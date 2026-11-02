using System;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Threading;

namespace Needle;

// N71: cross-process view of a NeedleChannel block. The rings are
// offset-based (cells follow their headers, cancel words follow the reply
// ring), so two processes mapping the same file each build a NeedleChannel
// over their own base address and read/write the same cells. The only
// process-local state in the block was the wake function pointer, which a
// foreign process must never call: shared blocks set the Shared flag in the
// block header, SignalWake becomes a no-op, and both sides poll with
// TryClaim/TryReceive (or the WaitFor* helpers below, which spin-then-yield
// and never sleep).
//
// Game-side code is unchanged: submit and receive through
// SharedChannel.Channel exactly as with NeedleHost.Start's channel.
// Never call Free() on a shared view; disposing this handle releases the
// mapping. Free() on a shared view would corrupt the owner's allocator.
public sealed unsafe class SharedChannel : IDisposable
{
    readonly MemoryMappedFile file;
    readonly MemoryMappedViewAccessor view;
    int disposed;

    SharedChannel(MemoryMappedFile file, MemoryMappedViewAccessor view, string path, NeedleChannel channel)
    {
        this.file = file;
        this.view = view;
        Path = path;
        Channel = channel;
    }

    public NeedleChannel Channel { get; }

    /// <summary>Backing file path: hand this to the other process (argv, env, inspector) for <see cref="Open"/>.</summary>
    public string Path { get; }

    public static string NewPath(string prefix = "needle-channel")
    {
        if (string.IsNullOrEmpty(prefix)) throw new ArgumentNullException(nameof(prefix));
        return System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"{prefix}-{Guid.NewGuid():N}.channel");
    }

    public static SharedChannel Create(string path, int requestCapacity = NeedleChannel.DefaultCapacity, int replyCapacity = NeedleChannel.DefaultCapacity)
    {
        if (string.IsNullOrEmpty(path)) throw new ArgumentNullException(nameof(path));
        NeedleChannel.ComputeLayout(requestCapacity, replyCapacity, out var requestCells, out var replyCells, out var words, out var total);
        // N71: file-backed, not named: named maps throw
        // PlatformNotSupportedException on Linux/macOS. The path travels
        // out-of-band; the last side out deletes it.
        var file = MemoryMappedFile.CreateFromFile(path, FileMode.Create, null, total, MemoryMappedFileAccess.ReadWrite);
        try
        {
            var view = file.CreateViewAccessor(0, total, MemoryMappedFileAccess.ReadWrite);
            byte* pointer = null;
            try
            {
                view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
                var block = pointer + view.PointerOffset;
                Mem.Clear(block, (nuint)total);
                var channel = NeedleChannel.InitializeBlock(block, requestCells, replyCells, words, shared: 1);
                return new SharedChannel(file, view, path, channel);
            }
            catch
            {
                if (pointer != null) view.SafeMemoryMappedViewHandle.ReleasePointer();
                view.Dispose();
                throw;
            }
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    public static SharedChannel Open(string path)
    {
        if (string.IsNullOrEmpty(path)) throw new ArgumentNullException(nameof(path));
        var length = new FileInfo(path).Length;
        if (length <= 0) throw new InvalidDataException($"Shared channel file \"{path}\" is missing or empty.");
        var file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, length, MemoryMappedFileAccess.ReadWrite);
        try
        {
            var view = file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.ReadWrite);
            byte* pointer = null;
            try
            {
                view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
                // Validates the magic and rebuilds ring pointers over this
                // process's base address; the live rings are never cleared.
                var channel = NeedleChannel.Attach(pointer + view.PointerOffset);
                if (!channel.IsShared)
                    throw new InvalidDataException($"Shared channel file \"{path}\" is not a shared channel block.");
                return new SharedChannel(file, view, path, channel);
            }
            catch
            {
                if (pointer != null) view.SafeMemoryMappedViewHandle.ReleasePointer();
                view.Dispose();
                throw;
            }
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    // Polls until any reply arrives (single-worker hosts publish in
    // submission order, so the first reply is the caller's; multi-flight
    // callers match reply.Ticket themselves as in ChannelTests.WaitForReply).
    // Zero allocation; never sleeps.
    public static bool WaitForReply(in NeedleChannel channel, out ChannelReply reply, int timeoutMilliseconds = 5000)
    {
        var start = Stopwatch.GetTimestamp();
        var budget = timeoutMilliseconds < 0 ? long.MaxValue : (long)timeoutMilliseconds * (Stopwatch.Frequency / 1000);
#if NETSTANDARD2_1
        var spin = new Spinner();
#else
        var spin = new SpinWait();
#endif
        while (Stopwatch.GetTimestamp() - start < budget)
        {
            if (channel.TryReceive(out reply)) return true;
#if NETSTANDARD2_1
            spin.SpinOnce();
#else
            spin.SpinOnce(-1);
#endif
        }
        reply = default;
        return false;
    }

    // Server-side poll: waits until a request is queued, then the caller
    // drains with TryClaim. Zero allocation; never sleeps.
    public static bool WaitForRequest(in NeedleChannel channel, int timeoutMilliseconds = 5000)
    {
        var start = Stopwatch.GetTimestamp();
        var budget = timeoutMilliseconds < 0 ? long.MaxValue : (long)timeoutMilliseconds * (Stopwatch.Frequency / 1000);
#if NETSTANDARD2_1
        var spin = new Spinner();
#else
        var spin = new SpinWait();
#endif
        while (Stopwatch.GetTimestamp() - start < budget)
        {
            if (channel.HasPendingRequest) return true;
#if NETSTANDARD2_1
            spin.SpinOnce();
#else
            spin.SpinOnce(-1);
#endif
        }
        return false;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0) return;
        view.SafeMemoryMappedViewHandle.ReleasePointer();
        view.Dispose();
        file.Dispose();
    }
}
