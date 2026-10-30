using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace Needle;

public enum ChannelRequestKind : byte
{
    Query = 0,
    ToolResult = 1,
}

public static class ChannelLimits
{
    public const int QueryCapacity = 512;
    public const int CallsCapacity = 1024;
    public const int TextCapacity = 2048;
}

#if NET10_0_OR_GREATER
[InlineArray(ChannelLimits.QueryCapacity)]
public struct QueryUtf8
{
    private byte e;
}

[InlineArray(ChannelLimits.CallsCapacity)]
public struct CallsUtf8
{
    private byte e;
}

[InlineArray(ChannelLimits.TextCapacity)]
public struct TextUtf8
{
    private byte e;
}
#else
public unsafe struct QueryUtf8
{
    public fixed byte e[ChannelLimits.QueryCapacity];
    public byte this[int index] { get => e[index]; set => e[index] = value; }
}

public unsafe struct CallsUtf8
{
    public fixed byte e[ChannelLimits.CallsCapacity];
    public byte this[int index] { get => e[index]; set => e[index] = value; }
}

public unsafe struct TextUtf8
{
    public fixed byte e[ChannelLimits.TextCapacity];
    public byte this[int index] { get => e[index]; set => e[index] = value; }
}
#endif

public static unsafe class Utf8
{
    public static Span<byte> AsSpan(ref QueryUtf8 payload, int length)
    {
#if NET10_0_OR_GREATER
        return MemoryMarshal.CreateSpan(ref payload[0], length);
#else
        fixed (byte* p = payload.e) return new Span<byte>(p, length);
#endif
    }

    public static Span<byte> AsSpan(ref CallsUtf8 payload, int length)
    {
#if NET10_0_OR_GREATER
        return MemoryMarshal.CreateSpan(ref payload[0], length);
#else
        fixed (byte* p = payload.e) return new Span<byte>(p, length);
#endif
    }

    public static Span<byte> AsSpan(ref TextUtf8 payload, int length)
    {
#if NET10_0_OR_GREATER
        return MemoryMarshal.CreateSpan(ref payload[0], length);
#else
        fixed (byte* p = payload.e) return new Span<byte>(p, length);
#endif
    }
}

public struct ChannelRequest
{
    public ChannelRequestKind Kind;
    public int Utf8Length;
    public QueryUtf8 Utf8;
}

public struct ChannelReply
{
    public ulong Ticket;
    public Reply Reply;
    public CallsUtf8 Calls;
    public TextUtf8 Reasoning;
    public TextUtf8 Response;
}

internal static unsafe class RingOps
{
    [StructLayout(LayoutKind.Explicit, Size = 256)]
    internal struct Cursors
    {
        [FieldOffset(0)] internal long Enqueue;
        [FieldOffset(128)] internal long Dequeue;
    }

    internal struct Header
    {
        internal Cursors Cursors;
        internal long Mask;
    }

    internal struct Cell<T> where T : unmanaged
    {
        internal long Sequence;
        internal T Slot;
    }

    internal static Cell<T>* Cells<T>(Header* header) where T : unmanaged => (Cell<T>*)(header + 1);

    internal static void Initialize<T>(Header* header, int capacity) where T : unmanaged
    {
        header->Mask = capacity - 1;
        var cells = Cells<T>(header);
        for (long i = 0; i < capacity; i++) Volatile.Write(ref cells[i].Sequence, i);
        Volatile.Write(ref header->Cursors.Enqueue, 0L);
        Volatile.Write(ref header->Cursors.Dequeue, 0L);
    }

    internal static bool TryEnqueue<T>(Header* header, in T slot, out long position) where T : unmanaged
    {
        var cells = Cells<T>(header);
        long pos;
        for (;;)
        {
            pos = Volatile.Read(ref header->Cursors.Enqueue);
            var cell = cells + (pos & header->Mask);
            var difference = Volatile.Read(ref cell->Sequence) - pos;
            if (difference == 0)
            {
                if (Interlocked.CompareExchange(ref header->Cursors.Enqueue, pos + 1, pos) == pos) break;
            }
            else if (difference < 0)
            {
                position = 0;
                return false;
            }
        }
        position = pos;
        var target = cells + (pos & header->Mask);
        target->Slot = slot;
        Volatile.Write(ref target->Sequence, pos + 1);
        return true;
    }

    internal static bool TryDequeue<T>(Header* header, out T slot, out long position) where T : unmanaged
    {
        slot = default;
        var cells = Cells<T>(header);
        long pos;
        for (;;)
        {
            pos = Volatile.Read(ref header->Cursors.Dequeue);
            var cell = cells + (pos & header->Mask);
            var difference = Volatile.Read(ref cell->Sequence) - (pos + 1);
            if (difference == 0)
            {
                if (Interlocked.CompareExchange(ref header->Cursors.Dequeue, pos + 1, pos) == pos)
                {
                    slot = cell->Slot;
                    Volatile.Write(ref cell->Sequence, pos + header->Mask + 1);
                    position = pos;
                    return true;
                }
            }
            else if (difference < 0)
            {
                position = 0;
                return false;
            }
        }
    }

    internal static bool HasItems<T>(Header* header) where T : unmanaged
    {
        var pos = Volatile.Read(ref header->Cursors.Dequeue);
        var cell = Cells<T>(header) + (pos & header->Mask);
        return Volatile.Read(ref cell->Sequence) == pos + 1;
    }
}

public readonly unsafe struct NeedleChannel
{
    public const int DefaultCapacity = 64;

    const int SleepingOffset = 0;
    const int WakeFunctionOffset = 128;
    const int WakeStateOffset = 136;
    const int RequestHeaderOffset = 256;

    internal readonly byte* block;
    internal readonly long* sleeping;
    internal readonly RingOps.Header* requests;
    internal readonly RingOps.Header* replies;
    internal readonly long* cancelWords;
    internal readonly int cancelMask;

    NeedleChannel(byte* block, long* sleeping, RingOps.Header* requests, RingOps.Header* replies, long* cancelWords, int cancelMask)
    {
        this.block = block;
        this.sleeping = sleeping;
        this.requests = requests;
        this.replies = replies;
        this.cancelWords = cancelWords;
        this.cancelMask = cancelMask;
    }

    public static NeedleChannel Allocate(int requestCapacity = DefaultCapacity, int replyCapacity = DefaultCapacity)
    {
        var requestCells = PowerOfTwo(requestCapacity);
        var replyCells = PowerOfTwo(replyCapacity);
        var words = Math.Max(1, (2 * Math.Max(requestCells, replyCells) + 63) / 64);
        var requestCellOffset = RequestHeaderOffset + sizeof(RingOps.Header);
        var replyHeaderOffset = Align(requestCellOffset + requestCells * sizeof(RingOps.Cell<ChannelRequest>));
        var replyCellOffset = replyHeaderOffset + sizeof(RingOps.Header);
        var cancelOffset = Align(replyCellOffset + replyCells * sizeof(RingOps.Cell<ChannelReply>));
        var total = Align(cancelOffset + words * sizeof(long));
        var block = (byte*)Mem.AlignedAlloc((nuint)total, 64);
        Mem.Clear(block, (nuint)total);
        var requests = (RingOps.Header*)(block + RequestHeaderOffset);
        var replies = (RingOps.Header*)(block + replyHeaderOffset);
        RingOps.Initialize<ChannelRequest>(requests, requestCells);
        RingOps.Initialize<ChannelReply>(replies, replyCells);
        return new NeedleChannel(block, (long*)(block + SleepingOffset), requests, replies, (long*)(block + cancelOffset), words - 1);
    }

    public void Free()
    {
        if (block != null) Mem.AlignedFree(block);
    }

    internal void SetWake(delegate*<void*, void> wake, void* state)
    {
        Volatile.Write(ref *(nint*)(block + WakeFunctionOffset), (nint)(void*)wake);
        Volatile.Write(ref *(nint*)(block + WakeStateOffset), (nint)state);
    }

    internal void SetSleeping(long value) => Volatile.Write(ref *sleeping, value);

    public bool TrySubmitQuery(ReadOnlySpan<byte> utf8, out ulong ticket)
    {
        fixed (byte* source = utf8) return Submit(source, utf8.Length, ChannelRequestKind.Query, out ticket);
    }

    public bool TrySubmitQuery(byte* utf8, int length, out ulong ticket) => Submit(utf8, length, ChannelRequestKind.Query, out ticket);

    public bool TrySubmitToolResult(ReadOnlySpan<byte> utf8, out ulong ticket)
    {
        fixed (byte* source = utf8) return Submit(source, utf8.Length, ChannelRequestKind.ToolResult, out ticket);
    }

    public bool TrySubmitToolResult(byte* utf8, int length, out ulong ticket) => Submit(utf8, length, ChannelRequestKind.ToolResult, out ticket);

    public bool TrySubmit(in ChannelRequest request, out ulong ticket)
    {
        if (request.Utf8Length > ChannelLimits.QueryCapacity)
        {
            ticket = 0;
            return false;
        }
        if (!RingOps.TryEnqueue(requests, in request, out var position))
        {
            ticket = 0;
            return false;
        }
        ticket = (ulong)position;
        SignalWake();
        return true;
    }

    internal bool TryClaim(out ChannelRequest request, out ulong ticket)
    {
        var claimed = RingOps.TryDequeue(requests, out request, out var position);
        ticket = (ulong)position;
        return claimed;
    }

    internal bool TryPublishReply(in ChannelReply reply) => RingOps.TryEnqueue(replies, in reply, out _);

    public bool TryReceive(out ChannelReply reply) => RingOps.TryDequeue(replies, out reply, out _);

    public bool HasPendingRequest => RingOps.HasItems<ChannelRequest>(requests);

    public bool HasCompletedReply => RingOps.HasItems<ChannelReply>(replies);

    public bool IsCancelled(ulong ticket) =>
        (Volatile.Read(ref cancelWords[CancelIndex(ticket)]) & (1L << (int)(ticket & 63))) != 0;

    public void Cancel(ulong ticket)
    {
        var word = cancelWords + CancelIndex(ticket);
        var flag = 1L << (int)(ticket & 63);
        var current = Volatile.Read(ref *word);
        while ((current & flag) == 0)
        {
            var observed = Interlocked.CompareExchange(ref *word, current | flag, current);
            if (observed == current) return;
            current = observed;
        }
    }

    nint CancelIndex(ulong ticket) => (nint)((ticket >> 6) & (ulong)cancelMask);

    bool Submit(byte* utf8, int length, ChannelRequestKind kind, out ulong ticket)
    {
        if (length < 0 || length > ChannelLimits.QueryCapacity)
        {
            ticket = 0;
            return false;
        }
        ChannelRequest request = default;
        request.Kind = kind;
        request.Utf8Length = length;
        for (var i = 0; i < length; i++) request.Utf8[i] = utf8[i];
        return TrySubmit(in request, out ticket);
    }

    void SignalWake()
    {
        var function = Volatile.Read(ref *(nint*)(block + WakeFunctionOffset));
        if (function == IntPtr.Zero) return;
        if (Interlocked.CompareExchange(ref *sleeping, 0, 0) == 0) return;
        var state = Volatile.Read(ref *(nint*)(block + WakeStateOffset));
        ((delegate*<void*, void>)(void*)function)((void*)state);
    }

    static int PowerOfTwo(int requested)
    {
        var capacity = 1;
        while (capacity < requested) capacity <<= 1;
        return capacity;
    }

    static int Align(int value) => (value + 7) & ~7;
}
