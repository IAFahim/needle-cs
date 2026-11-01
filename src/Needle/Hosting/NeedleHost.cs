using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace Needle;

public readonly struct HostOptions
{
    public string ModelPath { get; init; }
    public string ToolsJson { get; init; }
    public NeedleOptions? Session { get; init; }
    public int RequestCapacity { get; init; }
    public int ReplyCapacity { get; init; }
}

public sealed unsafe class NeedleHost : IDisposable
{
    readonly object gate = new();
    ThreadPolicy policy;
    Model? model;
    NeedleSession? session;
    Thread? worker;
    NeedleChannel channel;
    Runner runner = null!;
    byte* scratch;
    GCHandle self;
    long runningTicket = -1;
    int started;
    int disposed;
    int stopping;

    delegate Reply Runner(ref ChannelRequest request);

    public NeedleChannel Start(in HostOptions options) => StartCore(in options, Run);

    public NeedleChannel StartTyped<TSink, TCommand>(in HostOptions options)
        where TSink : struct, Tools.ICallSink<TCommand>
        where TCommand : unmanaged =>
        StartCore(in options, RunTyped<TSink, TCommand>);

    NeedleChannel StartCore(in HostOptions options, Runner run)
    {
        if (Interlocked.Exchange(ref started, 1) != 0 || Volatile.Read(ref disposed) != 0)
            throw new InvalidOperationException("NeedleHost was already started");
        try
        {
            runner = run;
            var sessionOptions = options.Session ?? new NeedleOptions();
            policy = sessionOptions.ResolvedThreading;
            model = Model.Open(options.ModelPath);
            session = new NeedleSession(model, options.ToolsJson, sessionOptions);
            channel = NeedleChannel.Allocate(
                options.RequestCapacity == 0 ? NeedleChannel.DefaultCapacity : options.RequestCapacity,
                options.ReplyCapacity == 0 ? NeedleChannel.DefaultCapacity : options.ReplyCapacity);
            scratch = (byte*)Mem.AlignedAlloc(
                (nuint)(ChannelLimits.CallsCapacity + 2 * ChannelLimits.TextCapacity), 64);
            self = GCHandle.Alloc(this);
            channel.SetWake(&WakeTrampoline, (void*)GCHandle.ToIntPtr(self));
            worker = new Thread(Loop) { IsBackground = true, Name = "needle-host" };
            worker.Start();
            return channel;
        }
        catch
        {
            ReleaseNative();
            Interlocked.Exchange(ref started, 0);
            throw;
        }
    }

    public void Cancel(ulong ticket)
    {
        channel.Cancel(ticket);
        if (Interlocked.Read(ref runningTicket) == (long)ticket) session!.RequestCancel();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0) return;
        Volatile.Write(ref stopping, 1);
        Pulse();
        worker?.Join();
        ReleaseNative();
        if (self.IsAllocated) self.Free();
    }

    bool TryProcessOne()
    {
        if (!channel.TryClaim(out var request, out var ticket)) return false;
        Volatile.Write(ref runningTicket, (long)ticket);
        ChannelReply reply = default;
        reply.Ticket = ticket;
        reply.Reply = channel.IsCancelled(ticket) ? CancelledReply() : runner(ref request);
        Volatile.Write(ref runningTicket, -1);
        new ReadOnlySpan<byte>(scratch, reply.Reply.Utf8Length)
            .CopyTo(Utf8.AsSpan(ref reply.Calls, reply.Reply.Utf8Length));
        new ReadOnlySpan<byte>(scratch + ChannelLimits.CallsCapacity, reply.Reply.ReasoningLength)
            .CopyTo(Utf8.AsSpan(ref reply.Reasoning, reply.Reply.ReasoningLength));
        new ReadOnlySpan<byte>(scratch + ChannelLimits.CallsCapacity + ChannelLimits.TextCapacity, reply.Reply.ResponseLength)
            .CopyTo(Utf8.AsSpan(ref reply.Response, reply.Reply.ResponseLength));
        var spin = new SpinWait();
        while (!channel.TryPublishReply(in reply))
#if NETSTANDARD2_1
            spin.SpinOnce();
#else
            spin.SpinOnce(-1);
#endif
        return true;
    }

    Reply Run(ref ChannelRequest request)
    {
        var queryLength = Math.Min(request.Utf8Length, ChannelLimits.QueryCapacity);
        var payload = Utf8.AsSpan(ref request.Utf8, queryLength);
        var calls = new Span<byte>(scratch, ChannelLimits.CallsCapacity);
        var reasoning = new Span<byte>(scratch + ChannelLimits.CallsCapacity, ChannelLimits.TextCapacity);
        var response = new Span<byte>(scratch + ChannelLimits.CallsCapacity + ChannelLimits.TextCapacity, ChannelLimits.TextCapacity);
        return request.Kind == ChannelRequestKind.ToolResult
            ? session!.CompleteWithToolResult(payload, calls, reasoning, response)
            : session!.Complete(payload, calls, reasoning, response);
    }

    Reply RunTyped<TSink, TCommand>(ref ChannelRequest request)
        where TSink : struct, Tools.ICallSink<TCommand>
        where TCommand : unmanaged
    {
        var queryLength = Math.Min(request.Utf8Length, ChannelLimits.QueryCapacity);
        var payload = Utf8.AsSpan(ref request.Utf8, queryLength);
        var capacity = Math.Max(1, ChannelLimits.CallsCapacity / sizeof(TCommand));
        var commands = new Span<TCommand>(scratch, capacity);
        var reasoning = new Span<byte>(scratch + ChannelLimits.CallsCapacity, ChannelLimits.TextCapacity);
        var response = new Span<byte>(scratch + ChannelLimits.CallsCapacity + ChannelLimits.TextCapacity, ChannelLimits.TextCapacity);
        var sink = new TSink();
        var reply = request.Kind == ChannelRequestKind.ToolResult
            ? session!.CompleteWithToolResult(payload, commands, reasoning, response, ref sink)
            : session!.Complete(payload, commands, reasoning, response, ref sink);
        var stored = Math.Min(reply.CallCount, capacity);
        return new Reply(reply.Status, reply.Failure, reply.Flags, reply.Confidence, reply.HeadConfidence,
            reply.DecodeConfidence, stored, stored * sizeof(TCommand), reply.ReasoningLength, reply.ResponseLength,
            reply.PromptTokens, reply.DecodeTokens, reply.PrefillTicks, reply.FirstTokenTicks, reply.ElapsedTicks);
    }

    static Reply CancelledReply() => new(
        ReplyStatus.Cancelled, FailureReason.None, ReplyFlags.None,
        0f, 0f, 0f, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    void Loop()
    {
        OsThreads.ConfigureCurrentThread(policy.Priority, policy.Placement, -1);
        while (Volatile.Read(ref stopping) == 0)
            if (!TryProcessOne()) Park();
    }

    void Park()
    {
        lock (gate)
        {
            channel.SetSleeping(1);
            while (Volatile.Read(ref stopping) == 0 && !channel.HasPendingRequest)
                Monitor.Wait(gate);
            channel.SetSleeping(0);
        }
    }

    void Pulse()
    {
        lock (gate) Monitor.PulseAll(gate);
    }

    static void WakeTrampoline(void* state) =>
        ((NeedleHost)GCHandle.FromIntPtr((nint)state).Target!).Pulse();

    void ReleaseNative()
    {
        if (scratch != null)
        {
            Mem.AlignedFree(scratch);
            scratch = null;
        }
        session?.Dispose();
        session = null;
        model?.Dispose();
        model = null;
        if (channel.block != null)
        {
            channel.Free();
            channel = default;
        }
    }
}
