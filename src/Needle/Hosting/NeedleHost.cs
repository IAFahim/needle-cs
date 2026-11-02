using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace Needle;

public readonly struct HostOptions
{
    public string ModelPath { get; init; }
    public string ToolsJson { get; init; }
    public NeedleOptions? Session { get; init; }
    /// <summary>
    /// Path to a baked primed-prefix file (<c>needle prime</c> output or the
    /// Editor <c>Needle/Bake Primed Prefix</c> menu, N64). When set, the
    /// session is constructed from the snapshot instead of running the
    /// tool-prompt prefill, so the host is ready in milliseconds. Requires
    /// <c>Session.AutoDate == false</c> (the bake is date-free, plan.md
    /// §10.5) and <c>Session.RetrievalTopK</c> at or above the tool count (a
    /// prefix cannot combine with tool retrieval). Leave null to prime at
    /// startup as before. No default flip: existing hosts behave unchanged.
    /// </summary>
    public string? PrimedPrefixPath { get; init; }
    public int RequestCapacity { get; init; }
    public int ReplyCapacity { get; init; }
    /// <summary>
    /// Treat every <see cref="ChannelRequestKind.Query"/> as a fresh conversation: the session is
    /// rewound to the primed tool prefix first, so unrelated orders never see each other and the
    /// context never fills up. <see cref="ChannelRequestKind.ToolResult"/> still continues the
    /// turn it answers. Leave off for chat-style use where each query builds on the last.
    /// </summary>
    public bool StatelessQueries { get; init; }
    /// <summary>
    /// Batch ceiling: the worker drains up to this many queued requests and
    /// decodes the stateless queries among them as one <c>BatchedSession</c>
    /// turn (N63). Decode is weight-bandwidth-bound, so the lanes share the
    /// stream. Unset or below 1 means 16; the value is clamped to 1–16 (the
    /// batched session's lane limit). 1 disables batching: every request
    /// takes the single-lane path exactly as before. Batching only applies
    /// to stateless query bursts on the untyped (<c>Start</c>) host; typed
    /// hosts, stateful hosts and <c>ToolResult</c> turns stay single-lane.
    /// The lanes prime once at <c>Start</c> (one tool-prefix prefill each),
    /// so a batching host starts slower but no drain ever pays a prime.
    /// </summary>
    public int MaxBatchSize { get; init; }
}

public sealed unsafe class NeedleHost : IDisposable
{
    // N63: the drain width. BatchedSession caps lanes at 16, so the scratch
    // staging below holds one single-lane-sized region per possible claim.
    const int MaxBatchCeiling = 16;
    const int LaneBytes = ChannelLimits.CallsCapacity + 2 * ChannelLimits.TextCapacity;

    readonly object gate = new();
    readonly ulong[] inFlight = new ulong[MaxBatchCeiling];
    ThreadPolicy policy;
    Model? model;
    NeedleSession? session;
    BatchedSession? batch;
    string batchTools = "";
    NeedleOptions batchOptions = new();
    int batchLanes;
    bool batchRetired;
    bool typedHost;
    int maxBatch = MaxBatchCeiling;
    Thread? worker;
    NeedleChannel channel;
    Runner runner = null!;
    byte* scratch;
    GCHandle self;
    long runningTicket = -1;
    int inFlightCount;
    bool statelessQueries;
    int started;
    int disposed;
    int stopping;

    delegate Reply Runner(ref ChannelRequest request, byte* bottom);

    public NeedleChannel Start(in HostOptions options)
    {
        typedHost = false;
        return StartCore(in options, Run);
    }

    public NeedleChannel StartTyped<TSink, TCommand>(in HostOptions options)
        where TSink : struct, Tools.ICallSink<TCommand>
        where TCommand : unmanaged
    {
        typedHost = true;
        return StartCore(in options, RunTyped<TSink, TCommand>);
    }

    NeedleChannel StartCore(in HostOptions options, Runner run)
    {
        if (Interlocked.Exchange(ref started, 1) != 0 || Volatile.Read(ref disposed) != 0)
            throw new InvalidOperationException("NeedleHost was already started");
        try
        {
            runner = run;
            statelessQueries = options.StatelessQueries;
            maxBatch = options.MaxBatchSize < 1 ? MaxBatchCeiling : Math.Min(MaxBatchCeiling, options.MaxBatchSize);
            var sessionOptions = options.Session ?? new NeedleOptions();
            // N62: stateless queries rewind before every Query, so a deferred
            // turn-closing is always dropped, never replayed: one engine pass
            // fewer per call. Continuations (ToolResult) replay it first, so
            // their history is unchanged. Explicit Session.Stateless wins.
            if (options.StatelessQueries && !sessionOptions.Stateless)
                sessionOptions = sessionOptions.WithStateless(true);
            batchOptions = sessionOptions;
            batchTools = options.ToolsJson;
            policy = sessionOptions.ResolvedThreading;
            model = Model.Open(options.ModelPath);
            byte[]? primedPrefix = null;
            if (!string.IsNullOrEmpty(options.PrimedPrefixPath))
            {
                // The bake is date-free; a dated system prompt would fail
                // prefix validation, so fail fast with the rule attached.
                if (sessionOptions.AutoDate) throw new ArgumentException("PrimedPrefixPath requires Session.AutoDate == false (plan.md §10.5).");
                primedPrefix = File.ReadAllBytes(options.PrimedPrefixPath);
            }
            session = primedPrefix != null
                ? new NeedleSession(model, options.ToolsJson, primedPrefix, batchOptions)
                : new NeedleSession(model, options.ToolsJson, batchOptions);
            // Lanes prime once here, up front: every later batch reuses them,
            // so no drain ever pays a prime mid-burst. Stateless untyped
            // hosts only; typed/stateful hosts and MaxBatchSize 1 never batch.
            if (!typedHost && batchOptions.Stateless && maxBatch >= 2) CreateBatchSession();
            channel = NeedleChannel.Allocate(
                options.RequestCapacity == 0 ? NeedleChannel.DefaultCapacity : options.RequestCapacity,
                options.ReplyCapacity == 0 ? NeedleChannel.DefaultCapacity : options.ReplyCapacity);
            scratch = (byte*)Mem.AlignedAlloc((nuint)MaxBatchCeiling * (nuint)LaneBytes, 64);
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
        // A lane in a running batch stops at the next shared step; its lane
        // then reports Cancelled, exactly like the single lane. RequestCancel
        // touches one managed int per lane, so this is safe at any time; the
        // flags are cleared at the next turn start.
        var active = batch;
        if (active == null) return;
        var count = Volatile.Read(ref inFlightCount);
        for (var i = 0; i < count; i++)
            if (Volatile.Read(ref inFlight[i]) == ticket) { active.RequestCancel(i); break; }
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

    // N63: serve the whole drain, not one claim. One queued request takes the
    // exact single-lane path as before (latency parity by construction);
    // two or more stateless queries share one batched turn and split the
    // weight reads. Replies always publish in claim order: the ring is FIFO
    // and callers match tickets, so batching must not reorder them.
    bool TryProcessBatch()
    {
        Span<ChannelRequest> requests = stackalloc ChannelRequest[MaxBatchCeiling];
        Span<ulong> tickets = stackalloc ulong[MaxBatchCeiling];
        var count = 0;
        while (count < maxBatch && channel.TryClaim(out requests[count], out tickets[count]))
            count++;
        if (count == 0) return false;
        ServeDrain(requests.Slice(0, count), tickets.Slice(0, count));
        return true;
    }

    void ServeDrain(Span<ChannelRequest> requests, Span<ulong> tickets)
    {
        var count = requests.Length;
        Span<Reply> results = stackalloc Reply[MaxBatchCeiling];
        Span<bool> cancelled = stackalloc bool[MaxBatchCeiling];
        Span<int> queryAt = stackalloc int[MaxBatchCeiling];
        Span<int> singleAt = stackalloc int[MaxBatchCeiling];
        var queries = 0;
        var singles = 0;
        var batchable = !typedHost && batchOptions.Stateless && !batchRetired;
        for (var i = 0; i < count; i++)
        {
            var skip = channel.IsCancelled(tickets[i]);
            cancelled[i] = skip;
            if (skip) continue;
            if (batchable && requests[i].Kind == ChannelRequestKind.Query)
                queryAt[queries++] = i;
            else
                singleAt[singles++] = i;
        }
        if (queries >= 2 && batch != null && !batchRetired)
        {
            // Singles first in claim order, then the shared turn: the single
            // session's lineage stays FIFO among the turns it actually serves.
            // The batch runs on its own engine, so re-prime the single
            // session afterwards: a later ToolResult then continues from the
            // primed prefix, the same lineage every batch lane starts from,
            // instead of an arbitrarily old single turn.
            for (var s = 0; s < singles; s++)
            {
                var i = singleAt[s];
                RunSingleAt(ref requests[i], tickets[i], Lane(i), ref results[i]);
            }
            RunBatchQueries(queryAt.Slice(0, queries), requests, tickets, results);
            session!.Reset();
        }
        else
        {
            for (var i = 0; i < count; i++)
                if (!cancelled[i]) RunSingleAt(ref requests[i], tickets[i], Lane(i), ref results[i]);
        }
        PublishDrain(requests, tickets, results, cancelled);
    }

    void RunSingleAt(ref ChannelRequest request, ulong ticket, byte* bottom, ref Reply result)
    {
        Volatile.Write(ref runningTicket, (long)ticket);
        result = channel.IsCancelled(ticket) ? CancelledReply() : runner(ref request, bottom);
        Volatile.Write(ref runningTicket, -1);
    }

    // All lanes open with the host's single tool set and prime once here; a
    // lane whose catalog exceeds RetrievalTopK throws there, which can only
    // happen when the single session is in retrieval mode — batching would
    // then narrow per query while lanes stay fixed, so retire instead and
    // serve everything single-lane.
    void CreateBatchSession()
    {
        BatchedSession? next = null;
        try
        {
            next = new BatchedSession(model!, batchOptions, maxBatch);
            for (var i = 0; i < maxBatch; i++) next.Open(batchTools);
        }
        catch (Exception)
        {
            // Batching is an optimization: it must never fail a request the
            // single lane can serve. Retire it for the host's lifetime.
            next?.Dispose();
            batchRetired = true;
            return;
        }
        batch = next;
        batchLanes = maxBatch;
    }

    void RunBatchQueries(Span<int> lanes, Span<ChannelRequest> requests, Span<ulong> tickets, Span<Reply> results)
    {
        var active = batch!;
        var width = batchLanes;
        for (var j = 0; j < lanes.Length; j++)
            Volatile.Write(ref inFlight[j], tickets[lanes[j]]);
        Volatile.Write(ref inFlightCount, lanes.Length);
        CallResult[] laneResults;
        try
        {
            // Every turn needs one query per open lane; lanes past the drain
            // get empty input, which fails fast before any engine work, so a
            // partial drain costs nothing extra. Lanes are stateless across
            // turns by design, matching one stateless single turn each.
            var inputs = new string[width];
            for (var j = 0; j < lanes.Length; j++)
            {
                var at = lanes[j];
                inputs[j] = Encoding.UTF8.GetString(Utf8.AsSpan(ref requests[at].Utf8, Math.Min(requests[at].Utf8Length, ChannelLimits.QueryCapacity)));
            }
            for (var j = lanes.Length; j < width; j++) inputs[j] = "";
            laneResults = active.Complete(inputs);
        }
        finally
        {
            Volatile.Write(ref inFlightCount, 0);
        }
        for (var j = 0; j < lanes.Length; j++)
            ConvertResult(laneResults[j], Lane(lanes[j]), ref results[lanes[j]]);
    }

    // The batched session reports strings and parsed calls; the channel
    // speaks Reply plus byte blobs in CallRestorer layout, so re-render here.
    // Calls re-render byte-identically (same escaping, verbatim arguments);
    // reasoning/response are trimmed by CallResult, so compare trimmed.
    void ConvertResult(CallResult lane, byte* bottom, ref Reply result)
    {
        var calls = new Span<byte>(bottom, ChannelLimits.CallsCapacity);
        var reasoning = new Span<byte>(bottom + ChannelLimits.CallsCapacity, ChannelLimits.TextCapacity);
        var response = new Span<byte>(bottom + ChannelLimits.CallsCapacity + ChannelLimits.TextCapacity, ChannelLimits.TextCapacity);
        var elapsedTicks = (long)(lane.TotalMs * Stopwatch.Frequency / 1000.0);
        var firstTokenTicks = (long)(lane.TimeToFirstTokenMs * Stopwatch.Frequency / 1000.0);
        var prefillTicks = lane.PrefillTokens > 0 && lane.PrefillTps > 0
            ? (long)(lane.PrefillTokens / lane.PrefillTps * Stopwatch.Frequency)
            : 0;
        if (!lane.Success)
        {
            if (lane.Error == "cancelled")
            {
                result = new Reply(ReplyStatus.Cancelled, FailureReason.None, ReplyFlags.None,
                    float.NaN, float.NaN, float.NaN, 0, 0, 0, 0, 0, 0, 0, 0, elapsedTicks);
                return;
            }
            var failedFlags = ReplyFlags.None;
            var failedReasoning = CopyText(lane.Reasoning, reasoning, ref failedFlags, ReplyFlags.ReasoningTruncated);
            result = new Reply(ReplyStatus.Failed, FailureFromError(lane.Error), failedFlags,
                float.NaN, float.NaN, float.NaN, 0, 0, failedReasoning, 0,
                lane.PrefillTokens, lane.DecodeTokens, prefillTicks, firstTokenTicks, elapsedTicks);
            return;
        }
        var refused = lane.SuppressedCalls.Count > 0;
        var status = refused ? ReplyStatus.Refused
            : lane.FunctionCalls.Count > 0 ? ReplyStatus.Called : ReplyStatus.Responded;
        var flags = ReplyFlags.None;
        var reasoningLength = CopyText(lane.Reasoning, reasoning, ref flags, ReplyFlags.ReasoningTruncated);
        var responseLength = status == ReplyStatus.Responded
            ? CopyText(lane.Response, response, ref flags, ReplyFlags.ResponseTruncated)
            : 0;
        var source = refused ? lane.SuppressedCalls : lane.FunctionCalls;
        var callBytes = 0;
        if (status != ReplyStatus.Responded)
        {
            callBytes = RenderCalls(source, calls);
            if (callBytes < 0)
            {
                // The lane buffer dwarfs the channel's; mirror the single
                // lane, which reports CallsBufferTooSmall with no call bytes.
                result = new Reply(ReplyStatus.Failed, FailureReason.CallsBufferTooSmall, flags,
                    lane.Confidence ?? float.NaN, lane.HeadConfidence ?? float.NaN, lane.DecodeConfidence ?? float.NaN,
                    source.Count, 0, reasoningLength, 0,
                    lane.PrefillTokens, lane.DecodeTokens, prefillTicks, firstTokenTicks, elapsedTicks);
                return;
            }
        }
        result = new Reply(status, FailureReason.None, flags,
            lane.Confidence ?? float.NaN, lane.HeadConfidence ?? float.NaN, lane.DecodeConfidence ?? float.NaN,
            source.Count, callBytes, reasoningLength, responseLength,
            lane.PrefillTokens, lane.DecodeTokens, prefillTicks, firstTokenTicks, elapsedTicks);
    }

    static int CopyText(string text, Span<byte> destination, ref ReplyFlags flags, ReplyFlags truncated)
    {
        if (string.IsNullOrEmpty(text)) return 0;
        if (Encoding.UTF8.GetByteCount(text) <= destination.Length)
            return Encoding.UTF8.GetBytes(text, destination);
        var encoder = Encoding.UTF8.GetEncoder();
        int written;
        fixed (char* chars = text)
        fixed (byte* bytes = destination)
            encoder.Convert(chars, text.Length, bytes, destination.Length, true, out _, out written, out _);
        flags |= truncated;
        return written;
    }

    static int RenderCalls(IReadOnlyList<FunctionCall> calls, Span<byte> destination)
    {
        var builder = new StringBuilder(256);
        builder.Append('[');
        for (var i = 0; i < calls.Count; i++)
        {
            if (i > 0) builder.Append(',');
            builder.Append("{\"name\":");
            JsonValue.WriteString(builder, calls[i].Name);
            builder.Append(",\"arguments\":");
            builder.Append(calls[i].ArgumentsJson);
            builder.Append('}');
        }
        builder.Append(']');
        var text = builder.ToString();
        if (Encoding.UTF8.GetByteCount(text) > destination.Length) return -1;
        return Encoding.UTF8.GetBytes(text, destination);
    }

    static FailureReason FailureFromError(string? error) => error switch
    {
        "the query is empty" => FailureReason.EmptyInput,
        "the input exceeds the maximum size" => FailureReason.InputTooLong,
        "context capacity exceeded" => FailureReason.ContextExhausted,
        "call exceeded the token budget" => FailureReason.CallTokenBudgetExceeded,
        "the generated call exceeds the call text buffer" => FailureReason.CallTextTooLong,
        "grammar admits no token" => FailureReason.GrammarDeadEnd,
        "grammar rejected the generated token" => FailureReason.GrammarRejectedToken,
        "the generated calls are malformed" => FailureReason.MalformedCalls,
        "frame budget exceeded" => FailureReason.FrameBudgetExceeded,
        _ => FailureReason.MalformedCalls,
    };

    void PublishDrain(Span<ChannelRequest> requests, Span<ulong> tickets, Span<Reply> results, Span<bool> cancelled)
    {
        for (var i = 0; i < requests.Length; i++)
        {
            ChannelReply reply = default;
            reply.Ticket = tickets[i];
            reply.Reply = cancelled[i] ? CancelledReply() : results[i];
            var bottom = Lane(i);
            new ReadOnlySpan<byte>(bottom, reply.Reply.Utf8Length)
                .CopyTo(Utf8.AsSpan(ref reply.Calls, reply.Reply.Utf8Length));
            new ReadOnlySpan<byte>(bottom + ChannelLimits.CallsCapacity, reply.Reply.ReasoningLength)
                .CopyTo(Utf8.AsSpan(ref reply.Reasoning, reply.Reply.ReasoningLength));
            new ReadOnlySpan<byte>(bottom + ChannelLimits.CallsCapacity + ChannelLimits.TextCapacity, reply.Reply.ResponseLength)
                .CopyTo(Utf8.AsSpan(ref reply.Response, reply.Reply.ResponseLength));
#if NETSTANDARD2_1
            var spin = new Spinner();
#else
            var spin = new SpinWait();
#endif
            while (!channel.TryPublishReply(in reply))
#if NETSTANDARD2_1
                spin.SpinOnce();
#else
                spin.SpinOnce(-1);
#endif
        }
    }

    byte* Lane(int index) => scratch + (nuint)index * (nuint)LaneBytes;

    Reply Run(ref ChannelRequest request, byte* bottom)
    {
        var queryLength = Math.Min(request.Utf8Length, ChannelLimits.QueryCapacity);
        var payload = Utf8.AsSpan(ref request.Utf8, queryLength);
        var calls = new Span<byte>(bottom, ChannelLimits.CallsCapacity);
        var reasoning = new Span<byte>(bottom + ChannelLimits.CallsCapacity, ChannelLimits.TextCapacity);
        var response = new Span<byte>(bottom + ChannelLimits.CallsCapacity + ChannelLimits.TextCapacity, ChannelLimits.TextCapacity);
        if (statelessQueries && request.Kind == ChannelRequestKind.Query) session!.Reset();
        return request.Kind == ChannelRequestKind.ToolResult
            ? session!.CompleteWithToolResult(payload, calls, reasoning, response)
            : session!.Complete(payload, calls, reasoning, response);
    }

    Reply RunTyped<TSink, TCommand>(ref ChannelRequest request, byte* bottom)
        where TSink : struct, Tools.ICallSink<TCommand>
        where TCommand : unmanaged
    {
        var queryLength = Math.Min(request.Utf8Length, ChannelLimits.QueryCapacity);
        var payload = Utf8.AsSpan(ref request.Utf8, queryLength);
        var capacity = Math.Max(1, ChannelLimits.CallsCapacity / sizeof(TCommand));
        var commands = new Span<TCommand>(bottom, capacity);
        var reasoning = new Span<byte>(bottom + ChannelLimits.CallsCapacity, ChannelLimits.TextCapacity);
        var response = new Span<byte>(bottom + ChannelLimits.CallsCapacity + ChannelLimits.TextCapacity, ChannelLimits.TextCapacity);
        var sink = new TSink();
        if (statelessQueries && request.Kind == ChannelRequestKind.Query) session!.Reset();
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
            if (!TryProcessBatch()) Park();
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
        batch?.Dispose();
        batch = null;
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
