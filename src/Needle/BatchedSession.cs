using System;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Needle;

// W26 multi-sequence batched decode: many NPCs for roughly the cost of one.
// Each lane owns one engine slot (KV, histories, pooling, position) plus its
// grammar state, masks and text buffers; one ForwardBatch per step feeds one
// token run per unfinished lane through the shared weight reads. Grammar
// masks differ per lane, so after each shared pass every lane applies its own
// mask and argmax serially — that is where the sharing stops, by architecture
// (per-sequence argmax, trie walk and token bookkeeping stay serial CPU work).
//
// The turn driver mirrors NeedleSession phase for phase (reason, decision,
// call, respond) with identical token feeds, feeds-per-token and logits
// requests, so every lane's output is identical to decoding it alone. What
// batching does NOT cover: prefill stays per sequence (FeedSlot); jump-forward
// chunking is dormant (every lane steps token by token, which equals the
// per-token greedy sequence either way); tool retrieval is rejected at Open
// (lanes decode with fixed tool sets); typed sinks and cross attention are
// unsupported. BatchedSession is not thread-safe: one thread drives steps,
// RequestCancel excepted.
public sealed unsafe class BatchedSession : IDisposable
{
    const int ImStart = 4, ImEnd = 5, ThinkStart = 6, ThinkEnd = 7, ToolCallStart = 10, ToolCallEnd = 11, EndOfText = 1;
    const int MaxPieceBytes = 64;
    const int SparseHeadThreshold = 256;
    const int MaxRunTokens = 4;
    const int MaxLanes = 16;
    const int InputOverheadBytes = 128;
    const int TokenizerInputBytes = 1 << 16;
    const int CallsHeadroomBytes = 256;

    static ReadOnlySpan<byte> ToolResultHead => "<|im_start|>tool\n<tool_result>"u8;
    static ReadOnlySpan<byte> ToolResultTail => "</tool_result><|im_end|>\n<|im_start|>assistant\n"u8;
    static ReadOnlySpan<byte> FirstQueryHead => "\n"u8;
    static ReadOnlySpan<byte> LaterQueryHead => "<|im_start|>user\n"u8;
    static ReadOnlySpan<byte> QueryTail => "<|im_end|>\n<|im_start|>assistant\n"u8;

    static readonly Regex DateKey = new(@"\bdate\s*:", RegexOptions.IgnoreCase);
    static readonly Regex IsoStamp = new(@"\d{4}-\d{2}-\d{2}");

    enum Phase : byte { Reasoning, ThinkHead, ThinkTail, Call, CallClosing, Respond, RespondClosing }

    enum Want : byte { None, Full, Sparse }

    struct Lane
    {
        public ToolSet Tools;
        public Grammar Grammar;
        public int Slot;
        public GrammarState State;
        public Phase Phase;
        public uint* Mask;
        public int* SparseRows;
        public float* ScratchLogits;
        public float* Logits;
        public float* Pooled;
        public float* ConfidenceScratch;
        public byte[] CallText;
        public int CallTextLength;
        public CallSpan[] CallSpans;
        public byte[] KeyScratch;
        public byte[] Calls;
        public byte[] Reasoning;
        public int ReasoningLength;
        public byte[] Response;
        public int ResponseLength;
        public int[] Pending;
        public int PendingCount;
        public Want Want;
        public int CompactIndex;
        public int LastRow;
        public int SparseCount;
        public uint* CurMask;
        public int Allowed;
        public bool Full;
        public bool First;
        public double LogProb;
        public int Steps;
        public int Current;
        public int PromptTokens;
        public int DecodeTokens;
        public long Started;
        public long PrefillTicks;
        public long FirstTokenTicks;
        public long ElapsedTicks;
        public ReplyFlags Flags;
        public bool TurnOne;
        public int CancelRequested;
        public int RestoredCallCount;
        public CallResult Result;
        public bool Finished;
    }

    readonly Model model;
    readonly NeedleOptions options;
    readonly BatchedEngine engine;
    readonly Tokenizer tokenizer;
    readonly TokenTrie trie;
    readonly string system;
    readonly int maskWords;
    readonly int vocabSize;
    readonly uint* reasoningMask;
    readonly uint* decisionMask;
    readonly uint* responseMask;
    readonly int[] thinkEndForced;
    readonly int[] callClosing;
    readonly int[] responseClosing;
    readonly byte[] inputBuffer;
    readonly int[] promptIds;
    readonly int maxInputBytes;
    readonly int contextCapacity;
    readonly int embeddingWidth;
    readonly Lane[] lanes;
    int laneCount;

    public BatchedSession(Model model, NeedleOptions? options = null, int maxSequences = 8)
    {
        if (maxSequences < 1 || maxSequences > MaxLanes) throw new ArgumentOutOfRangeException(nameof(maxSequences));
        this.model = model;
        this.options = options ?? new NeedleOptions();
        this.engine = new BatchedEngine(model, this.options.ResolvedThreading, this.options.Context, 64, maxSequences);
        var vocabulary = model.GetVocabulary();
        tokenizer = new Tokenizer(vocabulary, TokenizerInputBytes);
        trie = new TokenTrie(vocabulary.Trie);
        system = BuildSystem(this.options);
        maskWords = trie.Words;
        vocabSize = model.VocabSize;
        reasoningMask = CopySharedMask(vocabulary);
        decisionMask = CopyMask(reasoningMask, ToolCallStart, ImEnd, EndOfText);
        responseMask = CopyMask(reasoningMask, ImEnd, EndOfText);
        responseMask[ThinkEnd / 32] &= ~(1u << (ThinkEnd % 32));
        var newline = Tokens("\n", false);
        thinkEndForced = Concatenate(new[] { ThinkEnd }, newline);
        callClosing = Concatenate(new[] { ToolCallEnd, ImEnd }, newline);
        responseClosing = Concatenate(new[] { ImEnd }, newline);
        contextCapacity = Math.Min(this.options.Context, model.Geometry.MaxSeq);
        maxInputBytes = Math.Max(1, Math.Min(TokenizerInputBytes - InputOverheadBytes, this.options.Context * 16));
        inputBuffer = new byte[maxInputBytes + InputOverheadBytes];
        promptIds = new int[3 * (maxInputBytes + InputOverheadBytes) + 16];
        var head = model.Confidence;
        embeddingWidth = head != null ? head->Queries * model.Geometry.Dim : 0;
        lanes = new Lane[maxSequences];
    }

    public int LaneCount => laneCount;

    public int MaxSequences => lanes.Length;

    // Opens a lane with its own tool set and primes its slot. Tool retrieval
    // would re-prime mid-turn, so lanes whose catalog exceeds RetrievalTopK
    // are rejected: decode with the narrowed set instead.
    public int Open(string toolsJson)
    {
        if (laneCount >= lanes.Length) throw new InvalidOperationException("all batch lanes are open");
        var tools = new ToolSet(toolsJson);
        if (tools.Tools.Count > options.RetrievalTopK && model.Confidence != null)
            throw new NotSupportedException("batched decode needs fixed tool sets; raise RetrievalTopK past the catalog size");
        var lane = new Lane
        {
            Tools = tools,
            Grammar = new Grammar(tools.Tools),
            Slot = laneCount,
            TurnOne = true,
            Pending = new int[MaxRunTokens],
        };
        try
        {
            lane.Mask = (uint*)Mem.AllocZeroed((nuint)maskWords * 4);
            lane.SparseRows = (int*)Mem.Alloc((nuint)SparseHeadThreshold * 4);
            lane.ScratchLogits = (float*)Mem.Alloc((nuint)vocabSize * 4);
            lane.Logits = (float*)Mem.Alloc((nuint)vocabSize * 4);
            lane.Pooled = (float*)Mem.AllocZeroed((nuint)Math.Max(1, embeddingWidth) * 4);
            var confidence = model.Confidence;
            lane.ConfidenceScratch = confidence != null
                ? (float*)Mem.AllocZeroed((nuint)((model.Geometry.Layers + 1) * confidence->Probes * (model.Geometry.Dim + 1)) * 4)
                : null;
            var callBytes = Math.Max(1, options.MaxCallTokens) * MaxPieceBytes;
            lane.CallText = new byte[callBytes];
            lane.Calls = new byte[callBytes + CallsHeadroomBytes];
            lane.CallSpans = new CallSpan[Math.Max(1, options.MaxCallTokens)];
            lane.KeyScratch = new byte[Math.Max(64, tools.Aliases.LongestPropertyAlias + 2)];
            lane.Reasoning = new byte[Math.Max(0, options.MaxReasoningTokens) * MaxPieceBytes];
            lane.Response = new byte[Math.Max(0, options.MaxCallTokens) * MaxPieceBytes];
            Prime(ref lane);
        }
        catch
        {
            lane.Grammar.Dispose();
            if (lane.Mask != null) Mem.Free(lane.Mask);
            if (lane.SparseRows != null) Mem.Free(lane.SparseRows);
            if (lane.ScratchLogits != null) Mem.Free(lane.ScratchLogits);
            if (lane.Logits != null) Mem.Free(lane.Logits);
            if (lane.Pooled != null) Mem.Free(lane.Pooled);
            if (lane.ConfidenceScratch != null) Mem.Free(lane.ConfidenceScratch);
            throw;
        }
        lanes[laneCount] = lane;
        return laneCount++;
    }

    public void Reset(int lane)
    {
        engine.RewindSlot(lanes[lane].Slot);
        lanes[lane].TurnOne = true;
    }

    public void RequestCancel(int lane) => System.Threading.Volatile.Write(ref lanes[lane].CancelRequested, 1);

    public CallResult[] Complete(string[] queries)
    {
        if (queries.Length != laneCount) throw new ArgumentException("one query per open lane");
        var inputs = new byte[laneCount][];
        for (var i = 0; i < laneCount; i++) inputs[i] = Encoding.UTF8.GetBytes(queries[i] ?? "");
        RunBatch(inputs, false);
        var results = new CallResult[laneCount];
        for (var i = 0; i < laneCount; i++) results[i] = lanes[i].Result;
        return results;
    }

    public CallResult[] CompleteWithToolResult(string[] toolJsons)
    {
        if (toolJsons.Length != laneCount) throw new ArgumentException("one tool result per open lane");
        var inputs = new byte[laneCount][];
        for (var i = 0; i < laneCount; i++) inputs[i] = Encoding.UTF8.GetBytes(toolJsons[i] ?? "");
        RunBatch(inputs, true);
        var results = new CallResult[laneCount];
        for (var i = 0; i < laneCount; i++) results[i] = lanes[i].Result;
        return results;
    }

    void RunBatch(byte[][] inputs, bool toolResult)
    {
        var started = Stopwatch.GetTimestamp();
        for (var i = 0; i < laneCount; i++) PrefillLane(i, inputs[i], toolResult, started);
        var tokens = stackalloc int[MaxLanes * MaxRunTokens];
        var slots = stackalloc int[MaxLanes * MaxRunTokens];
        var heads = stackalloc int[MaxLanes * MaxRunTokens];
        while (true)
        {
            var rows = 0;
            var headCount = 0;
            var active = 0;
            for (var i = 0; i < laneCount; i++)
            {
                ref var lane = ref lanes[i];
                if (lane.Finished) continue;
                active++;
                if (System.Threading.Volatile.Read(ref lane.CancelRequested) != 0)
                {
                    FailLane(ref lane, ReplyStatus.Cancelled, FailureReason.None);
                    continue;
                }
                if (lane.PendingCount == 0) continue;
                if (engine.SlotLength(lane.Slot) + lane.PendingCount > engine.Capacity)
                {
                    FailLane(ref lane, ReplyStatus.Failed, FailureReason.ContextExhausted);
                    continue;
                }
                for (var k = 0; k < lane.PendingCount; k++)
                {
                    tokens[rows] = lane.Pending[k];
                    slots[rows] = lane.Slot;
                    rows++;
                }
                lane.LastRow = rows - 1;
                if (lane.Want == Want.Full)
                {
                    lane.CompactIndex = headCount;
                    heads[headCount++] = rows - 1;
                }
            }
            if (active == 0) break;
            if (rows > 0) engine.ForwardBatch(tokens, slots, rows, heads, headCount);
            for (var i = 0; i < laneCount; i++)
            {
                ref var lane = ref lanes[i];
                if (!lane.Finished) AdvanceLane(ref lane);
            }
        }
    }

    void PrefillLane(int index, byte[] input, bool toolResult, long started)
    {
        ref var lane = ref lanes[index];
        lane.Started = started;
        lane.PrefillTicks = 0;
        lane.FirstTokenTicks = 0;
        lane.ElapsedTicks = 0;
        lane.LogProb = 0;
        lane.Steps = 0;
        lane.PromptTokens = 0;
        lane.DecodeTokens = 0;
        lane.ReasoningLength = 0;
        lane.ResponseLength = 0;
        lane.CallTextLength = 0;
        lane.Flags = ReplyFlags.None;
        lane.PendingCount = 0;
        lane.First = true;
        lane.Finished = false;
        System.Threading.Volatile.Write(ref lane.CancelRequested, 0);
        engine.RewindSlot(lane.Slot);
        if (!toolResult && input.Length == 0) { FailLane(ref lane, ReplyStatus.Failed, FailureReason.EmptyInput); return; }
        if (input.Length > maxInputBytes) { FailLane(ref lane, ReplyStatus.Failed, FailureReason.InputTooLong); return; }
        var count = EncodePrompt(ref lane, input, toolResult);
        if (engine.SlotLength(lane.Slot) + count > engine.Capacity) { FailLane(ref lane, ReplyStatus.Failed, FailureReason.ContextExhausted); return; }
        lane.TurnOne = false;
        lane.PromptTokens = count;
        engine.FeedSlot(lane.Slot, new ReadOnlySpan<int>(promptIds, 0, count), true);
        Buffer.MemoryCopy(engine.BatchLogitsRow(0), lane.Logits, (nuint)vocabSize * 4, (nuint)vocabSize * 4);
        lane.PrefillTicks = Stopwatch.GetTimestamp() - started;
        lane.CurMask = reasoningMask;
        lane.Allowed = int.MaxValue;
        lane.Full = true;
        lane.Phase = Phase.Reasoning;
    }

    void AdvanceLane(ref Lane lane)
    {
        if (!lane.First && lane.Want == Want.Sparse)
            engine.LogitsForBatchRow(lane.LastRow, lane.SparseRows, lane.SparseCount, lane.ScratchLogits);
        switch (lane.Phase)
        {
            case Phase.Reasoning: AdvanceReason(ref lane); break;
            case Phase.ThinkHead: AdvanceThinkHead(ref lane); break;
            case Phase.ThinkTail: AdvanceThinkTail(ref lane); break;
            case Phase.Call: AdvanceCall(ref lane); break;
            case Phase.CallClosing: FinishCall(ref lane); break;
            case Phase.Respond: AdvanceRespond(ref lane); break;
            case Phase.RespondClosing: FinishRespond(ref lane); break;
        }
        lane.First = false;
    }

    void AdvanceReason(ref Lane lane)
    {
        if (lane.Steps >= options.MaxReasoningTokens)
        {
            SetPending(ref lane, ThinkEnd, Want.None);
            lane.Phase = Phase.ThinkHead;
            return;
        }
        var next = ArgMax(reasoningMask, PickLogits(ref lane));
        if (next < 0) { FailLane(ref lane, ReplyStatus.Failed, FailureReason.GrammarDeadEnd); return; }
        if (lane.FirstTokenTicks == 0) lane.FirstTokenTicks = Stopwatch.GetTimestamp() - lane.Started;
        if (next == ThinkEnd)
        {
            SetPending(ref lane, ThinkEnd, Want.None);
            lane.Phase = Phase.ThinkHead;
            return;
        }
        AppendPiece(next, lane.ReasoningLength == 0, lane.Reasoning, ref lane.ReasoningLength, ref lane.Flags, ReplyFlags.ReasoningTruncated);
        lane.Steps++;
        lane.DecodeTokens++;
        if (FrameBudgetHit(ref lane)) return;
        SetPending(ref lane, next, Want.Full);
    }

    void AdvanceThinkHead(ref Lane lane)
    {
        SetPending(ref lane, thinkEndForced.AsSpan(1, 1), Want.Full);
        lane.Phase = Phase.ThinkTail;
    }

    void AdvanceThinkTail(ref Lane lane)
    {
        var decision = ArgMax(decisionMask, engine.BatchLogitsRow(lane.CompactIndex));
        if (decision < 0) { FailLane(ref lane, ReplyStatus.Failed, FailureReason.GrammarDeadEnd); return; }
        lane.Steps = 0;
        if (decision == ImEnd || decision == EndOfText)
        {
            lane.Phase = Phase.RespondClosing;
            SetPending(ref lane, responseClosing, Want.None);
            return;
        }
        if (decision != ToolCallStart)
        {
            lane.Current = decision;
            lane.Phase = Phase.Respond;
            SetPending(ref lane, decision, Want.Full);
            return;
        }
        StartCall(ref lane);
        SetPending(ref lane, decision, Want.Full);
    }

    void StartCall(ref Lane lane)
    {
        lane.Phase = Phase.Call;
        lane.State = Grammar.Start(lane.Grammar.Program);
        trie.Allow(lane.Grammar.Program, lane.State, lane.Mask);
        lane.CurMask = lane.Mask;
        lane.Allowed = MaskCount(lane.Mask, maskWords);
        lane.Full = true;
    }

    void AdvanceCall(ref Lane lane)
    {
        if (lane.Steps >= options.MaxCallTokens) { FailLane(ref lane, ReplyStatus.Failed, FailureReason.CallTokenBudgetExceeded); return; }
        uint* mask = lane.CurMask;
        int pick;
        float probability;
        if (lane.Full) pick = ArgMaxWithProbability(PickLogits(ref lane), mask, lane.ScratchLogits, out probability);
        else pick = ArgMaxSparse(lane.ScratchLogits, lane.SparseRows, lane.SparseCount, out probability);
        if (pick < 0) { FailLane(ref lane, ReplyStatus.Failed, FailureReason.GrammarDeadEnd); return; }
        if (lane.FirstTokenTicks == 0) lane.FirstTokenTicks = Stopwatch.GetTimestamp() - lane.Started;
        Span<byte> buffer = stackalloc byte[256];
        int length;
        fixed (byte* bytes = buffer)
            length = tokenizer.ConstraintBytes(pick, bytes);
        if (lane.CallTextLength + length > lane.CallText.Length) { FailLane(ref lane, ReplyStatus.Failed, FailureReason.CallTextTooLong); return; }
        fixed (byte* bytes = buffer)
            fixed (GrammarState* state = &lane.State)
                for (var b = 0; b < length; b++)
                {
                    if (!Grammar.Advance(lane.Grammar.Program, state, bytes[b])) { FailLane(ref lane, ReplyStatus.Failed, FailureReason.GrammarRejectedToken); return; }
                    lane.CallText[lane.CallTextLength++] = bytes[b];
                }
        lane.LogProb += Math.Log(Math.Max(probability, 1e-9));
        lane.Steps++;
        lane.DecodeTokens++;
        if (FrameBudgetHit(ref lane)) return;
        if (Grammar.Complete(lane.State))
        {
            lane.Phase = Phase.CallClosing;
            SetPending(ref lane, pick, callClosing, Want.None);
            return;
        }
        trie.Allow(lane.Grammar.Program, lane.State, lane.Mask);
        var allowed = MaskCount(lane.Mask, maskWords);
        if (allowed == 0) { FailLane(ref lane, ReplyStatus.Failed, FailureReason.GrammarDeadEnd); return; }
        lane.CurMask = lane.Mask;
        lane.Allowed = allowed;
        if (allowed <= SparseHeadThreshold)
        {
            lane.SparseCount = ExtractIds(lane.Mask, maskWords, lane.SparseRows, SparseHeadThreshold);
            lane.Full = false;
            SetPending(ref lane, pick, Want.Sparse);
        }
        else
        {
            lane.Full = true;
            SetPending(ref lane, pick, Want.Full);
        }
    }

    void AdvanceRespond(ref Lane lane)
    {
        AppendPiece(lane.Current, lane.ResponseLength == 0, lane.Response, ref lane.ResponseLength, ref lane.Flags, ReplyFlags.ResponseTruncated);
        if (lane.Steps >= options.MaxCallTokens) { FailLane(ref lane, ReplyStatus.Failed, FailureReason.CallTokenBudgetExceeded); return; }
        lane.Steps++;
        lane.DecodeTokens++;
        if (FrameBudgetHit(ref lane)) return;
        var next = ArgMax(responseMask, engine.BatchLogitsRow(lane.CompactIndex));
        if (next < 0) { FailLane(ref lane, ReplyStatus.Failed, FailureReason.GrammarDeadEnd); return; }
        if (next == ImEnd || next == EndOfText)
        {
            SetPending(ref lane, responseClosing, Want.None);
            lane.Phase = Phase.RespondClosing;
            return;
        }
        lane.Current = next;
        SetPending(ref lane, next, Want.Full);
    }

    void FinishCall(ref Lane lane)
    {
        // The closing rows were fed by the shared pass just before this
        // advance; only local materialization remains.
        lane.PendingCount = 0;
        var head = HeadProbability(ref lane);
        var decode = Math.Exp(lane.LogProb / Math.Max(1, lane.Steps));
        var confidence = (float)Math.Min(head, decode);
        var calls = lane.Calls;
        if (!CallRestorer.TryRestore(lane.Tools.Aliases, new ReadOnlySpan<byte>(lane.CallText, 0, lane.CallTextLength), calls, lane.KeyScratch, lane.CallSpans, out var required, out var restored))
        {
            FailLane(ref lane, ReplyStatus.Failed, FailureReason.MalformedCalls);
            return;
        }
        lane.RestoredCallCount = restored;
        lane.ElapsedTicks = Stopwatch.GetTimestamp() - lane.Started;
        var refused = confidence < options.RefuseBelow;
        if (required > calls.Length)
        {
            var exact = new byte[required];
            CallRestorer.TryRestore(lane.Tools.Aliases, new ReadOnlySpan<byte>(lane.CallText, 0, lane.CallTextLength), exact, lane.KeyScratch, lane.CallSpans, out _, out _);
            lane.Result = ToCallResult(ref lane, ReplyStatus.Failed, FailureReason.CallsBufferTooSmall, float.NaN, float.NaN, float.NaN, restored, required, exact);
        }
        else
        {
            var status = refused ? ReplyStatus.Refused : ReplyStatus.Called;
            lane.Result = ToCallResult(ref lane, status, FailureReason.None, confidence, (float)head, (float)decode, restored, required, calls);
        }
        lane.Finished = true;
    }

    void FinishRespond(ref Lane lane)
    {
        lane.PendingCount = 0;
        var confidence = (float)HeadProbability(ref lane);
        lane.ElapsedTicks = Stopwatch.GetTimestamp() - lane.Started;
        lane.Result = ToCallResult(ref lane, ReplyStatus.Responded, FailureReason.None, confidence, confidence, float.NaN, 0, 0, Array.Empty<byte>());
        lane.Finished = true;
    }

    void FailLane(ref Lane lane, ReplyStatus status, FailureReason reason)
    {
        engine.RewindSlot(lane.Slot);
        lane.ElapsedTicks = Stopwatch.GetTimestamp() - lane.Started;
        lane.Result = ToCallResult(ref lane, status, reason, float.NaN, float.NaN, float.NaN, 0, 0, Array.Empty<byte>());
        lane.Finished = true;
    }

    float* PickLogits(ref Lane lane) => lane.First ? lane.Logits : engine.BatchLogitsRow(lane.CompactIndex);

    static void SetPending(ref Lane lane, int token, Want want)
    {
        lane.Pending[0] = token;
        lane.PendingCount = 1;
        lane.Want = want;
    }

    static void SetPending(ref Lane lane, int[] tokens, Want want)
    {
        tokens.CopyTo(lane.Pending, 0);
        lane.PendingCount = tokens.Length;
        lane.Want = want;
    }

    static void SetPending(ref Lane lane, int pick, int[] tail, Want want)
    {
        lane.Pending[0] = pick;
        tail.CopyTo(lane.Pending, 1);
        lane.PendingCount = 1 + tail.Length;
        lane.Want = want;
    }

    static void SetPending(ref Lane lane, Span<int> tokens, Want want)
    {
        for (var i = 0; i < tokens.Length; i++) lane.Pending[i] = tokens[i];
        lane.PendingCount = tokens.Length;
        lane.Want = want;
    }

    bool FrameBudgetHit(ref Lane lane)
    {
        var budget = options.FrameBudget;
        if (budget.IsUnlimited) return false;
        if (budget.MaxTokensPerSlice > 0 && lane.DecodeTokens >= budget.MaxTokensPerSlice)
        {
            FailLane(ref lane, ReplyStatus.Failed, FailureReason.FrameBudgetExceeded);
            return true;
        }
        if (budget.MaxMillisecondsPerSlice > 0 &&
            (Stopwatch.GetTimestamp() - lane.Started) * 1000.0 / Stopwatch.Frequency >= budget.MaxMillisecondsPerSlice)
        {
            FailLane(ref lane, ReplyStatus.Failed, FailureReason.FrameBudgetExceeded);
            return true;
        }
        return false;
    }

    double HeadProbability(ref Lane lane)
    {
        if (lane.ConfidenceScratch == null) return 1.0;
        var logit = engine.ConfidenceLogitForSlot(lane.Slot, lane.Pooled, lane.ConfidenceScratch);
        return 1.0 / (1.0 + Math.Exp(-logit));
    }

    int EncodePrompt(ref Lane lane, byte[] input, bool toolResult)
    {
        var head = toolResult ? ToolResultHead : lane.TurnOne ? FirstQueryHead : LaterQueryHead;
        var tail = toolResult ? ToolResultTail : QueryTail;
        var text = new Span<byte>(inputBuffer);
        head.CopyTo(text);
        new ReadOnlySpan<byte>(input).CopyTo(text[head.Length..]);
        tail.CopyTo(text[(head.Length + input.Length)..]);
        var count = tokenizer.Encode(text[..(head.Length + input.Length + tail.Length)], promptIds, false);
        promptIds[count] = ThinkStart;
        return count + 1;
    }

    void Prime(ref Lane lane)
    {
        var head = (system.Length > 0 ? "<|im_start|>system\n" + system + "<|im_end|>\n" : "") + "<|im_start|>user\n<tools>" + lane.Tools.Render() + "</tools>";
        var bytes = Encoding.UTF8.GetBytes(head);
        var count = tokenizer.Encode(bytes, promptIds, true) + 1;
        var ids = new int[count];
        ids[0] = tokenizer.BeginId;
        Array.Copy(promptIds, 0, ids, 1, count - 1);
        engine.ResetSlot(lane.Slot);
        engine.FeedSlot(lane.Slot, ids, false);
        engine.MarkSlot(lane.Slot);
        lane.TurnOne = true;
    }

    int[] Tokens(string text, bool leadingSpace)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        var ids = new int[bytes.Length * 3 + 8];
        var count = tokenizer.Encode(bytes, ids, leadingSpace);
        return ids[..count];
    }

    static int[] Concatenate(int[] head, int[] tail)
    {
        var joined = new int[head.Length + tail.Length];
        head.CopyTo(joined, 0);
        tail.CopyTo(joined, head.Length);
        return joined;
    }

    static string BuildSystem(NeedleOptions o)
    {
        var text = o.System ?? "";
        if (!o.AutoDate || DateKey.IsMatch(text) || IsoStamp.IsMatch(text)) return text;
        var fact = DateTime.Now.ToString("'date: 'yyyy-MM-dd ddd HH:mm", CultureInfo.InvariantCulture);
        if (string.IsNullOrWhiteSpace(text)) return fact;
        return text.TrimStart().StartsWith('{') ? text : fact + "; " + text;
    }

    uint* CopySharedMask(Vocabulary vocabulary)
    {
        var copy = (uint*)Mem.AllocZeroed((nuint)maskWords * 4);
        Buffer.MemoryCopy(vocabulary.ReasoningMask, copy, (nuint)maskWords * 4, (nuint)maskWords * 4);
        return copy;
    }

    uint* CopyMask(uint* source, params int[] extraTokens)
    {
        var copy = (uint*)Mem.Alloc((nuint)maskWords * 4);
        new ReadOnlySpan<uint>(source, maskWords).CopyTo(new Span<uint>(copy, maskWords));
        foreach (var id in extraTokens) copy[id / 32] |= 1u << (id % 32);
        return copy;
    }

    int ArgMax(uint* allowed, float* logits) => Ops.ArgMaxMasked(logits, allowed, maskWords, vocabSize);

    int ArgMaxWithProbability(float* logits, uint* allowed, float* scratch, out float probability)
    {
        var best = -1;
        var count = 0;
        var peak = float.NegativeInfinity;
        for (var word = 0; word < maskWords; word++)
        {
            var bits = allowed[word];
            while (bits != 0)
            {
                var id = word * 32 + Bits.TrailingZeros(bits);
                bits &= bits - 1;
                var value = logits[id];
                scratch[count++] = value;
                if (value > peak) { peak = value; best = id; }
            }
        }
        if (best < 0)
        {
            probability = 0f;
            return -1;
        }
        probability = Ops.BestProbability(scratch, count, peak);
        return best;
    }

    static int ArgMaxSparse(float* values, int* rows, int count, out float probability)
    {
        var best = -1;
        var peak = float.NegativeInfinity;
        for (var i = 0; i < count; i++)
        {
            var value = values[i];
            if (value > peak) { peak = value; best = rows[i]; }
        }
        probability = best < 0 ? 0f : Ops.BestProbability(values, count, peak);
        return best;
    }

    static int MaskCount(uint* mask, int words)
    {
        var total = 0;
        for (var w = 0; w < words; w++) total += Bits.PopCount(mask[w]);
        return total;
    }

    static int ExtractIds(uint* mask, int words, int* rows, int limit)
    {
        var count = 0;
        for (var w = 0; w < words && count < limit; w++)
        {
            var bits = mask[w];
            while (bits != 0 && count < limit)
            {
                rows[count++] = w * 32 + Bits.TrailingZeros(bits);
                bits &= bits - 1;
            }
        }
        return count;
    }

    void AppendPiece(int token, bool first, byte[] destination, ref int length, ref ReplyFlags flags, ReplyFlags truncated)
    {
        if (destination.Length == 0) return;
        Span<byte> piece = stackalloc byte[MaxPieceBytes];
        var count = tokenizer.Append(token, piece, first);
        var room = destination.Length - length;
        if (count > room) flags |= truncated;
        var written = Math.Min(count, room);
        piece[..written].CopyTo(destination[length..]);
        length += written;
    }

    CallResult ToCallResult(ref Lane lane, ReplyStatus status, FailureReason failure, float confidence, float head, float decode, int callCount, int required, byte[] calls)
    {
        var ticksPerSecond = (double)Stopwatch.Frequency;
        var reasoning = Encoding.UTF8.GetString(lane.Reasoning, 0, lane.ReasoningLength).Trim();
        if (status == ReplyStatus.Cancelled) return new CallResult { Success = false, Error = "cancelled", Reasoning = reasoning };
        if (status == ReplyStatus.Failed && failure != FailureReason.CallsBufferTooSmall)
            return new CallResult { Success = false, Error = Describe(failure), Reasoning = reasoning };

        var timing = new CallResult
        {
            PrefillTokens = lane.PromptTokens,
            DecodeTokens = lane.DecodeTokens,
            TimeToFirstTokenMs = lane.FirstTokenTicks * 1000.0 / ticksPerSecond,
            PrefillTps = lane.PromptTokens / Math.Max(1e-9, lane.PrefillTicks / ticksPerSecond),
            DecodeTps = lane.DecodeTokens / Math.Max(1e-9, (lane.ElapsedTicks - lane.PrefillTicks) / ticksPerSecond),
            TotalMs = lane.ElapsedTicks * 1000.0 / ticksPerSecond,
        };
        if (status == ReplyStatus.Responded)
            return new CallResult
            {
                Reasoning = reasoning,
                Response = Encoding.UTF8.GetString(lane.Response, 0, lane.ResponseLength).Trim(),
                Confidence = confidence,
                PrefillTokens = timing.PrefillTokens,
                DecodeTokens = timing.DecodeTokens,
                TimeToFirstTokenMs = timing.TimeToFirstTokenMs,
                PrefillTps = timing.PrefillTps,
                DecodeTps = timing.DecodeTps,
                TotalMs = timing.TotalMs,
            };
        var resultCalls = new FunctionCall[callCount];
        for (var i = 0; i < resultCalls.Length; i++)
        {
            var span = lane.CallSpans[i];
            resultCalls[i] = new FunctionCall(lane.Tools.Aliases.OriginalTool(span.Tool), Encoding.UTF8.GetString(calls, span.ArgumentsStart, span.ArgumentsLength));
        }
        var refused = status == ReplyStatus.Refused;
        return new CallResult
        {
            FunctionCalls = refused ? Array.Empty<FunctionCall>() : resultCalls,
            SuppressedCalls = refused ? resultCalls : Array.Empty<FunctionCall>(),
            Reasoning = reasoning,
            Confidence = confidence,
            HeadConfidence = head,
            DecodeConfidence = decode,
            PrefillTokens = timing.PrefillTokens,
            DecodeTokens = timing.DecodeTokens,
            TimeToFirstTokenMs = timing.TimeToFirstTokenMs,
            PrefillTps = timing.PrefillTps,
            DecodeTps = timing.DecodeTps,
            TotalMs = timing.TotalMs,
        };
    }

    static string Describe(FailureReason reason) => reason switch
    {
        FailureReason.EmptyInput => "the query is empty",
        FailureReason.InputTooLong => "the input exceeds the maximum size",
        FailureReason.ContextExhausted => "context capacity exceeded",
        FailureReason.CallTokenBudgetExceeded => "call exceeded the token budget",
        FailureReason.CallTextTooLong => "the generated call exceeds the call text buffer",
        FailureReason.GrammarDeadEnd => "grammar admits no token",
        FailureReason.GrammarRejectedToken => "grammar rejected the generated token",
        FailureReason.MalformedCalls => "the generated calls are malformed",
        FailureReason.FrameBudgetExceeded => "frame budget exceeded",
        _ => "the request failed",
    };

    public void Dispose()
    {
        for (var i = 0; i < laneCount; i++)
        {
            lanes[i].Grammar.Dispose();
            Mem.Free(lanes[i].Mask);
            Mem.Free(lanes[i].SparseRows);
            Mem.Free(lanes[i].ScratchLogits);
            Mem.Free(lanes[i].Logits);
            Mem.Free(lanes[i].Pooled);
            if (lanes[i].ConfidenceScratch != null) Mem.Free(lanes[i].ConfidenceScratch);
        }
        engine.Dispose();
        tokenizer.Dispose();
        trie.Dispose();
        Mem.Free(reasoningMask);
        Mem.Free(decisionMask);
        Mem.Free(responseMask);
    }
}
