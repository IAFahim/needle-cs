using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;

namespace Needle;

public sealed class NeedleOptions
{
    public int Threads { get; init; } = 1;
    public ThreadPolicy? Threading { get; init; }
    public int Context { get; init; } = 2048;
    public int MaxReasoningTokens { get; init; } = 96;
    public int MaxCallTokens { get; init; } = 384;
    public string? System { get; init; }
    public bool AutoDate { get; init; }
    public float RefuseBelow { get; init; } = 0.1f;
    public int RetrievalTopK { get; init; } = 5;

    internal ThreadPolicy ResolvedThreading => Threading ?? ThreadPolicy.Default.WithThreads(Threads);
}

public sealed record FunctionCall(string Name, string ArgumentsJson);

public sealed class CallResult
{
    public bool Success { get; init; } = true;
    public string? Error { get; init; }
    public IReadOnlyList<FunctionCall> FunctionCalls { get; init; } = Array.Empty<FunctionCall>();
    public IReadOnlyList<FunctionCall> SuppressedCalls { get; init; } = Array.Empty<FunctionCall>();
    public string Reasoning { get; init; } = "";
    public string Response { get; init; } = "";
    public float? Confidence { get; init; }
    public float? HeadConfidence { get; init; }
    public float? DecodeConfidence { get; init; }
    public int PrefillTokens { get; init; }
    public int DecodeTokens { get; init; }
    public double TimeToFirstTokenMs { get; init; }
    public double PrefillTps { get; init; }
    public double DecodeTps { get; init; }
    public double TotalMs { get; init; }

    public string ToJson()
    {
        var b = new StringBuilder("{\"success\":");
        b.Append(Success ? "true" : "false").Append(",\"error\":");
        if (Error == null) b.Append("null"); else JsonValue.WriteString(b, Error);
        b.Append(",\"type\":\"").Append(FunctionCalls.Count > 0 ? "call" : "respond").Append("\",\"function_calls\":");
        WriteCalls(b, FunctionCalls);
        b.Append(",\"suppressed_calls\":");
        WriteCalls(b, SuppressedCalls);
        b.Append(",\"reasoning\":");
        JsonValue.WriteString(b, Reasoning);
        b.Append(",\"response\":");
        JsonValue.WriteString(b, Response);
        b.Append(",\"confidence\":").Append(Confidence is { } c ? c.ToString("0.####", CultureInfo.InvariantCulture) : "null");
        b.Append(",\"confidence_head\":").Append(HeadConfidence is { } h ? h.ToString("0.####", CultureInfo.InvariantCulture) : "null");
        b.Append(",\"confidence_decode\":").Append(DecodeConfidence is { } d ? d.ToString("0.####", CultureInfo.InvariantCulture) : "null");
        b.Append(",\"time_to_first_token_ms\":").Append(TimeToFirstTokenMs.ToString("0.##", CultureInfo.InvariantCulture));
        b.Append(",\"prefill_tokens\":").Append(PrefillTokens).Append(",\"prefill_tps\":").Append(PrefillTps.ToString("0.#", CultureInfo.InvariantCulture));
        b.Append(",\"decode_tokens\":").Append(DecodeTokens).Append(",\"decode_tps\":").Append(DecodeTps.ToString("0.#", CultureInfo.InvariantCulture));
        b.Append(",\"total_time_ms\":").Append(TotalMs.ToString("0.##", CultureInfo.InvariantCulture)).Append('}');
        return b.ToString();
    }

    static void WriteCalls(StringBuilder b, IReadOnlyList<FunctionCall> calls)
    {
        b.Append('[');
        for (var i = 0; i < calls.Count; i++)
        {
            if (i > 0) b.Append(',');
            b.Append("{\"name\":");
            JsonValue.WriteString(b, calls[i].Name);
            b.Append(",\"arguments\":").Append(calls[i].ArgumentsJson).Append('}');
        }
        b.Append(']');
    }
}

public static class Extraction
{
    public static string ToolsJson(string schemaJson, string name = "extract", string description = "Extract the fields from the text.")
    {
        var schema = JsonValue.Parse(schemaJson);
        var tool = JsonValue.Object(new[]
        {
            new KeyValuePair<string, JsonValue>("name", JsonValue.Str(schema.GetString("title") ?? name)),
            new KeyValuePair<string, JsonValue>("description", JsonValue.Str(schema.GetString("description") ?? description)),
            new KeyValuePair<string, JsonValue>("parameters", schema),
        });
        return JsonValue.Array(new[] { tool }).ToString();
    }
}

public sealed unsafe partial class NeedleSession : IDisposable
{
    const int ImStart = 4, ImEnd = 5, ThinkStart = 6, ThinkEnd = 7, ToolCallStart = 10, ToolCallEnd = 11, EndOfText = 1;
    const int MaxPieceBytes = 64;
    const int InputOverheadBytes = 128;
    const int TokenizerInputBytes = 1 << 16;
    const int EmbedderContext = 512;
    const int MinKeyScratchBytes = 64;

    static readonly Regex DateKey = new(@"\bdate\s*:", RegexOptions.IgnoreCase);
    static readonly Regex IsoStamp = new(@"\d{4}-\d{2}-\d{2}");

    static ReadOnlySpan<byte> ToolResultHead => "<|im_start|>tool\n<tool_result>"u8;
    static ReadOnlySpan<byte> ToolResultTail => "</tool_result><|im_end|>\n<|im_start|>assistant\n"u8;
    static ReadOnlySpan<byte> FirstQueryHead => "\n"u8;
    static ReadOnlySpan<byte> LaterQueryHead => "<|im_start|>user\n"u8;
    static ReadOnlySpan<byte> QueryTail => "<|im_end|>\n<|im_start|>assistant\n"u8;

    struct TurnProgress
    {
        public long Started, PrefillTicks, FirstTokenTicks;
        public int PromptTokens, DecodeTokens, ReasoningLength, ResponseLength;
        public ReplyFlags Flags;
    }

    readonly Model model;
    readonly NeedleOptions options;
    readonly ToolSet allTools;
    readonly Engine engine;
    readonly Tokenizer tokenizer;
    readonly TokenTrie trie;
    readonly string system;
    readonly int maskWords;
    readonly uint* reasoningMask;
    readonly uint* decisionMask;
    readonly uint* responseMask;
    readonly uint* mask;
    readonly float* scratchLogits;
    readonly float* pooled;
    readonly float* confidenceScratch;
    readonly int embeddingWidth;
    readonly int contextCapacity;
    readonly int embedderCapacity;
    readonly int maxInputBytes;
    readonly byte[] inputBuffer;
    readonly int[] promptIds;
    readonly int[] thinkEndForced;
    readonly int[] callClosing;
    readonly int[] responseClosing;
    readonly byte[] callText;
    readonly byte[] keyScratch;
    readonly CallSpan[] callSpans;
    readonly bool retrieval;
    readonly float[] toolEmbeddings;
    readonly float[] toolScores;
    readonly int[] chosenTools;
    readonly int[] activeTools;
    int activeToolCount = -1;
    int callTextLength;
    int restoredCallCount;
    int cancelRequested;
    Engine? embedder;
    ToolSet active;
    Grammar grammar;
    bool turnOne = true;

    public NeedleSession(Model model, string toolsJson, NeedleOptions? options = null)
        : this(model, toolsJson, null, default, options)
    {
    }

    public NeedleSession(Model model, string toolsJson, ReadOnlySpan<byte> primedPrefix, NeedleOptions? options = null)
        : this(model, toolsJson, null, primedPrefix, options)
    {
    }

    internal NeedleSession(Model model, string toolsJson, GrammarProgram* precompiled, ReadOnlySpan<byte> primedPrefix, NeedleOptions? options)
    {
        this.model = model;
        this.options = options ?? new NeedleOptions();
        allTools = new ToolSet(toolsJson);
        engine = new Engine(model, this.options.Context, 64, this.options.ResolvedThreading, null) { Pooling = true };
        var vocabulary = model.GetVocabulary();
        tokenizer = new Tokenizer(vocabulary, TokenizerInputBytes);
        trie = new TokenTrie(vocabulary.Trie);
        system = BuildSystem(this.options);
        contextCapacity = Math.Min(this.options.Context, model.Geometry.MaxSeq);
        embedderCapacity = Math.Min(EmbedderContext, model.Geometry.MaxSeq);
        maxInputBytes = Math.Max(1, Math.Min(TokenizerInputBytes - InputOverheadBytes, this.options.Context * 16));
        maskWords = trie.Words;
        mask = (uint*)Mem.AllocZeroed((nuint)maskWords * 4);
        scratchLogits = (float*)Mem.Alloc((nuint)model.VocabSize * 4);
        var head = model.Confidence;
        embeddingWidth = head != null ? head->Queries * model.Geometry.Dim : 0;
        pooled = (float*)Mem.AllocZeroed((nuint)Math.Max(1, embeddingWidth) * 4);
        confidenceScratch = head != null ? (float*)Mem.AllocZeroed((nuint)((model.Geometry.Layers + 1) * head->Probes * (model.Geometry.Dim + 1)) * 4) : null;
        reasoningMask = CopySharedMask(vocabulary);
        decisionMask = CopyMask(reasoningMask, ToolCallStart, ImEnd, EndOfText);
        responseMask = CopyMask(reasoningMask, ImEnd, EndOfText);
        responseMask[ThinkEnd / 32] &= ~(1u << (ThinkEnd % 32));
        var newline = Tokens("\n", false);
        thinkEndForced = Concatenate(new[] { ThinkEnd }, newline);
        callClosing = Concatenate(new[] { ToolCallEnd, ImEnd }, newline);
        responseClosing = Concatenate(new[] { ImEnd }, newline);
        inputBuffer = new byte[maxInputBytes + InputOverheadBytes];
        promptIds = new int[3 * (maxInputBytes + InputOverheadBytes) + 16];
        callText = new byte[Math.Max(1, this.options.MaxCallTokens) * MaxPieceBytes];
        callSpans = new CallSpan[Math.Max(1, this.options.MaxCallTokens)];
        keyScratch = new byte[Math.Max(MinKeyScratchBytes, allTools.Aliases.LongestPropertyAlias + 2)];
        retrieval = allTools.Tools.Count > this.options.RetrievalTopK && model.Confidence != null;
        var topK = Math.Max(0, this.options.RetrievalTopK);
        toolEmbeddings = retrieval ? new float[allTools.Tools.Count * embeddingWidth] : Array.Empty<float>();
        toolScores = retrieval ? new float[allTools.Tools.Count] : Array.Empty<float>();
        chosenTools = new int[topK];
        activeTools = new int[topK];
        active = allTools;
        grammar = precompiled != null ? new Grammar(precompiled) : new Grammar(active.Tools);
        try
        {
            if (retrieval)
            {
                if (primedPrefix.Length > 0) throw new ArgumentException("a primed prefix cannot be combined with tool retrieval; raise RetrievalTopK or drop the prefix");
                IndexTools();
            }
            else if (primedPrefix.Length > 0) ImportPrimedPrefix(primedPrefix);
            else Prime(active);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public int ContextUsed => engine.Length;

    public int MaxInputBytes => maxInputBytes;

    public void RequestCancel() => System.Threading.Volatile.Write(ref cancelRequested, 1);

    static string BuildSystem(NeedleOptions o)
    {
        var text = o.System ?? "";
        if (!o.AutoDate || DateKey.IsMatch(text) || IsoStamp.IsMatch(text)) return text;
        var fact = DateTime.Now.ToString("'date: 'yyyy-MM-dd ddd HH:mm", CultureInfo.InvariantCulture);
        if (string.IsNullOrWhiteSpace(text)) return fact;
        return text.TrimStart().StartsWith('{') ? text : fact + "; " + text;
    }


    uint* CopyMask(uint* source, params int[] extraTokens)
    {
        var copy = (uint*)Mem.Alloc((nuint)maskWords * 4);
        new ReadOnlySpan<uint>(source, maskWords).CopyTo(new Span<uint>(copy, maskWords));
        foreach (var id in extraTokens) copy[id / 32] |= 1u << (id % 32);
        return copy;
    }

    static int[] Concatenate(int[] head, int[] tail)
    {
        var joined = new int[head.Length + tail.Length];
        head.CopyTo(joined, 0);
        tail.CopyTo(joined, head.Length);
        return joined;
    }

    int[] Tokens(string text, bool leadingSpace)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        var ids = new int[bytes.Length * 3 + 8];
        var count = tokenizer.Encode(bytes, ids, leadingSpace);
        return ids[..count];
    }

    void Prime(ToolSet tools)
    {
        var head = (system.Length > 0 ? "<|im_start|>system\n" + system + "<|im_end|>\n" : "") + "<|im_start|>user\n<tools>" + tools.Render() + "</tools>";
        var ids = new List<int> { tokenizer.BeginId };
        ids.AddRange(Tokens(head, true));
        engine.Reset();
        engine.Feed(ids.ToArray(), computeLogits: false);
        engine.Mark();
        turnOne = true;
    }

    public void Reset()
    {
        engine.Rewind();
        turnOne = true;
    }

    const int PrefixMagic = 0x3150444E;
    const int PrefixVersion = 1;
    const int PrefixHeaderInts = 10;

    static uint PrefixHash(string text)
    {
        var hash = 2166136261u;
        foreach (var b in Encoding.UTF8.GetBytes(text)) hash = (hash ^ b) * 16777619u;
        return hash;
    }

    void WritePrefixHeader(Span<byte> destination)
    {
        var ints = MemoryMarshal.Cast<byte, int>(destination.Slice(0, PrefixHeaderInts * sizeof(int)));
        ints[0] = PrefixMagic;
        ints[1] = PrefixVersion;
        ints[2] = model.Geometry.Dim;
        ints[3] = model.Geometry.Layers;
        ints[4] = model.VocabSize;
        ints[5] = contextCapacity;
        ints[6] = (int)PrefixHash(allTools.Render());
        ints[7] = (int)PrefixHash(system);
        ints[8] = engine.MarkedLength;
    }

    public byte[] ExportPrimedPrefix()
    {
        var bytes = new byte[PrefixHeaderInts * sizeof(int) + (int)engine.PrimedPrefixBytes];
        WritePrefixHeader(bytes);
        engine.ExportPrimedPrefix(bytes.AsSpan(PrefixHeaderInts * sizeof(int)));
        return bytes;
    }

    void ImportPrimedPrefix(ReadOnlySpan<byte> primedPrefix)
    {
        if (primedPrefix.Length < PrefixHeaderInts * sizeof(int)) throw new ArgumentException("primed prefix is truncated");
        var ints = MemoryMarshal.Cast<byte, int>(primedPrefix.Slice(0, PrefixHeaderInts * sizeof(int)));
        if (ints[0] != PrefixMagic || ints[1] != PrefixVersion) throw new ArgumentException("primed prefix format is not recognized");
        if (ints[2] != model.Geometry.Dim || ints[3] != model.Geometry.Layers || ints[4] != model.VocabSize ||
            ints[5] != contextCapacity || ints[6] != (int)PrefixHash(allTools.Render()) || ints[7] != (int)PrefixHash(system))
            throw new ArgumentException("primed prefix was created for a different model, tool set, system prompt or context");
        engine.ImportPrimedPrefix(primedPrefix.Slice(PrefixHeaderInts * sizeof(int)));
        turnOne = true;
    }

    bool warmed;

    public void WarmUp()
    {
        if (warmed) return;
        warmed = true;
        _ = Complete("Warm up the engine with a short greeting.");
        Reset();
    }

    uint* CopySharedMask(Vocabulary vocabulary)
    {
        var copy = (uint*)Mem.Alloc((nuint)maskWords * 4);
        Buffer.MemoryCopy(vocabulary.ReasoningMask, copy, (nuint)maskWords * 4, (nuint)maskWords * 4);
        return copy;
    }

    public Reply Complete(ReadOnlySpan<byte> utf8Query, Span<byte> utf8Calls) =>
        Run(utf8Query, false, utf8Calls, default, default);

    public Reply Complete(ReadOnlySpan<byte> utf8Query, Span<byte> utf8Calls, Span<byte> utf8Reasoning, Span<byte> utf8Response) =>
        Run(utf8Query, false, utf8Calls, utf8Reasoning, utf8Response);

    public Reply CompleteWithToolResult(ReadOnlySpan<byte> utf8Json, Span<byte> utf8Calls) =>
        Run(utf8Json, true, utf8Calls, default, default);

    public Reply CompleteWithToolResult(ReadOnlySpan<byte> utf8Json, Span<byte> utf8Calls, Span<byte> utf8Reasoning, Span<byte> utf8Response) =>
        Run(utf8Json, true, utf8Calls, utf8Reasoning, utf8Response);

    Reply Run(ReadOnlySpan<byte> input, bool toolResult, Span<byte> calls, Span<byte> reasoning, Span<byte> response)
    {
        var progress = new TurnProgress { Started = Stopwatch.GetTimestamp() };
        System.Threading.Volatile.Write(ref cancelRequested, 0);
        if (!toolResult && input.Length == 0) return Failed(FailureReason.EmptyInput, progress);
        if (input.Length > maxInputBytes) return Failed(FailureReason.InputTooLong, progress);
        if (retrieval && !toolResult)
        {
            var selection = SelectTools(input);
            if (selection != FailureReason.None) return Failed(selection, progress);
        }
        progress.PromptTokens = EncodePrompt(input, toolResult);
        if (engine.Length + progress.PromptTokens > contextCapacity) return Failed(FailureReason.ContextExhausted, progress);
        turnOne = false;
        engine.Feed(new ReadOnlySpan<int>(promptIds, 0, progress.PromptTokens));
        progress.PrefillTicks = Stopwatch.GetTimestamp() - progress.Started;
        return Generate(ref progress, calls, reasoning, response);
    }

    int EncodePrompt(ReadOnlySpan<byte> input, bool toolResult)
    {
        var head = toolResult ? ToolResultHead : turnOne ? FirstQueryHead : LaterQueryHead;
        var tail = toolResult ? ToolResultTail : QueryTail;
        var text = new Span<byte>(inputBuffer);
        head.CopyTo(text);
        input.CopyTo(text[head.Length..]);
        tail.CopyTo(text[(head.Length + input.Length)..]);
        var count = tokenizer.Encode(text[..(head.Length + input.Length + tail.Length)], promptIds, false);
        promptIds[count] = ThinkStart;
        return count + 1;
    }

    Reply Generate(ref TurnProgress progress, Span<byte> calls, Span<byte> reasoning, Span<byte> response)
    {
        var firstPiece = true;
        for (var i = 0; i < options.MaxReasoningTokens; i++)
        {
            if (CancelRequested) return Abandon(ReplyStatus.Cancelled, FailureReason.None, progress);
            var next = ArgMax(reasoningMask);
            if (next < 0) return Abandon(ReplyStatus.Failed, FailureReason.GrammarDeadEnd, progress);
            if (i == 0) progress.FirstTokenTicks = Stopwatch.GetTimestamp() - progress.Started;
            if (next == ThinkEnd) break;
            AppendPiece(next, firstPiece, reasoning, ref progress.ReasoningLength, ref progress.Flags, ReplyFlags.ReasoningTruncated);
            firstPiece = false;
            if (!TryFeed(next)) return Abandon(ReplyStatus.Failed, FailureReason.ContextExhausted, progress);
            progress.DecodeTokens++;
        }
        if (!TryFeed(thinkEndForced)) return Abandon(ReplyStatus.Failed, FailureReason.ContextExhausted, progress);
        var decision = ArgMax(decisionMask);
        if (decision < 0) return Abandon(ReplyStatus.Failed, FailureReason.GrammarDeadEnd, progress);
        if (decision != ToolCallStart) return Respond(decision, ref progress, response);
        if (!TryFeed(ToolCallStart)) return Abandon(ReplyStatus.Failed, FailureReason.ContextExhausted, progress);
        return GenerateCalls(ref progress, calls);
    }

    Reply GenerateCalls(ref TurnProgress progress, Span<byte> calls)
    {
        var state = Grammar.Start(grammar.Program);
        var logProbability = 0.0;
        var steps = 0;
        var buffer = stackalloc byte[256];
        callTextLength = 0;
        while (!Grammar.Complete(state))
        {
            if (CancelRequested) return Abandon(ReplyStatus.Cancelled, FailureReason.None, progress);
            if (steps >= options.MaxCallTokens) return Abandon(ReplyStatus.Failed, FailureReason.CallTokenBudgetExceeded, progress);
            trie.Allow(grammar.Program, state, mask);
            var pick = ArgMaxWithProbability(mask, out var probability);
            if (pick < 0) return Abandon(ReplyStatus.Failed, FailureReason.GrammarDeadEnd, progress);
            if (steps == 0 && progress.FirstTokenTicks == 0) progress.FirstTokenTicks = Stopwatch.GetTimestamp() - progress.Started;
            var length = tokenizer.ConstraintBytes(pick, buffer);
            if (callTextLength + length > callText.Length) return Abandon(ReplyStatus.Failed, FailureReason.CallTextTooLong, progress);
            var cursor = &state;
            for (var b = 0; b < length; b++)
            {
                if (!Grammar.Advance(grammar.Program, cursor, buffer[b])) return Abandon(ReplyStatus.Failed, FailureReason.GrammarRejectedToken, progress);
                callText[callTextLength++] = buffer[b];
            }
            logProbability += Math.Log(Math.Max(probability, 1e-9));
            steps++;
            progress.DecodeTokens++;
            if (!TryFeed(pick, computeLogits: !Grammar.Complete(state))) return Abandon(ReplyStatus.Failed, FailureReason.ContextExhausted, progress);
        }
        if (!TryFeed(callClosing, computeLogits: false)) return Abandon(ReplyStatus.Failed, FailureReason.ContextExhausted, progress);

        var head = HeadProbability();
        var decode = Math.Exp(logProbability / Math.Max(1, steps));
        var confidence = (float)Math.Min(head, decode);
        if (!CallRestorer.TryRestore(allTools.Aliases, new ReadOnlySpan<byte>(callText, 0, callTextLength), calls, keyScratch, callSpans, out var required, out restoredCallCount))
            return Failed(FailureReason.MalformedCalls, progress);
        var refused = confidence < options.RefuseBelow;
        var status = required > calls.Length ? ReplyStatus.Failed : refused ? ReplyStatus.Refused : ReplyStatus.Called;
        var failure = required > calls.Length ? FailureReason.CallsBufferTooSmall : FailureReason.None;
        return new Reply(status, failure, progress.Flags, confidence, (float)head, (float)decode, restoredCallCount, required,
            progress.ReasoningLength, 0, progress.PromptTokens, progress.DecodeTokens, progress.PrefillTicks, progress.FirstTokenTicks, Stopwatch.GetTimestamp() - progress.Started);
    }

    Reply Respond(int first, ref TurnProgress progress, Span<byte> response)
    {
        var next = first;
        for (var i = 0; i < options.MaxCallTokens && next != ImEnd && next != EndOfText; i++)
        {
            if (CancelRequested) return Abandon(ReplyStatus.Cancelled, FailureReason.None, progress);
            AppendPiece(next, i == 0, response, ref progress.ResponseLength, ref progress.Flags, ReplyFlags.ResponseTruncated);
            if (!TryFeed(next)) return Abandon(ReplyStatus.Failed, FailureReason.ContextExhausted, progress);
            progress.DecodeTokens++;
            next = ArgMax(responseMask);
            if (next < 0) return Abandon(ReplyStatus.Failed, FailureReason.GrammarDeadEnd, progress);
        }
        if (!TryFeed(responseClosing, computeLogits: false)) return Abandon(ReplyStatus.Failed, FailureReason.ContextExhausted, progress);
        var confidence = (float)HeadProbability();
        return new Reply(ReplyStatus.Responded, FailureReason.None, progress.Flags, confidence, confidence, float.NaN, 0, 0,
            progress.ReasoningLength, progress.ResponseLength, progress.PromptTokens, progress.DecodeTokens, progress.PrefillTicks, progress.FirstTokenTicks, Stopwatch.GetTimestamp() - progress.Started);
    }

    bool CancelRequested => System.Threading.Volatile.Read(ref cancelRequested) != 0;

    double HeadProbability() =>
        confidenceScratch == null ? 1.0 : 1.0 / (1.0 + Math.Exp(-engine.ConfidenceLogit(pooled, confidenceScratch)));

    Reply Failed(FailureReason reason, in TurnProgress progress) =>
        new(ReplyStatus.Failed, reason, progress.Flags, float.NaN, float.NaN, float.NaN, 0, 0,
            progress.ReasoningLength, progress.ResponseLength, progress.PromptTokens, progress.DecodeTokens, progress.PrefillTicks, progress.FirstTokenTicks, Stopwatch.GetTimestamp() - progress.Started);

    Reply Abandon(ReplyStatus status, FailureReason reason, in TurnProgress progress)
    {
        Reset();
        var reply = Failed(reason, progress);
        return status == ReplyStatus.Cancelled ? reply with { Status = ReplyStatus.Cancelled } : reply;
    }

    bool TryFeed(ReadOnlySpan<int> tokens, bool computeLogits = true)
    {
        if (engine.Length + tokens.Length > contextCapacity) return false;
        engine.Feed(tokens, computeLogits);
        return true;
    }

    bool TryFeed(int token, bool computeLogits = true)
    {
        var one = stackalloc int[1];
        one[0] = token;
        return TryFeed(new ReadOnlySpan<int>(one, 1), computeLogits);
    }

    void AppendPiece(int token, bool first, Span<byte> destination, ref int length, ref ReplyFlags flags, ReplyFlags truncated)
    {
        if (destination.IsEmpty) return;
        Span<byte> piece = stackalloc byte[MaxPieceBytes];
        var count = tokenizer.Append(token, piece, first);
        var room = destination.Length - length;
        if (count > room) flags |= truncated;
        var written = Math.Min(count, room);
        piece[..written].CopyTo(destination[length..]);
        length += written;
    }

    int ArgMax(uint* allowed)
    {
        fixed (float* logits = engine.Logits)
        {
            var best = -1;
            var peak = float.NegativeInfinity;
            for (var word = 0; word < maskWords; word++)
            {
                var bits = allowed[word];
                while (bits != 0)
                {
                    var id = word * 32 + Bits.TrailingZeros(bits);
                    bits &= bits - 1;
                    var value = logits[id];
                    if (value > peak) { peak = value; best = id; }
                }
            }
            return best;
        }
    }

    int ArgMaxWithProbability(uint* allowed, out float probability)
    {
        fixed (float* logits = engine.Logits)
        {
            var best = -1;
            var bestSlot = 0;
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
                    scratchLogits[count] = value;
                    if (value > peak) { peak = value; best = id; bestSlot = count; }
                    count++;
                }
            }
            if (best < 0)
            {
                probability = 0f;
                return -1;
            }
            Ops.Softmax(scratchLogits, count);
            probability = scratchLogits[bestSlot];
            return best;
        }
    }

    public float[] Embed(string text)
    {
        EnsureEmbedder();
        var ids = new int[Encoding.UTF8.GetByteCount(text) * 3 + 9];
        ids[0] = tokenizer.BeginId;
        var count = 1 + tokenizer.Encode(Encoding.UTF8.GetBytes(text), ids.AsSpan(1), true);
        if (EmbedTokens(new ReadOnlySpan<int>(ids, 0, count)) != FailureReason.None) throw new InvalidOperationException("context capacity exceeded");
        var result = new float[embeddingWidth];
        new ReadOnlySpan<float>(pooled, embeddingWidth).CopyTo(result);
        return result;
    }

    void EnsureEmbedder()
    {
        if (model.Confidence == null) throw new InvalidOperationException("the model has no confidence head");
        embedder ??= new Engine(model, EmbedderContext, 64, options.ResolvedThreading, engine.SharedPool) { Pooling = true };
    }

    FailureReason EmbedTokens(ReadOnlySpan<int> ids)
    {
        if (ids.Length > embedderCapacity) return FailureReason.ContextExhausted;
        embedder!.Reset();
        embedder.Feed(ids, computeLogits: false);
        embedder.ConfidenceLogit(pooled, confidenceScratch);
        var norm = MathF.Sqrt(Ops.SumOfSquares(pooled, embeddingWidth)) + 1e-12f;
        for (var i = 0; i < embeddingWidth; i++) pooled[i] /= norm;
        return FailureReason.None;
    }

    FailureReason EmbedQuery(ReadOnlySpan<byte> query)
    {
        promptIds[0] = tokenizer.BeginId;
        var count = 1 + tokenizer.Encode(query, new Span<int>(promptIds, 1, promptIds.Length - 1), true);
        return EmbedTokens(new ReadOnlySpan<int>(promptIds, 0, count));
    }

    void IndexTools()
    {
        EnsureEmbedder();
        for (var t = 0; t < allTools.Tools.Count; t++)
            Embed(allTools.Tools[t].ToString()).CopyTo(toolEmbeddings, t * embeddingWidth);
    }

    FailureReason SelectTools(ReadOnlySpan<byte> query)
    {
        var embedded = EmbedQuery(query);
        if (embedded != FailureReason.None) return embedded;
        var toolCount = allTools.Tools.Count;
        for (var t = 0; t < toolCount; t++)
        {
            var dot = 0f;
            var row = t * embeddingWidth;
            for (var j = 0; j < embeddingWidth; j++) dot += pooled[j] * toolEmbeddings[row + j];
            toolScores[t] = dot;
        }
        for (var k = 0; k < chosenTools.Length; k++)
        {
            var best = -1;
            for (var t = 0; t < toolCount; t++)
                if (!IsChosen(t, k) && (best < 0 || toolScores[t] > toolScores[best])) best = t;
            chosenTools[k] = best;
        }
        Array.Sort(chosenTools);
        if (activeToolCount == chosenTools.Length && chosenTools.AsSpan().SequenceEqual(activeTools)) return FailureReason.None;
        return RestrictTools();
    }

    bool IsChosen(int tool, int chosenCount)
    {
        for (var k = 0; k < chosenCount; k++) if (chosenTools[k] == tool) return true;
        return false;
    }

    FailureReason RestrictTools()
    {
        try
        {
            activeToolCount = -1;
            var subset = allTools.Subset(chosenTools);
            var rebuilt = new Grammar(subset.Tools);
            grammar.Dispose();
            grammar = rebuilt;
            active = subset;
            Prime(active);
            chosenTools.CopyTo(activeTools, 0);
            activeToolCount = chosenTools.Length;
            return FailureReason.None;
        }
        catch (Exception e) when (e is InvalidOperationException or FormatException or ArgumentException)
        {
            return FailureReason.ToolContextFailed;
        }
    }

    public void Pause() => engine.Pause();

    public void Resume() => engine.Resume();

    public void Dispose()
    {
        engine.Dispose();
        embedder?.Dispose();
        tokenizer.Dispose();
        trie.Dispose();
        grammar.Dispose();
        Mem.Free(mask);
        Mem.Free(reasoningMask);
        Mem.Free(decisionMask);
        Mem.Free(responseMask);
        Mem.Free(scratchLogits);
        Mem.Free(pooled);
        if (confidenceScratch != null) Mem.Free(confidenceScratch);
    }
}
