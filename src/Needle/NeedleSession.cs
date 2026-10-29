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
    public bool AutoDate { get; init; } = true;
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

public sealed unsafe class NeedleSession : IDisposable
{
    const int ImStart = 4, ImEnd = 5, ThinkStart = 6, ThinkEnd = 7, ToolCallStart = 10, ToolCallEnd = 11;
    static readonly Regex DateKey = new(@"\bdate\s*:", RegexOptions.IgnoreCase);
    static readonly Regex IsoStamp = new(@"\d{4}-\d{2}-\d{2}");

    readonly Model model;
    readonly NeedleOptions options;
    readonly ToolSet allTools;
    readonly Engine engine;
    readonly Tokenizer tokenizer;
    readonly TokenTrie trie;
    readonly string system;
    readonly uint[] reasoningMask;
    readonly uint[] decisionMask;
    readonly uint[] responseMask;
    readonly uint* mask;
    readonly float* scratchLogits;
    readonly float* pooled;
    readonly bool retrieval;
    readonly List<float[]> toolEmbeddings = new();
    Engine? embedder;
    ToolSet active;
    Grammar grammar;
    bool turnOne = true;
    string primedRender = "";

    public NeedleSession(Model model, string toolsJson, NeedleOptions? options = null)
    {
        this.model = model;
        this.options = options ?? new NeedleOptions();
        allTools = new ToolSet(toolsJson);
        engine = new Engine(model, this.options.Context, 64, this.options.ResolvedThreading, null) { Pooling = true };
        tokenizer = new Tokenizer(model, 1 << 16);
        trie = new TokenTrie(tokenizer, model.VocabSize);
        system = BuildSystem(this.options);
        mask = (uint*)Mem.AllocZeroed((nuint)trie.Words * 4);
        scratchLogits = (float*)Mem.Alloc((nuint)model.VocabSize * 4);
        pooled = (float*)Mem.AllocZeroed((nuint)(4 * model.Geometry.Dim * 4));
        reasoningMask = new uint[trie.Words];
        for (var id = 0; id < Math.Min(model.VocabSize, tokenizer.PieceCount); id++)
            if (tokenizer.IsConstrainable(id)) reasoningMask[id / 32] |= 1u << (id % 32);
        reasoningMask[ThinkEnd / 32] |= 1u << (ThinkEnd % 32);
        decisionMask = (uint[])reasoningMask.Clone();
        responseMask = (uint[])reasoningMask.Clone();
        foreach (var id in new[] { ToolCallStart, ImEnd, 1 }) decisionMask[id / 32] |= 1u << (id % 32);
        foreach (var id in new[] { ImEnd, 1 }) responseMask[id / 32] |= 1u << (id % 32);
        responseMask[ThinkEnd / 32] &= ~(1u << (ThinkEnd % 32));
        retrieval = allTools.Tools.Count > this.options.RetrievalTopK && model.Confidence != null;
        active = allTools;
        grammar = new Grammar(active.Tools);
        if (retrieval) IndexTools();
        else Prime(active);
    }

    public int ContextUsed => engine.Length;

    static string BuildSystem(NeedleOptions o)
    {
        var text = o.System ?? "";
        if (!o.AutoDate || DateKey.IsMatch(text) || IsoStamp.IsMatch(text)) return text;
        var fact = DateTime.Now.ToString("'date: 'yyyy-MM-dd ddd HH:mm", CultureInfo.InvariantCulture);
        if (string.IsNullOrWhiteSpace(text)) return fact;
        return text.TrimStart().StartsWith('{') ? text : fact + "; " + text;
    }

    int[] Tokens(string text, bool leadingSpace)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        var ids = new int[bytes.Length + 8];
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
        primedRender = tools.Render();
    }

    public void Reset()
    {
        engine.Rewind();
        turnOne = true;
    }

    public CallResult Complete(string query) => Turn(query, null);

    public CallResult CompleteWithToolResult(string resultJson) => Turn(null, resultJson);

    CallResult Turn(string? query, string? toolResult)
    {
        var started = Stopwatch.GetTimestamp();
        try
        {
            if (retrieval && query != null) SelectTools(query);
            var prompt = new List<int>();
            if (toolResult != null)
                prompt.AddRange(Tokens("<|im_start|>tool\n<tool_result>" + toolResult + "</tool_result><|im_end|>\n<|im_start|>assistant\n", false));
            else if (turnOne)
                prompt.AddRange(Tokens("\n" + query + "<|im_end|>\n<|im_start|>assistant\n", false));
            else
                prompt.AddRange(Tokens("<|im_start|>user\n" + query + "<|im_end|>\n<|im_start|>assistant\n", false));
            prompt.Add(ThinkStart);
            turnOne = false;
            engine.Feed(prompt.ToArray());
            var prefilled = Stopwatch.GetTimestamp();
            return Generate(started, prefilled, prompt.Count);
        }
        catch (Exception e) when (e is InvalidOperationException or FormatException or ArgumentException)
        {
            return new CallResult { Success = false, Error = e.Message };
        }
    }

    CallResult Generate(long started, long prefilled, int promptTokens)
    {
        var reasoning = new List<byte>();
        var piece = new byte[64];
        var first = true;
        var decoded = 0;
        var ttft = 0.0;
        for (var i = 0; i < options.MaxReasoningTokens; i++)
        {
            var next = ArgMaxMasked(reasoningMask, out _);
            if (i == 0) ttft = Elapsed(started, Stopwatch.GetTimestamp());
            if (next == ThinkEnd) break;
            reasoning.AddRange(piece.AsSpan(0, tokenizer.Append(next, piece, first)).ToArray());
            first = false;
            Feed(next);
            decoded++;
        }
        var forced = new List<int> { ThinkEnd };
        forced.AddRange(Tokens("\n", false));
        engine.Feed(forced.ToArray());
        var decision = ArgMaxMasked(decisionMask, out _);
        if (decision != ToolCallStart) return Respond(decision, reasoning, started, prefilled, promptTokens, decoded, ttft);
        Feed(ToolCallStart);

        var state = Grammar.Start(grammar.Program);
        var text = new List<byte>();
        var logProbability = 0.0;
        var steps = 0;
        var buffer = stackalloc byte[256];
        while (!Grammar.Complete(state))
        {
            if (steps >= options.MaxCallTokens) return new CallResult { Success = false, Error = "call exceeded the token budget", Reasoning = Encoding.UTF8.GetString(reasoning.ToArray()) };
            trie.Allow(grammar.Program, state, mask);
            var pick = ArgMaxMasked(mask, out var probability);
            if (steps == 0 && ttft == 0) ttft = Elapsed(started, Stopwatch.GetTimestamp());
            var length = tokenizer.ConstraintBytes(pick, buffer);
            var cursor = &state;
            for (var b = 0; b < length; b++)
                if (!Grammar.Advance(grammar.Program, cursor, buffer[b])) return new CallResult { Success = false, Error = "grammar rejected an allowed token" };
            for (var b = 0; b < length; b++) text.Add(buffer[b]);
            logProbability += Math.Log(Math.Max(probability, 1e-9));
            steps++;
            decoded++;
            Feed(pick, computeLogits: !Grammar.Complete(state));
        }
        var closing = new List<int> { ToolCallEnd, ImEnd };
        closing.AddRange(Tokens("\n", false));
        engine.Feed(closing.ToArray(), computeLogits: false);

        var head = 1.0 / (1.0 + Math.Exp(-engine.ConfidenceLogit(pooled)));
        var decode = Math.Exp(logProbability / Math.Max(1, steps));
        var confidence = (float)Math.Min(head, decode);
        var calls = ParseCalls(Encoding.UTF8.GetString(text.ToArray()));
        var done = Stopwatch.GetTimestamp();
        var refused = confidence < options.RefuseBelow;
        return new CallResult
        {
            FunctionCalls = refused ? Array.Empty<FunctionCall>() : calls,
            SuppressedCalls = refused ? calls : Array.Empty<FunctionCall>(),
            Reasoning = Encoding.UTF8.GetString(reasoning.ToArray()).Trim(),
            Confidence = confidence,
            HeadConfidence = (float)head,
            DecodeConfidence = (float)decode,
            PrefillTokens = promptTokens,
            DecodeTokens = decoded,
            TimeToFirstTokenMs = ttft,
            PrefillTps = promptTokens / Math.Max(1e-9, (prefilled - started) / (double)Stopwatch.Frequency),
            DecodeTps = decoded / Math.Max(1e-9, (done - prefilled) / (double)Stopwatch.Frequency),
            TotalMs = Elapsed(started, done),
        };
    }

    void Feed(int token, bool computeLogits = true)
    {
        var one = stackalloc int[1];
        one[0] = token;
        engine.Feed(new ReadOnlySpan<int>(one, 1), computeLogits);
    }

    CallResult Respond(int first, List<byte> reasoning, long started, long prefilled, int promptTokens, int decoded, double ttft)
    {
        var text = new List<byte>();
        var piece = new byte[64];
        var next = first;
        for (var i = 0; i < options.MaxCallTokens && next != ImEnd && next != 1; i++)
        {
            text.AddRange(piece.AsSpan(0, tokenizer.Append(next, piece, i == 0)).ToArray());
            Feed(next);
            decoded++;
            next = ArgMaxMasked(responseMask, out _);
        }
        var closing = new List<int> { ImEnd };
        closing.AddRange(Tokens("\n", false));
        engine.Feed(closing.ToArray(), computeLogits: false);
        var done = Stopwatch.GetTimestamp();
        return new CallResult
        {
            Reasoning = Encoding.UTF8.GetString(reasoning.ToArray()).Trim(),
            Response = Encoding.UTF8.GetString(text.ToArray()).Trim(),
            Confidence = (float)(1.0 / (1.0 + Math.Exp(-engine.ConfidenceLogit(pooled)))),
            PrefillTokens = promptTokens,
            DecodeTokens = decoded,
            TimeToFirstTokenMs = ttft,
            PrefillTps = promptTokens / Math.Max(1e-9, (prefilled - started) / (double)Stopwatch.Frequency),
            DecodeTps = decoded / Math.Max(1e-9, (done - prefilled) / (double)Stopwatch.Frequency),
            TotalMs = Elapsed(started, done),
        };
    }

    static double Elapsed(long from, long to) => (to - from) * 1000.0 / Stopwatch.Frequency;

    int ArgMaxMasked(ReadOnlySpan<uint> allowed, out float probability)
    {
        fixed (uint* m = allowed) return ArgMaxMasked(m, out probability);
    }

    int ArgMaxMasked(uint* allowed, out float probability)
    {
        var logits = engine.Logits;
        var vocab = model.VocabSize;
        var best = -1;
        var peak = float.NegativeInfinity;
        for (var id = 0; id < vocab; id++)
        {
            var open = (allowed[id / 32] >> (id % 32) & 1) != 0;
            scratchLogits[id] = open ? logits[id] : -1e30f;
            if (open && logits[id] > peak) { peak = logits[id]; best = id; }
        }
        if (best < 0) throw new InvalidOperationException("the grammar left no allowed token");
        Ops.Softmax(scratchLogits, vocab);
        probability = scratchLogits[best];
        return best;
    }

    IReadOnlyList<FunctionCall> ParseCalls(string json)
    {
        var parsed = JsonValue.Parse(json);
        var calls = new List<FunctionCall>();
        foreach (var call in parsed.Items)
        {
            var alias = call.GetString("name")!;
            var arguments = call.Get("arguments") ?? JsonValue.Object(Array.Empty<KeyValuePair<string, JsonValue>>());
            calls.Add(new FunctionCall(active.OriginalName(alias), Restore(arguments, alias).ToString()));
        }
        return calls;
    }

    JsonValue Restore(JsonValue value, string tool) => value.Kind switch
    {
        JsonKind.Object => JsonValue.Object(value.Members.ConvertAll(m => new KeyValuePair<string, JsonValue>(active.OriginalProperty(tool, m.Key), Restore(m.Value, tool)))),
        JsonKind.Array => JsonValue.Array(value.Items.ConvertAll(i => Restore(i, tool))),
        _ => value,
    };

    public float[] Embed(string text)
    {
        embedder ??= new Engine(model, 512, 64, options.ResolvedThreading, engine.SharedPool) { Pooling = true };
        var ids = new List<int> { tokenizer.BeginId };
        ids.AddRange(Tokens(text, true));
        embedder.Reset();
        embedder.Feed(ids.ToArray(), computeLogits: false);
        var width = model.Confidence->Queries * model.Geometry.Dim;
        var vector = (float*)Mem.AllocZeroed((nuint)width * 4);
        embedder.ConfidenceLogit(vector);
        var norm = MathF.Sqrt(Ops.SumOfSquares(vector, width)) + 1e-12f;
        var result = new float[width];
        for (var i = 0; i < width; i++) result[i] = vector[i] / norm;
        Mem.Free(vector);
        return result;
    }

    void IndexTools()
    {
        foreach (var tool in allTools.Tools) toolEmbeddings.Add(Embed(tool.ToString()));
    }

    void SelectTools(string query)
    {
        var q = Embed(query);
        var scored = new List<(float Score, int Index)>();
        for (var i = 0; i < toolEmbeddings.Count; i++)
        {
            var dot = 0f;
            for (var j = 0; j < q.Length; j++) dot += q[j] * toolEmbeddings[i][j];
            scored.Add((dot, i));
        }
        scored.Sort((a, b) => b.Score.CompareTo(a.Score));
        var chosen = scored.GetRange(0, options.RetrievalTopK).ConvertAll(s => s.Index);
        chosen.Sort();
        var subset = allTools.Subset(chosen);
        if (subset.Render() == primedRender) return;
        active = subset;
        grammar.Dispose();
        grammar = new Grammar(active.Tools);
        Prime(active);
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
        Mem.Free(scratchLogits);
        Mem.Free(pooled);
    }
}
