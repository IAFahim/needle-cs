using System;
using System.Collections.Generic;
using System.Text;

namespace Needle.Tests;

internal sealed unsafe class LegacySession : IDisposable
{
    const int ImEnd = 5, ThinkStart = 6, ThinkEnd = 7, ToolCallStart = 10, ToolCallEnd = 11;

    readonly Model model;
    readonly NeedleOptions options;
    readonly ToolSet tools;
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
    readonly Grammar grammar;
    bool turnOne = true;

    public LegacySession(Model model, string toolsJson, NeedleOptions options)
    {
        this.model = model;
        this.options = options;
        tools = new ToolSet(toolsJson);
        engine = new Engine(model, options.Context, 64, options.Threads) { Pooling = true };
        tokenizer = new Tokenizer(model, 1 << 16);
        trie = new TokenTrie(tokenizer, model.VocabSize);
        system = options.System ?? "";
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
        grammar = new Grammar(tools.Tools);
        var head = (system.Length > 0 ? "<|im_start|>system\n" + system + "<|im_end|>\n" : "") + "<|im_start|>user\n<tools>" + tools.Render() + "</tools>";
        var ids = new List<int> { tokenizer.BeginId };
        ids.AddRange(Tokens(head, true));
        engine.Reset();
        engine.Feed(ids.ToArray(), computeLogits: false);
        engine.Mark();
    }

    public void Reset()
    {
        engine.Rewind();
        turnOne = true;
    }

    public CallResult Complete(string query) => Turn(query, null);

    public CallResult CompleteWithToolResult(string resultJson) => Turn(null, resultJson);

    int[] Tokens(string text, bool leadingSpace)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        var ids = new int[bytes.Length + 8];
        var count = tokenizer.Encode(bytes, ids, leadingSpace);
        return ids[..count];
    }

    CallResult Turn(string? query, string? toolResult)
    {
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
        return Generate();
    }

    CallResult Generate()
    {
        var reasoning = new List<byte>();
        var piece = new byte[64];
        var first = true;
        for (var i = 0; i < options.MaxReasoningTokens; i++)
        {
            var next = ArgMaxMasked(reasoningMask, out _);
            if (next == ThinkEnd) break;
            reasoning.AddRange(piece.AsSpan(0, tokenizer.Append(next, piece, first)).ToArray());
            first = false;
            Feed(next);
        }
        var forced = new List<int> { ThinkEnd };
        forced.AddRange(Tokens("\n", false));
        engine.Feed(forced.ToArray());
        var decision = ArgMaxMasked(decisionMask, out _);
        if (decision != ToolCallStart) return Respond(decision, reasoning);
        Feed(ToolCallStart);

        var state = Grammar.Start(grammar.Program);
        var text = new List<byte>();
        var logProbability = 0.0;
        var steps = 0;
        var buffer = stackalloc byte[256];
        while (!Grammar.Complete(state))
        {
            if (steps >= options.MaxCallTokens) return new CallResult { Success = false, Error = "call exceeded the token budget" };
            trie.Allow(grammar.Program, state, mask);
            var pick = ArgMaxMasked(mask, out var probability);
            var length = tokenizer.ConstraintBytes(pick, buffer);
            var cursor = &state;
            for (var b = 0; b < length; b++)
                if (!Grammar.Advance(grammar.Program, cursor, buffer[b])) return new CallResult { Success = false, Error = "grammar rejected an allowed token" };
            for (var b = 0; b < length; b++) text.Add(buffer[b]);
            logProbability += Math.Log(Math.Max(probability, 1e-9));
            steps++;
            Feed(pick, computeLogits: !Grammar.Complete(state));
        }
        var closing = new List<int> { ToolCallEnd, ImEnd };
        closing.AddRange(Tokens("\n", false));
        engine.Feed(closing.ToArray(), computeLogits: false);

        var head = 1.0 / (1.0 + Math.Exp(-engine.ConfidenceLogit(pooled)));
        var decode = Math.Exp(logProbability / Math.Max(1, steps));
        var confidence = (float)Math.Min(head, decode);
        var calls = ParseCalls(Encoding.UTF8.GetString(text.ToArray()));
        var refused = confidence < options.RefuseBelow;
        return new CallResult
        {
            FunctionCalls = refused ? Array.Empty<FunctionCall>() : calls,
            SuppressedCalls = refused ? calls : Array.Empty<FunctionCall>(),
            Reasoning = Encoding.UTF8.GetString(reasoning.ToArray()).Trim(),
            Confidence = confidence,
            HeadConfidence = (float)head,
            DecodeConfidence = (float)decode,
        };
    }

    void Feed(int token, bool computeLogits = true)
    {
        var one = stackalloc int[1];
        one[0] = token;
        engine.Feed(new ReadOnlySpan<int>(one, 1), computeLogits);
    }

    CallResult Respond(int first, List<byte> reasoning)
    {
        var text = new List<byte>();
        var piece = new byte[64];
        var next = first;
        for (var i = 0; i < options.MaxCallTokens && next != ImEnd && next != 1; i++)
        {
            text.AddRange(piece.AsSpan(0, tokenizer.Append(next, piece, i == 0)).ToArray());
            Feed(next);
            next = ArgMaxMasked(responseMask, out _);
        }
        var closing = new List<int> { ImEnd };
        closing.AddRange(Tokens("\n", false));
        engine.Feed(closing.ToArray(), computeLogits: false);
        return new CallResult
        {
            Reasoning = Encoding.UTF8.GetString(reasoning.ToArray()).Trim(),
            Response = Encoding.UTF8.GetString(text.ToArray()).Trim(),
            Confidence = (float)(1.0 / (1.0 + Math.Exp(-engine.ConfidenceLogit(pooled)))),
        };
    }

    int ArgMaxMasked(uint[] allowed, out float probability)
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
            calls.Add(new FunctionCall(tools.OriginalName(alias), Restore(arguments, alias).ToString()));
        }
        return calls;
    }

    JsonValue Restore(JsonValue value, string tool) => value.Kind switch
    {
        JsonKind.Object => JsonValue.Object(value.Members.ConvertAll(m => new KeyValuePair<string, JsonValue>(tools.OriginalProperty(tool, m.Key), Restore(m.Value, tool)))),
        JsonKind.Array => JsonValue.Array(value.Items.ConvertAll(i => Restore(i, tool))),
        _ => value,
    };

    public void Dispose()
    {
        engine.Dispose();
        tokenizer.Dispose();
        trie.Dispose();
        grammar.Dispose();
        Mem.Free(mask);
        Mem.Free(scratchLogits);
        Mem.Free(pooled);
    }
}
