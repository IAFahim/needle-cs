using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using Needle.Cli;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

public sealed unsafe class MaskedDecodingTests : IDisposable
{
    readonly Model model;
    readonly ITestOutputHelper output;

    public MaskedDecodingTests(ITestOutputHelper output)
    {
        Assets.Require();
        model = Model.Open(Assets.Model);
        this.output = output;
    }

    public void Dispose() => model.Dispose();

    [Fact]
    public void ArgMaxMaskedMatchesScalarReference()
    {
        var random = new Random(1234);
        foreach (var length in new[] { 33, 1000, 8192 })
        {
            var logits = new float[length];
            for (var i = 0; i < length; i++) logits[i] = (float)(random.NextDouble() * 20.0 - 10.0);
            logits[3 % length] = 50f;
            logits[7 % length] = 50f;
            var words = (length + 31) / 32;
            var mask = new uint[words];
            var singles = new[] { 0, 1, 31, 32, length - 1 };
            for (var trial = 0; trial < 60; trial++)
            {
                Array.Clear(mask, 0, words);
                switch (trial % 6)
                {
                    case 0: break;
                    case 1:
                        for (var w = 0; w < words; w++) mask[w] = 0xFFFFFFFFu;
                        break;
                    case 2:
                        mask[singles[trial / 6 % singles.Length] / 32] |= 1u << (singles[trial / 6 % singles.Length] % 32);
                        break;
                    case 3:
                        for (var k = 0; k < 5; k++)
                        {
                            var id = random.Next(length);
                            mask[id / 32] |= 1u << (id % 32);
                        }
                        break;
                    case 4:
                    {
                        var start = random.Next(length);
                        var end = Math.Min(length, start + random.Next(2000));
                        for (var id = start; id < end; id++) mask[id / 32] |= 1u << (id % 32);
                        break;
                    }
                    default:
                        for (var w = 0; w < words; w += 2) mask[w] = 0xAAAAAAAAu;
                        break;
                }
                var expected = -1;
                var peak = float.NegativeInfinity;
                for (var id = 0; id < length; id++)
                    if ((mask[id / 32] >> (id % 32) & 1) != 0 && logits[id] > peak) { peak = logits[id]; expected = id; }
                int actual;
                fixed (float* x = logits)
                fixed (uint* m = mask)
                    actual = Ops.ArgMaxMasked(x, m, words, length);
                Assert.Equal(expected, actual);
            }
        }
    }

    [Fact]
    public void BestProbabilityMatchesSoftmaxAndDoubleReference()
    {
        var random = new Random(77);
        foreach (var count in new[] { 1, 2, 7, 8, 9, 15, 16, 17, 100, 1000, 8192 })
        {
            for (var spread = 0; spread < 3; spread++)
            {
                var values = new float[count];
                var scale = spread == 0 ? 8.0 : spread == 1 ? 1.0 : 30.0;
                var best = float.NegativeInfinity;
                for (var i = 0; i < count; i++)
                {
                    values[i] = (float)(random.NextDouble() * 2 * scale - scale);
                    if (values[i] > best) best = values[i];
                }
                var total = 0.0;
                for (var i = 0; i < count; i++) total += Math.Exp(values[i] - best);
                var expected = 1.0 / total;
                float actual;
                fixed (float* v = values)
                    actual = Ops.BestProbability(v, count, best);
                Assert.InRange(Math.Abs(actual - expected), 0.0, 1e-6);
                // The legacy path normalizes the whole allowed set; the best
                // slot must agree with the log-sum-exp shortcut.
                var legacy = new float[count];
                Array.Copy(values, legacy, count);
                var bestSlot = 0;
                for (var i = 1; i < count; i++) if (values[i] > values[bestSlot]) bestSlot = i;
                float legacyProb;
                fixed (float* l = legacy)
                {
                    Ops.Softmax(l, count);
                    legacyProb = l[bestSlot];
                }
                Assert.InRange(Math.Abs(legacyProb - actual), 0f, 1e-6f);
            }
        }
    }

    struct EventCounter : Tools.IGrammarSink
    {
        public int Tool, Property, StringBytes, NumberBytes, Enum, Bool, Null, Completed, Calls;
        public void ToolSelected(int tool) => Tool++;
        public void PropertySelected(int property) => Property++;
        public void StringByte(byte value) => StringBytes++;
        public void NumberByte(byte value) => NumberBytes++;
        public void EnumSelected(int literal) => Enum++;
        public void BoolSelected(bool value) => Bool++;
        public void NullSelected() => Null++;
        public void ValueCompleted() => Completed++;
        public void CallCompleted() => Calls++;
    }

    static void AdvancePrefix(GrammarProgram* program, GrammarState* state, string prefix)
    {
        foreach (var b in Encoding.UTF8.GetBytes(prefix))
            Assert.True(Grammar.Advance(program, state, b), $"prefix {prefix} rejected at byte {b}");
    }

    static int AdmissibleCount(GrammarProgram* program, in GrammarState state)
    {
        var count = 0;
        var probe = new Tools.NoSink();
        for (var b = 0; b < 256; b++)
        {
            var copy = state;
            if (Grammar.Advance(program, &copy, (byte)b, ref probe)) count++;
        }
        return count;
    }

    [Fact]
    public void ForcedBytesSpellsFixedLiteralsThenStopsAtChoices()
    {
        var set = new ToolSet(BenchPrompts.GameToolsJson);
        using var grammar = new Grammar(set.Tools);
        var program = grammar.Program;
        var destination = new byte[256];
        // From a fresh state only '[' opens a call list.
        var state = Grammar.Start(program);
        int count;
        fixed (byte* d = destination)
            count = Grammar.ForcedBytes(program, &state, new Span<byte>(d, destination.Length));
        Assert.Equal(new byte[] { (byte)'[' }, destination.AsSpan(0, count).ToArray());
        // After '[' the grammar branches (']' or '{'), so nothing is forced
        // and the state must be untouched.
        state = Grammar.Start(program);
        AdvancePrefix(program, &state, "[");
        var depth = state.Depth;
        fixed (byte* d = destination)
            count = Grammar.ForcedBytes(program, &state, new Span<byte>(d, destination.Length));
        Assert.Equal(0, count);
        Assert.Equal(depth, state.Depth);
        // After "[{" the call head `"name":"` is fixed; the tool name that
        // follows is a choice, so the run stops there.
        state = Grammar.Start(program);
        AdvancePrefix(program, &state, "[{");
        fixed (byte* d = destination)
            count = Grammar.ForcedBytes(program, &state, new Span<byte>(d, destination.Length));
        Assert.Equal(Encoding.UTF8.GetBytes("\"name\":\""), destination.AsSpan(0, count).ToArray());
        Assert.NotEqual(1, AdmissibleCount(program, state));
        // The sink overload reports the same bytes and leaves an equivalent
        // state: same admissibility profile plus identical sink events to
        // feeding the bytes one by one.
        var plain = Grammar.Start(program);
        AdvancePrefix(program, &plain, "[{");
        var sink = new EventCounter();
        int sunk;
        fixed (byte* d = destination)
            sunk = Grammar.ForcedBytes(program, &plain, new Span<byte>(d, destination.Length), ref sink);
        Assert.Equal(count, sunk);
        Assert.Equal(destination.AsSpan(0, count).ToArray(), destination.AsSpan(0, sunk).ToArray());
        var manual = Grammar.Start(program);
        AdvancePrefix(program, &manual, "[{");
        var expected = new EventCounter();
        for (var i = 0; i < sunk; i++)
            Assert.True(Grammar.Advance(program, &manual, destination[i], ref expected));
        Assert.Equal(expected, sink);
        Assert.Equal(AdmissibleCount(program, manual), AdmissibleCount(program, plain));
    }

    [Fact]
    public void ForcedBytesHandlesEmptySpanAndCompletion()
    {
        var set = new ToolSet(BenchPrompts.GameToolsJson);
        using var grammar = new Grammar(set.Tools);
        var program = grammar.Program;
        var state = Grammar.Start(program);
        Assert.Equal(0, Grammar.ForcedBytes(program, &state, Span<byte>.Empty));
        var done = new GrammarState();
        Assert.Equal(0, Grammar.ForcedBytes(program, &done, new Span<byte>(new byte[8])));
    }

    static List<GrammarState> SampleStates(GrammarProgram* program, TokenTrie trie, Tokenizer tokenizer, int vocab)
    {
        var states = new List<GrammarState> { Grammar.Start(program) };
        var mask = new uint[trie.Words];
        var scratch = new byte[256];
        var random = new Random(5);
        // A mid-string state exercises the string fast path (unbounded text).
        var text = Grammar.Start(program);
        AdvancePrefix(program, &text, "[{\"name\":\"say\",\"arguments\":{\"text\":\"hello");
        states.Add(text);
        var walk = Grammar.Start(program);
        for (var step = 0; step < 400 && !Grammar.Complete(walk); step++)
        {
            fixed (uint* m = mask) trie.Allow(program, walk, m);
            var allowed = new List<int>();
            for (var id = 0; id < vocab; id++) if ((mask[id / 32] >> (id % 32) & 1) != 0) allowed.Add(id);
            if (allowed.Count == 0) break;
            var pick = allowed[random.Next(allowed.Count)];
            int length;
            fixed (byte* p = scratch) length = tokenizer.ConstraintBytes(pick, p);
            for (var i = 0; i < length; i++)
                if (!Grammar.Advance(program, &walk, scratch[i])) break;
            if (step % 37 == 0) states.Add(walk);
        }
        return states;
    }

    // Mirrors TokenTrie.Allow's string fast-path gate: an unbounded string
    // value whose top frame is mid-literal. Passed by value; the struct copy
    // keeps the address-of below legal for an iteration variable.
    static bool IsStringFastPath(GrammarProgram* program, GrammarState state)
    {
        if (state.Depth == 0) return false;
        var top = state.Frames + state.Depth - 1;
        var node = program->Nodes + top->Node * Kind.Fields;
        return node[0] == Kind.Str && top->Phase == 1 && node[1] < 0;
    }

    [Fact]
    public void MemoizedMasksMatchBruteForceAdvance()
    {
        var set = new ToolSet(BenchPrompts.GameToolsJson);
        using var grammar = new Grammar(set.Tools);
        using var tokenizer = new Tokenizer(model);
        using var trie = new TokenTrie(tokenizer, model.VocabSize);
        var words = trie.Words;
        var first = new uint[words];
        var second = new uint[words];
        var scratch = new byte[256];
        var compared = 0;
        foreach (var state in SampleStates(grammar.Program, trie, tokenizer, model.VocabSize))
        {
            fixed (uint* f = first) trie.Allow(grammar.Program, state, f);
            fixed (uint* s = second) trie.Allow(grammar.Program, state, s);
            Assert.Equal(first, second);
            // The unbounded-string fast path is a conservative approximation:
            // it admits plain and quoted pieces only, while the grammar also
            // accepts bytes like DEL inside strings. There the memoized mask
            // must be a subset of the brute-force set (every bit is genuine);
            // everywhere else the trie walk is exact.
            var exact = !IsStringFastPath(grammar.Program, state);
            for (var id = 0; id < model.VocabSize; id++)
            {
                int length;
                fixed (byte* p = scratch) length = tokenizer.ConstraintBytes(id, p);
                var expected = false;
                if (length > 0)
                {
                    var copy = state;
                    expected = true;
                    for (var i = 0; i < length && expected; i++)
                        expected = Grammar.Advance(grammar.Program, &copy, scratch[i]);
                }
                var actual = (first[id / 32] >> (id % 32) & 1) != 0;
                if (exact) Assert.Equal(expected, actual);
                else if (actual) Assert.True(expected);
                compared++;
            }
        }
        Assert.True(compared > 10000, $"only {compared} token checks ran");
    }

    [Fact]
    public void SparseHeadMatchesFullHeadBitForBit()
    {
        using var engine = new Engine(model, 512, 64, 1);
        using var tokenizer = new Tokenizer(model);
        var prompt = "move to the old bridge and say hello"u8.ToArray();
        var ids = new int[4096];
        ids[0] = tokenizer.BeginId;
        var count = 1 + tokenizer.Encode(prompt, ids.AsSpan(1), true);
        engine.Feed(new ReadOnlySpan<int>(ids, 0, count));
        var full = engine.Logits.ToArray();
        var sets = new List<int[]>
        {
            new List<int> { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }.ToArray(),
            EveryNth(model.VocabSize, 997, 64),
            EveryNth(model.VocabSize, 27, 303),
            EveryNth(model.VocabSize, 1, 3000),
        };
        var sparse = new float[model.VocabSize];
        foreach (var rows in sets)
        {
            Array.Clear(sparse, 0, rows.Length);
            fixed (int* r = rows)
            fixed (float* d = sparse)
                engine.LogitsForRows(r, rows.Length, d);
            for (var i = 0; i < rows.Length; i++) Assert.Equal(full[rows[i]], sparse[i]);
            // The sparse argmax path must pick the same token with a
            // log-sum-exp probability matching the dense masked path.
            var mask = new uint[(model.VocabSize + 31) / 32];
            foreach (var id in rows) mask[id / 32] |= 1u << (id % 32);
            int denseBest, sparseBest;
            float denseProb, sparseProb;
            var gathered = new float[rows.Length];
            fixed (float* logits = full)
            fixed (float* g = gathered)
            fixed (uint* m = mask)
            {
                denseBest = ArgMaxWithProbabilityReference(logits, m, (model.VocabSize + 31) / 32, g, out denseProb);
                fixed (float* d = sparse)
                fixed (int* r = rows)
                    sparseBest = ArgMaxSparseReference(d, r, rows.Length, out sparseProb);
            }
            Assert.Equal(denseBest, sparseBest);
            Assert.InRange(Math.Abs(denseProb - sparseProb), 0f, 1e-6f);
        }
    }

    static int[] EveryNth(int vocab, int step, int take)
    {
        var rows = new List<int>();
        for (var id = 0; id < vocab && rows.Count < take; id += step) rows.Add(id);
        return rows.ToArray();
    }

    static int ArgMaxWithProbabilityReference(float* logits, uint* allowed, int words, float* gathered, out float probability)
    {
        var best = -1;
        var count = 0;
        var peak = float.NegativeInfinity;
        for (var word = 0; word < words; word++)
        {
            var bits = allowed[word];
            while (bits != 0)
            {
                var id = word * 32 + Needle.Bits.TrailingZeros(bits);
                bits &= bits - 1;
                gathered[count++] = logits[id];
                if (logits[id] > peak) { peak = logits[id]; best = id; }
            }
        }
        probability = Ops.BestProbability(gathered, count, peak);
        return best;
    }

    static int ArgMaxSparseReference(float* values, int* rows, int count, out float probability)
    {
        var best = -1;
        var peak = float.NegativeInfinity;
        for (var i = 0; i < count; i++)
        {
            if (values[i] > peak) { peak = values[i]; best = rows[i]; }
        }
        probability = Ops.BestProbability(values, count, peak);
        return best;
    }

    [Fact]
    public void SparseWalkMatchesFullWalkAndReportsOverhead()
    {
        var tools = new ToolSet(BenchPrompts.GameToolsJson);
        using var grammar = new Grammar(tools.Tools);
        using var tokenizer = new Tokenizer(model);
        using var trie = new TokenTrie(tokenizer, model.VocabSize);
        var prompt = Encoding.UTF8.GetBytes(BenchPrompts.Game[0]);
        var ids = new int[8192];
        ids[0] = tokenizer.BeginId;
        var promptCount = 1 + tokenizer.Encode(prompt, ids.AsSpan(1), true);
        using var full = new Engine(model, 1024, 64, 1);
        using var sparse = new Engine(model, 1024, 64, 1);
        full.Feed(new ReadOnlySpan<int>(ids, 0, promptCount));
        sparse.Feed(new ReadOnlySpan<int>(ids, 0, promptCount));
        var words = trie.Words;
        var mask = new uint[words];
        var single = new uint[words];
        var allowedIds = new int[256];
        var scratch = new float[model.VocabSize];
        var bytes = new byte[64];
        var picks = new List<int>();
        var probs = new List<float>();
        var allowTicks = 0L;
        var headTicks = 0L;
        var argmaxTicks = 0L;
        var sparseHeadTicks = 0L;
        var sparseArgmaxTicks = 0L;
        var sparseSteps = 0;
        var steps = 0;
        var state = Grammar.Start(grammar.Program);
        while (!Grammar.Complete(state) && steps < 384)
        {
            steps++;
            var tick = Stopwatch.GetTimestamp();
            fixed (uint* m = mask) trie.Allow(grammar.Program, state, m);
            allowTicks += Stopwatch.GetTimestamp() - tick;
            var allowed = 0;
            for (var w = 0; w < words; w++) allowed += System.Numerics.BitOperations.PopCount(mask[w]);
            Assert.True(allowed > 0, "grammar dead end on the full walk");
            // The raw greedy walk would end immediately with an empty call
            // list (']'); force the call branch once so the walk exercises a
            // realistic multi-step call. The singleton still flows through
            // both argmax paths, so equality still verifies them.
            uint[] effective = mask;
            if (steps == 2)
            {
                // The trie admits every tokenization of a forced span, so no
                // single token spells '{'; take the shortest '{'-leading one.
                Array.Clear(single, 0, words);
                var forced = -1;
                var forcedLength = int.MaxValue;
                for (var id = 0; id < model.VocabSize; id++)
                {
                    if ((mask[id / 32] >> (id % 32) & 1) == 0) continue;
                    int len;
                    fixed (byte* p = bytes) len = tokenizer.ConstraintBytes(id, p);
                    if (len > 0 && bytes[0] == (byte)'{' && len < forcedLength) { forced = id; forcedLength = len; }
                }
                Assert.True(forced >= 0, "no '{'-leading token in the mask");
                single[forced / 32] |= 1u << (forced % 32);
                effective = single;
                allowed = 1;
            }
            // Reference: masked gather over the full head plus a log-sum-exp
            // probability over the allowed set only. full.Logits is valid for
            // the current state: the previous feed computed it.
            int pick;
            float probability;
            tick = Stopwatch.GetTimestamp();
            var logits = full.Logits;
            fixed (float* c = logits)
            fixed (float* g = scratch)
            fixed (uint* m = effective)
                pick = ArgMaxWithProbabilityReference(c, m, words, g, out probability);
            argmaxTicks += Stopwatch.GetTimestamp() - tick;
            // Sparse twin: the sparse engine's last feed is the same token
            // the full engine fed last, so its hidden state matches and
            // LogitsForRows must reproduce the current full logits. It runs
            // before either engine feeds the new pick.
            var extracted = 0;
            var useSparse = allowed <= 256;
            if (useSparse)
            {
                for (var w = 0; w < words; w++)
                {
                    var bits = effective[w];
                    while (bits != 0)
                    {
                        allowedIds[extracted++] = w * 32 + Needle.Bits.TrailingZeros(bits);
                        bits &= bits - 1;
                    }
                }
            }
            // Timed head comparison: full Feed(true) covers forward+head while
            // the sparse side covers forward-without-logits plus LogitsForRows.
            int spick;
            float sprob;
            if (useSparse)
            {
                tick = Stopwatch.GetTimestamp();
                fixed (int* r = allowedIds)
                fixed (float* d = scratch)
                    sparse.LogitsForRows(r, extracted, d);
                sparse.Feed(new ReadOnlySpan<int>(&pick, 1), false);
                sparseHeadTicks += Stopwatch.GetTimestamp() - tick;
                tick = Stopwatch.GetTimestamp();
                fixed (int* r = allowedIds)
                fixed (float* d = scratch)
                    spick = ArgMaxSparseReference(d, r, extracted, out sprob);
                sparseArgmaxTicks += Stopwatch.GetTimestamp() - tick;
                sparseSteps++;
            }
            else
            {
                sparse.Feed(new ReadOnlySpan<int>(&pick, 1), true);
                spick = pick;
                sprob = probability;
            }
            Assert.Equal(pick, spick);
            Assert.InRange(Math.Abs(probability - sprob), 0f, 1e-6f);
            picks.Add(pick);
            probs.Add(probability);
            // Advance first so the feed flag matches the session: no logits
            // are needed once the new state completes.
            var cursor = state;
            AdvanceToken(grammar.Program, tokenizer, &cursor, pick, bytes);
            var done = Grammar.Complete(cursor);
            tick = Stopwatch.GetTimestamp();
            full.Feed(new ReadOnlySpan<int>(&pick, 1), !done);
            headTicks += Stopwatch.GetTimestamp() - tick;
            state = cursor;
        }
        Assert.True(picks.Count > 0, "walk produced no steps");
        var frequency = (double)Stopwatch.Frequency;
        output.WriteLine($"steps={steps} sparse={sparseSteps}/{steps} " +
            $"allow_us={allowTicks / frequency * 1e6 / steps:F2} " +
            $"full_head_us={headTicks / frequency * 1e6 / steps:F2} " +
            $"masked_argmax_us={argmaxTicks / frequency * 1e6 / steps:F2} " +
            $"sparse_head_us={(sparseSteps == 0 ? 0 : sparseHeadTicks / frequency * 1e6 / sparseSteps):F2} " +
            $"sparse_argmax_us={(sparseSteps == 0 ? 0 : sparseArgmaxTicks / frequency * 1e6 / sparseSteps):F2}");
    }

    static void AdvanceToken(GrammarProgram* program, Tokenizer tokenizer, GrammarState* state, int pick, byte[] bytes)
    {
        int length;
        fixed (byte* p = bytes) length = tokenizer.ConstraintBytes(pick, p);
        for (var i = 0; i < length; i++)
            Assert.True(Grammar.Advance(program, state, bytes[i]), $"grammar rejected token {pick}");
    }

}
