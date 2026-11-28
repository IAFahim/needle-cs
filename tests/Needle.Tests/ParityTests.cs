using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Xunit;

namespace Needle.Tests;

public sealed class ParityTests : IDisposable
{
    readonly Model model;

    public ParityTests()
    {
        Assets.Require();
        model = Model.Open(Assets.Model);
    }

    public void Dispose() => model.Dispose();

    static int[] Ids(System.Text.Json.JsonElement element) => element.EnumerateArray().Select(e => e.GetInt32()).ToArray();

    [Fact]
    public void TokenizerMatchesSentencePieceOnPromptsAndProbes()
    {
        using var tokenizer = new Tokenizer(model);
        var golden = Assets.GoldenJson();
        var buffer = new int[4096];
        foreach (var item in golden.GetProperty("cases").EnumerateArray())
        {
            var text = Encoding.UTF8.GetBytes(item.GetProperty("prompt").GetString()!);
            var expected = Ids(item.GetProperty("sp_ids"));
            var count = tokenizer.Encode(text, buffer.AsSpan(1));
            buffer[0] = tokenizer.BeginId;
            Assert.Equal(expected, buffer.AsSpan(0, count + 1).ToArray());
        }
        foreach (var item in golden.GetProperty("probes").EnumerateArray())
        {
            var text = Encoding.UTF8.GetBytes(item.GetProperty("text").GetString()!);
            var count = tokenizer.Encode(text, buffer);
            Assert.Equal(Ids(item.GetProperty("sp_ids")), buffer.AsSpan(0, count).ToArray());
        }
    }

    [Fact]
    public void TokenByTokenDecodeIsBitIdenticalAcrossThreadCounts()
    {
        var ids = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/ids.bin"))).ToArray();
        // Past 64 positions decode attention splits positions into partial softmaxes; their count must not
        // follow the thread count (it did on x64: 3+ threads merged two partials where 1-2 threads merged none).
        var serial = DecodeLogits(ids, 100, 1, false);
        foreach (var threads in new[] { 2, 3, 4, 6, 8 })
            AssertSameBits(serial, DecodeLogits(ids, 100, threads, false), $"threads {threads}");
        // A 4-thread pool whose adaptive team size changes token by token.
        AssertSameBits(serial, DecodeLogits(ids, 100, 4, true), "adaptive team");
    }

    [Fact]
    public void PrefillAndDecodeAreBitIdenticalAtEveryThreadAndTeamSize()
    {
        var ids = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/ids.bin"))).ToArray();
        ids = ids.AsSpan(0, Math.Min(ids.Length, 100)).ToArray();
        // Pooling (as NeedleSession runs it) prefills layer by layer instead of the fused team.
        foreach (var pooling in new[] { false, true })
        {
            float[][]? full = null;
            foreach (var (threads, team) in new[] { (4, 0), (4, 1), (4, 2), (4, 3), (1, 0), (2, 0), (3, 0), (6, 0), (8, 0) })
            {
                using var engine = new Engine(model, 512, 64, threads) { Pooling = pooling };
                engine.SharedPool.PinTeamMembers(team);
                var steps = new System.Collections.Generic.List<float[]>();
                engine.Feed(ids);
                steps.Add(engine.Logits.ToArray());
                var token = new int[1];
                for (var step = 0; step < 8; step++)
                {
                    token[0] = engine.Greedy();
                    engine.Feed(token);
                    steps.Add(engine.Logits.ToArray());
                }
                if (full == null) { full = steps.ToArray(); continue; }
                AssertSameBits(full, steps.ToArray(), $"pooling {pooling} threads {threads} team {team}");
            }
        }
    }

    float[][] DecodeLogits(int[] ids, int prompt, int threads, bool varyTeam)
    {
        using var engine = new Engine(model, 512, 64, threads);
        var steps = new System.Collections.Generic.List<float[]>();
        var token = new int[1];
        var fed = 0;
        void Step(int id)
        {
            if (varyTeam) engine.SharedPool.PinTeamMembers(1 + fed++ % threads);
            token[0] = id;
            engine.Feed(token);
            steps.Add(engine.Logits.ToArray());
        }
        foreach (var id in ids.AsSpan(0, Math.Min(ids.Length, prompt))) Step(id);
        for (var step = 0; step < 8; step++) Step(engine.Greedy());
        return steps.ToArray();
    }

    static void AssertSameBits(float[][] expected, float[][] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var step = 0; step < expected.Length; step++)
            for (var i = 0; i < expected[step].Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(expected[step][i]) == BitConverter.SingleToInt32Bits(actual[step][i]),
                    $"{label} step {step} logit {i}: {expected[step][i]} vs {actual[step][i]}");
    }

    [Fact]
    public void ForwardMatchesReferenceLogits()
    {
        var ids = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/ids.bin"))).ToArray();
        var expected = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/logits.bin"))).ToArray();
        foreach (var (chunk, threads) in new[] { (256, 1), (64, 1), (7, 1), (64, 8), (5, 6) })
        {
            using var engine = new Engine(model, 512, chunk, Assets.ThreadsFor(threads));
            engine.Feed(ids);
            var actual = engine.Logits.ToArray();
            Assert.Equal(Array.IndexOf(expected, expected.Max()), Array.IndexOf(actual, actual.Max()));
            Assert.True(Cosine(actual, expected) > 0.99995, $"chunk {chunk} threads {threads}: cosine {Cosine(actual, expected)}");
        }
    }

    [Fact]
    public void RewindRestoresExactState()
    {
        var ids = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/ids.bin"))).ToArray();
        using var engine = new Engine(model, 512, 64);
        engine.Feed(ids.AsSpan(0, 100), computeLogits: false);
        engine.Mark();
        engine.Feed(ids.AsSpan(100));
        var first = engine.Logits.ToArray();
        engine.Rewind();
        Assert.Equal(100, engine.Length);
        engine.Feed(ids.AsSpan(100));
        Assert.Equal(first, engine.Logits.ToArray());
    }

    const string LagosQuery = "what's it like in Lagos right now?";
    const string LagosTieBranch = "<think>\n'what's it like in Lagos right now' -> get_weather with city 'Lagos'.\n</think>\n<tool_call>[{\"name\":\"get_weather\",\"arguments\":{\"c";

    [Fact]
    public void GenerationMatchesReferenceGreedyDecode()
    {
        var tools = File.ReadAllBytes(Assets.Tools);
        var golden = Assets.GoldenJson();
        var budget = Assets.GoldenBudget(golden);
        foreach (var threads in new[] { 1, 8 })
        {
            using var caller = new ToolCaller(model, tools, threads: threads);
            var output = new byte[4096];
            foreach (var item in golden.GetProperty("cases").EnumerateArray())
            {
                var result = caller.Complete(Encoding.UTF8.GetBytes(item.GetProperty("query").GetString()!), output, budget);
                var expected = item.GetProperty("text").GetString()!;
                var actual = Encoding.UTF8.GetString(output, 0, result.Bytes);
                if (item.GetProperty("query").GetString() == LagosQuery)
                    Assert.True(actual.StartsWith(expected, StringComparison.Ordinal) || actual.StartsWith(LagosTieBranch, StringComparison.Ordinal),
                        $"Lagos greedy left both tie branches:\n{actual}");
                else
                    Assert.StartsWith(expected, actual);
            }
        }
    }

    [Fact]
    public unsafe void ConfidenceHeadMatchesReference()
    {
        var ids = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/ids.bin"))).ToArray();
        var expected = double.Parse(File.ReadAllText(Path.Combine(Assets.Golden, "case0/confidence.txt")), System.Globalization.CultureInfo.InvariantCulture);
        var reference = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/pooled.bin"))).ToArray();
        float* pooled = stackalloc float[3072];
        foreach (var chunk in new[] { 64, 9 })
        {
            using var engine = new Engine(model, 512, chunk, 4);
            engine.Pooling = true;
            engine.Feed(ids);
            var logit = engine.ConfidenceLogit(pooled);
            var cosine = Cosine(new ReadOnlySpan<float>(pooled, 3072).ToArray(), reference);
            Assert.True(cosine > 0.9995, $"chunk {chunk}: cosine {cosine} logit {logit} expected {expected}");
            static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
            Assert.InRange(Sigmoid(logit), Sigmoid(expected) - 0.05, Sigmoid(expected) + 0.05);
        }
    }

    static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return dot / Math.Sqrt(na * nb);
    }
}
