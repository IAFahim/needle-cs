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
    public void ForwardMatchesReferenceLogits()
    {
        var ids = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/ids.bin"))).ToArray();
        var expected = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/logits.bin"))).ToArray();
        foreach (var (chunk, threads) in new[] { (256, 1), (64, 1), (7, 1), (64, 8), (5, 6) })
        {
            using var engine = new Engine(model, 512, chunk, threads);
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
                Assert.StartsWith(expected, Encoding.UTF8.GetString(output, 0, result.Bytes));
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
            Assert.InRange(logit, expected - 0.15, expected + 0.15);
        }
    }

    static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return dot / Math.Sqrt(na * nb);
    }
}
