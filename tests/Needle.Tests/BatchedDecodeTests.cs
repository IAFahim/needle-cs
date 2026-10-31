using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Needle.Cli;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

// Issue #26: multi-sequence batched decode. One shared forward per step must
// leave every sequence's output identical to decoding it alone: same tokens,
// calls, reasoning, confidence and token counts. The tests below pin that
// identity at the engine level (bit-exact logits) and at the session level
// (game prompts, mixed grammars, ragged finishes, per-lane cancel), then
// report the throughput scaling the issue asks for.
public sealed unsafe class BatchedDecodeTests : IDisposable
{
    readonly Model model;
    readonly ITestOutputHelper output;
    readonly int[] ids;

    public BatchedDecodeTests(ITestOutputHelper output)
    {
        Assets.Require();
        this.output = output;
        model = Model.Open(Assets.Model);
        ids = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/ids.bin"))).ToArray();
    }

    public void Dispose() => model.Dispose();

    static NeedleOptions Options(int threads) => new() { Threads = threads, AutoDate = false, System = "", RetrievalTopK = 32 };

    static float[] LogitsOf(Engine engine)
    {
        var copy = new float[engine.Logits.Length];
        engine.Logits.CopyTo(copy);
        return copy;
    }

    static void AssertLogitsEqual(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actual[i], $"{label}: logit {i} differs ({expected[i]} vs {actual[i]})");
    }

    float PooledConfidence(Engine engine, float[] pooled, float[] scratch)
    {
        fixed (float* p = pooled)
        fixed (float* s = scratch)
            return engine.ConfidenceLogit(p, s);
    }

    float PooledConfidence(BatchedEngine engine, int slot, float[] pooled, float[] scratch)
    {
        fixed (float* p = pooled)
        fixed (float* s = scratch)
            return engine.ConfidenceLogitForSlot(slot, p, s);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void EngineBatchMatchesSingleFeed(int threads)
    {
        var policy = ThreadPolicy.Default.WithThreads(threads);
        using var single0 = new Engine(model, policy, 512, 16) { Pooling = true };
        using var batched0 = new BatchedEngine(model, policy, 512, 16, 1);
        single0.Feed(ids.AsSpan(0, 1), true);
        batched0.FeedSlot(0, ids.AsSpan(0, 1), true);
        AssertLogitsEqual(LogitsOf(single0), BatchRow(batched0, 0), "single token");
        using var singleA = new Engine(model, policy, 512, 16) { Pooling = true };
        using var singleB = new Engine(model, policy, 512, 16) { Pooling = true };
        using var batched = new BatchedEngine(model, policy, 512, 16, 2);
        var prefix = ids.AsSpan(0, 80);
        var queryA = ids.AsSpan(80, 40);
        var queryB = ids.AsSpan(120, 5);

        singleA.Feed(prefix, true);
        singleA.Mark();
        singleB.Feed(prefix, true);
        singleB.Mark();
        foreach (var slot in new[] { 0, 1 })
        {
            batched.ResetSlot(slot);
            batched.FeedSlot(slot, prefix, true);
        }
        AssertLogitsEqual(LogitsOf(singleA), BatchRow(batched, 0), "prefix slot 0");
        foreach (var slot in new[] { 0, 1 })
        {
            batched.ResetSlot(slot);
            batched.FeedSlot(slot, prefix, false);
            batched.MarkSlot(slot);
        }
        singleA.Feed(queryA);
        batched.FeedSlot(0, queryA);
        AssertLogitsEqual(LogitsOf(singleA), BatchRow(batched, 0), "prefill slot 0");
        singleB.Feed(queryB);
        batched.FeedSlot(1, queryB);
        AssertLogitsEqual(LogitsOf(singleB), BatchRow(batched, 0), "prefill slot 1");

        // One shared step over rows from both slots.
        var tokens = stackalloc int[2] { queryA[^1], queryB[^1] };
        var rowSlots = stackalloc int[2] { 0, 1 };
        var heads = stackalloc int[2] { 0, 1 };
        singleA.Feed(new ReadOnlySpan<int>(tokens, 1));
        singleB.Feed(new ReadOnlySpan<int>(tokens + 1, 1));
        batched.ForwardBatch(tokens, rowSlots, 2, heads, 2);
        AssertLogitsEqual(LogitsOf(singleA), BatchRow(batched, 0), "batched row slot 0");
        AssertLogitsEqual(LogitsOf(singleB), BatchRow(batched, 1), "batched row slot 1");

        var dim = model.Geometry.Dim;
        var queries = model.Confidence->Queries;
        var rows = (model.Geometry.Layers + 1) * model.Confidence->Probes;
        var pooledA = new float[queries * dim];
        var scratchA = new float[rows * (dim + 1)];
        var pooledB = new float[queries * dim];
        var scratchB = new float[rows * (dim + 1)];
        Assert.True(PooledConfidence(singleA, pooledA, scratchA) == PooledConfidence(batched, 0, pooledB, scratchB), "confidence slot 0 differs");
        Assert.True(pooledA.SequenceEqual(pooledB), "pooled embeddings slot 0 differ");

        // Rewind on both paths, then decode in the opposite slot order.
        singleA.Rewind();
        singleB.Rewind();
        batched.RewindSlot(0);
        batched.RewindSlot(1);
        var tokens2 = stackalloc int[2] { queryB[^1], queryA[^1] };
        var rowSlots2 = stackalloc int[2] { 1, 0 };
        singleB.Feed(new ReadOnlySpan<int>(tokens2, 1));
        singleA.Feed(new ReadOnlySpan<int>(tokens2 + 1, 1));
        batched.ForwardBatch(tokens2, rowSlots2, 2, heads, 2);
        AssertLogitsEqual(LogitsOf(singleB), BatchRow(batched, 0), "reordered row slot 1");
        AssertLogitsEqual(LogitsOf(singleA), BatchRow(batched, 1), "reordered row slot 0");
    }

    static float[] BatchRow(BatchedEngine engine, int compactRow)
    {
        var vocab = engine.VocabSize;
        var copy = new float[vocab];
        var source = new ReadOnlySpan<float>(engine.BatchLogitsRow(compactRow), vocab);
        source.CopyTo(copy);
        return copy;
    }

    static void AssertSameCallResult(string label, CallResult expected, CallResult actual)
    {
        Assert.True(expected.Success == actual.Success, $"{label}: success {expected.Success} vs {actual.Success} ({expected.Error} vs {actual.Error})");
        Assert.Equal(expected.Error, actual.Error);
        Assert.Equal(expected.FunctionCalls.Count, actual.FunctionCalls.Count);
        for (var i = 0; i < expected.FunctionCalls.Count; i++)
        {
            Assert.Equal(expected.FunctionCalls[i].Name, actual.FunctionCalls[i].Name);
            Assert.Equal(expected.FunctionCalls[i].ArgumentsJson, actual.FunctionCalls[i].ArgumentsJson);
        }
        Assert.Equal(expected.SuppressedCalls.Count, actual.SuppressedCalls.Count);
        Assert.Equal(expected.Reasoning, actual.Reasoning);
        Assert.Equal(expected.Response, actual.Response);
        Assert.True(expected.Confidence == actual.Confidence, $"{label}: confidence {expected.Confidence} vs {actual.Confidence}");
        Assert.True(expected.DecodeConfidence == actual.DecodeConfidence, $"{label}: decode confidence {expected.DecodeConfidence} vs {actual.DecodeConfidence}");
        Assert.Equal(expected.PrefillTokens, actual.PrefillTokens);
        Assert.Equal(expected.DecodeTokens, actual.DecodeTokens);
    }

    CallResult[] RunSequential(string tools, string[] queries, NeedleOptions options)
    {
        using var session = new NeedleSession(model, tools, options);
        return queries.Select(q =>
        {
            session.Reset();
            return session.Complete(q);
        }).ToArray();
    }

    [Fact]
    public void BatchedMatchesSequentialOnGamePrompts()
    {
        var options = Options(4);
        var tools = BenchPrompts.GameToolsJson;
        var queries = BenchPrompts.Game;
        var expected = RunSequential(tools, queries, options);
        using var batch = new BatchedSession(model, options, queries.Length);
        foreach (var _ in queries) batch.Open(tools);
        var actual = batch.Complete(queries);
        for (var i = 0; i < queries.Length; i++)
        {
            AssertSameCallResult($"game prompt {i} ({queries[i]})", expected[i], actual[i]);
            output.WriteLine($"[{i}] {queries[i]} -> {Describe(actual[i])} decode={actual[i].DecodeTokens} conf={actual[i].Confidence:0.###}");
        }
    }

    [Fact]
    public void MixedToolsetsDecodeIdentically()
    {
        var options = Options(4);
        var lanes = new[]
        {
            (Tools: File.ReadAllText(Assets.Tools), Query: "what's it like in Lagos right now?"),
            (Tools: BenchPrompts.GameToolsJson, Query: "move to 120, 45"),
            (Tools: """[{"name":"set_light","description":"Turn a light on or off.","parameters":{"type":"object","properties":{"state":{"type":"string","enum":["on","off"]},"room":{"type":"string","description":"Room name"}},"required":["state","room"]}}]""", Query: "turn off the kitchen light"),
            (Tools: """[{"type":"function","function":{"name":"getWeather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"cityName":{"type":"string","description":"City name"}},"required":["cityName"]}}}]""", Query: "weather in Paris please"),
            (Tools: Extraction.ToolsJson("""{"title":"booking","type":"object","properties":{"guest":{"type":"string"},"nights":{"type":"integer"},"city":{"type":"string"}},"required":["guest","nights"]}"""), Query: "Maria Lopez would like to stay 3 nights in Lisbon."),
        };
        var expected = lanes.Select(l =>
        {
            using var session = new NeedleSession(model, l.Tools, options);
            return session.Complete(l.Query);
        }).ToArray();
        using var batch = new BatchedSession(model, options, lanes.Length);
        foreach (var lane in lanes) batch.Open(lane.Tools);
        var actual = batch.Complete(lanes.Select(l => l.Query).ToArray());
        for (var i = 0; i < lanes.Length; i++)
        {
            AssertSameCallResult($"mixed lane {i} ({lanes[i].Query})", expected[i], actual[i]);
            output.WriteLine($"lane {i}: {Describe(actual[i])}");
        }
    }

    [Fact]
    public void CancelledLaneDoesNotDisturbOthers()
    {
        var options = Options(4);
        var tools = BenchPrompts.GameToolsJson;
        var queries = new[] { "stop everything", "give 50 gold to Borin then move to 10, 20", "form a circle" };
        var expected = RunSequential(tools, queries, options);
        using var batch = new BatchedSession(model, options, queries.Length);
        foreach (var _ in queries) batch.Open(tools);
        // A pre-turn cancel is cleared at turn start, exactly like the
        // single-sequence session clears RequestCancel on entry: the lane
        // runs normally and identically.
        batch.RequestCancel(1);
        var actual = batch.Complete(queries);
        for (var i = 0; i < queries.Length; i++) AssertSameCallResult($"lane {i}", expected[i], actual[i]);
    }

    [Fact]
    public void MidFlightCancelStopsOnlyThatLane()
    {
        var options = Options(4);
        var tools = BenchPrompts.GameToolsJson;
        var queries = BenchPrompts.Game;
        var expected = RunSequential(tools, queries, options);
        for (var attempt = 0; attempt < 8; attempt++)
        {
            using var batch = new BatchedSession(model, options, queries.Length);
            foreach (var _ in queries) batch.Open(tools);
            batch.TurnGate = 0;
            var task = Task.Run(() => batch.Complete(queries));
            var spin = new SpinWait();
            var deadline = Environment.TickCount64 + 120_000;
            while (Volatile.Read(ref batch.TurnGate) == 0 && Environment.TickCount64 < deadline) spin.SpinOnce();
            Assert.NotEqual(0, Volatile.Read(ref batch.TurnGate));
            batch.RequestCancel(3);
            var actual = task.Result;
            if (!actual[3].Success && actual[3].Error == "cancelled")
            {
                for (var i = 0; i < queries.Length; i++)
                    if (i != 3) AssertSameCallResult($"lane {i}", expected[i], actual[i]);
                output.WriteLine($"mid-flight cancel landed on attempt {attempt}");
                return;
            }
        }
        Assert.Fail("cancel never landed mid-flight in 8 attempts");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public void ThroughputScalesWithBatchSize(int size)
    {
        var options = Options(4);
        var tools = BenchPrompts.GameToolsJson;
        var queries = BenchPrompts.Game.Take(size).ToArray();
        var expected = RunSequential(tools, queries, options);
        using var batch = new BatchedSession(model, options, size);
        foreach (var _ in queries) batch.Open(tools);
        batch.Complete(queries);
        var sw = Stopwatch.StartNew();
        var actual = batch.Complete(queries);
        sw.Stop();
        for (var i = 0; i < size; i++) AssertSameCallResult($"batch {size} lane {i}", expected[i], actual[i]);
        var decodeTokens = actual.Sum(r => r.DecodeTokens);
        var seconds = sw.Elapsed.TotalSeconds;
        output.WriteLine($"batch {size}: {decodeTokens} decode tokens in {seconds * 1000:0} ms = {decodeTokens / seconds:0} tok/s aggregate");
    }

    [Fact]
    public void BatchedBatchBeatsSequential()
    {
        var options = Options(4);
        var tools = BenchPrompts.GameToolsJson;
        var queries = BenchPrompts.Game.Take(4).ToArray();
        using var batch = new BatchedSession(model, options, 4);
        foreach (var _ in queries) batch.Open(tools);
        batch.Complete(queries);
        var sw = Stopwatch.StartNew();
        var actual = batch.Complete(queries);
        var batchMs = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        var expected = RunSequential(tools, queries, options);
        var sequentialMs = sw.Elapsed.TotalMilliseconds;
        for (var i = 0; i < queries.Length; i++) AssertSameCallResult($"lane {i}", expected[i], actual[i]);
        output.WriteLine($"batch-4 {batchMs:0} ms vs sequential {sequentialMs:0} ms (x{sequentialMs / batchMs:0.0})");
        Assert.True(batchMs < sequentialMs, $"batched ({batchMs:0} ms) should beat sequential ({sequentialMs:0} ms)");
    }

    static string Describe(CallResult result) =>
        result.Success
            ? (result.FunctionCalls.Count > 0
                ? string.Join("+", result.FunctionCalls.Select(c => c.Name))
                : $"respond:{result.Response}")
            : $"FAIL:{result.Error}";
}
