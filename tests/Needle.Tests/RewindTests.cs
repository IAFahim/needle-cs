using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

public sealed unsafe class RewindTests : IDisposable
{
    const int PrefixLength = 100;
    const int QueryTurns = 50;
    readonly Model model;
    readonly ITestOutputHelper output;
    readonly int[] ids;

    public RewindTests(ITestOutputHelper output)
    {
        Assets.Require();
        this.output = output;
        model = Model.Open(Assets.Model);
        ids = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/ids.bin"))).ToArray();
    }

    public void Dispose() => model.Dispose();

    float PooledConfidence(Engine engine)
    {
        var pooled = new float[model.Confidence->Queries * model.Geometry.Dim];
        fixed (float* p = pooled) return engine.ConfidenceLogit(p);
    }

    float[] Observe(Engine engine) => engine.Logits.ToArray().Append(PooledConfidence(engine)).ToArray();

    Engine Primed()
    {
        var engine = new Engine(model, 512, 16) { Pooling = true };
        engine.Feed(ids.AsSpan(0, PrefixLength), computeLogits: false);
        engine.Mark();
        return engine;
    }

    float[] FreshTurn(ReadOnlySpan<int> query)
    {
        using var fresh = Primed();
        fresh.Feed(query);
        return Observe(fresh);
    }

    [Fact]
    public void RewindIsBitIdenticalToAFreshPrimedEngineForEveryQueryShape()
    {
        using var engine = Primed();
        var shapes = new[] { 1, 2, 3, 9, 10, 16, 17, 40, 33, 5, 1 };
        for (var round = 0; round < 3; round++)
            foreach (var length in shapes)
            {
                var query = ids.AsSpan(PrefixLength + round, length);
                engine.Rewind();
                Assert.Equal(PrefixLength, engine.Length);
                engine.Feed(query);
                Assert.Equal(FreshTurn(query), Observe(engine));
            }
    }

    [Fact]
    public void MarkRightAfterRewindKeepsThePrimedState()
    {
        using var engine = Primed();
        var query = ids.AsSpan(PrefixLength, 20);
        engine.Feed(query);
        engine.Rewind();
        engine.Mark();
        engine.Feed(query);
        Assert.Equal(FreshTurn(query), Observe(engine));
        engine.Rewind();
        engine.Feed(query);
        Assert.Equal(FreshTurn(query), Observe(engine));
    }

    [Fact]
    public void ConfidenceReadRightAfterRewindSeesThePrimedPool()
    {
        using var engine = Primed();
        engine.Feed(ids.AsSpan(PrefixLength, 12));
        engine.Rewind();
        var afterRewind = PooledConfidence(engine);
        using var fresh = Primed();
        Assert.Equal(PooledConfidence(fresh), afterRewind);
    }

    [Fact]
    public void FiftyAlternatingTurnsWithResetMatchAFreshSession()
    {
        var tools = File.ReadAllText(Assets.Tools);
        var options = new NeedleOptions { Threads = 4 };
        var queries = new[] { "what's it like in Lagos right now?", "set a timer for 5 minutes" };
        using var reused = new NeedleSession(model, tools, options);
        var expected = queries.Select(q =>
        {
            using var fresh = new NeedleSession(model, tools, options);
            return Outcome(fresh.Complete(q));
        }).ToArray();
        for (var turn = 0; turn < QueryTurns; turn++)
        {
            var index = turn % queries.Length;
            reused.Reset();
            Assert.Equal(expected[index], Outcome(reused.Complete(queries[index])));
        }
    }

    static string Outcome(CallResult reply) =>
        string.Join("|", reply.Success, reply.Confidence, reply.HeadConfidence, reply.DecodeConfidence, reply.Reasoning, reply.Response,
            string.Join(";", reply.FunctionCalls.Select(c => c.Name + c.ArgumentsJson)));

    [Fact]
    public void RewindAllocatesNothing()
    {
        using var engine = Primed();
        engine.Feed(ids.AsSpan(PrefixLength, 8));
        engine.Rewind();
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 1000; i++) engine.Rewind();
        Assert.Equal(before, GC.GetAllocatedBytesForCurrentThread());
    }

    [Fact]
    public void MeasureRewindCost()
    {
        using var engine = Primed();
        var query = ids.AsSpan(PrefixLength, 24);
        engine.Feed(query);
        var rewindBest = double.MaxValue;
        for (var batch = 0; batch < 9; batch++)
        {
            var clock = Stopwatch.StartNew();
            for (var i = 0; i < 1000; i++) engine.Rewind();
            rewindBest = Math.Min(rewindBest, clock.Elapsed.TotalMilliseconds);
        }
        var turnBest = double.MaxValue;
        for (var batch = 0; batch < 7; batch++)
        {
            var clock = Stopwatch.StartNew();
            for (var i = 0; i < 40; i++)
            {
                engine.Rewind();
                engine.Feed(query.Slice(0, 1), computeLogits: false);
            }
            turnBest = Math.Min(turnBest, clock.Elapsed.TotalMilliseconds * 1000 / 40);
        }
        output.WriteLine($"stateBytes={engine.StateBytes} rewind={rewindBest * 1000:F1} ns rewind+feed1={turnBest:F1} us");
    }
}
