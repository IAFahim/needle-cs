using System;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

// Engine chunk/sequential parity (issue #17 precondition). Prompt-lookup
// speculation feeds pick+drafts as one chunk and verifies each position
// against the chunk logits; that is only lossless when a chunk feed computes
// the same per-row logits as sequential single-token feeds. Single-threaded
// this holds bit-for-bit (all T=1..5 rows, every layer stream, argmax).
// Multi-threaded it does not (measured ~0.36 abs logit drift from layer 0 at
// 4 threads, same binaries, deterministic): distributing rows across workers
// changes the computed values, so a borderline argmax can flip. Until the
// kernel work (W13/W20) closes that gap, the W17 identity gate runs
// single-threaded and speculation stays default-off.
public sealed class ChunkParityTests : IDisposable
{
    readonly Model model;
    readonly ITestOutputHelper output;

    public ChunkParityTests(ITestOutputHelper output)
    {
        this.output = output;
        Assets.Require();
        model = Model.Open(Assets.Model);
    }

    public void Dispose() => model.Dispose();

    [Fact]
    public unsafe void ChunkFeedMatchesSequentialSinglesSingleThreaded()
    {
        var warmup = new int[100];
        for (var i = 0; i < warmup.Length; i++) warmup[i] = 100 + (i % 50);
        var toks = new int[] { 500, 600, 700, 800, 900 };
        for (var n = 1; n <= toks.Length; n++)
        {
            using var chunked = new Engine(model, 2048, 64, 1) { Pooling = true };
            using var sequential = new Engine(model, 2048, 64, 1) { Pooling = true };
            chunked.Feed(warmup, computeLogits: false);
            sequential.Feed(warmup, computeLogits: false);
            var slice = toks[..n];
            chunked.Feed(slice, computeLogits: true);
            foreach (var t in slice) sequential.Feed(new ReadOnlySpan<int>(&t, 1), computeLogits: true);
            Assert.True(chunked.Logits.ToArray().AsSpan().SequenceEqual(sequential.Logits.ToArray()), $"T={n}: chunk logits differ from sequential singles");
        }
        output.WriteLine("single-threaded T=1..5 chunk logits bit-identical to sequential singles");
    }
}
