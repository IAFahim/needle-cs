using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

public sealed class ChunkDiagTests : IDisposable
{
    readonly Model model;
    readonly ITestOutputHelper output;
    public ChunkDiagTests(ITestOutputHelper output) { this.output = output; Assets.Require(); model = Model.Open(Assets.Model); }
    public void Dispose() => model.Dispose();

    static readonly int[] Warmup = Enumerable.Range(0, 100).Select(i => 100 + (i % 50)).ToArray();
    static readonly int[] Toks = { 500, 600 };

    float[] Run(bool chunk, bool observe, bool pooling, List<float[]>? layers)
    {
        using var e = new Engine(model, 2048, 64, 1) { Pooling = pooling };
        e.Feed(Warmup, computeLogits: false);
        if (observe) e.Observer = (layer, stream) => layers?.Add(stream.ToArray());
        if (chunk) e.Feed(Toks, computeLogits: true);
        else foreach (var t in Toks) e.Feed(new[] { t }, computeLogits: true);
        return e.Logits.ToArray();
    }

    static string Diff(float[] a, float[] b)
    {
        var worst = 0.0; var at = -1; var n = 0;
        for (var i = 0; i < Math.Min(a.Length, b.Length); i++)
        {
            var d = Math.Abs((double)a[i] - b[i]);
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) n++;
            if (d > worst) { worst = d; at = i; }
        }
        return $"bitdiff={n}/{a.Length} maxabs={worst:G6}@{at}";
    }

    [Fact]
    public void Diagnose()
    {
        output.WriteLine($"tier={Tiers.Active} neon={Simd.UseNeon}");
        var tokenTeam = Run(false, false, true, null);
        var singlesLayers = new List<float[]>();
        var singlesLegacy = Run(false, true, true, singlesLayers);
        var chunkLayers = new List<float[]>();
        var chunkObserved = Run(true, true, true, chunkLayers);
        var chunkPool = Run(true, false, true, null);
        var chunkBatch = Run(true, false, false, null);
        var tokenNoPool = Run(false, false, false, null);
        output.WriteLine("tokenTeam vs singlesLegacy  " + Diff(tokenTeam, singlesLegacy));
        output.WriteLine("tokenTeam vs chunkPool      " + Diff(tokenTeam, chunkPool));
        output.WriteLine("tokenTeam vs chunkObserved  " + Diff(tokenTeam, chunkObserved));
        output.WriteLine("singlesLegacy vs chunkObs   " + Diff(singlesLegacy, chunkObserved));
        output.WriteLine("tokenNoPool vs chunkBatch   " + Diff(tokenNoPool, chunkBatch));
        output.WriteLine("tokenTeam vs tokenNoPool    " + Diff(tokenTeam, tokenNoPool));
        // singles observed: per call, layers -1..L-1 for one token; chunk: for two tokens.
        var perCall = singlesLayers.Count / 2;
        var width = singlesLayers[0].Length;
        output.WriteLine($"layers per call={perCall} width={width} chunk calls={chunkLayers.Count}");
        for (var i = 0; i < perCall && i < chunkLayers.Count; i++)
        {
            var c = chunkLayers[i];
            var first = c.AsSpan(0, width).ToArray();
            var second = c.AsSpan(width, width).ToArray();
            output.WriteLine($"layer {i - 1}: tok0 {Diff(singlesLayers[i], first)} | tok1 {Diff(singlesLayers[perCall + i], second)}");
        }
    }
}
