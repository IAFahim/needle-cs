using System;
using System.Linq;
using System.Reflection;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

public sealed unsafe class ChunkDiag2Tests : IDisposable
{
    readonly Model model;
    readonly ITestOutputHelper output;
    public ChunkDiag2Tests(ITestOutputHelper output) { this.output = output; Assets.Require(); model = Model.Open(Assets.Model); }
    public void Dispose() => model.Dispose();

    static readonly Type Core = typeof(EngineCore<NoClock>);
    static object F(object e, string name) => Core.GetField(name, BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(e)!;
    static void* P(object e, string name) => Pointer.Unbox(F(e, name));

    string Cmp(byte* a, byte* b, long bytes)
    {
        long n = 0, first = -1;
        for (long i = 0; i < bytes; i++) if (a[i] != b[i]) { n++; if (first < 0) first = i; }
        return n == 0 ? "same" : $"DIFF {n}/{bytes} first@{first}";
    }

    string CmpF(float* a, float* b, long count)
    {
        long n = 0; double worst = 0; long at = -1;
        for (long i = 0; i < count; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(b[i])) continue;
            n++; var d = Math.Abs((double)a[i] - b[i]); if (d > worst || at < 0) { worst = d; at = i; }
        }
        return n == 0 ? "same" : $"DIFF {n}/{count} max {worst:G6}@{at}";
    }

    [Fact]
    public void Diagnose()
    {
        var warmup = Enumerable.Range(0, 100).Select(i => 100 + (i % 50)).ToArray();
        using var legacy = new Engine(model, 2048, 64, 1) { Pooling = true };
        using var team = new Engine(model, 2048, 64, 1) { Pooling = true };
        legacy.Feed(warmup, computeLogits: false);
        team.Feed(warmup, computeLogits: false);
        legacy.Observer = (_, _) => { };
        legacy.Feed(new[] { 500 }, computeLogits: true);
        team.Feed(new[] { 500 }, computeLogits: true);
        var g = F(legacy, "g");
        int layers = (int)g.GetType().GetField("Layers")!.GetValue(g)!;
        int kvHeads = (int)g.GetType().GetField("KvHeads")!.GetValue(g)!;
        int capacity = (int)F(legacy, "capacity"), kW = (int)F(legacy, "kWidth"), vW = (int)F(legacy, "vWidth"), dim = (int)F(legacy, "dim");
        output.WriteLine($"tier={Tiers.Active} layers={layers} kv={kvHeads} cap={capacity} kW={kW} vW={vW}");
        output.WriteLine("logits " + CmpF((float*)P(legacy, "logits"), (float*)P(team, "logits"), 8192));
        const int pos = 100;
        for (var l = 0; l < layers; l++)
        {
            var k = Cmp((byte*)P(legacy, "keyCache") + ((long)l * capacity + pos) * kW, (byte*)P(team, "keyCache") + ((long)l * capacity + pos) * kW, kW);
            var v = Cmp((byte*)P(legacy, "valueCache") + ((long)l * capacity + pos) * vW, (byte*)P(team, "valueCache") + ((long)l * capacity + pos) * vW, vW);
            var ks = CmpF((float*)P(legacy, "keyScale") + ((long)l * capacity + pos) * kvHeads, (float*)P(team, "keyScale") + ((long)l * capacity + pos) * kvHeads, kvHeads);
            var vs = CmpF((float*)P(legacy, "valueScale") + ((long)l * capacity + pos) * kvHeads, (float*)P(team, "valueScale") + ((long)l * capacity + pos) * kvHeads, kvHeads);
            output.WriteLine($"layer {l}: K {k} V {v} Ks {ks} Vs {vs}");
        }
        foreach (var name in new[] { "residual", "attnProj", "attnOut", "gateOut", "qConv", "input", "preOut", "postOut", "resOut" })
            output.WriteLine($"{name} (first {dim}): " + CmpF((float*)P(legacy, name), (float*)P(team, name), dim / 4));
        output.WriteLine("state " + Cmp((byte*)P(legacy, "state"), (byte*)P(team, "state"), (long)(nuint)F(legacy, "stateBytes")));
    }
}
