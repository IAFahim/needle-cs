using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Xunit;

namespace Needle.Tests;

public sealed class TierDispatchTests
{
    static readonly TierSwitches None = default;

    [Fact]
    public void Zen4PicksEvexVnni() =>
        Assert.Equal(CpuTier.Avx512Vnni, Tiers.Resolve(null, CpuFeatures.Zen4, None));

    [Fact]
    public void AlderLakePicksAvxVnni() =>
        Assert.Equal(CpuTier.AvxVnni, Tiers.Resolve(null, CpuFeatures.AlderLake, None));

    [Fact]
    public void SapphireRapidsPicksEvexVnni() =>
        Assert.Equal(CpuTier.Avx512Vnni, Tiers.Resolve(null, CpuFeatures.SapphireRapids, None));

    [Fact]
    public void PlainAvx2PicksAvx2() =>
        Assert.Equal(CpuTier.Avx2, Tiers.Resolve(null, CpuFeatures.PlainAvx2, None));

    [Fact]
    public void Avx512WithoutVnniStaysOnAvx2()
    {
        var skylakeX = new CpuFeatures { Avx2 = true, Avx512 = true };
        Assert.Equal(CpuTier.Avx2, Tiers.Resolve(null, skylakeX, None));
    }

    [Fact]
    public void VnniInt8OutranksEveryOtherX86Tier()
    {
        var arrowLake = new CpuFeatures { Avx2 = true, AvxVnni = true, AvxVnniInt8 = true };
        Assert.Equal(CpuTier.AvxVnniInt8, Tiers.Resolve(null, arrowLake, None));
        var diamondRapids = new CpuFeatures { Avx2 = true, AvxVnni = true, AvxVnniInt8 = true, Avx512 = true, Avx512Vnni = true };
        Assert.Equal(CpuTier.AvxVnniInt8, Tiers.Resolve(null, diamondRapids, None));
    }

    [Fact]
    public void NoAvxFallsBackToScalarAndArmPicksNeon()
    {
        Assert.Equal(CpuTier.Scalar, Tiers.Resolve(null, default, None));
        Assert.Equal(CpuTier.NeonDot, Tiers.Resolve(null, new CpuFeatures { AdvSimd = true, Dp = true }, None));
        Assert.Equal(CpuTier.Neon, Tiers.Resolve(null, new CpuFeatures { AdvSimd = true, Dp = true }, new TierSwitches { NoDp = true }));
    }

    [Fact]
    public void SwitchesSteerTheAutoPick()
    {
        Assert.Equal(CpuTier.Avx2, Tiers.Resolve(null, CpuFeatures.Zen4, new TierSwitches { NoVnni = true }));
        Assert.Equal(CpuTier.Avx2, Tiers.Resolve(null, CpuFeatures.SapphireRapids, new TierSwitches { NoVnni = true }));
        Assert.Equal(CpuTier.Scalar, Tiers.Resolve(null, CpuFeatures.SapphireRapids, new TierSwitches { Scalar = true }));
    }

    [Fact]
    public void NoEvexSwitchFallsBackAndRefusesTheForcedTier()
    {
        var off = new TierSwitches { NoEvex = true };
        Assert.Equal(CpuTier.Avx2, Tiers.Resolve(null, CpuFeatures.Zen4, off));
        Assert.Equal(CpuTier.AvxVnni, Tiers.Resolve(null, CpuFeatures.SapphireRapids, off));
        Assert.Throws<InvalidOperationException>(() => Tiers.Resolve("avx512vnni", CpuFeatures.Zen4, off));
        Assert.Equal(CpuTier.Avx512Bw, Tiers.Resolve("avx512", CpuFeatures.Zen4, off));
    }

    [Fact]
    public void Zen5WithBothVnniFormsPicksEvexVnni()
    {
        var zen5 = new CpuFeatures { Avx2 = true, AvxVnni = true, Avx512 = true, Avx512Vnni = true };
        Assert.Equal(CpuTier.Avx512Vnni, Tiers.Resolve(null, zen5, None));
    }

    [Fact]
    public void IceLakeWithEvexVnniOnlyPicksEvexVnni()
    {
        var iceLake = new CpuFeatures { Avx2 = true, Avx512 = true, Avx512Vnni = true };
        Assert.Equal(CpuTier.Avx512Vnni, Tiers.Resolve(null, iceLake, None));
    }

    public static IEnumerable<object[]> Overrides() => new[]
    {
        new object[] { "zen4", "avx2", "Avx2" },
        new object[] { "zen4", "avx512", "Avx512Bw" },
        new object[] { "zen4", "AVX512VNNI", "Avx512Vnni" },
        new object[] { "zen4", "scalar", "Scalar" },
        new object[] { "sapphire", "avxvnni", "AvxVnni" },
        new object[] { "sapphire", "avx512vnni", "Avx512Vnni" },
        new object[] { "sapphire", " avx2 ", "Avx2" },
        new object[] { "alder", "avxvnni", "AvxVnni" },
        new object[] { "alder", "avx2", "Avx2" },
        new object[] { "avx2", "avx2", "Avx2" },
    };

    [Theory]
    [MemberData(nameof(Overrides))]
    public void NeedleTierForcesASupportedTier(string machine, string raw, string expected) =>
        Assert.Equal(expected, Tiers.Resolve(raw, Machine(machine), new TierSwitches { NoVnni = true, Scalar = false }).ToString());

    [Theory]
    [InlineData("zen4", "avxvnni")]
    [InlineData("alder", "avx512vnni")]
    [InlineData("alder", "avx512")]
    [InlineData("avx2", "avx512vnni")]
    [InlineData("avx2", "avxvnniint8")]
    [InlineData("zen4", "neondot")]
    [InlineData("zen4", "zen4")]
    public void NeedleTierRejectsWhatTheCpuLacks(string machine, string raw)
    {
        var ex = Assert.Throws<InvalidOperationException>(() => Tiers.Resolve(raw, Machine(machine), None));
        Assert.Contains(raw, ex.Message);
        Assert.Contains(Tiers.Resolve(null, Machine(machine), None).ToString(), ex.Message);
    }

    [Fact]
    public void BlankOverrideMeansAuto()
    {
        Assert.Equal(CpuTier.Avx512Vnni, Tiers.Resolve("", CpuFeatures.Zen4, None));
        Assert.Equal(CpuTier.Avx512Vnni, Tiers.Resolve("  ", CpuFeatures.Zen4, None));
    }

    [Fact]
    public void TierNameParses() =>
        Assert.True(Tiers.TryParseTier("avx512vnni", out var tier) && tier == CpuTier.Avx512Vnni);

    [Fact]
    public void DetectedFeaturesAgreeWithTheActiveTier()
    {
        if (Environment.GetEnvironmentVariable("NEEDLE_TIER") != null || Environment.GetEnvironmentVariable("NEEDLE_SCALAR") != null
            || Environment.GetEnvironmentVariable("NEEDLE_NOVNNI") != null || Environment.GetEnvironmentVariable("NEEDLE_NODP") != null
            || Environment.GetEnvironmentVariable("NEEDLE_NOEVEX") != null)
            return;
        Assert.Equal(Tiers.Resolve(null, Tiers.Detected, None), Tiers.Active);
        Assert.Equal(EvexVnni.Ready, Tiers.Detected.Avx512Vnni);
    }

    static CpuFeatures Machine(string name) => name switch
    {
        "zen4" => CpuFeatures.Zen4,
        "sapphire" => CpuFeatures.SapphireRapids,
        "alder" => CpuFeatures.AlderLake,
        _ => CpuFeatures.PlainAvx2,
    };
}

public sealed unsafe class EvexVnniKernelTests : IDisposable
{
    readonly List<nint> allocations = new();

    public static IEnumerable<object[]> Shapes()
    {
        foreach (var bits in new[] { 2, 4 })
            foreach (var (rows, cols) in new[] { (4, 128), (5, 256), (8, 768), (13, 768), (64, 1024), (97, 768), (576, 768), (24, 3072), (130, 3072), (33, 1152) })
                yield return new object[] { bits, rows, cols };
    }

    // The EVEX kernel must give exactly the managed block's bits, not merely close values.
    [Theory]
    [MemberData(nameof(Shapes))]
    public void EvexBlocksAreBitIdenticalToManagedBlocks(int bits, int rows, int cols)
    {
        if (!EvexVnni.Ready) return;
        var random = new Random(rows * 131 + cols * 7 + bits);
        var w = Weights(random, rows, cols, bits);
        for (var tokens = 1; tokens <= 3; tokens++)
        {
            var a = Prepared(random, tokens, cols, bits);
            for (var token = 0; token < tokens; token++)
            {
                foreach (var (start, end) in new[] { (0, rows), (1, rows), (0, rows - rows / 3), (rows / 2, rows) })
                {
                    if (start >= end) continue;
                    var expected = (float*)Allocate((nuint)rows * 4);
                    var actual = (float*)Allocate((nuint)rows * 4);
                    for (var i = 0; i < rows; i++) expected[i] = actual[i] = 7777f;
                    Gemv.RowsForTest(w, a, token, expected, start, end, evex: false);
                    Gemv.RowsForTest(w, a, token, actual, start, end, evex: true);
                    for (var i = 0; i < rows; i++)
                        Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                            $"bits {bits} rows {rows} cols {cols} token {token} range {start}..{end} row {i}: {expected[i]:R} vs {actual[i]:R}");
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void EvexPairsAreBitIdenticalToManagedPairs(int bits, int rows, int cols)
    {
        if (!EvexVnni.Ready) return;
        var random = new Random(rows * 17 + cols * 3 + bits);
        var w = Weights(random, rows, cols, bits);
        const int tokens = 8;
        var a = Prepared(random, tokens, cols, bits);
        foreach (var token in new[] { 0, 4 })
            foreach (var (start, end) in new[] { (0, rows), (1, rows), (0, rows - rows / 3), (rows / 2, rows) })
            {
                if (start >= end) continue;
                var expected = (float*)Allocate((nuint)(tokens * rows) * 4);
                var actual = (float*)Allocate((nuint)(tokens * rows) * 4);
                for (var i = 0; i < tokens * rows; i++) expected[i] = actual[i] = 7777f;
                Gemv.Tile4ForTest(w, a, token, expected + token * rows, rows, start, end, evex: false);
                Gemv.Tile4ForTest(w, a, token, actual + token * rows, rows, start, end, evex: true);
                for (var i = 0; i < tokens * rows; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                        $"bits {bits} rows {rows} cols {cols} token {token} range {start}..{end} index {i}: {expected[i]:R} vs {actual[i]:R}");
            }
    }

    void* Allocate(nuint bytes)
    {
        var p = (nint)NativeMemory.AlignedAlloc(bytes + 64, 64);
        allocations.Add(p);
        new Span<byte>((void*)p, (int)bytes + 64).Clear();
        return (void*)p;
    }

    Cq Weights(Random random, int rows, int cols, int bits)
    {
        var levels = stackalloc float[16];
        var count = bits == 2 ? 4 : 16;
        for (var i = 0; i < count; i++) levels[i] = (i - (count - 1) / 2f) * 0.4f + (float)(random.NextDouble() - 0.5) * 0.1f;
        var book = (Codebook*)Allocate((nuint)sizeof(Codebook));
        Codebook.Fill(book, levels, count);
        var rowBytes = cols * bits / 8;
        var indices = (byte*)Allocate((nuint)(rows * rowBytes));
        for (var i = 0; i < rows * rowBytes; i++) indices[i] = (byte)random.Next(256);
        var groups = cols / Cq.GroupSize;
        var norms = (ushort*)Allocate((nuint)(rows * groups) * 2);
        for (var i = 0; i < rows * groups; i++)
            norms[i] = BitConverter.HalfToUInt16Bits((Half)((i % 11 == 5 ? 0f : 0.2f + (float)random.NextDouble() * 1.8f) * (i % 3 == 1 ? -1 : 1)));
        return new Cq(indices, norms, book, rows, cols, bits);
    }

    Activations Prepared(Random random, int tokens, int cols, int bits)
    {
        var arena = new Arena((nuint)(tokens * cols * 10 + 4096));
        allocations.Add((nint)arena.Base);
        var a = new Activations(ref arena, tokens, cols);
        var x = (float*)Allocate((nuint)(tokens * cols) * 4);
        for (var i = 0; i < tokens * cols; i++) x[i] = (float)(random.NextDouble() - 0.5) * (1 + i % 7);
        Quant.Prepare(x, tokens, cols, bits, a);
        return a;
    }

    public void Dispose()
    {
        foreach (var p in allocations) NativeMemory.AlignedFree((void*)p);
    }
}
