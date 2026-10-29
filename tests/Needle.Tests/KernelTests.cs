using System;
using Xunit;

namespace Needle.Tests;

public sealed unsafe class KernelTests
{
    [Fact]
    public void HadamardIsOrthonormalAndSelfInverse()
    {
        var x = stackalloc float[128];
        var original = stackalloc float[128];
        var random = new Random(7);
        for (var i = 0; i < 128; i++) original[i] = x[i] = (float)random.NextDouble() - 0.5f;
        Quant.Hadamard(x);
        double before = 0, after = 0;
        for (var i = 0; i < 128; i++) { before += original[i] * original[i]; after += x[i] * x[i]; }
        Assert.Equal(before, after, 4);
        Quant.Hadamard(x);
        for (var i = 0; i < 128; i++) Assert.Equal(original[i], x[i], 5);
    }

    [Fact]
    public void FakeA8SnapsToTheInt8Grid()
    {
        var x = stackalloc float[16];
        for (var i = 0; i < 16; i++) x[i] = (i - 8) * 0.37f;
        var scale = Quant.FakeA8(x, 16);
        for (var i = 0; i < 16; i++) Assert.Equal(Math.Round(x[i] / scale), x[i] / scale, 4);
        Assert.Equal(127, Math.Round(Math.Abs(x[0]) / scale));
    }

    [Fact]
    public void ExpMatchesLibmWithinSingleRounding()
    {
        var input = stackalloc float[8];
        for (var i = 0; i < 8; i++) input[i] = -9f + 2.7f * i;
        var exp = Ops.Exp(System.Runtime.Intrinsics.X86.Avx.LoadVector256(input));
        for (var i = 0; i < 8; i++) Assert.InRange(exp[i] / MathF.Exp(input[i]), 1 - 2e-6, 1 + 2e-6);
    }
}

public sealed class BackendTests
{
    [Fact]
    public void ScalarBackendIsActiveWhenRequested()
    {
        var requested = Environment.GetEnvironmentVariable("NEEDLE_SCALAR") != null;
        Assert.Equal(!requested, Simd.Enabled || !System.Runtime.Intrinsics.X86.Avx2.IsSupported);
    }
}

public sealed unsafe class AttentionTileTests
{
    const int Positions = 45;

    [Theory]
    [InlineData(2, 6, 0, 45, false)]
    [InlineData(2, 6, 0, 45, true)]
    [InlineData(2, 6, 13, 32, true)]
    [InlineData(2, 6, 41, 4, true)]
    [InlineData(2, 3, 5, 40, false)]
    [InlineData(2, 5, 0, 45, true)]
    [InlineData(2, 2, 0, 17, false)]
    [InlineData(8, 1, 0, 45, true)]
    [InlineData(8, 1, 20, 25, false)]
    [InlineData(1, 6, 0, 1, true)]
    [InlineData(2, 6, 7, 2, false)]
    public void TileMatchesScalarReference(int kvHeads, int heads, int first, int count, bool capture)
    {
        var random = new Random(kvHeads * 1000 + heads * 100 + first + count);
        var queries = stackalloc sbyte[heads * 48];
        var queryScales = stackalloc float[heads];
        var keys = stackalloc sbyte[Positions * kvHeads * 48];
        var values = stackalloc sbyte[Positions * kvHeads * 64];
        var keyScales = stackalloc float[Positions * kvHeads];
        var valueScales = stackalloc float[Positions * kvHeads];
        for (var i = 0; i < heads * 48; i++) queries[i] = (sbyte)random.Next(-127, 128);
        for (var i = 0; i < Positions * kvHeads * 48; i++) keys[i] = (sbyte)random.Next(-127, 128);
        for (var i = 0; i < Positions * kvHeads * 64; i++) values[i] = (sbyte)random.Next(-127, 128);
        for (var i = 0; i < heads; i++) queryScales[i] = 0.4f + 0.1f * i;
        for (var i = 0; i < Positions * kvHeads; i++)
        {
            keyScales[i] = 0.0004f * (1 + random.Next(4));
            valueScales[i] = 0.003f * (1 + random.Next(4));
        }
        var kv = kvHeads - 1;
        var expectedOutput = stackalloc float[heads * 64];
        var expectedProbabilities = stackalloc float[heads * Positions];
        var expectedScores = stackalloc float[Attention.ScoreFloats(count)];
        Scalar.AttentionTile(Tile(queries, queryScales, heads, keys, keyScales, values, valueScales, kvHeads, kv, first, count, expectedScores, expectedOutput, capture ? expectedProbabilities : null));

        var actualOutput = stackalloc float[heads * 64];
        var actualProbabilities = stackalloc float[heads * Positions];
        var actualScores = stackalloc float[Attention.ScoreFloats(count)];
        Check(queries, queryScales, heads, keys, keyScales, values, valueScales, kvHeads, kv, first, count, capture, expectedOutput, expectedProbabilities, actualScores, actualOutput, actualProbabilities, 0);
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported)
            Check(queries, queryScales, heads, keys, keyScales, values, valueScales, kvHeads, kv, first, count, capture, expectedOutput, expectedProbabilities, actualScores, actualOutput, actualProbabilities, 1);
        if (System.Runtime.Intrinsics.X86.AvxVnni.IsSupported)
            Check(queries, queryScales, heads, keys, keyScales, values, valueScales, kvHeads, kv, first, count, capture, expectedOutput, expectedProbabilities, actualScores, actualOutput, actualProbabilities, 2);
    }

    static AttentionTile Tile(
        sbyte* queries, float* queryScales, int heads, sbyte* keys, float* keyScales, sbyte* values, float* valueScales,
        int kvHeads, int kv, int first, int count, float* scores, float* output, float* probabilities) => new()
    {
        Queries = queries, QueryScales = queryScales, Heads = heads,
        Keys = keys + kv * 48, KeyScales = keyScales + kv, Values = values + kv * 64, ValueScales = valueScales + kv,
        KeyStride = kvHeads * 48, ValueStride = kvHeads * 64, ScaleStride = kvHeads, First = first, Count = count,
        Scores = scores, Output = output, Probabilities = probabilities, ProbabilityStride = Positions,
    };

    static void Check(
        sbyte* queries, float* queryScales, int heads, sbyte* keys, float* keyScales, sbyte* values, float* valueScales,
        int kvHeads, int kv, int first, int count, bool capture, float* expectedOutput, float* expectedProbabilities,
        float* scores, float* output, float* probabilities, int variant)
    {
        for (var i = 0; i < heads * 64; i++) output[i] = float.NaN;
        for (var i = 0; i < heads * Positions; i++) probabilities[i] = 0f;
        var tile = Tile(queries, queryScales, heads, keys, keyScales, values, valueScales, kvHeads, kv, first, count, scores, output, capture ? probabilities : null);
        if (variant == 0) Attention.Run(tile);
        else if (variant == 1) Attention.GroupAvx2(tile);
        else Attention.GroupVnni(tile);
        var peak = 0f;
        for (var i = 0; i < heads * 64; i++) peak = MathF.Max(peak, MathF.Abs(expectedOutput[i]));
        var tolerance = variant != 1 && System.Runtime.Intrinsics.X86.AvxVnni.IsSupported ? 5e-4f : 2e-5f;
        for (var i = 0; i < heads * 64; i++) Assert.InRange(output[i] - expectedOutput[i], -tolerance * peak, tolerance * peak);
        if (!capture) return;
        for (var i = 0; i < heads * Positions; i++) Assert.InRange(probabilities[i] - expectedProbabilities[i], -1e-5f, 1e-5f);
    }
}

public sealed unsafe class AttentionBatchTests
{
    const int Capacity = 260;

    [Theory]
    [InlineData(1, 12, 2, 1, 150, 0, 0, false)]
    [InlineData(4, 12, 2, 1, 150, 0, 0, false)]
    [InlineData(4, 12, 2, 1, 20, 0, 0, false)]
    [InlineData(4, 12, 2, 1, 95, 0, 0, false)]
    [InlineData(8, 12, 2, 1, 200, 0, 0, false)]
    [InlineData(4, 12, 2, 1, 200, 64, 0, false)]
    [InlineData(4, 12, 2, 5, 40, 0, 0, false)]
    [InlineData(1, 12, 2, 5, 40, 0, 0, false)]
    [InlineData(4, 12, 2, 3, 100, 32, 0, false)]
    [InlineData(4, 8, 8, 3, 0, 0, 70, true)]
    [InlineData(4, 8, 8, 9, 0, 0, 9, false)]
    [InlineData(4, 12, 1, 2, 30, 0, 0, false)]
    [InlineData(2, 6, 1, 1, 90, 0, 0, false)]
    public void BatchMatchesScalarReference(int threads, int heads, int kvHeads, int tokens, int position, int window, int frames, bool capture)
    {
        var random = new Random(heads * 31 + tokens * 7 + position);
        var queries = new sbyte[tokens * heads * 48];
        var queryScales = new float[tokens * heads];
        var keys = new sbyte[Capacity * kvHeads * 48];
        var values = new sbyte[Capacity * kvHeads * 64];
        var keyScales = new float[Capacity * kvHeads];
        var valueScales = new float[Capacity * kvHeads];
        for (var i = 0; i < queries.Length; i++) queries[i] = (sbyte)random.Next(-127, 128);
        for (var i = 0; i < keys.Length; i++) keys[i] = (sbyte)random.Next(-127, 128);
        for (var i = 0; i < values.Length; i++) values[i] = (sbyte)random.Next(-127, 128);
        for (var i = 0; i < queryScales.Length; i++) queryScales[i] = 0.3f + 0.05f * (i % 7);
        for (var i = 0; i < keyScales.Length; i++) { keyScales[i] = 0.0003f * (1 + random.Next(4)); valueScales[i] = 0.003f * (1 + random.Next(4)); }
        var expected = new float[tokens * heads * 64];
        var expectedProbabilities = new float[tokens * heads * Math.Max(frames, 1)];
        var actual = new float[expected.Length];
        var actualProbabilities = new float[expectedProbabilities.Length];
        using var pool = new Pool(threads);
        fixed (sbyte* q = queries, k = keys, v = values)
        fixed (float* qs = queryScales, ks = keyScales, vs = valueScales, e = expected, ep = expectedProbabilities, a = actual, ap = actualProbabilities)
        {
            var scores = stackalloc float[Capacity + 16];
            var group = heads / kvHeads;
            for (var t = 0; t < tokens; t++)
                for (var h = 0; h < heads; h++)
                {
                    var kv = h / group;
                    var p = position + t;
                    var first = window == 0 ? 0 : Math.Max(0, p - window + 1);
                    var last = frames > 0 ? frames - 1 : p;
                    Scalar.AttentionHead(
                        q + (t * heads + h) * 48, qs[t * heads + h], k + kv * 48, ks + kv, v + kv * 64, vs + kv,
                        kvHeads * 48, kvHeads * 64, kvHeads, first, last, scores, e + (t * heads + h) * 64,
                        capture ? ep + (t * heads + h) * frames : null);
                }
            var batch = new AttentionBatch
            {
                Queries = q, QueryScales = qs, Keys = k, KeyScales = ks, Values = v, ValueScales = vs, Output = a,
                Probabilities = capture ? ap : null, Heads = heads, KvHeads = kvHeads, QkDim = 48, VDim = 64,
                Position = position, Window = window, Capacity = Capacity, Frames = frames,
            };
            AttentionBatch.Run(pool, &batch, tokens);
        }
        var tolerance = kvHeads < heads && System.Runtime.Intrinsics.X86.AvxVnni.IsSupported && Simd.Enabled ? 5e-4f : 3e-5f;
        for (var unit = 0; unit < tokens * heads; unit++)
        {
            var peak = 0f;
            for (var d = 0; d < 64; d++) peak = MathF.Max(peak, MathF.Abs(expected[unit * 64 + d]));
            for (var d = 0; d < 64; d++) Assert.InRange(actual[unit * 64 + d] - expected[unit * 64 + d], -tolerance * peak, tolerance * peak);
        }
        if (capture)
            for (var i = 0; i < actualProbabilities.Length; i++) Assert.InRange(actualProbabilities[i] - expectedProbabilities[i], -1e-5f, 1e-5f);
    }

    [Theory]
    [InlineData(0, 2)]
    [InlineData(1, 3)]
    [InlineData(2, 2)]
    public void SplitPartialsMergeToTheWholeResult(int variant, int pieces)
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported || !System.Runtime.Intrinsics.X86.Fma.IsSupported) return;
        if (variant == 2 && !System.Runtime.Intrinsics.X86.AvxVnni.IsSupported) return;
        const int positions = 101;
        const int heads = 6;
        var random = new Random(pieces * 13 + variant);
        var queries = stackalloc sbyte[heads * 48];
        var queryScales = stackalloc float[heads];
        var keys = stackalloc sbyte[positions * 48];
        var values = stackalloc sbyte[positions * 64];
        var keyScales = stackalloc float[positions];
        var valueScales = stackalloc float[positions];
        for (var i = 0; i < heads * 48; i++) queries[i] = (sbyte)random.Next(-127, 128);
        for (var i = 0; i < positions * 48; i++) keys[i] = (sbyte)random.Next(-127, 128);
        for (var i = 0; i < positions * 64; i++) values[i] = (sbyte)random.Next(-127, 128);
        for (var i = 0; i < heads; i++) queryScales[i] = 0.4f + 0.1f * i;
        for (var i = 0; i < positions; i++) { keyScales[i] = 0.0005f * (1 + random.Next(4)); valueScales[i] = 0.003f * (1 + random.Next(4)); }
        var whole = stackalloc float[heads * 64];
        var wholeScores = stackalloc float[Attention.ScoreFloats(positions)];
        var merged = stackalloc float[heads * 64];
        var partials = stackalloc float[pieces * Attention.PartialFloats];
        Run(variant, queries, queryScales, keys, keyScales, values, valueScales, 0, positions, wholeScores, whole, null);
        var length = (positions + pieces - 1) / pieces;
        var scores = stackalloc float[Attention.ScoreFloats(length)];
        for (var piece = 0; piece < pieces; piece++)
        {
            var begin = piece * length;
            var count = Math.Min(length, positions - begin);
            Run(variant, queries, queryScales, keys, keyScales, values, valueScales, begin, count, scores, partials + piece * Attention.PartialFloats, partials + piece * Attention.PartialFloats);
        }
        Attention.MergePartials(partials, pieces, heads, merged);
        var peak = 0f;
        for (var i = 0; i < heads * 64; i++) peak = MathF.Max(peak, MathF.Abs(whole[i]));
        var tolerance = variant == 2 ? 5e-4f : 3e-5f;
        for (var i = 0; i < heads * 64; i++) Assert.InRange(merged[i] - whole[i], -tolerance * peak, tolerance * peak);
    }

    static void Run(
        int variant, sbyte* queries, float* queryScales, sbyte* keys, float* keyScales, sbyte* values, float* valueScales,
        int first, int count, float* scores, float* output, float* partial)
    {
        var tile = new AttentionTile
        {
            Queries = queries, QueryScales = queryScales, Heads = 6, Keys = keys, KeyScales = keyScales, Values = values, ValueScales = valueScales,
            KeyStride = 48, ValueStride = 64, ScaleStride = 1, First = first, Count = count, Scores = scores, Output = output,
        };
        if (partial != null)
        {
            tile.Peaks = partial + Attention.MaxTileHeads * 64;
            tile.Sums = tile.Peaks + 8;
        }
        if (variant == 2) Attention.GroupVnni(tile);
        else Attention.GroupAvx2(tile);
    }
}
