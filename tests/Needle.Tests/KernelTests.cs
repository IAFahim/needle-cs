using System;
using System.Linq;
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

    // Ops.Exp(Vector256) is the x86 AVX/FMA path (mirrors Neon.Exp on ARM,
    // covered by NeonTests.ExpMatchesMathF). It faults on platforms without
    // AVX, so gate on the same predicate the engine uses (CI evidence: run
    // 41030471519 linux-arm64/macos-arm64 PlatformNotSupportedException).
    [Fact]
    public void ExpMatchesLibmWithinSingleRounding()
    {
        if (!Simd.UseAvx) return;
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

public sealed unsafe class HadamardMlpTests : IDisposable
{
    readonly System.Collections.Generic.List<nint> allocations = new();

    public void Dispose()
    {
        foreach (var allocation in allocations) Mem.Free((void*)allocation);
    }

    float* Allocate(int count, Random random, float spread, float offset = 0f)
    {
        var block = (float*)Mem.AllocZeroed((nuint)count * sizeof(float));
        allocations.Add((nint)block);
        for (var i = 0; i < count; i++) block[i] = offset + spread * (float)(random.NextDouble() * 2 - 1);
        return block;
    }

    int* Permutation(int count, Random random)
    {
        var block = (int*)Mem.AllocZeroed((nuint)count * sizeof(int));
        allocations.Add((nint)block);
        for (var i = 0; i < count; i++) block[i] = i;
        for (var i = count - 1; i > 0; i--)
        {
            var j = random.Next(i + 1);
            (block[i], block[j]) = (block[j], block[i]);
        }
        return block;
    }

    MlpBatch* Batch(int dim, int rows, int cols, float scale, int seed)
    {
        var random = new Random(seed);
        var width = rows * cols;
        var weights = (float**)Mem.AllocZeroed(13 * (nuint)sizeof(float*));
        allocations.Add((nint)weights);
        weights[0] = Allocate(width, random, 0.5f, 1f);
        weights[1] = Allocate(width, random, 0.5f, 1f);
        weights[2] = Allocate(width, random, 0.2f);
        weights[3] = Allocate(width, random, 0.5f, 1f);
        weights[4] = Allocate(width, random, 0.5f, 1f);
        for (var i = 5; i < 11; i++) weights[i] = Allocate(width, random, 0.3f);
        weights[11] = Allocate(dim * 8, random, 0.1f);
        weights[12] = Allocate(8 * width, random, 0.3f);
        var batch = (MlpBatch*)Mem.AllocZeroed((nuint)sizeof(MlpBatch));
        allocations.Add((nint)batch);
        *batch = new MlpBatch
        {
            Weights = (MlpWeights*)weights, PreNorm = Allocate(dim, random, 0.3f),
            PermA = Permutation(width, random), PermB = Permutation(width, random),
            Residual = null, Dim = dim, Rows = rows, Cols = cols, Scale = scale,
        };
        return batch;
    }

    static double[] Product(double[] z, float* left, float* right, int rows, int cols)
    {
        var middle = new double[rows * cols];
        var result = new double[rows * cols];
        for (var k = 0; k < rows; k++)
            for (var c = 0; c < cols; c++)
                for (var i = 0; i < rows; i++) middle[k * cols + c] += (double)left[i * rows + k] * z[i * cols + c];
        for (var k = 0; k < rows; k++)
            for (var c = 0; c < cols; c++)
                for (var j = 0; j < cols; j++) result[k * cols + c] += middle[k * cols + j] * right[j * cols + c];
        return result;
    }

    static double[] Reference(MlpBatch* batch, float* residual)
    {
        var w = batch->Weights;
        var dim = batch->Dim;
        var width = batch->Width;
        double energy = 0;
        for (var i = 0; i < dim; i++) energy += (double)residual[i] * residual[i];
        var inverse = 1 / Math.Sqrt(energy / dim + 1e-6);
        var normalized = new double[dim];
        for (var i = 0; i < dim; i++) normalized[i] = (1 + (double)batch->PreNorm[i]) * residual[i] * inverse;
        var logits = new double[8];
        for (var j = 0; j < 8; j++)
            for (var i = 0; i < dim; i++) logits[j] += normalized[i] * w->CondV[i * 8 + j];
        var peak = logits.Max();
        var weights = logits.Select(l => Math.Exp(l - peak)).ToArray();
        var total = weights.Sum();
        for (var j = 0; j < 8; j++) weights[j] /= total;

        var first = new double[width];
        for (var i = 0; i < dim; i++) first[i] = normalized[i] * w->D1[i];
        var firstProduct = Product(first, w->W1A, w->W1B, batch->Rows, batch->Cols);
        var second = new double[width];
        for (var n = 0; n < width; n++)
        {
            var condition = 1.0;
            for (var j = 0; j < 8; j++) condition += weights[j] * w->CondU[j * width + n];
            var v = w->D2[n] * condition * firstProduct[batch->PermA[n]] + w->B2[n];
            second[n] = v / (1 + Math.Exp(-v));
        }
        var secondProduct = Product(second, w->W2A, w->W2B, batch->Rows, batch->Cols);
        var third = new double[width];
        for (var n = 0; n < width; n++) third[n] = secondProduct[batch->PermB[n]] * w->D3[n];
        var thirdProduct = Product(third, w->W3A, w->W3B, batch->Rows, batch->Cols);
        var expected = new double[dim];
        for (var i = 0; i < dim; i++) expected[i] = residual[i] + batch->Scale * thirdProduct[i] * w->D4[i];
        return expected;
    }

    static float[] Run(MlpBatch* batch, float[] initial, int tokens, int threads)
    {
        var dim = batch->Dim;
        var residual = (float*)Mem.AllocZeroed((nuint)(tokens * dim) * sizeof(float));
        try
        {
            for (var i = 0; i < tokens * dim; i++) residual[i] = initial[i];
            batch->Residual = residual;
            using var pool = new Pool(threads);
            MlpBatch.Run(pool, batch, tokens);
            var result = new float[tokens * dim];
            for (var i = 0; i < result.Length; i++) result[i] = residual[i];
            return result;
        }
        finally { Mem.Free(residual); }
    }

    static float[] Initial(int count, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
    }

    [Theory]
    [InlineData(768, 32, 32, 1, 1)]
    [InlineData(768, 32, 32, 4, 1)]
    [InlineData(768, 32, 32, 4, 2)]
    [InlineData(768, 32, 32, 4, 7)]
    [InlineData(96, 8, 16, 3, 1)]
    [InlineData(48, 8, 8, 4, 1)]
    [InlineData(32, 8, 8, 1, 3)]
    public void MatchesDoublePrecisionReference(int dim, int rows, int cols, int threads, int tokens)
    {
        var batch = Batch(dim, rows, cols, 0.5f, 11);
        var initial = Initial(tokens * dim, 5);
        var actual = Run(batch, initial, tokens, threads);
        var scratch = (float*)Mem.AllocZeroed((nuint)(tokens * dim) * sizeof(float));
        try
        {
            for (var t = 0; t < tokens; t++)
            {
                for (var i = 0; i < dim; i++) scratch[t * dim + i] = initial[t * dim + i];
                var expected = Reference(batch, scratch + t * dim);
                for (var i = 0; i < dim; i++) Assert.InRange(actual[t * dim + i], expected[i] - 2e-4 * (1 + Math.Abs(expected[i])), expected[i] + 2e-4 * (1 + Math.Abs(expected[i])));
            }
        }
        finally { Mem.Free(scratch); }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    public void ResultIsIdenticalForEveryThreadCount(int tokens)
    {
        var batch = Batch(768, 32, 32, 1f, 21);
        var initial = Initial(tokens * 768, 9);
        var serial = Run(batch, initial, tokens, 1);
        foreach (var threads in new[] { 2, 3, 4, 8 })
            Assert.Equal(serial, Run(batch, initial, tokens, threads));
    }

    [Theory]
    [InlineData(32, 32, 24, 32)]
    [InlineData(32, 32, 32, 24)]
    [InlineData(16, 32, 16, 16)]
    [InlineData(8, 16, 8, 4)]
    public void KroneckerProductMatchesScalarReference(int rows, int cols, int activeRows, int resultRows)
    {
        var random = new Random(rows * 100 + cols);
        var width = rows * cols;
        var z = Allocate(width, random, 1f);
        var left = Allocate(rows * rows, random, 1f);
        var right = Allocate(cols * cols, random, 1f);
        for (var i = activeRows * cols; i < width; i++) z[i] = 0f;
        var expected = (float*)Mem.AllocZeroed((nuint)width * sizeof(float));
        var actual = (float*)Mem.AllocZeroed((nuint)width * sizeof(float));
        allocations.Add((nint)expected);
        allocations.Add((nint)actual);
        Scalar.Kron(z, activeRows, left, right, rows, cols, resultRows, expected);
        HadamardMlp.Kron(z, activeRows, left, right, rows, cols, resultRows, actual);
        for (var i = 0; i < resultRows * cols; i++) Assert.InRange(actual[i], expected[i] - 1e-4 * (1 + Math.Abs(expected[i])), expected[i] + 1e-4 * (1 + Math.Abs(expected[i])));
    }

    [Fact]
    public void SkippingZeroInputRowsDoesNotChangeTheProduct()
    {
        var random = new Random(3);
        var z = Allocate(1024, random, 1f);
        var left = Allocate(1024, random, 1f);
        var right = Allocate(1024, random, 1f);
        for (var i = 24 * 32; i < 1024; i++) z[i] = 0f;
        var skipped = (float*)Mem.AllocZeroed(1024 * sizeof(float));
        var full = (float*)Mem.AllocZeroed(1024 * sizeof(float));
        allocations.Add((nint)skipped);
        allocations.Add((nint)full);
        HadamardMlp.Kron(z, 24, left, right, 32, 32, 32, skipped);
        HadamardMlp.Kron(z, 32, left, right, 32, 32, 32, full);
        for (var i = 0; i < 1024; i++) Assert.Equal(full[i], skipped[i]);
    }

#if NEEDLE_SIMD
    [Fact]
    public void WideKroneckerProductMatchesAvx2BitForBit()
    {
        if (!HadamardMlp.Wide) return;
        var random = new Random(8);
        var z = Allocate(1024, random, 1f);
        var left = Allocate(1024, random, 1f);
        var right = Allocate(1024, random, 1f);
        var narrow = (float*)Mem.AllocZeroed(1024 * sizeof(float));
        var wide = (float*)Mem.AllocZeroed(1024 * sizeof(float));
        allocations.Add((nint)narrow);
        allocations.Add((nint)wide);
        HadamardMlp.KronAvx2(z, 32, left, right, 32, 32, 32, narrow);
        HadamardMlp.KronAvx512(z, 32, left, right, 32, 32, 32, wide);
        for (var i = 0; i < 1024; i++) Assert.Equal(narrow[i], wide[i]);
    }
#endif
}
