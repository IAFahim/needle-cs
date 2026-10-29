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
