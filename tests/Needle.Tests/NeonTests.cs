using System;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using Xunit;

namespace Needle.Tests;

public sealed unsafe class NeonTests
{
    const float Tolerance = 3e-5f;

    static void Fill(float* x, int n, int seed, float amplitude = 1f)
    {
        var random = new Random(seed);
        for (var i = 0; i < n; i++) x[i] = amplitude * ((float)random.NextDouble() * 2 - 1);
    }

    static void AssertClose(float* actual, float* expected, int n, string what)
    {
        for (var i = 0; i < n; i++)
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= Tolerance * MathF.Max(1f, MathF.Abs(expected[i])),
                $"{what}[{i}]: {actual[i]} vs {expected[i]}");
    }

    [Fact]
    public void ExpStepsMatchHand()
    {
        if (!Simd.UseNeon) return;
        var x = System.Runtime.Intrinsics.Vector128.Create(-20f, 0f, 1f, -1f);
        var n = Neon.RoundToNearestOf(x * 1.44269504f);
        var nExpected = new[] { -29f, 0f, 1f, -1f };
        for (var i = 0; i < 4; i++) Assert.Equal(nExpected[i], n[i], 4);
        var r = Neon.FusedMultiplySubtractOf(x, n, 0.693359375f);
        var r1Expected = new[] { 0.10742188f, 0f, 0.30664062f, -0.30664062f };
        for (var i = 0; i < 4; i++) Assert.Equal(r1Expected[i], r[i], 5);
        r = Neon.FusedMultiplySubtractOf(r, n, -2.12194440e-4f);
        var r2Expected = new[] { 0.10126838f, 0f, 0.30685281f, -0.30685281f };
        for (var i = 0; i < 4; i++) Assert.Equal(r2Expected[i], r[i], 5);
        var exponent = System.Runtime.Intrinsics.Arm.AdvSimd.ShiftLeftLogical(
            System.Runtime.Intrinsics.Arm.AdvSimd.Add(
                System.Runtime.Intrinsics.Arm.AdvSimd.ConvertToInt32RoundToEven(n), System.Runtime.Intrinsics.Vector128.Create(127)).AsUInt32(), 23);
        var scaleExpected = new[] { 1.8626451e-9f, 1f, 2f, 0.5f };
        var scale = exponent.AsSingle();
        for (var i = 0; i < 4; i++) Assert.Equal(scaleExpected[i], scale[i], 7);
    }

    [Fact]
    public void ExpMatchesMathF()
    {
        if (!Simd.UseNeon) return;
        var input = stackalloc float[4];
        var values = new[] { -20f, -1.5f, 0.3f, 12f };
        for (var i = 0; i < 4; i++) input[i] = values[i];
        var result = Neon.Exp(System.Runtime.Intrinsics.Vector128.Create(input[0], input[1], input[2], input[3]));
        for (var i = 0; i < 4; i++)
            Assert.True(MathF.Abs(result[i] - MathF.Exp(values[i])) <= 2e-6 * MathF.Exp(values[i]),
                $"Exp({values[i]}): {result[i]} vs {MathF.Exp(values[i])}");
    }

    [Fact]
    public void DecodeRowTwoBitPattern()
    {
        if (!Simd.UseNeon) return;
        var book = (Codebook*)Marshal.AllocHGlobal(sizeof(Codebook));
        for (var i = 0; i < 16; i++) book->Levels[i] = 0f;
        book->Levels[0] = 1f; book->Levels[1] = 2f; book->Levels[2] = 4f; book->Levels[3] = 8f;
        book->ByteScale = 1f;
        for (var i = 0; i < 32; i++) book->Bytes[i] = 0;
        var indices = (byte*)Marshal.AllocHGlobal(32);
        for (var i = 0; i < 32; i++) indices[i] = 0;
        indices[0] = 0b11_10_01_00;
        indices[1] = 0b00_00_00_01;
        var norms = (ushort*)Marshal.AllocHGlobal(2);
        norms[0] = 0x3C00;
        var w = new Cq(indices, norms, book, 1, 128, 2);
        var actual = stackalloc float[128];
        var expected = stackalloc float[128];
        Neon.DecodeRow(w, 0, actual);
        Scalar.DecodeRow(w, 0, expected);
        var want = new[] { 1f, 2f, 4f, 8f, 2f, 1f, 1f, 1f };
        for (var i = 0; i < 8; i++) Assert.Equal(want[i], actual[i], 4);
        Marshal.FreeHGlobal((nint)indices);
        Marshal.FreeHGlobal((nint)norms);
        Marshal.FreeHGlobal((nint)book);
    }

    [Fact]
    public void DecodeRowFourBitPattern()
    {
        if (!Simd.UseNeon) return;
        var book = (Codebook*)Marshal.AllocHGlobal(sizeof(Codebook));
        for (var i = 0; i < 16; i++) book->Levels[i] = i + 1f;
        book->ByteScale = 1f;
        for (var i = 0; i < 32; i++) book->Bytes[i] = 0;
        var indices = (byte*)Marshal.AllocHGlobal(64);
        for (var i = 0; i < 64; i++) indices[i] = 0;
        indices[0] = 0x21;
        indices[1] = 0xF0;
        var norms = (ushort*)Marshal.AllocHGlobal(2);
        norms[0] = 0x3C00;
        var w = new Cq(indices, norms, book, 1, 128, 4);
        var actual = stackalloc float[128];
        var expected = stackalloc float[128];
        Neon.DecodeRow(w, 0, actual);
        Scalar.DecodeRow(w, 0, expected);
        Assert.Equal(2f, actual[0], 4);
        Assert.Equal(3f, actual[1], 4);
        Marshal.FreeHGlobal((nint)indices);
        Marshal.FreeHGlobal((nint)norms);
        Marshal.FreeHGlobal((nint)book);
    }

    [Fact]
    public void HadamardMatchesScalar()
    {
        if (!Simd.UseNeon) return;
        var x = stackalloc float[128];
        var source = stackalloc float[128];
        var expected = stackalloc float[128];
        Fill(source, 128, 1);
        Neon.Hadamard(source, x);
        Scalar.Hadamard(source, expected);
        AssertClose(x, expected, 128, "Hadamard");
        Fill(x, 128, 2);
        for (var i = 0; i < 128; i++) expected[i] = x[i];
        Neon.Hadamard(x, x);
        Scalar.Hadamard(expected, expected);
        AssertClose(x, expected, 128, "Hadamard in-place");
    }

    [Fact]
    public void QuantizeInt8SnapsToTheInt8Grid()
    {
        if (!Simd.UseNeon) return;
        var x = stackalloc float[128];
        var actual = stackalloc sbyte[128];
        Fill(x, 128, 3);
        var scale = Neon.QuantizeInt8(x, 128, actual);
        Assert.Equal(Neon.AbsMax(x, 128) / 127f, scale, 5);
        for (var i = 0; i < 128; i++) Assert.Equal((int)(x[i] / scale), actual[i]);
    }

    [Fact]
    public void RoundTripMatchesScalar()
    {
        if (!Simd.UseNeon) return;
        var x = stackalloc float[128];
        var reference = stackalloc float[128];
        Fill(x, 128, 4);
        for (var i = 0; i < 128; i++) reference[i] = x[i];
        Neon.RoundTrip(x, 128, 3.5f, 0.25f);
        Scalar.RoundTrip(reference, 128, 3.5f, 0.25f);
        AssertClose(x, reference, 128, "RoundTrip");
    }

    static Cq MakeWeights(int rows, int cols, int bits, int seed)
    {
        var random = new Random(seed);
        var groups = cols / Cq.GroupSize;
        var rowBytes = cols * bits / 8;
        var book = (Codebook*)Marshal.AllocHGlobal(sizeof(Codebook));
        var peak = 0f;
        for (var i = 0; i < (bits == 2 ? 4 : 16); i++)
        {
            var level = (float)(random.NextDouble() * 2 - 1);
            book->Levels[i] = level;
            peak = MathF.Max(peak, MathF.Abs(level));
        }
        for (var i = (bits == 2 ? 4 : 16); i < 16; i++) book->Levels[i] = 0f;
        book->ByteScale = peak / Codebook.Bias;
        for (var i = 0; i < (bits == 2 ? 4 : 16); i++)
            book->Bytes[i] = (byte)Math.Clamp(MathF.Round(MathF.Abs(book->Levels[i]) / book->ByteScale), 0, 127);
        var indices = (byte*)Marshal.AllocHGlobal(rows * rowBytes);
        for (var i = 0; i < rows * rowBytes; i++) indices[i] = (byte)random.Next(256);
        var norms = (ushort*)Marshal.AllocHGlobal(rows * groups * 2);
        for (var i = 0; i < rows * groups; i++)
            norms[i] = FloatToHalfBits(0.01f + (float)random.NextDouble());
        return new Cq(indices, norms, book, rows, cols, bits);
    }

    static ushort FloatToHalfBits(float value)
    {
        var bits = *(int*)&value;
        return (ushort)(((bits >> 16) & 0x8000) | ((bits >> 13) & 0x7FFF));
    }

    static void FreeWeights(Cq w)
    {
        Marshal.FreeHGlobal((nint)w.Indices);
        Marshal.FreeHGlobal((nint)w.Norms);
        Marshal.FreeHGlobal((nint)w.Book);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void DecodeRowMatchesScalar(int bits)
    {
        if (!Simd.UseNeon) return;
        var w = MakeWeights(4, 768, bits, 10 + bits);
        var expected = stackalloc float[768];
        var actual = stackalloc float[768];
        for (var row = 0; row < w.Rows; row++)
        {
            Scalar.DecodeRow(w, row, expected);
            Neon.DecodeRow(w, row, actual);
            AssertClose(actual, expected, 768, $"DecodeRow{bits} row {row}");
        }
        FreeWeights(w);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void MatMulMatchesScalar(int bits)
    {
        if (!Simd.UseNeon) return;
        var w = MakeWeights(24, 768, bits, 20 + bits);
        var tokens = 5;
        var x = stackalloc float[tokens * 768];
        Fill(x, tokens * 768, 30 + bits);
        var values = stackalloc sbyte[tokens * 768];
        var scales = stackalloc float[tokens * 8];
        var offsets = stackalloc float[tokens * 8];
        for (var i = 0; i < tokens * 8; i++) { scales[i] = 0f; offsets[i] = 0f; }
        var activations = new Activations(values, scales, offsets, 768);
        Quant.Prepare(x, tokens, 768, bits, activations);
        var expected = stackalloc float[tokens * w.Rows];
        var actual = stackalloc float[tokens * w.Rows];
        Scalar.MatMulRows(w, activations, tokens, expected, w.Rows, 0, w.Rows);
        Neon.MatMulRows(w, activations, tokens, actual, w.Rows, 0, w.Rows);
        AssertClose(actual, expected, tokens * w.Rows, $"MatMul{bits}");
        FreeWeights(w);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void PrepareMatchesScalar(int bits)
    {
        if (!Simd.UseNeon) return;
        var x = stackalloc float[768];
        Fill(x, 768, 40 + bits);
        var valuesExpected = stackalloc sbyte[768];
        var valuesActual = stackalloc sbyte[768];
        var scalesExpected = stackalloc float[8];
        var scalesActual = stackalloc float[8];
        var offsetsExpected = stackalloc float[8];
        var offsetsActual = stackalloc float[8];
        var rotated = stackalloc float[128];
        var quantized = stackalloc sbyte[128];
        for (var g = 0; g < 6; g++)
        {
            Scalar.Hadamard(x + g * 128, rotated);
            var peak = 0f;
            for (var i = 0; i < 128; i++) peak = MathF.Max(peak, MathF.Abs(rotated[i]));
            scalesExpected[g] = peak > 0f ? peak / 127f : 1f;
            var inverse = 1f / scalesExpected[g];
            for (var i = 0; i < 128; i++) quantized[i] = (sbyte)Math.Clamp((int)(rotated[i] * inverse), -128, 127);
            Scalar.Planes(quantized, bits, valuesExpected + g * 128, out var sum);
            offsetsExpected[g] = Codebook.Bias * sum;
        }
        var actual = new Activations(valuesActual, scalesActual, offsetsActual, 768);
        Neon.Prepare(x, 1, 768, bits, actual);
        for (var i = 0; i < 768; i++)
            Assert.True(MathF.Abs(valuesActual[i] - valuesExpected[i]) <= 1,
                $"prepare{bits}[{i}]: neon={valuesActual[i]} scalar={valuesExpected[i]}");
        for (var i = 0; i < 6; i++)
        {
            Assert.True(MathF.Abs(scalesActual[i] - scalesExpected[i]) <= 1e-4f * scalesExpected[i],
                $"scales{bits}[{i}]: {scalesActual[i]} vs {scalesExpected[i]}");
            Assert.True(MathF.Abs(offsetsActual[i] - offsetsExpected[i]) <= 127f * 2f,
                $"offsets{bits}[{i}]: {offsetsActual[i]} vs {offsetsExpected[i]}");
        }
    }

    [Fact]
    public void FloatOpsMatchScalar()
    {
        if (!Simd.UseNeon) return;
        var l = stackalloc float[256];
        var r = stackalloc float[256];
        var expected = stackalloc float[256];
        var actual = stackalloc float[256];
        Fill(l, 256, 50, 4f);
        for (var i = 0; i < 256; i++) expected[i] = l[i];
        Neon.Silu(l, 256);
        Scalar.Silu(expected, 256);
        AssertClose(l, expected, 256, "Silu");

        Fill(r, 256, 51, 0.5f);
        for (var i = 0; i < 256; i++) expected[i] = r[i];
        Neon.Softmax(r, 256);
        Scalar.Softmax(expected, 256);
        AssertClose(r, expected, 256, "Softmax");

        Fill(l, 256, 52);
        Console.Error.WriteLine($"rms sum={Neon.SumOfSquares(l, 256)} scalarSum={Scalar.SumOfSquares(l, 256)} l0={l[0]}");
        Neon.RmsUnit(l, 256, actual);
        var inverse = 1f / MathF.Sqrt(Scalar.SumOfSquares(l, 256) / 256 + 1e-6f);
        for (var i = 0; i < 256; i++) expected[i] = l[i] * inverse;
        AssertClose(actual, expected, 256, "RmsUnit");

        Fill(l, 256, 53);
        Neon.ZeroCenteredRms(l, r, 256, actual);
        var centered = 1f / MathF.Sqrt(Scalar.SumOfSquares(l, 256) / 256 + 1e-6f);
        for (var i = 0; i < 256; i++) expected[i] = (1f + r[i]) * l[i] * centered;
        AssertClose(actual, expected, 256, "ZeroCenteredRms");

        Fill(l, 256, 55);
        Neon.Mix4(l, 64, 0.3f, -0.7f, 1.2f, 0.05f, actual);
        Scalar.Mix4(l, 64, 0.3f, -0.7f, 1.2f, 0.05f, expected);
        AssertClose(actual, expected, 64, "Mix4");

        Fill(l, 256, 56);
        Neon.Combine4(l, 64, 0.3f, -0.7f, 1.2f, 0.05f, r, 2f, actual);
        Scalar.Combine4(l, 64, 0.3f, -0.7f, 1.2f, 0.05f, r, 2f, expected);
        AssertClose(actual, expected, 64, "Combine4");

        Fill(l, 256, 57);
        var history = stackalloc float[192];
        Fill(history, 192, 57);
        var current = history + 128;
        Neon.Taps3(l, 64, current, actual);
        Scalar.Taps3(l, 64, current, expected);
        AssertClose(actual, expected, 64, "Taps3");

        Fill(l, 256, 57);
        Neon.Subtract(l, r, 64, actual);
        Scalar.Subtract(l, r, 64, expected);
        AssertClose(actual, expected, 64, "Subtract");

        Fill(l, 64, 58);
        Fill(r, 64, 59);
        for (var i = 0; i < 64; i++) { expected[i] = actual[i] = l[i]; }
        Neon.MultiplyAccumulate(actual, l, r, 64);
        Scalar.MultiplyAccumulate(expected, l, r, 64);
        AssertClose(actual, expected, 64, "MultiplyAccumulate");

        for (var i = 0; i < 64; i++) { expected[i] = actual[i] = l[i]; }
        Neon.ScaleAdd(actual, 2f, r, 0.5f, 64);
        Scalar.ScaleAdd(expected, 2f, r, 0.5f, 64);
        AssertClose(actual, expected, 64, "ScaleAdd");

        for (var i = 0; i < 64; i++) { expected[i] = actual[i] = l[i]; }
        Neon.AddScaled(actual, r, 1.5f, 64);
        Scalar.AddScaled(expected, r, 1.5f, 64);
        AssertClose(actual, expected, 64, "AddScaled");

        Neon.Glu(l, r, actual, 64);
        Scalar.Glu(l, r, expected, 64);
        AssertClose(actual, expected, 64, "Glu");
    }

    [Fact]
    public void RopeMatchesScalar()
    {
        if (!Simd.UseNeon) return;
        var x = stackalloc float[256];
        var cos = stackalloc float[32];
        var sin = stackalloc float[32];
        Fill(x, 256, 60, 4f);
        for (var i = 0; i < 32; i++) { cos[i] = MathF.Cos(0.13f * i); sin[i] = MathF.Sin(0.13f * i); }
        Neon.Rope(x, 4, 64, cos, sin);
        var reference = stackalloc float[256];
        Fill(reference, 256, 60, 4f);
        Scalar.Rope(reference, 4, 64, cos, sin);
        AssertClose(x, reference, 256, "Rope");
    }

    [Fact]
    public void AttentionHeadMatchesScalar()
    {
        if (!Simd.UseNeon) return;
        var random = new Random(70);
        var queries = stackalloc sbyte[48];
        var keys = stackalloc sbyte[8 * 48];
        var values = stackalloc sbyte[8 * 64];
        var keyScales = stackalloc float[8];
        var valueScales = stackalloc float[8];
        for (var i = 0; i < 48; i++) queries[i] = (sbyte)random.Next(-120, 120);
        for (var i = 0; i < 8 * 48; i++) keys[i] = (sbyte)random.Next(-120, 120);
        for (var i = 0; i < 8 * 64; i++) values[i] = (sbyte)random.Next(-120, 120);
        for (var i = 0; i < 8; i++) { keyScales[i] = 0.002f * (1 + i); valueScales[i] = 0.003f * (1 + i); }
        var expectedScores = stackalloc float[16];
        var actualScores = stackalloc float[16];
        var expectedOut = stackalloc float[64];
        var actualOut = stackalloc float[64];
        Scalar.AttentionHead(queries, 0.5f, keys, keyScales, values, valueScales, 48, 64, 1, 0, 7, expectedScores, expectedOut, null);
        Neon.AttentionHead(queries, 0.5f, keys, keyScales, values, valueScales, 48, 64, 1, 0, 7, actualScores, actualOut, null);
        AssertClose(actualScores, expectedScores, 8, "AttentionScores");
        AssertClose(actualOut, expectedOut, 64, "AttentionHead");
    }

    [Fact]
    public void MlpBlocksMatchScalar()
    {
        if (!Simd.UseNeon) return;
        var random = new Random(80);
        var dim = 128;
        var width = 32 * 32;
        var left = stackalloc float[width];
        var right = stackalloc float[width];
        var condV = stackalloc float[dim * 8];
        var condU = stackalloc float[8 * width];
        var d2 = stackalloc float[width];
        var b2 = stackalloc float[width];
        for (var i = 0; i < width; i++) { d2[i] = (float)random.NextDouble(); b2[i] = (float)(random.NextDouble() * 0.1); }
        for (var i = 0; i < width; i++) { left[i] = (float)(random.NextDouble() * 2 - 1); right[i] = (float)(random.NextDouble() * 2 - 1); }
        for (var i = 0; i < dim * 8; i++) condV[i] = (float)(random.NextDouble() * 2 - 1);
        for (var i = 0; i < 8 * width; i++) condU[i] = (float)(random.NextDouble() * 2 - 1);
        var input = stackalloc float[dim];
        for (var i = 0; i < dim; i++) input[i] = (float)(random.NextDouble() * 2 - 1);

        var permutation = stackalloc int[width];
        for (var i = 0; i < width; i++) permutation[i] = (i * 389 + 17) % width;

        var expectedWeights = stackalloc float[8];
        var actualWeights = stackalloc float[8];
        Scalar.ConditionWeights(condV, input, dim, expectedWeights);
        Neon.ConditionWeights(condV, input, dim, actualWeights);
        AssertClose(actualWeights, expectedWeights, 8, "ConditionWeights");

        var expectedInput = stackalloc float[width];
        var actualInput = stackalloc float[width];
        Scalar.ScaleInput(input, d2, dim, 160, expectedInput);
        Neon.ScaleInput(input, d2, dim, 160, actualInput);
        AssertClose(actualInput, expectedInput, 160, "ScaleInput");

        var source = stackalloc float[width];
        for (var i = 0; i < width; i++) source[i] = (float)(random.NextDouble() * 2 - 1);
        var expectedA = stackalloc float[width];
        var actualA = stackalloc float[width];
        Scalar.PermuteModulateActivate(permutation, source, d2, expectedWeights, condU, b2, width, expectedA);
        Neon.PermuteModulateActivate(permutation, source, d2, expectedWeights, condU, b2, width, actualA);
        AssertClose(actualA, expectedA, width, "PermuteModulateActivate");

        Scalar.PermuteScale(permutation, source, d2, width, expectedA);
        Neon.PermuteScale(permutation, source, d2, width, actualA);
        AssertClose(actualA, expectedA, width, "PermuteScale");

        var expectedOut = stackalloc float[width];
        var actualOut = stackalloc float[width];
        Scalar.Kron(expectedA, 24, left, right, 32, 32, 32, expectedOut);
        Neon.Kron(actualA, 24, left, right, 32, 32, 32, actualOut);
        AssertClose(actualOut, expectedOut, width, "Kron");

        var expectedResidual = stackalloc float[dim];
        var actualResidual = stackalloc float[dim];
        for (var i = 0; i < dim; i++) expectedResidual[i] = actualResidual[i] = input[i];
        Scalar.AccumulateOutput(expectedResidual, expectedOut, d2, 0.5f, dim);
        Neon.AccumulateOutput(actualResidual, actualOut, d2, 0.5f, dim);
        AssertClose(actualResidual, expectedResidual, dim, "AccumulateOutput");
    }
}
