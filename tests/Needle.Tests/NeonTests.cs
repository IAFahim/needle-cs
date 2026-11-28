using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
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

    static void AssertMatMulClose(float* actual, float* expected, int n, string what)
    {
        var magnitude = 1f;
        for (var i = 0; i < n; i++) magnitude = MathF.Max(magnitude, MathF.Abs(expected[i]));
        for (var i = 0; i < n; i++)
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= 1e-3f * magnitude,
                $"{what}[{i}]: {actual[i]} vs {expected[i]} (mag {magnitude})");
    }

    public static IEnumerable<object[]> MultiRowShapes()
    {
        foreach (var bits in new[] { 2, 4 })
            foreach (var (rows, cols) in new[] { (1, 128), (2, 128), (3, 256), (4, 256), (5, 768), (6, 384), (7, 768), (8, 768), (9, 768), (13, 768), (24, 768), (37, 768) })
                foreach (var tokens in new[] { 1, 2, 3, 4, 5, 7, 8, 9 })
                    yield return new object[] { bits, rows, cols, tokens };
    }

    [Theory]
    [MemberData(nameof(MultiRowShapes))]
    public void MultiRowMatMulMatchesScalar(int bits, int rows, int cols, int tokens)
    {
        if (!Simd.UseNeon) return;
        var w = MakeWeights(rows, cols, bits, rows * 131 + cols + bits * 17 + tokens);
        var groups = cols / Cq.GroupSize;
        var stride = (groups + 7) & ~7;
        var x = stackalloc float[tokens * cols];
        Fill(x, tokens * cols, rows * 1000 + tokens * 37 + bits);
        var values = stackalloc sbyte[tokens * cols];
        var scales = stackalloc float[tokens * stride];
        var offsets = stackalloc float[tokens * stride];
        var activations = new Activations(values, scales, offsets, cols);
        Quant.Prepare(x, tokens, cols, bits, activations);
        var expected = stackalloc float[tokens * rows];
        var actual = stackalloc float[tokens * rows];
        Scalar.MatMulRows(w, activations, tokens, expected, rows, 0, rows);
        Neon.MatMulRows(w, activations, tokens, actual, rows, 0, rows);
        AssertMatMulClose(actual, expected, tokens * rows, $"MultiRow{bits} {rows}x{cols}x{tokens}");
        FreeWeights(w);
    }

    [Theory]
    [InlineData(2, 9, 768, 1, 0, 9)]
    [InlineData(2, 9, 768, 1, 2, 6)]
    [InlineData(2, 9, 768, 4, 1, 8)]
    [InlineData(2, 9, 768, 5, 3, 9)]
    [InlineData(4, 9, 768, 1, 0, 9)]
    [InlineData(4, 9, 768, 7, 2, 7)]
    [InlineData(4, 13, 384, 3, 5, 13)]
    [InlineData(2, 24, 768, 8, 7, 18)]
    [InlineData(4, 5, 768, 6, 1, 4)]
    public void MultiRowSlicesMatchScalar(int bits, int rows, int cols, int tokens, int rowStart, int rowEnd)
    {
        if (!Simd.UseNeon) return;
        var w = MakeWeights(rows, cols, bits, rows * 57 + cols + bits + tokens);
        var groups = cols / Cq.GroupSize;
        var stride = (groups + 7) & ~7;
        var x = stackalloc float[tokens * cols];
        Fill(x, tokens * cols, rows * 2000 + tokens * 91 + bits);
        var values = stackalloc sbyte[tokens * cols];
        var scales = stackalloc float[tokens * stride];
        var offsets = stackalloc float[tokens * stride];
        var activations = new Activations(values, scales, offsets, cols);
        Quant.Prepare(x, tokens, cols, bits, activations);
        var expected = stackalloc float[tokens * rows];
        var actual = stackalloc float[tokens * rows];
        for (var i = 0; i < tokens * rows; i++) expected[i] = actual[i] = 12345f;
        Scalar.MatMulRows(w, activations, tokens, expected, rows, rowStart, rowEnd);
        Neon.MatMulRows(w, activations, tokens, actual, rows, rowStart, rowEnd);
        for (var t = 0; t < tokens; t++)
        {
            for (var n = 0; n < rows; n++)
                if (n < rowStart || n >= rowEnd) Assert.Equal(12345f, actual[t * rows + n]);
            AssertMatMulClose(actual + t * rows + rowStart, expected + t * rows + rowStart, rowEnd - rowStart, $"MultiRowSlice{bits} {rows}x{cols}x{tokens}[{rowStart},{rowEnd}) t{t}");
        }
        FreeWeights(w);
    }

    static void AssertBits(float* actual, float* expected, int n, string what)
    {
        for (var i = 0; i < n; i++)
            Assert.True(BitConverter.SingleToInt32Bits(actual[i]) == BitConverter.SingleToInt32Bits(expected[i]),
                $"{what}[{i}]: {actual[i]} vs {expected[i]}");
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 7)]
    [InlineData(2, 9)]
    [InlineData(3, 37)]
    [InlineData(4, 16)]
    [InlineData(5, 30)]
    [InlineData(6, 131)]
    [InlineData(7, 22)]
    [InlineData(12, 45)]
    public void AttentionTileIsBitExactToPerHeadTile(int heads, int count)
    {
        if (!Simd.UseNeon) return;
        const int kvHeads = 2;
        const int first = 3;
        var positions = first + count;
        var random = new Random(heads * 1000 + count);
        var queries = (sbyte*)NativeMemory.Alloc((nuint)(heads * 48));
        var queryScales = (float*)NativeMemory.Alloc((nuint)(heads * 4));
        var keys = (sbyte*)NativeMemory.Alloc((nuint)(positions * kvHeads * 48));
        var values = (sbyte*)NativeMemory.Alloc((nuint)(positions * kvHeads * 64));
        var keyScales = (float*)NativeMemory.Alloc((nuint)(positions * kvHeads * 4));
        var valueScales = (float*)NativeMemory.Alloc((nuint)(positions * kvHeads * 4));
        var scoreFloats = heads * count + 64;
        var expectedScores = (float*)NativeMemory.AllocZeroed((nuint)(scoreFloats * 4));
        var actualScores = (float*)NativeMemory.AllocZeroed((nuint)(scoreFloats * 4));
        var expectedOut = (float*)NativeMemory.AllocZeroed((nuint)(heads * 64 * 4));
        var actualOut = (float*)NativeMemory.AllocZeroed((nuint)(heads * 64 * 4));
        var expectedProbabilities = (float*)NativeMemory.AllocZeroed((nuint)(heads * positions * 4));
        var actualProbabilities = (float*)NativeMemory.AllocZeroed((nuint)(heads * positions * 4));
        try
        {
            for (var i = 0; i < heads * 48; i++) queries[i] = (sbyte)random.Next(-127, 128);
            for (var h = 0; h < heads; h++) queryScales[h] = 0.01f + 0.05f * (float)random.NextDouble();
            for (var i = 0; i < positions * kvHeads * 48; i++) keys[i] = (sbyte)random.Next(-127, 128);
            for (var i = 0; i < positions * kvHeads * 64; i++) values[i] = (sbyte)random.Next(-127, 128);
            for (var i = 0; i < positions * kvHeads; i++)
            {
                keyScales[i] = 0.001f + 0.01f * (float)random.NextDouble();
                valueScales[i] = 0.001f + 0.01f * (float)random.NextDouble();
            }
            AttentionTile Make(float* scores, float* output, float* probabilities) => new()
            {
                Queries = queries, QueryScales = queryScales, Heads = heads,
                Keys = keys + 48, KeyScales = keyScales + 1,
                Values = values + 64, ValueScales = valueScales + 1,
                KeyStride = kvHeads * 48, ValueStride = kvHeads * 64, ScaleStride = kvHeads,
                First = first, Count = count, Scores = scores, Output = output,
                Probabilities = probabilities, ProbabilityStride = positions,
            };
            Neon.AttentionTilePerHead(Make(expectedScores, expectedOut, expectedProbabilities));
            NeonAttention.Tile(Make(actualScores, actualOut, actualProbabilities));
            AssertBits(actualProbabilities, expectedProbabilities, heads * positions, "Probabilities");
            AssertBits(actualOut, expectedOut, heads * 64, "AttentionTile");
        }
        finally
        {
            NativeMemory.Free(queries); NativeMemory.Free(queryScales); NativeMemory.Free(keys); NativeMemory.Free(values);
            NativeMemory.Free(keyScales); NativeMemory.Free(valueScales); NativeMemory.Free(expectedScores); NativeMemory.Free(actualScores);
            NativeMemory.Free(expectedOut); NativeMemory.Free(actualOut); NativeMemory.Free(expectedProbabilities); NativeMemory.Free(actualProbabilities);
        }
    }

    // Neon.Kron is one fused multiply-add per term, accumulated in index order from zero; this is that
    // order written with MathF.FusedMultiplyAdd, so the comparison is bit for bit.
    [Theory]
    [InlineData(24, 32, 32, 32)]
    [InlineData(32, 32, 32, 16)]
    [InlineData(5, 8, 16, 8)]
    public void KronIsBitExactToFusedOrder(int activeRows, int rows, int cols, int resultRows)
    {
        if (!Simd.UseNeon) return;
        var random = new Random(activeRows * 7 + cols);
        var z = (float*)NativeMemory.Alloc((nuint)(rows * cols * 4));
        var left = (float*)NativeMemory.Alloc((nuint)(rows * rows * 4));
        var right = (float*)NativeMemory.Alloc((nuint)(cols * cols * 4));
        var expected = (float*)NativeMemory.Alloc((nuint)(rows * cols * 4));
        var actual = (float*)NativeMemory.Alloc((nuint)(rows * cols * 4));
        var tile = new float[cols];
        try
        {
            for (var i = 0; i < rows * cols; i++) z[i] = (float)(random.NextDouble() * 2 - 1);
            for (var i = 0; i < rows * rows; i++) left[i] = (float)(random.NextDouble() * 2 - 1);
            for (var i = 0; i < cols * cols; i++) right[i] = (float)(random.NextDouble() * 2 - 1);
            for (var k = 0; k < resultRows; k++)
            {
                for (var c = 0; c < cols; c++)
                {
                    var sum = 0f;
                    for (var i = 0; i < activeRows; i++) sum = MathF.FusedMultiplyAdd(left[i * rows + k], z[i * cols + c], sum);
                    tile[c] = sum;
                }
                for (var c = 0; c < cols; c++)
                {
                    var sum = 0f;
                    for (var j = 0; j < cols; j++) sum = MathF.FusedMultiplyAdd(tile[j], right[j * cols + c], sum);
                    expected[k * cols + c] = sum;
                }
            }
            Neon.Kron(z, activeRows, left, right, rows, cols, resultRows, actual);
            AssertBits(actual, expected, resultRows * cols, "Kron");
        }
        finally
        {
            NativeMemory.Free(z); NativeMemory.Free(left); NativeMemory.Free(right); NativeMemory.Free(expected); NativeMemory.Free(actual);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(13)]
    [InlineData(64)]
    [InlineData(8193)]
    public void PeakMatchesScalarMax(int n)
    {
        if (!Simd.UseNeon) return;
        var x = (float*)NativeMemory.Alloc((nuint)(n * 4));
        try
        {
            Fill(x, n, n, 30f);
            for (var round = 0; round < 3; round++)
            {
                if (round == 1) for (var i = 0; i < n; i++) x[i] = i % 2 == 0 ? -0f : -1f;
                if (round == 2) x[n - 1] = 1e30f;
                var expected = float.NegativeInfinity;
                for (var i = 0; i < n; i++) expected = MathF.Max(expected, x[i]);
                Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(Neon.Peak(x, n)));
            }
        }
        finally
        {
            NativeMemory.Free(x);
        }
    }

    // Neon.Prepare against Neon.Hadamard (the original vector rotation) and the original truncating
    // quantiser, bit for bit, including tokens after the first.
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void PrepareIsBitExactToHadamardAndTruncation(int bits)
    {
        if (!Simd.UseNeon) return;
        const int cols = 768;
        const int tokens = 3;
        const int groups = cols / 128;
        var x = (float*)NativeMemory.Alloc(cols * tokens * 4);
        var values = (sbyte*)NativeMemory.Alloc(cols * tokens);
        var scales = (float*)NativeMemory.Alloc(8 * tokens * 4);
        var offsets = (float*)NativeMemory.Alloc(8 * tokens * 4);
        var rotated = stackalloc float[128];
        var quantized = stackalloc sbyte[128];
        var planes = stackalloc sbyte[128];
        try
        {
            Fill(x, cols * tokens, 90 + bits, 3f);
            x[5] = 0f;
            var activations = new Activations(values, scales, offsets, cols);
            Neon.Prepare(x, tokens, cols, bits, activations);
            for (var t = 0; t < tokens; t++)
            {
                var slice = activations.Slice(t);
                for (var g = 0; g < groups; g++)
                {
                    Neon.Hadamard(x + t * cols + g * 128, rotated);
                    var peak = 0f;
                    for (var i = 0; i < 128; i++) peak = MathF.Max(peak, MathF.Abs(rotated[i]));
                    var scale = peak > 0f ? peak / 127f : 1f;
                    var inverse = 1f / scale;
                    for (var i = 0; i < 128; i++) quantized[i] = (sbyte)Math.Clamp((int)(rotated[i] * inverse), -128, 127);
                    Scalar.Planes(quantized, bits, planes, out var sum);
                    Assert.Equal(BitConverter.SingleToInt32Bits(scale), BitConverter.SingleToInt32Bits(slice.Scales[g]));
                    Assert.Equal(Codebook.Bias * sum, slice.Offsets[g]);
                    for (var i = 0; i < 128; i++)
                        Assert.True(planes[i] == slice.Values[g * 128 + i], $"prepare{bits} t{t} g{g}[{i}]: {slice.Values[g * 128 + i]} vs {planes[i]}");
                }
            }
        }
        finally
        {
            NativeMemory.Free(x); NativeMemory.Free(values); NativeMemory.Free(scales); NativeMemory.Free(offsets);
        }
    }

    // Sharding the hyper prepare by groups (Ops.RmsStats, then PrepareFakeGroupsRange per shard)
    // must reproduce PrepareRmsFakeA8 bit for bit wherever Quant.CanSplitRmsFakeA8 says so.
    [Theory]
    [InlineData(2, 768)]
    [InlineData(4, 3072)]
    public void SplitRmsFakeA8IsBitExact(int bits, int cols)
    {
        if (!Simd.Enabled || !Quant.CanSplitRmsFakeA8) return;
        var groups = cols / 128;
        var stride = (groups + 7) & ~7;
        var x = (float*)NativeMemory.Alloc((nuint)(cols * 4));
        var scratch = (float*)NativeMemory.Alloc((nuint)(cols * 4));
        var expectedValues = (sbyte*)NativeMemory.Alloc((nuint)cols);
        var actualValues = (sbyte*)NativeMemory.Alloc((nuint)cols);
        var expectedScales = (float*)NativeMemory.AllocZeroed((nuint)(stride * 8));
        var actualScales = (float*)NativeMemory.AllocZeroed((nuint)(stride * 8));
        try
        {
            Fill(x, cols, 500 + bits, 4f);
            var expected = new Activations(expectedValues, expectedScales, expectedScales + stride, cols);
            var actual = new Activations(actualValues, actualScales, actualScales + stride, cols);
            Quant.PrepareRmsFakeA8(x, scratch, cols, bits, expected);
            Ops.RmsStats(x, cols, out var inverse, out var rawPeak);
            var cut0 = groups / 3;
            var cut1 = 2 * groups / 3 + 1;
            Quant.PrepareFakeGroupsRange(x, rawPeak * inverse, inverse, bits, actual, cut1, groups);
            Quant.PrepareFakeGroupsRange(x, rawPeak * inverse, inverse, bits, actual, 0, cut0);
            Quant.PrepareFakeGroupsRange(x, rawPeak * inverse, inverse, bits, actual, cut0, cut1);
            for (var i = 0; i < cols; i++) Assert.True(expectedValues[i] == actualValues[i], $"values[{i}]");
            AssertBits(actualScales, expectedScales, 2 * stride, "scales and offsets");
        }
        finally
        {
            NativeMemory.Free(x); NativeMemory.Free(scratch); NativeMemory.Free(expectedValues); NativeMemory.Free(actualValues);
            NativeMemory.Free(expectedScales); NativeMemory.Free(actualScales);
        }
    }

    [Theory]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(23)]
    public void LoadFactorsTailMatchesScalarHalves(int groups)
    {
        if (!Simd.UseNeon) return;
        var norms = stackalloc ushort[groups];
        var specials = new ushort[] { 0x3C00, 0x0001, 0x03FF, 0x8000, 0x0000, 0x8123, 0x1555, 0xB800 };
        var random = new Random(groups);
        for (var g = 0; g < groups; g++) norms[g] = g < 4 ? (ushort)random.Next(0x0400, 0x7BFF) : specials[(g + groups) % specials.Length];
        var factors = stackalloc float[groups + 8];
        for (var i = 0; i < groups + 8; i++) factors[i] = float.NaN;
        Neon.LoadFactors(norms, groups, 0.37f, factors);
        for (var g = (groups & ~3); g < groups; g++)
            Assert.Equal(BitConverter.SingleToInt32Bits(Scalar.HalfToSingle(norms[g]) * 0.37f), BitConverter.SingleToInt32Bits(factors[g]));
    }

    // A token decoded alone (TapsHead) and inside a prefill chunk (Taps3) must convolve identically.
    [Fact]
    public void TapsHeadIsBitExactToTaps3()
    {
        if (!Simd.UseNeon) return;
        const int width = 96;
        var taps = stackalloc float[3 * width];
        var rows = stackalloc float[3 * width];
        var expected = stackalloc float[width];
        var actual = stackalloc float[width];
        var live = stackalloc float[2 * width];
        Fill(taps, 3 * width, 61);
        Fill(rows, 3 * width, 62, 5f);
        Neon.Taps3(taps, width, rows + 2 * width, expected);
        Neon.TapsHead(taps, width, rows, rows + width, rows + 2 * width, actual, live, live + width, width);
        AssertBits(actual, expected, width, "TapsHead");
        AssertBits(live, rows + width, 2 * width, "TapsHead history");
    }

    [Theory]
    [InlineData(1024)]
    [InlineData(1044)]
    public void PermuteModulateActivateIsBitExactToSingleVectors(int width)
    {
        if (!Simd.UseNeon) return;
        var permutation = (int*)NativeMemory.Alloc((nuint)width * 4);
        var source = (float*)NativeMemory.Alloc((nuint)width * 4);
        var d2 = (float*)NativeMemory.Alloc((nuint)width * 4);
        var b2 = (float*)NativeMemory.Alloc((nuint)width * 4);
        var condU = (float*)NativeMemory.Alloc((nuint)width * 8 * 4);
        var expected = (float*)NativeMemory.Alloc((nuint)width * 4);
        var actual = (float*)NativeMemory.Alloc((nuint)width * 4);
        var weights = stackalloc float[8];
        try
        {
            for (var i = 0; i < width; i++) permutation[i] = (i * 389 + 17) % width;
            Fill(source, width, 91, 6f);
            Fill(d2, width, 92, 2f);
            Fill(b2, width, 93);
            Fill(condU, width * 8, 94);
            Fill(weights, 8, 95);
            Neon.PermuteModulateActivateSingle(permutation, source, d2, weights, condU, b2, width, expected);
            Neon.PermuteModulateActivate(permutation, source, d2, weights, condU, b2, width, actual);
            AssertBits(actual, expected, width, "PermuteModulateActivate");
        }
        finally
        {
            NativeMemory.Free(permutation); NativeMemory.Free(source); NativeMemory.Free(d2); NativeMemory.Free(b2);
            NativeMemory.Free(condU); NativeMemory.Free(expected); NativeMemory.Free(actual);
        }
    }

    [Fact]
    public void CombineDiff4x4IsBitExactToFourCombines()
    {
        if (!Simd.UseNeon) return;
        const int dim = 132;
        var s = stackalloc float[4 * dim];
        var x = stackalloc float[dim];
        var u = stackalloc float[dim];
        var mix = stackalloc float[16];
        var post = stackalloc float[4];
        var expected = stackalloc float[4 * dim];
        var actual = stackalloc float[4 * dim];
        Fill(s, 4 * dim, 81, 3f);
        Fill(x, dim, 82, 2f);
        Fill(u, dim, 83, 2f);
        Fill(mix, 16, 84);
        Fill(post, 4, 85);
        for (var i = 0; i < 4; i++)
            Neon.CombineDiff4(s, dim, mix[i * 4], mix[i * 4 + 1], mix[i * 4 + 2], mix[i * 4 + 3], x, u, post[i], expected + i * dim);
        Ops.CombineDiff4x4(s, dim, mix, x, u, post, actual);
        AssertBits(actual, expected, 4 * dim, "CombineDiff4x4");
    }

    // The fused attention gate (per-head gate and peaks, then the groups at the folded peak) must
    // prepare the same activations as PrepareGateFakeA8 over the whole row.
    [Fact]
    public void FusedAttentionGateMatchesGateFakeA8()
    {
        if (!Simd.Enabled || !(Simd.UseNeon || Simd.UseAvx)) return;
        const int heads = 12;
        const int vDim = 64;
        const int cols = heads * vDim;
        var x = (float*)NativeMemory.Alloc(cols * 4);
        var fused = (float*)NativeMemory.Alloc(cols * 4);
        var gate = (float*)NativeMemory.Alloc(cols * 4);
        var peaks = (float*)NativeMemory.Alloc(heads * 16 * 4);
        var expectedValues = (sbyte*)NativeMemory.Alloc(cols);
        var actualValues = (sbyte*)NativeMemory.Alloc(cols);
        var expectedScales = (float*)NativeMemory.AllocZeroed(32 * 4);
        var actualScales = (float*)NativeMemory.AllocZeroed(32 * 4);
        try
        {
            Fill(x, cols, 71, 3f);
            Fill(gate, cols, 72, 4f);
            for (var i = 0; i < cols; i++) fused[i] = x[i];
            var expected = new Activations(expectedValues, expectedScales, expectedScales + 16, cols);
            var actual = new Activations(actualValues, actualScales, actualScales + 16, cols);
            Quant.PrepareGateFakeA8(x, gate, cols, 2, expected);
            Ops.ApplyGateAndHeadPeaks(fused, gate, heads, vDim, peaks);
            var peak = 0f;
            for (var h = 0; h < heads; h++) peak = MathF.Max(peak, peaks[h * 16]);
            Quant.PrepareFakeGroupsRange(fused, peak, 1f, 2, actual, 0, actual.Groups);
            for (var i = 0; i < cols; i++) Assert.True(expectedValues[i] == actualValues[i], $"values[{i}]");
            AssertBits(actualScales, expectedScales, 32, "scales and offsets");
        }
        finally
        {
            NativeMemory.Free(x); NativeMemory.Free(fused); NativeMemory.Free(gate); NativeMemory.Free(peaks);
            NativeMemory.Free(expectedValues); NativeMemory.Free(actualValues); NativeMemory.Free(expectedScales); NativeMemory.Free(actualScales);
        }
    }

    [Fact]
    public void LoadFactorsConvertsEveryHalfExactlyIncludingSubnormals()
    {
        if (!Simd.UseNeon) return;
        var norms = (ushort*)Marshal.AllocHGlobal(65536 * 2);
        var factors = (float*)Marshal.AllocHGlobal(65536 * 4);
        for (var i = 0; i < 65536; i++) norms[i] = (ushort)i;
        Neon.LoadFactors(norms, 65536, 0.75f, factors);
        for (var i = 0; i < 65536; i++)
        {
            var expected = Scalar.HalfToSingle((ushort)i) * 0.75f;
            Assert.True(factors[i] == expected || (float.IsNaN(expected) && float.IsNaN(factors[i])),
                $"half 0x{i:X4}: {factors[i]} vs {expected}");
        }
        Marshal.FreeHGlobal((nint)norms);
        Marshal.FreeHGlobal((nint)factors);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static float PoisonStack()
    {
        var poison = stackalloc float[16384];
        for (var i = 0; i < 16384; i++) poison[i] = float.NaN;
        return poison[16383];
    }

    [Theory]
    [InlineData(2, 9, 768, 1)]
    [InlineData(4, 9, 768, 3)]
    [InlineData(2, 5, 384, 2)]
    [InlineData(4, 13, 640, 5)]
    public void MatMulIgnoresWhateverWasOnTheStackBeforeIt(int bits, int rows, int cols, int tokens)
    {
        if (!Simd.UseNeon) return;
        var w = MakeWeights(rows, cols, bits, rows * 7 + cols + bits);
        var groups = cols / Cq.GroupSize;
        var stride = (groups + 7) & ~7;
        var x = (float*)Marshal.AllocHGlobal(tokens * cols * 4);
        Fill(x, tokens * cols, rows * 3 + tokens);
        var values = (sbyte*)Marshal.AllocHGlobal(tokens * cols);
        var scales = (float*)Marshal.AllocHGlobal(tokens * stride * 4);
        var offsets = (float*)Marshal.AllocHGlobal(tokens * stride * 4);
        var activations = new Activations(values, scales, offsets, cols);
        Quant.Prepare(x, tokens, cols, bits, activations);
        var expected = (float*)Marshal.AllocHGlobal(tokens * rows * 4);
        var actual = (float*)Marshal.AllocHGlobal(tokens * rows * 4);
        Scalar.MatMulRows(w, activations, tokens, expected, rows, 0, rows);
        PoisonStack();
        Neon.MatMulRows(w, activations, tokens, actual, rows, 0, rows);
        for (var i = 0; i < tokens * rows; i++) Assert.False(float.IsNaN(actual[i]), $"output {i} is NaN");
        AssertMatMulClose(actual, expected, tokens * rows, $"PoisonedStack{bits} {rows}x{cols}x{tokens}");
        foreach (var pointer in new[] { (nint)x, (nint)values, (nint)scales, (nint)offsets, (nint)expected, (nint)actual }) Marshal.FreeHGlobal(pointer);
        FreeWeights(w);
    }
}
