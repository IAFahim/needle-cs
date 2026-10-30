using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Xunit;

namespace Needle.Tests;

public sealed unsafe class MatMulTests : IDisposable
{
    const float Sentinel = 12345f;
    readonly List<nint> allocations = new();

    public static IEnumerable<object[]> Shapes()
    {
        foreach (var bits in new[] { 2, 4 })
            foreach (var (rows, cols) in new[] { (1, 128), (3, 128), (4, 256), (5, 768), (7, 384), (13, 768), (64, 1024), (97, 768), (576, 768), (24, 3072), (320, 3072) })
                yield return new object[] { bits, rows, cols };
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void MatchesScalarReferenceForOneToNineTokens(int bits, int rows, int cols)
    {
        var random = new Random(rows * 31 + cols + bits);
        var w = Weights(random, rows, cols, bits);
        for (var tokens = 1; tokens <= 9; tokens++)
        {
            var a = Prepared(random, tokens, cols, bits);
            var expected = (float*)Allocate((nuint)(tokens * rows) * 4);
            var actual = (float*)Allocate((nuint)(tokens * rows) * 4);
            Fill(expected, tokens * rows, Sentinel);
            Fill(actual, tokens * rows, Sentinel);
            Scalar.MatMulRows(w, a, tokens, expected, rows, 0, rows);
            Quant.MatMulRows(w, a, tokens, actual, rows, 0, rows);
            AssertClose(expected, actual, tokens * rows);
        }
    }

    public static IEnumerable<object[]> WideShapes()
    {
        foreach (var bits in new[] { 2, 4 })
            foreach (var (rows, cols) in new[] { (1, 128), (2, 128), (5, 768), (97, 768), (576, 768), (320, 3072) })
                foreach (var tokens in new[] { 10, 15, 16, 17, 24, 31, 32, 33, 64 })
                    yield return new object[] { bits, rows, cols, tokens };
    }

    [Theory]
    [MemberData(nameof(WideShapes))]
    public void MatchesScalarReferenceForWideTokens(int bits, int rows, int cols, int tokens)
    {
        var random = new Random(rows * 31 + cols + bits);
        var w = Weights(random, rows, cols, bits);
        var a = Prepared(random, tokens, cols, bits);
        var expected = (float*)Allocate((nuint)(tokens * rows) * 4);
        var actual = (float*)Allocate((nuint)(tokens * rows) * 4);
        Fill(expected, tokens * rows, Sentinel);
        Fill(actual, tokens * rows, Sentinel);
        Scalar.MatMulRows(w, a, tokens, expected, rows, 0, rows);
        Quant.MatMulRows(w, a, tokens, actual, rows, 0, rows);
        AssertClose(expected, actual, tokens * rows);
    }

    [Fact]
    public void HalfToSingleDecodesEveryFiniteBitPatternPerIeee754()
    {
        for (var bits = 0; bits < 0x10000; bits++)
        {
            var half = (ushort)bits;
            var magnitude = half & 0x7FFF;
            if (magnitude >= 0x7C00) continue;
            var expected = magnitude < 0x0400
                ? ((half & 0x8000) != 0 ? -1f : 1f) * magnitude * 5.9604645e-8f
                : BitConverter.Int32BitsToSingle(((magnitude << 13) + 0x38000000) | ((half & 0x8000) << 16));
            Assert.Equal(expected, Scalar.HalfToSingle(half));
        }
    }

    [Fact]
    public void HalfToSingleUpstreamDecodesSubnormalsWithImplicitExponent()
    {
        for (var bits = 0; bits < 0x10000; bits++)
        {
            var half = (ushort)bits;
            var magnitude = half & 0x7FFF;
            if (magnitude == 0 || magnitude >= 0x0400) continue;
            Assert.Equal(((half & 0x8000) != 0 ? -1f : 1f) * (0x0400 + magnitude) * 2.9802322e-8f, Scalar.HalfToSingleUpstream(half));
        }
    }

    [Theory]
    [InlineData(2, 37, 768, 5, 29)]
    [InlineData(2, 37, 768, 1, 2)]
    [InlineData(4, 37, 768, 3, 36)]
    [InlineData(4, 21, 256, 8, 21)]
    [InlineData(2, 50, 384, 0, 7)]
    public void LeavesRowsOutsideTheRangeUntouched(int bits, int rows, int cols, int rowStart, int rowEnd)
    {
        var random = new Random(rows + rowStart);
        var w = Weights(random, rows, cols, bits);
        foreach (var tokens in new[] { 1, 2, 3, 5, 6, 7, 8, 16 })
        {
            var a = Prepared(random, tokens, cols, bits);
            var ldy = rows + 3;
            var expected = (float*)Allocate((nuint)(tokens * ldy) * 4);
            var actual = (float*)Allocate((nuint)(tokens * ldy) * 4);
            Fill(expected, tokens * ldy, Sentinel);
            Fill(actual, tokens * ldy, Sentinel);
            Scalar.MatMulRows(w, a, tokens, expected, ldy, rowStart, rowEnd);
            Quant.MatMulRows(w, a, tokens, actual, ldy, rowStart, rowEnd);
            AssertClose(expected, actual, tokens * ldy);
        }
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void SlicedWeightsAtTheEndOfTheirBufferAreExact(int bits)
    {
        var random = new Random(bits);
        var parent = Weights(random, 40, 768, bits);
        var a = Prepared(random, 1, 768, bits);
        foreach (var (first, count) in new[] { (0, 4), (36, 4), (35, 5), (38, 2), (39, 1) })
        {
            var slice = parent.Slice(first, count);
            var expected = (float*)Allocate((nuint)count * 4);
            var actual = (float*)Allocate((nuint)count * 4);
            Scalar.MatMulRows(slice, a, 1, expected, count, 0, count);
            Quant.MatMulRows(slice, a, 1, actual, count, 0, count);
            AssertClose(expected, actual, count);
        }
    }

    void AssertClose(float* expected, float* actual, int count)
    {
        var magnitude = 1f;
        for (var i = 0; i < count; i++) if (expected[i] != Sentinel) magnitude = MathF.Max(magnitude, MathF.Abs(expected[i]));
        for (var i = 0; i < count; i++)
        {
            if (expected[i] == Sentinel) Assert.Equal(Sentinel, actual[i]);
            else Assert.InRange(actual[i] - expected[i], -1e-3f * magnitude, 1e-3f * magnitude);
        }
    }

    static void Fill(float* x, int count, float value)
    {
        for (var i = 0; i < count; i++) x[i] = value;
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
            norms[i] = BitConverter.HalfToUInt16Bits((Half)(i % 11 == 5 ? 0f : 0.2f + (float)random.NextDouble() * 1.8f));
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
