using System;
using Xunit;

namespace Needle.Tests;

// SIMD-04: stencil kernels use negative-offset unaligned loads
// (Taps3 reads current[c-width], current[c-2*width]; RoPE walks lo/hi pairs).
// These canary tests prove the vector loops never read or write past their
// arrays: guard lanes around every buffer must survive, and the AVX2 output
// must match the scalar reference.
public sealed unsafe class SimdCanaryTests
{
    const float Guard = 12345.678f;

    static void Fill(float* x, int n, int seed, float mag)
    {
        var r = new Random(seed);
        for (var i = 0; i < n; i++) x[i] = ((float)r.NextDouble() - 0.5f) * mag;
    }

    static void CheckGuards(float* g, int n, string name)
    {
        for (var i = 0; i < n; i++) Assert.Equal(Guard, g[i]);
    }

    [Theory]
    [InlineData(64)]
    [InlineData(768)]
    public void Taps3KeepsCanariesAndMatchesScalar(int width)
    {
        var taps = stackalloc float[3 * width + 16];
        var hist = stackalloc float[3 * width + 16];
        var dst = stackalloc float[width + 16];
        var expect = stackalloc float[width + 16];
        Fill(taps + 8, 3 * width, 11, 1f);
        Fill(hist + 8, 3 * width, 12, 1f);
        for (var i = 0; i < 8; i++) { taps[i] = Guard; taps[3 * width + 8 + i] = Guard; hist[i] = Guard; hist[3 * width + 8 + i] = Guard; }
        for (var i = 0; i < 8; i++) { dst[i] = Guard; dst[width + 8 + i] = Guard; expect[i] = Guard; expect[width + 8 + i] = Guard; }
        float* current = hist + 8 + 2 * width;
        Ops.Taps3(taps + 8, width, current, dst + 8);
        Scalar.Taps3(taps + 8, width, current, expect + 8);
        for (var i = 0; i < width; i++) Assert.InRange(dst[8 + i] - expect[8 + i], -1e-5f, 1e-5f);
        CheckGuards(taps, 8, "taps-lo"); CheckGuards(taps + 3 * width + 8, 8, "taps-hi");
        CheckGuards(hist, 8, "hist-lo"); CheckGuards(hist + 3 * width + 8, 8, "hist-hi");
        CheckGuards(dst, 8, "dst-lo"); CheckGuards(dst + width + 8, 8, "dst-hi");
    }

    [Theory]
    [InlineData(1, 48)]
    [InlineData(4, 48)]
    [InlineData(12, 48)]
    public void RopeKeepsCanariesAndMatchesScalar(int heads, int headDim)
    {
        var n = heads * headDim;
        var half = headDim / 2;
        var x = stackalloc float[n + 16];
        var expect = stackalloc float[n + 16];
        var cos = stackalloc float[half + 16];
        var sin = stackalloc float[half + 16];
        Fill(x + 8, n, 21, 2f);
        for (var i = 0; i < n; i++) expect[8 + i] = x[8 + i];
        for (var i = 0; i < half; i++) { cos[8 + i] = MathF.Cos(0.13f * i); sin[8 + i] = MathF.Sin(0.13f * i); }
        for (var i = 0; i < 8; i++) { x[i] = Guard; x[n + 8 + i] = Guard; expect[i] = Guard; expect[n + 8 + i] = Guard; }
        for (var i = 0; i < 8; i++) { cos[i] = Guard; cos[half + 8 + i] = Guard; sin[i] = Guard; sin[half + 8 + i] = Guard; }
        Ops.Rope(x + 8, heads, headDim, cos + 8, sin + 8);
        Scalar.Rope(expect + 8, heads, headDim, cos + 8, sin + 8);
        for (var i = 0; i < n; i++) Assert.InRange(x[8 + i] - expect[8 + i], -1e-5f, 1e-5f);
        CheckGuards(x, 8, "x-lo"); CheckGuards(x + n + 8, 8, "x-hi");
        CheckGuards(cos, 8, "cos-lo"); CheckGuards(cos + half + 8, 8, "cos-hi");
        CheckGuards(sin, 8, "sin-lo"); CheckGuards(sin + half + 8, 8, "sin-hi");
    }
}
