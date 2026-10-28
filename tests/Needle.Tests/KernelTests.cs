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
