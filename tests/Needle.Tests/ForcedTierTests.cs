using System;
using System.Runtime.Intrinsics.X86;
using Xunit;

namespace Needle.Tests;

public sealed class ForcedTierTests
{
    [Fact]
    public void UnsupportedForcedTierThrowsNamingRequestedAndAutoPick()
    {
        var raw = AvxVnni.IsSupported ? "neon" : "avxvnni";
        var ex = Assert.Throws<InvalidOperationException>(() => Tiers.Resolve(raw));
        Assert.Contains(raw, ex.Message);
        Assert.Contains(Tiers.Resolve(null).ToString(), ex.Message);
    }

    [Fact]
    public void UnknownForcedTierThrows()
    {
        Assert.Throws<InvalidOperationException>(() => Tiers.Resolve("zen4"));
    }

    [Fact]
    public void SupportedForcedTierStillResolves()
    {
        Assert.Equal(CpuTier.Scalar, Tiers.Resolve("scalar"));
        if (Avx2.IsSupported && Fma.IsSupported && Sse41.IsSupported)
            Assert.Equal(CpuTier.Avx2, Tiers.Resolve("avx2"));
    }

    [Fact]
    public void UnsetTierAutoPicksWithoutThrowing()
    {
        var expected = Tiers.Resolve(null);
        Assert.Equal(expected, Tiers.Resolve(""));
        Assert.Equal(expected, Tiers.Resolve("   "));
        if (Environment.GetEnvironmentVariable("NEEDLE_TIER") == null
            && Environment.GetEnvironmentVariable("NEEDLE_SCALAR") == null)
            Assert.Equal(Tiers.Active, expected);
    }
}
