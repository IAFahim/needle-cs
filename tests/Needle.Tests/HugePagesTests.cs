using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;

namespace Needle.Tests;

public sealed class HugePagesTests
{
    [Theory]
    [InlineData(null, false)]
    [InlineData("", false)]
    [InlineData("0", false)]
    [InlineData("1", true)]
    [InlineData("always", true)]
    public void HugePagesSettingParses(string? value, bool expected) =>
        Assert.Equal(expected, Paging.ParseHugePagesSetting(value));

    [Fact]
    public void LiveFlagMatchesEnvironment() =>
        Assert.Equal(Paging.ParseHugePagesSetting(Environment.GetEnvironmentVariable("NEEDLE_HUGEPAGES")), Paging.HugePagesEnabled);

    [Fact]
    public unsafe void AdviseAppliesOnlyWhenOptedIn()
    {
        var buffer = (byte*)Mem.AlignedAlloc(65536, 4096);
        try
        {
            Assert.Equal(Paging.HugePagesEnabled && OperatingSystem.IsLinux(), Paging.AdviseHugePages(buffer, 65536));
            Assert.False(Paging.AdviseHugePages(buffer, 0));
        }
        finally
        {
            Mem.AlignedFree(buffer);
        }
    }

    [Fact]
    public unsafe void LargeArenaAlignmentFollowsOptIn()
    {
        var arena = new Arena(Paging.HugePageBytes);
        try
        {
            Assert.Equal((nuint)0, (nuint)arena.Base % (Paging.HugePagesEnabled ? Paging.HugePageBytes : 64));
        }
        finally
        {
            arena.Free();
        }
    }

    [Fact]
    public unsafe void SmallArenaKeepsCacheLineAlignment()
    {
        var arena = new Arena(4096);
        try
        {
            Assert.Equal((nuint)0, (nuint)arena.Base % 64);
        }
        finally
        {
            arena.Free();
        }
    }

    [Fact]
    public void ModelLoadMatchesReferenceInBothModes()
    {
        Assets.Require();
        var ids = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/ids.bin"))).ToArray();
        var expected = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(Assets.Golden, "case0/logits.bin"))).ToArray();
        using var model = Model.Open(Assets.Model);
        using var engine = new Engine(model, 512, 64, 1);
        engine.Feed(ids);
        var actual = engine.Logits.ToArray();
        Assert.Equal(Array.IndexOf(expected, expected.Max()), Array.IndexOf(actual, actual.Max()));
    }
}
