using System;
using System.IO;
using Xunit;

namespace Needle.Tests;

public sealed class SessionStartupTests : IDisposable
{
    readonly Model model;

    public SessionStartupTests()
    {
        Assets.Require();
        model = Model.Open(Assets.Model);
    }

    public void Dispose() => model.Dispose();

    static string Tools => File.ReadAllText(Assets.Tools);

    static NeedleOptions Options => new() { Threads = 4 };

    [Fact]
    public void SecondSessionMatchesTheFirst()
    {
        using var first = new NeedleSession(model, Tools, Options);
        using var second = new NeedleSession(model, Tools, Options);
        var a = first.Complete("what's it like in Lagos right now?");
        var b = second.Complete("what's it like in Lagos right now?");
        Assert.Equal(Stable(a), Stable(b));
    }

    [Fact]
    public void PrimedPrefixRoundTripsAndValidatesItsKey()
    {
        using var first = new NeedleSession(model, Tools, Options);
        var prefix = first.ExportPrimedPrefix();
        Assert.True(prefix.Length > 1024);

        using var restored = new NeedleSession(model, Tools, prefix, Options);
        var expected = first.Complete("set a timer for 12 minutes");
        var actual = restored.Complete("set a timer for 12 minutes");
        Assert.Equal(Stable(expected), Stable(actual));

        var otherTools = "[{\"name\":\"noop\",\"description\":\"Do nothing.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}]";
        Assert.Throws<ArgumentException>(() => new NeedleSession(model, otherTools, prefix, Options));
        Assert.Throws<ArgumentException>(() => new NeedleSession(model, Tools, prefix.AsSpan(0, prefix.Length / 2), Options));
    }

    [Fact]
    public void WarmUpLeavesTheSessionPrimedAndIdentical()
    {
        using var warmed = new NeedleSession(model, Tools, Options);
        warmed.WarmUp();
        using var fresh = new NeedleSession(model, Tools, Options);
        var a = warmed.Complete("what's it like in Lagos right now?");
        var b = fresh.Complete("what's it like in Lagos right now?");
        Assert.Equal(Stable(b), Stable(a));

        var again = warmed.Complete("play some Miles Davis");
        Assert.True(again.Success);
    }

    [Fact]
    public void AutoDateIsOffUnlessRequested()
    {
        Assert.False(new NeedleOptions().AutoDate);
        using var session = new NeedleSession(model, Tools, Options);
        var a = session.Complete("what's it like in Lagos right now?");
        session.Reset();
        var b = session.Complete("what's it like in Lagos right now?");
        Assert.Equal(Stable(a), Stable(b));
    }

    static string Stable(CallResult result) => new CallResult
    {
        Success = result.Success,
        Error = result.Error,
        FunctionCalls = result.FunctionCalls,
        SuppressedCalls = result.SuppressedCalls,
        Reasoning = result.Reasoning,
        Response = result.Response,
        Confidence = result.Confidence,
        HeadConfidence = result.HeadConfidence,
        DecodeConfidence = result.DecodeConfidence,
        PrefillTokens = result.PrefillTokens,
        DecodeTokens = result.DecodeTokens,
    }.ToJson();
}
