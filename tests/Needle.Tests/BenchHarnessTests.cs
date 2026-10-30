using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Needle.Cli;
using Xunit;

namespace Needle.Tests;

public sealed class BenchHarnessTests : IDisposable
{
    readonly Model model;

    public BenchHarnessTests()
    {
        Assets.Require();
        model = Model.Open(Assets.Model);
    }

    public void Dispose() => model.Dispose();

    [Fact]
    public void EndToEndHarnessReturnsFiniteNumbersAndProfilesEveryPoolRun()
    {
        var options = new BenchE2EOptions { Threads = new[] { 1 }, Reps = 1, Warmups = 0, Sets = "golden", Profile = true, DecodeBudget = 8 };
        var report = BenchE2E.Measure(model, File.ReadAllText(Assets.Tools), options, "test", null);

        var run = Assert.Single(report.Runs);
        var set = Assert.Single(run.Sets);
        Assert.Equal(BenchPrompts.Golden.Length, set.Prompts.Count);
        Assert.True(set.PrefixTokens > 0);
        Assert.True(set.SessionCtorMs > 0);
        Assert.True(set.SessionCtorPrivateBytes > 0);
        foreach (var prompt in set.Prompts)
        {
            Assert.Equal(0, prompt.Failures);
            Assert.True(Spread.Of(prompt.PrefillMs).IsFinite && prompt.PrefillMs[0] > 0);
            Assert.True(Spread.Of(prompt.DecodeUsPerToken).IsFinite && prompt.DecodeUsPerToken[0] > 0);
            Assert.True(Spread.Of(prompt.TtftMs).IsFinite);
            Assert.True(Spread.Of(prompt.TotalMs).IsFinite && prompt.TotalMs[0] > 0);
            Assert.True(prompt.ManagedBytes[0] >= 0);
            Assert.True(prompt.DecodeTokens > 0);
        }

        var profile = Assert.IsType<ProfileResult>(run.Profile);
        var geometry = model.Geometry;
        Assert.Equal((double)(geometry.Layers + 1), profile.PoolRunsPerToken);
        Assert.Single(profile.Reps);
        for (var stage = 0; stage < StageProfile.StageCount; stage++) Assert.True(profile.DecodeStage(stage).IsFinite);
        Assert.True(profile.DecodeStage((int)Stage.Mlp).Best > 0);
        Assert.True(profile.DecodeStage((int)Stage.Logits).Best > 0);

        using var document = JsonDocument.Parse(BenchE2E.ToJson(report));
        var root = document.RootElement;
        Assert.Equal("needle.bench-e2e/1", root.GetProperty("schema").GetString());
        var first = root.GetProperty("runs")[0];
        Assert.Equal(1, first.GetProperty("threads").GetInt32());
        Assert.True(first.GetProperty("sets")[0].GetProperty("summary").GetProperty("total_ms").GetProperty("best").GetDouble() > 0);
        Assert.True(first.GetProperty("sets")[0].GetProperty("session_ctor_ms").GetDouble() > 0);
        Assert.True(first.GetProperty("sets")[0].GetProperty("session_ctor_private_bytes").GetInt64() > 0);
        Assert.True(first.GetProperty("profile").GetProperty("decode_us_per_token").GetProperty("best").GetDouble() > 0);
        Assert.Contains("mlp", BenchE2E.ToText(report));
        Assert.Contains("ctor", BenchE2E.ToText(report));
        Assert.Empty(BenchE2E.Violations(report));

        report.Options = options with { MaxManagedBytes = 0 };
        Assert.NotEmpty(BenchE2E.Violations(report));
    }

    [Fact]
    public void PromptSetsAndNativeScriptStayInSync()
    {
        var script = File.ReadAllText(Path.Combine(Assets.Root, "tools/native_bench.py"));
        foreach (var query in BenchPrompts.Golden.Concat(BenchPrompts.Game)) Assert.Contains($"\"{query}\"", script);
        Assert.Equal(File.ReadAllText(Path.Combine(Assets.Root, "tools/game_tools.json")), BenchPrompts.GameToolsJson);
        Assert.Equal(7, new ToolSet(BenchPrompts.GameToolsJson).Tools.Count);
    }

    [Fact]
    public void TimingAndJsonPlumbingAreWellFormed()
    {
        var counter = 0;
        var spread = BenchTiming.NanosecondsPerCall(() => counter++, rounds: 3, roundMilliseconds: 2);
        Assert.True(spread.IsFinite && spread.Best > 0 && spread.Best <= spread.Median);
        Assert.True(counter > 0);

        var json = new BenchJson().BeginObject().Field("name", "a\"b").Field("count", 3).Field("ratio", double.NaN).Field("spread", new Spread(1, 2))
            .Key("list").BeginArray().Value(1.5).Value(true).Null().EndArray().EndObject().ToString();
        using var document = JsonDocument.Parse(json);
        Assert.Equal("a\"b", document.RootElement.GetProperty("name").GetString());
        Assert.Equal(JsonValueKind.Null, document.RootElement.GetProperty("ratio").ValueKind);
        Assert.Equal(2, document.RootElement.GetProperty("spread").GetProperty("median").GetDouble());
        Assert.Equal(3, document.RootElement.GetProperty("list").GetArrayLength());
    }
}
