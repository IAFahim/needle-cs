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
        // Process-wide private bytes can dip while other test classes churn, so
        // retry the cheap measurement until the ctor delta is observable.
        var report = BenchE2E.Measure(model, File.ReadAllText(Assets.Tools), options, "test", null);
        for (var attempt = 0; attempt < 3 && report.Runs[0].Sets[0].SessionCtorPrivateBytes <= 0; attempt++)
            report = BenchE2E.Measure(model, File.ReadAllText(Assets.Tools), options, "test", null);

        var run = Assert.Single(report.Runs);
        var set = Assert.Single(run.Sets);
        Assert.Equal(BenchPrompts.Golden.Length, set.Prompts.Count);
        Assert.True(set.PrefixTokens > 0);
        Assert.True(set.SessionCtorMs > 0);
        // Process.PrivateMemorySize64 reports 0 on macOS, so the ctor delta
        // stays 0 there no matter how often we retry (CI evidence: run
        // 41030471519 macos-arm64). Keep the signal where the OS provides it.
        if (OperatingSystem.IsMacOS()) Assert.True(set.SessionCtorPrivateBytes >= 0);
        else Assert.True(set.SessionCtorPrivateBytes > 0);
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
        Assert.Equal((double)(geometry.Layers * 4 + geometry.SiteCount + 1), profile.PoolRunsPerToken);
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

    [Fact]
    public void SpanApiModeMatchesTextMode()
    {
        var text = new BenchE2EOptions { Threads = new[] { 1 }, Reps = 1, Warmups = 0, Sets = "golden", DecodeBudget = 8 };
        var textReport = BenchE2E.Measure(model, File.ReadAllText(Assets.Tools), text, "test", null);
        var spanReport = BenchE2E.Measure(model, File.ReadAllText(Assets.Tools), text with { Api = "span" }, "test", null);

        Assert.Equal("span", spanReport.Options.Api);
        Assert.Equal(textReport.Runs.Count, spanReport.Runs.Count);
        for (var r = 0; r < textReport.Runs.Count; r++)
        {
            Assert.Equal(textReport.Runs[r].Sets.Count, spanReport.Runs[r].Sets.Count);
            for (var s = 0; s < textReport.Runs[r].Sets.Count; s++)
            {
                var expected = textReport.Runs[r].Sets[s];
                var actual = spanReport.Runs[r].Sets[s];
                Assert.Equal(expected.Prompts.Count, actual.Prompts.Count);
                for (var p = 0; p < expected.Prompts.Count; p++)
                {
                    Assert.Equal(0, actual.Prompts[p].Failures);
                    Assert.Equal(expected.Prompts[p].Calls, actual.Prompts[p].Calls);
                    Assert.Equal(expected.Prompts[p].PrefillTokens, actual.Prompts[p].PrefillTokens);
                    Assert.Equal(expected.Prompts[p].DecodeTokens, actual.Prompts[p].DecodeTokens);
                    Assert.True(Spread.Of(actual.Prompts[p].TotalMs).IsFinite);
                    Assert.True(actual.Prompts[p].ManagedBytes.Count == 1 && actual.Prompts[p].ManagedBytes[0] >= 0);
                    Assert.True(actual.Prompts[p].NativeAllocs.Count == 1 && actual.Prompts[p].NativeAllocs[0] >= 0);
                }
            }
        }
        Assert.Empty(BenchE2E.Violations(spanReport));
        Assert.Contains("api span", BenchE2E.ToText(spanReport));
        using var document = JsonDocument.Parse(BenchE2E.ToJson(spanReport));
        Assert.Equal("span", document.RootElement.GetProperty("api").GetString());
        var native = document.RootElement.GetProperty("runs")[0].GetProperty("sets")[0].GetProperty("prompts")[0].GetProperty("native_allocs");
        if (Mem.CountersEnabled) Assert.True(native.GetProperty("median").GetDouble() >= 0);
        else Assert.Equal(JsonValueKind.Null, native.ValueKind);
    }

    [Fact]
    public void ApiOptionRejectsUnknownModes()
    {
        Assert.Equal("text", BenchE2EOptions.Parse(Array.Empty<string>()).Api);
        Assert.Equal("span", BenchE2EOptions.Parse(new[] { "--api", "span" }).Api);
        Assert.Equal("text", BenchE2EOptions.Parse(new[] { "--api", "TEXT" }).Api);
        Assert.Throws<ArgumentException>(() => BenchE2EOptions.Parse(new[] { "--api", "bogus" }));
    }

    static PromptMeasurement GatePrompt(long managedBytes, long nativeAllocs, int failures = 0) => new()
    {
        Query = "q",
        Calls = "get_weather",
        Failures = failures,
        PrefillTokens = 10,
        DecodeTokens = 5,
        PrefillMs = { 1 },
        DecodeUsPerToken = { 100 },
        TtftMs = { 1 },
        TotalMs = { 2 },
        PrefillTps = { 1000 },
        DecodeTps = { 10000 },
        ManagedBytes = { managedBytes },
        NativeAllocs = { nativeAllocs },
    };

    static E2EReport GateReport(PromptMeasurement prompt, long? maxManaged, long? maxNative)
    {
        var set = new SetMeasurement { Name = "golden" };
        set.Prompts.Add(prompt);
        var run = new ThreadMeasurement { Threads = 1 };
        run.Sets.Add(set);
        var report = new E2EReport { Model = "test", Options = new BenchE2EOptions { MaxManagedBytes = maxManaged, MaxNativeAllocs = maxNative } };
        report.Runs.Add(run);
        return report;
    }

    [Fact]
    public void ZeroGatesPassOnlyWhenNothingWasAllocated()
    {
        var clean = BenchE2E.Violations(GateReport(GatePrompt(0, 0), 0, 0));
        if (Mem.CountersEnabled) Assert.Empty(clean);
        else Assert.Contains(clean, v => v.Contains("NeedleAllocCounters"));

        Assert.Contains(BenchE2E.Violations(GateReport(GatePrompt(5, 0), 0, 0)), v => v.Contains("managed bytes"));
        var native = BenchE2E.Violations(GateReport(GatePrompt(0, 3), 0, 0));
        Assert.Contains(native, v => v.Contains(Mem.CountersEnabled ? "native allocations" : "NeedleAllocCounters"));

        Assert.Contains(BenchE2E.Violations(GateReport(GatePrompt(0, 0, failures: 1), 0, 0)), v => v.Contains("failed calls"));
        Assert.Empty(BenchE2E.Violations(GateReport(GatePrompt(732, 0), null, null)));
    }
}
