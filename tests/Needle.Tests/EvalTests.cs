using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Needle.Cli;
using Xunit;

namespace Needle.Tests;

public sealed class EvalDatasetTests
{
    static readonly string DatasetPath = Path.Combine(Assets.Root, "tests/eval");

    static ArgumentMatcher Matcher(string json)
    {
        using var document = JsonDocument.Parse(json);
        return ArgumentMatcher.Parse(document.RootElement);
    }

    static JsonElement Value(string json)
    {
        using var document = JsonDocument.Parse(json);
        return document.RootElement.Clone();
    }

    [Fact]
    public void DatasetCoversFiveToolSetsWithOutOfDomainPromptsAndATwentyPromptSubset()
    {
        var dataset = EvalDataset.Load(DatasetPath);
        Assert.Equal(new[] { "chains", "dialogue", "rpg", "rts", "settings" }, dataset.ToolSets.Keys.OrderBy(k => k, StringComparer.Ordinal));
        Assert.InRange(dataset.Cases.Count, 180, 240);
        Assert.InRange(dataset.Cases.Count(c => !c.InDomain), 25, 60);
        Assert.Equal(dataset.Cases.Count, dataset.Cases.Select(c => c.Id).Distinct().Count());
        Assert.Equal(20, EvalDataset.Load(DatasetPath, "ci").Cases.Count);
        Assert.Equal(12, EvalDataset.Load(DatasetPath, limit: 12).Cases.Count);
        foreach (var set in dataset.ToolSets.Values) Assert.True(new ToolSet(set).Tools.Count >= 6);
    }

    [Fact]
    public void MatchersImplementExactNearContainsOneOfAndAny()
    {
        Assert.True(Matcher("\"Lyra\"").Matches(Value("\"Lyra\"")));
        Assert.False(Matcher("\"Lyra\"").Matches(Value("\"lyra\"")));
        Assert.True(Matcher("3").Matches(Value("3.0")));
        Assert.False(Matcher("3").Matches(Value("\"3\"")));
        Assert.True(Matcher("true").Matches(Value("true")));
        Assert.False(Matcher("true").Matches(Value("false")));
        Assert.True(Matcher("{\"near\":100,\"tol\":2}").Matches(Value("101.5")));
        Assert.False(Matcher("{\"near\":100,\"tol\":2}").Matches(Value("103")));
        Assert.True(Matcher("{\"contains\":\"Orc\"}").Matches(Value("\"the ORC chieftain\"")));
        Assert.False(Matcher("{\"contains\":\"orc\"}").Matches(Value("7")));
        Assert.True(Matcher("{\"oneOf\":[\"line\",\"wedge\"]}").Matches(Value("\"wedge\"")));
        Assert.False(Matcher("{\"oneOf\":[\"line\",\"wedge\"]}").Matches(Value("\"circle\"")));
        Assert.True(Matcher("{\"any\":true}").Matches(Value("null")));
        Assert.Throws<FormatException>(() => Matcher("{\"unknown\":1}"));
    }

    [Fact]
    public void ScoringRequiresOrderedNamesMatchingArgumentsAndNoExtraArguments()
    {
        var expected = new[]
        {
            new ExpectedCall("give_item", new[]
            {
                KeyValuePair("item", "\"potion\""),
                KeyValuePair("count", "3"),
            }),
            new ExpectedCall("stop", Array.Empty<System.Collections.Generic.KeyValuePair<string, ArgumentMatcher>>()),
        };
        Assert.True(Score(expected, "[{\"name\":\"give_item\",\"arguments\":{\"item\":\"potion\",\"count\":3}},{\"name\":\"stop\",\"arguments\":{}}]").Exact);
        Assert.False(Score(expected, "[{\"name\":\"stop\",\"arguments\":{}},{\"name\":\"give_item\",\"arguments\":{\"item\":\"potion\",\"count\":3}}]").Exact);
        Assert.False(Score(expected, "[{\"name\":\"give_item\",\"arguments\":{\"item\":\"potion\",\"count\":3,\"extra\":1}},{\"name\":\"stop\",\"arguments\":{}}]").Exact);
        var partial = Score(expected, "[{\"name\":\"give_item\",\"arguments\":{\"item\":\"potion\",\"count\":4}}]");
        Assert.False(partial.Exact);
        Assert.Equal(1, partial.NamesMatched);
        Assert.Equal(2, partial.ArgumentsExpected);
        Assert.Equal(1, partial.ArgumentsMatched);
    }

    [Fact]
    public void PolicyRegistryParsesKnownSettingsAndRejectsUnknownOnes()
    {
        var defaults = EvalPolicy.Defaults(1);
        var policy = EvalPolicies.Apply(defaults, new[] { "RefuseBelow=0.3", "autodate=false", "MaxReasoningTokens=32" });
        Assert.Equal(0.3f, policy.RefuseBelow);
        Assert.False(policy.AutoDate);
        Assert.Equal(32, policy.MaxReasoningTokens);
        Assert.Equal("defaults", EvalPolicies.Describe(defaults, defaults));
        Assert.Equal("RefuseBelow=0.3 AutoDate=false MaxReasoningTokens=32", EvalPolicies.Describe(policy, defaults));
        Assert.Throws<ArgumentException>(() => EvalPolicies.Apply(defaults, new[] { "JumpForward=true" }));
        Assert.Throws<ArgumentException>(() => EvalPolicies.Apply(defaults, new[] { "RefuseBelow=high" }));
    }

    static System.Collections.Generic.KeyValuePair<string, ArgumentMatcher> KeyValuePair(string key, string matcherJson) => new(key, Matcher(matcherJson));

    static CaseScore Score(ExpectedCall[] expected, string callsJson)
    {
        Assert.True(ObservedCall.TryParse(callsJson, out var observed));
        return CaseScore.Of(expected, observed);
    }
}

public sealed class EvalTests : IDisposable
{
    const double MeasuredAccuracy = 17.0 / 20.0;
    const double AccuracyFloor = 0.65;

    readonly Model model;

    public EvalTests()
    {
        Assets.Require();
        model = Model.Open(Assets.Model);
    }

    public void Dispose() => model.Dispose();

    [Fact]
    public void TwentyPromptSubsetStaysAboveTheMeasuredAccuracyFloor()
    {
        var dataset = EvalDataset.Load(Path.Combine(Assets.Root, "tests/eval"), "ci");
        var policy = EvalPolicies.Apply(EvalPolicy.Defaults(2), new[] { "AutoDate=false", "RetrievalTopK=64" });
        var run = EvalRunner.Run(model, dataset, policy, "ci", null);
        var summary = EvalMetrics.Summarize("ci", run.Results);

        Assert.Equal(20, summary.Prompts);
        Assert.Equal(0, summary.Failures);
        Assert.True(summary.OverallAccuracy >= AccuracyFloor,
            $"overall accuracy {summary.OverallAccuracy:P0} fell below the {AccuracyFloor:P0} floor (measured {MeasuredAccuracy:P0} when the floor was set); misses: {string.Join("; ", run.Results.Where(r => !r.Correct).Select(r => $"{r.Case.Id} \"{r.Case.Query}\" -> {r.DescribeOutcome()}"))}");
    }
}
