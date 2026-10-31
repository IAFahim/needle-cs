using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json;

namespace Needle.Cli;

internal enum MatcherKind { Exact, Near, Contains, OneOf, Any }

internal sealed class ArgumentMatcher
{
    readonly MatcherKind kind;
    readonly JsonElement target;
    readonly double tolerance;
    readonly string fragment = "";
    readonly ArgumentMatcher[] alternatives = Array.Empty<ArgumentMatcher>();

    ArgumentMatcher(MatcherKind kind, JsonElement target = default, double tolerance = 0, string fragment = "", ArgumentMatcher[]? alternatives = null)
    {
        this.kind = kind;
        this.target = target;
        this.tolerance = tolerance;
        this.fragment = fragment;
        this.alternatives = alternatives ?? Array.Empty<ArgumentMatcher>();
    }

    public static ArgumentMatcher Parse(JsonElement specification)
    {
        if (specification.ValueKind != JsonValueKind.Object) return new ArgumentMatcher(MatcherKind.Exact, specification.Clone());
        if (specification.TryGetProperty("contains", out var contains)) return new ArgumentMatcher(MatcherKind.Contains, fragment: contains.GetString() ?? "");
        if (specification.TryGetProperty("near", out var near))
        {
            var tolerance = specification.TryGetProperty("tol", out var tol) ? tol.GetDouble() : 0;
            return new ArgumentMatcher(MatcherKind.Near, near.Clone(), tolerance);
        }
        if (specification.TryGetProperty("oneOf", out var choices))
            return new ArgumentMatcher(MatcherKind.OneOf, alternatives: choices.EnumerateArray().Select(choice => new ArgumentMatcher(MatcherKind.Exact, choice.Clone())).ToArray());
        if (specification.TryGetProperty("any", out _)) return new ArgumentMatcher(MatcherKind.Any);
        throw new FormatException($"unknown matcher {specification.GetRawText()}; use a value, {{\"near\":n,\"tol\":t}}, {{\"contains\":s}}, {{\"oneOf\":[...]}} or {{\"any\":true}}");
    }

    public bool Matches(JsonElement actual) => kind switch
    {
        MatcherKind.Exact => SameValue(target, actual),
        MatcherKind.Near => actual.ValueKind == JsonValueKind.Number && Math.Abs(actual.GetDouble() - target.GetDouble()) <= tolerance,
        MatcherKind.Contains => actual.ValueKind == JsonValueKind.String && actual.GetString()!.Contains(fragment, StringComparison.OrdinalIgnoreCase),
        MatcherKind.OneOf => alternatives.Any(alternative => alternative.Matches(actual)),
        _ => true,
    };

    public string Describe() => kind switch
    {
        MatcherKind.Exact => target.GetRawText(),
        MatcherKind.Near => string.Create(CultureInfo.InvariantCulture, $"{target.GetRawText()}+-{tolerance}"),
        MatcherKind.Contains => $"~\"{fragment}\"",
        MatcherKind.OneOf => string.Join("|", alternatives.Select(alternative => alternative.Describe())),
        _ => "*",
    };

    static bool SameValue(JsonElement expected, JsonElement actual)
    {
        if (expected.ValueKind != actual.ValueKind) return false;
        return expected.ValueKind switch
        {
            JsonValueKind.String => expected.GetString() == actual.GetString(),
            JsonValueKind.Number => expected.GetDouble() == actual.GetDouble(),
            JsonValueKind.True or JsonValueKind.False or JsonValueKind.Null => true,
            _ => expected.GetRawText() == actual.GetRawText(),
        };
    }
}

internal sealed record ExpectedCall(string Name, KeyValuePair<string, ArgumentMatcher>[] Arguments)
{
    public string Describe() => $"{Name}({string.Join(", ", Arguments.Select(argument => $"{argument.Key}={argument.Value.Describe()}"))})";
}

internal sealed record ObservedCall(string Name, Dictionary<string, JsonElement> Arguments)
{
    public string Describe() => $"{Name}({string.Join(", ", Arguments.Select(argument => $"{argument.Key}={argument.Value.GetRawText()}"))})";

    public static bool TryParse(string callsJson, out ObservedCall[] calls)
    {
        calls = Array.Empty<ObservedCall>();
        if (callsJson.Length == 0) return true;
        try
        {
            using var document = JsonDocument.Parse(callsJson);
            if (document.RootElement.ValueKind != JsonValueKind.Array) return false;
            var parsed = new List<ObservedCall>();
            foreach (var call in document.RootElement.EnumerateArray())
            {
                var arguments = new Dictionary<string, JsonElement>();
                if (call.TryGetProperty("arguments", out var argumentObject) && argumentObject.ValueKind == JsonValueKind.Object)
                    foreach (var property in argumentObject.EnumerateObject()) arguments[property.Name] = property.Value.Clone();
                parsed.Add(new ObservedCall(call.GetProperty("name").GetString() ?? "", arguments));
            }
            calls = parsed.ToArray();
            return true;
        }
        catch (Exception e) when (e is JsonException or KeyNotFoundException or InvalidOperationException)
        {
            return false;
        }
    }
}

internal readonly record struct CaseScore(bool Exact, int NamesMatched, int ArgumentsExpected, int ArgumentsMatched)
{
    public static CaseScore Of(ExpectedCall[] expected, ObservedCall[] observed)
    {
        var namesMatched = 0;
        var argumentsExpected = 0;
        var argumentsMatched = 0;
        var everyCallExact = expected.Length == observed.Length;
        for (var i = 0; i < expected.Length; i++)
        {
            argumentsExpected += expected[i].Arguments.Length;
            var namesAgree = i < observed.Length && observed[i].Name == expected[i].Name;
            if (!namesAgree)
            {
                everyCallExact = false;
                continue;
            }
            namesMatched++;
            var matchedHere = 0;
            foreach (var (key, matcher) in expected[i].Arguments)
                if (observed[i].Arguments.TryGetValue(key, out var value) && matcher.Matches(value)) matchedHere++;
            argumentsMatched += matchedHere;
            if (matchedHere != expected[i].Arguments.Length || observed[i].Arguments.Count != expected[i].Arguments.Length) everyCallExact = false;
        }
        return new CaseScore(everyCallExact, namesMatched, argumentsExpected, argumentsMatched);
    }
}
