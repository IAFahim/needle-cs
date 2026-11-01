using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace Needle.Cli;

internal static class EvalText
{
    const int MissLimit = 30;

    public static string Render(EvalRun run, EvalPolicy defaults, bool allMisses)
    {
        var text = new StringBuilder();
        var overall = EvalMetrics.Summarize("overall", run.Results);
        text.AppendLine($"== {run.Label} [{EvalPolicies.Describe(run.Policy, defaults)}]: {overall.Prompts} prompts, {overall.InDomain} in-domain, {overall.OutOfDomain} out-of-domain ==");
        text.AppendLine("set         prompts   exact    tool    args  no-call  refused  false-call  failed  p50 ms  p95 ms  ttft ms");
        foreach (var summary in EvalMetrics.BySet(run.Results).Append(overall)) text.AppendLine(SummaryRow(summary));
        text.AppendLine($"overall accuracy {Percent(overall.OverallAccuracy)} (exact calls plus correct no-calls over all prompts). tool and args columns score the model output before refusal gating.");
        text.AppendLine();
        AppendCalibration(text, run.Results);
        AppendCurve(text, run.Results, run.Policy.RefuseBelow);
        AppendMisses(text, run.Results, allMisses);
        return text.ToString();
    }

    public static string RenderComparison(EvalRun baseline, EvalRun candidate, EvalPolicy defaults)
    {
        var text = new StringBuilder();
        text.AppendLine($"== delta: {candidate.Label} [{EvalPolicies.Describe(candidate.Policy, defaults)}] versus {baseline.Label} [{EvalPolicies.Describe(baseline.Policy, defaults)}] ==");
        text.AppendLine("set         metric          baseline  candidate    delta");
        var baselineSets = EvalMetrics.BySet(baseline.Results).Append(EvalMetrics.Summarize("overall", baseline.Results)).ToList();
        var candidateSets = EvalMetrics.BySet(candidate.Results).Append(EvalMetrics.Summarize("overall", candidate.Results)).ToList();
        for (var i = 0; i < baselineSets.Count; i++)
        {
            var before = baselineSets[i];
            var after = candidateSets[i];
            text.AppendLine(DeltaRow(before.Name, "exact", before.ExactAccuracy, after.ExactAccuracy));
            text.AppendLine(DeltaRow("", "args", before.ArgumentAccuracy, after.ArgumentAccuracy));
            text.AppendLine(DeltaRow("", "no-call", before.NoCallAccuracy, after.NoCallAccuracy));
            text.AppendLine(DeltaRow("", "overall", before.OverallAccuracy, after.OverallAccuracy));
            text.AppendLine($"{"",-11} {"p50 ms",-14} {before.MedianMs,9:F0} {after.MedianMs,10:F0} {after.MedianMs - before.MedianMs,+8:F0}");
        }
        var flips = EvalMetrics.Flips(baseline, candidate);
        var lost = flips.Where(flip => flip.Baseline.Correct).ToList();
        var gained = flips.Where(flip => flip.Candidate.Correct).ToList();
        text.AppendLine($"regressions {lost.Count}, improvements {gained.Count}");
        foreach (var (before, after) in lost) text.AppendLine($"  regressed {before.Case.Id} \"{before.Case.Query}\": {before.DescribeOutcome()} -> {after.DescribeOutcome()}");
        foreach (var (before, after) in gained) text.AppendLine($"  improved  {before.Case.Id} \"{before.Case.Query}\": {before.DescribeOutcome()} -> {after.DescribeOutcome()}");
        return text.ToString();
    }

    static string SummaryRow(EvalSummary s) =>
        $"{s.Name,-11} {s.Prompts,7} {Percent(s.ExactAccuracy),7} {Percent(s.ToolAccuracy),7} {Percent(s.ArgumentAccuracy),7} {Percent(s.NoCallAccuracy),8} {s.RefusedInDomain,8} {s.FalseCalls,11} {s.Failures,7} {s.MedianMs,7:F0} {s.P95Ms,7:F0} {s.MedianFirstTokenMs,8:F0}";

    static string DeltaRow(string set, string metric, double before, double after) =>
        $"{set,-11} {metric,-14} {Percent(before),9} {Percent(after),10} {(double.IsNaN(before) || double.IsNaN(after) ? "-" : $"{(after - before) * 100:+0.0;-0.0;0.0} pp"),8}";

    static void AppendCalibration(StringBuilder text, IReadOnlyList<EvalResult> results)
    {
        var bins = EvalMetrics.Calibrate(results);
        text.AppendLine("calibration: replies that carry a call, by confidence (accurate = in-domain and exactly right)");
        text.AppendLine("confidence      calls  accurate  accuracy  mean confidence");
        foreach (var bin in bins)
            text.AppendLine($"{bin.Low:0.0}-{bin.High:0.0}  {bin.Count,9} {bin.Correct,9} {Percent(bin.Accuracy),9} {(bin.Count == 0 ? "-" : bin.MeanConfidence.ToString("F3")),16}");
        text.AppendLine($"expected calibration error {EvalMetrics.ExpectedCalibrationError(bins):F3}");
        text.AppendLine();
    }

    static void AppendCurve(StringBuilder text, IReadOnlyList<EvalResult> results, float configuredThreshold)
    {
        var curve = EvalMetrics.Curve(results);
        text.AppendLine("RefuseBelow curve: what the same replies would score at each threshold (calls below it are refused)");
        text.AppendLine("RefuseBelow  accuracy  coverage  precision  false-call  wrong executed");
        foreach (var point in curve)
            text.AppendLine($"{point.Threshold,11:0.0} {Percent(point.Accuracy),9} {Percent(point.Coverage),9} {Percent(point.Precision),10} {Percent(point.FalseCallRate),11} {point.WrongExecuted,15}{(Math.Abs(point.Threshold - configuredThreshold) < 1e-6 ? "  <- configured" : "")}");
        text.AppendLine($"best overall accuracy at RefuseBelow {EvalMetrics.RecommendThreshold(curve):0.0}");
        text.AppendLine();
    }

    static void AppendMisses(StringBuilder text, IReadOnlyList<EvalResult> results, bool all)
    {
        var misses = results.Where(r => !r.Correct).ToList();
        text.AppendLine($"misses: {misses.Count}{(!all && misses.Count > MissLimit ? $" (showing {MissLimit}, --verbose lists all)" : "")}");
        foreach (var miss in misses.Take(all ? int.MaxValue : MissLimit))
            text.AppendLine($"  {miss.Case.Id} \"{miss.Case.Query}\" conf {(double.IsNaN(miss.Confidence) ? "-" : miss.Confidence.ToString("F2"))}\n      expected {miss.Case.DescribeExpected()}\n      got      {miss.DescribeOutcome()}");
    }

    public static string Percent(double value) => double.IsNaN(value) ? "-" : $"{value * 100:F1}%";
}

internal static class EvalJson
{
    public static string Render(string model, string dataset, EvalPolicy defaults, IReadOnlyList<EvalRun> runs)
    {
        using var stream = new MemoryStream();
        using (var json = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            json.WriteStartObject();
            json.WriteString("model", model);
            json.WriteString("dataset", dataset);
            json.WriteStartArray("runs");
            foreach (var run in runs) WriteRun(json, run, defaults);
            json.WriteEndArray();
            if (runs.Count == 2) WriteFlips(json, runs[0], runs[1]);
            json.WriteEndObject();
        }
        return Encoding.UTF8.GetString(stream.ToArray());
    }

    static void WriteRun(Utf8JsonWriter json, EvalRun run, EvalPolicy defaults)
    {
        json.WriteStartObject();
        json.WriteString("label", run.Label);
        json.WriteString("policy", EvalPolicies.Describe(run.Policy, defaults));
        json.WriteStartObject("settings");
        foreach (var setting in EvalPolicies.Settings) json.WriteString(setting.Name, setting.Read(run.Policy));
        json.WriteEndObject();
        json.WriteStartObject("overall");
        WriteSummary(json, EvalMetrics.Summarize("overall", run.Results));
        json.WriteEndObject();
        json.WriteStartArray("sets");
        foreach (var summary in EvalMetrics.BySet(run.Results))
        {
            json.WriteStartObject();
            json.WriteString("name", summary.Name);
            WriteSummary(json, summary);
            json.WriteEndObject();
        }
        json.WriteEndArray();
        var bins = EvalMetrics.Calibrate(run.Results);
        json.WriteStartArray("calibration");
        foreach (var bin in bins)
        {
            json.WriteStartObject();
            json.WriteNumber("low", bin.Low);
            json.WriteNumber("high", bin.High);
            json.WriteNumber("calls", bin.Count);
            json.WriteNumber("accurate", bin.Correct);
            WriteNumber(json, "accuracy", bin.Accuracy);
            WriteNumber(json, "meanConfidence", bin.MeanConfidence);
            json.WriteEndObject();
        }
        json.WriteEndArray();
        WriteNumber(json, "expectedCalibrationError", EvalMetrics.ExpectedCalibrationError(bins));
        var curve = EvalMetrics.Curve(run.Results);
        json.WriteStartArray("refuseBelowCurve");
        foreach (var point in curve)
        {
            json.WriteStartObject();
            json.WriteNumber("threshold", point.Threshold);
            WriteNumber(json, "accuracy", point.Accuracy);
            WriteNumber(json, "coverage", point.Coverage);
            WriteNumber(json, "precision", point.Precision);
            WriteNumber(json, "falseCallRate", point.FalseCallRate);
            json.WriteNumber("wrongExecuted", point.WrongExecuted);
            json.WriteEndObject();
        }
        json.WriteEndArray();
        json.WriteNumber("recommendedRefuseBelow", EvalMetrics.RecommendThreshold(curve));
        json.WriteStartArray("cases");
        foreach (var result in run.Results) WriteCase(json, result);
        json.WriteEndArray();
        json.WriteEndObject();
    }

    static void WriteSummary(Utf8JsonWriter json, EvalSummary s)
    {
        json.WriteNumber("prompts", s.Prompts);
        json.WriteNumber("inDomain", s.InDomain);
        json.WriteNumber("outOfDomain", s.OutOfDomain);
        json.WriteNumber("exactCalls", s.ExactCalls);
        WriteNumber(json, "exactAccuracy", s.ExactAccuracy);
        WriteNumber(json, "toolAccuracy", s.ToolAccuracy);
        WriteNumber(json, "argumentAccuracy", s.ArgumentAccuracy);
        WriteNumber(json, "noCallAccuracy", s.NoCallAccuracy);
        WriteNumber(json, "overallAccuracy", s.OverallAccuracy);
        json.WriteNumber("refusedInDomain", s.RefusedInDomain);
        json.WriteNumber("falseCalls", s.FalseCalls);
        json.WriteNumber("failures", s.Failures);
        WriteNumber(json, "medianMs", s.MedianMs);
        WriteNumber(json, "p95Ms", s.P95Ms);
        WriteNumber(json, "medianFirstTokenMs", s.MedianFirstTokenMs);
    }

    static void WriteCase(Utf8JsonWriter json, EvalResult result)
    {
        json.WriteStartObject();
        json.WriteString("id", result.Case.Id);
        json.WriteString("toolSet", result.Case.ToolSet);
        json.WriteString("query", result.Case.Query);
        json.WriteString("expected", result.Case.DescribeExpected());
        json.WriteString("status", result.Status.ToString());
        json.WriteString("got", result.DescribeOutcome());
        WriteNumber(json, "confidence", result.Confidence);
        json.WriteBoolean("correct", result.Correct);
        json.WriteBoolean("exact", result.Case.InDomain && result.Score.Exact);
        WriteNumber(json, "totalMs", result.TotalMs);
        json.WriteNumber("decodeTokens", result.DecodeTokens);
        json.WriteEndObject();
    }

    static void WriteFlips(Utf8JsonWriter json, EvalRun baseline, EvalRun candidate)
    {
        json.WriteStartArray("flips");
        foreach (var (before, after) in EvalMetrics.Flips(baseline, candidate))
        {
            json.WriteStartObject();
            json.WriteString("id", before.Case.Id);
            json.WriteString("query", before.Case.Query);
            json.WriteString("change", after.Correct ? "improved" : "regressed");
            json.WriteString("baseline", before.DescribeOutcome());
            json.WriteString("candidate", after.DescribeOutcome());
            json.WriteEndObject();
        }
        json.WriteEndArray();
    }

    static void WriteNumber(Utf8JsonWriter json, string name, double value)
    {
        if (double.IsFinite(value)) json.WriteNumber(name, value);
        else json.WriteNull(name);
    }
}
