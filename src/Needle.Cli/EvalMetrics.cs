using System;
using System.Collections.Generic;
using System.Linq;

namespace Needle.Cli;

internal sealed record EvalSummary(
    string Name, int Prompts, int InDomain, int OutOfDomain, int ExactCalls,
    int ToolsRouted, int ToolsExpected, int ArgumentsMatched, int ArgumentsExpected,
    int NoCallCorrect, int RefusedInDomain, int FalseCalls, int Failures,
    double MedianMs, double P95Ms, double MedianFirstTokenMs)
{
    public double ExactAccuracy => Ratio(ExactCalls, InDomain);
    public double ToolAccuracy => Ratio(ToolsRouted, ToolsExpected);
    public double ArgumentAccuracy => Ratio(ArgumentsMatched, ArgumentsExpected);
    public double NoCallAccuracy => Ratio(NoCallCorrect, OutOfDomain);
    public double OverallAccuracy => Ratio(ExactCalls + NoCallCorrect, Prompts);

    static double Ratio(int part, int whole) => whole == 0 ? double.NaN : (double)part / whole;
}

internal readonly record struct CalibrationBin(double Low, double High, int Count, int Correct, double MeanConfidence)
{
    public double Accuracy => Count == 0 ? double.NaN : (double)Correct / Count;
}

internal readonly record struct CurvePoint(float Threshold, double Accuracy, double Coverage, double Precision, double FalseCallRate, int WrongExecuted);

internal static class EvalMetrics
{
    public const int BinCount = 10;
    public static readonly float[] Thresholds = { 0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f };

    public static EvalSummary Summarize(string name, IReadOnlyList<EvalResult> results)
    {
        var inDomain = results.Where(r => r.Case.InDomain).ToList();
        var outOfDomain = results.Where(r => !r.Case.InDomain).ToList();
        var produced = inDomain.Where(r => r.HasCalls).ToList();
        var latencies = results.Select(r => r.TotalMs).OrderBy(v => v).ToArray();
        var firstTokens = results.Select(r => r.FirstTokenMs).OrderBy(v => v).ToArray();
        return new EvalSummary(
            name, results.Count, inDomain.Count, outOfDomain.Count,
            inDomain.Count(r => r.Correct),
            produced.Sum(r => r.Score.NamesMatched), inDomain.Sum(r => r.Case.Expected.Length),
            produced.Sum(r => r.Score.ArgumentsMatched), inDomain.Sum(r => r.Score.ArgumentsExpected),
            outOfDomain.Count(r => r.Correct), inDomain.Count(r => r.Status == ReplyStatus.Refused),
            outOfDomain.Count(r => r.Executed), results.Count(r => r.Status == ReplyStatus.Failed),
            Percentile(latencies, 0.5), Percentile(latencies, 0.95), Percentile(firstTokens, 0.5));
    }

    public static IReadOnlyList<EvalSummary> BySet(IReadOnlyList<EvalResult> results) =>
        results.GroupBy(r => r.Case.ToolSet).Select(group => Summarize(group.Key, group.ToList())).ToList();

    public static CalibrationBin[] Calibrate(IReadOnlyList<EvalResult> results)
    {
        var bins = new CalibrationBin[BinCount];
        for (var b = 0; b < BinCount; b++)
        {
            var members = results.Where(r => r.HasCalls && BinOf(r.Confidence) == b).ToList();
            bins[b] = new CalibrationBin(
                (double)b / BinCount, (double)(b + 1) / BinCount, members.Count,
                members.Count(r => r.Case.InDomain && r.Score.Exact),
                members.Count == 0 ? double.NaN : members.Average(r => r.Confidence));
        }
        return bins;
    }

    public static double ExpectedCalibrationError(CalibrationBin[] bins)
    {
        var total = bins.Sum(b => b.Count);
        return total == 0 ? double.NaN : bins.Where(b => b.Count > 0).Sum(b => Math.Abs(b.Accuracy - b.MeanConfidence) * b.Count) / total;
    }

    public static CurvePoint[] Curve(IReadOnlyList<EvalResult> results)
    {
        var inDomain = results.Where(r => r.Case.InDomain).ToList();
        var outOfDomain = results.Where(r => !r.Case.InDomain).ToList();
        return Thresholds.Select(threshold =>
        {
            var executed = results.Where(r => r.ExecutedAt(threshold)).ToList();
            var exactExecuted = executed.Count(r => r.Case.InDomain && r.Score.Exact);
            return new CurvePoint(
                threshold,
                results.Count == 0 ? double.NaN : (double)results.Count(r => r.CorrectWhenExecuted(r.ExecutedAt(threshold))) / results.Count,
                inDomain.Count == 0 ? double.NaN : (double)inDomain.Count(r => r.ExecutedAt(threshold)) / inDomain.Count,
                executed.Count == 0 ? double.NaN : (double)exactExecuted / executed.Count,
                outOfDomain.Count == 0 ? double.NaN : (double)outOfDomain.Count(r => r.ExecutedAt(threshold)) / outOfDomain.Count,
                executed.Count - exactExecuted);
        }).ToArray();
    }

    public static float RecommendThreshold(CurvePoint[] curve) =>
        curve.Where(point => point.Threshold > 0).OrderByDescending(point => point.Accuracy).ThenBy(point => point.Threshold).First().Threshold;

    public static IReadOnlyList<(EvalResult Baseline, EvalResult Candidate)> Flips(EvalRun baseline, EvalRun candidate)
    {
        var byId = candidate.Results.ToDictionary(r => r.Case.Id);
        return baseline.Results
            .Where(r => byId.TryGetValue(r.Case.Id, out var other) && other.Correct != r.Correct)
            .Select(r => (r, byId[r.Case.Id]))
            .ToList();
    }

    static int BinOf(float confidence) => Math.Clamp((int)(confidence * BinCount), 0, BinCount - 1);

    static double Percentile(double[] sorted, double fraction) =>
        sorted.Length == 0 ? double.NaN : sorted[Math.Max(0, (int)Math.Ceiling(fraction * sorted.Length) - 1)];
}
