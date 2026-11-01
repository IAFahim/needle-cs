using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace Needle.Cli;

internal sealed record EvalResult(
    EvalCase Case,
    ReplyStatus Status,
    FailureReason Failure,
    float Confidence,
    string CallsJson,
    ObservedCall[] Calls,
    string Reasoning,
    string Response,
    CaseScore Score,
    double TotalMs,
    double FirstTokenMs,
    int PromptTokens,
    int DecodeTokens)
{
    public bool HasCalls => Status is ReplyStatus.Called or ReplyStatus.Refused;

    public bool Executed => Status == ReplyStatus.Called;

    public bool ExecutedAt(float threshold) => HasCalls && Confidence >= threshold;

    public bool CorrectWhenExecuted(bool executed) =>
        Status != ReplyStatus.Failed && (Case.InDomain ? executed && Score.Exact : !executed);

    public bool Correct => CorrectWhenExecuted(Executed);

    public string DescribeOutcome() => Status switch
    {
        ReplyStatus.Called or ReplyStatus.Refused => string.Join(" ; ", Calls.Select(call => call.Describe())) + (Status == ReplyStatus.Refused ? " [refused]" : ""),
        ReplyStatus.Responded => "(no call)",
        _ => $"(failed: {Failure})",
    };
}

internal sealed record EvalRun(string Label, EvalPolicy Policy, IReadOnlyList<EvalResult> Results);

internal static class EvalRunner
{
    const int CallsBufferBytes = 16384;
    const int TextBufferBytes = 8192;

    public static EvalRun Run(Model model, EvalDataset dataset, EvalPolicy policy, string label, TextWriter? progress)
    {
        var results = new List<EvalResult>(dataset.Cases.Count);
        var options = policy.ToOptions();
        foreach (var group in dataset.Cases.GroupBy(item => item.ToolSet))
        {
            using var session = new NeedleSession(model, dataset.ToolSets[group.Key], options);
            var done = 0;
            foreach (var item in group)
            {
                results.Add(Evaluate(session, item));
                progress?.Write($"\r{label}: {group.Key} {++done}/{group.Count()}   ");
            }
            progress?.WriteLine();
        }
        var order = dataset.Cases.Select((item, index) => (item.Id, index)).ToDictionary(entry => entry.Id, entry => entry.index);
        results.Sort((a, b) => order[a.Case.Id].CompareTo(order[b.Case.Id]));
        return new EvalRun(label, policy, results);
    }

    static EvalResult Evaluate(NeedleSession session, EvalCase item)
    {
        var calls = new byte[CallsBufferBytes];
        var reasoning = new byte[TextBufferBytes];
        var response = new byte[TextBufferBytes];
        session.Reset();
        var reply = session.Complete(Encoding.UTF8.GetBytes(item.Query), calls, reasoning, response);
        var callsJson = reply.HasCalls ? Encoding.UTF8.GetString(calls, 0, reply.Utf8Length) : "";
        var malformed = !ObservedCall.TryParse(callsJson, out var observed);
        if (malformed) observed = Array.Empty<ObservedCall>();
        var status = malformed ? ReplyStatus.Failed : reply.HasCalls && reply.CallCount == 0 ? ReplyStatus.Responded : reply.Status;
        var ticksToMs = 1000.0 / Stopwatch.Frequency;
        return new EvalResult(
            item, status, malformed ? FailureReason.MalformedCalls : reply.Failure,
            reply.Confidence, callsJson, observed,
            Encoding.UTF8.GetString(reasoning, 0, reply.ReasoningLength).Trim(), Encoding.UTF8.GetString(response, 0, reply.ResponseLength).Trim(),
            CaseScore.Of(item.Expected, observed),
            reply.ElapsedTicks * ticksToMs, reply.FirstTokenTicks * ticksToMs, reply.PromptTokens, reply.DecodeTokens);
    }
}
