using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using Xunit;

namespace Needle.Tests;

public sealed class SpanApiTests : IDisposable
{
    const float Tolerance = 1e-6f;

    readonly Model model;

    public SpanApiTests()
    {
        Assets.Require();
        model = Model.Open(Assets.Model);
    }

    public void Dispose() => model.Dispose();

    sealed record Outcome(ReplyStatus Status, string Calls, string Reasoning, string Response, float Confidence, float Head, float Decode);

    static string CallsJson(IEnumerable<FunctionCall> calls)
    {
        var builder = new StringBuilder("[");
        var first = true;
        foreach (var call in calls)
        {
            if (!first) builder.Append(',');
            first = false;
            builder.Append("{\"name\":");
            JsonValue.WriteString(builder, call.Name);
            builder.Append(",\"arguments\":").Append(call.ArgumentsJson).Append('}');
        }
        return builder.Append(']').ToString();
    }

    static Outcome FromResult(CallResult result)
    {
        Assert.True(result.Success, result.Error);
        if (result.FunctionCalls.Count > 0)
            return new Outcome(ReplyStatus.Called, CallsJson(result.FunctionCalls), result.Reasoning, "", result.Confidence!.Value, result.HeadConfidence!.Value, result.DecodeConfidence!.Value);
        if (result.SuppressedCalls.Count > 0)
            return new Outcome(ReplyStatus.Refused, CallsJson(result.SuppressedCalls), result.Reasoning, "", result.Confidence!.Value, result.HeadConfidence!.Value, result.DecodeConfidence!.Value);
        return new Outcome(ReplyStatus.Responded, "", result.Reasoning, result.Response, result.Confidence!.Value, float.NaN, float.NaN);
    }

    static Outcome FromReply(in Reply reply, byte[] calls, byte[] reasoning, byte[] response)
    {
        Assert.NotEqual(ReplyStatus.Failed, reply.Status);
        Assert.Equal(FailureReason.None, reply.Failure);
        Assert.Equal(ReplyFlags.None, reply.Flags);
        return new Outcome(reply.Status, reply.HasCalls ? Encoding.UTF8.GetString(calls, 0, reply.Utf8Length) : "",
            Encoding.UTF8.GetString(reasoning, 0, reply.ReasoningLength).Trim(), Encoding.UTF8.GetString(response, 0, reply.ResponseLength).Trim(),
            reply.Confidence, reply.HeadConfidence, reply.DecodeConfidence);
    }

    static void AssertSame(Outcome expected, Outcome actual, string label)
    {
        Assert.True(expected.Status == actual.Status, $"{label}: status {expected.Status} vs {actual.Status}");
        Assert.True(expected.Calls == actual.Calls, $"{label}: calls {expected.Calls} vs {actual.Calls}");
        Assert.True(expected.Reasoning == actual.Reasoning, $"{label}: reasoning {expected.Reasoning} vs {actual.Reasoning}");
        Assert.True(expected.Response == actual.Response, $"{label}: response {expected.Response} vs {actual.Response}");
        Assert.InRange(Math.Abs(expected.Confidence - actual.Confidence), 0f, Tolerance);
        if (expected.Status != ReplyStatus.Responded)
        {
            Assert.InRange(Math.Abs(expected.Head - actual.Head), 0f, Tolerance);
            Assert.InRange(Math.Abs(expected.Decode - actual.Decode), 0f, Tolerance);
        }
    }

    static Outcome RunSpan(NeedleSession session, SessionStep step)
    {
        var calls = new byte[8192];
        var reasoning = new byte[8192];
        var response = new byte[8192];
        var input = Encoding.UTF8.GetBytes(step.Input);
        var reply = step.Kind == StepKind.Query
            ? session.Complete(input, calls, reasoning, response)
            : session.CompleteWithToolResult(input, calls, reasoning, response);
        return FromReply(reply, calls, reasoning, response);
    }

    void AssertCasesMatchLegacy(int threads, Func<SessionCase, bool> include)
    {
        foreach (var sessionCase in SessionCases.All(threads).Where(include))
        {
            var tools = sessionCase.ToolsJson ?? SessionCases.DefaultTools();
            using var legacy = new LegacySession(model, tools, sessionCase.Options);
            using var span = new NeedleSession(model, tools, sessionCase.Options);
            using var text = new NeedleSession(model, tools, sessionCase.Options);
            var index = 0;
            foreach (var step in sessionCase.Steps)
            {
                var label = $"{sessionCase.Name}[{index++}] threads={threads}";
                if (step.Kind == StepKind.Reset)
                {
                    legacy.Reset();
                    span.Reset();
                    text.Reset();
                    continue;
                }
                var expected = FromResult(step.Kind == StepKind.Query ? legacy.Complete(step.Input) : legacy.CompleteWithToolResult(step.Input));
                AssertSame(expected, RunSpan(span, step), label + " span");
                AssertSame(expected, FromResult(step.Kind == StepKind.Query ? text.Complete(step.Input) : text.CompleteWithToolResult(step.Input)), label + " string");
            }
        }
    }

    [Fact]
    public void EveryExistingPromptYieldsTheLegacyResultsAtFourThreads() =>
        AssertCasesMatchLegacy(4, c => c.Name != "retrieval");

    [Fact]
    public void SingleThreadedRunsYieldTheLegacyResultsToo() =>
        AssertCasesMatchLegacy(1, c => c.Name is "weather" or "chain" or "toolResult" or "nestedAliases");

    [Fact]
    public void RetrievalSessionsAgreeBetweenSpanAndStringApis()
    {
        var sessionCase = SessionCases.All(4).Single(c => c.Name == "retrieval");
        using var span = new NeedleSession(model, sessionCase.ToolsJson!, sessionCase.Options);
        using var text = new NeedleSession(model, sessionCase.ToolsJson!, sessionCase.Options);
        var step = sessionCase.Steps[0];
        var fromSpan = RunSpan(span, step);
        var fromText = FromResult(text.Complete(step.Input));
        AssertSame(fromText, fromSpan, "retrieval");
        Assert.Contains("\"set_alarm\"", fromSpan.Calls);
        span.Reset();
        AssertSame(fromSpan, RunSpan(span, step), "retrieval again");
    }

    [Fact]
    public void ReplyCarriesCountsLengthsAndTimings()
    {
        using var session = new NeedleSession(model, SessionCases.DefaultTools(), new NeedleOptions { Threads = 4, AutoDate = false });
        var calls = new byte[4096];
        var reasoning = new byte[4096];
        var reply = session.Complete("set a timer for 12 minutes and play some Miles Davis"u8, calls, reasoning, default);
        Assert.Equal(ReplyStatus.Called, reply.Status);
        Assert.Equal(2, reply.CallCount);
        Assert.Equal("[{\"name\":\"set_timer\",\"arguments\":{\"minutes\":12}},{\"name\":\"play_music\",\"arguments\":{\"query\":\"Miles Davis\"}}]", Encoding.UTF8.GetString(calls, 0, reply.Utf8Length));
        Assert.True(reply.PromptTokens > 0 && reply.DecodeTokens > reply.CallCount);
        Assert.InRange(reply.FirstTokenTicks, 1, reply.PrefillTicks + reply.ElapsedTicks);
        Assert.InRange(reply.PrefillTicks, 1, reply.ElapsedTicks);
        Assert.Equal(Math.Min(reply.HeadConfidence, reply.DecodeConfidence), reply.Confidence);
        Assert.Contains("Miles Davis", Encoding.UTF8.GetString(reasoning, 0, reply.ReasoningLength));
        Assert.Equal(0, reply.ResponseLength);
    }

    [Fact]
    public void ToolResultsAreAnsweredThroughTheSpanApi()
    {
        using var session = new NeedleSession(model, SessionCases.DefaultTools(), new NeedleOptions { Threads = 4, AutoDate = false });
        var calls = new byte[4096];
        var response = new byte[4096];
        Assert.Equal(ReplyStatus.Called, session.Complete("what's it like in Lagos right now?"u8, calls).Status);
        var reply = session.CompleteWithToolResult("{\"city\":\"Lagos\",\"temp_c\":27,\"sky\":\"clear\"}"u8, calls, default, response);
        Assert.Equal(ReplyStatus.Responded, reply.Status);
        Assert.Equal(0, reply.CallCount);
        Assert.Contains("27", Encoding.UTF8.GetString(response, 0, reply.ResponseLength));
    }

    [Fact]
    public void CamelCaseAliasesAreRestoredInNestedArguments()
    {
        var sessionCase = SessionCases.All(4).Single(c => c.Name == "nestedAliases");
        using var session = new NeedleSession(model, sessionCase.ToolsJson!, sessionCase.Options);
        var calls = new byte[4096];
        var reply = session.Complete("move Lyra to x 120 y 45 and run fast, passing through bridge and gate"u8, calls);
        Assert.Equal(ReplyStatus.Called, reply.Status);
        var json = JsonValue.Parse(Encoding.UTF8.GetString(calls, 0, reply.Utf8Length));
        var call = Assert.Single(json.Items);
        Assert.Equal("moveCharacter", call.GetString("name"));
        var arguments = call.Get("arguments")!;
        Assert.Equal("Lyra", arguments.GetString("characterName"));
        Assert.Equal("120", arguments.Get("targetPosition")!.Get("xCoord")!.Text);
        Assert.Equal("45", arguments.Get("targetPosition")!.Get("yCoord")!.Text);
        Assert.DoesNotContain("character_name", Encoding.UTF8.GetString(calls, 0, reply.Utf8Length));
    }

    [Fact]
    public void FailuresReturnStatusesInsteadOfThrowing()
    {
        using var session = new NeedleSession(model, SessionCases.DefaultTools(), new NeedleOptions { Threads = 4, AutoDate = false });
        var calls = new byte[4096];

        var empty = session.Complete(ReadOnlySpan<byte>.Empty, calls);
        Assert.Equal(ReplyStatus.Failed, empty.Status);
        Assert.Equal(FailureReason.EmptyInput, empty.Failure);

        var tooLong = session.Complete(new byte[session.MaxInputBytes + 1], calls);
        Assert.Equal(ReplyStatus.Failed, tooLong.Status);
        Assert.Equal(FailureReason.InputTooLong, tooLong.Failure);

        var tiny = session.Complete("what's it like in Lagos right now?"u8, new byte[4]);
        Assert.Equal(ReplyStatus.Failed, tiny.Status);
        Assert.Equal(FailureReason.CallsBufferTooSmall, tiny.Failure);
        Assert.Equal("[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Lagos\"}}]".Length, tiny.Utf8Length);

        session.Reset();
        var recovered = session.Complete("what's it like in Lagos right now?"u8, calls);
        Assert.Equal(ReplyStatus.Called, recovered.Status);
        Assert.Equal(tiny.Utf8Length, recovered.Utf8Length);
        Assert.Equal(tiny.Confidence, recovered.Confidence);

        var textFailure = session.Complete(string.Empty);
        Assert.False(textFailure.Success);
        Assert.NotNull(textFailure.Error);
        Assert.False(session.Complete(new string('a', session.MaxInputBytes + 1)).Success);
    }

    [Fact]
    public void ExceedingTheCallTokenBudgetFailsWithoutThrowingAndLeavesTheSessionUsable()
    {
        var starved = new NeedleOptions { Threads = 4, AutoDate = false, MaxCallTokens = 3 };
        using var session = new NeedleSession(model, SessionCases.DefaultTools(), starved);
        var calls = new byte[4096];
        var primed = session.ContextUsed;
        var reply = session.Complete("what's it like in Lagos right now?"u8, calls);
        Assert.Equal(ReplyStatus.Failed, reply.Status);
        Assert.Equal(FailureReason.CallTokenBudgetExceeded, reply.Failure);
        Assert.True(reply.ReasoningLength == 0 && reply.DecodeTokens > 0);
        var result = session.Complete("what's it like in Lagos right now?");
        Assert.False(result.Success);
        Assert.Equal("call exceeded the token budget", result.Error);
        Assert.Equal(primed, session.ContextUsed);
    }

    [Fact]
    public void AFrameTokenBudgetYieldsAfterExactlyThatManyTokensAndLeavesTheSessionUsable()
    {
        var budgeted = new NeedleOptions { Threads = 4, AutoDate = false, FrameBudget = FrameBudget.FromTokens(3) };
        using var session = new NeedleSession(model, SessionCases.DefaultTools(), budgeted);
        var calls = new byte[4096];
        var primed = session.ContextUsed;
        var reply = session.Complete("what's it like in Lagos right now?"u8, calls);
        Assert.Equal(ReplyStatus.Failed, reply.Status);
        Assert.Equal(FailureReason.FrameBudgetExceeded, reply.Failure);
        Assert.Equal(3, reply.DecodeTokens);
        var result = session.Complete("what's it like in Lagos right now?");
        Assert.False(result.Success);
        Assert.Equal("the frame budget ran out", result.Error);
        Assert.Equal(primed, session.ContextUsed);
    }

    [Fact]
    public void AGenerousFrameBudgetLeavesTheCallBitIdentical()
    {
        var options = new NeedleOptions { Threads = 4, AutoDate = false, FrameBudget = FrameBudget.FromTokens(10000).WithMaxMillisecondsPerSlice(60000) };
        using var budgeted = new NeedleSession(model, SessionCases.DefaultTools(), options);
        using var plain = new NeedleSession(model, SessionCases.DefaultTools(), new NeedleOptions { Threads = 4, AutoDate = false });
        var budgetedCalls = new byte[4096];
        var plainCalls = new byte[4096];
        var capped = budgeted.Complete("what's it like in Lagos right now?"u8, budgetedCalls);
        var free = plain.Complete("what's it like in Lagos right now?"u8, plainCalls);
        Assert.Equal(free.Status, capped.Status);
        Assert.Equal(free.Failure, capped.Failure);
        Assert.Equal(free.DecodeTokens, capped.DecodeTokens);
        Assert.Equal(free.Utf8Length, capped.Utf8Length);
        Assert.Equal(free.Confidence, capped.Confidence);
        Assert.Equal(Encoding.UTF8.GetString(plainCalls, 0, free.Utf8Length), Encoding.UTF8.GetString(budgetedCalls, 0, capped.Utf8Length));
    }

    [Fact]
    public void ContextExhaustionFailsWithoutThrowing()
    {
        using var session = new NeedleSession(model, SessionCases.DefaultTools(), new NeedleOptions { Threads = 4, AutoDate = false, Context = 300 });
        var calls = new byte[4096];
        var statuses = new List<Reply>();
        for (var i = 0; i < 12; i++) statuses.Add(session.Complete("set a timer for 12 minutes and play some Miles Davis"u8, calls));
        Assert.Contains(statuses, r => r.Failure == FailureReason.ContextExhausted);
        session.Reset();
        Assert.Equal(ReplyStatus.Called, session.Complete("what's it like in Lagos right now?"u8, calls).Status);
    }

    [Fact]
    public void OutputSpansTruncateAndFlagInsteadOfOverflowing()
    {
        using var session = new NeedleSession(model, SessionCases.DefaultTools(), new NeedleOptions { Threads = 4, AutoDate = false });
        var calls = new byte[4096];
        var reasoning = new byte[8];
        var reply = session.Complete("what's it like in Lagos right now?"u8, calls, reasoning, default);
        Assert.Equal(ReplyStatus.Called, reply.Status);
        Assert.Equal(8, reply.ReasoningLength);
        Assert.True(reply.Flags.HasFlag(ReplyFlags.ReasoningTruncated));
        Assert.False(reply.Flags.HasFlag(ReplyFlags.ResponseTruncated));
    }

    [Fact]
    public void ACancelRequestedMidFlightEndsTheCallAndKeepsTheSessionUsable()
    {
        using var session = new NeedleSession(model, SessionCases.DefaultTools(), new NeedleOptions { Threads = 1, AutoDate = false });
        var calls = new byte[4096];
        // A single 30 ms timer can lose to a fast call, so retry until a cancel
        // lands mid-flight; a broken RequestCancel still fails by never winning.
        var reply = default(Reply);
        for (var attempt = 0; attempt < 5 && reply.Status != ReplyStatus.Cancelled; attempt++)
        {
            var canceller = new Thread(() =>
            {
                Thread.Sleep(30);
                session.RequestCancel();
            });
            canceller.Start();
            reply = session.Complete("what's it like in Lagos right now?"u8, calls);
            canceller.Join();
        }
        Assert.Equal(ReplyStatus.Cancelled, reply.Status);
        Assert.Equal(FailureReason.None, reply.Failure);
        var again = session.Complete("what's it like in Lagos right now?"u8, calls);
        Assert.Equal(ReplyStatus.Called, again.Status);
        Assert.Equal("get_weather", JsonValue.Parse(Encoding.UTF8.GetString(calls, 0, again.Utf8Length)).Items[0].GetString("name"));
    }
}
