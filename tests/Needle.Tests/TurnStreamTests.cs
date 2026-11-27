using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

/// <summary>Model-free checks of the pieces behind streamed turns.</summary>
public class CallScannerTests
{
    const string Text = "[{\"name\":\"set_timer\",\"arguments\":{\"minutes\":12}},{\"name\":\"send_message\",\"arguments\":{\"to\":\"Zo\\\"}{ë\",\"nested\":{\"a\":[1,{\"b\":\"]}\"}]}}},{\"name\":\"stop\",\"arguments\":{}}]";

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(1000)]
    public void FindsNamesArgumentsAndClosesAtAnyChunking(int chunk)
    {
        var bytes = Encoding.UTF8.GetBytes(Text);
        var scanner = CallScanner.Create();
        var seen = new List<string>();
        var longest = new Dictionary<int, string>();
        for (var end = Math.Min(chunk, bytes.Length); ; end = Math.Min(end + chunk, bytes.Length))
        {
            var visible = new ReadOnlySpan<byte>(bytes, 0, end);
            for (;;)
            {
                var scan = scanner.Next(visible, out var at);
                if (scanner.Call >= 0 && scanner.ArgumentsStart >= 0)
                    longest[scanner.Call] = Encoding.UTF8.GetString(bytes, scanner.ArgumentsStart, scanner.ArgumentsLength(visible));
                if (scan == CallScan.None) break;
                if (scan == CallScan.Named) seen.Add("named " + scanner.Call + " " + Encoding.UTF8.GetString(bytes, scanner.NameStart, scanner.NameEnd - scanner.NameStart));
                else seen.Add("closed " + scanner.Call + " " + Encoding.UTF8.GetString(bytes, scanner.CallStart, at - scanner.CallStart));
            }
            if (end == bytes.Length) break;
        }
        Assert.Equal(new[]
        {
            "named 0 set_timer",
            "closed 0 {\"name\":\"set_timer\",\"arguments\":{\"minutes\":12}}",
            "named 1 send_message",
            "closed 1 {\"name\":\"send_message\",\"arguments\":{\"to\":\"Zo\\\"}{ë\",\"nested\":{\"a\":[1,{\"b\":\"]}\"}]}}}",
            "named 2 stop",
            "closed 2 {\"name\":\"stop\",\"arguments\":{}}",
        }, seen);
        Assert.Equal("{\"minutes\":12}", longest[0]);
        Assert.Equal("{\"to\":\"Zo\\\"}{ë\",\"nested\":{\"a\":[1,{\"b\":\"]}\"}]}}", longest[1]);
        Assert.Equal("{}", longest[2]);
    }

    [Fact]
    public void AnEmptyListHasNoCalls()
    {
        var scanner = CallScanner.Create();
        Assert.Equal(CallScan.None, scanner.Next("[]"u8, out _));
        Assert.Equal(-1, scanner.Call);
    }

    [Fact]
    public void DefaultTurnsFailCleanly()
    {
        NeedleTurn turn = default;
        Assert.Throws<ObjectDisposedException>(() => turn.Run(1));
        Assert.Throws<ObjectDisposedException>(() => turn.IsDone);
        turn.Cancel();
        TurnEvents events = default;
        Assert.Equal(0, events.Count);
        NeedleAgent agent = default;
        Assert.Throws<ObjectDisposedException>(() => agent.Begin("hello"));
        Assert.Throws<ObjectDisposedException>(() => agent.TryPoll(out _));
    }
}

/// <summary>Streamed turns against the model: the answer is Ask's bit for bit, and the events describe it.</summary>
public sealed class TurnStreamTests : IDisposable
{
    readonly ITestOutputHelper output;
    readonly NeedleModel model;

    public TurnStreamTests(ITestOutputHelper output)
    {
        this.output = output;
        Assets.Require();
        model = NeedleModel.Open(Assets.Model);
    }

    public void Dispose() => model.Dispose();

    static readonly FrameBudget[] Budgets =
    {
        FrameBudget.FromTokens(1),
        FrameBudget.FromTokens(16),
        FrameBudget.FromMilliseconds(0.5),
        FrameBudget.Unlimited,
    };

    static AgentOptions ToAgent(NeedleOptions o) => new()
    {
        Threads = o.Threads,
        AutoDate = o.AutoDate ? Toggle.On : Toggle.Off,
        RefuseBelow = o.RefuseBelow,
        RetrievalTopK = o.RetrievalTopK == int.MaxValue ? 0 : o.RetrievalTopK,
        JumpForward = o.JumpForward ? Toggle.On : Toggle.Default,
        PromptLookup = o.PromptLookup ? Toggle.On : Toggle.Default,
    };

    internal static string Fingerprint(Answer answer)
    {
        var d = answer.Details;
        var b = new StringBuilder();
        b.Append(answer.Kind).Append('|').Append(answer.Why).Append('|').Append(d.Status).Append('|').Append(d.Failure).Append('|').Append(d.Flags);
        b.Append('|').Append(BitConverter.SingleToInt32Bits(d.Confidence)).Append('|').Append(BitConverter.SingleToInt32Bits(d.HeadConfidence)).Append('|').Append(BitConverter.SingleToInt32Bits(d.DecodeConfidence));
        b.Append('|').Append(d.CallCount).Append('|').Append(d.Utf8Length).Append('|').Append(d.ReasoningLength).Append('|').Append(d.ResponseLength).Append('|').Append(d.PromptTokens).Append('|').Append(d.DecodeTokens);
        foreach (var call in answer.Calls) b.Append("|call ").Append(call.ToString());
        foreach (var call in answer.Suppressed) b.Append("|suppressed ").Append(call.ToString());
        b.Append("|text ").Append(answer.Text.ToString()).Append("|thinking ").Append(answer.Thinking.ToString()).Append("|ungrounded ").Append(answer.Ungrounded.ToString());
        return b.ToString();
    }

    sealed class Collected
    {
        public readonly List<string> Kinds = new();
        public readonly StringBuilder Thinking = new();
        public readonly StringBuilder Speaking = new();
        public readonly List<string> Started = new();
        public readonly List<string> Closed = new();
        public int Finals;
        public int Slices;
        public TurnEventKind Last;
    }

    static void Collect(TurnEvents events, Collected into)
    {
        foreach (var e in events)
        {
            Assert.Equal(0, into.Finals);
            into.Kinds.Add(e.Kind.ToString());
            into.Last = e.Kind;
            switch (e.Kind)
            {
                case TurnEventKind.Thinking: into.Thinking.Append(e.Text.ToString()); break;
                case TurnEventKind.Speaking: into.Speaking.Append(e.Text.ToString()); break;
                case TurnEventKind.CallStarted:
                    Assert.Equal(into.Started.Count, e.CallIndex);
                    into.Started.Add(e.Name.ToString());
                    break;
                case TurnEventKind.ArgumentProgress:
                    Assert.Equal(into.Started.Count - 1, e.CallIndex);
                    Assert.Equal((byte)'{', e.Arguments.Bytes[0]);
                    break;
                case TurnEventKind.CallClosed:
                    Assert.Equal(into.Closed.Count, e.CallIndex);
                    Assert.True(e.IsProvisional);
                    into.Closed.Add(e.IsWithheld ? "withheld " + e.Name.ToString() : e.Name.ToString() + "(" + e.Arguments.ToString() + ")");
                    break;
                default:
                    Assert.True(e.IsFinal);
                    into.Finals++;
                    break;
            }
        }
    }

    static string Stream(NeedleAgent agent, SessionStep step, FrameBudget budget, Collected collected)
    {
        var turn = step.Kind == StepKind.ToolResult ? agent.BeginContinue(step.Input) : agent.Begin(step.Input);
        var guard = 0;
        while (!turn.IsDone)
        {
            Collect(turn.Run(budget), collected);
            collected.Slices++;
            Assert.True(++guard < 100_000, "the turn never ended");
        }
        Collect(turn.Run(budget), collected);
        Assert.Equal(1, collected.Finals);
        CheckEvents(turn.Answer, collected);
        return Fingerprint(turn.Answer);
    }

    static void CheckEvents(Answer answer, Collected c)
    {
        var expectedFinal = answer.Kind switch
        {
            AnswerKind.Calls or AnswerKind.Text or AnswerKind.Declined => TurnEventKind.Done,
            AnswerKind.Refused => TurnEventKind.Refused,
            AnswerKind.Cancelled => TurnEventKind.Cancelled,
            _ => TurnEventKind.Failed,
        };
        Assert.Equal(expectedFinal, c.Last);
        Assert.Equal(answer.Thinking.ToString(), c.Thinking.ToString().Trim());
        Assert.Equal(answer.Text.ToString(), c.Speaking.ToString().Trim());
        if (answer.Kind is AnswerKind.Calls or AnswerKind.Refused)
        {
            Assert.Equal(c.Started.Count, c.Closed.Count);
            var final = answer.Kind == AnswerKind.Calls ? answer.Calls : answer.Suppressed;
            // With one call, the provisional repair saw the same list as the final one.
            if (c.Closed.Count == 1 && final.Count == 1)
                Assert.Equal(final[0].ToString(), c.Closed[0]);
            if ((answer.Details.Flags & (ReplyFlags.Repaired | ReplyFlags.Negated | ReplyFlags.Invalid)) == 0)
            {
                Assert.Equal(final.Count, c.Closed.Count);
                for (var i = 0; i < final.Count; i++) Assert.Equal(final[i].ToString(), c.Closed[i]);
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void RunMatchesAskBitForBitAtEveryBudget(int threads)
    {
        threads = Assets.ThreadsFor(threads);
        var turns = 0;
        foreach (var @case in SessionCases.All(threads))
        {
            var tools = @case.ToolsJson ?? SessionCases.DefaultTools();
            var options = ToAgent(@case.Options);
            var expected = new List<string>();
            using (var asker = model.Agent(tools, options))
                foreach (var step in @case.Steps)
                {
                    if (step.Kind == StepKind.Reset) { asker.Forget(); expected.Add("reset"); continue; }
                    expected.Add(Fingerprint(step.Kind == StepKind.ToolResult ? asker.Continue(step.Input) : asker.Ask(step.Input)));
                }
            foreach (var budget in Budgets)
            {
                using var streamer = model.Agent(tools, options);
                for (var i = 0; i < @case.Steps.Length; i++)
                {
                    var step = @case.Steps[i];
                    if (step.Kind == StepKind.Reset) { streamer.Forget(); continue; }
                    var collected = new Collected();
                    var actual = Stream(streamer, step, budget, collected);
                    Assert.True(expected[i] == actual, $"{@case.Name} step {i} budget {budget.MaxTokensPerSlice}t/{budget.MaxMillisecondsPerSlice}ms:\n ask    {expected[i]}\n stream {actual}");
                    turns++;
                    if (budget.MaxTokensPerSlice == 1) Assert.True(collected.Slices > 3, $"{@case.Name}: one-token slices took only {collected.Slices} slices");
                }
            }
        }
        output.WriteLine($"{turns} streamed turns at {threads} thread(s) matched Ask bit for bit");
    }

    [Fact]
    public void EventsDescribeTheAnswer()
    {
        var threads = Assets.ThreadsFor(2);
        foreach (var @case in SessionCases.All(threads))
        {
            var tools = @case.ToolsJson ?? SessionCases.DefaultTools();
            using var agent = model.Agent(tools, ToAgent(@case.Options));
            foreach (var step in @case.Steps)
            {
                if (step.Kind == StepKind.Reset) { agent.Forget(); continue; }
                var collected = new Collected();
                var turn = step.Kind == StepKind.ToolResult ? agent.BeginContinue(step.Input) : agent.Begin(step.Input);
                while (!turn.IsDone) Collect(turn.Run(FrameBudget.FromTokens(3)), collected);
                Collect(turn.Run(FrameBudget.FromTokens(3)), collected);
                var answer = turn.Answer;
                CheckEvents(answer, collected);
                output.WriteLine($"{@case.Name}: {string.Join(" ", collected.Kinds)} -> {answer.ToString()}");
            }
        }
    }

    [Fact]
    public void EventViewsGoStaleWithTheNextTurn()
    {
        using var agent = model.Agent(SessionCases.DefaultTools(), new AgentOptions { Threads = 1, AutoDate = Toggle.Off });
        var turn = agent.Begin("set a timer for 12 minutes");
        var events = turn.Run(FrameBudget.Unlimited);
        Assert.True(turn.IsDone);
        Assert.True(events.TryGetEnd(out var end));
        Assert.Equal(TurnEventKind.Done, end.Kind);
        Assert.Equal(AnswerKind.Calls, end.Answer.Kind);
        agent.Ask("set a timer for 5 minutes");
        try
        {
            _ = events[0].Kind;
            Assert.Fail("a stale event must throw");
        }
        catch (InvalidOperationException)
        {
        }
        Assert.Throws<InvalidOperationException>(() => turn.Run(1));
        Assert.Throws<InvalidOperationException>(() => turn.IsDone);
    }

    [Fact]
    public void CancelForgetsAndBusyTurnsRefuseOtherWork()
    {
        var tools = SessionCases.DefaultTools();
        var options = new AgentOptions { Threads = 1, AutoDate = Toggle.Off };
        string fresh;
        using (var reference = model.Agent(tools, options)) fresh = Fingerprint(reference.Ask("play some Miles Davis"));

        using var agent = model.Agent(tools, options);
        Assert.True(agent.Ask("set a timer for 12 minutes").Ok);
        var turn = agent.Begin("what's it like in Lagos right now?");
        turn.Run(1);
        turn.Run(1);
        Assert.False(turn.IsDone);
        Assert.Throws<InvalidOperationException>(() => agent.Ask("hello"));
        Assert.Throws<InvalidOperationException>(() => agent.Begin("hello"));
        Assert.Throws<InvalidOperationException>(() => agent.TryPost("hello"));
        turn.Cancel();
        Assert.True(turn.IsDone);
        Assert.Equal(AnswerKind.Cancelled, turn.Answer.Kind);
        var events = turn.Run(1);
        Assert.True(events.TryGetEnd(out var end));
        Assert.Equal(TurnEventKind.Cancelled, end.Kind);
        Assert.Equal(fresh, Fingerprint(agent.Ask("play some Miles Davis")));

        var abandoned = agent.Begin("what's it like in Lagos right now?");
        abandoned.Run(2);
        agent.Forget();
        Assert.Equal(fresh, Fingerprint(agent.Ask("play some Miles Davis")));
    }

    [Fact]
    public void LosslessSpeedupsStreamTheSameAnswer()
    {
        var tools = SessionCases.DefaultTools();
        foreach (var options in new[]
        {
            new AgentOptions { Threads = 1, AutoDate = Toggle.Off, JumpForward = Toggle.On },
            new AgentOptions { Threads = 1, AutoDate = Toggle.Off, PromptLookup = Toggle.On },
            new AgentOptions { Threads = 1, AutoDate = Toggle.Off, ReasoningBudget = AgentOptions.NoReasoning },
        })
        {
            foreach (var query in new[] { "set a timer for 12 minutes and play some Miles Davis", "text Sam that dinner is at 7" })
            {
                string expected;
                using (var asker = model.Agent(tools, options)) expected = Fingerprint(asker.Ask(query));
                using var streamer = model.Agent(tools, options);
                var collected = new Collected();
                Assert.Equal(expected, Stream(streamer, new SessionStep(StepKind.Query, query), FrameBudget.FromTokens(2), collected));
            }
        }
    }

    [Fact]
    public void PostedTurnsReportProgressAndMatchAsking()
    {
        var tools = SessionCases.DefaultTools();
        var options = new AgentOptions { Threads = Assets.ThreadsFor(2), AutoDate = Toggle.Off, Stateless = true };
        const string query = "set a timer for 12 minutes and play some Miles Davis";
        string expected;
        using (var asker = model.Agent(tools, options)) expected = Fingerprint(asker.Ask(query));
        using var agent = model.Agent(tools, options);
        for (var round = 0; round < 2; round++)
        {
            Assert.True(agent.TryPost(query));
            var collected = new Collected();
            var clock = Stopwatch.StartNew();
            string? actual = null;
            while (clock.Elapsed < TimeSpan.FromMinutes(2))
            {
                if (agent.TryPoll(out var events)) Collect(events, collected);
                if (agent.TryTake(out var answer))
                {
                    if (agent.TryPoll(out var rest)) Collect(rest, collected);
                    actual = Fingerprint(answer);
                    CheckEvents(answer, collected);
                    break;
                }
                Thread.Sleep(0);
            }
            Assert.Equal(expected, actual);
            Assert.Equal(1, collected.Finals);
            Assert.Contains("CallClosed", collected.Kinds);
            output.WriteLine($"posted round {round}: {collected.Kinds.Count} events, {string.Join(", ", collected.Closed)}");
        }
    }

    [Fact]
    public void StreamingDoesNotAllocate()
    {
        using var agent = model.Agent(SessionCases.DefaultTools(), new AgentOptions { Threads = 1, AutoDate = Toggle.Off });
        const string query = "set a timer for 12 minutes and play some Miles Davis";
        for (var i = 0; i < 3; i++) { agent.Forget(); Drain(agent.Begin(query)); }
        var before = GC.GetAllocatedBytesForCurrentThread();
        var total = 0;
        for (var i = 0; i < 6; i++) { agent.Forget(); total += Drain(agent.Begin(query)); }
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.True(total > 0);
    }

    static int Drain(NeedleTurn turn)
    {
        var total = 0;
        while (!turn.IsDone)
            foreach (var e in turn.Run(FrameBudget.FromTokens(4)))
            {
                total += (int)e.Kind + e.Name.Length + e.Arguments.Length + e.Text.Length;
                if (e.Kind == TurnEventKind.CallClosed && e.TryGetInt32("minutes", out var minutes)) total += minutes;
            }
        foreach (var call in turn.Answer) total += call.Arguments.Length;
        return total;
    }
}
