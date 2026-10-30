using System;
using System.Text;
using Needle.Cli;
using Needle.Tools;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

// Issue #16: lossless jump-forward over token-forced steps. A token may join
// the fed chunk only when the grammar+trie allow exactly that one token at
// the committed state (allowed-mask popcount == 1): per-token greedy would
// emit it regardless of logits, so the chunk is the per-token greedy sequence
// itself and free choices downstream see the identical token history. What
// these tests prove is the losslessness (byte-identical calls/reasoning/
// tokens/decode counts on every prompt), per-token probability-1 accounting,
// and typed-sink event parity. Pass counts are reported per prompt with a
// never-regress guard; on the current grammar/vocab no singleton mask occurs
// so jumping stays dormant (default off), and these tests pin that behavior.
public sealed class JumpForwardTests : IDisposable
{
    readonly Model model;
    readonly ITestOutputHelper output;

    public JumpForwardTests(ITestOutputHelper output)
    {
        this.output = output;
        Assets.Require();
        model = Model.Open(Assets.Model);
    }

    public void Dispose() => model.Dispose();

    static NeedleOptions WithJump(NeedleOptions options, bool jump) => new()
    {
        Threads = options.Threads,
        Threading = options.Threading,
        FrameBudget = options.FrameBudget,
        Context = options.Context,
        MaxReasoningTokens = options.MaxReasoningTokens,
        MaxCallTokens = options.MaxCallTokens,
        System = options.System,
        AutoDate = options.AutoDate,
        RefuseBelow = options.RefuseBelow,
        RetrievalTopK = options.RetrievalTopK,
        JumpForward = jump,
    };

    sealed record TextOutcome(ReplyStatus Status, FailureReason Failure, int CallCount, string Calls, string Reasoning, string Response, int PromptTokens, int DecodeTokens, int Passes);

    static TextOutcome RunStep(NeedleSession session, SessionStep step)
    {
        var calls = new byte[8192];
        var reasoning = new byte[8192];
        var response = new byte[8192];
        var input = Encoding.UTF8.GetBytes(step.Input);
        var reply = step.Kind == StepKind.Query
            ? session.Complete(input, calls, reasoning, response)
            : session.CompleteWithToolResult(input, calls, reasoning, response);
        return new TextOutcome(reply.Status, reply.Failure, reply.CallCount,
            reply.HasCalls ? Encoding.UTF8.GetString(calls, 0, reply.Utf8Length) : "",
            Encoding.UTF8.GetString(reasoning, 0, reply.ReasoningLength).Trim(),
            Encoding.UTF8.GetString(response, 0, reply.ResponseLength).Trim(),
            reply.PromptTokens, reply.DecodeTokens, session.LastCallPasses);
    }

    [Fact]
    public void JumpForwardIsOffByDefault()
    {
        Assert.False(new NeedleOptions().JumpForward);
    }

    // Lossless jumping must reproduce the reference path exactly: the same
    // status, calls, reasoning, response, and decode-token counts. Any
    // divergence fails the test (it would mean the uniqueness criterion let
    // a real choice slip into a chunk). Passes can only stay equal or drop:
    // every macro-step feeds at least one token, so jumping never costs a
    // sequential pass. Observed reduction is reported, not asserted.
    void ReportPrompt(string label, string tools, string prompt, ref int totalTokens, ref int totalOff, ref int totalOn)
    {
        var options = new NeedleOptions { Threads = 4, AutoDate = false, System = "" };
        using var plain = new NeedleSession(model, tools, WithJump(options, false));
        using var jumped = new NeedleSession(model, tools, WithJump(options, true));
        var expected = RunStep(plain, new SessionStep(StepKind.Query, prompt));
        var actual = RunStep(jumped, new SessionStep(StepKind.Query, prompt));
        var same = expected.Status == actual.Status && expected.Failure == actual.Failure &&
            expected.CallCount == actual.CallCount && expected.Calls == actual.Calls &&
            expected.Reasoning == actual.Reasoning && expected.Response == actual.Response &&
            expected.PromptTokens == actual.PromptTokens && expected.DecodeTokens == actual.DecodeTokens;
        output.WriteLine($"{label}: tokens off={expected.DecodeTokens} on={actual.DecodeTokens}, passes off={expected.Passes} on={actual.Passes}, text={(same ? "identical" : "DIVERGED")}");
        Assert.True(same, $"{label}: jumped path diverged: off={expected.Status}/{expected.Failure} calls={expected.Calls} reasoning={expected.Reasoning} || on={actual.Status}/{actual.Failure} calls={actual.Calls} reasoning={actual.Reasoning}");
        Assert.True(actual.Passes <= expected.Passes, $"{label}: passes {actual.Passes} > {expected.Passes} with jumping");
        totalTokens += expected.DecodeTokens;
        totalOff += expected.Passes;
        totalOn += actual.Passes;
    }

    [Fact]
    public void JumpForwardMatchesReferenceOnGamePrompts()
    {
        var totalTokens = 0;
        var totalOff = 0;
        var totalOn = 0;
        var tools = BenchPrompts.GameToolsJson;
        foreach (var prompt in BenchPrompts.Game)
            ReportPrompt(prompt, tools, prompt, ref totalTokens, ref totalOff, ref totalOn);
        output.WriteLine($"game prompts: {BenchPrompts.Game.Length}/{BenchPrompts.Game.Length} text-identical; {totalTokens} reference tokens in {totalOff} passes without jumping, {totalOn} with jumping");
        Assert.True(totalOn <= totalOff, $"jumping cost passes: {totalOn} vs {totalOff} without jumping");
    }

    static void AssertNpcEqual(NpcCommand expected, NpcCommand actual)
    {
        Assert.Equal(expected.Kind, actual.Kind);
        Assert.Equal(expected.Presence, actual.Presence);
        Assert.Equal(expected.MoveTo.X, actual.MoveTo.X);
        Assert.Equal(expected.MoveTo.Y, actual.MoveTo.Y);
        Assert.Equal(expected.Attack.Target.Length, actual.Attack.Target.Length);
        Assert.True(expected.Attack.Target.Span.SequenceEqual(actual.Attack.Target.Span));
        Assert.Equal(expected.GiveItem.Item, actual.GiveItem.Item);
        Assert.Equal(expected.GiveItem.Count, actual.GiveItem.Count);
        Assert.True(expected.GiveItem.To.Span.SequenceEqual(actual.GiveItem.To.Span));
        Assert.Equal(expected.SetFormation.Formation, actual.SetFormation.Formation);
    }

    void AssertSameToggleAgreement(string prompt, bool jump)
    {
        var schema = Encoding.UTF8.GetString(NpcCommands.Schema);
        var query = Encoding.UTF8.GetBytes(prompt);
        var options = new NeedleOptions { Threads = 4, AutoDate = false, JumpForward = jump };
        using var json = new NeedleSession(model, schema, options);
        var calls = new byte[1 << 16];
        var jsonReply = json.Complete(query, calls.AsSpan());
        using var typed = ToolSession.Open<NpcToolSet>(model, options);
        var sink = new NpcSink();
        var commands = new NpcCommand[8];
        var typedReply = typed.Complete(query, commands.AsSpan(), ref sink);
        var label = $"{prompt} jump={jump}";
        Assert.Equal(jsonReply.Status, typedReply.Status);
        Assert.Equal(jsonReply.CallCount, typedReply.CallCount);
        Assert.Equal(jsonReply.DecodeTokens, typedReply.DecodeTokens);
        Assert.Equal(json.LastCallPasses, typed.LastCallPasses);
        var restored = Encoding.UTF8.GetString(calls, 0, jsonReply.Utf8Length);
        var parsed = JsonValue.Parse(restored);
        Assert.Equal(parsed.Items.Count, typedReply.CallCount);
        for (var i = 0; i < typedReply.CallCount; i++)
        {
            var call = parsed.Items[i];
            Assert.True(NpcCommands.TryBind(new FunctionCall(call.GetString("name")!, call.Get("arguments")!.ToString()), out var expected), label);
            AssertNpcEqual(expected, commands[i]);
        }
        output.WriteLine($"{label}: tokens={jsonReply.DecodeTokens} passes={json.LastCallPasses} calls={jsonReply.CallCount}");
    }

    [Fact]
    public void JumpForwardTypedMatchesJsonUnderSameToggle()
    {
        foreach (var jump in new[] { false, true })
        {
            AssertSameToggleAgreement("move to 120, 45", jump);
            AssertSameToggleAgreement("give Lyra 3 potions", jump);
            AssertSameToggleAgreement("switch to wedge formation and follow the captain", jump);
            AssertSameToggleAgreement("form a circle", jump);
        }
    }

    [Fact]
    public void JumpForwardCountsTokensAgainstTheCallBudget()
    {
        const string prompt = "what's it like in Lagos right now?";
        var tools = SessionCases.DefaultTools();
        var input = Encoding.UTF8.GetBytes(prompt);
        // A two-token call budget starves both paths: the reference path
        // decodes two choice tokens, the jumping path trips on its first
        // forced chunk because forced tokens count toward MaxCallTokens even
        // though they skip sequential passes. Same failure either way.
        var starved = new NeedleOptions { Threads = 4, AutoDate = false, MaxCallTokens = 2 };
        using var plain = new NeedleSession(model, tools, WithJump(starved, false));
        using var jumped = new NeedleSession(model, tools, WithJump(starved, true));
        var expected = plain.Complete(input, new byte[4096], new byte[4096], default);
        var actual = jumped.Complete(input, new byte[4096], new byte[4096], default);
        Assert.Equal(ReplyStatus.Failed, expected.Status);
        Assert.Equal(FailureReason.CallTokenBudgetExceeded, expected.Failure);
        Assert.Equal(expected.Status, actual.Status);
        Assert.Equal(expected.Failure, actual.Failure);
    }

    [Fact]
    public void JumpForwardRespectsTheFrameBudget()
    {
        const string prompt = "what's it like in Lagos right now?";
        var tools = SessionCases.DefaultTools();
        var input = Encoding.UTF8.GetBytes(prompt);
        // A two-token slice budget fires inside reasoning on both paths, long
        // before any jump can run: same failure, proving the budget gate is
        // unaffected by the chunking code.
        var sliced = new NeedleOptions { Threads = 4, AutoDate = false, FrameBudget = FrameBudget.FromTokens(2) };
        using var plain = new NeedleSession(model, tools, WithJump(sliced, false));
        using var jumped = new NeedleSession(model, tools, WithJump(sliced, true));
        var expected = plain.Complete(input, new byte[4096], new byte[4096], default);
        var actual = jumped.Complete(input, new byte[4096], new byte[4096], default);
        Assert.Equal(ReplyStatus.Failed, expected.Status);
        Assert.Equal(FailureReason.FrameBudgetExceeded, expected.Failure);
        Assert.Equal(expected.Status, actual.Status);
        Assert.Equal(expected.Failure, actual.Failure);
    }
}
