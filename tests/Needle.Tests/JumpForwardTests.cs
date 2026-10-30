using System;
using System.Collections.Generic;
using System.Text;
using Needle.Cli;
using Needle.Tools;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

// Issue #16: jump-forward over grammar-forced spans. The toggle under test
// is NeedleOptions.JumpForward (off by default until the W28 accuracy gate).
// Canonical chunk tokenization can split the same bytes differently from the
// model's own split, so free choices downstream may flip (documented
// divergences, gated by W28). What jump-forward guarantees — and what these
// tests prove — is: forced bytes are byte-identical by construction, every
// fed chunk token stays grammar-admissible, the typed sink observes the same
// event sequence as per-byte decoding, and sequential engine passes drop.
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

    // Returns true when both paths emitted identical text. Reduction is
    // asserted only then: when the canonical split steers a free choice
    // elsewhere the pass counts are not comparable, and the divergence is
    // reported below for the W28 accuracy gate instead of asserted.
    bool ReportPrompt(string label, string tools, string prompt, ref int totalTokens, ref int totalOff, ref int totalOn)
    {
        var options = new NeedleOptions { Threads = 4, AutoDate = false, System = "" };
        using var plain = new NeedleSession(model, tools, WithJump(options, false));
        using var jumped = new NeedleSession(model, tools, WithJump(options, true));
        var expected = RunStep(plain, new SessionStep(StepKind.Query, prompt));
        var actual = RunStep(jumped, new SessionStep(StepKind.Query, prompt));
        var same = expected.Status == actual.Status && expected.Failure == actual.Failure &&
            expected.CallCount == actual.CallCount && expected.Calls == actual.Calls &&
            expected.Reasoning == actual.Reasoning && expected.Response == actual.Response;
        output.WriteLine($"{label}: tokens off={expected.DecodeTokens} on={actual.DecodeTokens}, passes off={expected.Passes} on={actual.Passes}, text={(same ? "identical" : "DIVERGED")}");
        if (!same)
        {
            output.WriteLine($"  off: {expected.Status}/{expected.Failure} calls={expected.Calls}");
            output.WriteLine($"  on:  {actual.Status}/{actual.Failure} calls={actual.Calls}");
            return false;
        }
        // Identical text with at least one forced span jumped (every call
        // opens with several) must take strictly fewer sequential passes.
        Assert.True(actual.Passes < expected.Passes, $"{label}: passes {actual.Passes} >= {expected.Passes} without jumping");
        totalTokens += expected.DecodeTokens;
        totalOff += expected.Passes;
        totalOn += actual.Passes;
        return true;
    }

    [Fact]
    public void JumpForwardReducesPassesOnGamePrompts()
    {
        var totalTokens = 0;
        var totalOff = 0;
        var totalOn = 0;
        var agreed = 0;
        var tools = BenchPrompts.GameToolsJson;
        foreach (var prompt in BenchPrompts.Game)
            if (ReportPrompt(prompt, tools, prompt, ref totalTokens, ref totalOff, ref totalOn)) agreed++;
        output.WriteLine($"game prompts: {agreed}/{BenchPrompts.Game.Length} text-identical; {totalTokens} reference tokens in {totalOff} passes without jumping, {totalOn} with jumping");
        Assert.True(agreed > 0, "no prompt agreed, nothing to compare");
        Assert.True(totalOn < totalOff, $"expected fewer passes with jumping, got {totalOn} vs {totalOff}");
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
    public void JumpForwardChunkTokensStayGrammarAdmissible()
    {
        // Every token of a canonical forced-span chunk must advance the
        // grammar from the span start and spell exactly the forced bytes:
        // feeding the chunk cannot change the outcome versus per-byte steps.
        using var tokenizer = new Tokenizer(model);
        using var trie = new TokenTrie(tokenizer, model.VocabSize);
        CheckChunks(tokenizer, trie, SessionCases.DefaultTools(), model.VocabSize);
        CheckChunks(tokenizer, trie, BenchPrompts.GameToolsJson, model.VocabSize);
    }

    static unsafe void CheckChunks(Tokenizer tokenizer, TokenTrie trie, string toolsJson, int vocabSize)
    {
        using var grammar = new Grammar(new ToolSet(toolsJson).Tools);
        var program = grammar.Program;
        var random = new Random(7);
        var mask = new uint[trie.Words];
        var scratch = new byte[256];
        var forced = new byte[256];
        var ids = new int[776];
        var decoded = new byte[1024];
        var states = new List<GrammarState> { Grammar.Start(program) };
        var walk = Grammar.Start(program);
        for (var step = 0; step < 300 && !Grammar.Complete(walk); step++)
        {
            fixed (uint* m = mask) trie.Allow(program, walk, m);
            var allowed = new List<int>();
            for (var id = 0; id < vocabSize; id++) if ((mask[id / 32] >> (id % 32) & 1) != 0) allowed.Add(id);
            if (allowed.Count == 0) break;
            var pick = allowed[random.Next(allowed.Count)];
            int length;
            fixed (byte* p = scratch) length = tokenizer.ConstraintBytes(pick, p);
            var ok = true;
            for (var i = 0; i < length && ok; i++) ok = Grammar.Advance(program, &walk, scratch[i]);
            if (!ok) break;
            states.Add(walk);
        }
        var checkedSpans = 0;
        foreach (var start in states)
        {
            var state = start;
            int count;
            fixed (byte* f = forced) count = Grammar.ForcedBytes(program, &state, new Span<byte>(f, forced.Length));
            if (count == 0) continue;
            checkedSpans++;
            var span = forced.AsSpan(0, count);
            var encoded = tokenizer.Encode(span, ids.AsSpan(), leadingSpace: false);
            Assert.True(encoded > 0, "forced span encoded to no tokens");
            var probe = start;
            var at = 0;
            for (var i = 0; i < encoded; i++)
            {
                int dlen;
                fixed (byte* p = scratch) dlen = tokenizer.ConstraintBytes(ids[i], p);
                Assert.True(dlen > 0, $"chunk token {ids[i]} has no bytes");
                for (var b = 0; b < dlen; b++)
                    Assert.True(Grammar.Advance(program, &probe, scratch[b]), $"chunk token {ids[i]} rejected at byte {b}");
                scratch.AsSpan(0, dlen).CopyTo(decoded.AsSpan(at));
                at += dlen;
            }
            Assert.True(new ReadOnlySpan<byte>(decoded, 0, at).SequenceEqual(span), "chunk bytes differ from the forced span");
            Assert.Equal(Grammar.Complete(state), Grammar.Complete(probe));
        }
        Assert.True(checkedSpans > 0, "no forced spans sampled");
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
