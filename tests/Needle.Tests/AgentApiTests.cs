using System;
using Xunit;

namespace Needle.Tests;

public class AgentApiTests
{
    [Fact]
    public void DefaultHandlesFailCleanly()
    {
        NeedleAgent agent = default;
        NeedleModel model = default;
        Assert.False(agent.IsOpen);
        Assert.False(model.IsOpen);
        Assert.Throws<ObjectDisposedException>(() => agent.Ask("hello"));
        Assert.Throws<ObjectDisposedException>(() => model.Agent("[]"));
        agent.Dispose();
        model.Dispose();
        agent.Cancel();
    }

    [Fact]
    public void ReusedSlotsDoNotRevive()
    {
        HandleTable<object>.Add(new object(), out var slot, out var generation);
        Assert.NotNull(HandleTable<object>.Remove(slot, generation));
        HandleTable<object>.Add(new object(), out var slot2, out var generation2);
        Assert.Equal(slot, slot2);
        Assert.NotEqual(generation, generation2);
        Assert.False(HandleTable<object>.TryGet(slot, generation, out _));
        Assert.True(HandleTable<object>.TryGet(slot2, generation2, out _));
        Assert.Null(HandleTable<object>.Remove(slot, generation));
        HandleTable<object>.Remove(slot2, generation2);
    }

    [Fact]
    public void DefaultOptionsMeanUpstream()
    {
        var options = default(AgentOptions).ToNeedleOptions(null);
        Assert.Equal(2048, options.Context);
        Assert.Equal(96, options.ReasoningBudget);
        Assert.Equal(384, options.MaxCallTokens);
        Assert.Equal(0.1f, options.RefuseBelow);
        Assert.True(options.AutoDate);
        Assert.True(options.Repair);
        Assert.Equal(int.MaxValue, options.RetrievalTopK);
        Assert.Equal(ThreadPolicy.AutoThreads, options.ResolvedThreading.Threads);

        var custom = new AgentOptions { Threads = 3, ReasoningBudget = AgentOptions.NoReasoning, RefuseBelow = -1, AutoDate = Toggle.Off, Repair = Toggle.Off, JumpForward = Toggle.On }.ToNeedleOptions(null);
        Assert.Equal(3, custom.ResolvedThreading.Threads);
        Assert.Equal(0, custom.ReasoningBudget);
        Assert.Equal(0f, custom.RefuseBelow);
        Assert.False(custom.AutoDate);
        Assert.False(custom.Repair);
        Assert.True(custom.JumpForward);
    }

    [Fact]
    public void ArgumentsReadWithoutAParseTree()
    {
        var json = "{\"room\": \"kitchen\", \"level\":42, \"ratio\":0.5,\"on\":true,\"note\":\"say \\\"hi\\\"\\n\\u00e9\",\"nested\":{\"a\":[1,{\"b\":\"}\"}]},\"last\":null}"u8;
        Assert.True(JsonArgs.TryFind(json, "room".AsSpan(), out var room, out var isString));
        Assert.True(isString);
        Assert.True(new Utf8Span(room).Equals("kitchen"));
        Assert.True(JsonArgs.TryFind(json, "level"u8, out var level, out isString));
        Assert.False(isString);
        Assert.True(new Utf8Span(level).Equals("42"));
        Assert.True(JsonArgs.TryFind(json, "last".AsSpan(), out var last, out _));
        Assert.True(new Utf8Span(last).Equals("null"));
        Assert.True(JsonArgs.TryFind(json, "nested".AsSpan(), out var nested, out _));
        Assert.Equal("{\"a\":[1,{\"b\":\"}\"}]}", new Utf8Span(nested).ToString());
        Assert.False(JsonArgs.TryFind(json, "missing".AsSpan(), out _, out _));
        Assert.False(JsonArgs.TryFind("[1,2]"u8, "a".AsSpan(), out _, out _));

        Assert.True(JsonArgs.TryFind(json, "note".AsSpan(), out var note, out _));
        Span<byte> buffer = stackalloc byte[64];
        Assert.True(JsonArgs.TryUnescape(note, buffer, out var written));
        Assert.Equal("say \"hi\"\né", System.Text.Encoding.UTF8.GetString(buffer.Slice(0, written)));
        Assert.False(JsonArgs.TryUnescape(note, buffer.Slice(0, 3), out _));
        Assert.True(JsonArgs.TryUnescape("\\ud83d\\ude00"u8, buffer, out written));
        Assert.Equal("😀", System.Text.Encoding.UTF8.GetString(buffer.Slice(0, written)));
    }

    [Fact]
    public void Utf8SpanComparesWithStrings()
    {
        var text = new Utf8Span("café"u8);
        Assert.True(text.Equals("café"));
        Assert.False(text.Equals("cafe"));
        Assert.False(text.Equals((string?)null));
        Assert.True(new Utf8Span("set_timer"u8).Equals("set_timer"));
        Assert.False(new Utf8Span("set_timer"u8).Equals("set_time"));
        Span<char> chars = stackalloc char[8];
        Assert.True(text.TryCopyTo(chars, out var count));
        Assert.Equal("café", chars.Slice(0, count).ToString());
        Assert.False(text.TryCopyTo(chars.Slice(0, 2), out _));
        Assert.Equal("café", text.ToString());
    }
}

public class AgentModelTests
{
    [Fact]
    public void AskReturnsCallsAndAnswersGoStale()
    {
        Assets.Require();
        using var model = NeedleModel.Open(Assets.Model);
        using var agent = model.Agent(SessionCases.DefaultTools(), new AgentOptions { Threads = Assets.ThreadsFor(2), AutoDate = Toggle.Off });
        var answer = agent.Ask("set a timer for 12 minutes");
        Assert.True(answer.Ok, answer.ToString());
        Assert.Equal(AnswerKind.Calls, answer.Kind);
        Assert.Equal(AnswerWhy.None, answer.Why);
        var seen = 0;
        foreach (var call in answer)
        {
            seen++;
            Assert.False(call.Name.IsEmpty);
            Assert.True(call.Arguments.Bytes[0] == (byte)'{');
        }
        Assert.Equal(answer.Calls.Count, seen);
        Assert.True(seen > 0);

        agent.Forget();
        try
        {
            _ = answer.Kind;
            Assert.Fail("a stale answer must throw");
        }
        catch (InvalidOperationException)
        {
        }
    }

    [Fact]
    public void MatchesTheSessionItWraps()
    {
        Assets.Require();
        var tools = SessionCases.DefaultTools();
        using var engineModel = Model.Open(Assets.Model);
        using var session = new NeedleSession(engineModel, tools, new NeedleOptions { Threads = 1, AutoDate = false });
        var expected = session.Complete("set a timer for 12 minutes and play some Miles Davis");

        using var model = NeedleModel.Open(Assets.Model);
        using var agent = model.Agent(tools, new AgentOptions { Threads = 1, AutoDate = Toggle.Off });
        var answer = agent.Ask("set a timer for 12 minutes and play some Miles Davis");
        Assert.Equal(expected.FunctionCalls.Count, answer.Calls.Count);
        for (var i = 0; i < answer.Calls.Count; i++)
        {
            var call = answer.Calls[i];
            Assert.True(call.Is(expected.FunctionCalls[i].Name));
            Assert.Equal(expected.FunctionCalls[i].ArgumentsJson, call.Arguments.ToString());
        }
    }

    [Fact]
    public void ModelOutlivesItsHandleWhileAgentsUseIt()
    {
        Assets.Require();
        var model = NeedleModel.Open(Assets.Model);
        using var agent = model.Agent(SessionCases.DefaultTools(), new AgentOptions { Threads = 1, AutoDate = Toggle.Off });
        model.Dispose();
        Assert.False(model.IsOpen);
        Assert.True(agent.Ask("set a timer for 5 minutes").Ok);
    }

    [Fact]
    public void SetSystemStartsOverWithNewFacts()
    {
        Assets.Require();
        using var model = NeedleModel.Open(Assets.Model);
        using var agent = model.Agent(SessionCases.DefaultTools(), new AgentOptions { Threads = 1, AutoDate = Toggle.Off });
        Assert.True(agent.Ask("set a timer for 5 minutes").Ok);
        agent.SetSystem("date: 2026-11-26 Thu 09:00");
        Assert.True(agent.Ask("set a timer for 5 minutes").Ok);
    }

    [Fact]
    public void AskingDoesNotAllocate()
    {
        Assets.Require();
        using var model = NeedleModel.Open(Assets.Model);
        using var agent = model.Agent(SessionCases.DefaultTools(), new AgentOptions { Threads = 1, AutoDate = Toggle.Off });
        const string query = "set a timer for 12 minutes";
        for (var i = 0; i < 4; i++) { agent.Forget(); Read(agent.Ask(query)); }
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 8; i++) { agent.Forget(); Read(agent.Ask(query)); }
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    static int Read(Answer answer)
    {
        var total = 0;
        foreach (var call in answer)
        {
            total += call.Name.Length + call.Arguments.Length;
            if (call.TryGetInt32("minutes", out var minutes)) total += minutes;
        }
        return total + answer.Text.Length;
    }
}
