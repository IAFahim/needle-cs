using System;
using System.Collections.Generic;
using System.Text;
using Needle.Tools;
using Xunit;

namespace Needle.Tests;

[NeedleTools]
public partial interface IHomeCommands
{
    [Tool("Turn on a light.")] void TurnOnLight([Param("Room")] string room);
    [Tool("Turn off a light.")] void TurnOffLight([Param("Room")] string room);
    [Tool("Dim or brighten a light.")] void SetBrightness([Param("Room")] string room, [Param("Percent")] int level);
    [Tool("Lock or unlock a door.")] void LockDoor([Param("Door")] string door, [Param("Locked")] bool locked);
}

// Records every grammar event, so a replay can be compared with a direct decode of the aliased text.
struct TraceSink : ICallSink<int>
{
    public List<string> Events;
    int pending;
    bool ready;

    public int CallCount { get; private set; }

    public bool TryTakeCall(out int command)
    {
        command = pending;
        if (!ready) return false;
        ready = false;
        return true;
    }

    public void ToolSelected(int tool) { Events.Add("tool " + tool); pending = tool; }
    public void PropertySelected(int property) => Events.Add("property " + property);
    public void StringByte(byte value) => Events.Add("s " + (char)value);
    public void NumberByte(byte value) => Events.Add("n " + (char)value);
    public void EnumSelected(int literal) => Events.Add("enum " + literal);
    public void BoolSelected(bool value) => Events.Add("bool " + value);
    public void NullSelected() => Events.Add("null");
    public void ValueCompleted() => Events.Add("value");

    public void CallCompleted()
    {
        Events.Add("call");
        ready = true;
        CallCount++;
    }
}

// Model-free: drives the generated grammar the way GenerateCallsTyped does, then runs the same repair step.
public sealed unsafe class TypedRepairTests
{
    sealed class Rig : IDisposable
    {
        readonly string schema = Encoding.UTF8.GetString(HomeCommands.Schema);
        readonly ToolSet set;
        readonly RepairPlan plan;
        readonly RepairScratch scratch;
        readonly RepairText request = new(1024);
        readonly RepairText history = new(8192);
        readonly byte[] restored = new byte[8192];
        readonly byte[] keyScratch = new byte[64];
        readonly CallSpan[] spans = new CallSpan[16];
        readonly GrammarProgram* program;

        public Rig()
        {
            set = new ToolSet(schema);
            plan = new RepairPlan(set, schema);
            scratch = new RepairScratch(spans.Length, 64, restored.Length + plan.LongestValue * 8);
            program = GrammarLoader.Load(HomeCommands.GrammarNodes, HomeCommands.GrammarProperties, HomeCommands.GrammarLiterals,
                HomeCommands.GrammarUnions, HomeCommands.GrammarTools, HomeCommands.GrammarBytes, HomeCommands.GrammarRoot, HomeCommands.GrammarToolCount);
        }

        public void Dispose() => GrammarLoader.Free(program);

        // Bytes allocated by the repair step alone, after one warm-up pass.
        public long RepairAllocations(string query, string generated, int rounds)
        {
            var q = Encoding.UTF8.GetBytes(query);
            request.Load(q);
            history.Load(default);
            history.Append(q);
            var text = Encoding.UTF8.GetBytes(generated);
            var commands = new HomeCommand[8];
            var before = new HomeSink();
            long allocated = 0;
            for (var round = 0; round <= rounds; round++)
            {
                var sink = before;
                var stored = 0;
                var start = GC.GetAllocatedBytesForCurrentThread();
                CallReplay.Repair(program, set.Aliases, plan, scratch, request, history, null, text, restored, keyScratch, spans,
                    in before, ref sink, commands.AsSpan(), ref stored, out _, out _, out var failure);
                if (round > 0) allocated += GC.GetAllocatedBytesForCurrentThread() - start;
                Assert.Equal(FailureReason.None, failure);
            }
            return allocated;
        }

        public (HomeCommand[] Commands, RepairOutcome Outcome, FailureReason Failure, int Calls) Run(string query, string generated, HomeSink sink = default)
        {
            var q = Encoding.UTF8.GetBytes(query);
            request.Load(q);
            history.Load(default);
            history.Append(q);
            var commands = new HomeCommand[8];
            var startCount = sink.CallCount;
            var before = sink;
            var stored = 0;
            var text = Encoding.UTF8.GetBytes(generated);
            var state = Grammar.Start(program);
            foreach (var b in text)
            {
                Assert.True(Grammar.Advance(program, &state, b, ref sink), generated);
                if (sink.TryTakeCall(out var call) && stored < commands.Length) commands[stored++] = call;
            }
            Assert.True(Grammar.Complete(state));
            var outcome = CallReplay.Repair(program, set.Aliases, plan, scratch, request, history, null, text, restored, keyScratch, spans,
                in before, ref sink, commands.AsSpan(), ref stored, out var callCount, out _, out var failure);
            Assert.Equal(callCount, sink.CallCount - startCount);
            return (commands[..stored], outcome, failure, callCount);
        }
    }

    static string Call(string name, string args) => $$"""[{"name":"{{name}}","arguments":{{args}}}]""";

    static string Text(Utf8Text128 text) => Encoding.UTF8.GetString(text.Span);

    [Fact]
    public void CleanTypedCallKeepsTheSinkCommands()
    {
        using var rig = new Rig();
        var (commands, outcome, failure, calls) = rig.Run("dim the bedroom lights to 40 percent", Call("set_brightness", """{"room":"bedroom","level":40}"""));
        Assert.Equal(FailureReason.None, failure);
        Assert.Equal(0, (int)(outcome & CallReplay.Rewrites));
        Assert.Equal(1, calls);
        var command = Assert.Single(commands);
        Assert.Equal(HomeCommandKind.SetBrightness, command.Kind);
        Assert.Equal("bedroom", Text(command.SetBrightness.Room));
        Assert.Equal(40, command.SetBrightness.Level);
        Assert.Equal(ReplyStatus.Called, CallReplay.Status(calls, 0.9f, 0.1f, 0, outcome));
    }

    [Fact]
    public void PairedToolIsReplayedAsTheCorrectedCommand()
    {
        using var rig = new Rig();
        var (commands, outcome, failure, _) = rig.Run("turn off the kitchen light", Call("turn_on_light", """{"room":"kitchen"}"""));
        Assert.Equal(FailureReason.None, failure);
        Assert.True(outcome.HasFlag(RepairOutcome.Repaired));
        var command = Assert.Single(commands);
        Assert.Equal(HomeCommandKind.TurnOffLight, command.Kind);
        Assert.Equal("kitchen", Text(command.TurnOffLight.Room));
        Assert.Equal(1u, command.Presence);
        Assert.Equal(ReplyFlags.Repaired, NeedleSession.ToFlags(outcome) & ReplyFlags.Repaired);
    }

    [Fact]
    public void BooleanRepairReachesTheTypedCommand()
    {
        using var rig = new Rig();
        var (commands, outcome, failure, _) = rig.Run("unlock the front door", Call("lock_door", """{"door":"front","locked":true}"""));
        Assert.Equal(FailureReason.None, failure);
        Assert.True(outcome.HasFlag(RepairOutcome.Repaired));
        var command = Assert.Single(commands);
        Assert.Equal(HomeCommandKind.LockDoor, command.Kind);
        Assert.Equal("front", Text(command.LockDoor.Door));
        Assert.False(command.LockDoor.Locked);
    }

    [Fact]
    public void NegatedTypedCallIsDroppedAndDeclined()
    {
        using var rig = new Rig();
        var (commands, outcome, failure, calls) = rig.Run("don't turn on the kitchen light", Call("turn_on_light", """{"room":"kitchen"}"""));
        Assert.Equal(FailureReason.None, failure);
        Assert.True(outcome.HasFlag(RepairOutcome.Negated));
        Assert.Empty(commands);
        Assert.Equal(0, calls);
        Assert.Equal(ReplyStatus.Declined, CallReplay.Status(calls, 0.9f, 0.1f, 0, outcome));
        Assert.True(NeedleSession.ToFlags(outcome).HasFlag(ReplyFlags.Negated));
    }

    [Fact]
    public void NegationKeepsTheOtherTypedCall()
    {
        using var rig = new Rig();
        var generated = """[{"name":"turn_on_light","arguments":{"room":"kitchen"}},{"name":"turn_off_light","arguments":{"room":"hallway"}}]""";
        var (commands, outcome, failure, calls) = rig.Run("turn on the kitchen light but don't turn off the hallway light", generated);
        Assert.Equal(FailureReason.None, failure);
        Assert.True(outcome.HasFlag(RepairOutcome.Negated));
        Assert.Equal(1, calls);
        var command = Assert.Single(commands);
        Assert.Equal(HomeCommandKind.TurnOnLight, command.Kind);
        Assert.Equal("kitchen", Text(command.TurnOnLight.Room));
    }

    [Fact]
    public void ReplayContinuesTheSinksRunningCount()
    {
        using var rig = new Rig();
        var sink = new HomeSink();
        var (_, _, _, _) = rig.Run("turn on the hallway light", Call("turn_on_light", """{"room":"hallway"}"""), sink);
        // A sink that already counted calls in earlier turns keeps counting from there after a replay.
        var (commands, outcome, _, calls) = rig.Run("turn off the kitchen light", Call("turn_on_light", """{"room":"kitchen"}"""), sink);
        Assert.True(outcome.HasFlag(RepairOutcome.Repaired));
        Assert.Equal(1, calls);
        Assert.Equal(HomeCommandKind.TurnOffLight, Assert.Single(commands).Kind);
    }

    [Fact]
    public void UngroundedTypedCallIsRefused()
    {
        using var rig = new Rig();
        var (commands, outcome, failure, calls) = rig.Run("dim the bedroom lights", Call("set_brightness", """{"room":"bedroom","level":30}"""));
        Assert.Equal(FailureReason.None, failure);
        Assert.True(outcome.HasFlag(RepairOutcome.Ungrounded));
        var command = Assert.Single(commands);
        Assert.Equal(30, command.SetBrightness.Level);
        // The gate withholds the call whatever the confidence, and even when a trigger routed it.
        Assert.Equal(ReplyStatus.Refused, CallReplay.Status(calls, 0.99f, 0.1f, 0, outcome));
        Assert.Equal(ReplyStatus.Refused, CallReplay.Status(calls, 0.99f, 0.1f, 1, outcome));
        Assert.True(NeedleSession.ToFlags(outcome).HasFlag(ReplyFlags.Ungrounded));
    }

    [Fact]
    public void RepairAndReplayAllocateNothingAfterWarmUp()
    {
        using var rig = new Rig();
        Assert.Equal(0, rig.RepairAllocations("turn off the kitchen light", Call("turn_on_light", """{"room":"kitchen"}"""), 50));
        Assert.Equal(0, rig.RepairAllocations("dim the bedroom lights to 40 percent", Call("set_brightness", """{"room":"bedroom","level":40}"""), 50));
    }

    [Fact]
    public void LowConfidenceRefusalStillHonoursTheTriggerRoute()
    {
        Assert.Equal(ReplyStatus.Refused, CallReplay.Status(1, 0.05f, 0.1f, 0, RepairOutcome.None));
        Assert.Equal(ReplyStatus.Called, CallReplay.Status(1, 0.05f, 0.1f, 1, RepairOutcome.None));
    }

    [Fact]
    public void ReplayFindsTheAliasForANameReusedAtTwoDepths()
    {
        const string tools = """
            [{"name":"placeOrder","description":"d","parameters":{"type":"object","properties":{
            "name":{"type":"string"},
            "item":{"type":"object","properties":{"name":{"type":"string"},"qty":{"type":"integer"}},"required":["name"]},
            "rushDelivery":{"type":["boolean","null"]}},
            "required":["name","item"]}},
             {"name":"cancelOrder","description":"d","parameters":{"type":"object","properties":{"orderId":{"type":"string"}}}}]
            """;
        var set = new ToolSet(tools);
        using var grammar = new Grammar(set.Tools);
        var generated = Encoding.UTF8.GetBytes("""[{"name":"cancel_order","arguments":{"order_id":"A\"7"}},{"name":"place_order","arguments":{"item":{"qty":2,"name_2":"tea"},"name":"Ada","rush_delivery":null}}]""");
        var direct = new TraceSink { Events = new List<string>() };
        var state = Grammar.Start(grammar.Program);
        foreach (var b in generated) Assert.True(Grammar.Advance(grammar.Program, &state, b, ref direct));
        Assert.True(Grammar.Complete(state));

        var restored = new byte[1024];
        var spans = new CallSpan[4];
        Assert.True(CallRestorer.TryRestore(set.Aliases, generated, restored, new byte[64], spans, out var length, out var count));
        Assert.Contains("\"orderId\"", Encoding.UTF8.GetString(restored, 0, length));
        var replay = new TraceSink { Events = new List<string>() };
        var commands = new int[4];
        var stored = 0;
        Assert.True(CallReplay.Replay(grammar.Program, set.Aliases, restored.AsSpan(0, length), spans.AsSpan(0, count), ref replay, commands.AsSpan(), ref stored));
        Assert.Equal(direct.Events, replay.Events);
        Assert.Equal(2, stored);
        Assert.Equal(new[] { 1, 0 }, commands[..2]);
    }

    [Fact]
    public void ReplayRejectsAKeyTheSchemaDoesNotHave()
    {
        var set = new ToolSet(Encoding.UTF8.GetString(HomeCommands.Schema));
        using var grammar = new Grammar(set.Tools);
        var json = Encoding.UTF8.GetBytes("""[{"name":"turn_on_light","arguments":{"floor":"2"}}]""");
        var arguments = Encoding.UTF8.GetString(json).IndexOf("{\"floor\"", StringComparison.Ordinal);
        var spans = new[] { new CallSpan(0, arguments, "{\"floor\":\"2\"}".Length) };
        var sink = new HomeSink();
        var commands = new HomeCommand[2];
        var stored = 0;
        Assert.False(CallReplay.Replay(grammar.Program, set.Aliases, json, spans, ref sink, commands.AsSpan(), ref stored));
    }
}
