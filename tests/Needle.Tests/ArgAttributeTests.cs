using System;
using System.ComponentModel;
using System.Text;
using Needle.Tools;
using Xunit;

namespace Needle.Tests;

[Tools]
public partial interface IArena
{
    [Description("Set the music volume.")] void SetVolume([Description("Percent"), Arg(Min = 0, Max = 100)] int level);
    [Description("Walk to a map position.")]
    void MoveTo([Arg(Min = -50, Max = 50, Inferred = true)] float x, [Arg(Min = -50, Max = 50, Inferred = true)] float y);
    [Description("Walk to a position the player names.")] void MoveToSaid(float x, float y);
    [Description("Aim the bow.")] void Aim([Arg(Inferred = true)] float angle = 0f, [Arg(Inferred = true)] Shape spread = Shape.Line);
    [Description("Name the squad.")] void Rename([Arg(MaxLength = 16)] Utf8Text32 name);
    [Description("Set the speed.")] void Speed([Arg(Min = 0.5, Max = 2.5)] double factor);
}

[NeedleTools]
public partial interface IRangeCommands
{
    [Tool("Set the music volume.")] void Loudness([Param("Percent")][Arg(Min = 0, Max = 100)] int level);
    [Tool("Walk to a map position.")] void Walk([Arg(Min = -50, Max = 50, Inferred = true)] float x, [Arg(Min = -50, Max = 50, Inferred = true)] float y);
}

// [Arg(Min, Max, MaxLength, Inferred)]: what the generator writes into the schema, and how the repair step
// enforces it on the JSON path (NeedleAgent, Call.Is) and on the typed path (CallReplay into a generated sink).
public sealed unsafe class ArgAttributeTests
{
    [Fact]
    public void LimitsReachTheSchema()
    {
        var schema = IArena.Calls.Schema;
        Assert.Contains("\"level\":{\"type\":\"integer\",\"description\":\"Percent\",\"minimum\":0,\"maximum\":100}", schema);
        Assert.Contains("\"x\":{\"type\":\"number\",\"minimum\":-50,\"maximum\":50}", schema);
        Assert.Contains("\"factor\":{\"type\":\"number\",\"minimum\":0.5,\"maximum\":2.5}", schema);
        Assert.Contains("\"name\":{\"type\":\"string\",\"maxLength\":16}", schema);
        Assert.Contains("\"required\":[\"x\",\"y\"]},\"inferred\":[\"x\",\"y\"]}", schema);
        Assert.Contains("\"inferred\":[\"angle\",\"spread\"]", schema);
        Assert.Contains("\"minimum\":0,\"maximum\":100", Encoding.UTF8.GetString(RangeCommands.Schema));
    }

    [Fact]
    public void TheModelNeverSeesTheInferredList()
    {
        var rendered = new ToolSet(IArena.Calls.Schema).Render();
        Assert.DoesNotContain("inferred", rendered);
        Assert.Contains("\"minimum\":-50", rendered);
    }

    [Fact]
    public void PlanReadsLimitsAndInferredFields()
    {
        var plan = new RepairPlan(new ToolSet(IArena.Calls.Schema), IArena.Calls.Schema);
        var volume = plan.Tools[0].Params[0];
        Assert.Equal(0, volume.Minimum);
        Assert.Equal(100, volume.Maximum);
        Assert.False(volume.Inferred);
        Assert.True(plan.Tools[1].Params[0].Inferred);
        Assert.True(plan.Tools[1].Params[1].Inferred);
        Assert.False(plan.Tools[2].Params[0].Inferred);
        Assert.True(plan.Tools[3].Params[1].Inferred);
    }

    sealed class JsonRig
    {
        readonly ToolSet set = new(IArena.Calls.Schema);
        readonly RepairPlan plan;
        readonly RepairScratch scratch = new(16, 64, 4096);
        readonly RepairText request = new(1024);
        readonly RepairText history = new(8192);

        public JsonRig() => plan = new RepairPlan(set, IArena.Calls.Schema);

        public (string Json, RepairOutcome Outcome, string Ungrounded) Run(string query, string generated)
        {
            var q = Encoding.UTF8.GetBytes(query);
            request.Load(q);
            history.Load(default);
            history.Append(q);
            var buffer = new byte[4096];
            var spans = new CallSpan[16];
            Assert.True(CallRestorer.TryRestore(set.Aliases, Encoding.UTF8.GetBytes(generated), buffer, new byte[64], spans, out var length, out var count));
            var outcome = ArgumentRepair.Apply(plan, scratch, request, history, null, buffer, length, spans, ref count, out var required);
            var ungrounded = Encoding.UTF8.GetString(scratch.Ungrounded, 0, scratch.UngroundedLength).Replace('\n', ' ').Trim();
            return (Encoding.UTF8.GetString(buffer, 0, required), outcome, ungrounded);
        }
    }

    static string Call(string name, string args) => $$"""[{"name":"{{name}}","arguments":{{args}}}]""";

    [Fact]
    public void InRangeCallStandsUntouched()
    {
        var generated = Call("set_volume", """{"level":100}""");
        var (json, outcome, _) = new JsonRig().Run("set the volume to 100", generated);
        Assert.Equal(generated, json);
        Assert.Equal(RepairOutcome.None, outcome);
    }

    [Fact]
    public void OutOfRangeCallIsWithheldNotClamped()
    {
        var (json, outcome, _) = new JsonRig().Run("set the volume to 150", Call("set_volume", """{"level":150}"""));
        Assert.Equal("[]", json);
        Assert.True(outcome.HasFlag(RepairOutcome.Invalid));
        Assert.Equal(ReplyFlags.Invalid, NeedleSession.ToFlags(outcome) & ReplyFlags.Invalid);
    }

    [Fact]
    public void FractionalLimitsHold()
    {
        var (json, outcome, _) = new JsonRig().Run("speed up to 3", Call("speed", """{"factor":3}"""));
        Assert.Equal("[]", json);
        Assert.True(outcome.HasFlag(RepairOutcome.Invalid));
        var kept = Call("speed", """{"factor":2.5}""");
        Assert.Equal(kept, new JsonRig().Run("speed up to 2.5", kept).Json);
    }

    [Fact]
    public void OnlyTheBrokenCallIsDropped()
    {
        var generated = """[{"name":"set_volume","arguments":{"level":400}},{"name":"move_to","arguments":{"x":12.5,"y":-3}}]""";
        var (json, outcome, _) = new JsonRig().Run("set the volume to 400 and go to the tower", generated);
        Assert.Equal("""[{"name":"move_to","arguments":{"x":12.5,"y":-3}}]""", json);
        Assert.True(outcome.HasFlag(RepairOutcome.Invalid));
    }

    [Fact]
    public void InferredCoordinatesPassTheGroundingGate()
    {
        var generated = Call("move_to", """{"x":12.5,"y":-3}""");
        var (json, outcome, ungrounded) = new JsonRig().Run("go to the tower", generated);
        Assert.Equal(generated, json);
        Assert.False(outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.Equal("", ungrounded);
    }

    [Fact]
    public void UninferredCoordinatesAreStillFlagged()
    {
        var (_, outcome, ungrounded) = new JsonRig().Run("go to the tower", Call("move_to_said", """{"x":12.5,"y":-3}"""));
        Assert.True(outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.Equal("move_to_said.x move_to_said.y", ungrounded);
    }

    [Fact]
    public void InferredCoordinatesStillObeyTheirRange()
    {
        var (json, outcome, _) = new JsonRig().Run("go to the tower", Call("move_to", """{"x":80,"y":-3}"""));
        Assert.Equal("[]", json);
        Assert.True(outcome.HasFlag(RepairOutcome.Invalid));
    }

    [Fact]
    public void InferredOptionalValuesAreNotDropped()
    {
        var generated = Call("aim", """{"angle":37.5,"spread":"wedge"}""");
        var (json, outcome, ungrounded) = new JsonRig().Run("aim at the archer", generated);
        Assert.Equal(generated, json);
        Assert.Equal(RepairOutcome.None, outcome);
        Assert.Equal("", ungrounded);
    }

    [Fact]
    public void DeclinedForRangeIsOutOfRange()
    {
        var state = new AgentState(new[] { "set_volume" }, Array.Empty<(int, string)>());
        state.Record(default(Reply) with { Status = ReplyStatus.Declined, Flags = ReplyFlags.Invalid }, state.Calls);
        Assert.Equal(AnswerKind.Declined, state.Kind);
        Assert.Equal(AnswerWhy.OutOfRange, state.Why);
        state.Record(default(Reply) with { Status = ReplyStatus.Declined }, state.Calls);
        Assert.Equal(AnswerWhy.Declined, state.Why);
    }

    // The typed path: the generated grammar drives the generated sink, then CallReplay repairs and replays.
    sealed class TypedRig : IDisposable
    {
        readonly string schema = Encoding.UTF8.GetString(RangeCommands.Schema);
        readonly ToolSet set;
        readonly RepairPlan plan;
        readonly RepairScratch scratch;
        readonly RepairText request = new(1024);
        readonly RepairText history = new(8192);
        readonly byte[] restored = new byte[8192];
        readonly byte[] keyScratch = new byte[64];
        readonly CallSpan[] spans = new CallSpan[16];
        readonly GrammarProgram* program;

        public TypedRig()
        {
            set = new ToolSet(schema);
            plan = new RepairPlan(set, schema);
            scratch = new RepairScratch(spans.Length, 64, restored.Length + plan.LongestValue * 8);
            program = GrammarLoader.Load(RangeCommands.GrammarNodes, RangeCommands.GrammarProperties, RangeCommands.GrammarLiterals,
                RangeCommands.GrammarUnions, RangeCommands.GrammarTools, RangeCommands.GrammarBytes, RangeCommands.GrammarRoot, RangeCommands.GrammarToolCount);
        }

        public void Dispose() => GrammarLoader.Free(program);

        public (RangeCommand[] Commands, RepairOutcome Outcome, int Calls, long Allocated) Run(string query, string generated, int rounds = 0)
        {
            var q = Encoding.UTF8.GetBytes(query);
            request.Load(q);
            history.Load(default);
            history.Append(q);
            var text = Encoding.UTF8.GetBytes(generated);
            var commands = new RangeCommand[8];
            var before = new RangeSink();
            RepairOutcome outcome = default;
            var callCount = 0;
            var stored = 0;
            long allocated = 0;
            for (var round = 0; round <= rounds; round++)
            {
                var sink = before;
                stored = 0;
                var state = Grammar.Start(program);
                foreach (var b in text)
                {
                    Assert.True(Grammar.Advance(program, &state, b, ref sink), generated);
                    if (sink.TryTakeCall(out var call) && stored < commands.Length) commands[stored++] = call;
                }
                Assert.True(Grammar.Complete(state));
                var start = GC.GetAllocatedBytesForCurrentThread();
                outcome = CallReplay.Repair(program, set.Aliases, plan, scratch, request, history, null, text, restored, keyScratch, spans,
                    in before, ref sink, commands.AsSpan(), ref stored, out callCount, out _, out var failure);
                if (round > 0) allocated += GC.GetAllocatedBytesForCurrentThread() - start;
                Assert.Equal(FailureReason.None, failure);
            }
            return (commands[..stored], outcome, callCount, allocated);
        }
    }

    [Fact]
    public void TypedPathWithholdsTheOutOfRangeCommand()
    {
        using var rig = new TypedRig();
        var (commands, outcome, calls, _) = rig.Run("set the volume to 150", Call("loudness", """{"level":150}"""));
        Assert.True(outcome.HasFlag(RepairOutcome.Invalid));
        Assert.Empty(commands);
        Assert.Equal(0, calls);
        Assert.Equal(ReplyStatus.Declined, CallReplay.Status(calls, 0.9f, 0.1f, 0, outcome));
    }

    [Fact]
    public void TypedPathKeepsInferredCoordinates()
    {
        using var rig = new TypedRig();
        var (commands, outcome, calls, _) = rig.Run("go to the tower", Call("walk", """{"x":12.5,"y":-3}"""));
        Assert.False(outcome.HasFlag(RepairOutcome.Ungrounded));
        Assert.Equal(1, calls);
        var command = Assert.Single(commands);
        Assert.Equal(RangeCommandKind.Walk, command.Kind);
        Assert.Equal(12.5f, command.Walk.X);
        Assert.Equal(-3f, command.Walk.Y);
        Assert.Equal(ReplyStatus.Called, CallReplay.Status(calls, 0.9f, 0.1f, 0, outcome));
    }

    [Fact]
    public void RangeChecksAllocateNothingPerTurn()
    {
        using var rig = new TypedRig();
        var generated = """[{"name":"loudness","arguments":{"level":150}},{"name":"walk","arguments":{"x":12.5,"y":-3}}]""";
        var (commands, outcome, _, allocated) = rig.Run("set the volume to 150 and go to the tower", generated, rounds: 20);
        Assert.True(outcome.HasFlag(RepairOutcome.Invalid));
        Assert.Equal(RangeCommandKind.Walk, Assert.Single(commands).Kind);
        Assert.Equal(0, allocated);
    }
}
