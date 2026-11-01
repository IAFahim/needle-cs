using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Needle.Tools;
using Xunit;

namespace Needle.Tests;

[NeedleTools]
public partial interface IGameCommands
{
    [Tool("Get the current weather for a city.")] void GetWeather([Param("City name")] string city);
    [Tool("Set a countdown timer.")] void SetTimer([Param("Length in minutes")] int minutes);
    [Tool("Play a song or artist.")] void PlayMusic([Param("Song or artist")] string query);
}

public enum Item { Potion, Sword, Shield, Arrow, Gold }
public enum Formation { Line, Wedge, Circle, Column }

[NeedleTools]
public partial interface INpcCommands
{
    [Tool("Move the selected unit to a map position.")] void MoveTo(float x, float y);
    [Tool("Attack a target.")] void Attack(Utf8Text64 target);
    [Tool("Give items to a party member.")] void GiveItem(Item item, int count, Utf8Text32 to);
    [Tool("Change the squad formation.")] void SetFormation(Formation formation = Formation.Line);
}

[NeedleTools]
public partial interface IDeviceCommands
{
    [Tool("Toggle the device power.")] void SetPower(bool on, long delayMs = 0);
    [Tool("Calibrate the sensor.")] void Calibrate(double gain);
}

public sealed unsafe class ToolGenTests
{
    const string NpcToolsJson = """
    [{"name":"move_to","description":"Move the selected unit to a map position.","parameters":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"}},"required":["x","y"]}},
     {"name":"attack","description":"Attack a target.","parameters":{"type":"object","properties":{"target":{"type":"string","maxLength":64}},"required":["target"]}},
     {"name":"give_item","description":"Give items to a party member.","parameters":{"type":"object","properties":{"item":{"type":"string","enum":["potion","sword","shield","arrow","gold"]},"count":{"type":"integer"},"to":{"type":"string","maxLength":32}},"required":["item","count","to"]}},
     {"name":"set_formation","description":"Change the squad formation.","parameters":{"type":"object","properties":{"formation":{"type":"string","enum":["line","wedge","circle","column"]}}}}]
    """;

    const string DeviceToolsJson = """
    [{"name":"set_power","description":"Toggle the device power.","parameters":{"type":"object","properties":{"on":{"type":"boolean"},"delay_ms":{"type":"integer"}},"required":["on"]}},
     {"name":"calibrate","description":"Calibrate the sensor.","parameters":{"type":"object","properties":{"gain":{"type":"number"}},"required":["gain"]}}]
    """;

    [Fact]
    public void GameSchemaIsByteIdenticalToTheToolSetRender()
    {
        var rendered = Encoding.UTF8.GetBytes(new ToolSet(File.ReadAllText(Assets.Tools)).Render());
        Assert.True(rendered.AsSpan().SequenceEqual(GameCommands.Schema));
    }

    [Fact]
    public void GeneratedSchemaIsAFixedPointOfToolSet()
    {
        AssertFixedPoint(GameCommands.Schema);
        AssertFixedPoint(NpcCommands.Schema);
        AssertFixedPoint(DeviceCommands.Schema);
    }

    static void AssertFixedPoint(ReadOnlySpan<byte> schema)
    {
        var text = Encoding.UTF8.GetString(schema);
        Assert.Equal(text, new ToolSet(text).Render());
    }

    [Fact]
    public void NpcAndDeviceSchemasMatchTheirEquivalentJson()
    {
        AssertSchemaMatches(NpcToolsJson, NpcCommands.Schema);
        AssertSchemaMatches(DeviceToolsJson, DeviceCommands.Schema);
    }

    static void AssertSchemaMatches(string json, ReadOnlySpan<byte> schema)
    {
        var rendered = Encoding.UTF8.GetBytes(new ToolSet(json).Render());
        Assert.True(rendered.AsSpan().SequenceEqual(schema), Encoding.UTF8.GetString(rendered));
    }

    [Fact]
    public void GeneratedGrammarTablesMatchTheRuntimeGrammar()
    {
        AssertGrammarMatches(NpcToolsJson, NpcCommands.Schema, NpcCommands.GrammarNodes, NpcCommands.GrammarProperties, NpcCommands.GrammarLiterals, NpcCommands.GrammarUnions, NpcCommands.GrammarTools, NpcCommands.GrammarBytes, NpcCommands.GrammarRoot, NpcCommands.GrammarToolCount);
        AssertGrammarMatches(DeviceToolsJson, DeviceCommands.Schema, DeviceCommands.GrammarNodes, DeviceCommands.GrammarProperties, DeviceCommands.GrammarLiterals, DeviceCommands.GrammarUnions, DeviceCommands.GrammarTools, DeviceCommands.GrammarBytes, DeviceCommands.GrammarRoot, DeviceCommands.GrammarToolCount);
        AssertGrammarMatches(File.ReadAllText(Assets.Tools), GameCommands.Schema, GameCommands.GrammarNodes, GameCommands.GrammarProperties, GameCommands.GrammarLiterals, GameCommands.GrammarUnions, GameCommands.GrammarTools, GameCommands.GrammarBytes, GameCommands.GrammarRoot, GameCommands.GrammarToolCount);
    }

    static unsafe void AssertGrammarMatches(
        string json,
        ReadOnlySpan<byte> schema,
        ReadOnlySpan<int> nodes,
        ReadOnlySpan<int> properties,
        ReadOnlySpan<int> literals,
        ReadOnlySpan<int> unions,
        ReadOnlySpan<int> tools,
        ReadOnlySpan<byte> bytes,
        int root,
        int toolCount)
    {
        var text = Encoding.UTF8.GetString(schema);
        using var grammar = new Grammar(new ToolSet(text).Tools);
        var program = grammar.Program;
        Assert.Equal(toolCount, program->ToolCount);
        Assert.Equal(root, program->Root);
        Assert.Equal(nodes.Length, grammar.TableLengths.Nodes);
        AssertEqual(nodes, program->Nodes);
        Assert.Equal(properties.Length, grammar.TableLengths.Properties);
        AssertEqual(properties, program->Properties);
        Assert.Equal(literals.Length, grammar.TableLengths.Literals);
        AssertEqual(literals, program->Literals);
        Assert.Equal(unions.Length, grammar.TableLengths.Unions);
        AssertEqual(unions, program->Unions);
        Assert.Equal(tools.Length, grammar.TableLengths.Tools);
        AssertEqual(tools, program->Tools);
        Assert.Equal(bytes.Length, grammar.TableLengths.Bytes);
        AssertEqual(bytes, program->Bytes);
    }

    static unsafe void AssertEqual(ReadOnlySpan<int> expected, int* actual)
    {
        for (var i = 0; i < expected.Length; i++)
            if (expected[i] != actual[i])
                throw new InvalidOperationException($"int table differs at {i}: generated {expected[i]}, runtime {actual[i]}");
    }

    static unsafe void AssertEqual(ReadOnlySpan<byte> expected, byte* actual)
    {
        for (var i = 0; i < expected.Length; i++)
            if (expected[i] != actual[i])
                throw new InvalidOperationException($"byte table differs at {i}: generated {expected[i]}, runtime {actual[i]}");
    }

    [Fact]
    public void GeneratedGrammarAcceptsTheTrainedCallFormat()
    {
        var program = GrammarLoader.Load(NpcCommands.GrammarNodes, NpcCommands.GrammarProperties, NpcCommands.GrammarLiterals, NpcCommands.GrammarUnions, NpcCommands.GrammarTools, NpcCommands.GrammarBytes, NpcCommands.GrammarRoot, NpcCommands.GrammarToolCount);
        try
        {
            bool Accepts(string text)
            {
                var state = Grammar.Start(program);
                foreach (var b in Encoding.UTF8.GetBytes(text))
                    if (!Grammar.Advance(program, &state, b)) return false;
                return Grammar.Complete(state);
            }
            Assert.True(Accepts("[]"));
            Assert.True(Accepts("[{\"name\":\"move_to\",\"arguments\":{\"x\":1.5,\"y\":-2}}]"));
            Assert.True(Accepts("[{\"name\":\"attack\",\"arguments\":{\"target\":\"Lagos tower\"}}]"));
            Assert.True(Accepts("[{\"name\":\"give_item\",\"arguments\":{\"item\":\"gold\",\"count\":3,\"to\":\"Zoe\"}}]"));
            Assert.True(Accepts("[{\"name\":\"set_formation\",\"arguments\":{\"formation\":\"wedge\"}}]"));
            Assert.True(Accepts("[{\"name\":\"set_formation\",\"arguments\":{}}]"));
            Assert.False(Accepts("[{\"name\":\"nope\",\"arguments\":{}}]"));
            Assert.False(Accepts("[{\"name\":\"give_item\",\"arguments\":{\"item\":\"gold\",\"count\":3}}]"));
            Assert.False(Accepts("[{\"name\":\"set_formation\",\"arguments\":{\"formation\":\"mythril\"}}]"));
            Assert.False(Accepts("[{\"name\":\"attack\",\"arguments\":{\"target\":\"" + new string('x', 65) + "\"}}]"));
        }
        finally
        {
            GrammarLoader.Free(program);
        }
    }

    [Fact]
    public void ArgumentRecordsAreBlittable()
    {
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<GameCommand>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<GetWeatherArgs>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<SetTimerArgs>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<PlayMusicArgs>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<NpcCommand>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<MoveToArgs>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<AttackArgs>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<GiveItemArgs>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<SetFormationArgs>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<Utf8Text64>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<Utf8Text32>());
        Assert.Equal(8, Unsafe.SizeOf<MoveToArgs>());
        Assert.Equal(4 + 4 + 8 + (64 + 4) + (4 + 4 + 36) + 4, Unsafe.SizeOf<NpcCommand>());
        Assert.Equal(64 + 4, Unsafe.SizeOf<Utf8Text64>());
        Assert.True(Marshal.SizeOf<MoveToArgs>() == 8);
    }

    [Fact]
    public void TryBindFillsTypedCommandsAndTracksPresence()
    {
        Assert.True(NpcCommands.TryBind(new FunctionCall("set_formation", "{\"formation\":\"wedge\"}"), out var wedge));
        Assert.Equal(NpcCommandKind.SetFormation, wedge.Kind);
        Assert.Equal(Formation.Wedge, wedge.SetFormation.Formation);
        Assert.Equal(1u, wedge.Presence);

        Assert.True(NpcCommands.TryBind(new FunctionCall("set_formation", "{}"), out var absent));
        Assert.Equal(NpcCommandKind.SetFormation, absent.Kind);
        Assert.Equal(0u, absent.Presence);

        Assert.True(NpcCommands.TryBind(new FunctionCall("give_item", "{\"item\":\"gold\",\"count\":3,\"to\":\"Zoe\"}"), out var give));
        Assert.Equal(Item.Gold, give.GiveItem.Item);
        Assert.Equal(3, give.GiveItem.Count);
        Assert.Equal("Zoe", Encoding.UTF8.GetString(give.GiveItem.To.Span));
        Assert.Equal(0b111u, give.Presence);

        Assert.True(NpcCommands.TryBind(new FunctionCall("move_to", "{\"x\":1.5,\"y\":-2.25}"), out var move));
        Assert.Equal(1.5f, move.MoveTo.X);
        Assert.Equal(-2.25f, move.MoveTo.Y);

        Assert.True(NpcCommands.TryBind(new FunctionCall("attack", "{\"target\":\"Lagos tower\"}"), out var attack));
        Assert.Equal("Lagos tower", Encoding.UTF8.GetString(attack.Attack.Target.Span));

        Assert.True(DeviceCommands.TryBind(new FunctionCall("set_power", "{\"on\":true}"), out var power));
        Assert.True(power.SetPower.On);
        Assert.Equal(0L, power.SetPower.DelayMs);
        Assert.Equal(1u, power.Presence);

        Assert.False(NpcCommands.TryBind(new FunctionCall("nope", "{}"), out _));
        Assert.False(GameCommands.TryBind(new FunctionCall("nope", "{}"), out _));
    }

    [Fact]
    public void DispatchInvokesTheRightMethodsWithDefaultsApplied()
    {
        var handler = new NpcHandler();
        Assert.True(NpcCommands.TryBind(new FunctionCall("move_to", "{\"x\":12,\"y\":34}"), out var move));
        NpcCommands.Dispatch(ref handler, move);
        Assert.True(NpcCommands.TryBind(new FunctionCall("attack", "{\"target\":\"ogre\"}"), out var attack));
        NpcCommands.Dispatch(ref handler, attack);
        Assert.True(NpcCommands.TryBind(new FunctionCall("give_item", "{\"item\":\"potion\",\"count\":2,\"to\":\"Zoë ✓\"}"), out var give));
        NpcCommands.Dispatch(ref handler, give);
        Assert.True(NpcCommands.TryBind(new FunctionCall("set_formation", "{\"formation\":\"wedge\"}"), out var wedge));
        NpcCommands.Dispatch(ref handler, wedge);
        Assert.True(NpcCommands.TryBind(new FunctionCall("set_formation", "{}"), out var fallback));
        NpcCommands.Dispatch(ref handler, fallback);

        Assert.Equal(new[] { "move 12 34", "attack ogre", "Potion x2 to Zoë ✓" }, handler.Log);
        Assert.Equal(Formation.Wedge, handler.Formations[0]);
        Assert.Equal(Formation.Line, handler.Formations[1]);

        var devices = new DeviceHandler();
        Assert.True(DeviceCommands.TryBind(new FunctionCall("set_power", "{\"on\":true,\"delayMs\":7}"), out var power));
        DeviceCommands.Dispatch(ref devices, power);
        Assert.True(devices.LastOn);
        Assert.Equal(7L, devices.LastDelayMs);
        Assert.True(DeviceCommands.TryBind(new FunctionCall("set_power", "{\"on\":false}"), out var off));
        DeviceCommands.Dispatch(ref devices, off);
        Assert.False(devices.LastOn);
        Assert.Equal(0L, devices.LastDelayMs);
        Assert.True(DeviceCommands.TryBind(new FunctionCall("calibrate", "{\"gain\":1.25}"), out var calibrate));
        DeviceCommands.Dispatch(ref devices, calibrate);
        Assert.Equal(1.25, devices.LastGain);
    }

    [Fact]
    public void Utf8TextSlotsTruncateAndSurviveRoundTrips()
    {
        var slot = new Utf8Text64();
        slot.Set("hello");
        Assert.Equal(5, slot.Length);
        Assert.Equal("hello", Encoding.UTF8.GetString(slot.Span));
        Assert.Equal(64, Utf8Text64.Capacity);
        var long_ = new Utf8Text32();
        long_.Set(new string('ß', 40));
        Assert.Equal(32, long_.Length);
        Assert.True(long_.Span.Length == 32);
        long_.Set("abcdefghij");
        Assert.Equal("abcdefghij", long_.ToString());
    }
}

internal struct NpcHandler : INpcCommands
{
    public List<string> Log = new();
    public List<Formation> Formations = new();

    public NpcHandler()
    {
        Log = new List<string>();
        Formations = new List<Formation>();
    }

    public void MoveTo(float x, float y) => Log.Add($"move {x} {y}");
    public void Attack(Utf8Text64 target) => Log.Add("attack " + Encoding.UTF8.GetString(target.Span));
    public void GiveItem(Item item, int count, Utf8Text32 to) => Log.Add($"{item} x{count} to {Encoding.UTF8.GetString(to.Span)}");
    public void SetFormation(Formation formation) => Formations.Add(formation);
}

internal struct DeviceHandler : IDeviceCommands
{
    public bool LastOn;
    public long LastDelayMs = -1;
    public double LastGain;

    public DeviceHandler() { }

    public void SetPower(bool on, long delayMs = 0)
    {
        LastOn = on;
        LastDelayMs = delayMs;
    }

    public void Calibrate(double gain) => LastGain = gain;
}

public sealed class ToolGenSessionTests : IDisposable
{
    readonly Model model;

    public ToolGenSessionTests()
    {
        Assets.Require();
        model = Model.Open(Assets.Model);
    }

    public void Dispose() => model.Dispose();

    [Fact]
    public void GeneratedSchemaFeedsTheSessionAndCallsDispatchWithoutJsonOrReflection()
    {
        using var session = ToolSession.Open<GameToolSet>(model, new NeedleOptions { Threads = 4 });
        var handler = new GameHandler();

        var weather = session.Complete("what's it like in Lagos right now?");
        Assert.True(weather.Success);
        var call = Assert.Single(weather.FunctionCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.True(GameCommands.TryBind(call, out var command));
        GameCommands.Dispatch(ref handler, command);
        Assert.Equal("Lagos", handler.LastCity);

        var chained = session.Complete("set a timer for 12 minutes and play some Miles Davis");
        Assert.Equal(new[] { "set_timer", "play_music" }, chained.FunctionCalls.Select(c => c.Name));
        foreach (var chainedCall in chained.FunctionCalls)
        {
            Assert.True(GameCommands.TryBind(chainedCall, out command));
            GameCommands.Dispatch(ref handler, command);
        }
        Assert.Equal(12, handler.LastMinutes);
        Assert.Equal("Miles Davis", handler.LastQuery);
    }
}

internal struct GameHandler : IGameCommands
{
    public string LastCity = "";
    public int LastMinutes = -1;
    public string LastQuery = "";

    public GameHandler() { }

    public void GetWeather(string city) => LastCity = city;
    public void SetTimer(int minutes) => LastMinutes = minutes;
    public void PlayMusic(string query) => LastQuery = query;
}
