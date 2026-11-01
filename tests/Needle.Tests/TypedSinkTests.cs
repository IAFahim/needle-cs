using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using Needle.Tools;
using Xunit;

namespace Needle.Tests;

struct ProtocolSink : IGrammarSink
{
    public int Tools;
    public int Properties;
    public int Values;
    public int Calls;
    public int ToolCount;
    bool inCall;

    public bool Balanced => !inCall;

    public void ToolSelected(int tool)
    {
        Assert.True(tool >= 0 && tool < ToolCount);
        Assert.False(inCall);
        inCall = true;
        Tools++;
    }

    public void PropertySelected(int property)
    {
        Assert.True(inCall);
        Properties++;
    }

    public void StringByte(byte value)
    {
        Assert.True(inCall);
    }

    public void NumberByte(byte value)
    {
        Assert.True(inCall);
    }

    public void EnumSelected(int literal)
    {
        Assert.True(inCall);
    }

    public void BoolSelected(bool value)
    {
        Assert.True(inCall);
    }

    public void NullSelected()
    {
        Assert.True(inCall);
    }

    public void ValueCompleted()
    {
        Assert.True(inCall);
        Values++;
    }

    public void CallCompleted()
    {
        Assert.True(inCall);
        inCall = false;
        Calls++;
    }
}

public sealed unsafe class TypedSinkTests : IDisposable
{
    readonly Model model;

    public TypedSinkTests()
    {
        Assets.Require();
        model = Model.Open(Assets.Model);
    }

    public void Dispose() => model.Dispose();

    static GrammarProgram* LoadNpc()
    {
        return GrammarLoader.Load(NpcCommands.GrammarNodes, NpcCommands.GrammarProperties, NpcCommands.GrammarLiterals,
            NpcCommands.GrammarUnions, NpcCommands.GrammarTools, NpcCommands.GrammarBytes,
            NpcCommands.GrammarRoot, NpcCommands.GrammarToolCount);
    }

    static void Feed<TSink>(GrammarProgram* program, string text, ref TSink sink) where TSink : struct, IGrammarSink
    {
        var state = Grammar.Start(program);
        var bytes = Encoding.UTF8.GetBytes(text);
        for (var i = 0; i < bytes.Length; i++)
            Assert.True(Grammar.Advance(program, &state, bytes[i], ref sink), $"typed advance rejected byte {i} of {text}");
        Assert.True(Grammar.Complete(state), $"typed advance left {text} incomplete");
    }

    [Fact]
    public void NoSinkMatchesPlainAdvanceOnRandomWalks()
    {
        const string tools = """
            [{"name":"setLight","description":"d","parameters":{"type":"object","properties":{
            "roomName":{"type":"string","maxLength":6},
            "mode":{"type":"string","enum":["on","off","dim"]},
            "level":{"type":"integer"},
            "tags":{"type":"array","items":{"type":"string","maxLength":3},"maxItems":2},
            "opts":{"type":"object","properties":{"fade":{"type":"boolean"},"ratio":{"type":"number"}},"required":["fade"]},
            "note":{"type":["string","null"],"maxLength":4}},
            "required":["roomName","mode"]}},
            {"name":"ping","description":"d","parameters":{"type":"object","properties":{}}}]
            """;
        var set = new ToolSet(tools);
        using var grammar = new Grammar(set.Tools);
        using var tokenizer = new Tokenizer(model);
        using var trie = new TokenTrie(tokenizer, model.VocabSize);
        var random = new Random(11);
        var mask = new uint[trie.Words];
        var scratch = new byte[256];
        var completed = 0;
        for (var run = 0; run < 150; run++)
        {
            var state = Grammar.Start(grammar.Program);
            var trail = Grammar.Start(grammar.Program);
            var probe = new ProtocolSink { ToolCount = grammar.Program->ToolCount };
            var text = new List<byte>();
            for (var step = 0; step < 400 && !Grammar.Complete(state); step++)
            {
                fixed (uint* m = mask) trie.Allow(grammar.Program, state, m);
                var allowed = new List<int>();
                for (var id = 0; id < model.VocabSize; id++) if ((mask[id / 32] >> (id % 32) & 1) != 0) allowed.Add(id);
                Assert.True(allowed.Count > 0, $"dead state after: {Encoding.UTF8.GetString(text.ToArray())} (depth {state.Depth})");
                var pick = allowed[random.Next(allowed.Count)];
                int length;
                fixed (byte* p = scratch) length = tokenizer.ConstraintBytes(pick, p);
                for (var i = 0; i < length; i++)
                {
                    var plain = Grammar.Advance(grammar.Program, &state, scratch[i]);
                    var typed = Grammar.Advance(grammar.Program, &trail, scratch[i], ref probe);
                    Assert.Equal(plain, typed);
                    Assert.True(plain);
                    text.Add(scratch[i]);
                }
            }
            if (!Grammar.Complete(state)) continue;
            completed++;
            Assert.True(Grammar.Complete(trail));
            Assert.Equal(probe.Tools, probe.Calls);
            Assert.True(probe.Balanced);
        }
        Assert.True(completed > 40, $"only {completed} walks completed");
    }

    [Fact]
    public void TypedAndPlainAdvanceAgreeOnNearMisses()
    {
        var program = LoadNpc();
        try
        {
            var misses = new[]
            {
                """[{"name":"nope","arguments":{}}]""",
                """[{"name":"move_to","arguments":{"x":1.5}}]""",
                "[{\"name\":\"attack\",\"arguments\":{\"target\":\"" + new string('x', 65) + "\"}}]",
                """[{"name":"set_formation","arguments":{"formation":"mythril"}}]""",
                """[{"name":"give_item","arguments":{"item":"gold","count":3}}]""",
                """[{"name":"move_to","arguments":{"x":01,"y":2}}]""",
                """[{"name":"move_to","arguments":{"x":1.5,"y":}}]""",
                """[{"name":"move_to","arguments":{"x":1.5,"y":2}}""",
            };
            foreach (var text in misses)
            {
                var bytes = Encoding.UTF8.GetBytes(text);
                var plain = Grammar.Start(program);
                var plainOk = true;
                foreach (var b in bytes) plainOk &= Grammar.Advance(program, &plain, b);
                plainOk &= Grammar.Complete(plain);
                var typed = Grammar.Start(program);
                var probe = new ProtocolSink { ToolCount = NpcCommands.GrammarToolCount };
                var typedOk = true;
                foreach (var b in bytes) typedOk &= Grammar.Advance(program, &typed, b, ref probe);
                typedOk &= Grammar.Complete(typed);
                Assert.Equal(plainOk, typedOk);
                Assert.False(plainOk, text);
            }
        }
        finally
        {
            GrammarLoader.Free(program);
        }
    }

    [Fact]
    public void SinksAreBlittable()
    {
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<NpcSink>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<DeviceSink>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<GameSink>());
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
        Assert.Equal(expected.GiveItem.To.Length, actual.GiveItem.To.Length);
        Assert.True(expected.GiveItem.To.Span.SequenceEqual(actual.GiveItem.To.Span));
        Assert.Equal(expected.SetFormation.Formation, actual.SetFormation.Formation);
    }

    static void AssertDeviceEqual(DeviceCommand expected, DeviceCommand actual)
    {
        Assert.Equal(expected.Kind, actual.Kind);
        Assert.Equal(expected.Presence, actual.Presence);
        Assert.Equal(expected.SetPower.On, actual.SetPower.On);
        Assert.Equal(expected.SetPower.DelayMs, actual.SetPower.DelayMs);
        Assert.Equal(expected.Calibrate.Gain, actual.Calibrate.Gain);
    }

    static void AssertGameEqual(GameCommand expected, GameCommand actual)
    {
        Assert.Equal(expected.Kind, actual.Kind);
        Assert.Equal(expected.Presence, actual.Presence);
        Assert.Equal(expected.GetWeather.City.Length, actual.GetWeather.City.Length);
        Assert.True(expected.GetWeather.City.Span.SequenceEqual(actual.GetWeather.City.Span));
        Assert.Equal(expected.SetTimer.Minutes, actual.SetTimer.Minutes);
        Assert.Equal(expected.PlayMusic.Query.Length, actual.PlayMusic.Query.Length);
        Assert.True(expected.PlayMusic.Query.Span.SequenceEqual(actual.PlayMusic.Query.Span));
    }

    static NpcCommand BindNpc(string name, string arguments)
    {
        Assert.True(NpcCommands.TryBind(new FunctionCall(name, arguments), out var command), name + arguments);
        return command;
    }

    static NpcCommand FeedNpc(string calls, out int count)
    {
        var program = LoadNpc();
        try
        {
            var sink = new NpcSink();
            Feed(program, calls, ref sink);
            count = sink.CallCount;
            Assert.True(sink.TryTakeCall(out var first));
            return first;
        }
        finally
        {
            GrammarLoader.Free(program);
        }
    }

    [Fact]
    public void TypedSinkMatchesTryBindOnFixtures()
    {
        var cases = new (string Name, string Arguments)[]
        {
            ("move_to", """{"x":1.5,"y":-2.25}"""),
            ("move_to", """{"x":12,"y":34}"""),
            ("move_to", """{"x":1.5e2,"y":-0.5}"""),
            ("move_to", """{"x":0,"y":-0.0}"""),
            ("move_to", """{"x":12.0,"y":1E-3}"""),
            ("attack", """{"target":"Lagos tower"}"""),
            ("attack", """{"target":"Zoë ✓ café"}"""),
            ("attack", """{"target":"a\"b\\c\nd\te\/f\bg\u0041\u00e9"}"""),
            ("attack", """{"target":"😀 raw astral"}"""),
            ("attack", """{"target":"tab\there"}"""),
            ("give_item", """{"item":"gold","count":3,"to":"Zoe"}"""),
            ("give_item", """{"item":"potion","count":0,"to":"Lyra"}"""),
            ("give_item", """{"item":"sword","count":-12,"to":"Bo"}"""),
            ("give_item", """{"item":"arrow","count":99999999999999999999999,"to":"Zoe"}"""),
            ("give_item", """{"to":"Zoe","count":2,"item":"shield"}"""),
            ("set_formation", """{"formation":"wedge"}"""),
            ("set_formation", """{"formation":"line"}"""),
            ("set_formation", """{"formation":"circle"}"""),
            ("set_formation", """{"formation":"column"}"""),
            ("set_formation", """{}"""),
        };
        foreach (var (name, arguments) in cases)
        {
            var expected = BindNpc(name, arguments);
            var actual = FeedNpc($"[{{\"name\":\"{name}\",\"arguments\":{arguments}}}]", out var count);
            Assert.Equal(1, count);
            AssertNpcEqual(expected, actual);
        }
    }

    [Fact]
    public void TypedSinkTruncatesTextLikeTryBind()
    {
        var longUnicode = new string('é', 40);
        AssertNpcEqual(BindNpc("attack", $"{{\"target\":\"{longUnicode}\"}}"),
            FeedNpc($"[{{\"name\":\"attack\",\"arguments\":{{\"target\":\"{longUnicode}\"}}}}]", out _));
        var boundary = new string('a', 62) + "éé";
        AssertNpcEqual(BindNpc("attack", $"{{\"target\":\"{boundary}\"}}"),
            FeedNpc($"[{{\"name\":\"attack\",\"arguments\":{{\"target\":\"{boundary}\"}}}}]", out _));
        var split = new string('a', 63) + "é";
        AssertNpcEqual(BindNpc("attack", $"{{\"target\":\"{split}\"}}"),
            FeedNpc($"[{{\"name\":\"attack\",\"arguments\":{{\"target\":\"{split}\"}}}}]", out _));
    }

    [Fact]
    public void TypedSinkMatchesTryBindForDevicesAndGames()
    {
        var program = GrammarLoader.Load(DeviceCommands.GrammarNodes, DeviceCommands.GrammarProperties, DeviceCommands.GrammarLiterals,
            DeviceCommands.GrammarUnions, DeviceCommands.GrammarTools, DeviceCommands.GrammarBytes,
            DeviceCommands.GrammarRoot, DeviceCommands.GrammarToolCount);
        try
        {
            var cases = new (string Name, string Arguments, string BindArguments)[]
            {
                ("set_power", """{"on":true}""", """{"on":true}"""),
                ("set_power", """{"on":false}""", """{"on":false}"""),
                ("set_power", """{"on":true,"delay_ms":7}""", """{"on":true,"delayMs":7}"""),
                ("set_power", """{"on":false,"delay_ms":-5}""", """{"on":false,"delayMs":-5}"""),
                ("set_power", """{"on":true,"delay_ms":99999999999999999999}""", """{"on":true,"delayMs":99999999999999999999}"""),
                ("calibrate", """{"gain":1.25}""", """{"gain":1.25}"""),
                ("calibrate", """{"gain":1e3}""", """{"gain":1e3}"""),
                ("calibrate", """{"gain":-2.5e-3}""", """{"gain":-2.5e-3}"""),
            };
            foreach (var (name, arguments, bind) in cases)
            {
                Assert.True(DeviceCommands.TryBind(new FunctionCall(name, bind), out var expected), name + bind);
                var sink = new DeviceSink();
                Feed(program, $"[{{\"name\":\"{name}\",\"arguments\":{arguments}}}]", ref sink);
                Assert.Equal(1, sink.CallCount);
                Assert.True(sink.TryTakeCall(out var actual));
                AssertDeviceEqual(expected, actual);
            }
        }
        finally
        {
            GrammarLoader.Free(program);
        }

        var game = GrammarLoader.Load(GameCommands.GrammarNodes, GameCommands.GrammarProperties, GameCommands.GrammarLiterals,
            GameCommands.GrammarUnions, GameCommands.GrammarTools, GameCommands.GrammarBytes,
            GameCommands.GrammarRoot, GameCommands.GrammarToolCount);
        try
        {
            var cases = new (string Name, string Arguments)[]
            {
                ("get_weather", """{"city":"Lagos"}"""),
                ("get_weather", """{"city":"München ✓"}"""),
                ("set_timer", """{"minutes":12}"""),
                ("set_timer", """{"minutes":-3}"""),
                ("play_music", """{"query":"Miles Davis"}"""),
                ("get_weather", "{\"city\":\"" + new string('y', 200) + "\"}"),
            };
            foreach (var (name, arguments) in cases)
            {
                Assert.True(GameCommands.TryBind(new FunctionCall(name, arguments), out var expected), name + arguments);
                var sink = new GameSink();
                Feed(game, $"[{{\"name\":\"{name}\",\"arguments\":{arguments}}}]", ref sink);
                Assert.Equal(1, sink.CallCount);
                Assert.True(sink.TryTakeCall(out var actual));
                AssertGameEqual(expected, actual);
            }
        }
        finally
        {
            GrammarLoader.Free(game);
        }
    }

    [Fact]
    public void StreamingSurfacesFirstCallBeforeSecondDecodes()
    {
        var program = GrammarLoader.Load(NpcCommands.GrammarNodes, NpcCommands.GrammarProperties, NpcCommands.GrammarLiterals,
            NpcCommands.GrammarUnions, NpcCommands.GrammarTools, NpcCommands.GrammarBytes,
            NpcCommands.GrammarRoot, NpcCommands.GrammarToolCount);
        try
        {
            const string calls = """[{"name":"set_formation","arguments":{"formation":"wedge"}},{"name":"attack","arguments":{"target":"ogre"}}]""";
            var bytes = Encoding.UTF8.GetBytes(calls);
            var state = Grammar.Start(program);
            var sink = new NpcSink();
            var firstAt = -1;
            NpcCommand first = default;
            for (var i = 0; i < bytes.Length; i++)
            {
                Assert.True(Grammar.Advance(program, &state, bytes[i], ref sink));
                if (firstAt < 0 && sink.TryTakeCall(out first)) firstAt = i;
            }
            Assert.True(Grammar.Complete(state));
            Assert.True(firstAt >= 0);
            Assert.Equal(2, sink.CallCount);
            Assert.True(sink.TryTakeCall(out var second));
            Assert.False(sink.TryTakeCall(out _));
            var rest = Encoding.UTF8.GetString(bytes, firstAt + 1, bytes.Length - firstAt - 1);
            Assert.Contains("\"attack\"", rest);
            Assert.Equal(NpcCommandKind.SetFormation, first.Kind);
            Assert.Equal(Formation.Wedge, first.SetFormation.Formation);
            Assert.Equal(NpcCommandKind.Attack, second.Kind);
            Assert.Equal("ogre", Encoding.UTF8.GetString(second.Attack.Target.Span));
        }
        finally
        {
            GrammarLoader.Free(program);
        }
    }

    void AssertSessionMatches(string prompt, int threads, int maxCommands = 8)
    {
        var schema = Encoding.UTF8.GetString(NpcCommands.Schema);
        using var typed = ToolSession.Open<NpcToolSet>(model, new NeedleOptions { Threads = threads, AutoDate = false });
        using var json = new NeedleSession(model, schema, new NeedleOptions { Threads = threads, AutoDate = false });
        var query = Encoding.UTF8.GetBytes(prompt);
        var sink = new NpcSink();
        var commands = new NpcCommand[maxCommands];
        var typedReply = typed.Complete(query, commands.AsSpan(), ref sink);
        var calls = new byte[1 << 16];
        var jsonReply = json.Complete(query, calls.AsSpan());
        Assert.Equal(jsonReply.Status, typedReply.Status);
        Assert.Equal(jsonReply.CallCount, typedReply.CallCount);
        Assert.True(typedReply.CallCount > 0, prompt);
        Assert.Equal(jsonReply.Confidence, typedReply.Confidence);
        Assert.Equal(jsonReply.DecodeTokens, typedReply.DecodeTokens);
        var restored = Encoding.UTF8.GetString(calls, 0, jsonReply.Utf8Length);
        var parsed = JsonValue.Parse(restored);
        Assert.Equal(parsed.Items.Count, typedReply.CallCount);
        for (var i = 0; i < typedReply.CallCount; i++)
        {
            var call = parsed.Items[i];
            var expected = BindNpc(call.GetString("name")!, call.Get("arguments")!.ToString());
            AssertNpcEqual(expected, commands[i]);
        }
    }

    [Fact]
    public void SessionTypedMatchesJsonPathOnGamePrompts()
    {
        AssertSessionMatches("give Lyra 3 potions", 4);
        AssertSessionMatches("switch to wedge formation and follow the captain", 4);
        AssertSessionMatches("move to 120, 45", 4);
        AssertSessionMatches("form a circle", 4);
    }

    [Fact]
    public void SessionTypedMatchesJsonPathOnGoldenPrompts()
    {
        using var typed = ToolSession.Open<GameToolSet>(model, new NeedleOptions { Threads = 4, AutoDate = false });
        using var json = new NeedleSession(model, Encoding.UTF8.GetString(GameCommands.Schema), new NeedleOptions { Threads = 4, AutoDate = false });
        foreach (var prompt in new[] { "what's it like in Lagos right now?", "set a timer for 12 minutes and play some Miles Davis" })
        {
            var query = Encoding.UTF8.GetBytes(prompt);
            var sink = new GameSink();
            var commands = new GameCommand[8];
            var typedReply = typed.Complete(query, commands.AsSpan(), ref sink);
            var calls = new byte[1 << 16];
            var jsonReply = json.Complete(query, calls.AsSpan());
            Assert.Equal(jsonReply.Status, typedReply.Status);
            Assert.Equal(jsonReply.CallCount, typedReply.CallCount);
            Assert.True(typedReply.CallCount > 0, prompt);
            Assert.Equal(jsonReply.Confidence, typedReply.Confidence);
            var restored = Encoding.UTF8.GetString(calls, 0, jsonReply.Utf8Length);
            var parsed = JsonValue.Parse(restored);
            for (var i = 0; i < typedReply.CallCount; i++)
            {
                var call = parsed.Items[i];
                Assert.True(GameCommands.TryBind(new FunctionCall(call.GetString("name")!, call.Get("arguments")!.ToString()), out var expected));
                AssertGameEqual(expected, commands[i]);
            }
        }
    }

    [Fact]
    public void RefusedTypedCallsSurfaceBeforeTheConfidenceGate()
    {
        const string prompt = "what's it like in Lagos right now?";
        var query = Encoding.UTF8.GetBytes(prompt);
        using var refused = ToolSession.Open<GameToolSet>(model, new NeedleOptions { Threads = 4, AutoDate = false, RefuseBelow = 2f });
        var refusedSink = new GameSink();
        var refusedCommands = new GameCommand[8];
        var refusedReply = refused.Complete(query, refusedCommands.AsSpan(), ref refusedSink);
        Assert.Equal(ReplyStatus.Refused, refusedReply.Status);
        Assert.Equal(1, refusedReply.CallCount);
        using var called = ToolSession.Open<GameToolSet>(model, new NeedleOptions { Threads = 4, AutoDate = false });
        var calledSink = new GameSink();
        var calledCommands = new GameCommand[8];
        var calledReply = called.Complete(query, calledCommands.AsSpan(), ref calledSink);
        Assert.Equal(ReplyStatus.Called, calledReply.Status);
        AssertGameEqual(calledCommands[0], refusedCommands[0]);
    }

    [Fact]
    public void TypedCallsAllocateNothingManaged()
    {
        using var session = ToolSession.Open<GameToolSet>(model, new NeedleOptions { Threads = 1, AutoDate = false });
        var query = Encoding.UTF8.GetBytes("what's it like in Lagos right now?");
        for (var round = 0; round < 3; round++)
        {
            session.Reset();
            var warmSink = new GameSink();
            var warm = new GameCommand[8];
            var reply = session.Complete(query, warm.AsSpan(), ref warmSink);
            Assert.Equal(ReplyStatus.Called, reply.Status);
        }
        session.Reset();
        var commands = new GameCommand[8];
        var before = GC.GetAllocatedBytesForCurrentThread();
        var sink = new GameSink();
        var result = session.Complete(query, commands.AsSpan(), ref sink);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(ReplyStatus.Called, result.Status);
        Assert.Equal(0, allocated);
    }
}
