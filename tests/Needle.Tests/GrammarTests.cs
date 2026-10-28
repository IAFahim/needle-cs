using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;

namespace Needle.Tests;

public sealed unsafe class GrammarTests : IDisposable
{
    const string Tools = """
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

    readonly Model model;

    public GrammarTests()
    {
        Assets.Require();
        model = Model.Open(Assets.Model);
    }

    public void Dispose() => model.Dispose();

    [Fact]
    public void ToolsAreNormalisedAndAliased()
    {
        var set = new ToolSet(Tools);
        Assert.Equal("set_light", set.Tools[0].GetString("name"));
        Assert.Equal("setLight", set.OriginalName("set_light"));
        Assert.Equal("roomName", set.OriginalProperty("set_light", "room_name"));
        Assert.Contains("\"room_name\"", set.Render());
    }

    [Fact]
    public void RandomWalksThroughTheMaskAlwaysYieldValidCalls()
    {
        var set = new ToolSet(Tools);
        using var grammar = new Grammar(set.Tools);
        using var tokenizer = new Tokenizer(model);
        using var trie = new TokenTrie(tokenizer, model.VocabSize);
        var random = new Random(11);
        var mask = new uint[trie.Words];
        var scratch = new byte[256];
        var completed = 0;
        var names = new HashSet<string>();
        for (var run = 0; run < 150; run++)
        {
            var state = Grammar.Start(grammar.Program);
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
                    Assert.True(Grammar.Advance(grammar.Program, &state, scratch[i]));
                    text.Add(scratch[i]);
                }
            }
            if (!Grammar.Complete(state)) continue;
            completed++;
            var json = JsonValue.Parse(Encoding.UTF8.GetString(text.ToArray()));
            Assert.Equal(JsonKind.Array, json.Kind);
            foreach (var call in json.Items)
            {
                var name = call.GetString("name")!;
                names.Add(name);
                Assert.Contains(name, new[] { "set_light", "ping" });
                var args = call.Get("arguments")!;
                Assert.Equal(JsonKind.Object, args.Kind);
                if (name == "ping") Assert.Empty(args.Members);
                else
                {
                    var keys = args.Members.Select(m => m.Key).ToList();
                    Assert.Equal(keys.Count, keys.Distinct().Count());
                    Assert.Contains("room_name", keys);
                    Assert.Contains("mode", keys);
                    Assert.All(keys, k => Assert.Contains(k, new[] { "room_name", "mode", "level", "tags", "opts", "note" }));
                    Assert.Contains(args.Get("mode")!.Text, new[] { "on", "off", "dim" });
                    if (args.Get("level") is { } level) Assert.Equal(JsonKind.Number, level.Kind);
                    if (args.Get("opts") is { } opts) Assert.NotNull(opts.Get("fade"));
                    if (args.Get("tags") is { } tags) Assert.InRange(tags.Items.Count, 0, 2);
                }
            }
        }
        Assert.True(completed > 40, $"only {completed} walks completed");
        Assert.Equal(2, names.Count);
    }

    [Fact]
    public void ExactCallsAreAcceptedAndNearMissesRejected()
    {
        var set = new ToolSet(Tools);
        using var grammar = new Grammar(set.Tools);
        bool Accepts(string text)
        {
            var state = Grammar.Start(grammar.Program);
            foreach (var b in Encoding.UTF8.GetBytes(text)) if (!Grammar.Advance(grammar.Program, &state, b)) return false;
            return Grammar.Complete(state);
        }
        Assert.True(Accepts("[]"));
        Assert.True(Accepts("[{\"name\":\"ping\",\"arguments\":{}}]"));
        Assert.True(Accepts("[{\"name\":\"set_light\",\"arguments\":{\"room_name\":\"den\",\"mode\":\"dim\",\"level\":-3,\"opts\":{\"fade\":true,\"ratio\":1.5e2},\"note\":null}},{\"name\":\"ping\",\"arguments\":{}}]"));
        Assert.False(Accepts("[{\"name\":\"nope\",\"arguments\":{}}]"));
        Assert.False(Accepts("[{\"name\":\"set_light\",\"arguments\":{\"room_name\":\"den\"}}]"));
        Assert.False(Accepts("[{\"name\":\"set_light\",\"arguments\":{\"mode\":\"loud\",\"room_name\":\"x\"}}]"));
        Assert.False(Accepts("[{\"name\":\"set_light\",\"arguments\":{\"room_name\":\"toolongname\",\"mode\":\"on\"}}]"));
        Assert.False(Accepts("[{\"name\":\"ping\",\"arguments\":{\"x\":1}}]"));
        Assert.False(Accepts("[{\"name\": \"ping\"}]"));
    }
}
