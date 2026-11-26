using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Xunit;

namespace Needle.Tests.Bughunt;

public sealed unsafe class UpstreamBughuntTests
{
    static string Tool(string name, string properties = "{}") =>
        $"{{\"name\":\"{name}\",\"description\":\"d\",\"parameters\":{{\"type\":\"object\",\"properties\":{properties}}}}}";

    static bool Accepts(Grammar grammar, string text)
    {
        var state = Grammar.Start(grammar.Program);
        foreach (var b in Encoding.UTF8.GetBytes(text))
            if (!Grammar.Advance(grammar.Program, &state, b)) return false;
        return Grammar.Complete(state);
    }

    [Fact]
    public void GrammarRejectsToolNamesThatNoToolHasWhenThereAreMoreThanSixtyFourTools()
    {
        var tools = new List<string> { Tool("ax") };
        for (var i = 1; i < 64; i++) tools.Add(Tool("c" + i));
        tools.Add(Tool("by"));
        var set = new ToolSet("[" + string.Join(",", tools) + "]");
        using var grammar = new Grammar(set.Tools);
        Assert.True(Accepts(grammar, "[{\"name\":\"by\",\"arguments\":{}}]"));
        Assert.False(Accepts(grammar, "[{\"name\":\"bx\",\"arguments\":{}}]"));
    }

    [Fact]
    public void GrammarUsesTheSelectedToolsArgumentsPastSixtyFourTools()
    {
        var tools = new List<string> { Tool("ax", "{\"n\":{\"type\":\"integer\"}}") };
        for (var i = 1; i < 64; i++) tools.Add(Tool("c" + i));
        tools.Add(Tool("bx", "{\"s\":{\"type\":\"string\"}}"));
        var set = new ToolSet("[" + string.Join(",", tools) + "]");
        using var grammar = new Grammar(set.Tools);
        Assert.True(Accepts(grammar, "[{\"name\":\"bx\",\"arguments\":{\"s\":\"hi\"}}]"));
    }

    [Fact]
    public void NestedRequiredNamesStayTheNamesOfTheNestedProperties()
    {
        var set = new ToolSet("""
            [{"name":"pick","description":"d","parameters":{"type":"object","properties":{
              "name":{"type":"string"},
              "target":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}},
              "required":["name","target"]}}]
            """);
        var target = set.Tools[0].Get("parameters")!.Get("properties")!.Get("target")!;
        var nestedKeys = target.Get("properties")!.Members.Select(m => m.Key).ToList();
        var nestedRequired = target.Get("required")!.Items.Select(i => i.Text).ToList();
        Assert.All(nestedRequired, r => Assert.Contains(r, nestedKeys));
        Assert.Equal(new[] { "name" }, nestedKeys);
    }

    [Fact]
    public void NestedRequiredIsEnforcedByTheGrammar()
    {
        var set = new ToolSet("""
            [{"name":"pick","description":"d","parameters":{"type":"object","properties":{
              "name":{"type":"string"},
              "target":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}},
              "required":["name","target"]}}]
            """);
        using var grammar = new Grammar(set.Tools);
        Assert.False(Accepts(grammar, "[{\"name\":\"pick\",\"arguments\":{\"name\":\"a\",\"target\":{}}}]"));
    }

    [Fact]
    public void RequiredListedBeforePropertiesNamesTheRightAlias()
    {
        var set = new ToolSet("""
            [{"name":"t","description":"d","parameters":{"type":"object","required":["user_id"],
              "properties":{"userId":{"type":"string"},"user_id":{"type":"integer"}}}}]
            """);
        var parameters = set.Tools[0].Get("parameters")!;
        var required = parameters.Get("required")!.Items.Single().Text;
        Assert.Equal("user_id", set.OriginalProperty("t", required));
    }

    [Fact]
    public void NestedPropertyIsRestoredToItsOwnOriginalName()
    {
        var set = new ToolSet("""
            [{"name":"t","description":"d","parameters":{"type":"object","properties":{
              "inner":{"type":"object","properties":{"user_id":{"type":"string"}}},
              "userId":{"type":"string"}}}}]
            """);
        var inner = set.Tools[0].Get("parameters")!.Get("properties")!.Get("inner")!.Get("properties")!.Members.Single().Key;
        var outer = set.Tools[0].Get("parameters")!.Get("properties")!.Members[1].Key;
        Assert.Equal("user_id", set.OriginalProperty("t", inner));
        Assert.Equal("userId", set.OriginalProperty("t", outer));
    }

    [Theory]
    [InlineData("{\"a\":trux}")]
    [InlineData("[nope]")]
    [InlineData("{\"a\":nul}")]
    [InlineData("{\"a\":\"b")]
    [InlineData("{\"a\":")]
    [InlineData("{")]
    [InlineData("[1,")]
    [InlineData("\"\\u12\"")]
    public void MalformedJsonIsAFormatException(string text)
    {
        Assert.Throws<FormatException>(() => JsonValue.Parse(text));
    }

    static byte[] SyntheticBlob(IList<(string Piece, float Score, byte Type)> pieces, bool dummyPrefix)
    {
        var bytes = new List<byte>();
        void U32(uint v) => bytes.AddRange(BitConverter.GetBytes(v));
        U32((uint)pieces.Count);
        U32(0);
        U32(1);
        U32(2);
        U32(3);
        bytes.Add(dummyPrefix ? (byte)1 : (byte)0);
        bytes.Add(1);
        bytes.Add(0);
        bytes.Add(0);
        foreach (var (piece, score, type) in pieces)
        {
            var raw = Encoding.UTF8.GetBytes(piece);
            bytes.AddRange(BitConverter.GetBytes(score));
            bytes.Add(type);
            bytes.AddRange(BitConverter.GetBytes((ushort)raw.Length));
            bytes.AddRange(raw);
        }
        return bytes.ToArray();
    }

    static Tokenizer SyntheticTokenizer(byte* blob)
    {
        var model = (Model)RuntimeHelpers.GetUninitializedObject(typeof(Model));
        typeof(Model).GetField("TokenizerBlob", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, Pointer.Box(blob, typeof(byte*)));
        return new Tokenizer(model, 4096);
    }

    static List<(string, float, byte)> BaseVocabulary()
    {
        var pieces = new List<(string, float, byte)> { ("<pad>", 0, 2), ("</s>", 0, 2), ("<s>", 0, 2), ("<unk>", 0, 1) };
        for (var b = 0; b < 256; b++) pieces.Add(($"<0x{b:X2}>", 0, 4));
        return pieces;
    }

    [Fact]
    public void TextThatSpellsAControlPieceIsNotEncodedAsThatControlToken()
    {
        var pieces = BaseVocabulary();
        foreach (var p in new[] { "<", "/", "s", ">", "</", "</s" }) pieces.Add((p, -1, 0));
        var blob = SyntheticBlob(pieces, false);
        fixed (byte* p = blob)
        {
            using var tokenizer = SyntheticTokenizer(p);
            var ids = new int[64];
            var count = tokenizer.Encode("</s>"u8, ids, false);
            Assert.DoesNotContain(1, ids[..count]);
        }
    }

    [Fact]
    public void MaskCacheIsNotReusedForADifferentGrammarAtARecycledAddress()
    {
        var pieces = BaseVocabulary();
        foreach (var p in new[] { "[", "{", "\"", "name", "\":\"", "alpha", "beta", "a", "b" }) pieces.Add((p, -1, 0));
        var blob = SyntheticBlob(pieces, false);
        fixed (byte* p = blob)
        {
            using var tokenizer = SyntheticTokenizer(p);
            using var trie = new TokenTrie(tokenizer, tokenizer.PieceCount);
            var alpha = new ToolSet("[" + Tool("alpha") + "]");
            var beta = new ToolSet("[" + Tool("beta") + "]");
            var alphaId = pieces.FindIndex(x => x.Item1 == "alpha");
            var betaId = pieces.FindIndex(x => x.Item1 == "beta");
            var mask = new uint[trie.Words];
            var recycled = false;
            for (var attempt = 0; attempt < 64 && !recycled; attempt++)
            {
                var first = new Grammar(alpha.Tools);
                var address = (nuint)first.Program;
                fixed (uint* m = mask) trie.Allow(first.Program, Advanced(first, "[{\"name\":\""), m);
                var second = new Grammar(beta.Tools);
                first.Dispose();
                var third = new Grammar(beta.Tools);
                second.Dispose();
                if ((nuint)third.Program == address)
                {
                    recycled = true;
                    fixed (uint* m = mask) trie.Allow(third.Program, Advanced(third, "[{\"name\":\""), m);
                }
                third.Dispose();
            }
            Assert.True(recycled, "the allocator never handed a disposed grammar's address to a new one");
            Assert.True((mask[betaId / 32] >> (betaId % 32) & 1) != 0, "beta must be allowed by the grammar now living at that address");
            Assert.False((mask[alphaId / 32] >> (alphaId % 32) & 1) != 0, "alpha belongs to the disposed grammar");
        }
    }

    [Theory]
    [InlineData(typeof(NeedleSession), "Due Date: Friday", true)]
    [InlineData(typeof(NeedleSession), "meeting date : noon", true)]
    [InlineData(typeof(NeedleSession), "ship_date: 2026-12-01 maybe", false)]
    [InlineData(typeof(NeedleSession), "order_date: friday", false)]
    [InlineData(typeof(BatchedSession), "Due Date: Friday", true)]
    [InlineData(typeof(BatchedSession), "order_date: friday", false)]
    public void AutoDatePrefixesTheFactExactlyWhenUpstreamWould(Type session, string system, bool prefixed)
    {
        var build = session.GetMethod("BuildSystem", BindingFlags.Static | BindingFlags.NonPublic)!;
        var text = (string)build.Invoke(null, new object[] { new NeedleOptions { AutoDate = true, System = system } })!;
        Assert.Equal(prefixed, text.StartsWith("date: ", StringComparison.Ordinal) && text.EndsWith("; " + system, StringComparison.Ordinal));
        if (!prefixed) Assert.Equal(system, text);
    }

    static GrammarState Advanced(Grammar grammar, string text)
    {
        var state = Grammar.Start(grammar.Program);
        foreach (var b in Encoding.UTF8.GetBytes(text)) Assert.True(Grammar.Advance(grammar.Program, &state, b));
        return state;
    }
}
