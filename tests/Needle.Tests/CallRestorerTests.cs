using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;

namespace Needle.Tests;

public sealed class CallRestorerTests
{
    const string Tools = """[{"name":"moveCharacter","description":"Move.","parameters":{"type":"object","properties":{"characterName":{"type":"string"},"targetPosition":{"type":"object","properties":{"xCoord":{"type":"number"},"yCoord":{"type":"number"}}},"runFast":{"type":"boolean"},"waypointList":{"type":"array","items":{"type":"string"}},"metaData":{"type":"object"}}}},{"name":"SetTimer","description":"Timer.","parameters":{"type":"object","properties":{"durationMinutes":{"type":"integer"},"label":{"type":"string"}}}},{"name":"ping","description":"Ping.","parameters":{"type":"object","properties":{}}}]""";

    static readonly ToolSet Set = new(Tools);

    static string LegacyRestore(string generated)
    {
        var builder = new StringBuilder("[");
        var first = true;
        foreach (var call in JsonValue.Parse(generated).Items)
        {
            if (!first) builder.Append(',');
            first = false;
            var alias = call.GetString("name")!;
            builder.Append("{\"name\":");
            JsonValue.WriteString(builder, Set.OriginalName(alias));
            builder.Append(",\"arguments\":").Append(Restore(call.Get("arguments")!, alias).ToString()).Append('}');
        }
        return builder.Append(']').ToString();
    }

    static JsonValue Restore(JsonValue value, string tool) => value.Kind switch
    {
        JsonKind.Object => JsonValue.Object(value.Members.ConvertAll(m => new KeyValuePair<string, JsonValue>(Set.OriginalProperty(tool, m.Key), Restore(m.Value, tool)))),
        JsonKind.Array => JsonValue.Array(value.Items.ConvertAll(i => Restore(i, tool))),
        _ => value,
    };

    static bool Restored(ReadOnlySpan<byte> generated, Span<byte> destination, out int required, out int count, CallSpan[]? spans = null)
    {
        var scratch = new byte[Math.Max(64, Set.Aliases.LongestPropertyAlias + 2)];
        return CallRestorer.TryRestore(Set.Aliases, generated, destination, scratch, spans ?? new CallSpan[16], out required, out count);
    }

    public static IEnumerable<object[]> Generated() => new[]
    {
        "[]",
        """[{"name":"ping","arguments":{}}]""",
        """[{"name":"move_character","arguments":{"character_name":"Lyra","target_position":{"x_coord":1.5,"y_coord":-2e3},"run_fast":true,"waypoint_list":["a","b\n\"c\"\\"],"meta_data":{}}}]""",
        """[{"name":"move_character","arguments":{"character_name":"Zo\u00eb \ud83d\ude00 \u0007 caf\u00e9 \u2713 \/ \u0022","waypoint_list":[]}},{"name":"set_timer","arguments":{"duration_minutes":12,"label":"tea"}}]""",
        "[{\"name\":\"move_character\",\"arguments\":{\"character_name\":\"Zoë café ✓ \U0001F600\"}}]",
        """[{"name":"move_character","arguments":{"meta_data":{"character_name":1,"other":[{"x_coord":null},[true,false]],"run_fast":"x"}}}]""",
        """[{"name":"set_timer","arguments":{"duration_minutes":0,"label":"tab\t nul \u0000 bell \u001f"}}]""",
        """[{"name":"set_timer","arguments":{"d\u0075ration_minutes":3}}]""",
    }.Select(g => new object[] { g });

    [Theory]
    [MemberData(nameof(Generated))]
    public void MatchesTheTreeBasedRestoration(string generated)
    {
        var destination = new byte[1024];
        Assert.True(Restored(Encoding.UTF8.GetBytes(generated), destination, out var required, out var count), generated);
        Assert.Equal(LegacyRestore(generated), Encoding.UTF8.GetString(destination, 0, required));
        Assert.Equal(JsonValue.Parse(generated).Items.Count, count);
    }

    [Fact]
    public void LoneSurrogateEscapesBecomeReplacementCharacters()
    {
        var generated = Encoding.UTF8.GetBytes("""[{"name":"set_timer","arguments":{"label":"a\ud800b\udc00c"}}]""");
        var destination = new byte[256];
        Assert.True(Restored(generated, destination, out var required, out _));
        Assert.Equal("[{\"name\":\"SetTimer\",\"arguments\":{\"label\":\"a\uFFFDb\uFFFDc\"}}]", Encoding.UTF8.GetString(destination, 0, required));
    }

    [Fact]
    public void InvalidUtf8InsideStringsBecomesReplacementCharactersLikeTheDecoder()
    {
        var generated = new List<byte>(Encoding.UTF8.GetBytes("[{\"name\":\"set_timer\",\"arguments\":{\"label\":\""));
        generated.AddRange(new byte[] { 0xC3, 0x28, 0xE2, 0x82, 0xF0, 0x9F, 0x98, 0xFF, 0xED, 0xA0, 0x80, 0x41 });
        generated.AddRange(Encoding.UTF8.GetBytes("\"}}]"));
        var destination = new byte[1024];
        Assert.True(Restored(generated.ToArray(), destination, out var required, out _));
        Assert.Equal(LegacyRestore(Encoding.UTF8.GetString(generated.ToArray())), Encoding.UTF8.GetString(destination, 0, required));
    }

    [Fact]
    public void ReportsTheRequiredLengthWhenTheDestinationIsTooSmall()
    {
        var generated = Encoding.UTF8.GetBytes("""[{"name":"move_character","arguments":{"character_name":"Lyra"}}]""");
        var full = new byte[256];
        Assert.True(Restored(generated, full, out var required, out _));
        Assert.True(Restored(generated, new byte[5], out var small, out _));
        Assert.Equal(required, small);
    }

    [Fact]
    public void RecordsWhereEachCallsArgumentsLive()
    {
        var generated = Encoding.UTF8.GetBytes("""[{"name":"set_timer","arguments":{"duration_minutes":12}},{"name":"ping","arguments":{}}]""");
        var destination = new byte[256];
        var spans = new CallSpan[4];
        Assert.True(Restored(generated, destination, out var required, out var count, spans));
        Assert.Equal(2, count);
        Assert.Equal("{\"durationMinutes\":12}", Encoding.UTF8.GetString(destination, spans[0].ArgumentsStart, spans[0].ArgumentsLength));
        Assert.Equal("SetTimer", Set.Aliases.OriginalTool(spans[0].Tool));
        Assert.Equal("{}", Encoding.UTF8.GetString(destination, spans[1].ArgumentsStart, spans[1].ArgumentsLength));
        Assert.Equal("ping", Set.Aliases.OriginalTool(spans[1].Tool));
    }

    [Theory]
    [InlineData("")]
    [InlineData("[")]
    [InlineData("""[{"name":"nobody","arguments":{}}]""")]
    [InlineData("""[{"name":"ping","arguments":{"a":""")]
    [InlineData("""[{"name":"ping","arguments":{}}""")]
    [InlineData("""[{"name":"ping","arguments":{}}]trailing""")]
    [InlineData("""{"name":"ping"}""")]
    public void MalformedInputIsRejectedWithoutThrowing(string generated)
    {
        Assert.False(Restored(Encoding.UTF8.GetBytes(generated), new byte[256], out _, out _));
    }

    [Fact]
    public void AliasesAreTheSnakeCaseNamesAndOriginalsAreEscapedForJson()
    {
        Assert.True(Set.Aliases.TryFindTool("set_timer"u8, out var tool));
        Assert.Equal("SetTimer", Set.Aliases.OriginalTool(tool));
        Assert.True(Set.Aliases.TryFindProperty(tool, "duration_minutes"u8, out var original));
        Assert.Equal("durationMinutes", Encoding.UTF8.GetString(original));
        Assert.False(Set.Aliases.TryFindProperty(tool, "unknown"u8, out _));
    }
}
