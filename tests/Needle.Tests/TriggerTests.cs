using System.Text;
using Xunit;

namespace Needle.Tests;

public sealed unsafe class TriggerTests
{
    const string Tools = """
[{"name":"toggle_light","description":"Turn a light on or off.","triggers":["\\b(turn|switch)\\b.*\\b(on|off)\\b"],"parameters":{"type":"object","properties":{"on":{"type":"boolean"}},"required":["on"]}},
 {"name":"set_alarm","description":"Set an alarm.","parameters":{"type":"object","properties":{"time":{"type":"string"}},"required":["time"]}}]
""";

    static bool Accepts(Grammar grammar, ulong route, string text)
    {
        var state = Grammar.Start(grammar.Program, route);
        foreach (var b in Encoding.UTF8.GetBytes(text))
            if (!Grammar.Advance(grammar.Program, &state, b)) return false;
        return Grammar.Complete(state);
    }

    [Fact]
    public void TriggerMatchesCaseInsensitively()
    {
        var set = new ToolSet(Tools);
        var triggers = Triggers.Compile(set, Tools, 256)!;
        Assert.Equal(1UL, triggers.Match("Please SWITCH the porch light ON"u8));
        Assert.Equal(0UL, triggers.Match("wake me at 7"u8));
    }

    [Fact]
    public void NoTriggersMeansNothingToCompile() =>
        Assert.Null(Triggers.Compile(new ToolSet(Tools.Replace("\"triggers\":[\"\\\\b(turn|switch)\\\\b.*\\\\b(on|off)\\\\b\"],", "")), Tools.Replace("\"triggers\"", "\"x\""), 256));

    [Fact]
    public void RoutedGrammarRequiresACallToTheTriggeredTool()
    {
        var set = new ToolSet(Tools);
        using var grammar = new Grammar(set.Tools);
        Assert.True(Accepts(grammar, 0, "[]"));
        Assert.False(Accepts(grammar, 1, "[]"));
        Assert.False(Accepts(grammar, 1, """[{"name":"set_alarm","arguments":{"time":"7"}}]"""));
        Assert.True(Accepts(grammar, 1, """[{"name":"toggle_light","arguments":{"on":true}},{"name":"set_alarm","arguments":{"time":"7"}}]"""));
    }
}
