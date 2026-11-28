using System;
using System.ComponentModel;
using System.IO;
using System.Text;
using Needle.Tools;
using Xunit;

namespace Needle.Tests;

[Tools]
public partial interface IKitchen
{
    [Description("Get the current weather for a city.")] void GetWeather([Description("City name")] string city);
    [Description("Set a countdown timer.")] void SetTimer([Description("Length in minutes")] int minutes);
    [Description("Play a song or artist.")] void PlayMusic([Description("Song or artist")] string query);
}

public enum Shape { Line, Wedge, Circle }

[Tools]
public partial interface ISquad
{
    [Description("Move to a map position.")] void MoveTo(float x, float y);
    [Description("Give items to a member.")] void Give(Utf8Text32 member, int count = 1);
    [Description("Change formation.")] void Form(Shape shape = Shape.Line);
    [Description("Toggle stealth.")] void Stealth(bool on, double fade = 0.5);
}

public class ToolsInterfaceTests
{
    [Fact]
    public void SchemaMatchesTheHandWrittenTools()
    {
        var rendered = new ToolSet(File.ReadAllText(Assets.Tools)).Render();
        Assert.Equal(IKitchen.Calls.Schema, rendered);
        var attribute = (ToolSchemaAttribute)typeof(IKitchen).GetCustomAttributes(typeof(ToolSchemaAttribute), false)[0];
        Assert.Equal(IKitchen.Calls.Schema, attribute.Json);
        Assert.Equal(IKitchen.Calls.Schema, ToolCatalog<IKitchen>.Json);
    }

    [Fact]
    public void ArgumentsReadFromCalls()
    {
        using var fake = FakeCalls.Of(ISquad.Calls.Schema,
            ("move_to", "{\"x\":1.5,\"y\":-2}"),
            ("give", "{\"member\":\"Lyra the Very Long Named Ranger of the North\"}"),
            ("form", "{\"shape\":\"wedge\"}"),
            ("stealth", "{\"on\":true}"),
            ("form", "{\"shape\":\"square\"}"));
        var answer = fake.Answer;
        Assert.True(answer.Calls[0].Is(out ISquad.MoveToArgs move));
        Assert.Equal(1.5f, move.X);
        Assert.Equal(-2f, move.Y);
        Assert.False(answer.Calls[0].Is(out ISquad.GiveArgs _));
        Assert.True(answer.Calls[1].Is(out ISquad.GiveArgs give));
        Assert.Equal(32, give.Member.Length);
        Assert.Equal("Lyra the Very Long Named Ranger ", give.Member.ToString());
        Assert.Equal(1, give.Count);
        Assert.True(answer.Calls[2].Is(out ISquad.FormArgs form));
        Assert.Equal(Shape.Wedge, form.Shape);
        Assert.True(answer.Calls[3].Is(out ISquad.StealthArgs stealth));
        Assert.True(stealth.On);
        Assert.Equal(0.5, stealth.Fade);
        Assert.False(answer.Calls[4].Is(out ISquad.FormArgs _));
    }

    sealed class Recorder : ISquad
    {
        public readonly StringBuilder Log = new();
        public void MoveTo(float x, float y) => Log.Append($"move {x} {y};");
        public void Give(Utf8Text32 member, int count = 1) => Log.Append($"give {member} {count};");
        public void Form(Shape shape = Shape.Line) => Log.Append($"form {shape};");
        public void Stealth(bool on, double fade = 0.5) => Log.Append($"stealth {on} {fade};");
    }

    [Fact]
    public void DispatchInvokesTheImplementation()
    {
        using var fake = FakeCalls.Of(ISquad.Calls.Schema, ("give", "{\"member\":\"Lyra\",\"count\":3}"), ("form", "{\"shape\":\"circle\"}"), ("form", "{}"));
        var recorder = new Recorder();
        Assert.Equal(3, ISquad.Calls.Dispatch(fake.Answer, recorder));
        Assert.Equal("give Lyra 3;form Circle;form Line;", recorder.Log.ToString());
    }

    [Fact]
    public void TextIsCutAtAWholeCharacter()
    {
        Span<byte> buffer = stackalloc byte[4];
        JsonArgs.UnescapeTruncating("abcé"u8, buffer, out var written);
        Assert.Equal(3, written);
        JsonArgs.UnescapeTruncating("ab\\u00e9x"u8, buffer, out written);
        Assert.Equal(4, written);
        Assert.Equal("abé", Encoding.UTF8.GetString(buffer.Slice(0, written)));
    }
}

internal sealed class FakeCalls : IDisposable
{
    readonly AgentState state;

    FakeCalls(AgentState state) => this.state = state;

    public static FakeCalls Of(string schema, params (string Name, string Arguments)[] calls)
    {
        var tools = new ToolSet(schema);
        var names = new string[tools.Tools.Count];
        for (var i = 0; i < names.Length; i++) names[i] = tools.Aliases.OriginalTool(i);
        var indexed = new (int, string)[calls.Length];
        for (var i = 0; i < calls.Length; i++) indexed[i] = (Array.IndexOf(names, calls[i].Name), calls[i].Arguments);
        return new FakeCalls(new AgentState(names, indexed));
    }

    public Answer Answer => new(state, state.Version);

    public void Dispose()
    {
    }
}
