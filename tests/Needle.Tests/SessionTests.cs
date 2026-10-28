using System;
using System.IO;
using System.Linq;
using Xunit;

namespace Needle.Tests;

public sealed class SessionTests : IDisposable
{
    readonly Model model;

    public SessionTests()
    {
        Assets.Require();
        model = Model.Open(Assets.Model);
    }

    public void Dispose() => model.Dispose();

    NeedleSession Create(string? tools = null, NeedleOptions? options = null) =>
        new(model, tools ?? File.ReadAllText(Assets.Tools), options ?? new NeedleOptions { Threads = 4 });

    [Fact]
    public void CallsTheRightToolWithGrammarGuaranteedArguments()
    {
        using var session = Create();
        var result = session.Complete("what's it like in Lagos right now?");
        Assert.True(result.Success);
        var call = Assert.Single(result.FunctionCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Lagos", JsonValue.Parse(call.ArgumentsJson).GetString("city"));
        Assert.NotNull(result.Confidence);
        Assert.Contains("Lagos", result.Reasoning);
        Assert.StartsWith("{\"success\":true", result.ToJson());
    }

    [Fact]
    public void ChainsTwoCallsInOrder()
    {
        using var session = Create();
        var result = session.Complete("set a timer for 12 minutes and play some Miles Davis");
        Assert.Equal(new[] { "set_timer", "play_music" }, result.FunctionCalls.Select(c => c.Name));
        Assert.Equal("12", JsonValue.Parse(result.FunctionCalls[0].ArgumentsJson).Get("minutes")!.Text);
    }

    [Fact]
    public void CallsBelowTheConfidenceFloorAreSuppressedNotReturned()
    {
        using var session = Create(options: new NeedleOptions { Threads = 4, RefuseBelow = 2f });
        var result = session.Complete("what's it like in Lagos right now?");
        Assert.Empty(result.FunctionCalls);
        Assert.Equal("get_weather", Assert.Single(result.SuppressedCalls).Name);
        Assert.Equal("respond", JsonValue.Parse(result.ToJson()).GetString("type"));
    }

    [Fact]
    public void ResetRestoresTheFreshState()
    {
        using var session = Create();
        var first = session.Complete("what's it like in Lagos right now?");
        session.Complete("set a timer for 5 minutes");
        session.Reset();
        var again = session.Complete("what's it like in Lagos right now?");
        Assert.Equal(first.FunctionCalls[0].ArgumentsJson, again.FunctionCalls[0].ArgumentsJson);
        Assert.Equal(first.Reasoning, again.Reasoning);
    }

    [Fact]
    public void ToolResultsAreAnsweredInPlainLanguage()
    {
        using var session = Create();
        session.Complete("what's it like in Lagos right now?");
        var next = session.CompleteWithToolResult("{\"city\":\"Lagos\",\"temp_c\":27,\"sky\":\"clear\"}");
        Assert.True(next.Success);
        Assert.Empty(next.FunctionCalls);
        Assert.Contains("27", next.Response);
    }

    [Fact]
    public void CamelCaseNamesAreAliasedAndRestored()
    {
        const string tools = """[{"type":"function","function":{"name":"getWeather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"cityName":{"type":"string","description":"City name"}},"required":["cityName"]}}}]""";
        using var session = Create(tools);
        var call = Assert.Single(session.Complete("weather in Paris please").FunctionCalls);
        Assert.Equal("getWeather", call.Name);
        Assert.Equal("Paris", JsonValue.Parse(call.ArgumentsJson).GetString("cityName"));
    }

    [Fact]
    public void EnumArgumentsStayInsideTheirOptions()
    {
        const string tools = """[{"name":"set_light","description":"Turn a light on or off.","parameters":{"type":"object","properties":{"state":{"type":"string","enum":["on","off"]},"room":{"type":"string","description":"Room name"}},"required":["state","room"]}}]""";
        using var session = Create(tools);
        var call = Assert.Single(session.Complete("turn off the kitchen light").FunctionCalls);
        var args = JsonValue.Parse(call.ArgumentsJson);
        Assert.Equal("off", args.GetString("state"));
        Assert.Contains("itchen", args.GetString("room"));
    }

    [Fact]
    public void ExtractionIsToolCallingWithOneSchema()
    {
        const string schema = """{"title":"booking","type":"object","properties":{"guest":{"type":"string"},"nights":{"type":"integer"},"city":{"type":"string"}},"required":["guest","nights"]}""";
        using var session = Create(Extraction.ToolsJson(schema));
        var call = Assert.Single(session.Complete("Maria Lopez would like to stay 3 nights in Lisbon.").FunctionCalls);
        var args = JsonValue.Parse(call.ArgumentsJson);
        Assert.Contains("Lopez", args.GetString("guest"));
        Assert.Equal("3", args.Get("nights")!.Text);
    }

    [Fact]
    public void LargeCatalogsAreNarrowedToTheRelevantTools()
    {
        var names = new[] { "get_weather", "set_timer", "play_music", "send_message", "call_contact", "set_alarm", "turn_on_lights", "open_app" };
        var tools = "[" + string.Join(",", names.Select(n => $"{{\"name\":\"{n}\",\"description\":\"{n.Replace('_', ' ')} for the user.\",\"parameters\":{{\"type\":\"object\",\"properties\":{{\"value\":{{\"type\":\"string\",\"description\":\"The value\"}}}},\"required\":[\"value\"]}}}}")) + "]";
        using var session = Create(tools, new NeedleOptions { Threads = 4, RetrievalTopK = 5 });
        var result = session.Complete("set an alarm for seven in the morning");
        Assert.True(result.Success);
        Assert.Single(result.FunctionCalls);
        Assert.Equal("set_alarm", result.FunctionCalls[0].Name);
    }

    [Fact]
    public void EmbeddingsAreUnitVectors()
    {
        using var session = Create();
        var v = session.Embed("turn on the lights");
        Assert.Equal(3072, v.Length);
        Assert.InRange(Math.Sqrt(v.Sum(x => (double)x * x)), 0.999, 1.001);
        var same = session.Embed("turn on the lights");
        Assert.Equal(v, same);
    }
}
