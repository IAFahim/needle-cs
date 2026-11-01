using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Needle.Tests;

internal enum StepKind { Query, ToolResult, Reset }

internal sealed record SessionStep(StepKind Kind, string Input = "");

internal sealed record SessionCase(string Name, string? ToolsJson, NeedleOptions Options, SessionStep[] Steps);

internal static class SessionCases
{
    const string EnumTools = """[{"name":"set_light","description":"Turn a light on or off.","parameters":{"type":"object","properties":{"state":{"type":"string","enum":["on","off"]},"room":{"type":"string","description":"Room name"}},"required":["state","room"]}}]""";

    const string CamelTools = """[{"type":"function","function":{"name":"getWeather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"cityName":{"type":"string","description":"City name"}},"required":["cityName"]}}}]""";

    const string NestedTools = """[{"name":"moveCharacter","description":"Move a character to a map position.","parameters":{"type":"object","properties":{"characterName":{"type":"string","description":"Who moves"},"targetPosition":{"type":"object","properties":{"xCoord":{"type":"number"},"yCoord":{"type":"number"}},"required":["xCoord","yCoord"]},"runFast":{"type":"boolean"},"waypointList":{"type":"array","items":{"type":"string"}}},"required":["characterName","targetPosition"]}},{"name":"sendMessage","description":"Send a chat message to a person.","parameters":{"type":"object","properties":{"recipientName":{"type":"string","description":"Who receives it"},"messageText":{"type":"string","description":"The message"}},"required":["recipientName","messageText"]}}]""";

    const string ExtractionSchema = """{"title":"booking","type":"object","properties":{"guest":{"type":"string"},"nights":{"type":"integer"},"city":{"type":"string"}},"required":["guest","nights"]}""";

    static readonly string[] CatalogNames = { "get_weather", "set_timer", "play_music", "send_message", "call_contact", "set_alarm", "turn_on_lights", "open_app" };

    public static string Catalog() => "[" + string.Join(",", CatalogNames.Select(n => $"{{\"name\":\"{n}\",\"description\":\"{n.Replace('_', ' ')} for the user.\",\"parameters\":{{\"type\":\"object\",\"properties\":{{\"value\":{{\"type\":\"string\",\"description\":\"The value\"}}}},\"required\":[\"value\"]}}}}")) + "]";

    public static string DefaultTools() => File.ReadAllText(Assets.Tools);

    public static IReadOnlyList<SessionCase> All(int threads) => new[]
    {
        new SessionCase("weather", null, Options(threads), new[] { Query("what's it like in Lagos right now?") }),
        new SessionCase("chain", null, Options(threads), new[] { Query("set a timer for 12 minutes and play some Miles Davis") }),
        new SessionCase("refused", null, new NeedleOptions { Threads = threads, AutoDate = false, RefuseBelow = 2f }, new[] { Query("what's it like in Lagos right now?") }),
        new SessionCase("reset", null, Options(threads), new[]
        {
            Query("what's it like in Lagos right now?"), Query("set a timer for 5 minutes"), new SessionStep(StepKind.Reset), Query("what's it like in Lagos right now?"),
        }),
        new SessionCase("toolResult", null, Options(threads), new[]
        {
            Query("what's it like in Lagos right now?"), new SessionStep(StepKind.ToolResult, "{\"city\":\"Lagos\",\"temp_c\":27,\"sky\":\"clear\"}"),
        }),
        new SessionCase("camelCase", CamelTools, Options(threads), new[] { Query("weather in Paris please") }),
        new SessionCase("enum", EnumTools, Options(threads), new[] { Query("turn off the kitchen light") }),
        new SessionCase("extraction", Extraction.ToolsJson(ExtractionSchema), Options(threads), new[] { Query("Maria Lopez would like to stay 3 nights in Lisbon.") }),
        new SessionCase("retrieval", Catalog(), new NeedleOptions { Threads = threads, AutoDate = false, RetrievalTopK = 5 }, new[] { Query("set an alarm for seven in the morning") }),
        new SessionCase("nestedAliases", NestedTools, Options(threads), new[]
        {
            Query("move Lyra to x 120 y 45 and run fast, passing through bridge and gate"), Query("tell Zoe: \"meet me at the cafe\" at noon"),
        }),
        new SessionCase("unicode", NestedTools, Options(threads), new[] { Query("tell Zoë: \"meet me at the café ✓\" \\ now") }),
    };

    static NeedleOptions Options(int threads) => new() { Threads = threads, AutoDate = false };

    static SessionStep Query(string text) => new(StepKind.Query, text);
}
