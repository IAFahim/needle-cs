using System;
using System.IO;

namespace Needle.Cli;

internal sealed record PromptSet(string Name, string ToolsJson, string[] Queries);

internal static class BenchPrompts
{
    const string GameToolsResource = "game_tools.json";

    public static readonly string[] Golden =
    {
        "what's it like in Lagos right now?",
        "set a timer for 12 minutes and play some Miles Davis",
    };

    public static readonly string[] Game =
    {
        "move to 120, 45",
        "attack the orc chieftain",
        "give Lyra 3 potions",
        "switch to wedge formation and follow the captain",
        "say hello traveler, the bridge is out",
        "stop everything",
        "give 50 gold to Borin then move to 10, 20",
        "follow Mira",
        "attack the watchtower near the river",
        "form a circle",
    };

    public static string GameToolsJson
    {
        get
        {
            using var stream = typeof(BenchPrompts).Assembly.GetManifestResourceStream(GameToolsResource)
                ?? throw new InvalidOperationException($"embedded resource {GameToolsResource} is missing");
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }
    }

    public static PromptSet[] Select(string sets, string goldenToolsJson)
    {
        var golden = new PromptSet("golden", goldenToolsJson, Golden);
        var game = new PromptSet("game", GameToolsJson, Game);
        return sets switch
        {
            "golden" => new[] { golden },
            "game" => new[] { game },
            "all" => new[] { golden, game },
            _ => throw new ArgumentException($"--sets expects golden, game or all, got {sets}"),
        };
    }
}
