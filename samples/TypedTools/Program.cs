using System;
using System.Text;
using Needle;
using Needle.Tools;

namespace Needle.Samples.TypedTools;

[NeedleTools]
public partial interface IGameCommands
{
    [Tool("Get the current weather for a city.")] void GetWeather([Param("City name")] string city);
    [Tool("Set a countdown timer.")] void SetTimer([Param("Length in minutes")] int minutes);
    [Tool("Play a song or artist.")] void PlayMusic([Param("Song or artist")] string query);
}

internal struct Handler : IGameCommands
{
    public string LastCity = "";
    public int LastMinutes = -1;
    public string LastQuery = "";

    public Handler() { }

    public void GetWeather(string city) => LastCity = city;
    public void SetTimer(int minutes) => LastMinutes = minutes;
    public void PlayMusic(string query) => LastQuery = query;
}

internal static class Program
{
    static int Main()
    {
        Console.WriteLine($"schema bytes: {GameCommands.Schema.Length}");
        Console.WriteLine($"grammar tools: {GameCommands.GrammarToolCount}, nodes: {GameCommands.GrammarNodes.Length}");

        var handler = new Handler();
        if (!GameCommands.TryBind(new FunctionCall("get_weather", "{\"city\":\"Lagos\"}"), out var command)) return 1;
        GameCommands.Dispatch(ref handler, command);
        if (handler.LastCity != "Lagos") return 2;

        if (!GameCommands.TryBind(new FunctionCall("set_timer", "{\"minutes\":12}"), out command)) return 3;
        GameCommands.Dispatch(ref handler, command);
        if (handler.LastMinutes != 12) return 4;

        Console.WriteLine($"dispatched: city={handler.LastCity} minutes={handler.LastMinutes}");
        Console.WriteLine(Encoding.UTF8.GetString(GameCommands.Schema.Slice(0, 60)));
        return 0;
    }
}
