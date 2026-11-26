using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using Needle.Tools;
using Xunit;

namespace Needle.Tests.Bughunt;

[NeedleTools]
public partial interface IWideCommands
{
    [Tool("Configure many knobs.")]
    void Tune(int a, int b = 5, int c = 6, int d = 7, int e = 8, int f = 9, int g = 10, int h = 11, int i = 12);
    [Tool("Place an order.")]
    void PlaceOrder(int itemCount, bool giftWrap = false);
}

public struct WideHandler : IWideCommands
{
    public int[] Last;
    public int LastCount;
    public bool LastGift;

    public void Tune(int a, int b, int c, int d, int e, int f, int g, int h, int i) => Last = new[] { a, b, c, d, e, f, g, h, i };

    public void PlaceOrder(int itemCount, bool giftWrap)
    {
        LastCount = itemCount;
        LastGift = giftWrap;
    }
}

public sealed unsafe class ApiBughuntTests
{
    [Theory]
    [InlineData("call", "/nonexistent/needle3.cact", "/nonexistent/tools.json", "hi")]
    [InlineData("bench", "/nonexistent/needle3.cact")]
    [InlineData("transcribe", "/nonexistent/whistle.cact", "/nonexistent/clip.wav")]
    [InlineData("transcribe", "/nonexistent/whistle.cact", "/nonexistent/clip.wav", "--language")]
    public void Cli_MissingFilesOrOptionValues_ExitWithAnErrorInsteadOfAborting(params string[] arguments)
    {
        var directory = new System.IO.DirectoryInfo(AppContext.BaseDirectory);
        while (directory != null && !System.IO.File.Exists(System.IO.Path.Combine(directory.FullName, "src", "Needle.Cli", "Needle.Cli.csproj")))
            directory = directory.Parent;
        var cli = System.IO.Path.Combine(directory!.FullName, "src", "Needle.Cli", "bin", "Release", "net10.0", "needle.dll");
        Assert.True(System.IO.File.Exists(cli), "build the CLI first: dotnet build src/Needle.Cli -c Release");
        var start = new System.Diagnostics.ProcessStartInfo("dotnet") { RedirectStandardError = true, RedirectStandardOutput = true };
        start.ArgumentList.Add(cli);
        foreach (var argument in arguments) start.ArgumentList.Add(argument);
        using var process = System.Diagnostics.Process.Start(start)!;
        var error = process.StandardError.ReadToEnd();
        process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        Assert.DoesNotContain("Unhandled exception", error);
        Assert.InRange(process.ExitCode, 1, 2);
    }

    [Fact]
    public void ToolSet_NestedPropertySharingATopLevelName_KeepsItsNameAndItsRequiredConstraint()
    {
        var set = new ToolSet("""[{"name":"book","parameters":{"type":"object","properties":{"city":{"type":"string"},"stop":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},"required":["city","stop"]}}]""");
        var stop = set.Tools[0].Get("parameters")!.Get("properties")!.Get("stop")!;
        Assert.NotNull(stop.Get("properties")!.Get("city"));
        Assert.Equal("city", stop.Get("required")!.Items[0].Text);

        using var grammar = new Grammar(set.Tools);
        bool Accepts(string text)
        {
            var state = Grammar.Start(grammar.Program);
            foreach (var b in Encoding.UTF8.GetBytes(text)) if (!Grammar.Advance(grammar.Program, &state, b)) return false;
            return Grammar.Complete(state);
        }
        Assert.True(Accepts("""[{"name":"book","arguments":{"city":"a","stop":{"city":"b"}}}]"""));
        Assert.False(Accepts("""[{"name":"book","arguments":{"city":"a","stop":{}}}]"""));
    }

    [Fact]
    public void Generator_TryBind_AcceptsTheSnakeCasePropertyNamesTheSchemaAdvertises()
    {
        using var schema = JsonDocument.Parse(Encoding.UTF8.GetString(WideCommands.Schema));
        var properties = schema.RootElement[1].GetProperty("parameters").GetProperty("properties");
        Assert.True(properties.TryGetProperty("item_count", out _));
        Assert.True(properties.TryGetProperty("gift_wrap", out _));

        Assert.True(WideCommands.TryBind(new FunctionCall("place_order", "{\"item_count\":3,\"gift_wrap\":true}"), out var command));
        Assert.Equal(0b11u, command.Presence);
        var handler = new WideHandler();
        WideCommands.Dispatch(ref handler, command);
        Assert.Equal(3, handler.LastCount);
        Assert.True(handler.LastGift);
    }

    [Fact]
    public void Generator_TryBind_SetsOnlyThePresenceBitOfTheArgumentThatWasGiven()
    {
        Assert.True(WideCommands.TryBind(new FunctionCall("tune", "{\"a\":1,\"e\":40}"), out var fifth));
        Assert.Equal((1u << 0) | (1u << 4), fifth.Presence);
        Assert.True(WideCommands.TryBind(new FunctionCall("tune", "{\"a\":1,\"h\":70}"), out var eighth));
        Assert.Equal((1u << 0) | (1u << 7), eighth.Presence);

        var handler = new WideHandler();
        WideCommands.Dispatch(ref handler, fifth);
        Assert.Equal(new[] { 1, 5, 6, 7, 40, 9, 10, 11, 12 }, handler.Last);
        WideCommands.Dispatch(ref handler, eighth);
        Assert.Equal(new[] { 1, 5, 6, 7, 8, 9, 10, 70, 12 }, handler.Last);
    }

    [Fact]
    public void Generator_GrammarSink_SetsOnlyThePresenceBitOfTheArgumentThatWasGiven()
    {
        var sink = new WideSink();
        sink.ToolSelected(0);
        Feed(ref sink, 0, "1");
        Feed(ref sink, 7, "70");
        sink.CallCompleted();
        Assert.True(sink.TryTakeCall(out var command));
        Assert.Equal((1u << 0) | (1u << 7), command.Presence);
        var handler = new WideHandler();
        WideCommands.Dispatch(ref handler, command);
        Assert.Equal(new[] { 1, 5, 6, 7, 8, 9, 10, 70, 12 }, handler.Last);

        static void Feed(ref WideSink sink, int property, string number)
        {
            sink.PropertySelected(property);
            foreach (var b in Encoding.ASCII.GetBytes(number)) sink.NumberByte(b);
            sink.ValueCompleted();
        }
    }
}
