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

public enum BigItem { Aa, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31, V32, V33, V34, V35, V36, V37, V38, V39, V40, V41, V42, V43, V44, V45, V46, V47, V48, V49, V50, V51, V52, V53, V54, V55, V56, V57, V58, V59, V60, V61, V62, V63, Bb }

[NeedleTools]
public partial interface IBigEnumCommands
{
    [Tool("Pick an item.")] void Pick(BigItem item);
}

public struct ToolPickSink : IGrammarSink
{
    public int Tool;
    public int Literal;
    public void ToolSelected(int tool) => Tool = tool;
    public void PropertySelected(int property) { }
    public void StringByte(byte value) { }
    public void NumberByte(byte value) { }
    public void EnumSelected(int literal) => Literal = literal;
    public void BoolSelected(bool value) { }
    public void NullSelected() { }
    public void ValueCompleted() { }
    public void CallCompleted() { }
}

public sealed unsafe class ApiBughuntTests
{
    [Fact]
    public void Utf8Text_SetFromBytes_TruncatesAtACharacterBoundary()
    {
        var bytes = Encoding.UTF8.GetBytes("aaaaaaa\u00e9");
        var slot = new Utf8Text8();
        slot.Set(bytes);
        Assert.Equal("aaaaaaa", slot.ToString());
        var wide = new Utf8Text16();
        wide.Set(Encoding.UTF8.GetBytes("aaaaaaaaaaaaaa\u20ac"));
        Assert.Equal("aaaaaaaaaaaaaa", wide.ToString());
    }

    [Theory]
    [InlineData("{")]
    [InlineData("[")]
    [InlineData("[1")]
    [InlineData("{\"a\":1")]
    [InlineData("\"abc")]
    [InlineData("\"a\\")]
    [InlineData("\"\\u12\"")]
    [InlineData("[t]")]
    [InlineData("tru3")]
    [InlineData("{\"on\":nope}")]
    [InlineData("[falsy]")]
    public void Json_MalformedInput_ThrowsFormatException(string text) =>
        Assert.Throws<FormatException>(() => JsonValue.Parse(text));

    [Fact]
    public void Json_PathologicallyDeepNesting_ThrowsFormatExceptionInsteadOfOverflowingTheStack()
    {
        var deep = new string('[', 2000) + new string(']', 2000);
        Assert.Throws<FormatException>(() => JsonValue.Parse(deep));
    }

    [Fact]
    public void Services_CalledBeforeStart_FailWithInvalidOperationNotNullReference()
    {
        using var needle = new NeedleService();
        var call = needle.Complete("set a timer");
        Assert.IsType<InvalidOperationException>(Assert.Throws<AggregateException>(() => call.Wait(5000)).InnerException);
        using var speech = new SpeechService();
        var transcription = speech.Transcribe(new float[16000]);
        Assert.IsType<InvalidOperationException>(Assert.Throws<AggregateException>(() => transcription.Wait(5000)).InnerException);
    }

    [Fact]
    public void Services_DisposeTwice_IsANoOp()
    {
        var needle = new NeedleService();
        needle.Dispose();
        needle.Dispose();
        var speech = new SpeechService();
        speech.Dispose();
        speech.Dispose();
        var thread = new InferenceThread();
        thread.Dispose();
        thread.Dispose();
    }

    static bool Accepts(Grammar grammar, string text, ref ToolPickSink sink)
    {
        var state = Grammar.Start(grammar.Program);
        foreach (var b in Encoding.UTF8.GetBytes(text)) if (!Grammar.Advance(grammar.Program, &state, b, ref sink)) return false;
        return Grammar.Complete(state);
    }

    [Fact]
    public void Generator_EnumWithMoreThan64Members_GrammarRejectsMixedLiteralsAndBindsTheRightMember()
    {
        var program = GrammarLoader.Load(BigEnumCommands.GrammarNodes, BigEnumCommands.GrammarProperties, BigEnumCommands.GrammarLiterals,
            BigEnumCommands.GrammarUnions, BigEnumCommands.GrammarTools, BigEnumCommands.GrammarBytes, BigEnumCommands.GrammarRoot, BigEnumCommands.GrammarToolCount);
        using var grammar = new Grammar(program);
        bool Run(string text, out BigEnumCommand command)
        {
            var sink = new BigEnumSink();
            var state = Grammar.Start(grammar.Program);
            command = default;
            foreach (var b in Encoding.UTF8.GetBytes(text)) if (!Grammar.Advance(grammar.Program, &state, b, ref sink)) return false;
            return Grammar.Complete(state) && sink.TryTakeCall(out command);
        }
        Assert.False(Run("[{\"name\":\"pick\",\"arguments\":{\"item\":\"ab\"}}]", out _));
        Assert.True(Run("[{\"name\":\"pick\",\"arguments\":{\"item\":\"bb\"}}]", out var command));
        Assert.Equal(BigItem.Bb, command.Pick.Item);
    }

    [Fact]
    public void Grammar_MoreThan64Tools_NeitherMixesNamesNorPicksTheWrongTool()
    {
        var tools = new List<string>();
        for (var i = 0; i < 65; i++)
        {
            var name = i == 0 ? "aa" : i == 64 ? "bb" : "tool" + i;
            tools.Add("{\"name\":\"" + name + "\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}");
        }
        var set = new ToolSet("[" + string.Join(",", tools) + "]");
        using var grammar = new Grammar(set.Tools);
        var sink = new ToolPickSink();
        Assert.False(Accepts(grammar, "[{\"name\":\"ab\",\"arguments\":{}}]", ref sink));
        sink = new ToolPickSink();
        Assert.True(Accepts(grammar, "[{\"name\":\"bb\",\"arguments\":{}}]", ref sink));
        Assert.Equal(64, sink.Tool);
    }

    [Fact]
    public void Grammar_EnumWithMoreThan64Literals_NeitherMixesLiteralsNorPicksTheWrongOne()
    {
        var literals = new List<string>();
        for (var i = 0; i < 65; i++) literals.Add(i == 0 ? "\"aa\"" : i == 64 ? "\"bb\"" : "\"v" + i + "\"");
        var set = new ToolSet("[{\"name\":\"pick\",\"parameters\":{\"type\":\"object\",\"properties\":{\"item\":{\"type\":\"string\",\"enum\":[" + string.Join(",", literals) + "]}},\"required\":[\"item\"]}}]");
        using var grammar = new Grammar(set.Tools);
        var sink = new ToolPickSink();
        Assert.False(Accepts(grammar, "[{\"name\":\"pick\",\"arguments\":{\"item\":\"ab\"}}]", ref sink));
        sink = new ToolPickSink();
        Assert.True(Accepts(grammar, "[{\"name\":\"pick\",\"arguments\":{\"item\":\"bb\"}}]", ref sink));
        Assert.Equal(64, sink.Literal);
    }

    [Fact]
    public void SharedChannel_FileShorterThanItsHeaderedLayout_IsRejectedOnOpen()
    {
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "needle-bughunt-" + Guid.NewGuid().ToString("N") + ".chan");
        try
        {
            using (SharedChannel.Create(path, 64, 64)) { }
            using (var stream = new System.IO.FileStream(path, System.IO.FileMode.Open)) stream.SetLength(4096);
            Assert.Throws<System.IO.InvalidDataException>(() => { using var opened = SharedChannel.Open(path); });
        }
        finally
        {
            System.IO.File.Delete(path);
        }
    }

    [Fact]
    public void Channel_CapacityTooLargeForTheBlock_IsRejectedInsteadOfUndersizingTheAllocation()
    {
        var replyCell = sizeof(ChannelReply);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            NeedleChannel.ComputeLayout(64, 1_000_000, out _, out var replyCells, out _, out var total);
            Assert.True((long)total >= (long)replyCells * replyCell, $"block of {total} bytes cannot hold {replyCells} reply cells of {replyCell} bytes");
        });
    }

    [Fact]
    public void Channel_CapacityAboveTheLargestPowerOfTwo_IsRejectedInsteadOfHanging()
    {
        var layout = System.Threading.Tasks.Task.Run(() =>
        {
            try { NeedleChannel.ComputeLayout((1 << 30) + 1, 64, out _, out _, out _, out _); return "returned"; }
            catch (ArgumentOutOfRangeException) { return "rejected"; }
        });
        Assert.True(layout.Wait(TimeSpan.FromSeconds(5)), "ComputeLayout never returned");
        Assert.Equal("rejected", layout.Result);
    }

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

        Assert.True(WideCommands.TryBind(new FunctionCall("place_order", "{\"itemCount\":4,\"giftWrap\":true}"), out var camel));
        Assert.Equal(0b11u, camel.Presence);
        WideCommands.Dispatch(ref handler, camel);
        Assert.Equal(4, handler.LastCount);
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
