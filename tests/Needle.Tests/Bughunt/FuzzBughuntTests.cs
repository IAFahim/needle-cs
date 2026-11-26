using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using Needle.Tools;
using Xunit;

namespace Needle.Tests.Bughunt;

public sealed class FuzzBughuntTests
{
    static bool Folded(string a, string b) => VoiceStream.FoldedEqual(Encoding.UTF8.GetBytes(a), Encoding.UTF8.GetBytes(b));

    [Theory]
    [InlineData("cats", "cat")]
    [InlineData("cat", "cats")]
    [InlineData("a", "ab")]
    [InlineData("ten", "ten1")]
    public void StreamWordAgreementRejectsAWordThatGainedATrailingLetter(string current, string previous)
    {
        Assert.False(Folded(current, previous), $"'{current}' must not agree with '{previous}'");
    }

    [Theory]
    [InlineData("Cat,", "cat")]
    [InlineData("cat", "CAT!")]
    [InlineData("\"hello\"", "hello")]
    [InlineData("", "")]
    [InlineData("...", "")]
    public void StreamWordAgreementStillIgnoresCaseAndPunctuation(string a, string b)
    {
        Assert.True(Folded(a, b));
        Assert.True(Folded(b, a));
    }

    [Theory]
    [InlineData("{")]
    [InlineData("[")]
    [InlineData("{ ")]
    [InlineData("[1")]
    [InlineData("[1,")]
    [InlineData("{\"a\":1")]
    [InlineData("{\"a\"")]
    [InlineData("\"abc")]
    [InlineData("\"\\")]
    [InlineData("\"\\u12")]
    [InlineData("[tru")]
    [InlineData("[nul")]
    public void TruncatedJsonThrowsFormatException(string text)
    {
        var error = Xunit.Record.Exception(() => JsonValue.Parse(text));
        Assert.IsType<FormatException>(error);
    }

    [Theory]
    [InlineData("trux")]
    [InlineData("nope")]
    [InlineData("fals3")]
    [InlineData("[tree]")]
    [InlineData("-")]
    [InlineData("1.2.3")]
    [InlineData("+1")]
    [InlineData("1e")]
    [InlineData("--1")]
    [InlineData("٣")]
    [InlineData("\"\\u 12 \"")]
    public void InvalidJsonIsRejected(string text)
    {
        var error = Xunit.Record.Exception(() => JsonValue.Parse(text));
        Assert.IsType<FormatException>(error);
    }

    [Fact]
    public void ToolSetDoesNotReEmitAnInvalidNumberIntoThePrompt()
    {
        var error = Xunit.Record.Exception(() => new ToolSet("[{\"name\":\"move\",\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"number\",\"maximum\":1.2.3}}}}]"));
        Assert.IsType<FormatException>(error);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-0")]
    [InlineData("12")]
    [InlineData("-3.25")]
    [InlineData("1e9")]
    [InlineData("1E-9")]
    [InlineData("2.5e+3")]
    [InlineData("true")]
    [InlineData("false")]
    [InlineData("null")]
    [InlineData("[true,false,null]")]
    [InlineData("{\"a\":[1,{\"b\":\"\\u00e9\\n\"}]}")]
    public void ValidJsonStillRoundTrips(string text)
    {
        Assert.Equal(text.Replace("\\u00e9", "é"), JsonValue.Parse(text).ToString());
    }

    [Fact]
    public void DeeplyNestedJsonIsRejectedInsteadOfOverflowingTheStack()
    {
        Exception? error = null;
        var thread = new Thread(() =>
        {
            var text = new string('[', 20000) + new string(']', 20000);
            error = Xunit.Record.Exception(() => JsonValue.Parse(text));
        }, 256 * 1024);
        thread.Start();
        thread.Join();
        Assert.IsType<FormatException>(error);
    }

    [Fact]
    public void Utf8TextSetFromBytesKeepsWholeCodePoints()
    {
        var text = default(Utf8Text8);
        text.Set(Encoding.UTF8.GetBytes("日本語"));
        var fromString = default(Utf8Text8);
        fromString.Set("日本語");
        Assert.Equal("日本", text.ToString());
        Assert.Equal(fromString.Length, text.Length);
    }

    [Fact]
    public void Utf8TextSetFromBytesKeepsWholeCodePointsAtEveryCapacity()
    {
        var bytes = Encoding.UTF8.GetBytes(string.Concat(System.Linq.Enumerable.Repeat("é😀日", 60)));
        var t16 = default(Utf8Text16); t16.Set(bytes);
        var t32 = default(Utf8Text32); t32.Set(bytes);
        var t64 = default(Utf8Text64); t64.Set(bytes);
        var t128 = default(Utf8Text128); t128.Set(bytes);
        var t256 = default(Utf8Text256); t256.Set(bytes);
        foreach (var span in new[] { t16.Span.ToArray(), t32.Span.ToArray(), t64.Span.ToArray(), t128.Span.ToArray(), t256.Span.ToArray() })
            Assert.Equal(span, Encoding.UTF8.GetBytes(Encoding.UTF8.GetString(span)));
    }

    static byte[] WavBytes(params (string Id, byte[] Body)[] chunks)
    {
        var stream = new MemoryStream();
        var w = new BinaryWriter(stream);
        w.Write("RIFF"u8.ToArray());
        w.Write(0);
        w.Write("WAVE"u8.ToArray());
        foreach (var (id, body) in chunks)
        {
            w.Write(Encoding.ASCII.GetBytes(id));
            w.Write(body.Length);
            w.Write(body);
            if ((body.Length & 1) == 1) w.Write((byte)0);
        }
        return stream.ToArray();
    }

    static byte[] Fmt(int tag, int channels, int rate, int bits)
    {
        var stream = new MemoryStream();
        var w = new BinaryWriter(stream);
        w.Write((ushort)tag);
        w.Write((ushort)channels);
        w.Write(rate);
        w.Write(rate * channels * bits / 8);
        w.Write((ushort)(channels * bits / 8));
        w.Write((ushort)bits);
        return stream.ToArray();
    }

    [Theory]
    [InlineData(8000)]
    [InlineData(11025)]
    [InlineData(22050)]
    [InlineData(44100)]
    [InlineData(48000)]
    [InlineData(96000)]
    public void ResampledSineKeepsItsFrequencyAndAmplitude(int rate)
    {
        var input = new float[rate];
        for (var i = 0; i < input.Length; i++) input[i] = (float)(0.5 * Math.Sin(2 * Math.PI * 440 * i / rate));
        var output = Wav.Resample(input, rate, 16000);
        Assert.Equal(16000, output.Length);
        double worst = 0;
        for (var n = 400; n < 15600; n++) worst = Math.Max(worst, Math.Abs(output[n] - 0.5 * Math.Sin(2 * Math.PI * 440 * n / 16000.0)));
        Assert.True(worst < 0.01, $"{rate} Hz: worst deviation {worst}");
    }

    [Fact]
    public void WavDecodesEveryPcmLayout()
    {
        var pcm16 = new byte[8];
        BitConverter.GetBytes((short)16384).CopyTo(pcm16, 0);
        BitConverter.GetBytes((short)-16384).CopyTo(pcm16, 2);
        Assert.Equal(new[] { 0.5f, -0.5f, 0f, 0f }, Wav.Decode(WavBytes(("LIST", new byte[5]), ("fmt ", Fmt(1, 1, 16000, 16)), ("data", pcm16))));
        var pcm24 = new byte[] { 0x00, 0x00, 0xC0 };
        Assert.Equal(new[] { -0.5f }, Wav.Decode(WavBytes(("fmt ", Fmt(1, 1, 16000, 24)), ("data", pcm24))));
        var stereo8 = new byte[] { 192, 64 };
        Assert.Equal(new[] { 0f }, Wav.Decode(WavBytes(("fmt ", Fmt(1, 2, 16000, 8)), ("data", stereo8))));
        var streaming = WavBytes(("fmt ", Fmt(3, 1, 16000, 32)), ("data", BitConverter.GetBytes(0.25f)));
        BitConverter.GetBytes(uint.MaxValue).CopyTo(streaming, streaming.Length - 8);
        Assert.Equal(new[] { 0.25f }, Wav.Decode(streaming));
    }

    [Fact]
    public void FuzzedWavInputOnlyThrowsInvalidData()
    {
        var failures = Fuzz(0x5EED, 4000, TimeSpan.FromSeconds(20), random =>
        {
            var bits = new[] { 8, 16, 24, 32, 64, 0, 7, 12 }[random.Next(8)];
            var tag = new[] { 1, 3, 0xFFFE, 0, 2 }[random.Next(5)];
            var channels = random.Next(4) == 0 ? random.Next(0, 70000) : random.Next(1, 9);
            var rate = random.Next(4) == 0 ? random.Next(int.MinValue, int.MaxValue) : new[] { 8000, 11025, 16000, 22050, 44100, 48000, 96000, 0, 1 }[random.Next(9)];
            var fmt = Fmt(tag, channels, rate, bits);
            if (tag == 0xFFFE)
            {
                var extended = new byte[40];
                fmt.CopyTo(extended, 0);
                BitConverter.GetBytes((ushort)new[] { 1, 3, 2 }[random.Next(3)]).CopyTo(extended, 24);
                fmt = extended;
            }
            if (random.Next(5) == 0) fmt = fmt.AsSpan(0, random.Next(fmt.Length)).ToArray();
            var data = new byte[random.Next(0, 600)];
            random.NextBytes(data);
            var bytes = random.Next(3) == 0 ? WavBytes(("data", data), ("fmt ", fmt)) : WavBytes(("LIST", new byte[random.Next(9)]), ("fmt ", fmt), ("data", data));
            switch (random.Next(6))
            {
                case 0: bytes = bytes.AsSpan(0, random.Next(bytes.Length)).ToArray(); break;
                case 1: bytes[random.Next(bytes.Length)] = (byte)random.Next(256); break;
                case 2: BitConverter.GetBytes((uint)random.Next()).CopyTo(bytes, 12 + 4 + random.Next(2) * (8 + fmt.Length + (fmt.Length & 1))); break;
            }
            return bytes;
        }, bytes => Wav.Decode(bytes), typeof(InvalidDataException));
        Assert.True(failures.Length == 0, failures);
    }

    [Fact]
    public void FuzzedJsonOnlyThrowsFormatException()
    {
        const string seed = "{\"tools\":[{\"name\":\"get_weather\",\"description\":\"Get the weather \\u00e9 \\\"now\\\"\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\",\"minimum\":-1.5e3}},\"required\":[\"city\"]},\"strict\":true,\"x\":null,\"y\":false}]}";
        const string alphabet = "{}[]\",:\\/ \t\nutnfrlase0123456789-+.eEé٣\uD83D‏\0";
        var failures = Fuzz(0xC0FFEE, 6000, TimeSpan.FromSeconds(20), random =>
        {
            var text = new StringBuilder(seed);
            for (var edits = random.Next(1, 6); edits > 0; edits--)
            {
                var at = random.Next(text.Length + 1);
                switch (random.Next(4))
                {
                    case 0: if (at < text.Length) text.Remove(at, Math.Min(text.Length - at, random.Next(1, 8))); break;
                    case 1: text.Insert(at, alphabet[random.Next(alphabet.Length)]); break;
                    case 2: if (at < text.Length) text[at] = alphabet[random.Next(alphabet.Length)]; break;
                    default: text.Length = at; break;
                }
            }
            return text.ToString();
        }, text =>
        {
            var parsed = JsonValue.Parse(text);
            Assert.Equal(parsed.ToString(), JsonValue.Parse(parsed.ToString()).ToString());
            try { new ToolSet(text).Render(); }
            catch (FormatException) { }
        }, typeof(FormatException));
        Assert.True(failures.Length == 0, failures);
    }

    [Fact]
    public void ToolSetHandlesHostileNames()
    {
        var names = new[] { "", "\0", "😀", "😀😀", "שלום", "é", "a b", "A", "a", "a_2", "__", "GetHTTPResponse", new string('x', 100000) };
        var json = new StringBuilder("[");
        foreach (var name in names)
        {
            if (json.Length > 1) json.Append(',');
            json.Append("{\"name\":");
            JsonValue.WriteString(json, name);
            json.Append(",\"parameters\":{\"type\":\"object\",\"properties\":{");
            JsonValue.WriteString(json, name);
            json.Append(":{\"type\":\"string\"}},\"required\":[");
            JsonValue.WriteString(json, name);
            json.Append("]}}");
        }
        json.Append(']');
        var set = new ToolSet(json.ToString());
        Assert.Equal(names.Length, set.NameAliases.Count);
        foreach (var name in names) Assert.Contains(name, set.NameAliases.Values);
        Assert.Equal(set.Render(), JsonValue.Parse(set.Render()).ToString());
    }

    static void AssertRequiredNamesItsOwnProperties(JsonValue schema)
    {
        if (schema.Kind == JsonKind.Array) foreach (var item in schema.Items) AssertRequiredNamesItsOwnProperties(item);
        if (schema.Kind != JsonKind.Object) return;
        if (schema.Get("required") is { Kind: JsonKind.Array } required && schema.Get("properties") is { Kind: JsonKind.Object } properties)
            foreach (var name in required.Items) Assert.Contains(name.Text, properties.Members.ConvertAll(m => m.Key));
        foreach (var member in schema.Members) AssertRequiredNamesItsOwnProperties(member.Value);
    }

    [Theory]
    [InlineData("{\"name\":\"t\",\"parameters\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},\"required\":[\"a\"]}},\"required\":[\"a\"]}}")]
    [InlineData("{\"name\":\"t\",\"parameters\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"pet\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"}},\"required\":[\"name\",\"age\"]}},\"required\":[\"name\",\"pet\"]}}")]
    [InlineData("{\"name\":\"t\",\"parameters\":{\"type\":\"object\",\"required\":[\"userName\"],\"properties\":{\"user_name\":{\"type\":\"string\"},\"userName\":{\"type\":\"string\"}}}}")]
    public void RenderedRequiredListsNameTheirOwnRenamedProperties(string tool)
    {
        var set = new ToolSet(tool);
        foreach (var rendered in set.Tools) AssertRequiredNamesItsOwnProperties(rendered);
        var original = JsonValue.Parse(tool).Get("parameters")!.Get("required")!.Items.ConvertAll(r => r.Text);
        var renamed = set.Tools[0].Get("parameters")!.Get("required")!.Items.ConvertAll(r => set.OriginalProperty("t", r.Text));
        Assert.Equal(original, renamed);
    }

    [Fact]
    public void NestedPropertyKeepsItsOwnOriginalNameWhenAnOuterOneSnakeCasesOntoIt()
    {
        var set = new ToolSet("{\"name\":\"t\",\"parameters\":{\"type\":\"object\",\"properties\":{\"info\":{\"type\":\"object\",\"properties\":{\"user_name\":{\"type\":\"string\"}}},\"userName\":{\"type\":\"string\"}}}}");
        var properties = set.Tools[0].Get("parameters")!.Get("properties")!;
        var nested = properties.Get("info")!.Get("properties")!.Members[0].Key;
        var outer = properties.Members[1].Key;
        var generated = Encoding.UTF8.GetBytes($"[{{\"name\":\"t\",\"arguments\":{{\"info\":{{\"{nested}\":\"x\"}},\"{outer}\":\"y\"}}}}]");
        var destination = new byte[512];
        Assert.True(CallRestorer.TryRestore(set.Aliases, generated, destination, new byte[64], new CallSpan[4], out var length, out _));
        Assert.Equal("[{\"name\":\"t\",\"arguments\":{\"info\":{\"user_name\":\"x\"},\"userName\":\"y\"}}]", Encoding.UTF8.GetString(destination, 0, length));
    }

    static string Fuzz<T>(int seed, int cases, TimeSpan cap, Func<Random, T> generate, Action<T> run, Type allowed)
    {
        var random = new Random(seed);
        var report = new StringBuilder();
        var clock = Stopwatch.StartNew();
        for (var i = 0; i < cases && clock.Elapsed < cap; i++)
        {
            var input = generate(random);
            var started = clock.Elapsed;
            Exception? error = null;
            try { run(input); }
            catch (Exception e) { error = e; }
            if (clock.Elapsed - started > TimeSpan.FromSeconds(2)) report.AppendLine($"case {i}: took {(clock.Elapsed - started).TotalSeconds:F1} s");
            if (error != null && error.GetType() != allowed && report.Length < 4000)
                report.AppendLine($"case {i}: {error.GetType().Name}: {error.Message} input={Describe(input)}");
        }
        return report.ToString();
    }

    static string Describe<T>(T input) => input switch
    {
        byte[] b => Convert.ToHexString(b.AsSpan(0, Math.Min(b.Length, 64))),
        string s => s.Length > 120 ? s[..120] + "..." : s,
        _ => input?.ToString() ?? "null",
    };
}
