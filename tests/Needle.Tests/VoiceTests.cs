using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

public sealed class VoiceTests : IDisposable
{
    static string WhistlePath => Path.Combine(Assets.Root, "ref/hf/whistle/whistle.cact");
    static float[] Clip(string name) => Wav.Read(Path.Combine(Assets.Root, "ref/audio", name));

    readonly Model model;
    readonly Whistle whistle;
    readonly ITestOutputHelper output;

    public VoiceTests(ITestOutputHelper output)
    {
        this.output = output;
        if (!File.Exists(WhistlePath) || !File.Exists(Path.Combine(Assets.Root, "ref/audio/record.wav"))) throw new InvalidOperationException("missing whistle.cact or ref/audio; run tools/fetch.sh");
        model = Model.Open(WhistlePath);
        whistle = new Whistle(model, 4) { BeamWidth = 1 };
    }

    public void Dispose()
    {
        whistle.Dispose();
        model.Dispose();
    }

    static byte[] Utf8Buffer(Whistle w) => new byte[w.MaxTranscriptTokens * 64];
    static WordSlice[] WordBuffer(Whistle w) => new WordSlice[w.MaxTranscriptTokens + 1];

    static void AssertSameTranscript(Transcription expected, SpeechSpan actual, byte[] utf8)
    {
        Assert.False(actual.Truncated);
        Assert.Equal(expected.Text, Encoding.UTF8.GetString(utf8, 0, actual.Utf8Length));
        Assert.Equal(expected.Language, actual.Language);
        var words = expected.Words ?? (IReadOnlyList<Word>)Array.Empty<Word>();
        Assert.Equal(words.Count, actual.WordCount);
    }

    static void AssertSameWords(Transcription expected, SpeechSpan actual, byte[] utf8, WordSlice[] slices)
    {
        var words = expected.Words ?? (IReadOnlyList<Word>)Array.Empty<Word>();
        for (var i = 0; i < words.Count; i++)
        {
            Assert.Equal(words[i].Text, Encoding.UTF8.GetString(utf8, slices[i].Utf8Offset, slices[i].Utf8Length));
            Assert.True(Math.Abs(words[i].Start - slices[i].Start) < 1e-9, $"word {i} start {slices[i].Start} != {words[i].Start}");
            Assert.True(Math.Abs(words[i].End - slices[i].End) < 1e-9, $"word {i} end");
            Assert.True(Math.Abs(words[i].Probability - slices[i].Probability) < 1e-9, $"word {i} prob");
        }
    }

    [Fact]
    public void SpanMatchesOfflineOnRefClips()
    {
        foreach (var name in new[] { "record.wav", "test.wav", "hotword.wav" })
        {
            var pcm = Clip(name);
            var expected = whistle.Transcribe(pcm, wordTimestamps: true);
            var utf8 = Utf8Buffer(whistle);
            var slices = WordBuffer(whistle);
            var actual = whistle.TranscribeUtf8(pcm, utf8, slices);
            AssertSameTranscript(expected, actual, utf8);
            AssertSameWords(expected, actual, utf8, slices);
        }
    }

    [Fact]
    public void SpanMatchesOfflineWithKeywords()
    {
        var biased = new List<string> { "Omeprazole" };
        var unrelated = new List<string> { "Siobhan", "Krzysztof" };
        var hotword = Clip("hotword.wav");
        var expectedBiased = whistle.Transcribe(hotword, keywords: biased);
        var utf8 = Utf8Buffer(whistle);
        var slices = WordBuffer(whistle);
        var actualBiased = whistle.TranscribeUtf8(hotword, utf8, slices, keywords: biased);
        Assert.False(actualBiased.Truncated);
        Assert.Equal(expectedBiased.Text, Encoding.UTF8.GetString(utf8, 0, actualBiased.Utf8Length));
        Assert.Contains("prazole", Encoding.UTF8.GetString(utf8, 0, actualBiased.Utf8Length));

        var record = Clip("record.wav");
        var expectedPlain = whistle.Transcribe(record);
        var actualPlain = whistle.TranscribeUtf8(record, utf8, slices, keywords: unrelated);
        Assert.Equal(expectedPlain.Text, Encoding.UTF8.GetString(utf8, 0, actualPlain.Utf8Length));
    }

    [Fact]
    public void SpanTranscriptionAllocatesNothing()
    {
        // Process-wide GC bytes include background host threads, so measure in a quiet
        // child process exactly like ZeroAllocationTests does.
        var measurement = RunVoiceProbe(4);
        Assert.Equal(0, measurement.Unexpected);
        Assert.Equal(0, measurement.CallerThread);
        Assert.Equal(0, measurement.WholeProcess);
        if (measurement.NativeCounted) Assert.Equal(0, measurement.NativeCalls);
    }

    static VoiceProbe.Measurement RunVoiceProbe(int threads)
    {
        var host = Environment.ProcessPath is { } path && Path.GetFileNameWithoutExtension(path) == "dotnet" ? path : Environment.GetEnvironmentVariable("DOTNET_HOST_PATH") ?? "dotnet";
        var assembly = typeof(VoiceTests).Assembly.Location;
        var stem = Path.Combine(Path.GetDirectoryName(assembly)!, Path.GetFileNameWithoutExtension(assembly));
        var start = new ProcessStartInfo(host) { RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (var argument in new[] { "exec", "--runtimeconfig", stem + ".runtimeconfig.json", "--depsfile", stem + ".deps.json", assembly }) start.ArgumentList.Add(argument);
        start.Environment[VoiceProbe.Variable] = threads.ToString();
        using var process = Process.Start(start)!;
        var error = process.StandardError.ReadToEndAsync();
        var output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        var line = output.Split('\n', StringSplitOptions.RemoveEmptyEntries)[^1].Trim();
        Assert.True(line.StartsWith("voiceprobe ", StringComparison.Ordinal), $"{line} {error.Result}");
        var fields = line["voiceprobe ".Length..].Split(' ');
        long Field(int index) => long.Parse(fields[index][(fields[index].IndexOf('=') + 1)..]);
        return new VoiceProbe.Measurement(Field(0), Field(1), Field(2), fields[3].EndsWith("True", StringComparison.Ordinal), (int)Field(4));
    }

    [Fact]
    public void LongSpanMatchesOffline()
    {
        var pcm = Clip("test_long.wav");
        var expected = whistle.Transcribe(pcm, "en", wordTimestamps: true);
        var utf8 = Utf8Buffer(whistle);
        var slices = WordBuffer(whistle);
        var actual = whistle.TranscribeUtf8(pcm, utf8, slices, "en");
        AssertSameTranscript(expected, actual, utf8);
        AssertSameWords(expected, actual, utf8, slices);
    }

    [Fact]
    public void PushToTalkRoundTripsInAudioThreadChunks()
    {
        var pcm = Clip("record.wav");
        var expected = whistle.Transcribe(pcm).Text;
        var talk = new PushToTalk();
        Assert.False(talk.IsHeld);
        const int Chunk = 1600;
        for (var at = 0; at < pcm.Length; at += Chunk)
            Assert.True(talk.Write(pcm.AsSpan(at, Math.Min(Chunk, pcm.Length - at))));
        Assert.True(talk.IsHeld);
        Assert.InRange(talk.BufferedSeconds, 4, 5);
        var utf8 = Utf8Buffer(whistle);
        var slices = WordBuffer(whistle);
        var released = talk.Release(whistle, utf8, slices);
        Assert.False(released.Truncated);
        Assert.False(talk.IsHeld);
        Assert.Equal(expected, Encoding.UTF8.GetString(utf8, 0, released.Utf8Length));

        // Empty release transcribes nothing.
        var empty = talk.Release(whistle, utf8, slices);
        Assert.Equal(0, empty.Utf8Length);
        Assert.Null(empty.Language);

        // Overflow is reported, never grown.
        var small = new PushToTalk(maxSeconds: 1);
        Assert.True(small.Write(pcm.AsSpan(0, 8000)));
        Assert.False(small.Write(pcm.AsSpan(0, 8001)));
        Assert.False(small.Write(pcm));
    }

    [Fact]
    public void StreamingSpanConvergesOnTheOfflineTranscript()
    {
        var pcm = Clip("record.wav");
        var expected = whistle.Transcribe(pcm).Text;
        var stream = new VoiceStream(whistle);
        var committedText = new byte[4096];
        var committedWords = new WordSlice[256];
        var pendingText = new byte[4096];
        var pendingWords = new WordSlice[256];
        var transcript = new StringBuilder();
        for (var at = 0; at < pcm.Length; at += Wav.SampleRate)
        {
            Assert.True(stream.Write(pcm.AsSpan(at, Math.Min(Wav.SampleRate, pcm.Length - at))));
            var step = stream.Update(committedText, committedWords, pendingText, pendingWords);
            Assert.False(step.Truncated);
            if (step.CommittedUtf8 > 0)
            {
                if (transcript.Length > 0) transcript.Append(' ');
                transcript.Append(Encoding.UTF8.GetString(committedText, 0, step.CommittedUtf8));
            }
        }
        var tail = stream.Stop(committedText, committedWords, pendingText, pendingWords);
        if (tail.CommittedUtf8 > 0)
        {
            if (transcript.Length > 0) transcript.Append(' ');
            transcript.Append(Encoding.UTF8.GetString(committedText, 0, tail.CommittedUtf8));
        }
        Assert.Contains("speed of sound at sea level", transcript.ToString());
        Assert.Contains("speed of sound at sea level", expected);
        Assert.InRange(stream.ReceivedSeconds, 4, 5);
    }

    [Fact]
    public void VoiceSessionSpanMatchesStringPath()
    {
        Assets.Require();
        using var needle = Model.Open(Assets.Model);
        using var voice = new VoiceSession(needle, model, File.ReadAllText(Assets.Tools), new NeedleOptions { Threads = 2, AutoDate = false });
        var pcm = Clip("record.wav");
        var calls = new byte[4096];
        var reasoning = new byte[4096];
        var response = new byte[4096];
        var transcript = Utf8Buffer(whistle);
        var words = WordBuffer(whistle);
        var (reply, speech) = voice.CompleteSpan(pcm, calls, reasoning, response, transcript, words);
        Assert.False(speech.Truncated);
        var text = Encoding.UTF8.GetString(transcript, 0, speech.Utf8Length);
        Assert.Contains("speed of sound", text);
        Assert.NotEqual(ReplyStatus.Failed, reply.Status);

        var silent = voice.CompleteSpan(new float[Wav.SampleRate], calls, reasoning, response, transcript, words);
        Assert.Equal(0, silent.Speech.Utf8Length);
        Assert.Equal(ReplyStatus.Failed, silent.Reply.Status);
        Assert.Equal(FailureReason.EmptyInput, silent.Reply.Failure);
    }

    [Fact]
    public void VoiceToTypedCommandEndToEnd()
    {
        Assets.Require();
        var schema = Encoding.UTF8.GetString(GameCommands.Schema);
        using var host = new VoiceHost();
        var channel = host.Start(new VoiceHostOptions
        {
            WhistleModelPath = WhistlePath,
            NeedleModelPath = Assets.Model,
            ToolsJson = schema,
            Session = new NeedleOptions { Threads = 2, AutoDate = false },
            WhistleThreads = 2,
        });

        // Leg A: audio in, transcribed reply out.
        var pcm = Clip("record.wav");
        const int Chunk = 4800;
        for (var at = 0; at < pcm.Length; at += Chunk)
            Assert.True(channel.WriteAudio(pcm.AsSpan(at, Math.Min(Chunk, pcm.Length - at))));
        Assert.True(channel.TryEndUtterance(out var utterance));
        Assert.False(channel.TryEndUtterance(out _));
        var voiceReply = WaitForVoice(ref channel, utterance, TimeSpan.FromSeconds(180));
        var transcript = Encoding.UTF8.GetString(Utf8.AsSpan(ref voiceReply.Transcript, voiceReply.TranscriptLength));
        output.WriteLine($"voice transcript ({voiceReply.TranscriptLength} bytes): {transcript}");
        Assert.Contains("speed of sound", transcript);
        Assert.False(voiceReply.QueryTruncated);
        Assert.NotEqual(ReplyStatus.Failed, voiceReply.Reply.Status);

        // Leg B: command text through the same hub channel, typed dispatch at the caller.
        Assert.True(host.Channel.TrySubmitQuery("set a timer for 12 minutes and play some Miles Davis"u8, out _));
        var direct = WaitForDirect(ref channel, TimeSpan.FromSeconds(180));
        Assert.Equal(ReplyStatus.Called, direct.Reply.Status);
        var callsJson = Encoding.UTF8.GetString(Utf8.AsSpan(ref direct.Calls, direct.Reply.Utf8Length));
        using var document = JsonDocument.Parse(callsJson);
        var names = new List<string>();
        foreach (var element in document.RootElement.EnumerateArray())
        {
            var call = new FunctionCall(element.GetProperty("name").GetString()!, element.GetProperty("arguments").GetRawText());
            Assert.True(GameCommands.TryBind(call, out var command));
            names.Add(call.Name);
            var handler = new GameHandler();
            GameCommands.Dispatch(ref handler, command);
            if (call.Name == "set_timer") Assert.Equal(12, handler.LastMinutes);
            if (call.Name == "play_music") Assert.Equal("Miles Davis", handler.LastQuery);
        }
        Assert.Equal(new[] { "set_timer", "play_music" }, names);
    }

    static VoiceReply WaitForVoice(ref VoiceChannel channel, ulong utterance, TimeSpan budget)
    {
        var deadline = Stopwatch.GetTimestamp() + (long)(budget.TotalSeconds * Stopwatch.Frequency);
        for (;;)
        {
            while (channel.TryReceive(out var reply))
                if (reply.Utterance == utterance) return reply;
            if (Stopwatch.GetTimestamp() > deadline) throw new TimeoutException("no voice reply arrived");
            Thread.Sleep(20);
        }
    }

    static ChannelReply WaitForDirect(ref VoiceChannel channel, TimeSpan budget)
    {
        var deadline = Stopwatch.GetTimestamp() + (long)(budget.TotalSeconds * Stopwatch.Frequency);
        for (;;)
        {
            while (channel.TryReceiveDirect(out var reply)) return reply;
            if (Stopwatch.GetTimestamp() > deadline) throw new TimeoutException("no direct reply arrived");
            Thread.Sleep(20);
        }
    }

    internal static float[] LoudestWindow(float[] pcm, int length)
    {
        var best = 0;
        var bestEnergy = double.NegativeInfinity;
        for (var at = 0; at + length <= pcm.Length; at += length / 4)
        {
            double energy = 0;
            for (var i = 0; i < length; i += 16) energy += Math.Abs(pcm[at + i]);
            if (energy > bestEnergy) { bestEnergy = energy; best = at; }
        }
        return pcm.AsSpan(best, length).ToArray();
    }

    [Fact]
    public void LatencyIsReportedAgainstNativeAndTheAllocatingPath()
    {
        var pcm = LoudestWindow(Clip("record.wav"), 3 * Wav.SampleRate);
        var utf8 = Utf8Buffer(whistle);
        var slices = WordBuffer(whistle);
        whistle.TranscribeUtf8(pcm, utf8, slices);
        var bestSpan = double.MaxValue;
        SpeechSpan span = default;
        for (var i = 0; i < 5; i++)
        {
            var start = Stopwatch.GetTimestamp();
            span = whistle.TranscribeUtf8(pcm, utf8, slices);
            bestSpan = Math.Min(bestSpan, (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency);
        }
        whistle.Transcribe(pcm);
        var bestManaged = double.MaxValue;
        Transcription managed = new();
        for (var i = 0; i < 2; i++)
        {
            var start = Stopwatch.GetTimestamp();
            managed = whistle.Transcribe(pcm, wordTimestamps: true);
            bestManaged = Math.Min(bestManaged, (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency);
        }
        var spanText = Encoding.UTF8.GetString(utf8, 0, span.Utf8Length);
        Assert.False(span.Truncated);
        Assert.Equal(managed.Text, spanText);

        var nativeMs = double.NaN;
        var nativeText = NativeTranscribe.TryTranscribe(pcm, out nativeMs);
        var line = $"voice 3s clip: span {bestSpan:0.0} ms (encode {span.EncodeMs:0.0}, ttft {span.TimeToFirstTokenMs:0.0}, {span.DecodeTps:0.0} tps), allocating {bestManaged:0.0} ms";
        if (nativeText != null)
        {
            line += $", native {nativeMs:0.0} ms";
            Assert.Equal(nativeText, spanText);
        }
        else line += ", native unavailable";
        output.WriteLine(line);
        Assert.True(span.Utf8Length > 0);
    }

    static unsafe class NativeTranscribe
    {
        public static string? TryTranscribe(float[] pcm, out double ms)
        {
            ms = double.NaN;
            try
            {
                var lib = LoadLibrary();
                if (lib == IntPtr.Zero) return null;
                if (!NativeLibrary.TryGetExport(lib, "needle_load", out var loadAddress)) return null;
                if (!NativeLibrary.TryGetExport(lib, "needle_transcribe", out var transcribeAddress)) return null;
                var load = (delegate* unmanaged[Cdecl]<byte*, ulong, int>)loadAddress;
                var transcribe = (delegate* unmanaged[Cdecl]<float*, int, byte*, byte*, int, byte*, int, int>)transcribeAddress;
                var cact = File.ReadAllBytes(WhistlePath);
                int loaded;
                fixed (byte* cactPtr = cact)
                    loaded = load(cactPtr, (ulong)cact.Length);
                if (loaded < 0) return null;
                var buffer = new byte[1 << 16];
                int tokens;
                var start = Stopwatch.GetTimestamp();
                fixed (float* pcmPtr = pcm)
                fixed (byte* outPtr = buffer)
                    tokens = transcribe(pcmPtr, pcm.Length, null, null, 0, outPtr, buffer.Length);
                ms = (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
                if (tokens < 0) return null;
                using var document = JsonDocument.Parse(Encoding.UTF8.GetString(buffer, 0, Array.IndexOf(buffer, (byte)0)));
                return document.RootElement.GetProperty("text").GetString();
            }
            catch { return null; }
        }

        static IntPtr LoadLibrary()
        {
            var dir = Path.Combine(Path.GetTempPath(), "needle-w27-native");
            var lib = Path.Combine(dir, "libneedle3.so");
            if (!File.Exists(lib))
            {
                string? wheel = null;
                foreach (var candidate in Directory.GetFiles(Path.Combine(Assets.Root, "ref/hf/wheel"), "*.whl"))
                {
                    wheel = candidate;
                    break;
                }
                if (wheel == null) return IntPtr.Zero;
                Directory.CreateDirectory(dir);
                using var archive = ZipFile.OpenRead(wheel!);
                foreach (var entry in archive.Entries)
                    if (entry.FullName.EndsWith("libneedle3.so", StringComparison.Ordinal))
                    {
                        entry.ExtractToFile(lib, overwrite: true);
                        break;
                    }
                if (!File.Exists(lib)) return IntPtr.Zero;
            }
            try { return NativeLibrary.Load(lib); }
            catch { return IntPtr.Zero; }
        }
    }
}

internal static class VoiceProbe
{
    public const string Variable = "NEEDLE_VOICE_PROBE";

    public readonly record struct Measurement(long CallerThread, long WholeProcess, long NativeCalls, bool NativeCounted, int Unexpected);

    public static Measurement Measure(int threads)
    {
        using var model = Model.Open(Path.Combine(Assets.Root, "ref/hf/whistle/whistle.cact"));
        using var whistle = new Whistle(model, threads) { BeamWidth = 1 };
        var pcm = VoiceTests.LoudestWindow(Wav.Read(Path.Combine(Assets.Root, "ref/audio/record.wav")), 3 * Wav.SampleRate);
        var utf8 = new byte[whistle.MaxTranscriptTokens * 64];
        var words = new WordSlice[whistle.MaxTranscriptTokens + 1];
        for (var round = 0; round < 5; round++) whistle.TranscribeUtf8(pcm, utf8, words);
        var unexpected = 0;
        var silence = new float[3 * Wav.SampleRate];
        var nativeBefore = MallocProbe.Available ? MallocProbe.Count : 0;
        var threadBefore = GC.GetAllocatedBytesForCurrentThread();
        var processBefore = GC.GetTotalAllocatedBytes(true);
        for (var round = 0; round < 30; round++)
        {
            var speech = whistle.TranscribeUtf8(pcm, utf8, words);
            if (speech.Truncated || speech.Utf8Length == 0) unexpected++;
        }
        for (var round = 0; round < 10; round++)
            if (whistle.TranscribeUtf8(silence, utf8, words).Utf8Length != 0) unexpected++;
        var processAfter = GC.GetTotalAllocatedBytes(true);
        var threadAfter = GC.GetAllocatedBytesForCurrentThread();
        var nativeAfter = MallocProbe.Available ? MallocProbe.Count : 0;
        return new Measurement(threadAfter - threadBefore, processAfter - processBefore, nativeAfter - nativeBefore, MallocProbe.Available, unexpected);
    }
}

internal static class VoiceProbeEntry
{
#pragma warning disable CA2255
    [ModuleInitializer]
    internal static void Run()
#pragma warning restore CA2255
    {
        var request = Environment.GetEnvironmentVariable(VoiceProbe.Variable);
        if (request == null) return;
        var measurement = VoiceProbe.Measure(int.Parse(request));
        Console.Out.WriteLine($"voiceprobe caller={measurement.CallerThread} process={measurement.WholeProcess} native={measurement.NativeCalls} nativeCounted={measurement.NativeCounted} unexpected={measurement.Unexpected}");
        Console.Out.Flush();
        Environment.Exit(0);
    }
}
