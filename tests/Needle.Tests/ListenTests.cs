using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

/// <summary>Voice activity and the microphone ring. No model needed.</summary>
public sealed class VoiceActivityTests
{
    static float[] Tone(double seconds, float amplitude)
    {
        var pcm = new float[(int)(seconds * Wav.SampleRate)];
        for (var i = 0; i < pcm.Length; i++) pcm[i] = amplitude * (float)Math.Sin(i * 2 * Math.PI * 220 / Wav.SampleRate);
        return pcm;
    }

    static float[] Noise(double seconds, float amplitude, int seed)
    {
        var random = new Random(seed);
        var pcm = new float[(int)(seconds * Wav.SampleRate)];
        for (var i = 0; i < pcm.Length; i++) pcm[i] = amplitude * (float)(random.NextDouble() * 2 - 1);
        return pcm;
    }

    [Fact]
    public void SpeechStartsAndEndsWithHangover()
    {
        var vad = VoiceActivity.Create(0.7);
        var pcm = Noise(1, 0.001f, 1).Concat(Tone(1, 0.1f)).Concat(Noise(1.5, 0.001f, 2)).ToArray();
        var edges = new List<(int Edge, int At)>();
        for (var i = 0; i < pcm.Length; i++)
        {
            var edge = vad.Push(pcm[i]);
            if (edge != 0) edges.Add((edge, i + 1));
        }
        Assert.Equal(2, edges.Count);
        Assert.Equal(1, edges[0].Edge);
        Assert.Equal(-1, edges[1].Edge);
        // Speech is confirmed after three loud frames and ended after 0.7 s of quiet ones.
        Assert.Equal(Wav.SampleRate + VoiceActivity.StartFrames * VoiceActivity.FrameSamples, edges[0].At);
        Assert.Equal(2 * Wav.SampleRate + 35 * VoiceActivity.FrameSamples, edges[1].At);
    }

    [Fact]
    public void QuietNoiseIsNotSpeechAndLoudNoiseStopsBeingSpeech()
    {
        var vad = VoiceActivity.Create(0.7);
        foreach (var x in Noise(2, 0.003f, 3)) Assert.Equal(0, vad.Push(x));
        Assert.False(vad.InSpeech);
        var started = false;
        foreach (var x in Tone(0.5, 0.2f)) started |= vad.Push(x) == 1;
        Assert.True(started);
        foreach (var x in new float[Wav.SampleRate]) vad.Push(x);
        Assert.False(vad.InSpeech);
        // A steady background loud enough to pass for speech at first is learned as the floor within seconds.
        var edges = new List<int>();
        foreach (var x in Noise(20, 0.02f, 4))
        {
            var edge = vad.Push(x);
            if (edge != 0) edges.Add(edge);
        }
        Assert.Equal(new[] { 1, -1 }, edges);
        started = false;
        foreach (var x in Tone(0.5, 0.3f)) started |= vad.Push(x) == 1;
        Assert.True(started);
    }

    [Fact]
    public void PausesShorterThanTheEndpointDoNotEndSpeech()
    {
        var vad = VoiceActivity.Create(0.7);
        var pcm = Tone(0.5, 0.1f).Concat(new float[(int)(0.5 * Wav.SampleRate)]).Concat(Tone(0.5, 0.1f)).ToArray();
        var ends = 0;
        foreach (var x in pcm) ends += vad.Push(x) < 0 ? 1 : 0;
        Assert.Equal(0, ends);
        Assert.True(vad.InSpeech);
    }

    sealed class Sink : ISampleSink
    {
        public readonly List<float> Samples = new();
        public bool Write(ReadOnlySpan<float> samples)
        {
            foreach (var x in samples) Samples.Add(x);
            return true;
        }
    }

    [Fact]
    public void TheRingKeepsOrderAcrossTheWrapAndDropsWhatDoesNotFit()
    {
        var ring = new AudioRing(10);
        var sink = new Sink();
        var next = 0f;
        float[] Run(int n)
        {
            var run = new float[n];
            for (var i = 0; i < n; i++) run[i] = next++;
            return run;
        }
        Assert.Equal(7, ring.Write(Run(7)));
        Assert.Equal(5, ring.Drain(sink, 5));
        Assert.Equal(8, ring.Write(Run(8)));
        Assert.Equal(10, ring.Available);
        Assert.Equal(0, ring.Write(Run(3)));
        Assert.Equal(3, ring.Dropped);
        Assert.Equal(10, ring.Drain(sink, int.MaxValue));
        Assert.Equal(Enumerable.Range(0, 15).Select(i => (float)i), sink.Samples);
        Assert.Equal(0, ring.Drain(sink, int.MaxValue));
    }
}

/// <summary>Incremental streaming speech against offline transcription. Needs whistle.cact and ref/audio.</summary>
public sealed class VoiceStreamTests : IDisposable
{
    static string WhistlePath => Path.Combine(Assets.Root, "ref/hf/whistle/whistle.cact");
    static float[] Clip(string name) => Wav.Read(Path.Combine(Assets.Root, "ref/audio", name));

    readonly ITestOutputHelper output;
    readonly Model model;
    readonly Whistle whistle;

    public VoiceStreamTests(ITestOutputHelper output)
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

    byte[] Text => new byte[whistle.MaxTranscriptTokens * 64];
    WordSlice[] Words => new WordSlice[whistle.MaxTranscriptTokens + 1];

    static string Render(byte[] text, SpeechSpan span, WordSlice[] words)
    {
        var b = new StringBuilder(Encoding.UTF8.GetString(text, 0, span.Utf8Length)).Append('|').Append(span.Language);
        for (var i = 0; i < span.WordCount; i++)
            b.Append('|').Append(words[i].Utf8Offset).Append(',').Append(words[i].Utf8Length).Append(',')
                .Append(BitConverter.DoubleToInt64Bits(words[i].Start)).Append(',').Append(BitConverter.DoubleToInt64Bits(words[i].End)).Append(',')
                .Append(BitConverter.DoubleToInt64Bits(words[i].Probability));
        return b.ToString();
    }

    [Theory]
    [InlineData("record.wav", false)]
    [InlineData("test.wav", false)]
    [InlineData("hotword.wav", true)]
    public void ASteppedPassIsTheOfflinePassBitForBit(string clip, bool keywords)
    {
        var pcm = Clip(clip);
        if (pcm.Length > Whistle.MaxSeconds * Wav.SampleRate) pcm = pcm.AsSpan(0, Whistle.MaxSeconds * Wav.SampleRate).ToArray();
        IReadOnlyList<string>? words = keywords ? new[] { "Lyra", "Kestrel" } : null;
        var offlineText = Text;
        var offlineWords = Words;
        var offline = whistle.TranscribeUtf8(pcm, offlineText, offlineWords, null, words);
        var steps = 0;
        if (whistle.BeginSpan(pcm, null, words))
            while (!whistle.StepSpan()) steps++;
        var steppedText = Text;
        var steppedWords = Words;
        var stepped = whistle.FinishSpan(steppedText, steppedWords, 0);
        output.WriteLine($"{clip}: {steps} steps: {Encoding.UTF8.GetString(steppedText, 0, stepped.Utf8Length)}");
        Assert.True(steps > whistle.Frames / 100, "a pass takes many steps");
        Assert.Equal(Render(offlineText, offline, offlineWords), Render(steppedText, stepped, steppedWords));
    }

    static float[] Repeated(float[] clip, int times, double gapSeconds)
    {
        var gap = new float[(int)(gapSeconds * Wav.SampleRate)];
        var all = new List<float>(gap);
        for (var i = 0; i < times; i++)
        {
            all.AddRange(clip);
            all.AddRange(gap);
        }
        return all.ToArray();
    }

    sealed record Pass(double Start, double Seconds, string Committed, bool Final);

    static List<Pass> Stream(VoiceStream stream, float[] pcm, double chunkSeconds, out string transcript)
    {
        var committedText = new byte[8192];
        var committedWords = new WordSlice[512];
        var pendingText = new byte[8192];
        var pendingWords = new WordSlice[512];
        var passes = new List<Pass>();
        var all = new StringBuilder();
        void Take(StreamDelta d)
        {
            Assert.False(d.Truncated);
            var text = Encoding.UTF8.GetString(committedText, 0, d.CommittedUtf8);
            if (d.PassSeconds > 0) passes.Add(new Pass(d.PassStartSeconds, d.PassSeconds, text, d.Final));
            if (d.CommittedUtf8 == 0) return;
            if (all.Length > 0) all.Append(' ');
            all.Append(text);
        }
        var chunk = (int)(chunkSeconds * Wav.SampleRate);
        for (var at = 0; at < pcm.Length; at += chunk)
        {
            Assert.True(stream.Write(pcm.AsSpan(at, Math.Min(chunk, pcm.Length - at))));
            Take(stream.Update(committedText, committedWords, pendingText, pendingWords));
        }
        Take(stream.Stop(committedText, committedWords, pendingText, pendingWords));
        transcript = all.ToString();
        return passes;
    }

    [Fact]
    public void EveryCommittingPassIsTheOfflineTranscriptOfItsAudio()
    {
        var pcm = Repeated(Clip("record.wav"), 3, 1.2);
        var stream = new VoiceStream(whistle, "en");
        var passes = Stream(stream, pcm, 0.25, out var transcript);
        output.WriteLine(transcript);
        var finals = passes.Where(p => p.Final && p.Committed.Length > 0).ToList();
        Assert.True(finals.Count >= 3, $"{finals.Count} phrases");
        Assert.Equal(3, CountOf(transcript, "speed of sound"));
        var text = Text;
        var words = Words;
        foreach (var pass in finals)
        {
            var start = (int)Math.Round(pass.Start * Wav.SampleRate);
            var length = (int)Math.Round(pass.Seconds * Wav.SampleRate);
            var offline = whistle.TranscribeUtf8(pcm.AsSpan(start, length), text, words, "en");
            Assert.Equal(Encoding.UTF8.GetString(text, 0, offline.Utf8Length), pass.Committed);
        }
        // Committed audio is never transcribed again: final passes do not overlap.
        for (var i = 1; i < finals.Count; i++) Assert.True(finals[i].Start >= finals[i - 1].Start + finals[i - 1].Seconds - 1e-9);
    }

    [Fact]
    public void CostGrowsLinearlyWithTheAudio()
    {
        var clip = Clip("record.wav");
        var shortStream = new VoiceStream(whistle, "en");
        Stream(shortStream, Repeated(clip, 2, 1.2), 0.25, out _);
        var longStream = new VoiceStream(whistle, "en");
        Stream(longStream, Repeated(clip, 6, 1.2), 0.25, out var transcript);
        var shortRatio = shortStream.EncodedSeconds / shortStream.ReceivedSeconds;
        var longRatio = longStream.EncodedSeconds / longStream.ReceivedSeconds;
        output.WriteLine($"short {shortStream.EncodedSeconds:0.0}/{shortStream.ReceivedSeconds:0.0} s, long {longStream.EncodedSeconds:0.0}/{longStream.ReceivedSeconds:0.0} s");
        Assert.Equal(6, CountOf(transcript, "speed of sound"));
        // Re-reading everything would make the long stream's ratio about three times the short one's.
        Assert.InRange(longRatio, 0, shortRatio * 1.25);
        Assert.True(longStream.BufferedSeconds < 2, "committed audio is dropped");
    }

    [Fact]
    public void LongSpeechWithoutPausesIsCommittedInWindows()
    {
        var clip = Clip("record.wav");
        var pcm = Repeated(clip, 4, 0.1);
        var stream = new VoiceStream(whistle, "en") { MaxWindowSeconds = 6 };
        var passes = Stream(stream, pcm, 0.25, out var transcript);
        output.WriteLine(transcript);
        Assert.Contains(passes, p => !p.Final && p.Committed.Length > 0);
        Assert.True(stream.EncodedSeconds < stream.ReceivedSeconds * (6 / stream.PartialSeconds + 2));
        Assert.True(CountOf(transcript, "sound") >= 3, transcript);
    }

    static int CountOf(string text, string word)
    {
        var count = 0;
        for (var at = text.IndexOf(word, StringComparison.OrdinalIgnoreCase); at >= 0; at = text.IndexOf(word, at + 1, StringComparison.OrdinalIgnoreCase)) count++;
        return count;
    }
}

/// <summary>agent.Listen: speech into the same agent, a slice at a time. Needs the Needle and Whistle models.</summary>
public sealed class ListenTests : IDisposable
{
    static string WhistlePath => Path.Combine(Assets.Root, "ref/hf/whistle/whistle.cact");
    static float[] Clip(string name) => Wav.Read(Path.Combine(Assets.Root, "ref/audio", name));

    readonly ITestOutputHelper output;
    readonly NeedleModel needle;
    readonly NeedleModel speech;
    readonly string tools;

    public ListenTests(ITestOutputHelper output)
    {
        this.output = output;
        Assets.Require();
        if (!File.Exists(WhistlePath)) throw new InvalidOperationException("missing whistle.cact; run tools/fetch.sh");
        needle = NeedleModel.Open(Assets.Model);
        speech = NeedleModel.Open(WhistlePath);
        tools = File.ReadAllText(Assets.Tools);
    }

    public void Dispose()
    {
        speech.Dispose();
        needle.Dispose();
    }

    static readonly AgentOptions Options = new() { Threads = 2, AutoDate = Toggle.Off };

    sealed class Heard
    {
        public readonly List<HearingKind> Kinds = new();
        public readonly List<string> Partials = new();
        public readonly List<string> Calls = new();
        public string Transcript = "";
        public string Fingerprint = "";
        public bool Answered;
    }

    static Heard Listen(NeedleVoice voice, float[] pcm, FrameBudget budget, bool endPhrase, int chunk = Wav.SampleRate / 50)
    {
        var heard = new Heard();
        void Drain()
        {
            while (voice.Step(budget, out var e))
            {
                heard.Kinds.Add(e.Kind);
                switch (e.Kind)
                {
                    case HearingKind.Partial:
                        Assert.StartsWith(e.Committed.ToString(), e.Transcript.ToString());
                        heard.Partials.Add(e.Transcript.ToString());
                        break;
                    case HearingKind.CallReady:
                        heard.Calls.Add(e.Call.Name.ToString());
                        break;
                    case HearingKind.Done:
                    case HearingKind.Unsure:
                        heard.Transcript = e.Transcript.ToString();
                        heard.Answered = e.HasAnswer;
                        if (e.HasAnswer) heard.Fingerprint = TurnStreamTests.Fingerprint(e.Answer);
                        return;
                }
            }
        }
        for (var at = 0; at < pcm.Length; at += chunk)
        {
            Assert.Equal(Math.Min(chunk, pcm.Length - at), voice.Write(pcm.AsSpan(at, Math.Min(chunk, pcm.Length - at))));
            Drain();
        }
        if (endPhrase) voice.EndPhrase();
        var guard = 0;
        while (heard.Kinds.Count == 0 || heard.Kinds[^1] is not (HearingKind.Done or HearingKind.Unsure))
        {
            Drain();
            Assert.True(++guard < 1_000_000, "the voice never finished");
        }
        return heard;
    }

    static float[] WithSilence(float[] clip, double before, double after)
    {
        var pcm = new float[(int)(before * Wav.SampleRate) + clip.Length + (int)(after * Wav.SampleRate)];
        clip.CopyTo(pcm, (int)(before * Wav.SampleRate));
        return pcm;
    }

    [Theory]
    [InlineData(0, 0.0)]
    [InlineData(1, 0.0)]
    [InlineData(0, 2.0)]
    public void APhraseIsAnsweredExactlyAsAskingItsTranscript(int tokens, double ms)
    {
        var budget = new FrameBudget(tokens, ms);
        using var agent = needle.Agent(tools, Options);
        using var voice = agent.Listen(speech, new ListenOptions { Language = "en" });
        var heard = Listen(voice, WithSilence(Clip("record.wav"), 0.5, 1.5), budget, endPhrase: false);
        output.WriteLine(string.Join(" ", heard.Kinds) + " : " + heard.Transcript);
        Assert.Equal(HearingKind.SpeechStarted, heard.Kinds[0]);
        Assert.Equal(1, heard.Kinds.Count(k => k == HearingKind.SpeechStarted));
        if (budget.IsUnlimited) Assert.Contains(HearingKind.Partial, heard.Kinds);
        Assert.Contains("speed of sound", heard.Transcript);
        Assert.True(heard.Answered);
        using var asked = needle.Agent(tools, Options);
        Assert.Equal(TurnStreamTests.Fingerprint(asked.Ask(heard.Transcript)), heard.Fingerprint);
        Assert.Equal(heard.Calls.Count, heard.Kinds.Count(k => k == HearingKind.CallReady));
    }

    [Fact]
    public void PushToTalkEndsTheTurnOnRelease()
    {
        using var agent = needle.Agent(tools, Options);
        using var voice = agent.Listen(speech, new ListenOptions { Language = "en", PushToTalk = true });
        voice.Hold();
        var clip = Clip("record.wav");
        var heard = new List<HearingKind>();
        var chunk = Wav.SampleRate / 50;
        for (var at = 0; at < clip.Length; at += chunk)
        {
            voice.Write(clip.AsSpan(at, Math.Min(chunk, clip.Length - at)));
            while (voice.Step(FrameBudget.FromMilliseconds(5), out var e)) heard.Add(e.Kind);
        }
        Assert.DoesNotContain(HearingKind.Done, heard);
        voice.Release();
        string transcript = "";
        for (var guard = 0; guard < 1_000_000; guard++)
        {
            if (!voice.Step(FrameBudget.FromMilliseconds(5), out var e)) continue;
            heard.Add(e.Kind);
            if (e.Kind is HearingKind.Done or HearingKind.Unsure)
            {
                transcript = e.Transcript.ToString();
                break;
            }
        }
        Assert.Equal(HearingKind.SpeechStarted, heard[0]);
        Assert.Contains("speed of sound", transcript);
        Assert.False(voice.IsBusy);
    }

    [Fact]
    public void SilenceIsNeverHeardAndAnEmptyPhraseIsUnsure()
    {
        using var agent = needle.Agent(tools, Options);
        using var voice = agent.Listen(speech);
        voice.Write(new float[3 * Wav.SampleRate]);
        Assert.False(voice.Step(FrameBudget.Unlimited, out _));
        Assert.False(voice.IsBusy);
        voice.Hold();
        voice.Write(new float[Wav.SampleRate]);
        voice.Release();
        var kinds = new List<HearingKind>();
        while (voice.Step(FrameBudget.Unlimited, out var e))
        {
            kinds.Add(e.Kind);
            if (e.Kind == HearingKind.Unsure)
            {
                Assert.False(e.HasAnswer);
                Assert.True(e.Transcript.IsEmpty);
            }
        }
        Assert.Equal(new[] { HearingKind.SpeechStarted, HearingKind.Unsure }, kinds);
    }

    [Fact]
    public void HearingsGoStaleAndCancelStopsTheTurn()
    {
        using var agent = needle.Agent(tools, Options);
        using var voice = agent.Listen(speech, new ListenOptions { Language = "en" });
        voice.Write(WithSilence(Clip("record.wav"), 0.3, 1.0));
        Assert.True(voice.Step(FrameBudget.FromTokens(1), out var started));
        Assert.Equal(HearingKind.SpeechStarted, started.Kind);
        voice.Step(FrameBudget.FromTokens(1), out _);
        var threw = false;
        try
        {
            _ = started.Kind;
        }
        catch (InvalidOperationException)
        {
            threw = true;
        }
        Assert.True(threw, "a hearing goes stale with the next step");
        for (var guard = 0; guard < 1_000_000 && !ThinkingOf(voice); guard++) voice.Step(FrameBudget.FromTokens(1), out _);
        Assert.True(ThinkingOf(voice));
        Assert.True(voice.IsBusy);
        Assert.Throws<InvalidOperationException>(() => { agent.Ask("hello"); });
        voice.Cancel();
        Assert.False(voice.IsBusy);
        Assert.False(voice.Step(FrameBudget.Unlimited, out _));
        Assert.NotEqual(AnswerKind.Failed, agent.Ask("hello").Kind);
    }

    static bool ThinkingOf(NeedleVoice voice) => voice.State.Phase == VoicePhase.Thinking;

    [Fact]
    public void ListeningAllocatesNothingOnceWarm()
    {
        using var agent = needle.Agent(tools, Options);
        using var voice = agent.Listen(speech, new ListenOptions { Language = "en" });
        var pcm = WithSilence(Clip("record.wav"), 0.3, 1.2);
        Listen(voice, pcm, FrameBudget.FromMilliseconds(2), endPhrase: false);
        var chunk = Wav.SampleRate / 50;
        var before = GC.GetAllocatedBytesForCurrentThread();
        var done = false;
        for (var at = 0; at < pcm.Length || !done; at += chunk)
        {
            if (at < pcm.Length) voice.Write(pcm.AsSpan(at, Math.Min(chunk, pcm.Length - at)));
            while (voice.Step(FrameBudget.FromMilliseconds(2), out var e))
                if (e.Kind is HearingKind.Done or HearingKind.Unsure) done = true;
        }
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        output.WriteLine($"allocated {allocated} bytes");
        Assert.Equal(0, allocated);
    }
}
