using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Xunit;

namespace Needle.Tests;

public sealed class SpeechTests : IDisposable
{
    static string Whistle => Path.Combine(Assets.Root, "ref/hf/whistle/whistle.cact");
    static float[] Clip(string name) => Wav.Read(Path.Combine(Assets.Root, "ref/audio", name));

    readonly Model model;
    readonly Whistle whistle;

    public SpeechTests()
    {
        if (!File.Exists(Whistle) || !File.Exists(Path.Combine(Assets.Root, "ref/audio/record.wav"))) throw new InvalidOperationException("missing whistle.cact or ref/audio; run tools/fetch.sh");
        model = Model.Open(Whistle);
        whistle = new Whistle(model, 4);
    }

    public void Dispose()
    {
        whistle.Dispose();
        model.Dispose();
    }

    [Fact]
    public void TranscribesSpeechAndDetectsTheLanguage()
    {
        var result = whistle.Transcribe(Clip("record.wav"));
        Assert.Equal("en", result.Language);
        Assert.Contains("speed of sound at sea level", result.Text);
        Assert.Contains("voice memo", whistle.Transcribe(Clip("test.wav")).Text);
    }

    [Fact]
    public void RecordMatchesUpstreamExactly()
    {
        Assert.Equal("In one short sentence, what is the speed of sound at sea level?", whistle.Transcribe(Clip("record.wav")).Text);
    }

    [Fact]
    public void LongClipMatchesUpstreamThroughTheQueenPassage()
    {
        var firstTwentyFiveSeconds = Clip("test_long.wav").AsSpan(0, 25 * 16000);
        var text = whistle.Transcribe(firstTwentyFiveSeconds, "en").Text;
        AssertCloseTo("a new queen in Victoria, the first popular monarch in centuries, and her husband Albert", text, "test_long.wav");
    }

    static void AssertCloseTo(string expected, string actual, string clip)
    {
        static string[] Words(string text) => text.ToLowerInvariant().Split(new[] { ' ', ',', '.', '?', '!', ';', ':' }, StringSplitOptions.RemoveEmptyEntries);
        var want = Words(expected);
        var have = Words(actual);
        var table = new int[want.Length + 1, have.Length + 1];
        for (var i = 1; i <= want.Length; i++)
            for (var j = 1; j <= have.Length; j++)
                table[i, j] = want[i - 1] == have[j - 1] ? table[i - 1, j - 1] + 1 : Math.Max(table[i - 1, j], table[i, j - 1]);
        var overlap = (double)table[want.Length, have.Length] / want.Length;
        Assert.True(overlap >= 0.75, $"{clip}: only {overlap:P0} of \"{expected}\" found in: {actual}");
    }

    [Fact]
    public void BeamSearchRecoversWordsGreedyDecodingMisses()
    {
        var clip = Clip("test_long.wav").AsSpan(0, 25 * 16000).ToArray();
        whistle.BeamWidth = 1;
        var greedy = whistle.Transcribe(clip, "en").Text;
        whistle.BeamWidth = 4;
        var beam = whistle.Transcribe(clip, "en").Text;
        Assert.Contains("queen in victory", greedy);
        Assert.Contains("queen in Victoria", beam);
    }

    [Fact]
    public void TestClipAndHotwordClipKeepUpstreamPhrases()
    {
        var test = whistle.Transcribe(Clip("test.wav")).Text;
        Assert.True(test.Contains("creating a way of file"), $"test.wav: {test}");
        Assert.True(test.Contains("Wi-Fi"), $"test.wav: {test}");
        Assert.True(test.Contains("this will transcribe properly"), $"test.wav: {test}");
        var hotword = whistle.Transcribe(Clip("hotword.wav")).Text;
        AssertCloseTo("part of the cactus team and, you know, I also", hotword, "hotword.wav");
        Assert.True(hotword.Contains("called Omep Brazil"), $"hotword.wav: {hotword}");
    }

    [Fact]
    public void ResamplesFortyEightKilohertzStereo()
    {
        Assert.Contains("hello", whistle.Transcribe(Clip("hotword.wav")).Text.ToLowerInvariant());
    }

    [Fact]
    public void ForcedLanguageIsHonouredAndUnknownOnesAreRejected()
    {
        Assert.Equal("de", whistle.Transcribe(Clip("record.wav"), "de").Language);
        Assert.Throws<ArgumentException>(() => whistle.Transcribe(Clip("record.wav"), "xx"));
    }

    [Fact]
    public void SilenceYieldsNothing()
    {
        var result = whistle.Transcribe(new float[16000 * 3]);
        Assert.Equal("", result.Text);
        Assert.Null(result.Language);
    }

    [Fact]
    public void WordTimesAreOrderedAndInsideTheClip()
    {
        var pcm = Clip("record.wav");
        var words = whistle.Transcribe(pcm, wordTimestamps: true).Words!;
        Assert.Equal(13, words.Count);
        Assert.Equal(words.Select(w => w.Start).OrderBy(x => x), words.Select(w => w.Start));
        Assert.All(words, w => { Assert.InRange(w.Start, 0, pcm.Length / 16000.0 + 0.1); Assert.True(w.End >= w.Start); Assert.InRange(w.Probability, 0, 1.0001); });
        Assert.InRange(words[0].Start, 0, 1.5);
    }

    [Fact]
    public void LongAudioIsWindowedAndTimesCountFromTheStart()
    {
        var pcm = Clip("test_long.wav");
        var result = whistle.Transcribe(pcm, wordTimestamps: true);
        Assert.Contains("Parliament", result.Text);
        var words = result.Words!;
        Assert.True(words.Count > 100);
        Assert.True(words[^1].End > 60 && words[^1].End <= pcm.Length / 16000.0 + 1);
        for (var i = 1; i < words.Count; i++) Assert.True(words[i].Start >= words[i - 1].Start - 0.5);
    }

    [Fact]
    public void KeywordsSteerSpellingWithoutInventingWords()
    {
        var steered = whistle.Transcribe(Clip("hotword.wav"), keywords: new[] { "Omeprazole" }).Text;
        Assert.Contains("prazole", steered);
        var plain = whistle.Transcribe(Clip("record.wav")).Text;
        Assert.Equal(plain, whistle.Transcribe(Clip("record.wav"), keywords: new[] { "Siobhan", "Krzysztof" }).Text);
    }

    [Fact]
    public void StreamingConvergesOnTheOfflineTranscript()
    {
        var pcm = Clip("record.wav");
        var stream = whistle.Stream();
        var committed = new System.Collections.Generic.List<string>();
        for (var at = 0; at < pcm.Length; at += 16000)
        {
            var step = stream.Process(pcm.AsSpan(at, Math.Min(16000, pcm.Length - at)));
            if (step.Text.Length > 0) committed.Add(step.Text);
        }
        var tail = stream.Stop();
        if (tail.Text.Length > 0) committed.Add(tail.Text);
        Assert.Contains("speed of sound at sea level", string.Join(" ", committed));
    }

    [Fact]
    public void ThreadCountDoesNotChangeTheTranscript()
    {
        using var single = new Whistle(model, 1);
        Assert.Equal(single.Embed(Clip("test.wav")), whistle.Embed(Clip("test.wav")));
        Assert.Equal(single.Transcribe(Clip("record.wav")).Text, whistle.Transcribe(Clip("record.wav")).Text);
    }

    [Fact]
    public void EmbeddingHasOneRowPerEightyMilliseconds()
    {
        var pcm = Clip("record.wav");
        var embedding = whistle.Embed(pcm);
        Assert.Equal(0, embedding.Length % 512);
        Assert.InRange(embedding.Length / 512, pcm.Length / 1280, pcm.Length / 1280 + 2);
        Assert.All(embedding, v => Assert.True(float.IsFinite(v)));
    }

    static void AssertSpanMatches(Transcription expected, SpeechSpan actual, byte[] utf8, WordSlice[] slices, string clip)
    {
        Assert.False(actual.Truncated, clip);
        Assert.Equal(expected.Text, Encoding.UTF8.GetString(utf8, 0, actual.Utf8Length));
        Assert.Equal(expected.Language, actual.Language);
        var words = expected.Words!;
        Assert.Equal(words.Count, actual.WordCount);
        for (var i = 0; i < words.Count; i++)
        {
            Assert.Equal(words[i].Text, Encoding.UTF8.GetString(utf8, slices[i].Utf8Offset, slices[i].Utf8Length));
            Assert.True(Math.Abs(words[i].Start - slices[i].Start) < 1e-9, $"{clip} word {i} start");
            Assert.True(Math.Abs(words[i].End - slices[i].End) < 1e-9, $"{clip} word {i} end");
            Assert.True(Math.Abs(words[i].Probability - slices[i].Probability) < 1e-6, $"{clip} word {i} probability");
        }
    }

    [Fact]
    public void BeamSpanMatchesTheAllocatingTranscript()
    {
        Assert.Equal(3, whistle.BeamWidth);
        var utf8 = new byte[whistle.MaxTranscriptTokens * 64];
        var slices = new WordSlice[whistle.MaxTranscriptTokens + 1];
        foreach (var name in new[] { "record.wav", "test.wav", "hotword.wav" })
        {
            var pcm = Clip(name);
            AssertSpanMatches(whistle.Transcribe(pcm, wordTimestamps: true), whistle.TranscribeUtf8(pcm, utf8, slices), utf8, slices, name);
        }
        var queen = Clip("test_long.wav").AsSpan(0, 25 * 16000).ToArray();
        AssertSpanMatches(whistle.Transcribe(queen, "en", wordTimestamps: true), whistle.TranscribeUtf8(queen, utf8, slices, "en"), utf8, slices, "test_long.wav");
        Assert.Contains("queen in Victoria", Encoding.UTF8.GetString(utf8, 0, whistle.TranscribeUtf8(queen, utf8, slices, "en").Utf8Length));
        var keywords = new[] { "Omeprazole" };
        var hotword = Clip("hotword.wav");
        AssertSpanMatches(whistle.Transcribe(hotword, keywords: keywords, wordTimestamps: true), whistle.TranscribeUtf8(hotword, utf8, slices, keywords: keywords), utf8, slices, "hotword.wav keywords");
    }

    [Fact]
    public void BeamSpanTranscriptionAllocatesNothingAfterWarmUp()
    {
        var pcm = Clip("record.wav");
        var utf8 = new byte[whistle.MaxTranscriptTokens * 64];
        var slices = new WordSlice[whistle.MaxTranscriptTokens + 1];
        for (var i = 0; i < 2; i++) whistle.TranscribeUtf8(pcm, utf8, slices);
        var before = GC.GetAllocatedBytesForCurrentThread();
        var speech = whistle.TranscribeUtf8(pcm, utf8, slices);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(speech.Utf8Length > 0);
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void AnEmptyWordSpanStillReturnsTheText()
    {
        var pcm = Clip("record.wav");
        var utf8 = new byte[whistle.MaxTranscriptTokens * 64];
        var speech = whistle.TranscribeUtf8(pcm, utf8, Span<WordSlice>.Empty);
        Assert.Equal(whistle.Transcribe(pcm).Text, Encoding.UTF8.GetString(utf8, 0, speech.Utf8Length));
        Assert.Equal(0, speech.WordCount);
        Assert.False(speech.Truncated);
        Assert.Equal("en", speech.Language);
    }

    [Fact]
    public void NonFiniteAudioIsTreatedAsSilence()
    {
        var pcm = Clip("record.wav");
        pcm[pcm.Length / 2] = float.NaN;
        Assert.Equal("", whistle.Transcribe(pcm).Text);
        var utf8 = new byte[1024];
        var slices = new WordSlice[64];
        Assert.Equal(0, whistle.TranscribeUtf8(pcm, utf8, slices).Utf8Length);
        pcm[pcm.Length / 2] = float.PositiveInfinity;
        Assert.Equal("", whistle.Transcribe(pcm).Text);
        Assert.Equal("In one short sentence, what is the speed of sound at sea level?", whistle.Transcribe(Clip("record.wav")).Text);
    }

    [Fact]
    public void OverlappingCallsOnOneInstanceThrowInsteadOfSharingState()
    {
        var pcm = Clip("record.wav");
        Exception? overlap = null;
        var raised = 0;
        whistle.Observer = (stage, _) =>
        {
            if (stage != "mel" || Interlocked.Exchange(ref raised, 1) != 0) return;
            var other = new Thread(() =>
            {
                try { whistle.TranscribeUtf8(pcm, new byte[1024], new WordSlice[64]); }
                catch (Exception error) { overlap = error; }
            });
            other.Start();
            other.Join();
        };
        try { Assert.Equal("In one short sentence, what is the speed of sound at sea level?", whistle.Transcribe(pcm).Text); }
        finally { whistle.Observer = null; }
        Assert.IsType<InvalidOperationException>(overlap);
        Assert.Equal("In one short sentence, what is the speed of sound at sea level?", whistle.Transcribe(pcm).Text);
    }

    [Fact]
    public void DisposeIsIdempotentAndLaterCallsThrow()
    {
        var spare = new Whistle(model, 1);
        Assert.Contains("speed of sound", spare.Transcribe(Clip("record.wav")).Text);
        spare.Dispose();
        spare.Dispose();
        Assert.Throws<ObjectDisposedException>(() => spare.Transcribe(Clip("record.wav")));
        Assert.Throws<ObjectDisposedException>(() => spare.TranscribeUtf8(Clip("record.wav"), new byte[64], new WordSlice[4]));
    }

    [Fact]
    public void FullLengthWindowsTranscribeWithoutOverrunningTheirBuffers()
    {
        var pcm = Clip("test_long.wav").AsSpan(0, Needle.Whistle.MaxSeconds * 16000).ToArray();
        var first = whistle.Transcribe(pcm, "en").Text;
        Assert.Contains("queen", first);
        Assert.Equal(first, whistle.Transcribe(pcm, "en").Text);
    }

    static byte[] WavBytes(ushort tag, ushort channels, uint rate, ushort blockAlign, int fmtLength = 16, int dataBytes = 4)
    {
        var fmt = new byte[fmtLength];
        BitConverter.GetBytes(tag).CopyTo(fmt, 0);
        if (fmtLength >= 4) BitConverter.GetBytes(channels).CopyTo(fmt, 2);
        if (fmtLength >= 8) BitConverter.GetBytes(rate).CopyTo(fmt, 4);
        if (fmtLength >= 14) BitConverter.GetBytes(blockAlign).CopyTo(fmt, 12);
        var bytes = new System.Collections.Generic.List<byte>();
        bytes.AddRange("RIFF"u8.ToArray());
        bytes.AddRange(BitConverter.GetBytes(4 + 8 + fmtLength + 8 + dataBytes));
        bytes.AddRange("WAVE"u8.ToArray());
        bytes.AddRange("fmt "u8.ToArray());
        bytes.AddRange(BitConverter.GetBytes(fmtLength));
        bytes.AddRange(fmt);
        bytes.AddRange("data"u8.ToArray());
        bytes.AddRange(BitConverter.GetBytes(dataBytes));
        bytes.AddRange(new byte[dataBytes]);
        return bytes.ToArray();
    }

    static float[] DirectSinc(float[] input, int from, int to)
    {
        var ratio = (double)to / from;
        var scale = Math.Min(1.0, ratio);
        var radius = 24 / scale;
        var output = new float[(int)(input.Length * ratio)];
        for (var n = 0; n < output.Length; n++)
        {
            var center = n / ratio;
            double sum = 0, weight = 0;
            for (var i = Math.Max(0, (int)Math.Ceiling(center - radius)); i <= Math.Min(input.Length - 1, (int)Math.Floor(center + radius)); i++)
            {
                var t = (i - center) * scale;
                var sinc = t == 0 ? 1.0 : Math.Sin(Math.PI * t) / (Math.PI * t);
                var kernel = sinc * (0.5 + 0.5 * Math.Cos(Math.PI * (i - center) / radius));
                sum += input[i] * kernel;
                weight += kernel;
            }
            output[n] = (float)(weight == 0 ? 0 : sum / weight);
        }
        return output;
    }

    [Fact]
    public void PolyphaseResamplingMatchesTheDirectWindowedSinc()
    {
        var random = new Random(7);
        foreach (var rate in new[] { 8000, 11025, 22050, 32000, 44100, 44101, 48000, 96000 })
        {
            var input = new float[rate + 37];
            for (var i = 0; i < input.Length; i++) input[i] = (float)(0.5 * Math.Sin(i * 0.013) + random.NextDouble() - 0.5);
            var expected = DirectSinc(input, rate, Wav.SampleRate);
            var actual = Wav.Resample(input, rate, Wav.SampleRate);
            Assert.Equal(expected.Length, actual.Length);
            for (var i = 0; i < expected.Length; i++) Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-6f, $"{rate} Hz sample {i}: {expected[i]} vs {actual[i]}");
        }
        var pcm = new float[48000];
        var resampled = new float[Wav.ResampledLength(pcm.Length, 48000, Wav.SampleRate)];
        Wav.Resample(pcm, 48000, Wav.SampleRate, resampled);
        var before = GC.GetAllocatedBytesForCurrentThread();
        Wav.Resample(pcm, 48000, Wav.SampleRate, resampled);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.Throws<ArgumentException>(() => Wav.Resample(pcm, 48000, Wav.SampleRate, new float[10]));
    }

    [Fact]
    public void MalformedWavHeadersAreRejectedAsInvalidData()
    {
        Assert.Equal(2, Wav.Decode(WavBytes(1, 1, 16000, 2)).Length);
        Assert.Throws<InvalidDataException>(() => Wav.Decode(WavBytes(1, 0, 16000, 2)));
        Assert.Throws<InvalidDataException>(() => Wav.Decode(WavBytes(1, 1, 16000, 2, fmtLength: 8)));
        Assert.Throws<InvalidDataException>(() => Wav.Decode(WavBytes(1, 1, 16000, 0)));
        Assert.Throws<InvalidDataException>(() => Wav.Decode(WavBytes(1, 1, 0, 2)));
        Assert.Throws<InvalidDataException>(() => Wav.Decode(WavBytes(0xFFFE, 1, 16000, 2)));
        Assert.Throws<InvalidDataException>(() => Wav.Decode("RIFF0000WAVX"u8.ToArray()));
    }
}
