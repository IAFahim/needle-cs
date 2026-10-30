using System;
using System.IO;
using System.Linq;
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
        Assert.Contains("a new queen in Victoria, the first popular monarch in centuries, and her husband Albert", text);
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
        Assert.Contains("this will transcribe properly", whistle.Transcribe(Clip("test.wav")).Text);
        var hotword = whistle.Transcribe(Clip("hotword.wav")).Text;
        Assert.Contains("part of the cactus team and, you know, I also", hotword);
        Assert.Contains("called Omep Brazil", hotword);
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
}
