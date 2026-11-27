using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using Xunit;

namespace Needle.Tests;

public sealed class TranscriberSpeechTests
{
    const string RecordText = "In one short sentence, what is the speed of sound at sea level?";

    static string ModelPath => Path.Combine(Assets.Root, "ref/hf/whistle/whistle.cact");

    static float[] Clip(string name) => Wav.Read(Path.Combine(Assets.Root, "ref/audio", name));

    static SpeechModel OpenModel()
    {
        if (!File.Exists(ModelPath)) throw new InvalidOperationException("missing whistle.cact; run tools/fetch.sh");
        return SpeechModel.Open(ModelPath);
    }

    static string Read(Utf8Span text) => text.ToString();

    [Fact]
    public void DefaultHandlesFailCleanly()
    {
        SpeechModel model = default;
        Transcriber ears = default;
        Assert.False(model.IsOpen);
        Assert.False(ears.IsOpen);
        Assert.Throws<ObjectDisposedException>(() => model.Transcriber());
        Assert.Throws<ObjectDisposedException>(() => { ears.Transcribe(new float[16]); });
        Assert.False(model.TryTranscriber(default, out _, out var error));
        Assert.Equal(NeedleError.Disposed, error);
        ears.Cancel();
        ears.Dispose();
        model.Dispose();
        Assert.False(SpeechModel.TryOpen(Path.Combine(Assets.Root, "ref/does-not-exist.cact"), out _, out error));
        Assert.Equal(NeedleError.NotFound, error);
    }

    [Fact]
    public void ANeedleModelIsNotASpeechModel()
    {
        var needle = Path.Combine(Assets.Root, "ref/hf/needle3.cact");
        if (!File.Exists(needle)) throw new InvalidOperationException("missing needle3.cact; run tools/fetch.sh");
        Assert.Throws<InvalidDataException>(() => SpeechModel.Open(needle));
        Assert.False(SpeechModel.TryOpen(needle, out _, out var error));
        Assert.Equal(NeedleError.InvalidModel, error);
    }

    [Fact]
    public void TranscribesAt16kAndAtTheMicrophoneRate()
    {
        using var model = OpenModel();
        using var ears = model.Transcriber(new TranscriberOptions { Threads = 2 });
        var pcm = Clip("record.wav");
        var heard = ears.Transcribe(pcm);
        Assert.True(heard.Ok);
        Assert.Equal(TranscriptKind.Speech, heard.Kind);
        Assert.Equal(TranscriptWhy.None, heard.Why);
        Assert.Equal(RecordText, Read(heard.Text));
        Assert.True(heard.Language.Equals("en"));
        Assert.Equal(0, heard.Words.Count);

        var mic = Wav.Resample(pcm, Wav.SampleRate, 48000);
        var fromMic = ears.Transcribe(mic, 48000);
        Assert.True(fromMic.Ok);
        Assert.Contains("speed of sound at sea level", Read(fromMic.Text));

        using var legacy = new Whistle(model.Engine, 2);
        Assert.Equal(legacy.Transcribe(pcm).Text, Read(ears.Transcribe(pcm).Text));
    }

    [Fact]
    public void ATranscriptGoesStaleAfterTheNextCall()
    {
        using var model = OpenModel();
        using var ears = model.Transcriber(new TranscriberOptions { Threads = 2 });
        var first = ears.Transcribe(Clip("record.wav"));
        Assert.True(first.Ok);
        ears.Transcribe(new float[Wav.SampleRate]);
        Exception? stale = null;
        try { _ = first.Text; }
        catch (Exception error) { stale = error; }
        Assert.IsType<InvalidOperationException>(stale);
    }

    [Fact]
    public void LanguagesSilenceAndBadInputAreReportedNotThrown()
    {
        using var model = OpenModel();
        using var ears = model.Transcriber(new TranscriberOptions { Threads = 2 });
        var pcm = Clip("record.wav");
        Assert.True(ears.LanguageCount > 1);
        var found = false;
        for (var i = 0; i < ears.LanguageCount; i++) found |= ears.Language(i).Equals("de");
        Assert.True(found);

        var german = ears.Transcribe(pcm, "de"u8);
        Assert.True(german.Language.Equals("de"));

        var unknown = ears.Transcribe(pcm, "xx"u8);
        Assert.False(unknown.Ok);
        Assert.Equal(TranscriptKind.Failed, unknown.Kind);
        Assert.Equal(TranscriptWhy.UnknownLanguage, unknown.Why);

        var silence = ears.Transcribe(new float[Wav.SampleRate * 2]);
        Assert.Equal(TranscriptKind.Silence, silence.Kind);
        Assert.Equal(TranscriptWhy.Silence, silence.Why);
        Assert.True(silence.Text.IsEmpty);
        Assert.True(silence.Language.IsEmpty);

        Assert.Equal(TranscriptWhy.BadSampleRate, ears.Transcribe(pcm, 10).Why);
        using var small = model.Transcriber(new TranscriberOptions { Threads = 1, MaxSeconds = 1 });
        Assert.Equal(TranscriptWhy.InputTooLong, small.Transcribe(new float[48000 * 2], 48000).Why);
    }

    [Fact]
    public void WordsCarryTimesAndConfidence()
    {
        using var model = OpenModel();
        using var ears = model.Transcriber(new TranscriberOptions { Threads = 2, Words = Toggle.On });
        var heard = ears.Transcribe(Clip("record.wav"));
        Assert.True(heard.Ok);
        Assert.True(heard.Words.Count >= 10);
        var joined = new StringBuilder();
        var previous = -1.0;
        foreach (var word in heard)
        {
            if (joined.Length > 0) joined.Append(' ');
            joined.Append(word.Text.ToString());
            Assert.True(word.Start >= previous - 1e-9);
            Assert.True(word.End >= word.Start);
            Assert.InRange(word.Probability, 0f, 1f);
            previous = word.Start;
        }
        Assert.Equal(Read(heard.Text), joined.ToString());
    }

    [Fact]
    public void RepeatCallsAllocateNothing()
    {
        using var model = OpenModel();
        using var ears = model.Transcriber(new TranscriberOptions { Threads = 2 });
        ears.SetKeywords("Lyra", "Omep");
        var pcm = Clip("record.wav");
        var mic = Wav.Resample(pcm, Wav.SampleRate, 48000);
        ears.WarmUp();
        ears.Transcribe(pcm);
        ears.Transcribe(mic, 48000, "en"u8);
        var before = GC.GetAllocatedBytesForCurrentThread();
        var a = ears.Transcribe(pcm).Text.Length;
        var b = ears.Transcribe(mic, 48000, "en"u8).Text.Length;
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(a > 0 && b > 0);
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void CancelStopsATranscriptionInFlight()
    {
        using var model = OpenModel();
        var ears = model.Transcriber(new TranscriberOptions { Threads = 2 });
        try
        {
            ears.Engine.Observer = (stage, _) => { if (stage == "mel") ears.Cancel(); };
            var heard = ears.Transcribe(Clip("test_long.wav"));
            Assert.Equal(TranscriptKind.Cancelled, heard.Kind);
            Assert.Equal(TranscriptWhy.Cancelled, heard.Why);
            Assert.False(heard.Ok);
            ears.Engine.Observer = null;
            Assert.True(ears.Transcribe(Clip("record.wav")).Ok);
        }
        finally
        {
            ears.Dispose();
        }
    }

    [Fact]
    public void TheModelLivesUntilItsLastTranscriberIsDisposed()
    {
        var model = OpenModel();
        var ears = model.Transcriber(new TranscriberOptions { Threads = 1 });
        var copy = ears;
        model.Dispose();
        model.Dispose();
        Assert.False(model.IsOpen);
        Assert.Equal(RecordText, Read(ears.Transcribe(Clip("record.wav")).Text));
        ears.Dispose();
        copy.Dispose();
        Assert.False(copy.IsOpen);
        Assert.Throws<ObjectDisposedException>(() => { copy.Transcribe(new float[16]); });
    }

    static string TakeText(Transcriber ears)
    {
        var clock = Stopwatch.StartNew();
        while (clock.Elapsed < TimeSpan.FromMinutes(2))
        {
            if (ears.TryTake(out var heard)) return heard.Ok ? Read(heard.Text) : heard.ToString();
            Thread.Sleep(1);
        }
        throw new TimeoutException("the posted transcription did not finish");
    }

    [Fact]
    public void PostedTranscriptsMatchTranscribing()
    {
        using var model = OpenModel();
        using var ears = model.Transcriber(new TranscriberOptions { Threads = 2 });
        var pcm = Clip("record.wav");
        var direct = Read(ears.Transcribe(pcm).Text);
        Assert.Equal(RecordText, direct);

        Assert.False(ears.TryTake(out _));
        Assert.True(ears.TryPost(pcm));
        Array.Clear(pcm, 0, pcm.Length);
        Assert.False(ears.TryPost(pcm));
        if (ears.IsBusy) Assert.Throws<InvalidOperationException>(() => { ears.Transcribe(pcm); });
        Assert.Equal(direct, TakeText(ears));
        Assert.False(ears.IsBusy);
        Assert.False(ears.TryTake(out _));

        var mic = Wav.Resample(Clip("record.wav"), Wav.SampleRate, 48000);
        ears.Post(mic, 48000, "en"u8);
        var heard = ears.Take();
        Assert.True(heard.Ok);
        Assert.True(heard.Language.Equals("en"));
        Assert.Contains("speed of sound at sea level", Read(heard.Text));

        Assert.True(ears.TryPost(pcm, "xx"u8));
        Assert.True(ears.TryTake(out var refused));
        Assert.Equal(TranscriptWhy.UnknownLanguage, refused.Why);
        Assert.Throws<InvalidOperationException>(() => { ears.Take(); });
    }

    [Fact]
    public void PostingAllocatesNothingOnTheCallingThread()
    {
        using var model = OpenModel();
        using var ears = model.Transcriber(new TranscriberOptions { Threads = 2 });
        var pcm = Clip("record.wav");
        var mic = Wav.Resample(pcm, Wav.SampleRate, 48000);
        for (var i = 0; i < 2; i++)
        {
            ears.Post(pcm);
            ears.Take();
            ears.Post(mic, 48000);
            ears.Take();
        }
        var before = GC.GetAllocatedBytesForCurrentThread();
        var bytes = 0;
        for (var i = 0; i < 3; i++)
        {
            ears.Post(i % 2 == 0 ? pcm : mic, i % 2 == 0 ? Wav.SampleRate : 48000);
            Transcript heard;
            while (!ears.TryTake(out heard)) Thread.SpinWait(64);
            bytes += heard.Text.Length;
        }
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.True(bytes > 0);
    }

    [Fact]
    public void CancelAndDisposeStopAPostedTranscription()
    {
        using var model = OpenModel();
        var ears = model.Transcriber(new TranscriberOptions { Threads = 2 });
        try
        {
            var queen = Clip("test_long.wav");
            ears.Engine.Observer = (stage, _) => { if (stage == "mel") ears.Cancel(); };
            ears.Post(queen);
            var heard = ears.Take();
            Assert.Equal(TranscriptKind.Cancelled, heard.Kind);
            ears.Engine.Observer = null;
            Assert.Equal(RecordText, Read(ears.Transcribe(Clip("record.wav")).Text));
            ears.Post(queen);
            var clock = Stopwatch.StartNew();
            ears.Dispose();
            Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30));
            Assert.False(ears.IsOpen);
        }
        finally
        {
            ears.Dispose();
        }
    }
}
