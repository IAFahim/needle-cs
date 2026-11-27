using System;
using System.Collections.Generic;
using System.Text;

namespace Needle;

/// <summary>What came out of a piece of audio.</summary>
public enum TranscriptKind : byte
{
    /// <summary>Someone spoke; the words are in <see cref="Transcript.Text"/>.</summary>
    Speech,
    /// <summary>The audio was silent or held no words.</summary>
    Silence,
    /// <summary>The audio could not be transcribed; <see cref="Transcript.Why"/> says why.</summary>
    Failed,
    /// <summary><see cref="Transcriber.Cancel"/> stopped the call. Any words decoded before that are in <see cref="Transcript.Text"/>.</summary>
    Cancelled,
}

/// <summary>Why a transcript is not <see cref="Transcript.Ok"/>.</summary>
public enum TranscriptWhy : byte
{
    None,
    Silence,
    Cancelled,
    UnknownLanguage,
    InputTooLong,
    BadSampleRate,
    Failed,
}

internal sealed partial class TranscriberState
{
    public readonly ModelState Model;
    public readonly Whistle Whistle;
    public readonly byte[] Text;
    public readonly WordSlice[] Words;
    public readonly float[] Resampled;
    public readonly byte[][] LanguagesUtf8;
    public readonly bool WantWords;
    public IReadOnlyList<string>? Keywords;
    public int Version;
    public bool Disposed;
    public bool HasResult;
    public SpeechSpan Last;
    public TranscriptKind Kind;
    public TranscriptWhy Why;
    public int LanguageIndex = -1;
    public double AudioSeconds;

    public TranscriberState(ModelState model, in TranscriberOptions options)
    {
        Model = model;
        Whistle = new Whistle(model.Model, options.Policy);
        try
        {
            Whistle.BeamWidth = options.Beam;
            WantWords = options.Words == Toggle.On;
            Text = new byte[options.TextBytes];
            Words = WantWords ? new WordSlice[Math.Max(64, options.TextBytes / 2)] : Array.Empty<WordSlice>();
            Resampled = new float[options.Seconds * Wav.SampleRate];
            var languages = Whistle.Languages;
            LanguagesUtf8 = new byte[languages.Count][];
            for (var i = 0; i < languages.Count; i++) LanguagesUtf8[i] = Encoding.UTF8.GetBytes(languages[i]);
        }
        catch
        {
            Whistle.Dispose();
            throw;
        }
    }

    public int FindLanguage(ReadOnlySpan<byte> code)
    {
        for (var i = 0; i < LanguagesUtf8.Length; i++)
            if (code.SequenceEqual(LanguagesUtf8[i])) return i;
        return -1;
    }

    public void Invalidate()
    {
        Version = Version == int.MaxValue ? 1 : Version + 1;
        HasResult = false;
    }

    public void Fail(TranscriptWhy why, double seconds)
    {
        Last = default;
        LanguageIndex = -1;
        AudioSeconds = seconds;
        Kind = TranscriptKind.Failed;
        Why = why;
        HasResult = true;
    }

    public void Record(in SpeechSpan span, double seconds, bool cancelled)
    {
        Last = span;
        AudioSeconds = seconds;
        LanguageIndex = -1;
        if (span.Language != null)
            for (var i = 0; i < Whistle.Languages.Count; i++)
                if (ReferenceEquals(Whistle.Languages[i], span.Language) || Whistle.Languages[i] == span.Language) { LanguageIndex = i; break; }
        if (cancelled) (Kind, Why) = (TranscriptKind.Cancelled, TranscriptWhy.Cancelled);
        else if (span.Utf8Length == 0) (Kind, Why) = (TranscriptKind.Silence, TranscriptWhy.Silence);
        else (Kind, Why) = (TranscriptKind.Speech, TranscriptWhy.None);
        HasResult = true;
    }
}

/// <summary>
/// Turns audio into text. A transcriber is a handle: copying it copies the handle, not the decoder, and a disposed or
/// <c>default</c> transcriber throws <see cref="ObjectDisposedException"/>. One transcriber serves one thread at a time;
/// after the first call on each audio length it allocates nothing.
/// </summary>
public readonly partial struct Transcriber : IDisposable, IEquatable<Transcriber>
{
    readonly int slot;
    readonly int generation;

    Transcriber(int slot, int generation)
    {
        this.slot = slot;
        this.generation = generation;
    }

    /// <summary>Loads the model and makes a transcriber that owns it. Disposing the transcriber frees the model.</summary>
    public static Transcriber Open(string modelPath, in TranscriberOptions options = default)
    {
        using var model = SpeechModel.Open(modelPath);
        return model.Transcriber(options);
    }

    internal static Transcriber Create(ModelState model, in TranscriberOptions options)
    {
        model.AddReference();
        TranscriberState state;
        try
        {
            state = new TranscriberState(model, options);
        }
        catch
        {
            model.Release();
            throw;
        }
        HandleTable<TranscriberState>.Add(state, out var slot, out var generation);
        return new Transcriber(slot, generation);
    }

    public bool IsOpen => HandleTable<TranscriberState>.TryGet(slot, generation, out _);

    TranscriberState State => HandleTable<TranscriberState>.Get(slot, generation, nameof(Transcriber));

    TranscriberState Idle
    {
        get
        {
            var state = State;
            if (state.Mailbox is { Busy: true }) throw new InvalidOperationException("a posted transcription is still running on this transcriber; TryTake it first");
            return state;
        }
    }

    /// <summary>The engine-level transcriber, for APIs that still take one.</summary>
    public Whistle Engine => State.Whistle;

    /// <summary>Language codes the model knows, such as <c>en</c> and <c>de</c>.</summary>
    public int LanguageCount => State.LanguagesUtf8.Length;

    public Utf8Span Language(int index) => new(State.LanguagesUtf8[index]);

    /// <summary>Transcribes 16 kHz mono audio. The transcript stays readable until the next call on this transcriber.</summary>
    public Transcript Transcribe(ReadOnlySpan<float> pcm16k) => Run(Idle, pcm16k, -1, pcm16k.Length / (double)Wav.SampleRate);

    /// <summary>Transcribes 16 kHz mono audio in a given language, such as <c>"en"u8</c>. An empty code detects the language.</summary>
    public Transcript Transcribe(ReadOnlySpan<float> pcm16k, ReadOnlySpan<byte> language)
    {
        var state = Idle;
        var index = -1;
        if (!language.IsEmpty && (index = state.FindLanguage(language)) < 0)
        {
            state.Invalidate();
            state.Fail(TranscriptWhy.UnknownLanguage, pcm16k.Length / (double)Wav.SampleRate);
            return new Transcript(state, state.Version);
        }
        return Run(state, pcm16k, index, pcm16k.Length / (double)Wav.SampleRate);
    }

    /// <summary>Transcribes mono audio at any sample rate, such as a 48 kHz microphone. It is resampled into the transcriber's own buffer.</summary>
    public Transcript Transcribe(ReadOnlySpan<float> pcm, int sampleRate, ReadOnlySpan<byte> language = default)
    {
        var state = Idle;
        if (sampleRate == Wav.SampleRate) return Transcribe(pcm, language);
        var seconds = sampleRate > 0 ? pcm.Length / (double)sampleRate : 0;
        if (sampleRate < 1000 || sampleRate > 768000)
        {
            state.Invalidate();
            state.Fail(TranscriptWhy.BadSampleRate, seconds);
            return new Transcript(state, state.Version);
        }
        var length = Wav.ResampledLength(pcm.Length, sampleRate, Wav.SampleRate);
        if (length > state.Resampled.Length)
        {
            state.Invalidate();
            state.Fail(TranscriptWhy.InputTooLong, seconds);
            return new Transcript(state, state.Version);
        }
        Wav.Resample(pcm, sampleRate, Wav.SampleRate, state.Resampled);
        return Transcribe(new ReadOnlySpan<float>(state.Resampled, 0, length), language);
    }

    /// <summary>Words to listen for, such as character and place names. Set them once; each call then allocates nothing.</summary>
    public void SetKeywords(params string[]? keywords)
    {
        var state = Idle;
        state.Invalidate();
        state.Keywords = keywords == null || keywords.Length == 0 ? null : (string[])keywords.Clone();
    }

    /// <summary>Stops a transcription running on another thread as soon as it can. Safe to call from any thread.</summary>
    public void Cancel()
    {
        if (!HandleTable<TranscriberState>.TryGet(slot, generation, out var state)) return;
        state.Mailbox?.Cancel();
        state.Whistle.Cancel();
    }

    /// <summary>Runs the model once on silence-free noise so the first real call is not the slow one.</summary>
    public void WarmUp()
    {
        var state = Idle;
        Span<float> noise = stackalloc float[Wav.SampleRate / 2];
        var seed = 12345u;
        for (var i = 0; i < noise.Length; i++)
        {
            seed = seed * 1664525u + 1013904223u;
            noise[i] = ((seed >> 9) / (float)(1 << 23) - 0.5f) * 0.02f;
        }
        Run(state, noise, -1, noise.Length / (double)Wav.SampleRate);
        state.Invalidate();
    }

    public void Dispose()
    {
        var state = HandleTable<TranscriberState>.Remove(slot, generation);
        if (state == null) return;
        try
        {
            state.StopMailbox();
        }
        finally
        {
            state.Invalidate();
            state.Disposed = true;
        }
        try
        {
            state.Whistle.Dispose();
        }
        finally
        {
            state.Model.Release();
        }
    }

    internal static Transcript Run(TranscriberState state, ReadOnlySpan<float> pcm, int language, double seconds)
    {
        state.Invalidate();
        SpeechSpan span;
        try
        {
            span = state.Whistle.TranscribeUtf8(pcm, state.Text, state.Words, language < 0 ? null : state.Whistle.Languages[language], state.Keywords);
        }
        catch (InvalidOperationException)
        {
            throw;
        }
        catch (Exception)
        {
            state.Fail(TranscriptWhy.Failed, seconds);
            return new Transcript(state, state.Version);
        }
        state.Record(span, seconds, state.Whistle.WasCancelled);
        return new Transcript(state, state.Version);
    }

    public bool Equals(Transcriber other) => slot == other.slot && generation == other.generation;

    public override bool Equals(object? obj) => obj is Transcriber other && Equals(other);

    public override int GetHashCode() => (slot * 397) ^ generation;

    public static bool operator ==(Transcriber left, Transcriber right) => left.Equals(right);

    public static bool operator !=(Transcriber left, Transcriber right) => !left.Equals(right);
}

/// <summary>
/// The text of one piece of audio. It reads the transcriber's own buffers, so it costs nothing, and it is only valid
/// until the next call on the same transcriber; reading it after that throws <see cref="InvalidOperationException"/>.
/// <c>foreach</c> over a transcript walks its words when word timings are on.
/// </summary>
public readonly ref struct Transcript
{
    readonly TranscriberState state;
    readonly int version;

    internal Transcript(TranscriberState state, int version)
    {
        this.state = state;
        this.version = version;
    }

    TranscriberState Live
    {
        get
        {
            if (state == null) throw new InvalidOperationException("this transcript is empty; get one from Transcriber.Transcribe");
            if (state.Version != version || !state.HasResult)
            {
                if (state.Disposed) throw new ObjectDisposedException(nameof(Transcriber), "the transcriber behind this transcript has been disposed");
                throw new InvalidOperationException("a later call on the same transcriber replaced this transcript; read what you need before you transcribe again");
            }
            return state;
        }
    }

    public TranscriptKind Kind => Live.Kind;

    /// <summary>True when someone spoke and the text is complete.</summary>
    public bool Ok => Live.Kind == TranscriptKind.Speech;

    public TranscriptWhy Why => Live.Why;

    /// <summary>What was said, as UTF-8.</summary>
    public Utf8Span Text
    {
        get
        {
            var live = Live;
            return new Utf8Span(new ReadOnlySpan<byte>(live.Text, 0, live.Last.Utf8Length));
        }
    }

    /// <summary>The spoken language code, such as <c>en</c>. Empty for silence.</summary>
    public Utf8Span Language
    {
        get
        {
            var live = Live;
            return live.LanguageIndex < 0 ? default : new Utf8Span(live.LanguagesUtf8[live.LanguageIndex]);
        }
    }

    /// <summary>The text did not fit <see cref="TranscriberOptions.MaxTextBytes"/>, so the end is missing.</summary>
    public bool Truncated => Live.Last.Truncated;

    /// <summary>Each word with its time and confidence. Empty unless <see cref="TranscriberOptions.Words"/> is on.</summary>
    public WordList Words
    {
        get
        {
            var live = Live;
            return new WordList(live, version, live.Last.WordCount);
        }
    }

    /// <summary>Length of the audio, in seconds.</summary>
    public double AudioSeconds => Live.AudioSeconds;

    /// <summary>Time spent encoding the audio, in milliseconds.</summary>
    public double EncodeMs => Live.Last.EncodeMs;

    /// <summary>Time from the call to the first text token, in milliseconds.</summary>
    public double TimeToFirstTokenMs => Live.Last.TimeToFirstTokenMs;

    /// <summary>Decoded text tokens per second.</summary>
    public double DecodeTps => Live.Last.DecodeTps;

    public WordList.Enumerator GetEnumerator() => Words.GetEnumerator();

    public override string ToString()
    {
        if (state == null) return "(empty transcript)";
        var live = Live;
        return live.Kind == TranscriptKind.Speech ? Text.ToString() : $"{live.Kind}: {live.Why}";
    }

    internal static TranscriberState Check(TranscriberState state, int version)
    {
        if (state.Version != version || !state.HasResult)
            throw new InvalidOperationException("a later call on the same transcriber replaced this transcript; read what you need before you transcribe again");
        return state;
    }
}

/// <summary>One recognised word.</summary>
public readonly ref struct SpokenWord
{
    internal SpokenWord(Utf8Span text, double start, double end, float probability)
    {
        Text = text;
        Start = start;
        End = end;
        Probability = probability;
    }

    public Utf8Span Text { get; }

    /// <summary>Start of the word in the audio, in seconds.</summary>
    public double Start { get; }

    /// <summary>End of the word in the audio, in seconds.</summary>
    public double End { get; }

    /// <summary>How sure the model is of this word, from 0 to 1.</summary>
    public float Probability { get; }

    public override string ToString() => Text.ToString();
}

/// <summary>The words of a transcript, valid as long as the transcript is.</summary>
public readonly ref struct WordList
{
    readonly TranscriberState state;
    readonly int version;

    internal WordList(TranscriberState state, int version, int count)
    {
        this.state = state;
        this.version = version;
        Count = count;
    }

    public int Count { get; }

    public SpokenWord this[int index]
    {
        get
        {
            if ((uint)index >= (uint)Count) throw new ArgumentOutOfRangeException(nameof(index));
            var live = Transcript.Check(state, version);
            var w = live.Words[index];
            return new SpokenWord(new Utf8Span(new ReadOnlySpan<byte>(live.Text, w.Utf8Offset, w.Utf8Length)), w.Start, w.End, (float)w.Probability);
        }
    }

    public Enumerator GetEnumerator() => new(this);

    public ref struct Enumerator
    {
        readonly WordList list;
        int index;

        internal Enumerator(WordList list)
        {
            this.list = list;
            index = -1;
        }

        public bool MoveNext() => ++index < list.Count;

        public SpokenWord Current => list[index];
    }
}
