using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace Needle;

public sealed class PushToTalk
{
    readonly float[] buffer;
    int count;

    public PushToTalk(int maxSeconds = Whistle.MaxSeconds)
    {
        if (maxSeconds < 1 || maxSeconds > Whistle.MaxSeconds) throw new ArgumentOutOfRangeException(nameof(maxSeconds));
        buffer = new float[maxSeconds * Wav.SampleRate];
    }

    public int CapacitySamples => buffer.Length;

    public int BufferedSamples => Volatile.Read(ref count);

    public double BufferedSeconds => BufferedSamples / (double)Wav.SampleRate;

    public bool IsHeld => BufferedSamples > 0;

    public void Hold() => Volatile.Write(ref count, 0);

    public bool Write(ReadOnlySpan<float> pcm)
    {
        var current = Volatile.Read(ref count);
        if (pcm.Length == 0) return true;
        if (current + pcm.Length > buffer.Length) return false;
        pcm.CopyTo(buffer.AsSpan(current));
        Volatile.Write(ref count, current + pcm.Length);
        return true;
    }

    public SpeechSpan Preview(Whistle whistle, Span<byte> utf8, Span<WordSlice> words, string? language = null, IReadOnlyList<string>? keywords = null)
    {
        var n = Volatile.Read(ref count);
        return whistle.TranscribeUtf8(buffer.AsSpan(0, n), utf8, words, language, keywords);
    }

    public SpeechSpan Release(Whistle whistle, Span<byte> utf8, Span<WordSlice> words, string? language = null, IReadOnlyList<string>? keywords = null)
    {
        var speech = Preview(whistle, utf8, words, language, keywords);
        Volatile.Write(ref count, 0);
        return speech;
    }
}

public readonly struct StreamDelta
{
    public readonly int CommittedUtf8;
    public readonly int CommittedWords;
    public readonly int PendingUtf8;
    public readonly int PendingWords;
    public readonly double ReceivedSeconds;
    public readonly double PassMs;
    public readonly bool Truncated;
    /// <summary>Where in the stream the audio of this call's pass began, in seconds; 0 when no pass ran.</summary>
    public readonly double PassStartSeconds;
    /// <summary>How much audio this call's pass covered, in seconds; 0 when no pass ran.</summary>
    public readonly double PassSeconds;
    /// <summary>True when this call's pass committed every word it heard (a pause, or the end of the stream).</summary>
    public readonly bool Final;

    public StreamDelta(int committedUtf8, int committedWords, int pendingUtf8, int pendingWords, double receivedSeconds, double passMs, bool truncated)
        : this(committedUtf8, committedWords, pendingUtf8, pendingWords, receivedSeconds, passMs, truncated, 0, 0, false)
    {
    }

    public StreamDelta(int committedUtf8, int committedWords, int pendingUtf8, int pendingWords, double receivedSeconds, double passMs, bool truncated,
        double passStartSeconds, double passSeconds, bool final)
    {
        CommittedUtf8 = committedUtf8;
        CommittedWords = committedWords;
        PendingUtf8 = pendingUtf8;
        PendingWords = pendingWords;
        ReceivedSeconds = receivedSeconds;
        PassMs = passMs;
        Truncated = truncated;
        PassStartSeconds = passStartSeconds;
        PassSeconds = passSeconds;
        Final = final;
    }
}

/// <summary>
/// Streaming transcription that never encodes committed audio again. Words are committed, and their audio dropped,
/// at a pause (voice activity detection), at <see cref="Stop"/>, or when the uncommitted window outgrows
/// <see cref="MaxWindowSeconds"/>; until then <see cref="Update"/> shows the window's words as pending, re-reading
/// the window at most every <see cref="PartialSeconds"/> of new audio. Silence before speech is dropped down to a
/// short pre-roll and never transcribed. So every second of audio is encoded a bounded number of times and the total
/// cost grows linearly with the audio, where re-reading the whole stream on every pass grew quadratically.
/// A committing pass reads exactly the audio between two commit points, so its words are what
/// <see cref="Whistle.TranscribeUtf8"/> gives for that audio, bit for bit.
/// </summary>
public sealed class VoiceStream
{
    readonly SpeechStream core;

    public VoiceStream(Whistle whistle, string? language = null, IReadOnlyList<string>? keywords = null, int maxSeconds = Whistle.MaxSeconds)
    {
        if (maxSeconds < 1 || maxSeconds > Whistle.MaxSeconds) throw new ArgumentOutOfRangeException(nameof(maxSeconds));
        core = new SpeechStream(whistle, language, keywords, maxSeconds);
    }

    public double ReceivedSeconds => core.ReceivedSeconds;

    public double BufferedSeconds => core.BufferedSeconds;

    /// <summary>Seconds of audio the stream has transcribed so far, counting every pass: the stream's whole cost.</summary>
    public double EncodedSeconds => core.EncodedSeconds;

    /// <summary>Re-read the pending window after at least this much new audio. 0.5 s by default.</summary>
    public double PartialSeconds { get => core.PartialSeconds; set => core.PartialSeconds = value; }

    /// <summary>Silence after speech that ends a phrase and commits its words. 0.7 s by default; 0 turns pause commits off.</summary>
    public double EndpointSeconds { get => core.EndpointSeconds; set => core.EndpointSeconds = value; }

    /// <summary>The longest uncommitted window before its older words are committed anyway. 8 s by default.</summary>
    public double MaxWindowSeconds { get => core.MaxWindowSeconds; set => core.MaxWindowSeconds = value; }

    public bool Write(ReadOnlySpan<float> chunk) => core.Write(chunk);

    public StreamDelta Update(Span<byte> committedText, Span<WordSlice> committedWords, Span<byte> pendingText, Span<WordSlice> pendingWords) =>
        core.RunToEnd(final: false, committedText, committedWords, pendingText, pendingWords);

    public StreamDelta Stop(Span<byte> committedText, Span<WordSlice> committedWords, Span<byte> pendingText, Span<WordSlice> pendingWords) =>
        core.RunToEnd(final: true, committedText, committedWords, pendingText, pendingWords);

    internal static bool FoldedEqual(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b) => SpeechStream.FoldedEqual(a, b);
}

/// <summary>
/// Voice activity in 20 ms frames: an energy gate over an adaptive noise floor, with hangover at both edges.
/// Speech must clear both an absolute floor (about -48 dBFS) and 2.5 times the background.
/// </summary>
internal struct VoiceActivity
{
    public const int FrameSamples = Wav.SampleRate / 50;
    public const float MinSpeechRms = 0.004f;
    public const float NoiseFactor = 2.5f;
    public const int StartFrames = 3;

    double squares;
    int filled;
    float noise;
    int speechRun;
    int silenceRun;

    public bool InSpeech;
    public int EndFrames;

    public static VoiceActivity Create(double endpointSeconds) => new() { EndFrames = Frames(endpointSeconds) };

    public static int Frames(double seconds) => seconds <= 0 ? int.MaxValue : Math.Max(1, (int)Math.Round(seconds * 50));

    /// <summary>Feeds one sample. Returns the edge that a frame completed with: +1 speech began, -1 speech ended, 0 none.</summary>
    public int Push(float sample)
    {
        squares += sample * (double)sample;
        if (++filled < FrameSamples) return 0;
        var rms = (float)Math.Sqrt(squares / FrameSamples);
        squares = 0;
        filled = 0;
        return Frame(rms);
    }

    public int Frame(float rms)
    {
        var speech = rms > Math.Max(MinSpeechRms, noise * NoiseFactor);
        // The floor follows quiet frames quickly and loud ones slowly (about ten seconds), so a steady loud
        // background stops counting as speech after a while but a long sentence does not.
        if (!speech) noise = noise == 0 ? rms : Math.Min(rms, noise * 0.95f + rms * 0.05f);
        else if (rms > noise) noise += (rms - noise) * 0.002f;
        if (speech)
        {
            silenceRun = 0;
            if (++speechRun >= StartFrames && !InSpeech)
            {
                InSpeech = true;
                return 1;
            }
            return 0;
        }
        speechRun = 0;
        if (InSpeech && ++silenceRun >= EndFrames)
        {
            InSpeech = false;
            silenceRun = 0;
            return -1;
        }
        return 0;
    }

    public int SilentFrames => silenceRun;
}

internal enum SpeechPassKind : byte { None, Partial, Window, Phrase, Final }

/// <summary>
/// The incremental core behind <see cref="VoiceStream"/> and <see cref="NeedleVoice"/>: the window of uncommitted
/// audio, voice activity, and one pass at a time that can run whole or a step at a time.
/// </summary>
internal sealed class SpeechStream : ISampleSink
{
    const double PreRollSeconds = 0.3;
    const double TailSeconds = 0.2;
    const double KeepSeconds = 2.0;

    readonly Whistle whistle;
    readonly string? language;
    readonly IReadOnlyList<string>? keywords;
    readonly float[] pcm;
    readonly byte[] text;
    readonly WordSlice[] words;
    readonly byte[] pendingText;
    readonly WordSlice[] pendingWords;
    int count;
    long origin;
    long received;
    long encoded;
    VoiceActivity vad;
    bool hadSpeech;
    long speechStart = -1;
    long lastSpeechEnd = -1;
    long endpoint = -1;
    int lastPassAt;
    bool held;
    string? detected;
    int pendingUtf8;
    int pendingCount;
    SpeechPassKind running;
    int passLength;
    long passStarted;
    bool passDirect;

    public SpeechStream(Whistle whistle, string? language, IReadOnlyList<string>? keywords, int maxSeconds)
    {
        this.whistle = whistle;
        this.language = language;
        this.keywords = keywords;
        pcm = new float[maxSeconds * Wav.SampleRate];
        var cap = whistle.MaxTranscriptTokens * 64;
        text = new byte[cap];
        words = new WordSlice[whistle.MaxTranscriptTokens + 1];
        pendingText = new byte[cap];
        pendingWords = new WordSlice[whistle.MaxTranscriptTokens + 1];
        EndpointSeconds = 0.7;
    }

    public double PartialSeconds { get; set; } = 0.5;

    public double EndpointSeconds
    {
        get => endpointSeconds;
        set
        {
            endpointSeconds = value;
            vad.EndFrames = VoiceActivity.Frames(value);
        }
    }

    double endpointSeconds;

    public double MaxWindowSeconds { get; set; } = 8.0;

    public double ReceivedSeconds => received / (double)Wav.SampleRate;

    public double BufferedSeconds => count / (double)Wav.SampleRate;

    public double EncodedSeconds => encoded / (double)Wav.SampleRate;

    /// <summary>True from the moment speech is heard (or <see cref="Hold"/>) until its words are committed.</summary>
    public bool HasSpeech => hadSpeech || held;

    public bool InSpeech => vad.InSpeech || held;

    public bool Running => running != SpeechPassKind.None;

    public SpeechPassKind RunningKind => running;

    public ReadOnlySpan<byte> PendingText => new(pendingText, 0, pendingUtf8);

    public int PendingWordCount => pendingCount;

    public ReadOnlySpan<WordSlice> PendingWords => new(pendingWords, 0, pendingCount);

    public double OriginSeconds => origin / (double)Wav.SampleRate;

    /// <summary>
    /// Adds audio. False, and nothing added, when the window has no room (Update has not been called for too long).
    /// Runs voice activity over it at once, so a pause is noticed even before the next pass.
    /// </summary>
    public bool Write(ReadOnlySpan<float> chunk)
    {
        if (chunk.Length == 0) return true;
        if (count + chunk.Length > pcm.Length) return false;
        chunk.CopyTo(pcm.AsSpan(count));
        for (var i = 0; i < chunk.Length; i++)
        {
            var edge = vad.Push(chunk[i]);
            if (edge == 0) continue;
            var at = origin + count + i + 1;
            if (edge > 0)
            {
                hadSpeech = true;
                speechStart = at - VoiceActivity.FrameSamples * VoiceActivity.StartFrames;
            }
            else
            {
                lastSpeechEnd = at - (long)VoiceActivity.FrameSamples * vad.EndFrames;
                if (!held) endpoint = Math.Min(at, lastSpeechEnd + (long)(TailSeconds * Wav.SampleRate));
            }
        }
        count += chunk.Length;
        received += chunk.Length;
        return true;
    }

    /// <summary>Push-to-talk: from now until <see cref="Release"/> everything is speech and pauses do not end the phrase.</summary>
    public void Hold()
    {
        held = true;
        hadSpeech = true;
        endpoint = -1;
        speechStart = origin + count;
    }

    /// <summary>Push-to-talk: the phrase ends here.</summary>
    public void Release()
    {
        if (!held) return;
        held = false;
        endpoint = origin + count;
    }

    /// <summary>Samples <see cref="Write"/> can take now.</summary>
    public int Room => pcm.Length - count;

    /// <summary>Ends the phrase here, as a pause would: the next pass commits everything heard so far.</summary>
    public void EndPhrase()
    {
        held = false;
        if (count > 0 && (hadSpeech || pendingCount > 0)) endpoint = origin + count;
    }

    /// <summary>Forgets the window and anything pending, keeping the detected language.</summary>
    public void Clear()
    {
        if (running != SpeechPassKind.None) running = SpeechPassKind.None;
        origin += count;
        count = 0;
        pendingUtf8 = pendingCount = 0;
        hadSpeech = held = false;
        endpoint = speechStart = lastSpeechEnd = -1;
        lastPassAt = 0;
        vad.InSpeech = false;
    }

    /// <summary>Which pass the window needs now, if any. Drops silence before speech down to the pre-roll.</summary>
    public SpeechPassKind Due(bool final)
    {
        if (final) return count > 0 || pendingCount > 0 ? SpeechPassKind.Final : SpeechPassKind.None;
        if (endpoint >= 0) return SpeechPassKind.Phrase;
        if (!hadSpeech && !held)
        {
            var keep = (int)(PreRollSeconds * Wav.SampleRate);
            if (count > keep) Drop(count - keep);
            return SpeechPassKind.None;
        }
        if (count - lastPassAt < Math.Max(1, PartialSeconds * Wav.SampleRate)) return SpeechPassKind.None;
        return count > MaxWindowSeconds * Wav.SampleRate ? SpeechPassKind.Window : SpeechPassKind.Partial;
    }

    /// <summary>Starts a pass. True when it needs steps; false when it already ended.</summary>
    public bool Begin(SpeechPassKind kind)
    {
        running = kind;
        passStarted = System.Diagnostics.Stopwatch.GetTimestamp();
        passLength = kind == SpeechPassKind.Phrase ? (int)Math.Max(0, Math.Min(count, endpoint - origin)) : count;
        encoded += passLength;
        var window = new ReadOnlySpan<float>(pcm, 0, passLength);
        passDirect = whistle.BeamWidth != 1;
        if (passDirect) return false;
        return whistle.BeginSpan(window, language ?? detected, keywords);
    }

    /// <summary>Runs one step of the pass. True when it has ended.</summary>
    public bool Step() => passDirect || whistle.StepSpan();

    /// <summary>
    /// Ends the pass: commits what it should, drops the committed audio and keeps the rest pending. The committed
    /// words of this pass and every pending word are copied out.
    /// </summary>
    public StreamDelta Finish(Span<byte> committedText, Span<WordSlice> committedWords, Span<byte> pendingOut, Span<WordSlice> pendingWordsOut)
    {
        var kind = running;
        running = SpeechPassKind.None;
        var shift = origin / (double)Wav.SampleRate;
        var speech = passDirect
            ? whistle.TranscribeUtf8(new ReadOnlySpan<float>(pcm, 0, passLength), text, words, language ?? detected, keywords)
            : whistle.FinishSpan(text, words, 0);
        detected ??= speech.Language;
        var truncated = speech.Truncated;
        var commit = kind is SpeechPassKind.Phrase or SpeechPassKind.Final ? speech.WordCount : 0;
        var cutSamples = passLength;
        if (kind == SpeechPassKind.Window)
        {
            // Commit the words that end well before the window does; the last few seconds stay pending.
            var limit = passLength / (double)Wav.SampleRate - KeepSeconds;
            while (commit < speech.WordCount && words[commit].End <= limit) commit++;
            if (commit == 0 && speech.WordCount > 1) commit = speech.WordCount - 1;
            // Cut halfway through the gap after the last committed word; with nothing to commit, drop what comes
            // before the first word (or all but the last seconds of a window with no words), so the window stays bounded.
            cutSamples = commit > 0
                ? commit < speech.WordCount
                    ? (int)Math.Min(passLength, (words[commit - 1].End + words[commit].Start) / 2 * Wav.SampleRate)
                    : (int)Math.Min(passLength, (words[commit - 1].End + 0.2) * Wav.SampleRate)
                : (int)Math.Max(0, Math.Min(passLength, (speech.WordCount > 0 ? words[0].Start - 0.1 : limit) * Wav.SampleRate));
        }
        truncated |= CopyRange(text, words, 0, commit, shift, committedText, committedWords, out var committedBytes, out var committedCount);
        var passStart = shift;
        var passSeconds = passLength / (double)Wav.SampleRate;
        if (kind == SpeechPassKind.Partial)
        {
            lastPassAt = passLength;
            StorePending(speech.WordCount, shift);
        }
        else if (kind == SpeechPassKind.Window)
        {
            StorePending(speech.WordCount, shift, commit);
            Drop(cutSamples);
            lastPassAt = count;
        }
        else
        {
            pendingUtf8 = pendingCount = 0;
            Drop(passLength);
            lastPassAt = 0;
            endpoint = -1;
            hadSpeech = held || vad.InSpeech || speechStart >= origin;
            if (kind == SpeechPassKind.Final)
            {
                hadSpeech = held;
                vad.InSpeech = false;
            }
        }
        truncated |= CopyPending(pendingOut, pendingWordsOut, out var pendingBytes, out var pendingWordCount);
        var ms = (System.Diagnostics.Stopwatch.GetTimestamp() - passStarted) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        return new StreamDelta(committedBytes, committedCount, pendingBytes, pendingWordCount, ReceivedSeconds, ms, truncated,
            passStart, passSeconds, kind is SpeechPassKind.Phrase or SpeechPassKind.Final);
    }

    /// <summary>The synchronous path: runs the pass that is due, if any, to its end.</summary>
    public StreamDelta RunToEnd(bool final, Span<byte> committedText, Span<WordSlice> committedWords, Span<byte> pendingOut, Span<WordSlice> pendingWordsOut)
    {
        var kind = Due(final);
        if (kind == SpeechPassKind.None)
        {
            var truncated = CopyPending(pendingOut, pendingWordsOut, out var bytes, out var wordCount);
            return new StreamDelta(0, 0, bytes, wordCount, ReceivedSeconds, 0, truncated);
        }
        if (Begin(kind))
            while (!Step()) { }
        // A final stop commits everything in one pass, including a phrase whose pause was not yet taken.
        return Finish(committedText, committedWords, pendingOut, pendingWordsOut);
    }

    void Drop(int samples)
    {
        if (samples <= 0) return;
        samples = Math.Min(samples, count);
        Array.Copy(pcm, samples, pcm, 0, count - samples);
        count -= samples;
        origin += samples;
        lastPassAt = Math.Max(0, lastPassAt - samples);
    }

    void StorePending(int wordCount, double shift, int from = 0)
    {
        pendingUtf8 = 0;
        pendingCount = 0;
        for (var i = from; i < wordCount; i++)
        {
            var w = words[i];
            if (pendingCount >= pendingWords.Length || pendingUtf8 + 1 + w.Utf8Length > pendingText.Length) break;
            if (pendingCount > 0) pendingText[pendingUtf8++] = (byte)' ';
            new ReadOnlySpan<byte>(text, w.Utf8Offset, w.Utf8Length).CopyTo(pendingText.AsSpan(pendingUtf8));
            pendingWords[pendingCount++] = new WordSlice(pendingUtf8, w.Utf8Length, w.Start + shift, w.End + shift, w.Probability);
            pendingUtf8 += w.Utf8Length;
        }
    }

    bool CopyPending(Span<byte> destText, Span<WordSlice> destWords, out int utf8Length, out int wordCount)
    {
        utf8Length = 0;
        wordCount = 0;
        var truncated = false;
        for (var i = 0; i < pendingCount; i++)
        {
            var w = pendingWords[i];
            if (wordCount >= destWords.Length) { truncated = true; break; }
            if (utf8Length + (wordCount > 0 ? 1 : 0) + w.Utf8Length > destText.Length) { truncated = true; break; }
            if (wordCount > 0) destText[utf8Length++] = (byte)' ';
            new ReadOnlySpan<byte>(pendingText, w.Utf8Offset, w.Utf8Length).CopyTo(destText.Slice(utf8Length));
            destWords[wordCount++] = new WordSlice(utf8Length, w.Utf8Length, w.Start, w.End, w.Probability);
            utf8Length += w.Utf8Length;
        }
        return truncated;
    }

    static bool CopyRange(byte[] source, WordSlice[] slices, int from, int length, double shift, Span<byte> destText, Span<WordSlice> destWords, out int utf8Length, out int wordCount)
    {
        utf8Length = 0;
        wordCount = 0;
        var truncated = false;
        for (var i = 0; i < length; i++)
        {
            var w = slices[from + i];
            if (wordCount >= destWords.Length) { truncated = true; break; }
            if (utf8Length + (wordCount > 0 ? 1 : 0) + w.Utf8Length > destText.Length) { truncated = true; break; }
            if (wordCount > 0) destText[utf8Length++] = (byte)' ';
            new ReadOnlySpan<byte>(source, w.Utf8Offset, w.Utf8Length).CopyTo(destText.Slice(utf8Length));
            destWords[wordCount++] = new WordSlice(utf8Length, w.Utf8Length, w.Start + shift, w.End + shift, w.Probability);
            utf8Length += w.Utf8Length;
        }
        return truncated;
    }

    internal static bool FoldedEqual(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        var i = 0;
        var j = 0;
        while (i < a.Length && j < b.Length)
        {
            var x = FoldStep(a, ref i);
            var y = FoldStep(b, ref j);
            if (x != y) return false;
            if (x < 0) break;
        }
        while (i < a.Length && FoldStep(a, ref i) < 0) { }
        while (j < b.Length && FoldStep(b, ref j) < 0) { }
        return i >= a.Length && j >= b.Length;
    }

    static int FoldStep(ReadOnlySpan<byte> s, ref int i)
    {
        while (i < s.Length)
        {
            var c = s[i++];
            if (c >= (byte)'A' && c <= (byte)'Z') return c + 32;
            if ((c >= (byte)'a' && c <= (byte)'z') || (c >= (byte)'0' && c <= (byte)'9')) return c;
            if (c < 0x80) continue;
            return c;
        }
        return -2;
    }
}
