using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace Needle;

// Push-to-talk capture: the game audio thread appends PCM lock-free while the key is
// held; the game thread transcribes on release, or previews incrementally mid-hold.
// Single writer (audio), single reader (game). Transcription itself reuses Whistle's
// zero-allocation span path, so steady-state capture plus release allocates nothing.
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

    // Audio thread: appends a chunk. Returns false when the buffer is full (chunk dropped).
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

    // Game thread: transcribes whatever was captured and clears the buffer.
    public SpeechSpan Release(Whistle whistle, Span<byte> utf8, Span<WordSlice> words, string? language = null, IReadOnlyList<string>? keywords = null)
    {
        var speech = Preview(whistle, utf8, words, language, keywords);
        Volatile.Write(ref count, 0);
        return speech;
    }
}

// Incremental game-loop streaming over the zero-allocation span path. The caller pushes
// audio with Write, then calls Update once per frame: newly firmed words arrive in the
// committed spans (delta since the last Update), the unstable tail in the pending spans.
public readonly struct StreamDelta
{
    public readonly int CommittedUtf8;
    public readonly int CommittedWords;
    public readonly int PendingUtf8;
    public readonly int PendingWords;
    public readonly double ReceivedSeconds;
    public readonly double PassMs;
    public readonly bool Truncated;

    public StreamDelta(int committedUtf8, int committedWords, int pendingUtf8, int pendingWords, double receivedSeconds, double passMs, bool truncated)
    {
        CommittedUtf8 = committedUtf8;
        CommittedWords = committedWords;
        PendingUtf8 = pendingUtf8;
        PendingWords = pendingWords;
        ReceivedSeconds = receivedSeconds;
        PassMs = passMs;
        Truncated = truncated;
    }
}

public sealed class VoiceStream
{
    readonly Whistle whistle;
    readonly string? language;
    readonly IReadOnlyList<string>? keywords;
    readonly float[] pcm;
    readonly byte[] text;
    readonly WordSlice[] words;
    readonly byte[] previous;
    readonly int[] previousBounds;
    int previousCount;
    int count;
    double received;
    double bufferStart;
    string? detected;

    public VoiceStream(Whistle whistle, string? language = null, IReadOnlyList<string>? keywords = null, int maxSeconds = Whistle.MaxSeconds)
    {
        if (maxSeconds < 1 || maxSeconds > Whistle.MaxSeconds) throw new ArgumentOutOfRangeException(nameof(maxSeconds));
        this.whistle = whistle;
        this.language = language;
        this.keywords = keywords;
        pcm = new float[maxSeconds * Wav.SampleRate];
        var cap = whistle.MaxTranscriptTokens * 64;
        text = new byte[cap];
        words = new WordSlice[whistle.MaxTranscriptTokens + 1];
        previous = new byte[cap];
        previousBounds = new int[(whistle.MaxTranscriptTokens + 1) * 2];
    }

    public double ReceivedSeconds => received;

    public double BufferedSeconds => count / (double)Wav.SampleRate;

    // Returns false when the buffer is full (chunk dropped).
    public bool Write(ReadOnlySpan<float> chunk)
    {
        if (chunk.Length == 0) return true;
        if (count + chunk.Length > pcm.Length) return false;
        chunk.CopyTo(pcm.AsSpan(count));
        count += chunk.Length;
        received += chunk.Length / (double)Wav.SampleRate;
        return true;
    }

    public StreamDelta Update(Span<byte> committedText, Span<WordSlice> committedWords, Span<byte> pendingText, Span<WordSlice> pendingWords) =>
        Pass(committedText, committedWords, pendingText, pendingWords, final: false);

    public StreamDelta Stop(Span<byte> committedText, Span<WordSlice> committedWords, Span<byte> pendingText, Span<WordSlice> pendingWords) =>
        Pass(committedText, committedWords, pendingText, pendingWords, final: true);

    StreamDelta Pass(Span<byte> committedText, Span<WordSlice> committedWords, Span<byte> pendingText, Span<WordSlice> pendingWords, bool final)
    {
        var started = Stopwatch.GetTimestamp();
        var window = Math.Min(count, Whistle.MaxSeconds * Wav.SampleRate);
        var speech = whistle.TranscribeUtf8(pcm.AsSpan(count - window, window), text, words, language ?? detected, keywords);
        detected ??= speech.Language;
        var shift = bufferStart + (count - window) / (double)Wav.SampleRate;
        var agreed = 0;
        if (final) agreed = speech.WordCount;
        else while (agreed < speech.WordCount && agreed < previousCount && WordAgrees(agreed)) agreed++;
        var truncated = speech.Truncated;
        truncated |= CopyRange(text, words, 0, agreed, shift, committedText, committedWords, out var committedBytes, out var committedCount);
        truncated |= CopyRange(text, words, agreed, speech.WordCount - agreed, shift, pendingText, pendingWords, out var pendingBytes, out var pendingCount);
        if (!final && agreed > 0)
        {
            var cutSeconds = agreed < speech.WordCount
                ? (words[agreed - 1].End + words[agreed].Start) / 2 + shift
                : words[agreed - 1].End + shift;
            var drop = (int)Math.Clamp((cutSeconds - bufferStart) * Wav.SampleRate, 0, count);
            Array.Copy(pcm, drop, pcm, 0, count - drop);
            count -= drop;
            bufferStart += drop / (double)Wav.SampleRate;
            StorePrevious(agreed, speech.WordCount - agreed);
        }
        else StorePrevious(0, final ? 0 : speech.WordCount);
        var ms = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
        return new StreamDelta(committedBytes, committedCount, pendingBytes, pendingCount, received, ms, truncated);
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

    bool WordAgrees(int index)
    {
        var w = words[index];
        var a = new ReadOnlySpan<byte>(text, w.Utf8Offset, w.Utf8Length);
        var b = new ReadOnlySpan<byte>(previous, previousBounds[index * 2], previousBounds[index * 2 + 1]);
        return FoldedEqual(a, b);
    }

    void StorePrevious(int from, int length)
    {
        var at = 0;
        for (var i = 0; i < length; i++)
        {
            var w = words[from + i];
            new ReadOnlySpan<byte>(text, w.Utf8Offset, w.Utf8Length).CopyTo(previous.AsSpan(at));
            previousBounds[i * 2] = at;
            previousBounds[i * 2 + 1] = w.Utf8Length;
            at += w.Utf8Length;
        }
        previousCount = length;
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

    // Next ASCII alphanumeric folded to lowercase, -1 for anything skipped, -2 at end.
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
