using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Needle;

// Blittable word entry: byte offsets into the caller's UTF-8 transcript buffer plus times.
public readonly struct WordSlice
{
    public readonly int Utf8Offset;
    public readonly int Utf8Length;
    public readonly double Start;
    public readonly double End;
    public readonly double Probability;

    public WordSlice(int utf8Offset, int utf8Length, double start, double end, double probability)
    {
        Utf8Offset = utf8Offset;
        Utf8Length = utf8Length;
        Start = start;
        End = end;
        Probability = probability;
    }
}

// Zero-allocation transcription result. Language references the cached language table;
// no managed strings are created on the steady-state path.
public readonly struct SpeechSpan
{
    public readonly int Utf8Length;
    public readonly int WordCount;
    public readonly string? Language;
    public readonly double TimeToFirstTokenMs;
    public readonly double DecodeTps;
    public readonly double EncodeMs;
    public readonly bool Truncated;

    public SpeechSpan(int utf8Length, int wordCount, string? language, double ttftMs, double tps, double encodeMs, bool truncated)
    {
        Utf8Length = utf8Length;
        WordCount = wordCount;
        Language = language;
        TimeToFirstTokenMs = ttftMs;
        DecodeTps = tps;
        EncodeMs = encodeMs;
        Truncated = truncated;
    }
}

public sealed unsafe partial class Whistle
{
    // Greedy-only: the span path never runs beam search. Callers that need BeamWidth > 1
    // should use Transcribe; TranscribeUtf8 with BeamWidth != 1 falls back to it and copies out.
    const int SpanCurrentChars = 512;

    int[]? spanTokens;
    float[]? spanProbs;
    float[]? alignA;
    float[]? alignB;
    float[]? alignRow;
    int[]? alignStarts;
    byte[]? longScratch;
    WordSlice[]? longWords;
    byte[]? wordScratch;
    char[]? spanCurrentChars;
    IReadOnlyList<string>? spanKeywordSource = new List<string>();
    string[] spanKeywordStrings = Array.Empty<string>();

    // Maximum words one span transcription can report. Size caller word buffers to this
    // and the UTF-8 buffer to at least 64 bytes per word; check SpeechSpan.Truncated.
    public int MaxTranscriptTokens => g.MaxSeq;

    void EnsureSpanBuffers()
    {
        if (spanTokens != null) return;
        spanTokens = new int[g.MaxSeq];
        spanProbs = new float[g.MaxSeq];
        alignA = new float[(g.MaxSeq + 1) * maxFrames];
        alignB = new float[(g.MaxSeq + 1) * maxFrames];
        alignRow = new float[maxFrames];
        alignStarts = new int[g.MaxSeq + 1];
        longScratch = new byte[g.MaxSeq * 64];
        longWords = new WordSlice[g.MaxSeq + 1];
        spanCurrentChars = new char[SpanCurrentChars];
    }

    void CacheKeywords(IReadOnlyList<string>? keywords)
    {
        if (ReferenceEquals(keywords, spanKeywordSource)) return;
        spanKeywordSource = keywords;
        if (keywords == null || keywords.Count == 0)
        {
            spanKeywordStrings = Array.Empty<string>();
            return;
        }
        var trimmed = new string[keywords.Count];
        for (var i = 0; i < keywords.Count; i++) trimmed[i] = keywords[i].Trim();
        spanKeywordStrings = trimmed;
    }

    public SpeechSpan TranscribeUtf8(ReadOnlySpan<float> pcm, Span<byte> utf8, Span<WordSlice> words, string? language = null, IReadOnlyList<string>? keywords = null)
    {
        if (language != null && Array.IndexOf(languages, language) < 0) throw new ArgumentException($"language is one of {string.Join(" ", languages)}");
        if (BeamWidth != 1)
        {
            var fallback = Transcribe(pcm, language, keywords, true);
            return CopyFallback(fallback, utf8, words);
        }
        var maxSamples = MaxSeconds * audio->SampleRate;
        if (pcm.Length <= maxSamples) return TranscribeShortUtf8(pcm, utf8, words, language, keywords);
        return TranscribeLongUtf8(pcm, utf8, words, language, keywords);
    }

    static SpeechSpan CopyFallback(Transcription fallback, Span<byte> utf8, Span<WordSlice> words)
    {
        var bytes = Encoding.UTF8.GetBytes(fallback.Text);
        var truncated = bytes.Length > utf8.Length;
        var count = Math.Min(bytes.Length, utf8.Length);
        bytes.AsSpan(0, count).CopyTo(utf8);
        var kept = 0;
        var offset = 0;
        if (fallback.Words != null)
        {
            foreach (var w in fallback.Words)
            {
                var start = fallback.Text.IndexOf(w.Text, offset, StringComparison.Ordinal);
                if (start < 0) break;
                offset = start + w.Text.Length;
                if (kept >= words.Length) { truncated = true; break; }
                var byteStart = Encoding.UTF8.GetByteCount(fallback.Text, 0, start);
                var byteLength = Encoding.UTF8.GetByteCount(w.Text);
                if (byteStart + byteLength > count) { truncated = true; break; }
                words[kept++] = new WordSlice(byteStart, byteLength, w.Start, w.End, w.Probability);
            }
        }
        return new SpeechSpan(count, kept, fallback.Text.Length > 0 ? fallback.Language : null,
            fallback.TimeToFirstTokenMs, fallback.DecodeTps, fallback.EncodeMs, truncated);
    }

    SpeechSpan TranscribeShortUtf8(ReadOnlySpan<float> pcm, Span<byte> utf8, Span<WordSlice> words, string? language, IReadOnlyList<string>? keywords)
    {
        EnsureDecoderBuffers();
        EnsureSpanBuffers();
        CacheKeywords(keywords);
        var tokens = DecodeSpan(pcm, language, out var detected, out var ttftMs, out var tps, out var encodeMs, out var empty);
        if (empty) return new SpeechSpan(0, 0, null, 0, 0, 0, false);
        return RenderSpan(tokens, detected, utf8, words, 0, ttftMs, tps, encodeMs);
    }

    SpeechSpan TranscribeLongUtf8(ReadOnlySpan<float> pcm, Span<byte> utf8, Span<WordSlice> words, string? language, IReadOnlyList<string>? keywords)
    {
        EnsureDecoderBuffers();
        EnsureSpanBuffers();
        CacheKeywords(keywords);
        var rate = audio->SampleRate;
        var window = MaxSeconds * rate;
        var scratch = longScratch!;
        var scratchWords = longWords!;
        var totalUtf8 = 0;
        var totalWords = 0;
        var truncated = false;
        string? detected = language;
        string? firstLanguage = null;
        double ttftMs = 0, tps = 0, encodeMs = 0;
        var offset = 0;
        var first = true;
        while (offset < pcm.Length)
        {
            var length = Math.Min(window, pcm.Length - offset);
            var tokens = DecodeSpan(pcm.Slice(offset, length), detected, out var windowLanguage, out var windowTtft, out var windowTps, out var windowEncode, out _);
            if (first) { firstLanguage = windowLanguage; ttftMs = windowTtft; tps = windowTps; encodeMs = windowEncode; first = false; }
            detected ??= windowLanguage;
            var rendered = RenderSpan(tokens, windowLanguage, scratch, scratchWords, 0, 0, 0, 0);
            if (rendered.Truncated) truncated = true;
            var base_ = offset / (double)rate;
            var last = offset + length >= pcm.Length;
            var available = rendered.WordCount;
            var cut = available;
            if (!last)
            {
                cut = -1;
                for (var i = available - 1; i >= 0; i--)
                    if (WordEndsSentence(scratch, scratchWords[i])) { cut = i + 1; break; }
                if (cut <= 0) cut = available;
            }
            for (var i = 0; i < cut; i++)
            {
                var w = scratchWords[i];
                if (totalWords >= words.Length) { truncated = true; break; }
                if (totalUtf8 + w.Utf8Length + (totalWords > 0 ? 1 : 0) > utf8.Length) { truncated = true; break; }
                if (totalWords > 0) utf8[totalUtf8++] = (byte)' ';
                new Span<byte>(scratch, w.Utf8Offset, w.Utf8Length).CopyTo(utf8.Slice(totalUtf8));
                words[totalWords++] = new WordSlice(totalUtf8, w.Utf8Length, w.Start + base_, w.End + base_, w.Probability);
                totalUtf8 += w.Utf8Length;
            }
            if (last) break;
            var advance = cut > 0 && cut < available
                ? Math.Max(1.0, scratchWords[cut - 1].End) * rate
                : length;
            offset += (int)Math.Min(length, advance);
        }
        return new SpeechSpan(totalUtf8, totalWords, totalUtf8 > 0 ? detected ?? firstLanguage : null, ttftMs, tps, encodeMs, truncated);
    }

    static bool WordEndsSentence(byte[] scratch, WordSlice w)
    {
        if (w.Utf8Length == 0) return false;
        var last = scratch[w.Utf8Offset + w.Utf8Length - 1];
        if (last is (byte)'.' or (byte)'?' or (byte)'!') return true;
        if (w.Utf8Length >= 3)
        {
            var a = scratch[w.Utf8Offset + w.Utf8Length - 3];
            var b = scratch[w.Utf8Offset + w.Utf8Length - 2];
            // 。(E3 80 82) ？(EF BC 9F) ！(EF BC 81)
            if (a == 0xE3 && b == 0x80 && last == 0x82) return true;
            if (a == 0xEF && b == 0xBC && (last == 0x9F || last == 0x81)) return true;
        }
        return false;
    }

    // Decodes greedily into the reusable token buffer. Returns the token count;
    // empty is set when the input is silence or too short to encode.
    int DecodeSpan(ReadOnlySpan<float> pcm, string? language, out string? detected, out double ttftMs, out double tps, out double encodeMs, out bool empty)
    {
        detected = null;
        ttftMs = 0;
        tps = 0;
        encodeMs = 0;
        empty = true;
        if (Rms(pcm) < SilenceRms) return 0;
        var started = Stopwatch.GetTimestamp();
        if (Encode(pcm) == 0) return 0;
        BuildCrossMemory();
        encodeMs = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
        detected = PrimeSpan(language);
        var tokens = spanTokens!;
        var probs = spanProbs!;
        var count = 0;
        var one = stackalloc int[1];
        var firstToken = Stopwatch.GetTimestamp();
        var decodeStart = firstToken;
        for (var step = 0; step < g.MaxSeq - 3; step++)
        {
            var logits = decoder.Logits;
            for (var i = 0; i < textVocab; i++) scores[i] = logits[i];
            scores[0] = scores[2] = scores[3] = -1e30f;
            ApplyKeywordBiasSpan(new ReadOnlySpan<int>(tokens, 0, count), scores);
            var best = 0;
            for (var i = 1; i < textVocab; i++) if (scores[i] > scores[best]) best = i;
            if (step == 0) ttftMs = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
            Ops.Softmax(scores, textVocab);
            probs[count] = scores[best];
            if (best == Eos)
            {
                one[0] = Eos;
                decoder.Feed(new ReadOnlySpan<int>(one, 1));
                break;
            }
            tokens[count++] = best;
            one[0] = best;
            decoder.Feed(new ReadOnlySpan<int>(one, 1));
        }
        var seconds = Math.Max(1e-9, (Stopwatch.GetTimestamp() - decodeStart) / (double)Stopwatch.Frequency);
        tps = count / seconds;
        empty = false;
        return count;
    }

    string PrimeSpan(string? language)
    {
        decoder.Reset();
        decoder.Alignment = null;
        Mem.Clear(alignment, (nuint)g.MaxSeq * (nuint)maxFrames * 4);
        decoder.Alignment = alignment;
        decoder.AlignmentFromLayer = g.Layers / 2;
        var one = stackalloc int[1];
        one[0] = Bos;
        decoder.Feed(new ReadOnlySpan<int>(one, 1));
        int tag;
        if (language != null) tag = textVocab + Array.IndexOf(languages, language);
        else
        {
            var logits = decoder.Logits;
            tag = textVocab;
            for (var i = 1; i < languages.Length; i++) if (logits[textVocab + i] > logits[tag]) tag = textVocab + i;
        }
        one[0] = tag;
        decoder.Feed(new ReadOnlySpan<int>(one, 1));
        return languages[tag - textVocab];
    }

    // String-semantics keyword bias identical to ApplyKeywordBias, but the current word
    // is rebuilt into a reusable char buffer instead of a StringBuilder, so the
    // steady-state path allocates nothing. Skips bias for words over the scratch.
    void ApplyKeywordBiasSpan(ReadOnlySpan<int> generated, float* target)
    {
        var keys = spanKeywordStrings;
        if (keys.Length == 0) return;
        var pieces = Pieces();
        var chars = spanCurrentChars!;
        var currentLength = WordSoFarChars(generated, pieces, chars);
        if (currentLength < 0) return;
        var current = new ReadOnlySpan<char>(chars, 0, currentLength);
        for (var k = 0; k < keys.Length; k++)
        {
            var word = keys[k];
            if (word.Length <= currentLength || !word.AsSpan().StartsWith(current, StringComparison.OrdinalIgnoreCase)) continue;
            var remaining = word.AsSpan(currentLength);
            var bias = currentLength == 0 ? FirstKeywordBias : ContinuedKeywordBias;
            for (var id = 4; id < textVocab; id++)
            {
                var piece = pieces[id];
                if (piece.Length == 0) continue;
                var startsWord = piece[0] == '\u2581';
                if (startsWord != (currentLength == 0)) continue;
                var text = startsWord ? piece.AsSpan(1) : piece.AsSpan();
                if (text.Length == 0 || (currentLength == 0 && text.Length < 2 && remaining.Length > 1) || text.Length > remaining.Length) continue;
                if (!remaining.StartsWith(text, StringComparison.OrdinalIgnoreCase)) continue;
                var exact = remaining.StartsWith(text, StringComparison.Ordinal);
                target[id] += bias * (exact ? 1f : 0.5f) * (text.Length == remaining.Length ? 1.3f : 1f);
            }
        }
    }

    // Rebuilds the current word's chars from generated tokens. Returns -1 when the word
    // overflows the scratch, in which case keyword bias is skipped for this step.
    int WordSoFarChars(ReadOnlySpan<int> generated, string[] pieces, Span<char> scratch)
    {
        var start = generated.Length;
        for (var i = generated.Length - 1; i >= 0; i--)
        {
            var piece = pieces[generated[i]];
            if (piece.Length > 0 && piece[0] == '\u2581') { start = i; break; }
            start = i;
        }
        var total = 0;
        for (var i = start; i < generated.Length; i++)
        {
            var piece = pieces[generated[i]];
            var skip = i == start && piece.Length > 0 && piece[0] == '\u2581' ? 1 : 0;
            total += piece.Length - skip;
            if (total > scratch.Length) return -1;
        }
        var at = 0;
        for (var i = start; i < generated.Length; i++)
        {
            var piece = pieces[generated[i]];
            var skip = i == start && piece.Length > 0 && piece[0] == '\u2581' ? 1 : 0;
            piece.AsSpan(skip).CopyTo(scratch.Slice(at));
            at += piece.Length - skip;
        }
        return total;
    }

    bool IsWordStart(int id)
    {
        var piece = tokenizer.PiecePointer(id);
        return Tokenizer.PieceStartsWord(piece, tokenizer.PieceByteLength(id));
    }

    SpeechSpan RenderSpan(int tokens, string? detected, Span<byte> utf8, Span<WordSlice> words, double shift, double ttftMs, double tps, double encodeMs)
    {
        var tokenArray = spanTokens!;
        var probs = spanProbs!;
        var starts = alignStarts!;
        var scratch = wordScratch ??= new byte[1024];
        var frameStarts = words.Length > 0 && tokens > 0 ? TokenFramesSpan(tokens) : null;
        var utf8Length = 0;
        var wordCount = 0;
        var truncated = false;
        var w = 0;
        while (w < tokens)
        {
            var from = w;
            var to = w + 1;
            while (to < tokens && !IsWordStart(tokenArray[to])) to++;
            // Render the word into scratch, trimming padding like BuildText does.
            var at = 0;
            var overflow = false;
            for (var i = from; i < to; i++)
            {
                // Append emits at most the raw piece bytes, so this reserves enough.
                if (at + tokenizer.PieceByteLength(tokenArray[i]) > scratch.Length) { overflow = true; break; }
                at += tokenizer.Append(tokenArray[i], scratch.AsSpan(at), i == from);
            }
            if (overflow) { truncated = true; break; }
            var start = 0;
            while (start < at && (scratch[start] == (byte)' ' || scratch[start] == (byte)'\t' || scratch[start] == (byte)'\n' || scratch[start] == (byte)'\r')) start++;
            while (at > start && (scratch[at - 1] == (byte)' ' || scratch[at - 1] == (byte)'\t' || scratch[at - 1] == (byte)'\n' || scratch[at - 1] == (byte)'\r')) at--;
            if (at == start) { w = to; continue; }
            var length = at - start;
            if (wordCount >= words.Length) { truncated = true; break; }
            if (utf8Length + (wordCount > 0 ? 1 : 0) + length > utf8.Length) { truncated = true; break; }
            if (wordCount > 0) utf8[utf8Length++] = (byte)' ';
            new Span<byte>(scratch, start, length).CopyTo(utf8.Slice(utf8Length));
            var probability = 0.0;
            for (var i = from; i < to; i++) probability += probs[i];
            probability /= to - from;
            var wordStart = frameStarts != null ? starts[from] * FrameSeconds : 0;
            var wordEnd = frameStarts != null ? starts[to] * FrameSeconds : 0;
            words[wordCount++] = new WordSlice(utf8Length, length, wordStart + shift, wordEnd + shift, probability);
            utf8Length += length;
            w = to;
        }
        return new SpeechSpan(utf8Length, wordCount, utf8Length > 0 ? detected : null, ttftMs, tps, encodeMs, truncated);
    }

    // Allocation-free port of TokenFrames into the reusable align buffers.
    int[] TokenFramesSpan(int tokens)
    {
        var rows = tokens + 1;
        var columns = frames;
        var cost = alignA!;
        var total = alignB!;
        var filtered = alignRow!;
        var starts = alignStarts!;
        for (var j = 0; j < columns; j++)
        {
            double mean = 0;
            for (var i = 0; i < rows; i++) mean += alignment[(1 + i) * frames + j];
            mean /= rows;
            double squares = 0;
            for (var i = 0; i < rows; i++) { var d = alignment[(1 + i) * frames + j] - mean; squares += d * d; }
            var std = Math.Sqrt(squares / rows) + 1e-9;
            for (var i = 0; i < rows; i++) total[i * columns + j] = (float)((alignment[(1 + i) * frames + j] - mean) / std);
        }
        Span<float> window = stackalloc float[7];
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < columns; j++)
            {
                var n = 0;
                for (var k = -3; k <= 3; k++) window[n++] = total[i * columns + Math.Clamp(j + k, 0, columns - 1)];
                for (var a2 = 1; a2 < 7; a2++)
                {
                    var v = window[a2];
                    var b2 = a2 - 1;
                    while (b2 >= 0 && window[b2] > v) { window[b2 + 1] = window[b2]; b2--; }
                    window[b2 + 1] = v;
                }
                filtered[j] = window[3];
            }
            for (var j = 0; j < columns; j++) cost[i * columns + j] = -filtered[j];
        }
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < columns; j++)
            {
                float best;
                if (i == 0) best = j > 0 ? Math.Min(0f, total[j - 1]) : 0f;
                else
                {
                    best = total[(i - 1) * columns + j];
                    if (j > 0) best = Math.Min(best, Math.Min(total[(i - 1) * columns + j - 1], total[i * columns + j - 1]));
                }
                total[i * columns + j] = cost[i * columns + j] + best;
            }
        for (var i = 0; i < rows; i++) starts[i] = int.MaxValue;
        var r = rows - 1;
        var c = 0;
        for (var j = 1; j < columns; j++) if (total[r * columns + j] < total[r * columns + c]) c = j;
        for (;;)
        {
            starts[r] = Math.Min(starts[r], c);
            if (r == 0)
            {
                if (c > 0 && total[c - 1] < 0f) { c--; continue; }
                break;
            }
            var diagonal = c > 0 ? total[(r - 1) * columns + c - 1] : float.PositiveInfinity;
            var up = total[(r - 1) * columns + c];
            var left = c > 0 ? total[r * columns + c - 1] : float.PositiveInfinity;
            if (diagonal <= up && diagonal <= left) { r--; c--; }
            else if (up <= left) r--;
            else c--;
        }
        for (var i = rows - 2; i >= 0; i--) if (starts[i] == int.MaxValue) starts[i] = starts[i + 1];
        for (var i = 1; i < rows; i++) if (starts[i] == int.MaxValue) starts[i] = starts[i - 1];
        for (var i = 1; i < rows; i++) starts[i] = Math.Max(starts[i], starts[i - 1]);
        return starts;
    }
}
