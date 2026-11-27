using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace Needle;

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

    bool spanAlign = true;

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
        var maxSamples = MaxSeconds * audio->SampleRate;
        Enter();
        try
        {
            if (pcm.Length <= maxSamples) return TranscribeShortUtf8(pcm, utf8, words, language, keywords);
            return TranscribeLongUtf8(pcm, utf8, words, language, keywords);
        }
        finally { Exit(); }
    }

    SpeechSpan TranscribeShortUtf8(ReadOnlySpan<float> pcm, Span<byte> utf8, Span<WordSlice> words, string? language, IReadOnlyList<string>? keywords)
    {
        spanAlign = words.Length > 0;
        EnsureDecoderBuffers();
        EnsureSpanBuffers();
        CacheKeywords(keywords);
        var tokens = DecodeSpan(pcm, language, out var detected, out var ttftMs, out var tps, out var encodeMs, out var empty);
        if (empty) return new SpeechSpan(0, 0, null, 0, 0, 0, false);
        return RenderSpan(tokens, detected, utf8, words, 0, ttftMs, tps, encodeMs);
    }

    SpeechSpan TranscribeLongUtf8(ReadOnlySpan<float> pcm, Span<byte> utf8, Span<WordSlice> words, string? language, IReadOnlyList<string>? keywords)
    {
        spanAlign = true;
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
        var textOnly = words.Length == 0;
        string? detected = language;
        string? firstLanguage = null;
        double ttftMs = 0, tps = 0, encodeMs = 0;
        var offset = 0;
        var first = true;
        while (offset < pcm.Length)
        {
            if (offset > 0 && CancelPending()) break;
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
                if (!textOnly && totalWords >= words.Length) { truncated = true; break; }
                if (totalUtf8 + w.Utf8Length + (totalUtf8 > 0 ? 1 : 0) > utf8.Length) { truncated = true; break; }
                if (totalUtf8 > 0) utf8[totalUtf8++] = (byte)' ';
                new Span<byte>(scratch, w.Utf8Offset, w.Utf8Length).CopyTo(utf8.Slice(totalUtf8));
                if (!textOnly) words[totalWords++] = new WordSlice(totalUtf8, w.Utf8Length, w.Start + base_, w.End + base_, w.Probability);
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
            if (a == 0xE3 && b == 0x80 && last == 0x82) return true;
            if (a == 0xEF && b == 0xBC && (last == 0x9F || last == 0x81)) return true;
        }
        return false;
    }

    enum SpanStage : byte { Idle, Stem, EncoderStart, EncoderLayer, EncoderFinish, Memory, Prime, Decode, Done }

    // Everything one greedy pass carries from one step to the next. TranscribeUtf8 runs the steps back to back; a
    // speech stream runs them a slice at a time. Same code, same order, so the transcripts are the same bit for bit.
    struct SpanPass
    {
        public SpanStage Stage;
        public int L0;
        public int Layer;
        public int Count;
        public FrameJob Job;
        public long Started;
        public long DecodeStart;
        public double TtftMs;
        public double Tps;
        public double EncodeMs;
        public string? Language;
        public string? Detected;
        public bool Empty;
        public bool HasKeywords;
    }

    SpanPass span;

    int DecodeSpan(ReadOnlySpan<float> pcm, string? language, out string? detected, out double ttftMs, out double tps, out double encodeMs, out bool empty)
    {
        if (BeamWidth > 1) return BeamSpan(pcm, language, out detected, out ttftMs, out tps, out encodeMs, out empty);
        if (BeginSpanPass(pcm, language))
            while (!StepSpanPass()) { }
        detected = span.Detected;
        ttftMs = span.TtftMs;
        tps = span.Tps;
        encodeMs = span.EncodeMs;
        empty = span.Empty;
        return span.Empty ? 0 : span.Count;
    }

    int BeamSpan(ReadOnlySpan<float> pcm, string? language, out string? detected, out double ttftMs, out double tps, out double encodeMs, out bool empty)
    {
        detected = null;
        ttftMs = 0;
        tps = 0;
        encodeMs = 0;
        empty = true;
        if (Silent(pcm)) return 0;
        var started = Stopwatch.GetTimestamp();
        if (EncodeCore(pcm) == 0) return 0;
        BuildCrossMemory();
        encodeMs = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
        detected = PrimeSpan(language);
        var decodeStart = Stopwatch.GetTimestamp();
        var count = BeamSearch(true);
        ttftMs = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
        if (spanAlign) FeedForAlignment(count);
        tps = count / Math.Max(1e-9, (Stopwatch.GetTimestamp() - decodeStart) / (double)Stopwatch.Frequency);
        empty = false;
        return count;
    }

    /// <summary>
    /// Starts a greedy pass over <paramref name="pcm"/> (at most <see cref="MaxSeconds"/>): the audio is copied and
    /// turned into features at once, the rest runs in <see cref="StepSpanPass"/>. False when the pass already ended
    /// (silence or too little audio).
    /// </summary>
    bool BeginSpanPass(ReadOnlySpan<float> pcm, string? language)
    {
        span = default;
        span.Language = language;
        span.Empty = true;
        span.Stage = SpanStage.Done;
        if (Silent(pcm)) return false;
        span.Started = Stopwatch.GetTimestamp();
        var l0 = EncodeFeatures(pcm);
        if (l0 < 1) return false;
        span.L0 = l0;
        span.Stage = SpanStage.Stem;
        return true;
    }

    /// <summary>Runs one stage of the pass: the stem, one encoder layer, the cross memory, priming or one token. True when done.</summary>
    bool StepSpanPass()
    {
        switch (span.Stage)
        {
            case SpanStage.Stem:
                EncodeStem(span.L0);
                span.Stage = SpanStage.EncoderStart;
                return false;
            case SpanStage.EncoderStart:
                span.Job = EncoderStart(frames);
                span.Layer = 0;
                span.Stage = audio->EncoderLayers > 0 ? SpanStage.EncoderLayer : SpanStage.EncoderFinish;
                return false;
            case SpanStage.EncoderLayer:
                EncoderLayerStep(ref span.Job, span.Layer, frames);
                if (++span.Layer >= audio->EncoderLayers) span.Stage = SpanStage.EncoderFinish;
                return false;
            case SpanStage.EncoderFinish:
                EncoderFinish(ref span.Job, frames);
                span.Stage = SpanStage.Memory;
                return false;
            case SpanStage.Memory:
                BuildCrossMemory();
                span.EncodeMs = (Stopwatch.GetTimestamp() - span.Started) * 1000.0 / Stopwatch.Frequency;
                span.Stage = SpanStage.Prime;
                return false;
            case SpanStage.Prime:
                span.Detected = PrimeSpan(span.Language);
                span.Count = 0;
                span.HasKeywords = spanKeywordStrings.Length > 0;
                span.DecodeStart = Stopwatch.GetTimestamp();
                span.Stage = SpanStage.Decode;
                return false;
            case SpanStage.Decode:
                return DecodeSpanStep();
            default:
                return true;
        }
    }

    bool DecodeSpanStep()
    {
        var tokens = spanTokens!;
        var probs = spanProbs!;
        var step = span.Count;
        if (step >= g.MaxSeq - 3) return FinishSpanPass();
        if (CancelPending()) return FinishSpanPass();
        var one = stackalloc int[1];
        var logits = decoder.Logits;
        int best;
        float prob;
        if (!span.HasKeywords)
        {
            var lp = 0f;
            var p = 0f;
            fixed (float* src = logits)
                Ops.TopKLogSoftmax(src, textVocab, 1, &best, &lp, &p, true);
            prob = p;
        }
        else
        {
            logits.Slice(0, textVocab).CopyTo(new Span<float>(scores, textVocab));
            scores[0] = scores[2] = scores[3] = -1e30f;
            ApplyKeywordBiasSpan(new ReadOnlySpan<int>(tokens, 0, span.Count), scores);
            best = Ops.ArgMax(scores, textVocab);
            prob = Ops.BestProbability(scores, textVocab, scores[best]);
        }
        if (step == 0) span.TtftMs = (Stopwatch.GetTimestamp() - span.Started) * 1000.0 / Stopwatch.Frequency;
        probs[span.Count] = prob;
        if (best == Eos)
        {
            one[0] = Eos;
            decoder.Feed(new ReadOnlySpan<int>(one, 1));
            return FinishSpanPass();
        }
        tokens[span.Count++] = best;
        one[0] = best;
        decoder.Feed(new ReadOnlySpan<int>(one, 1));
        return false;
    }

    bool FinishSpanPass()
    {
        var seconds = Math.Max(1e-9, (Stopwatch.GetTimestamp() - span.DecodeStart) / (double)Stopwatch.Frequency);
        span.Tps = span.Count / seconds;
        span.Empty = false;
        span.Stage = SpanStage.Done;
        return true;
    }

    /// <summary>Starts a greedy pass to be run with <see cref="StepSpan"/>. False when it ended at once (silence).</summary>
    internal bool BeginSpan(ReadOnlySpan<float> pcm, string? language, IReadOnlyList<string>? keywords)
    {
        if (language != null && Array.IndexOf(languages, language) < 0) throw new ArgumentException($"language is one of {string.Join(" ", languages)}");
        if (pcm.Length > MaxSeconds * audio->SampleRate) throw new ArgumentException("a stepped pass takes at most MaxSeconds of audio");
        EnsureDecoderBuffers();
        EnsureSpanBuffers();
        CacheKeywords(keywords);
        Volatile.Write(ref cancelRequested, 0);
        cancelled = false;
        return BeginSpanPass(pcm, language);
    }

    internal bool StepSpan() => StepSpanPass();

    internal bool SpanDone => span.Stage == SpanStage.Done;

    /// <summary>Writes the finished pass's words, with times shifted by <paramref name="shift"/> seconds.</summary>
    internal SpeechSpan FinishSpan(Span<byte> utf8, Span<WordSlice> words, double shift)
    {
        if (span.Empty) return new SpeechSpan(0, 0, null, 0, 0, 0, false);
        return RenderSpan(span.Count, span.Detected, utf8, words, shift, span.TtftMs, span.Tps, span.EncodeMs);
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
        var textOnly = words.Length == 0;
        var frameStarts = !textOnly && tokens > 0 ? TokenFramesSpan(tokens) : null;
        var utf8Length = 0;
        var wordCount = 0;
        var truncated = false;
        var w = 0;
        while (w < tokens)
        {
            var from = w;
            var to = w + 1;
            while (to < tokens && !IsWordStart(tokenArray[to])) to++;
            var at = 0;
            var overflow = false;
            for (var i = from; i < to; i++)
            {
                if (at + tokenizer.PieceByteLength(tokenArray[i]) > scratch.Length) { overflow = true; break; }
                at += tokenizer.Append(tokenArray[i], scratch.AsSpan(at), i == from);
            }
            if (overflow) { truncated = true; break; }
            var start = 0;
            while (start < at && (scratch[start] == (byte)' ' || scratch[start] == (byte)'\t' || scratch[start] == (byte)'\n' || scratch[start] == (byte)'\r')) start++;
            while (at > start && (scratch[at - 1] == (byte)' ' || scratch[at - 1] == (byte)'\t' || scratch[at - 1] == (byte)'\n' || scratch[at - 1] == (byte)'\r')) at--;
            if (at == start) { w = to; continue; }
            var length = at - start;
            if (!textOnly && wordCount >= words.Length) { truncated = true; break; }
            if (utf8Length + (utf8Length > 0 ? 1 : 0) + length > utf8.Length) { truncated = true; break; }
            if (utf8Length > 0) utf8[utf8Length++] = (byte)' ';
            new Span<byte>(scratch, start, length).CopyTo(utf8.Slice(utf8Length));
            var probability = 0.0;
            for (var i = from; i < to; i++) probability += probs[i];
            probability /= to - from;
            var wordStart = frameStarts != null ? starts[from] * FrameSeconds : 0;
            var wordEnd = frameStarts != null ? starts[to] * FrameSeconds : 0;
            if (!textOnly) words[wordCount++] = new WordSlice(utf8Length, length, wordStart + shift, wordEnd + shift, probability);
            utf8Length += length;
            w = to;
        }
        return new SpeechSpan(utf8Length, wordCount, utf8Length > 0 ? detected : null, ttftMs, tps, encodeMs, truncated);
    }

    int[] TokenFramesSpan(int tokens)
    {
        var columns = frames;
        if (tokens < 1 || tokens + 2 > g.MaxSeq || columns < 1 || columns > maxFrames)
            throw new InvalidOperationException($"alignment exceeds its buffers: tokens={tokens} (max {g.MaxSeq - 2}), frames={columns} (max {maxFrames})");
        try
        {
            return AlignTokens(tokens, columns);
        }
        catch (IndexOutOfRangeException error)
        {
            throw new InvalidOperationException($"word alignment failed: tokens={tokens}, frames={columns}, maxFrames={maxFrames}, maxSeq={g.MaxSeq}, cost={alignA!.Length}, total={alignB!.Length}, row={alignRow!.Length}, starts={alignStarts!.Length}", error);
        }
    }

    int[] AlignTokens(int tokens, int columns)
    {
        var rows = tokens + 1;
        var cost = alignA!;
        var total = alignB!;
        var filtered = alignRow!;
        var starts = alignStarts!;
        for (var j = 0; j < columns; j++)
        {
            double mean = 0;
            for (var i = 0; i < rows; i++) mean += alignment[(long)(1 + i) * frames + j];
            mean /= rows;
            double squares = 0;
            for (var i = 0; i < rows; i++) { var d = alignment[(long)(1 + i) * frames + j] - mean; squares += d * d; }
            var std = Math.Sqrt(squares / rows) + 1e-9;
            for (var i = 0; i < rows; i++) total[i * columns + j] = (float)((alignment[(long)(1 + i) * frames + j] - mean) / std);
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
            else if (c > 0) c--;
            else r--;
        }
        for (var i = rows - 2; i >= 0; i--) if (starts[i] == int.MaxValue) starts[i] = starts[i + 1];
        for (var i = 1; i < rows; i++) if (starts[i] == int.MaxValue) starts[i] = starts[i - 1];
        for (var i = 1; i < rows; i++) starts[i] = Math.Max(starts[i], starts[i - 1]);
        return starts;
    }
}
