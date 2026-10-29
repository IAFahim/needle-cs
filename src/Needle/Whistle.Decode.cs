using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Needle;

public sealed unsafe partial class Whistle
{
    const float SilenceRms = 2e-4f;
    public float FirstKeywordBias { get; set; } = 3f;
    public float ContinuedKeywordBias { get; set; } = 8f;
    const double FrameSeconds = 0.08;

    float* alignment;
    float* scores;
    bool decoderReady;

    sealed class Pass
    {
        public List<int> Tokens = new();
        public List<float> Probabilities = new();
        public string Text = "";
        public string? Language;
        public double TtftMs, Tps, EncodeMs;
        public List<Word> Words = new();
    }

    public Transcription Transcribe(ReadOnlySpan<float> pcm, string? language = null, IReadOnlyList<string>? keywords = null, bool wordTimestamps = false)
    {
        if (language != null && Array.IndexOf(languages, language) < 0) throw new ArgumentException($"language is one of {string.Join(" ", languages)}");
        var maxSamples = MaxSeconds * audio->SampleRate;
        if (pcm.Length <= maxSamples)
        {
            var pass = Decode(pcm, language, keywords, wordTimestamps);
            return Present(pass, pass.Text, wordTimestamps ? pass.Words : null);
        }
        return TranscribeLong(pcm, language, keywords, wordTimestamps);
    }

    static Transcription Present(Pass p, string text, IReadOnlyList<Word>? words) => new()
    {
        Text = text, Language = text.Length > 0 ? p.Language : null, TimeToFirstTokenMs = p.TtftMs, DecodeTps = p.Tps, EncodeMs = p.EncodeMs, Words = words,
    };

    Transcription TranscribeLong(ReadOnlySpan<float> pcm, string? language, IReadOnlyList<string>? keywords, bool wordTimestamps)
    {
        var rate = audio->SampleRate;
        var window = MaxSeconds * rate;
        var words = new List<Word>();
        var offset = 0;
        Pass? first = null;
        string? detected = language;
        while (offset < pcm.Length)
        {
            var length = Math.Min(window, pcm.Length - offset);
            var pass = Decode(pcm.Slice(offset, length), detected, keywords, true);
            first ??= pass;
            detected ??= pass.Language;
            var base_ = offset / (double)rate;
            var last = offset + length >= pcm.Length;
            var cut = pass.Words.Count;
            if (!last)
            {
                cut = -1;
                for (var i = pass.Words.Count - 1; i >= 0; i--)
                    if (EndsSentence(pass.Words[i].Text)) { cut = i + 1; break; }
                if (cut <= 0) cut = pass.Words.Count;
            }
            for (var i = 0; i < cut; i++)
            {
                var w = pass.Words[i];
                words.Add(w with { Start = w.Start + base_, End = w.End + base_ });
            }
            if (last) break;
            var advance = cut > 0 && cut < pass.Words.Count ? Math.Max(1.0, pass.Words[cut - 1].End) * rate : length;
            offset += (int)Math.Min(length, advance);
        }
        var text = string.Join(" ", words.ConvertAll(w => w.Text));
        return Present(first ?? new Pass(), text, wordTimestamps ? words : null);
    }

    static bool EndsSentence(string word) => word.Length > 0 && word[^1] is '.' or '?' or '!' or '。' or '？' or '！';

    static float Rms(ReadOnlySpan<float> pcm)
    {
        double sum = 0;
        foreach (var x in pcm) sum += x * x;
        return pcm.Length == 0 ? 0 : (float)Math.Sqrt(sum / pcm.Length);
    }

    string[]? pieceTexts;

    string[] Pieces()
    {
        if (pieceTexts != null) return pieceTexts;
        pieceTexts = new string[textVocab];
        for (var id = 0; id < textVocab; id++) pieceTexts[id] = tokenizer.IsPlain(id) && id > 3 ? tokenizer.PieceText(id) : "";
        return pieceTexts;
    }

    void EnsureDecoderBuffers()
    {
        if (decoderReady) return;
        alignment = (float*)Mem.AllocZeroed((nuint)g.MaxSeq * (nuint)maxFrames * 4);
        scores = (float*)Mem.Alloc((nuint)model.VocabSize * 4);
        decoderReady = true;
    }

    Pass Decode(ReadOnlySpan<float> pcm, string? language, IReadOnlyList<string>? keywords, bool align)
    {
        var pass = new Pass();
        if (Rms(pcm) < SilenceRms) return pass;
        EnsureDecoderBuffers();
        var started = Stopwatch.GetTimestamp();
        if (Encode(pcm) == 0) return pass;
        BuildCrossMemory();
        pass.EncodeMs = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
        return DecodeMemory(pass, started, language, keywords, align);
    }

    internal string TranscribeEmbedding(ReadOnlySpan<float> embedding, string? language)
    {
        EnsureDecoderBuffers();
        frames = embedding.Length / dim;
        embedding.CopyTo(new Span<float>(encoderOutput, embedding.Length));
        BuildCrossMemory();
        return DecodeMemory(new Pass(), Stopwatch.GetTimestamp(), language, null, false).Text;
    }

    Pass DecodeMemory(Pass pass, long started, string? language, IReadOnlyList<string>? keywords, bool align)
    {
        Prime(language, align, pass);
        var one = stackalloc int[1];
        var logits = decoder.Logits;

        var firstToken = Stopwatch.GetTimestamp();
        var decodeStart = firstToken;
        var steps = 0;
        var forced = BeamWidth > 1 ? BeamSearch(keywords) : null;
        if (forced != null && !align)
        {
            pass.Tokens.AddRange(forced);
            pass.TtftMs = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
            steps = forced.Count;
        }
        for (var step = 0; step < g.MaxSeq - 3 && (forced == null || align); step++)
        {
            logits = decoder.Logits;
            for (var i = 0; i < textVocab; i++) scores[i] = logits[i];
            scores[0] = scores[2] = scores[3] = -1e30f;
            ApplyKeywordBias(keywords, pass.Tokens, scores);
            var best = 0;
            for (var i = 1; i < textVocab; i++) if (scores[i] > scores[best]) best = i;
            if (forced != null) best = step < forced.Count ? forced[step] : Eos;
            if (step == 0) pass.TtftMs = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
            if (align)
            {
                Ops.Softmax(scores, textVocab);
                pass.Probabilities.Add(scores[best]);
            }
            if (best == Eos)
            {
                if (align)
                {
                    one[0] = Eos;
                    decoder.Feed(new ReadOnlySpan<int>(one, 1));
                }
                break;
            }
            pass.Tokens.Add(best);
            one[0] = best;
            decoder.Feed(new ReadOnlySpan<int>(one, 1));
            steps++;
        }
        var seconds = Math.Max(1e-9, (Stopwatch.GetTimestamp() - decodeStart) / (double)Stopwatch.Frequency);
        pass.Tps = steps / seconds;
        BuildText(pass, align);
        return pass;
    }

    public int BeamWidth { get; set; } = 4;
    public float BeamLengthPenalty { get; set; }

    Engine[]? beamEngines;
    float* beamScores;

    void EnsureBeams(int width)
    {
        if (beamEngines != null && beamEngines.Length == 2 * width) return;
        DisposeBeams();
        beamEngines = new Engine[2 * width];
        for (var i = 0; i < beamEngines.Length; i++) beamEngines[i] = new Engine(model, g.MaxSeq, 64, policy, pool);
        beamScores = (float*)Mem.Alloc((nuint)(width * textVocab) * 4);
    }

    void DisposeBeams()
    {
        if (beamEngines != null) foreach (var engine in beamEngines) engine.Dispose();
        if (beamScores != null) Mem.Free(beamScores);
        beamEngines = null;
        beamScores = null;
    }

    float Normalized(float score, int length) => BeamLengthPenalty == 0f ? score : score / MathF.Pow(length, BeamLengthPenalty);

    List<int> BeamSearch(IReadOnlyList<string>? keywords)
    {
        var width = BeamWidth;
        EnsureBeams(width);
        foreach (var engine in beamEngines!) engine.Attach(memory);
        var live = beamEngines.AsSpan(0, width).ToArray();
        var next = beamEngines.AsSpan(width, width).ToArray();
        var liveTokens = new List<int>[width];
        var liveScores = new float[width];
        liveTokens[0] = new List<int>();
        live[0].CopyFrom(decoder);
        var liveCount = 1;
        var finished = new List<(float Score, List<int> Tokens)>();
        var candidates = new List<(float Score, int Parent, int Token)>();
        var parents = new int[width];
        var picks = new int[width];
        var pickScores = new float[width];
        var one = stackalloc int[1];
        for (var step = 0; step < g.MaxSeq - 4; step++)
        {
            candidates.Clear();
            for (var i = 0; i < liveCount; i++)
            {
                var row = beamScores + (long)i * textVocab;
                var logits = live[i].Logits;
                for (var v = 0; v < textVocab; v++) row[v] = logits[v];
                row[0] = row[2] = row[3] = -1e30f;
                ApplyKeywordBias(keywords, liveTokens[i], row);
                LogSoftmax(row, textVocab);
                for (var k = 0; k < width; k++)
                {
                    var best = 0;
                    for (var v = 1; v < textVocab; v++) if (row[v] > row[best]) best = v;
                    candidates.Add((liveScores[i] + row[best], i, best));
                    row[best] = float.NegativeInfinity;
                }
            }
            candidates.Sort((a, b) => b.Score.CompareTo(a.Score));
            var kept = 0;
            foreach (var c in candidates)
            {
                if (c.Token == Eos) finished.Add((c.Score, liveTokens[c.Parent]));
                else
                {
                    parents[kept] = c.Parent;
                    picks[kept] = c.Token;
                    pickScores[kept] = c.Score;
                    kept++;
                }
                if (kept >= width) break;
            }
            if (kept == 0) break;
            if (finished.Count >= width)
            {
                var bestFinished = float.NegativeInfinity;
                foreach (var f in finished) bestFinished = MathF.Max(bestFinished, Normalized(f.Score, f.Tokens.Count + 1));
                var bestLive = float.NegativeInfinity;
                for (var j = 0; j < kept; j++) bestLive = MathF.Max(bestLive, Normalized(pickScores[j], liveTokens[parents[j]].Count + 1));
                if (bestFinished >= bestLive) break;
            }
            var nextTokens = new List<int>[width];
            for (var j = 0; j < kept; j++)
            {
                next[j].CopyFrom(live[parents[j]]);
                one[0] = picks[j];
                next[j].Feed(new ReadOnlySpan<int>(one, 1));
                nextTokens[j] = new List<int>(liveTokens[parents[j]]) { picks[j] };
                liveScores[j] = pickScores[j];
            }
            (live, next) = (next, live);
            liveTokens = nextTokens;
            liveCount = kept;
        }
        var winner = default(List<int>);
        var winnerScore = float.NegativeInfinity;
        foreach (var f in finished)
        {
            var score = Normalized(f.Score, f.Tokens.Count + 1);
            if (score > winnerScore) { winnerScore = score; winner = f.Tokens; }
        }
        if (winner != null) return winner;
        for (var j = 0; j < liveCount; j++)
        {
            var score = Normalized(liveScores[j], liveTokens[j].Count);
            if (score > winnerScore) { winnerScore = score; winner = liveTokens[j]; }
        }
        return winner ?? new List<int>();
    }

    static void LogSoftmax(float* x, int n)
    {
        var peak = float.NegativeInfinity;
        for (var i = 0; i < n; i++) peak = MathF.Max(peak, x[i]);
        double sum = 0;
        for (var i = 0; i < n; i++) sum += Math.Exp(x[i] - peak);
        var shift = peak + (float)Math.Log(sum);
        for (var i = 0; i < n; i++) x[i] -= shift;
    }

    void Prime(string? language, bool align, Pass pass)
    {
        decoder.Reset();
        decoder.Alignment = null;
        if (align)
        {
            Mem.Clear(alignment, (nuint)g.MaxSeq * (nuint)maxFrames * 4);
            decoder.Alignment = alignment;
            decoder.AlignmentFromLayer = g.Layers / 2;
        }
        var one = stackalloc int[1];
        one[0] = Bos;
        decoder.Feed(new ReadOnlySpan<int>(one, 1));
        var logits = decoder.Logits;
        Observer?.Invoke("logits_bos", logits);
        int tag;
        if (language != null) tag = textVocab + Array.IndexOf(languages, language);
        else
        {
            tag = textVocab;
            for (var i = 1; i < languages.Length; i++) if (logits[textVocab + i] > logits[tag]) tag = textVocab + i;
        }
        pass.Language = languages[tag - textVocab];
        one[0] = tag;
        decoder.Feed(new ReadOnlySpan<int>(one, 1));
        Observer?.Invoke("logits_tag", decoder.Logits);
    }

    string WordSoFar(List<int> generated)
    {
        var pieces = Pieces();
        var word = new StringBuilder();
        for (var i = generated.Count - 1; i >= 0; i--)
        {
            var piece = pieces[generated[i]];
            if (piece.StartsWith('\u2581')) { word.Insert(0, piece[1..]); return word.ToString(); }
            word.Insert(0, piece);
        }
        return word.ToString();
    }

    void ApplyKeywordBias(IReadOnlyList<string>? keywords, List<int> generated, float* target)
    {
        if (keywords == null || keywords.Count == 0) return;
        var pieces = Pieces();
        var current = WordSoFar(generated);
        foreach (var keyword in keywords)
        {
            var word = keyword.Trim();
            if (word.Length <= current.Length || !word.StartsWith(current, StringComparison.OrdinalIgnoreCase)) continue;
            var remaining = word.AsSpan(current.Length);
            var bias = current.Length == 0 ? FirstKeywordBias : ContinuedKeywordBias;
            for (var id = 4; id < textVocab; id++)
            {
                var piece = pieces[id];
                if (piece.Length == 0) continue;
                var startsWord = piece[0] == '\u2581';
                if (startsWord != (current.Length == 0)) continue;
                var text = startsWord ? piece.AsSpan(1) : piece.AsSpan();
                if (text.Length == 0 || (current.Length == 0 && text.Length < 2 && remaining.Length > 1) || text.Length > remaining.Length) continue;
                if (!remaining.StartsWith(text, StringComparison.OrdinalIgnoreCase)) continue;
                var exact = remaining.StartsWith(text, StringComparison.Ordinal);
                target[id] += bias * (exact ? 1f : 0.5f) * (text.Length == remaining.Length ? 1.3f : 1f);
            }
        }
    }

    string TokenText(IReadOnlyList<int> ids, int from, int to)
    {
        var bytes = new List<byte>();
        var piece = new byte[64];
        for (var i = from; i < to; i++) bytes.AddRange(new ArraySegment<byte>(piece, 0, tokenizer.Append(ids[i], piece, i == from)));
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    void BuildText(Pass pass, bool align)
    {
        var starts = new List<int>();
        for (var i = 0; i < pass.Tokens.Count; i++)
            if (i == 0 || tokenizer.PieceText(pass.Tokens[i]).StartsWith('▁')) starts.Add(i);
        var text = new StringBuilder();
        var rows = pass.Tokens.Count;
        var frameStarts = align && rows > 0 ? TokenFrames(rows) : null;
        for (var w = 0; w < starts.Count; w++)
        {
            var from = starts[w];
            var to = w + 1 < starts.Count ? starts[w + 1] : rows;
            var word = TokenText(pass.Tokens, from, to).Trim();
            if (word.Length == 0) continue;
            if (text.Length > 0) text.Append(' ');
            text.Append(word);
            if (frameStarts == null) continue;
            var probability = 0.0;
            for (var i = from; i < to; i++) probability += pass.Probabilities[i];
            pass.Words.Add(new Word(word, frameStarts[from] * FrameSeconds, frameStarts[to] * FrameSeconds, probability / (to - from)));
        }
        pass.Text = text.ToString();
    }

    int[] TokenFrames(int tokens)
    {
        var rows = tokens + 1;
        var columns = frames;
        var cost = new float[rows, columns];
        var filtered = new float[columns];
        var window = stackalloc float[7];
        var normalized = new float[rows, columns];
        for (var j = 0; j < columns; j++)
        {
            double mean = 0, squares = 0;
            for (var i = 0; i < rows; i++) mean += alignment[(long)(1 + i) * frames + j];
            mean /= rows;
            for (var i = 0; i < rows; i++) { var d = alignment[(long)(1 + i) * frames + j] - mean; squares += d * d; }
            var std = Math.Sqrt(squares / rows) + 1e-9;
            for (var i = 0; i < rows; i++) normalized[i, j] = (float)((alignment[(long)(1 + i) * frames + j] - mean) / std);
        }
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < columns; j++)
            {
                var n = 0;
                for (var k = -3; k <= 3; k++) window[n++] = normalized[i, Math.Clamp(j + k, 0, columns - 1)];
                for (var a2 = 1; a2 < 7; a2++)
                {
                    var v = window[a2];
                    var b2 = a2 - 1;
                    while (b2 >= 0 && window[b2] > v) { window[b2 + 1] = window[b2]; b2--; }
                    window[b2 + 1] = v;
                }
                filtered[j] = window[3];
            }
            for (var j = 0; j < columns; j++) cost[i, j] = -filtered[j];
        }
        var total = new float[rows, columns];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < columns; j++)
            {
                float best;
                if (i == 0) best = j > 0 ? Math.Min(0f, total[0, j - 1]) : 0f;
                else
                {
                    best = total[i - 1, j];
                    if (j > 0) best = Math.Min(best, Math.Min(total[i - 1, j - 1], total[i, j - 1]));
                }
                total[i, j] = cost[i, j] + best;
            }
        var starts = new int[rows];
        Array.Fill(starts, int.MaxValue);
        var r = rows - 1;
        var c = 0;
        for (var j = 1; j < columns; j++) if (total[r, j] < total[r, c]) c = j;
        for (;;)
        {
            starts[r] = Math.Min(starts[r], c);
            if (r == 0)
            {
                if (c > 0 && total[0, c - 1] < 0f) { c--; continue; }
                break;
            }
            var diagonal = c > 0 ? total[r - 1, c - 1] : float.PositiveInfinity;
            var up = total[r - 1, c];
            var left = c > 0 ? total[r, c - 1] : float.PositiveInfinity;
            if (diagonal <= up && diagonal <= left) { r--; c--; }
            else if (up <= left) r--;
            else c--;
        }
        for (var i = rows - 2; i >= 0; i--) if (starts[i] == int.MaxValue) starts[i] = starts[i + 1];
        for (var i = 1; i < rows; i++) if (starts[i] == int.MaxValue) starts[i] = starts[i - 1];
        for (var i = 1; i < rows; i++) starts[i] = Math.Max(starts[i], starts[i - 1]);
        return starts;
    }

    public WhistleStream Stream(string? language = null, IReadOnlyList<string>? keywords = null) => new(this, language, keywords);

    internal Transcription Pass_(ReadOnlySpan<float> pcm, string? language, IReadOnlyList<string>? keywords) =>
        Transcribe(pcm, language, keywords, true);
}

public readonly record struct StreamStep(string Text, IReadOnlyList<Word> Words, string Pending, string? Language, double Received, double PassMs);

public sealed class WhistleStream
{
    readonly Whistle whistle;
    readonly string? language;
    readonly IReadOnlyList<string>? keywords;
    readonly List<float> buffer = new();
    List<Word> previous = new();
    double bufferStart;
    double received;
    string? detected;

    internal WhistleStream(Whistle whistle, string? language, IReadOnlyList<string>? keywords)
    {
        this.whistle = whistle;
        this.language = language;
        this.keywords = keywords;
    }

    static string Key(string word) => new string(Array.FindAll(word.ToLowerInvariant().ToCharArray(), char.IsLetterOrDigit));

    public StreamStep Process(ReadOnlySpan<float> chunk)
    {
        buffer.AddRange(chunk.ToArray());
        received += chunk.Length / (double)Wav.SampleRate;
        return Pass(final: false);
    }

    public StreamStep Stop() => Pass(final: true);

    StreamStep Pass(bool final)
    {
        var started = Stopwatch.GetTimestamp();
        var window = buffer.Count > Whistle.MaxSeconds * Wav.SampleRate ? buffer.GetRange(buffer.Count - Whistle.MaxSeconds * Wav.SampleRate, Whistle.MaxSeconds * Wav.SampleRate) : buffer;
        var result = whistle.Transcribe(window.ToArray(), language ?? detected, keywords, true);
        detected ??= result.Language;
        var shift = bufferStart + (buffer.Count - window.Count) / (double)Wav.SampleRate;
        var words = new List<Word>();
        foreach (var w in result.Words ?? Array.Empty<Word>()) words.Add(w with { Start = w.Start + shift, End = w.End + shift });

        var agreed = 0;
        if (final) agreed = words.Count;
        else
            while (agreed < words.Count && agreed < previous.Count && Key(words[agreed].Text) == Key(previous[agreed].Text)) agreed++;
        var committed = words.GetRange(0, agreed);
        var pending = words.GetRange(agreed, words.Count - agreed);
        if (committed.Count > 0 && !final)
        {
            var cutSeconds = pending.Count > 0 ? (committed[^1].End + pending[0].Start) / 2 : committed[^1].End;
            var drop = (int)Math.Clamp((cutSeconds - bufferStart) * Wav.SampleRate, 0, buffer.Count);
            buffer.RemoveRange(0, drop);
            bufferStart += drop / (double)Wav.SampleRate;
            previous = pending;
        }
        else previous = words;
        var text = string.Join(" ", committed.ConvertAll(w => w.Text));
        return new StreamStep(text, committed, string.Join(" ", pending.ConvertAll(w => w.Text)), result.Language ?? detected, received, (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency);
    }
}
