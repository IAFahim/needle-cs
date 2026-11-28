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
        Enter();
        try
        {
            if (pcm.Length <= maxSamples)
            {
                var pass = Decode(pcm, language, keywords, wordTimestamps);
                return Present(pass, pass.Text, wordTimestamps ? pass.Words : null);
            }
            return TranscribeLong(pcm, language, keywords, wordTimestamps);
        }
        finally { Exit(); }
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
            if (offset > 0 && CancelPending()) break;
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

    static bool Silent(ReadOnlySpan<float> pcm)
    {
        var rms = Rms(pcm);
        return !(rms >= SilenceRms) || float.IsInfinity(rms);
    }

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

    void FreeDecoderBuffers()
    {
        if (!decoderReady) return;
        Mem.Free(alignment);
        Mem.Free(scores);
        alignment = null;
        scores = null;
        decoderReady = false;
    }

    Pass Decode(ReadOnlySpan<float> pcm, string? language, IReadOnlyList<string>? keywords, bool align)
    {
        var pass = new Pass();
        if (Silent(pcm)) return pass;
        EnsureDecoderBuffers();
        var started = Stopwatch.GetTimestamp();
        if (EncodeCore(pcm) == 0) return pass;
        BuildCrossMemory();
        pass.EncodeMs = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
        return DecodeMemory(pass, started, language, keywords, align);
    }

    internal string TranscribeEmbedding(ReadOnlySpan<float> embedding, string? language)
    {
        if (embedding.Length % dim != 0 || embedding.Length / dim > maxFrames) throw new ArgumentException($"embedding holds whole {dim}-wide rows, at most {maxFrames}", nameof(embedding));
        Enter();
        try
        {
            EnsureDecoderBuffers();
            frames = embedding.Length / dim;
            embedding.CopyTo(new Span<float>(encoderOutput, embedding.Length));
            BuildCrossMemory();
            return DecodeMemory(new Pass(), Stopwatch.GetTimestamp(), language, null, false).Text;
        }
        finally { Exit(); }
    }

    Pass DecodeMemory(Pass pass, long started, string? language, IReadOnlyList<string>? keywords, bool align)
    {
        Prime(language, align, pass);
        var one = stackalloc int[1];
        var logits = decoder.Logits;

        var firstToken = Stopwatch.GetTimestamp();
        var decodeStart = firstToken;
        var steps = 0;
        var beam = BeamWidth > 1;
        if (beam)
        {
            CacheKeywords(keywords);
            var count = BeamSearch(align);
            pass.TtftMs = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
            steps = count;
            for (var i = 0; i < count; i++) pass.Tokens.Add(spanTokens![i]);
            if (align)
            {
                for (var i = 0; i < count; i++) pass.Probabilities.Add(spanProbs![i]);
                FeedForAlignment(count);
            }
        }
        var hasKeywords = keywords != null && keywords.Count > 0;
        for (var step = 0; step < g.MaxSeq - 3 && !beam; step++)
        {
            if (CancelPending()) break;
            logits = decoder.Logits;
            int best;
            float prob;
            if (!hasKeywords)
            {
                var lp = 0f;
                var p = 0f;
                fixed (float* src = logits)
                    Ops.TopKLogSoftmax(src, textVocab, 1, &best, &lp, align ? &p : null, true);
                prob = p;
            }
            else
            {
                logits.Slice(0, textVocab).CopyTo(new Span<float>(scores, textVocab));
                scores[0] = scores[2] = scores[3] = -1e30f;
                ApplyKeywordBias(keywords, pass.Tokens, scores);
                best = Ops.ArgMax(scores, textVocab);
                prob = align ? Ops.BestProbability(scores, textVocab, scores[best]) : 0f;
            }
            if (step == 0) pass.TtftMs = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
            if (align) pass.Probabilities.Add(prob);
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

    public int BeamWidth { get; set; } = 3;
    public float BeamLengthPenalty { get; set; } = 1f;

    struct BeamCandidate
    {
        public float Score;
        public float Prob;
        public int Parent;
        public int Token;
    }

    static bool Before(in BeamCandidate a, in BeamCandidate b) => b.Score.CompareTo(a.Score) < 0;

    static void SwapIfAfter(BeamCandidate[] items, int i, int j)
    {
        if (!Before(items[j], items[i])) return;
        (items[i], items[j]) = (items[j], items[i]);
    }

    static void SortCandidates(BeamCandidate[] items, int count)
    {
        if (count == 2) { SwapIfAfter(items, 0, 1); return; }
        if (count == 3)
        {
            SwapIfAfter(items, 0, 1);
            SwapIfAfter(items, 0, 2);
            SwapIfAfter(items, 1, 2);
            return;
        }
        for (var i = 0; i < count - 1; i++)
        {
            var item = items[i + 1];
            var j = i;
            while (j >= 0 && Before(item, items[j]))
            {
                items[j + 1] = items[j];
                j--;
            }
            items[j + 1] = item;
        }
    }

    Engine[]? beamEngines;
    Engine[]? beamLive;
    Engine[]? beamNext;
    int[]? beamTokensA;
    int[]? beamTokensB;
    float[]? beamProbsA;
    float[]? beamProbsB;
    int[]? beamLengthsA;
    int[]? beamLengthsB;
    float[]? beamLiveScores;
    BeamCandidate[]? beamCandidates;
    float* beamScores;
    bool beamParallel;
    GCHandle beamSelf;
    Engine[]? beamTaskEngines;
    Engine[]? beamTaskSources;
    int[]? beamTaskTokens;

    bool ParallelBeams(int width)
    {
        if (width < 2 || policy.Threads < 2) return false;
        var choice = Environment.GetEnvironmentVariable("NEEDLE_BEAM_PARALLEL");
        if (choice != null) return choice != "0";
        return policy.Threads >= width || RuntimeInformation.ProcessArchitecture != Architecture.Arm64;
    }

    void EnsureBeams(int width)
    {
        if (width < 1) throw new InvalidOperationException("BeamWidth must be at least 1");
        if (beamEngines != null && beamEngines.Length == 2 * width) return;
        DisposeBeams();
        var parallel = ParallelBeams(width);
        var engines = new Engine[2 * width];
        try
        {
            for (var i = 0; i < engines.Length; i++)
                engines[i] = parallel
                    ? new Engine(model, g.MaxSeq, 64, policy.WithThreads(1).WithCaller(CallerRole.Participates), null)
                    : new Engine(model, g.MaxSeq, 64, policy, pool);
        }
        catch
        {
            foreach (var engine in engines) engine?.Dispose();
            throw;
        }
        beamEngines = engines;
        beamLive = new Engine[width];
        beamNext = new Engine[width];
        beamTokensA = new int[width * g.MaxSeq];
        beamTokensB = new int[width * g.MaxSeq];
        beamProbsA = new float[width * g.MaxSeq];
        beamProbsB = new float[width * g.MaxSeq];
        beamLengthsA = new int[width];
        beamLengthsB = new int[width];
        beamLiveScores = new float[width];
        beamCandidates = new BeamCandidate[width * width];
        beamScores = (float*)Mem.Alloc((nuint)(width * textVocab) * 4);
        beamParallel = parallel;
        if (parallel)
        {
            beamTaskEngines = new Engine[width];
            beamTaskSources = new Engine[width];
            beamTaskTokens = new int[width];
            beamSelf = GCHandle.Alloc(this, GCHandleType.Weak);
        }
    }

    void RunBeamTasks(delegate*<void*, int, void> function, int count)
    {
        if (count == 0) return;
        if (count == 1) { function((void*)GCHandle.ToIntPtr(beamSelf), 0); return; }
        pool.Run(function, (void*)GCHandle.ToIntPtr(beamSelf), count);
    }

    static Whistle BeamOwner(void* context) => (Whistle)GCHandle.FromIntPtr((IntPtr)context).Target!;

    static void ExecuteBeamCopy(void* context, int task)
    {
        var owner = BeamOwner(context);
        owner.beamTaskEngines![task].CopyFrom(owner.beamTaskSources![task], copyLogits: false);
    }

    static void ExecuteBeamFeed(void* context, int task)
    {
        var owner = BeamOwner(context);
        var token = owner.beamTaskTokens![task];
        owner.beamTaskEngines![task].Feed(new ReadOnlySpan<int>(&token, 1));
    }

    void DisposeBeams()
    {
        if (beamEngines != null) foreach (var engine in beamEngines) engine.Dispose();
        if (beamScores != null) Mem.Free(beamScores);
        beamEngines = null;
        beamLive = null;
        beamNext = null;
        beamScores = null;
        if (beamSelf.IsAllocated) beamSelf.Free();
        beamParallel = false;
    }

    float Normalized(float score, int length) => BeamLengthPenalty == 0f ? score : score / MathF.Pow(length, BeamLengthPenalty);

    int BeamSearch(bool trackProbs)
    {
        var width = BeamWidth;
        EnsureBeams(width);
        EnsureSpanBuffers();
        var engines = beamEngines!;
        foreach (var engine in engines) engine.Attach(memory);
        var live = beamLive!;
        var next = beamNext!;
        for (var i = 0; i < width; i++)
        {
            live[i] = engines[i];
            next[i] = engines[width + i];
        }
        var stride = g.MaxSeq;
        var liveTokens = beamTokensA!;
        var nextTokens = beamTokensB!;
        var liveProbs = beamProbsA!;
        var nextProbs = beamProbsB!;
        var liveLengths = beamLengthsA!;
        var nextLengths = beamLengthsB!;
        var liveScores = beamLiveScores!;
        var candidates = beamCandidates!;
        var outTokens = spanTokens!;
        var outProbs = spanProbs!;
        liveLengths[0] = 0;
        liveScores[0] = 0f;
        live[0].CopyFrom(decoder);
        var liveCount = 1;
        var finishedCount = 0;
        var bestFinished = float.NegativeInfinity;
        var winnerFinished = float.NegativeInfinity;
        var bestFinishedLength = -1;
        var parents = stackalloc int[width];
        var picks = stackalloc int[width];
        var pickScores = stackalloc float[width];
        var pickProbs = stackalloc float[width];
        var lastChild = stackalloc int[width];
        var topIdx = stackalloc int[width];
        var topLogProbs = stackalloc float[width];
        var topProbs = stackalloc float[width];
        var one = stackalloc int[1];
        var hasKeywords = spanKeywordStrings.Length > 0;
        for (var step = 0; step < g.MaxSeq - 4; step++)
        {
            if (CancelPending()) break;
            var n = 0;
            for (var i = 0; i < liveCount; i++)
            {
                var logits = live[i].Logits;
                if (hasKeywords)
                {
                    var row = beamScores + (long)i * textVocab;
                    logits.Slice(0, textVocab).CopyTo(new Span<float>(row, textVocab));
                    row[0] = row[2] = row[3] = -1e30f;
                    ApplyKeywordBiasSpan(new ReadOnlySpan<int>(liveTokens, i * stride, liveLengths[i]), row);
                    Ops.TopKLogSoftmax(row, textVocab, width, topIdx, topLogProbs, trackProbs ? topProbs : null, false);
                }
                else
                {
                    fixed (float* lp = logits)
                        Ops.TopKLogSoftmax(lp, textVocab, width, topIdx, topLogProbs, trackProbs ? topProbs : null, true);
                }
                for (var k = 0; k < width; k++)
                    candidates[n++] = new BeamCandidate { Score = liveScores[i] + topLogProbs[k], Prob = trackProbs ? topProbs[k] : 0f, Parent = i, Token = topIdx[k] };
            }
            SortCandidates(candidates, n);
            var kept = 0;
            for (var c = 0; c < n; c++)
            {
                var candidate = candidates[c];
                if (candidate.Token == Eos)
                {
                    var length = liveLengths[candidate.Parent];
                    var normalized = Normalized(candidate.Score, length + 1);
                    bestFinished = MathF.Max(bestFinished, normalized);
                    finishedCount++;
                    if (normalized > winnerFinished)
                    {
                        winnerFinished = normalized;
                        bestFinishedLength = length;
                        Array.Copy(liveTokens, candidate.Parent * stride, outTokens, 0, length);
                        if (trackProbs) Array.Copy(liveProbs, candidate.Parent * stride, outProbs, 0, length);
                    }
                }
                else
                {
                    parents[kept] = candidate.Parent;
                    picks[kept] = candidate.Token;
                    pickScores[kept] = candidate.Score;
                    pickProbs[kept] = candidate.Prob;
                    kept++;
                }
                if (kept >= width) break;
            }
            if (kept == 0) break;
            if (finishedCount >= width)
            {
                var bestLive = float.NegativeInfinity;
                for (var j = 0; j < kept; j++) bestLive = MathF.Max(bestLive, Normalized(pickScores[j], liveLengths[parents[j]] + 1));
                if (bestFinished >= bestLive) break;
            }
            for (var p = 0; p < liveCount; p++) lastChild[p] = -1;
            for (var j = 0; j < kept; j++) lastChild[parents[j]] = j;
            if (beamParallel)
            {
                var taskEngines = beamTaskEngines!;
                var tasks = 0;
                for (var j = 0; j < kept; j++)
                {
                    var p = parents[j];
                    if (j == lastChild[p]) continue;
                    taskEngines[tasks] = next[j];
                    beamTaskSources![tasks] = live[p];
                    tasks++;
                }
                RunBeamTasks(&ExecuteBeamCopy, tasks);
                for (var j = 0; j < kept; j++)
                {
                    var p = parents[j];
                    taskEngines[j] = j == lastChild[p] ? live[p] : next[j];
                    beamTaskTokens![j] = picks[j];
                }
                RunBeamTasks(&ExecuteBeamFeed, kept);
                for (var j = 0; j < kept; j++)
                {
                    var p = parents[j];
                    if (j == lastChild[p]) (next[j], live[p]) = (live[p], next[j]);
                }
            }
            else
            {
                for (var j = 0; j < kept; j++)
                {
                    var p = parents[j];
                    if (j == lastChild[p]) continue;
                    next[j].CopyFrom(live[p], copyLogits: false);
                    one[0] = picks[j];
                    next[j].Feed(new ReadOnlySpan<int>(one, 1));
                }
                for (var j = 0; j < kept; j++)
                {
                    var p = parents[j];
                    if (j != lastChild[p]) continue;
                    one[0] = picks[j];
                    live[p].Feed(new ReadOnlySpan<int>(one, 1));
                    (next[j], live[p]) = (live[p], next[j]);
                }
            }
            for (var j = 0; j < kept; j++)
            {
                var p = parents[j];
                var length = liveLengths[p];
                Array.Copy(liveTokens, p * stride, nextTokens, j * stride, length);
                nextTokens[j * stride + length] = picks[j];
                if (trackProbs)
                {
                    Array.Copy(liveProbs, p * stride, nextProbs, j * stride, length);
                    nextProbs[j * stride + length] = pickProbs[j];
                }
                nextLengths[j] = length + 1;
                liveScores[j] = pickScores[j];
            }
            (live, next) = (next, live);
            (liveTokens, nextTokens) = (nextTokens, liveTokens);
            (liveProbs, nextProbs) = (nextProbs, liveProbs);
            (liveLengths, nextLengths) = (nextLengths, liveLengths);
            liveCount = kept;
        }
        if (bestFinishedLength >= 0) return bestFinishedLength;
        var winner = -1;
        var winnerScore = float.NegativeInfinity;
        for (var j = 0; j < liveCount; j++)
        {
            var score = Normalized(liveScores[j], liveLengths[j]);
            if (score > winnerScore) { winnerScore = score; winner = j; }
        }
        if (winner < 0) return 0;
        Array.Copy(liveTokens, winner * stride, outTokens, 0, liveLengths[winner]);
        if (trackProbs) Array.Copy(liveProbs, winner * stride, outProbs, 0, liveLengths[winner]);
        return liveLengths[winner];
    }

    void FeedForAlignment(int count)
    {
        var tokens = spanTokens!;
        tokens[count] = Eos;
        decoder.Feed(new ReadOnlySpan<int>(tokens, 0, count + 1), computeLogits: false);
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
        int[]? frameStarts = null;
        if (align && rows > 0)
        {
            EnsureSpanBuffers();
            frameStarts = TokenFramesSpan(rows);
        }
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
