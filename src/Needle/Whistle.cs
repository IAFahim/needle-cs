using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Needle;

internal delegate void StageObserver(string stage, ReadOnlySpan<float> values);

public readonly record struct Word(string Text, double Start, double End, double Probability);

public sealed class Transcription
{
    public string Text { get; init; } = "";
    public string? Language { get; init; }
    public double TimeToFirstTokenMs { get; init; }
    public double DecodeTps { get; init; }
    public double EncodeMs { get; init; }
    public IReadOnlyList<Word>? Words { get; init; }
}

public sealed unsafe partial class Whistle : IDisposable
{
    public const int MaxSeconds = 30;
    const int StemBlock = 16;
    const int Eos = 1, Bos = 2;

    readonly Model model;
    readonly Pool pool;
    readonly AudioWeights* audio;
    readonly Geometry g;
    readonly Engine decoder;
    readonly Tokenizer tokenizer;
    readonly int dim, maxFrames, textVocab;
    readonly string[] languages;
    Arena arena;
    Frontend front;

    float* padded, features, featurePeaks, stemOut;
    float* streamA, streamB, normalized, residual, input, hidden, attnOut, gateOut, attnProj, glu, conv;
    float* preOut, postOut, resOut, postGate, residualMix, rowScratch, quantizedRows;
    sbyte* qInt8, kInt8, vInt8;
    float* qScale, kScale, vScale, qRaw, kRaw, vRaw;
    float* ropeCos, ropeSin;
    float* stage0, stage1, stageDepthwise, stagePointwise, zeroRow, stemFlat;
    Activations streamAct, dimAct, stemAct, flatAct, memoryAct;
    float* encoderOutput, crossKRaw, crossVRaw;
    CrossMemory memory;
    int frames;

    internal StageObserver? Observer { get; set; }

    public int Frames => frames;
    public IReadOnlyList<string> Languages => languages;

    public Whistle(Model model, int threads = 1)
        : this(model, ThreadPolicy.Default.WithThreads(threads))
    {
    }

    public Whistle(Model model, ThreadPolicy policy)
    {
        if (model.Audio == null) throw new ArgumentException("this .cact has no speech model");
        this.model = model;
        audio = model.Audio;
        g = model.Geometry;
        dim = g.Dim;
        pool = new Pool(policy);
        decoder = new Engine(model, g.MaxSeq, 64, policy, pool);
        tokenizer = new Tokenizer(model, 1 << 12);
        var rawFrames = MaxSeconds * audio->SampleRate / audio->Hop;
        maxFrames = ((((rawFrames + 1) / 2 + 1) / 2 + 1) / 2) + 2;
        languages = ReadLanguages();
        textVocab = model.VocabSize - languages.Length;
        front = Frontend.Create(*audio);
        AllocateBuffers();
    }

    string[] ReadLanguages()
    {
        var found = new List<string>();
        for (var id = 0; id < tokenizer.PieceCount; id++)
        {
            var text = tokenizer.PieceText(id);
            if (text.StartsWith("<|") && text.EndsWith("|>")) found.Add(text[2..^2]);
            else found.Clear();
        }
        return found.ToArray();
    }

    void AllocateBuffers()
    {
        var samples = MaxSeconds * audio->SampleRate + audio->Window;
        var rawFrames = samples / audio->Hop + 2;
        arena = new Arena((nuint)(256u << 20));
        padded = arena.TakeUncleared<float>((nuint)samples + 512);
        features = arena.TakeUncleared<float>((nuint)(rawFrames * audio->Mels));
        featurePeaks = arena.TakeUncleared<float>((nuint)FeatureJob.Tasks(rawFrames));
        var f = maxFrames;
        stemOut = arena.TakeUncleared<float>((nuint)(f * dim));
        var streamWidth = 4 * dim;
        streamA = arena.TakeUncleared<float>((nuint)(f * streamWidth));
        streamB = arena.TakeUncleared<float>((nuint)(f * streamWidth));
        normalized = arena.TakeUncleared<float>((nuint)(f * streamWidth));
        residual = arena.TakeUncleared<float>((nuint)(f * dim));
        input = arena.TakeUncleared<float>((nuint)(f * dim));
        hidden = arena.TakeUncleared<float>((nuint)(f * dim));
        attnOut = arena.TakeUncleared<float>((nuint)(f * g.OutputWidth));
        gateOut = arena.TakeUncleared<float>((nuint)(f * g.OutputWidth));
        attnProj = arena.TakeUncleared<float>((nuint)(f * dim));
        glu = arena.TakeUncleared<float>((nuint)(f * 2 * dim));
        conv = arena.TakeUncleared<float>((nuint)(f * dim));
        preOut = arena.TakeUncleared<float>((nuint)(f * 4));
        postOut = arena.TakeUncleared<float>((nuint)(f * 4));
        resOut = arena.TakeUncleared<float>((nuint)(f * 16));
        postGate = arena.TakeUncleared<float>((nuint)(f * 4));
        residualMix = arena.TakeUncleared<float>((nuint)(f * 16));
        qRaw = arena.TakeUncleared<float>((nuint)(f * g.QueryWidth));
        kRaw = arena.TakeUncleared<float>((nuint)(f * g.KeyWidth));
        vRaw = arena.TakeUncleared<float>((nuint)(f * g.ValueWidth));
        qInt8 = arena.TakeUncleared<sbyte>((nuint)(f * g.QueryWidth));
        kInt8 = arena.TakeUncleared<sbyte>((nuint)(f * g.KeyWidth));
        vInt8 = arena.TakeUncleared<sbyte>((nuint)(f * g.ValueWidth));
        qScale = arena.TakeUncleared<float>((nuint)(f * g.Heads));
        kScale = arena.TakeUncleared<float>((nuint)(f * g.KvHeads));
        vScale = arena.TakeUncleared<float>((nuint)(f * g.KvHeads));
        rowScratch = arena.TakeUncleared<float>((nuint)(2 * dim + 4096));
        quantizedRows = arena.TakeUncleared<float>((nuint)(f * dim));
        ropeCos = arena.TakeUncleared<float>((nuint)(f * g.QkDim / 2));
        ropeSin = arena.TakeUncleared<float>((nuint)(f * g.QkDim / 2));
        for (var p = 0; p < f; p++)
            for (var i = 0; i < g.QkDim / 2; i++)
            {
                var angle = p / MathF.Pow(g.RopeTheta, 2f * i / g.QkDim);
                ropeCos[p * (g.QkDim / 2) + i] = MathF.Cos(angle);
                ropeSin[p * (g.QkDim / 2) + i] = MathF.Sin(angle);
            }
        streamAct = new Activations(ref arena, f, streamWidth);
        dimAct = new Activations(ref arena, f, dim);
        memoryAct = new Activations(ref arena, f, dim);
        var s = audio->StemWidth;
        stage0 = arena.TakeUncleared<float>((nuint)((4 * StemBlock + 8) * 40 * s));
        stage1 = arena.TakeUncleared<float>((nuint)((2 * StemBlock + 4) * 20 * s));
        stageDepthwise = arena.TakeUncleared<float>((nuint)((2 * StemBlock + 4) * 20 * s));
        stagePointwise = arena.TakeUncleared<float>((nuint)((2 * StemBlock + 4) * 20 * s));
        zeroRow = arena.Take<float>((nuint)(40 * s + 80));
        stemFlat = arena.TakeUncleared<float>((nuint)(StemBlock * 10 * s));
        stemAct = new Activations(ref arena, (2 * StemBlock + 4) * 20, s);
        flatAct = new Activations(ref arena, StemBlock, 10 * s);
        encoderOutput = arena.TakeUncleared<float>((nuint)(f * dim));
        crossKRaw = arena.TakeUncleared<float>((nuint)(f * g.Heads * g.QkDim));
        crossVRaw = arena.TakeUncleared<float>((nuint)(f * g.Heads * g.VDim));
        memory = new CrossMemory
        {
            Keys = arena.TakeUncleared<sbyte>((nuint)(g.Layers * f * g.Heads * g.QkDim)),
            KeyScales = arena.TakeUncleared<float>((nuint)(g.Layers * f * g.Heads)),
            Values = arena.TakeUncleared<sbyte>((nuint)(g.Layers * f * g.Heads * g.VDim)),
            ValueScales = arena.TakeUncleared<float>((nuint)(g.Layers * f * g.Heads)),
        };
    }

    public int Encode(ReadOnlySpan<float> pcm)
    {
        var samples = Math.Min(pcm.Length, MaxSeconds * audio->SampleRate);
        var l0 = samples / audio->Hop;
        frames = 0;
        if (l0 < 1) return 0;
        var half = audio->Window / 2;
        Mem.Clear(padded, (nuint)(samples + 2 * half + audio->Hop + audio->Fft) * 4);
        for (var i = 0; i < samples; i++) padded[half + i] = pcm[i];
        fixed (Frontend* fp = &front)
        {
            var job = new FeatureJob { Front = fp, Padded = padded, Output = features, TaskPeaks = featurePeaks, Frames = l0 };
            FeatureJob.Run(pool, &job);
        }
        Normalize(features, l0, audio->Mels);
        Observer?.Invoke("mel", new ReadOnlySpan<float>(features, l0 * audio->Mels));
        frames = Stem(l0);
        Observer?.Invoke("stem", new ReadOnlySpan<float>(stemOut, frames * dim));
        RunEncoder(frames);
        return frames;
    }

    static void Normalize(float* x, int rows, int cols)
    {
        for (var c = 0; c < cols; c++)
        {
            double sum = 0;
            for (var r = 0; r < rows; r++) sum += x[r * cols + c];
            var mean = sum / rows;
            double squares = 0;
            for (var r = 0; r < rows; r++) squares += (x[r * cols + c] - mean) * (x[r * cols + c] - mean);
            var std = Math.Sqrt(squares / Math.Max(rows - 1, 1));
            for (var r = 0; r < rows; r++) x[r * cols + c] = (float)((x[r * cols + c] - mean) / (std + 1e-5));
        }
    }

    int Stem(int l0)
    {
        var s = audio->StemWidth;
        var l1 = (l0 + 1) / 2;
        var l2 = (l1 + 1) / 2;
        var l3 = (l2 + 1) / 2;
        for (var r0 = 0; r0 < l3; r0 += StemBlock)
        {
            var r1 = Math.Min(l3, r0 + StemBlock);
            var q0 = 2 * r0 - 1;
            var q1 = 2 * r1 + 1;
            var p0 = 2 * q0 - 1;
            var p1 = 2 * (q1 - 1) + 2;
            StageFirst(l0, l1, p0, p1);
            StageSecond(p0, l1, l2, q0, q1);
            StageThird(q0, l2, l3, r0, r1);
        }
        return l3;
    }

    float* Row0(int row, int length) => row >= 0 && row < length ? features + (long)row * audio->Mels : zeroRow;

    void StageFirst(int l0, int l1, int p0, int p1)
    {
        var s = audio->StemWidth;
        for (var p = p0; p < p1; p++)
        {
            var o = stage0 + (long)(p - p0) * 40 * s;
            if (p < 0 || p >= l1) { Mem.Clear(o, (nuint)(40 * s * 4)); continue; }
            for (var f = 0; f < 40; f++)
            {
                Ops.Zero(o + f * s, s);
                for (var dt = 0; dt < 3; dt++)
                {
                    var row = Row0(2 * p - 1 + dt, l0);
                    for (var df = 0; df < 3; df++)
                    {
                        var freq = 2 * f - 1 + df;
                        if (freq < 0 || freq >= audio->Mels) continue;
                        Ops.AddScaled(o + f * s, audio->StemFirst + (dt * 3 + df) * s, row[freq], s);
                    }
                }
            }
            Ops.Silu(o, 40 * s);
        }
    }

    void Depthwise(float* weights, float* r0, float* r1, float* r2, int freqIn, int freqOut, float* output)
    {
        var s = audio->StemWidth;
        var rows = stackalloc float*[3];
        rows[0] = r0;
        rows[1] = r1;
        rows[2] = r2;
        for (var f = 0; f < freqOut; f++)
        {
            var o = output + f * s;
            Ops.Zero(o, s);
            for (var dt = 0; dt < 3; dt++)
                for (var df = 0; df < 3; df++)
                {
                    var freq = 2 * f - 1 + df;
                    if (freq < 0 || freq >= freqIn) continue;
                    Ops.MultiplyAccumulate(o, rows[dt] + freq * s, weights + (dt * 3 + df) * s, s);
                }
        }
    }

    float* Stage(float* buffer, int first, int row, int length, int rowFloats) =>
        row >= 0 && row < length && row >= first ? buffer + (long)(row - first) * rowFloats : zeroRow;

    void Pointwise(in Cq weights, float* source, int tokens, float* destination)
    {
        var s = audio->StemWidth;
        Quant.Prepare(source, tokens, s, 2, stemAct);
        var batch = new MatMulBatch(stemAct, tokens);
        batch.Add(weights, destination, s);
        MatMulBatch.Run(pool, &batch);
    }

    void StageSecond(int p0, int l1, int l2, int q0, int q1)
    {
        var s = audio->StemWidth;
        var rowFloats = 40 * s;
        for (var q = q0; q < q1; q++)
        {
            var o = stageDepthwise + (long)(q - q0) * 20 * s;
            if (q < 0 || q >= l2) { Mem.Clear(o, (nuint)(20 * s * 4)); continue; }
            Depthwise(audio->StemDepthwise1, Stage(stage0, p0, 2 * q - 1, l1, rowFloats), Stage(stage0, p0, 2 * q, l1, rowFloats), Stage(stage0, p0, 2 * q + 1, l1, rowFloats), 40, 20, o);
        }
        Pointwise(audio->StemPointwise1, stageDepthwise, (q1 - q0) * 20, stagePointwise);
        for (var q = q0; q < q1; q++)
        {
            var o = stage1 + (long)(q - q0) * 20 * s;
            if (q < 0 || q >= l2) { Mem.Clear(o, (nuint)(20 * s * 4)); continue; }
            Buffer.MemoryCopy(stagePointwise + (long)(q - q0) * 20 * s, o, 20 * s * 4, 20 * s * 4);
            Ops.Silu(o, 20 * s);
        }
    }

    void StageThird(int q0, int l2, int l3, int r0, int r1)
    {
        var s = audio->StemWidth;
        var rowFloats = 20 * s;
        var rows = r1 - r0;
        for (var r = r0; r < r1; r++)
            Depthwise(audio->StemDepthwise2, Stage(stage1, q0, 2 * r - 1, l2, rowFloats), Stage(stage1, q0, 2 * r, l2, rowFloats), Stage(stage1, q0, 2 * r + 1, l2, rowFloats), 20, 10, stageDepthwise + (long)(r - r0) * 10 * s);
        Pointwise(audio->StemPointwise2, stageDepthwise, rows * 10, stagePointwise);
        for (var r = 0; r < rows; r++)
        {
            var row = stagePointwise + (long)r * 10 * s;
            Ops.Silu(row, 10 * s);
            var flat = stemFlat + (long)r * 10 * s;
            for (var f = 0; f < 10; f++)
                for (var c = 0; c < s; c++) flat[c * 10 + f] = row[f * s + c];
            Quant.FakeA8(flat, 10 * s);
        }
        Quant.Prepare(stemFlat, rows, 10 * s, 2, flatAct);
        var batch = new MatMulBatch(flatAct, rows);
        batch.Add(audio->StemOut, stemOut + (long)r0 * dim, dim);
        MatMulBatch.Run(pool, &batch);
    }

    void RunEncoder(int count)
    {
        var width = 4 * dim;
        for (var t = 0; t < count; t++)
            for (var lane = 0; lane < 4; lane++) Buffer.MemoryCopy(stemOut + (long)t * dim, streamA + (long)t * width + lane * dim, dim * 4, dim * 4);
        var stream = streamA;
        var next = streamB;
        var logits = stackalloc float[16];
        var hpre = stackalloc float[4];
        for (var l = 0; l < audio->EncoderLayers; l++)
        {
            EncoderLayer* w = audio->Encoder + l;
            ref readonly var hyper = ref audio->EncoderHyper;
            var lane = l % 4;
            for (var t = 0; t < count; t++)
            {
                Ops.RmsUnit(stream + (long)t * width, width, normalized + (long)t * width);
                Quant.FakeA8(normalized + (long)t * width, width);
            }
            Quant.Prepare(normalized, count, width, 4, streamAct);
            var mix = new MatMulBatch(streamAct, count);
            mix.Add(hyper.PhiPre.Slice(l * 4, 4), preOut, 4);
            mix.Add(hyper.PhiPost.Slice(l * 4, 4), postOut, 4);
            mix.Add(hyper.PhiRes.Slice(l * 16, 16), resOut, 16);
            MatMulBatch.Run(pool, &mix);
            for (var t = 0; t < count; t++)
            {
                for (var i = 0; i < 4; i++)
                {
                    hpre[i] = Ops.Sigmoid(hyper.APre[l] * preOut[t * 4 + i] + hyper.BPre[l * 4 + i] + (i == lane ? 4f : -4f));
                    postGate[t * 4 + i] = 2f * Ops.Sigmoid(hyper.APost[l] * postOut[t * 4 + i] + hyper.BPost[l * 4 + i] + (i == lane ? 0f : -4f));
                }
                Ops.Mix4(stream + (long)t * width, dim, hpre[0], hpre[1], hpre[2], hpre[3], input + (long)t * dim);
                for (var i = 0; i < 16; i++) logits[i] = hyper.ARes[l] * resOut[t * 16 + i] + hyper.BRes[l * 16 + i];
                Sinkhorn.Normalize(logits, residualMix + t * 16);
                Buffer.MemoryCopy(input + (long)t * dim, residual + (long)t * dim, dim * 4, dim * 4);
            }

            RunMlp(&w->Mlp0, w->PreMlpNorm0, count);
            RunSelfAttention(w, count);
            RunConvolution(w, count);
            RunMlp(&w->Mlp, w->PreMlpNorm, count);

            for (var t = 0; t < count; t++)
            {
                Ops.Subtract(residual + (long)t * dim, input + (long)t * dim, dim, rowScratch);
                for (var i = 0; i < 4; i++)
                    Ops.Combine4(stream + (long)t * width, dim, residualMix[t * 16 + i * 4], residualMix[t * 16 + i * 4 + 1], residualMix[t * 16 + i * 4 + 2], residualMix[t * 16 + i * 4 + 3], rowScratch, postGate[t * 4 + i], next + (long)t * width + i * dim);
            }
            var swap = stream;
            stream = next;
            next = swap;
            Observer?.Invoke("layer" + l, new ReadOnlySpan<float>(stream, count * width));
        }
        for (var t = 0; t < count; t++)
        {
            Ops.Mix4(stream + (long)t * width, dim, 0.25f, 0.25f, 0.25f, 0.25f, rowScratch);
            Ops.ZeroCenteredRms(rowScratch, audio->EncoderFinalNorm, dim, encoderOutput + (long)t * dim);
        }
        Observer?.Invoke("encoder", new ReadOnlySpan<float>(encoderOutput, count * dim));
    }

    void RunMlp(MlpWeights* weights, float* preNorm, int count)
    {
        var batch = new MlpBatch
        {
            Weights = weights, PreNorm = preNorm, PermA = model.Hyper.PermA, PermB = model.Hyper.PermB,
            Residual = residual, Dim = dim, Rows = g.HadaRows, Cols = g.HadaCols, Scale = 0.5f,
        };
        MlpBatch.Run(pool, &batch, count);
    }

    void RunSelfAttention(EncoderLayer* w, int count)
    {
        for (var t = 0; t < count; t++)
        {
            var r = quantizedRows + (long)t * dim;
            Ops.ZeroCenteredRms(residual + (long)t * dim, w->AttnNorm, dim, r);
            Quant.FakeA8(r, dim);
        }
        Quant.Prepare(quantizedRows, count, dim, 2, dimAct);
        var projections = new MatMulBatch(dimAct, count);
        projections.Add(w->Q, qRaw, g.QueryWidth);
        projections.Add(w->K, kRaw, g.KeyWidth);
        projections.Add(w->V, vRaw, g.ValueWidth);
        projections.Add(w->Gate, gateOut, g.OutputWidth);
        MatMulBatch.Run(pool, &projections);
        var half = g.QkDim / 2;
        for (var t = 0; t < count; t++)
        {
            var q = qRaw + (long)t * g.QueryWidth;
            var k = kRaw + (long)t * g.KeyWidth;
            var v = vRaw + (long)t * g.ValueWidth;
            for (var h = 0; h < g.Heads; h++) Ops.ZeroCenteredRms(q + h * g.QkDim, w->QNorm, g.QkDim, q + h * g.QkDim);
            for (var h = 0; h < g.KvHeads; h++) Ops.ZeroCenteredRms(k + h * g.QkDim, w->KNorm, g.QkDim, k + h * g.QkDim);
            Ops.Rope(q, g.Heads, g.QkDim, ropeCos + t * half, ropeSin + t * half);
            Ops.Rope(k, g.KvHeads, g.QkDim, ropeCos + t * half, ropeSin + t * half);
            for (var h = 0; h < g.Heads; h++) qScale[t * g.Heads + h] = Quant.QuantizeInt8(q + h * g.QkDim, g.QkDim, qInt8 + (long)t * g.QueryWidth + h * g.QkDim);
            for (var h = 0; h < g.KvHeads; h++)
            {
                kScale[t * g.KvHeads + h] = Quant.QuantizeInt8(k + h * g.QkDim, g.QkDim, kInt8 + (long)t * g.KeyWidth + h * g.QkDim);
                vScale[t * g.KvHeads + h] = Quant.QuantizeInt8(v + h * g.VDim, g.VDim, vInt8 + (long)t * g.ValueWidth + h * g.VDim);
            }
        }
        var attention = new AttentionBatch
        {
            Queries = qInt8, QueryScales = qScale, Keys = kInt8, KeyScales = kScale, Values = vInt8, ValueScales = vScale,
            Output = attnOut, Heads = g.Heads, KvHeads = g.KvHeads, QkDim = g.QkDim, VDim = g.VDim, Position = 0, Window = 0,
            Capacity = count, Frames = count,
        };
        AttentionBatch.Run(pool, &attention, count);
        for (var t = 0; t < count; t++)
        {
            var o = attnOut + (long)t * g.OutputWidth;
            Ops.MultiplyBySigmoid(o, gateOut + (long)t * g.OutputWidth, g.OutputWidth);
            Quant.FakeA8(o, g.OutputWidth);
        }
        Quant.Prepare(attnOut, count, g.OutputWidth, 2, dimAct);
        var output = new MatMulBatch(dimAct, count);
        output.Add(w->Out, attnProj, dim);
        MatMulBatch.Run(pool, &output);
        for (var t = 0; t < count; t++)
        {
            Ops.ZeroCenteredRms(attnProj + (long)t * dim, w->PostAttnNorm, dim, rowScratch);
            Ops.AddScaled(residual + (long)t * dim, rowScratch, w->AttnGate, dim);
        }
    }

    void RunConvolution(EncoderLayer* w, int count)
    {
        for (var t = 0; t < count; t++)
        {
            var r = quantizedRows + (long)t * dim;
            Ops.ZeroCenteredRms(residual + (long)t * dim, w->ConvNorm, dim, r);
            Quant.FakeA8(r, dim);
        }
        Quant.Prepare(quantizedRows, count, dim, 2, dimAct);
        var expand = new MatMulBatch(dimAct, count);
        expand.Add(w->Pw1, glu, 2 * dim);
        MatMulBatch.Run(pool, &expand);
        for (var t = 0; t < count; t++)
        {
            var row = glu + (long)t * 2 * dim;
            var o = hidden + (long)t * dim;
            Ops.Glu(row, row + dim, o, dim);
        }
        var kernel = audio->ConvKernel;
        var pad = kernel / 2;
        for (var t = 0; t < count; t++)
        {
            var o = conv + (long)t * dim;
            Ops.Zero(o, dim);
            for (var j = 0; j < kernel; j++)
            {
                var source = t + j - pad;
                if (source < 0 || source >= count) continue;
                Ops.MultiplyAccumulate(o, w->Depthwise + j * dim, hidden + (long)source * dim, dim);
            }
            var r = quantizedRows + (long)t * dim;
            Ops.ZeroCenteredRms(o, w->ConvOutNorm, dim, r);
            Ops.Silu(r, dim);
        }
        Quant.Prepare(quantizedRows, count, dim, 2, dimAct);
        var project = new MatMulBatch(dimAct, count);
        project.Add(w->Pw2, attnProj, dim);
        MatMulBatch.Run(pool, &project);
        for (var t = 0; t < count; t++) Ops.AddScaled(residual + (long)t * dim, attnProj + (long)t * dim, 1f, dim);
    }

    public float[] Embed(ReadOnlySpan<float> pcm)
    {
        var count = Encode(pcm);
        var result = new float[count * dim];
        new ReadOnlySpan<float>(encoderOutput, result.Length).CopyTo(result);
        return result;
    }

    void BuildCrossMemory()
    {
        var count = frames;
        var heads = g.Heads;
        var angleStep = -Math.Log(10000.0) / (dim / 2 - 1);
        for (var t = 0; t < count; t++)
        {
            var row = quantizedRows + (long)t * dim;
            for (var i = 0; i < dim / 2; i++)
            {
                var angle = t * Math.Exp(angleStep * i);
                row[i] = encoderOutput[(long)t * dim + i] + audio->PositionMix * (float)Math.Sin(angle);
                row[dim / 2 + i] = encoderOutput[(long)t * dim + dim / 2 + i] + audio->PositionMix * (float)Math.Cos(angle);
            }
            Quant.FakeA8(row, dim);
        }
        Observer?.Invoke("memory", new ReadOnlySpan<float>(quantizedRows, count * dim));
        Quant.Prepare(quantizedRows, count, dim, 2, memoryAct);
        for (var l = 0; l < g.Layers; l++)
        {
            var c = audio->Cross + l;
            var batch = new MatMulBatch(memoryAct, count);
            batch.Add(c->K, crossKRaw, heads * g.QkDim);
            batch.Add(c->V, crossVRaw, heads * g.VDim);
            MatMulBatch.Run(pool, &batch);
            for (var t = 0; t < count; t++)
                for (var h = 0; h < heads; h++)
                {
                    var k = crossKRaw + (long)t * heads * g.QkDim + h * g.QkDim;
                    Ops.ZeroCenteredRms(k, c->KNorm, g.QkDim, k);
                    var slot = (long)l * count * heads + (long)t * heads + h;
                    memory.KeyScales[slot] = Quant.QuantizeInt8(k, g.QkDim, memory.Keys + slot * g.QkDim);
                    memory.ValueScales[slot] = Quant.QuantizeInt8(crossVRaw + (long)t * heads * g.VDim + h * g.VDim, g.VDim, memory.Values + slot * g.VDim);
                }
        }
        memory.Frames = count;
        decoder.Attach(memory);
        if (Observer != null)
        {
            Observer("cross_key0", DequantizedCross(memory.Keys, memory.KeyScales, g.QkDim, 0));
            Observer("cross_value_last", DequantizedCross(memory.Values, memory.ValueScales, g.VDim, g.Layers - 1));
        }
    }

    float[] DequantizedCross(sbyte* values, float* scales, int width, int layer)
    {
        var slots = frames * g.Heads;
        var result = new float[slots * width];
        var first = (long)layer * slots;
        for (var slot = 0; slot < slots; slot++)
            for (var i = 0; i < width; i++) result[slot * width + i] = values[(first + slot) * width + i] * scales[first + slot];
        return result;
    }

    internal void Observe(ReadOnlySpan<float> pcm, string? language)
    {
        if (Encode(pcm) == 0) return;
        EnsureDecoderBuffers();
        BuildCrossMemory();
        Prime(language, false, new Pass());
    }

    public void Pause() => pool.Pause();

    public void Resume() => pool.Resume();

    public void Dispose()
    {
        decoder.Dispose();
        tokenizer.Dispose();
        pool.Dispose();
        front.Free();
        arena.Free();
    }
}
