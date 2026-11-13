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
    readonly ThreadPolicy policy;
    readonly string[] languages;
    Arena arena;
    Frontend front;

    float* padded, features, featurePeaks, stemOut;
    float* streamA, streamB, normalized, residual, input, hidden, attnOut, gateOut, attnProj, glu, conv;
    float* preOut, postOut, resOut, postGate, residualMix, rowScratch, quantizedRows;
    sbyte* qInt8, kInt8, vInt8;
    float* qScale, kScale, vScale, qRaw, kRaw, vRaw;
    float* ropeCos, ropeSin, crossPosSin, crossPosCos;
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
        this.policy = policy;
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
        var halfDim = dim / 2;
        crossPosSin = arena.TakeUncleared<float>((nuint)(f * halfDim));
        crossPosCos = arena.TakeUncleared<float>((nuint)(f * halfDim));
        var angleStep = -Math.Log(10000.0) / (halfDim - 1);
        for (var i = 0; i < halfDim; i++)
        {
            var invFreq = Math.Exp(angleStep * i);
            for (var p = 0; p < f; p++)
            {
                var angle = p * invFreq;
                crossPosSin[p * halfDim + i] = (float)Math.Sin(angle);
                crossPosCos[p * halfDim + i] = (float)Math.Cos(angle);
            }
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

    static void SliceRange(int task, int tasks, int total, out int first, out int end)
    {
        first = task * total / tasks;
        end = (task + 1) * total / tasks;
    }

    struct NormalizeJob
    {
        public float* X;
        public int Rows, Cols, Tasks;
    }

    static void ExecuteNormalize(void* context, int task)
    {
        var j = (NormalizeJob*)context;
        SliceRange(task, j->Tasks, j->Cols, out var first, out var end);
        var rows = j->Rows;
        var cols = j->Cols;
        var denom = Math.Max(rows - 1, 1);
        var c = first;
#if NEEDLE_SIMD
        if (Simd.UseAvx && rows > 0)
        {
            var invRows = System.Runtime.Intrinsics.Vector256.Create(1.0 / rows);
            var invDenom = System.Runtime.Intrinsics.Vector256.Create(1.0 / denom);
            var eps = System.Runtime.Intrinsics.Vector256.Create(1e-5);
            for (; c + 8 <= end; c += 8)
            {
                var sum0 = System.Runtime.Intrinsics.Vector256<double>.Zero;
                var sum1 = System.Runtime.Intrinsics.Vector256<double>.Zero;
                var p = j->X + c;
                for (var r = 0; r < rows; r++, p += cols)
                {
                    sum0 = System.Runtime.Intrinsics.X86.Avx.Add(sum0, System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse.LoadVector128(p)));
                    sum1 = System.Runtime.Intrinsics.X86.Avx.Add(sum1, System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse.LoadVector128(p + 4)));
                }
                var mean0 = System.Runtime.Intrinsics.X86.Avx.Multiply(sum0, invRows);
                var mean1 = System.Runtime.Intrinsics.X86.Avx.Multiply(sum1, invRows);
                var sq0 = System.Runtime.Intrinsics.Vector256<double>.Zero;
                var sq1 = System.Runtime.Intrinsics.Vector256<double>.Zero;
                p = j->X + c;
                for (var r = 0; r < rows; r++, p += cols)
                {
                    var d0 = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse.LoadVector128(p)), mean0);
                    var d1 = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse.LoadVector128(p + 4)), mean1);
                    sq0 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d0, d0, sq0);
                    sq1 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d1, d1, sq1);
                }
                var invStd0 = System.Runtime.Intrinsics.X86.Avx.Divide(System.Runtime.Intrinsics.Vector256.Create(1.0), System.Runtime.Intrinsics.X86.Avx.Add(System.Runtime.Intrinsics.X86.Avx.Sqrt(System.Runtime.Intrinsics.X86.Avx.Multiply(sq0, invDenom)), eps));
                var invStd1 = System.Runtime.Intrinsics.X86.Avx.Divide(System.Runtime.Intrinsics.Vector256.Create(1.0), System.Runtime.Intrinsics.X86.Avx.Add(System.Runtime.Intrinsics.X86.Avx.Sqrt(System.Runtime.Intrinsics.X86.Avx.Multiply(sq1, invDenom)), eps));
                p = j->X + c;
                for (var r = 0; r < rows; r++, p += cols)
                {
                    var n0 = System.Runtime.Intrinsics.X86.Avx.Multiply(System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse.LoadVector128(p)), mean0), invStd0);
                    var n1 = System.Runtime.Intrinsics.X86.Avx.Multiply(System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.ConvertToVector256Double(System.Runtime.Intrinsics.X86.Sse.LoadVector128(p + 4)), mean1), invStd1);
                    System.Runtime.Intrinsics.X86.Sse.Store(p, System.Runtime.Intrinsics.X86.Avx.ConvertToVector128Single(n0));
                    System.Runtime.Intrinsics.X86.Sse.Store(p + 4, System.Runtime.Intrinsics.X86.Avx.ConvertToVector128Single(n1));
                }
            }
        }
#endif
        for (; c < end; c++)
        {
            double sum = 0;
            for (var r = 0; r < rows; r++) sum += j->X[r * cols + c];
            var mean = sum / rows;
            double squares = 0;
            for (var r = 0; r < rows; r++) squares += (j->X[r * cols + c] - mean) * (j->X[r * cols + c] - mean);
            var std = Math.Sqrt(squares / denom);
            for (var r = 0; r < rows; r++) j->X[r * cols + c] = (float)((j->X[r * cols + c] - mean) / (std + 1e-5));
        }
    }

    void Normalize(float* x, int rows, int cols)
    {
        var tasks = pool.Members <= 1 ? 1 : Math.Min(cols, pool.Members * 4);
        var job = new NormalizeJob { X = x, Rows = rows, Cols = cols, Tasks = tasks };
        pool.Run(&ExecuteNormalize, &job, tasks);
    }

    int Stem(int l0)
    {
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

    struct StemJob
    {
        public AudioWeights* Audio;
        public float* Features, ZeroRow, Stage0, Stage1, StageDepthwise, StagePointwise, StemFlat;
        public int L0, L1, L2, L3, P0, Q0, R0, TotalCount, Tasks;
    }

    void RunStem(delegate*<void*, int, void> fn, int l0, int l1, int l2, int l3, int p0, int q0, int r0, int count)
    {
        var tasks = pool.Members <= 1 ? 1 : Math.Min(count, pool.Members * 4);
        var job = new StemJob
        {
            Audio = audio, Features = features, ZeroRow = zeroRow,
            Stage0 = stage0, Stage1 = stage1, StageDepthwise = stageDepthwise,
            StagePointwise = stagePointwise, StemFlat = stemFlat,
            L0 = l0, L1 = l1, L2 = l2, L3 = l3, P0 = p0, Q0 = q0, R0 = r0,
            TotalCount = count, Tasks = tasks,
        };
        pool.Run(fn, &job, tasks);
    }

    static float* Stage(float* buffer, int first, int row, int length, int rowFloats, float* zero) =>
        row >= 0 && row < length && row >= first ? buffer + (long)(row - first) * rowFloats : zero;

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void Depthwise(int s, float* zero, float* weights, float* r0, float* r1, float* r2, int freqIn, int freqOut, float* output)
    {
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

    static void ExecuteStageFirst(void* context, int task)
    {
        var j = (StemJob*)context;
        var a = j->Audio;
        var s = a->StemWidth;
        var mels = a->Mels;
        SliceRange(task, j->Tasks, j->TotalCount, out var first, out var end);
        for (var idx = first; idx < end; idx++)
        {
            var p = j->P0 + idx;
            var o = j->Stage0 + (long)idx * 40 * s;
            if (p < 0 || p >= j->L1) { Mem.Clear(o, (nuint)(40 * s * 4)); continue; }
            for (var f = 0; f < 40; f++)
            {
                Ops.Zero(o + f * s, s);
                for (var dt = 0; dt < 3; dt++)
                {
                    var r = 2 * p - 1 + dt;
                    var row = r >= 0 && r < j->L0 ? j->Features + (long)r * mels : j->ZeroRow;
                    for (var df = 0; df < 3; df++)
                    {
                        var freq = 2 * f - 1 + df;
                        if (freq < 0 || freq >= mels) continue;
                        Ops.AddScaled(o + f * s, a->StemFirst + (dt * 3 + df) * s, row[freq], s);
                    }
                }
            }
            Ops.Silu(o, 40 * s);
        }
    }

    static void ExecuteStageSecondDepthwise(void* context, int task)
    {
        var j = (StemJob*)context;
        var s = j->Audio->StemWidth;
        var rowFloats = 40 * s;
        SliceRange(task, j->Tasks, j->TotalCount, out var first, out var end);
        for (var idx = first; idx < end; idx++)
        {
            var q = j->Q0 + idx;
            var o = j->StageDepthwise + (long)idx * 20 * s;
            if (q < 0 || q >= j->L2) { Mem.Clear(o, (nuint)(20 * s * 4)); continue; }
            Depthwise(s, j->ZeroRow, j->Audio->StemDepthwise1,
                Stage(j->Stage0, j->P0, 2 * q - 1, j->L1, rowFloats, j->ZeroRow),
                Stage(j->Stage0, j->P0, 2 * q, j->L1, rowFloats, j->ZeroRow),
                Stage(j->Stage0, j->P0, 2 * q + 1, j->L1, rowFloats, j->ZeroRow), 40, 20, o);
        }
    }

    static void ExecuteStageSecondSilu(void* context, int task)
    {
        var j = (StemJob*)context;
        var s = j->Audio->StemWidth;
        SliceRange(task, j->Tasks, j->TotalCount, out var first, out var end);
        for (var idx = first; idx < end; idx++)
        {
            var q = j->Q0 + idx;
            var o = j->Stage1 + (long)idx * 20 * s;
            if (q < 0 || q >= j->L2) { Mem.Clear(o, (nuint)(20 * s * 4)); continue; }
            Buffer.MemoryCopy(j->StagePointwise + (long)idx * 20 * s, o, 20 * s * 4, 20 * s * 4);
            Ops.Silu(o, 20 * s);
        }
    }

    static void ExecuteStageThirdDepthwise(void* context, int task)
    {
        var j = (StemJob*)context;
        var s = j->Audio->StemWidth;
        var rowFloats = 20 * s;
        SliceRange(task, j->Tasks, j->TotalCount, out var first, out var end);
        for (var idx = first; idx < end; idx++)
        {
            var r = j->R0 + idx;
            Depthwise(s, j->ZeroRow, j->Audio->StemDepthwise2,
                Stage(j->Stage1, j->Q0, 2 * r - 1, j->L2, rowFloats, j->ZeroRow),
                Stage(j->Stage1, j->Q0, 2 * r, j->L2, rowFloats, j->ZeroRow),
                Stage(j->Stage1, j->Q0, 2 * r + 1, j->L2, rowFloats, j->ZeroRow),
                20, 10, j->StageDepthwise + (long)idx * 10 * s);
        }
    }

    static void ExecuteStageThirdTranspose(void* context, int task)
    {
        var j = (StemJob*)context;
        var s = j->Audio->StemWidth;
        SliceRange(task, j->Tasks, j->TotalCount, out var first, out var end);
        for (var r = first; r < end; r++)
        {
            var row = j->StagePointwise + (long)r * 10 * s;
            Ops.Silu(row, 10 * s);
            var flat = j->StemFlat + (long)r * 10 * s;
            for (var f = 0; f < 10; f++)
                for (var c = 0; c < s; c++) flat[c * 10 + f] = row[f * s + c];
        }
    }

    void StageFirst(int l0, int l1, int p0, int p1) =>
        RunStem(&ExecuteStageFirst, l0, l1, 0, 0, p0, 0, 0, p1 - p0);

    void Pointwise(in Cq weights, float* source, int tokens, float* destination)
    {
        var s = audio->StemWidth;
        Quant.Prepare(pool, source, tokens, s, 2, stemAct);
        var batch = new MatMulBatch(stemAct, tokens);
        batch.Add(weights, destination, s);
        MatMulBatch.Run(pool, &batch);
    }

    void StageSecond(int p0, int l1, int l2, int q0, int q1)
    {
        var s = audio->StemWidth;
        var rows = q1 - q0;
        RunStem(&ExecuteStageSecondDepthwise, 0, l1, l2, 0, p0, q0, 0, rows);
        Pointwise(audio->StemPointwise1, stageDepthwise, rows * 20, stagePointwise);
        RunStem(&ExecuteStageSecondSilu, 0, l1, l2, 0, p0, q0, 0, rows);
    }

    void StageThird(int q0, int l2, int l3, int r0, int r1)
    {
        var s = audio->StemWidth;
        var rows = r1 - r0;
        RunStem(&ExecuteStageThirdDepthwise, 0, 0, l2, l3, 0, q0, r0, rows);
        Pointwise(audio->StemPointwise2, stageDepthwise, rows * 10, stagePointwise);
        RunStem(&ExecuteStageThirdTranspose, 0, 0, l2, l3, 0, q0, r0, rows);
        Quant.Prepare(pool, stemFlat, rows, 10 * s, 2, flatAct);
        var batch = new MatMulBatch(flatAct, rows);
        batch.Add(audio->StemOut, stemOut + (long)r0 * dim, dim);
        MatMulBatch.Run(pool, &batch);
    }

    struct FrameJob
    {
        public AudioWeights* Audio;
        public EncoderLayer* Encoder;
        public CrossMemory* Memory;
        public Activations StreamAct, DimAct, MemoryAct;
        public float* StemOut, Stream, Next, Normalized, Residual, Input, Hidden, AttnOut, GateOut, AttnProj, Glu, Conv;
        public float* PreOut, PostOut, ResOut, PostGate, ResidualMix, QuantizedRows, EncoderOutput;
        public float* QRaw, KRaw, VRaw, QScale, KScale, VScale, RopeCos, RopeSin;
        public sbyte* QInt8, KInt8, VInt8;
        public float* CrossPosSin, CrossPosCos, CrossKRaw, CrossVRaw;
        public int Dim, Width, Heads, KvHeads, QkDim, VDim, OutputWidth, QueryWidth, KeyWidth, ValueWidth, ConvKernel;
        public int Layer, Layers, Count, Tasks;
    }

    FrameJob CreateFrameJob(int count) => new()
    {
        Audio = audio,
        StreamAct = streamAct,
        DimAct = dimAct,
        MemoryAct = memoryAct,
        StemOut = stemOut,
        Normalized = normalized,
        Residual = residual,
        Input = input,
        Hidden = hidden,
        AttnOut = attnOut,
        GateOut = gateOut,
        AttnProj = attnProj,
        Glu = glu,
        Conv = conv,
        PreOut = preOut,
        PostOut = postOut,
        ResOut = resOut,
        PostGate = postGate,
        ResidualMix = residualMix,
        QuantizedRows = quantizedRows,
        EncoderOutput = encoderOutput,
        QRaw = qRaw,
        KRaw = kRaw,
        VRaw = vRaw,
        QInt8 = qInt8,
        KInt8 = kInt8,
        VInt8 = vInt8,
        QScale = qScale,
        KScale = kScale,
        VScale = vScale,
        RopeCos = ropeCos,
        RopeSin = ropeSin,
        CrossPosSin = crossPosSin,
        CrossPosCos = crossPosCos,
        CrossKRaw = crossKRaw,
        CrossVRaw = crossVRaw,
        Dim = dim,
        Width = 4 * dim,
        Heads = g.Heads,
        KvHeads = g.KvHeads,
        QkDim = g.QkDim,
        VDim = g.VDim,
        OutputWidth = g.OutputWidth,
        QueryWidth = g.QueryWidth,
        KeyWidth = g.KeyWidth,
        ValueWidth = g.ValueWidth,
        ConvKernel = audio->ConvKernel,
        Layers = g.Layers,
        Count = count,
    };

    void RunFrames(delegate*<void*, int, void> fn, ref FrameJob job, int count)
    {
        var tasks = pool.Members <= 1 ? 1 : Math.Min(count, pool.Members * 4);
        job.Count = count;
        job.Tasks = tasks;
        fixed (FrameJob* jp = &job)
            pool.Run(fn, jp, tasks);
    }

    static void ExecuteCopyLanes(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var dim = j->Dim;
        var width = j->Width;
        for (var t = first; t < end; t++)
            for (var lane = 0; lane < 4; lane++)
                Buffer.MemoryCopy(j->StemOut + (long)t * dim, j->Stream + (long)t * width + lane * dim, dim * 4, dim * 4);
    }

    static void ExecuteRmsAndPrepareStream(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var width = j->Width;
        for (var t = first; t < end; t++)
        {
            var dst = j->Normalized + (long)t * width;
            Ops.RmsUnit(j->Stream + (long)t * width, width, dst);
            Quant.Prepare(dst, 1, width, 4, j->StreamAct.Slice(t));
        }
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void ExecutePreHyper(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var l = j->Layer;
        var lane = l % 4;
        var dim = j->Dim;
        var width = j->Width;
        ref readonly var hyper = ref j->Audio->EncoderHyper;
        var logits = stackalloc float[16];
        var hpre = stackalloc float[4];
        for (var t = first; t < end; t++)
        {
            for (var i = 0; i < 4; i++)
            {
                hpre[i] = Ops.Sigmoid(hyper.APre[l] * j->PreOut[t * 4 + i] + hyper.BPre[l * 4 + i] + (i == lane ? 4f : -4f));
                j->PostGate[t * 4 + i] = 2f * Ops.Sigmoid(hyper.APost[l] * j->PostOut[t * 4 + i] + hyper.BPost[l * 4 + i] + (i == lane ? 0f : -4f));
            }
            Ops.Mix4(j->Stream + (long)t * width, dim, hpre[0], hpre[1], hpre[2], hpre[3], j->Input + (long)t * dim);
            for (var i = 0; i < 16; i++) logits[i] = hyper.ARes[l] * j->ResOut[t * 16 + i] + hyper.BRes[l * 16 + i];
            Ops.Sinkhorn4x4(logits, j->ResidualMix + t * 16);
            Buffer.MemoryCopy(j->Input + (long)t * dim, j->Residual + (long)t * dim, dim * 4, dim * 4);
        }
    }

    static void ExecuteCombine(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var dim = j->Dim;
        var width = j->Width;
        for (var t = first; t < end; t++)
            Ops.CombineDiff4x4(j->Stream + (long)t * width, dim, j->ResidualMix + t * 16, j->Residual + (long)t * dim, j->Input + (long)t * dim, j->PostGate + t * 4, j->Next + (long)t * width);
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void ExecuteEncoderFinal(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var dim = j->Dim;
        var width = j->Width;
        var scratch = stackalloc float[dim];
        for (var t = first; t < end; t++)
        {
            Ops.Mix4(j->Stream + (long)t * width, dim, 0.25f, 0.25f, 0.25f, 0.25f, scratch);
            Ops.ZeroCenteredRms(scratch, j->Audio->EncoderFinalNorm, dim, j->EncoderOutput + (long)t * dim);
        }
    }

    static void ExecuteAttnNormPrepare(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var dim = j->Dim;
        var norm = j->Encoder->AttnNorm;
        for (var t = first; t < end; t++)
        {
            var r = j->QuantizedRows + (long)t * dim;
            Ops.ZeroCenteredRms(j->Residual + (long)t * dim, norm, dim, r);
            Quant.Prepare(r, 1, dim, 2, j->DimAct.Slice(t));
        }
    }

    static void ExecuteAttnRopeQuantize(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var heads = j->Heads;
        var kvHeads = j->KvHeads;
        var qkDim = j->QkDim;
        var vDim = j->VDim;
        var half = qkDim / 2;
        var w = j->Encoder;
        for (var t = first; t < end; t++)
        {
            var q = j->QRaw + (long)t * j->QueryWidth;
            var k = j->KRaw + (long)t * j->KeyWidth;
            var v = j->VRaw + (long)t * j->ValueWidth;
            for (var h = 0; h < heads; h++) Ops.ZeroCenteredRms(q + h * qkDim, w->QNorm, qkDim, q + h * qkDim);
            for (var h = 0; h < kvHeads; h++) Ops.ZeroCenteredRms(k + h * qkDim, w->KNorm, qkDim, k + h * qkDim);
            Ops.Rope(q, heads, qkDim, j->RopeCos + (long)t * half, j->RopeSin + (long)t * half);
            Ops.Rope(k, kvHeads, qkDim, j->RopeCos + (long)t * half, j->RopeSin + (long)t * half);
            for (var h = 0; h < heads; h++)
                j->QScale[t * heads + h] = Quant.QuantizeInt8(q + h * qkDim, qkDim, j->QInt8 + (long)t * j->QueryWidth + h * qkDim);
            for (var h = 0; h < kvHeads; h++)
            {
                j->KScale[t * kvHeads + h] = Quant.QuantizeInt8(k + h * qkDim, qkDim, j->KInt8 + (long)t * j->KeyWidth + h * qkDim);
                j->VScale[t * kvHeads + h] = Quant.QuantizeInt8(v + h * vDim, vDim, j->VInt8 + (long)t * j->ValueWidth + h * vDim);
            }
        }
    }

    static void ExecuteAttnGatePrepare(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var width = j->OutputWidth;
        for (var t = first; t < end; t++)
        {
            var o = j->AttnOut + (long)t * width;
            Ops.MultiplyBySigmoid(o, j->GateOut + (long)t * width, width);
            Quant.Prepare(o, 1, width, 2, j->DimAct.Slice(t));
        }
    }

    static void ExecuteAttnPostNorm(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var dim = j->Dim;
        var w = j->Encoder;
        for (var t = first; t < end; t++)
            Ops.PostNormAdd(j->Residual + (long)t * dim, j->AttnProj + (long)t * dim, w->PostAttnNorm, w->AttnGate, dim);
    }

    static void ExecuteConvNormPrepare(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var dim = j->Dim;
        var norm = j->Encoder->ConvNorm;
        for (var t = first; t < end; t++)
        {
            var r = j->QuantizedRows + (long)t * dim;
            Ops.ZeroCenteredRms(j->Residual + (long)t * dim, norm, dim, r);
            Quant.Prepare(r, 1, dim, 2, j->DimAct.Slice(t));
        }
    }

    static void ExecuteConvGlu(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var dim = j->Dim;
        for (var t = first; t < end; t++)
        {
            var row = j->Glu + (long)t * 2 * dim;
            Ops.Glu(row, row + dim, j->Hidden + (long)t * dim, dim);
        }
    }

    static void ExecuteConvDepthwisePrepare(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var dim = j->Dim;
        var kernel = j->ConvKernel;
        var pad = kernel / 2;
        var count = j->Count;
        var w = j->Encoder;
        for (var t = first; t < end; t++)
        {
            var o = j->Conv + (long)t * dim;
            Ops.Zero(o, dim);
            for (var k = 0; k < kernel; k++)
            {
                var source = t + k - pad;
                if (source < 0 || source >= count) continue;
                Ops.MultiplyAccumulate(o, w->Depthwise + k * dim, j->Hidden + (long)source * dim, dim);
            }
            var r = j->QuantizedRows + (long)t * dim;
            Ops.ZeroCenteredRms(o, w->ConvOutNorm, dim, r);
            Ops.Silu(r, dim);
            Quant.Prepare(r, 1, dim, 2, j->DimAct.Slice(t));
        }
    }

    static void ExecuteConvAdd(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var dim = j->Dim;
        for (var t = first; t < end; t++)
            Ops.AddScaled(j->Residual + (long)t * dim, j->AttnProj + (long)t * dim, 1f, dim);
    }

    static void ExecuteCrossPrepare(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var dim = j->Dim;
        var half = dim / 2;
        var mix = j->Audio->PositionMix;
        for (var t = first; t < end; t++)
        {
            var row = j->QuantizedRows + (long)t * dim;
            Buffer.MemoryCopy(j->EncoderOutput + (long)t * dim, row, dim * 4, dim * 4);
            Ops.AddScaled(row, j->CrossPosSin + (long)t * half, mix, half);
            Ops.AddScaled(row + half, j->CrossPosCos + (long)t * half, mix, half);
            Quant.FakeA8(row, dim);
            Quant.Prepare(row, 1, dim, 2, j->MemoryAct.Slice(t));
        }
    }

    static void ExecuteCrossQuantize(void* context, int task)
    {
        var j = (FrameJob*)context;
        SliceRange(task, j->Tasks, j->Count, out var first, out var end);
        var count = j->Count;
        var heads = j->Heads;
        var qkDim = j->QkDim;
        var vDim = j->VDim;
        var c = j->Audio->Cross + j->Layer;
        var mem = j->Memory;
        for (var t = first; t < end; t++)
            for (var h = 0; h < heads; h++)
            {
                var k = j->CrossKRaw + (long)t * heads * qkDim + h * qkDim;
                Ops.ZeroCenteredRms(k, c->KNorm, qkDim, k);
                var slot = (long)j->Layer * count * heads + (long)t * heads + h;
                mem->KeyScales[slot] = Quant.QuantizeInt8(k, qkDim, mem->Keys + slot * qkDim);
                mem->ValueScales[slot] = Quant.QuantizeInt8(j->CrossVRaw + (long)t * heads * vDim + h * vDim, vDim, mem->Values + slot * vDim);
            }
    }

    void RunEncoder(int count)
    {
        var width = 4 * dim;
        var job = CreateFrameJob(count);
        job.Stream = streamA;
        job.Next = streamB;
        RunFrames(&ExecuteCopyLanes, ref job, count);
        for (var l = 0; l < audio->EncoderLayers; l++)
        {
            EncoderLayer* w = audio->Encoder + l;
            job.Layer = l;
            job.Encoder = w;
            RunFrames(&ExecuteRmsAndPrepareStream, ref job, count);
            ref readonly var hyper = ref audio->EncoderHyper;
            var mix = new MatMulBatch(streamAct, count);
            mix.Add(hyper.PhiPre.Slice(l * 4, 4), preOut, 4);
            mix.Add(hyper.PhiPost.Slice(l * 4, 4), postOut, 4);
            mix.Add(hyper.PhiRes.Slice(l * 16, 16), resOut, 16);
            MatMulBatch.Run(pool, &mix);
            RunFrames(&ExecutePreHyper, ref job, count);

            RunMlp(&w->Mlp0, w->PreMlpNorm0, count);
            RunSelfAttention(w, ref job, count);
            RunConvolution(w, ref job, count);
            RunMlp(&w->Mlp, w->PreMlpNorm, count);

            RunFrames(&ExecuteCombine, ref job, count);
            var swap = job.Stream;
            job.Stream = job.Next;
            job.Next = swap;
            Observer?.Invoke("layer" + l, new ReadOnlySpan<float>(job.Stream, count * width));
        }
        RunFrames(&ExecuteEncoderFinal, ref job, count);
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

    void RunSelfAttention(EncoderLayer* w, ref FrameJob job, int count)
    {
        RunFrames(&ExecuteAttnNormPrepare, ref job, count);
        var projections = new MatMulBatch(dimAct, count);
        projections.Add(w->Q, qRaw, g.QueryWidth);
        projections.Add(w->K, kRaw, g.KeyWidth);
        projections.Add(w->V, vRaw, g.ValueWidth);
        projections.Add(w->Gate, gateOut, g.OutputWidth);
        MatMulBatch.Run(pool, &projections);
        RunFrames(&ExecuteAttnRopeQuantize, ref job, count);
        var attention = new AttentionBatch
        {
            Queries = qInt8, QueryScales = qScale, Keys = kInt8, KeyScales = kScale, Values = vInt8, ValueScales = vScale,
            Output = attnOut, Heads = g.Heads, KvHeads = g.KvHeads, QkDim = g.QkDim, VDim = g.VDim, Position = 0, Window = 0,
            Capacity = count, Frames = count,
        };
        AttentionBatch.Run(pool, &attention, count);
        RunFrames(&ExecuteAttnGatePrepare, ref job, count);
        var output = new MatMulBatch(dimAct, count);
        output.Add(w->Out, attnProj, dim);
        MatMulBatch.Run(pool, &output);
        RunFrames(&ExecuteAttnPostNorm, ref job, count);
    }

    void RunConvolution(EncoderLayer* w, ref FrameJob job, int count)
    {
        RunFrames(&ExecuteConvNormPrepare, ref job, count);
        var expand = new MatMulBatch(dimAct, count);
        expand.Add(w->Pw1, glu, 2 * dim);
        MatMulBatch.Run(pool, &expand);
        RunFrames(&ExecuteConvGlu, ref job, count);
        RunFrames(&ExecuteConvDepthwisePrepare, ref job, count);
        var project = new MatMulBatch(dimAct, count);
        project.Add(w->Pw2, attnProj, dim);
        MatMulBatch.Run(pool, &project);
        RunFrames(&ExecuteConvAdd, ref job, count);
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
        var job = CreateFrameJob(count);
        fixed (CrossMemory* mp = &memory)
        {
            job.Memory = mp;
            RunFrames(&ExecuteCrossPrepare, ref job, count);
            Observer?.Invoke("memory", new ReadOnlySpan<float>(quantizedRows, count * dim));
            for (var l = 0; l < g.Layers; l++)
            {
                var c = audio->Cross + l;
                var batch = new MatMulBatch(memoryAct, count);
                batch.Add(c->K, crossKRaw, heads * g.QkDim);
                batch.Add(c->V, crossVRaw, heads * g.VDim);
                MatMulBatch.Run(pool, &batch);
                job.Layer = l;
                RunFrames(&ExecuteCrossQuantize, ref job, count);
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
        DisposeBeams();
        tokenizer.Dispose();
        pool.Dispose();
        front.Free();
        arena.Free();
    }
}
