using System;
using System.Runtime.InteropServices;

namespace Needle;

internal unsafe struct Frontend
{
    public int Fft, Window, Hop, Mels;
    public float* HannWindow, Cosines, Sines, MelBank;
    public int* BitReverse, MelFirst, MelCount;
    public float* Padded;
    public float* Log;
    public int Frames;

    public static Frontend Create(in AudioWeights a)
    {
        var f = new Frontend { Fft = a.Fft, Window = a.Window, Hop = a.Hop, Mels = a.Mels, MelBank = a.Mel };
        f.HannWindow = (float*)Mem.Alloc((nuint)a.Window * 4);
        for (var n = 0; n < a.Window; n++) f.HannWindow[n] = (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * n / (a.Window - 1)));
        f.Cosines = (float*)Mem.Alloc((nuint)a.Fft / 2 * 4);
        f.Sines = (float*)Mem.Alloc((nuint)a.Fft / 2 * 4);
        for (var k = 0; k < a.Fft / 2; k++)
        {
            f.Cosines[k] = (float)Math.Cos(2 * Math.PI * k / a.Fft);
            f.Sines[k] = (float)-Math.Sin(2 * Math.PI * k / a.Fft);
        }
        f.BitReverse = (int*)Mem.Alloc((nuint)a.Fft * 4);
        var bits = BitOperationsLog2(a.Fft);
        for (var i = 0; i < a.Fft; i++)
        {
            var r = 0;
            for (var b = 0; b < bits; b++) r |= (i >> b & 1) << (bits - 1 - b);
            f.BitReverse[i] = r;
        }
        var bins = a.Fft / 2 + 1;
        f.MelFirst = (int*)Mem.Alloc((nuint)bins * 4);
        f.MelCount = (int*)Mem.Alloc((nuint)bins * 4);
        for (var k = 0; k < bins; k++)
        {
            var row = a.Mel + k * a.Mels;
            var first = a.Mels;
            var last = -1;
            for (var m = 0; m < a.Mels; m++)
                if (row[m] != 0f) { if (first == a.Mels) first = m; last = m; }
            f.MelFirst[k] = first;
            f.MelCount[k] = last >= first ? last - first + 1 : 0;
        }
        return f;
    }

    static int BitOperationsLog2(int n) => n.BitLength() - 1;

    public void Free()
    {
        Mem.Free(HannWindow);
        Mem.Free(Cosines);
        Mem.Free(Sines);
        Mem.Free(BitReverse);
        Mem.Free(MelFirst);
        Mem.Free(MelCount);
    }
}

internal unsafe struct FeatureJob
{
    public Frontend* Front;
    public float* Padded;
    public float* Output;
    public float* TaskPeaks;
    public float Floor;
    public int Frames;

    public const int FramesPerTask = 16;
    public const float RelativeFloor = 1e-8f, MinimumFloor = 1e-30f;

    public static int Tasks(int frames) => (frames + FramesPerTask - 1) / FramesPerTask;

    public static void Run(Pool pool, FeatureJob* job)
    {
        var tasks = Tasks(job->Frames);
        pool.Run(&Execute, job, tasks);
        var peak = 0f;
        for (var t = 0; t < tasks; t++) peak = MathF.Max(peak, job->TaskPeaks[t]);
        job->Floor = MathF.Max(peak * RelativeFloor, MinimumFloor);
        pool.Run(&Logarithm, job, tasks);
    }

    static void Logarithm(void* context, int task)
    {
        var job = (FeatureJob*)context;
        var mels = job->Front->Mels;
        var first = task * FramesPerTask * mels;
        var last = Math.Min(job->Frames, (task + 1) * FramesPerTask) * mels;
        for (var i = first; i < last; i++) job->Output[i] = MathF.Log(job->Output[i] + job->Floor);
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void Execute(void* context, int task)
    {
        var job = (FeatureJob*)context;
        var f = job->Front;
        var re = stackalloc float[f->Fft];
        var im = stackalloc float[f->Fft];
        var power = stackalloc float[f->Fft / 2 + 1];
        var mel = stackalloc float[f->Mels];
        var peak = 0f;
        for (var frame = task * FramesPerTask; frame < Math.Min(job->Frames, (task + 1) * FramesPerTask); frame++)
        {
            var segment = job->Padded + (long)frame * f->Hop;
            for (var i = 0; i < f->Fft; i++)
            {
                re[f->BitReverse[i]] = i < f->Window ? segment[i] * f->HannWindow[i] : 0f;
                im[i] = 0f;
            }
            Butterflies(f, re, im);
            for (var k = 0; k <= f->Fft / 2; k++) power[k] = re[k] * re[k] + im[k] * im[k];
            Ops.Zero(mel, f->Mels);
            for (var k = 0; k <= f->Fft / 2; k++)
            {
                var count = f->MelCount[k];
                if (count == 0) continue;
                var first = f->MelFirst[k];
                Ops.AddScaled(mel + first, f->MelBank + k * f->Mels + first, power[k], count);
            }
            var row = job->Output + (long)frame * f->Mels;
            for (var m = 0; m < f->Mels; m++)
            {
                row[m] = mel[m];
                peak = MathF.Max(peak, mel[m]);
            }
        }
        job->TaskPeaks[task] = peak;
    }

    static void Butterflies(Frontend* f, float* re, float* im)
    {
#if NEEDLE_SIMD
        if (Simd.UseAvx)
        {
            var fft = f->Fft;
            for (var i = 0; i < fft; i += 2)
            {
                var ar = re[i];
                var br = re[i + 1];
                re[i] = ar + br;
                re[i + 1] = ar - br;
            }
            var cosPtr = f->Cosines;
            var sinPtr = f->Sines;
            var s4 = sinPtr[fft >> 2];
            for (var start = 0; start < fft; start += 4)
            {
                var r0 = re[start];
                var r2 = re[start + 2];
                re[start] = r0 + r2;
                re[start + 2] = r0 - r2;
                var r1 = re[start + 1];
                var ti = re[start + 3] * s4;
                re[start + 1] = r1;
                re[start + 3] = r1;
                im[start + 1] = ti;
                im[start + 3] = -ti;
            }
            for (var size = 8; size <= fft; size <<= 1)
            {
                var half = size >> 1;
                var step = fft / size;
                if (half >= 8)
                {
                    for (var k = 0; k < half; k += 8)
                    {
                        var kv = k * step;
                        var c = System.Runtime.Intrinsics.Vector256.Create(
                            cosPtr[kv], cosPtr[kv + step], cosPtr[kv + 2 * step], cosPtr[kv + 3 * step],
                            cosPtr[kv + 4 * step], cosPtr[kv + 5 * step], cosPtr[kv + 6 * step], cosPtr[kv + 7 * step]);
                        var s = System.Runtime.Intrinsics.Vector256.Create(
                            sinPtr[kv], sinPtr[kv + step], sinPtr[kv + 2 * step], sinPtr[kv + 3 * step],
                            sinPtr[kv + 4 * step], sinPtr[kv + 5 * step], sinPtr[kv + 6 * step], sinPtr[kv + 7 * step]);
                        for (var start = 0; start < fft; start += size)
                        {
                            var a = start + k;
                            var b = a + half;
                            var rb = System.Runtime.Intrinsics.X86.Avx.LoadVector256(re + b);
                            var ib = System.Runtime.Intrinsics.X86.Avx.LoadVector256(im + b);
                            var ra = System.Runtime.Intrinsics.X86.Avx.LoadVector256(re + a);
                            var ia = System.Runtime.Intrinsics.X86.Avx.LoadVector256(im + a);
                            var tr = System.Runtime.Intrinsics.X86.Fma.MultiplySubtract(rb, c, System.Runtime.Intrinsics.X86.Avx.Multiply(ib, s));
                            var ti = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(rb, s, System.Runtime.Intrinsics.X86.Avx.Multiply(ib, c));
                            System.Runtime.Intrinsics.X86.Avx.Store(re + b, System.Runtime.Intrinsics.X86.Avx.Subtract(ra, tr));
                            System.Runtime.Intrinsics.X86.Avx.Store(im + b, System.Runtime.Intrinsics.X86.Avx.Subtract(ia, ti));
                            System.Runtime.Intrinsics.X86.Avx.Store(re + a, System.Runtime.Intrinsics.X86.Avx.Add(ra, tr));
                            System.Runtime.Intrinsics.X86.Avx.Store(im + a, System.Runtime.Intrinsics.X86.Avx.Add(ia, ti));
                        }
                    }
                }
                else
                {
                    for (var start = 0; start < fft; start += size)
                        for (var k = 0; k < half; k++)
                        {
                            var c = cosPtr[k * step];
                            var s = sinPtr[k * step];
                            var a = start + k;
                            var b = a + half;
                            var tr = re[b] * c - im[b] * s;
                            var ti = re[b] * s + im[b] * c;
                            re[b] = re[a] - tr;
                            im[b] = im[a] - ti;
                            re[a] += tr;
                            im[a] += ti;
                        }
                }
            }
            return;
        }
#endif
        for (var size = 2; size <= f->Fft; size <<= 1)
        {
            var half = size / 2;
            var step = f->Fft / size;
            for (var start = 0; start < f->Fft; start += size)
                for (var k = 0; k < half; k++)
                {
                    var c = f->Cosines[k * step];
                    var s = f->Sines[k * step];
                    var a = start + k;
                    var b = a + half;
                    var tr = re[b] * c - im[b] * s;
                    var ti = re[b] * s + im[b] * c;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
        }
    }
}
