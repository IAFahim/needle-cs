using System;
using System.Runtime.InteropServices;

namespace Needle;

internal unsafe struct Frontend
{
    public int Fft, Window, Hop, Mels;
    public float* HannWindow, Cosines, Sines, MelBank;
    public int* BitReverse;
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
        return f;
    }

    static int BitOperationsLog2(int n) => n.BitLength() - 1;

    public void Free()
    {
        Mem.Free(HannWindow);
        Mem.Free(Cosines);
        Mem.Free(Sines);
        Mem.Free(BitReverse);
    }
}

internal unsafe struct FeatureJob
{
    public Frontend* Front;
    public float* Padded;
    public float* Output;
    public int Frames;

    public const int FramesPerTask = 16;

    public static void Run(Pool pool, FeatureJob* job) => pool.Run(&Execute, job, (job->Frames + FramesPerTask - 1) / FramesPerTask);

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void Execute(void* context, int task)
    {
        var job = (FeatureJob*)context;
        var f = job->Front;
        var re = stackalloc float[f->Fft];
        var im = stackalloc float[f->Fft];
        var power = stackalloc float[f->Fft / 2 + 1];
        var mel = stackalloc float[f->Mels];
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
            for (var k = 0; k <= f->Fft / 2; k++) Ops.AddScaled(mel, f->MelBank + k * f->Mels, power[k], f->Mels);
            var row = job->Output + (long)frame * f->Mels;
            for (var m = 0; m < f->Mels; m++) row[m] = MathF.Log(mel[m] + 5.9604645e-8f);
        }
    }

    static void Butterflies(Frontend* f, float* re, float* im)
    {
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
