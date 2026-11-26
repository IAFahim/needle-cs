using System;
using System.Runtime.CompilerServices;
using System.Threading;
#if NEEDLE_SIMD
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace Needle;

internal readonly unsafe struct Activations
{
    public readonly sbyte* Values;
    public readonly float* Scales;
    public readonly float* Offsets;
    public readonly int Cols;
    public readonly int Groups;
    public readonly int Stride;

    public Activations(sbyte* values, float* scales, float* offsets, int cols)
    {
        Values = values;
        Scales = scales;
        Offsets = offsets;
        Cols = cols;
        Groups = cols / Cq.GroupSize;
        Stride = (Groups + 7) & ~7;
    }

    public Activations Slice(int token) =>
        new(Values + (long)token * Cols, Scales + token * Stride, Offsets + token * Stride, Cols);

    public Activations(ref Arena arena, int tokens, int cols)
    {
        Cols = cols;
        Groups = cols / Cq.GroupSize;
        Stride = (Groups + 7) & ~7;
        Values = arena.Take<sbyte>((nuint)(tokens * cols));
        Scales = arena.Take<float>((nuint)(tokens * Stride));
        Offsets = arena.Take<float>((nuint)(tokens * Stride));
    }
}

[SkipLocalsInit]
internal static unsafe class Quant
{
    const float HadamardScale = 0.08838834764831845f;
    const int TokenTile = 4;
    internal const int RowBlock = 4;

    public static float AbsMax(float* x, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { return Neon.AbsMax(x, n); }
#endif
        if (!Simd.Enabled) return Scalar.AbsMax(x, n);
#if NEEDLE_SIMD
        var peak = Vector256<float>.Zero;
        var i = 0;
        for (; i + 8 <= n; i += 8) peak = Avx.Max(peak, Vector256.Abs(Avx.LoadVector256(x + i)));
        var folded = Sse.Max(peak.GetLower(), peak.GetUpper());
        folded = Sse.Max(folded, Sse.Shuffle(folded, folded, 0b01_00_11_10));
        var best = Sse.Max(folded, Sse.Shuffle(folded, folded, 0b10_11_00_01)).ToScalar();
        for (; i < n; i++) best = MathF.Max(best, MathF.Abs(x[i]));
        return best;
#endif
    }

    public static float FakeA8(float* x, int n)
    {
        var kernelFp = Kernels.Active.FakeA8;
        if ((void*)kernelFp != null)
        {
            return InvokeFakeA8(kernelFp, x, n);
        }
        var peak = AbsMax(x, n);
        var scale = peak > 0f ? peak / 127f : 1f;
        RoundTrip(x, n, 1f / scale, scale);
        return scale;
    }

    public static void RoundTrip(float* x, int n, float inverse, float scale)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.RoundTrip(x, n, inverse, scale); return; }
#endif
        if (!Simd.Enabled) { Scalar.RoundTrip(x, n, inverse, scale); return; }
#if NEEDLE_SIMD
        var inv = Vector256.Create(inverse);
        var sc = Vector256.Create(scale);
        var low = Vector256.Create(-128f);
        var high = Vector256.Create(127f);
        var i = 0;
        for (; i + 8 <= n; i += 8)
        {
            var q = Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(x + i), inv));
            Avx.Store(x + i, Avx.Multiply(Avx.Min(Avx.Max(q, low), high), sc));
        }
        for (; i < n; i++) x[i] = MathF.Min(MathF.Max(MathF.Round(x[i] * inverse), -128f), 127f) * scale;
#endif
    }

    public static float QuantizeInt8(float* x, int n, sbyte* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { return Neon.QuantizeInt8(x, n, destination); }
#endif
        var kernelFp = Kernels.Active.QuantizeInt8;
        if ((void*)kernelFp != null)
        {
            return InvokeQuantizeInt8(kernelFp, x, n, destination);
        }
        if (!Simd.Enabled) return Scalar.QuantizeInt8(x, n, destination);
#if NEEDLE_SIMD
        var peak = AbsMax(x, n);
        var scale = peak > 0f ? peak / 127f : 1f;
        var scalarInv = 1f / scale;
        var inv = Vector256.Create(scalarInv);
        var i = 0;
        var interleave = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
        for (; i + 32 <= n; i += 32)
            Avx.Store(destination + i, Pack32(x + i, inv, interleave));
        for (; i + 8 <= n; i += 8)
        {
            var q = Avx.ConvertToVector256Int32(Avx.Multiply(Avx.LoadVector256(x + i), inv));
            var packed = Sse2.PackSignedSaturate(q.GetLower(), q.GetUpper());
            *(long*)(destination + i) = Sse2.PackSignedSaturate(packed, packed).AsInt64().ToScalar();
        }
        for (; i < n; i++) destination[i] = (sbyte)MathF.Min(MathF.Max(MathF.Round(x[i] * scalarInv), -128f), 127f);
        return scale;
#endif
    }

#if NEEDLE_SIMD
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<sbyte> Pack32(float* source, Vector256<float> inv, Vector256<int> interleave)
    {
        var c0 = Avx.ConvertToVector256Int32(Avx.Multiply(Avx.LoadVector256(source), inv));
        var c1 = Avx.ConvertToVector256Int32(Avx.Multiply(Avx.LoadVector256(source + 8), inv));
        var c2 = Avx.ConvertToVector256Int32(Avx.Multiply(Avx.LoadVector256(source + 16), inv));
        var c3 = Avx.ConvertToVector256Int32(Avx.Multiply(Avx.LoadVector256(source + 24), inv));
        var p01 = Avx2.PackSignedSaturate(c0, c1);
        var p23 = Avx2.PackSignedSaturate(c2, c3);
        var packed = Avx2.PackSignedSaturate(p01, p23);
        return Avx2.PermuteVar8x32(packed.AsInt32(), interleave).AsSByte();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> Intra8Scale(
        Vector256<float> v,
        Vector256<float> pairSign,
        Vector256<float> quadSign,
        Vector256<float> halfSign,
        Vector256<float> scale)
    {
        v = Fma.MultiplyAdd(v, pairSign, Avx.Permute(v, 0b10_11_00_01));
        v = Fma.MultiplyAdd(v, quadSign, Avx.Permute(v, 0b01_00_11_10));
        v = Fma.MultiplyAdd(v, halfSign, Avx.Permute2x128(v, v, 0x01));
        return Avx.Multiply(v, scale);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float HadamardTailAvx2(float* p)
    {
        var pairSign = Vector256.Create(1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f);
        var quadSign = Vector256.Create(1f, 1f, -1f, -1f, 1f, 1f, -1f, -1f);
        var halfSign = Vector256.Create(1f, 1f, 1f, 1f, -1f, -1f, -1f, -1f);
        var scale = Vector256.Create(HadamardScale);
        var peak = Vector256<float>.Zero;
        for (var b = 0; b < Cq.GroupSize; b += 32)
        {
            var r0 = Avx.LoadVector256(p + b);
            var r1 = Avx.LoadVector256(p + b + 8);
            var r2 = Avx.LoadVector256(p + b + 16);
            var r3 = Avx.LoadVector256(p + b + 24);
            var p0 = Avx.Add(r0, r2);
            var p1 = Avx.Subtract(r0, r2);
            var p2 = Avx.Add(r1, r3);
            var p3 = Avx.Subtract(r1, r3);
            var v0 = Intra8Scale(Avx.Add(p0, p2), pairSign, quadSign, halfSign, scale);
            var v1 = Intra8Scale(Avx.Subtract(p0, p2), pairSign, quadSign, halfSign, scale);
            var v2 = Intra8Scale(Avx.Add(p1, p3), pairSign, quadSign, halfSign, scale);
            var v3 = Intra8Scale(Avx.Subtract(p1, p3), pairSign, quadSign, halfSign, scale);
            Avx.Store(p + b, v0);
            Avx.Store(p + b + 8, v1);
            Avx.Store(p + b + 16, v2);
            Avx.Store(p + b + 24, v3);
            var m01 = Avx.Max(Vector256.Abs(v0), Vector256.Abs(v1));
            var m23 = Avx.Max(Vector256.Abs(v2), Vector256.Abs(v3));
            peak = Avx.Max(peak, Avx.Max(m01, m23));
        }
        var folded = Sse.Max(peak.GetLower(), peak.GetUpper());
        folded = Sse.Max(folded, Sse.Shuffle(folded, folded, 0b01_00_11_10));
        return Sse.Max(folded, Sse.Shuffle(folded, folded, 0b10_11_00_01)).ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float HadamardPeakAvx2(float* source, float* p)
    {
        for (var i = 0; i < 32; i += 8)
        {
            var a = Avx.LoadVector256(source + i);
            var b = Avx.LoadVector256(source + i + 32);
            var c = Avx.LoadVector256(source + i + 64);
            var d = Avx.LoadVector256(source + i + 96);
            var ab0 = Avx.Add(a, c);
            var ab1 = Avx.Subtract(a, c);
            var cd0 = Avx.Add(b, d);
            var cd1 = Avx.Subtract(b, d);
            Avx.Store(p + i, Avx.Add(ab0, cd0));
            Avx.Store(p + i + 32, Avx.Subtract(ab0, cd0));
            Avx.Store(p + i + 64, Avx.Add(ab1, cd1));
            Avx.Store(p + i + 96, Avx.Subtract(ab1, cd1));
        }
        return HadamardTailAvx2(p);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> QuantFake(Vector256<float> value, Vector256<float> a8Inv, Vector256<float> a8Scale, Vector256<float> low, Vector256<float> high) =>
        Avx.Multiply(Avx.Min(Avx.Max(Avx.RoundToNearestInteger(Avx.Multiply(value, a8Inv)), low), high), a8Scale);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float HadamardPeakRmsFakeAvx2(float* source, float* p, Vector256<float> rmsInv, Vector256<float> a8Inv, Vector256<float> a8Scale, Vector256<float> low, Vector256<float> high)
    {
        for (var i = 0; i < 32; i += 8)
        {
            var a = QuantFake(Avx.Multiply(Avx.LoadVector256(source + i), rmsInv), a8Inv, a8Scale, low, high);
            var b = QuantFake(Avx.Multiply(Avx.LoadVector256(source + i + 32), rmsInv), a8Inv, a8Scale, low, high);
            var c = QuantFake(Avx.Multiply(Avx.LoadVector256(source + i + 64), rmsInv), a8Inv, a8Scale, low, high);
            var d = QuantFake(Avx.Multiply(Avx.LoadVector256(source + i + 96), rmsInv), a8Inv, a8Scale, low, high);
            var ab0 = Avx.Add(a, c);
            var ab1 = Avx.Subtract(a, c);
            var cd0 = Avx.Add(b, d);
            var cd1 = Avx.Subtract(b, d);
            Avx.Store(p + i, Avx.Add(ab0, cd0));
            Avx.Store(p + i + 32, Avx.Subtract(ab0, cd0));
            Avx.Store(p + i + 64, Avx.Add(ab1, cd1));
            Avx.Store(p + i + 96, Avx.Subtract(ab1, cd1));
        }
        return HadamardTailAvx2(p);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float HadamardPeakCenteredRmsFakeAvx2(float* source, float* scalePtr, float* p, Vector256<float> one, Vector256<float> rmsInv, Vector256<float> a8Inv, Vector256<float> a8Scale, Vector256<float> low, Vector256<float> high)
    {
        for (var i = 0; i < 32; i += 8)
        {
            var a = QuantFake(Avx.Multiply(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scalePtr + i)), Avx.LoadVector256(source + i)), rmsInv), a8Inv, a8Scale, low, high);
            var b = QuantFake(Avx.Multiply(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scalePtr + i + 32)), Avx.LoadVector256(source + i + 32)), rmsInv), a8Inv, a8Scale, low, high);
            var c = QuantFake(Avx.Multiply(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scalePtr + i + 64)), Avx.LoadVector256(source + i + 64)), rmsInv), a8Inv, a8Scale, low, high);
            var d = QuantFake(Avx.Multiply(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scalePtr + i + 96)), Avx.LoadVector256(source + i + 96)), rmsInv), a8Inv, a8Scale, low, high);
            var ab0 = Avx.Add(a, c);
            var ab1 = Avx.Subtract(a, c);
            var cd0 = Avx.Add(b, d);
            var cd1 = Avx.Subtract(b, d);
            Avx.Store(p + i, Avx.Add(ab0, cd0));
            Avx.Store(p + i + 32, Avx.Subtract(ab0, cd0));
            Avx.Store(p + i + 64, Avx.Add(ab1, cd1));
            Avx.Store(p + i + 96, Avx.Subtract(ab1, cd1));
        }
        return HadamardTailAvx2(p);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void PackGroup(float* rotated, float peak, int bits, in Activations a, int token, int g, Vector256<byte> ones, Vector256<int> interleave)
    {
        var slot = token * a.Stride + g;
        var scale = peak > 0f ? peak / 127f : 1f;
        a.Scales[slot] = scale;
        var inv = Vector256.Create(1f / scale);
        var q0 = Pack32(rotated, inv, interleave);
        var q1 = Pack32(rotated + 32, inv, interleave);
        var q2 = Pack32(rotated + 64, inv, interleave);
        var q3 = Pack32(rotated + 96, inv, interleave);
        var sum = Accumulate(Accumulate(Accumulate(Accumulate(Vector256<int>.Zero, ones, q0), ones, q1), ones, q2), ones, q3);
        a.Offsets[slot] = Codebook.Bias * (float)Vector256.Sum(sum);
        var destination = a.Values + ((long)token * a.Groups + g) * Cq.GroupSize;
        if (bits == 2) Planes2(q0, q1, q2, q3, destination);
        else Planes4(q0, q1, q2, q3, destination);
    }
#endif

#if NEEDLE_SIMD
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void PrepareFakeGroups(float* x, float a8Peak, float rmsInv, int bits, in Activations a) =>
        PrepareFakeGroupsRange(x, a8Peak, rmsInv, bits, a, 0, a.Groups);

    [SkipLocalsInit]
    internal static void PrepareFakeGroupsRange(float* x, float a8Peak, float rmsInv, int bits, in Activations a, int groupStart, int groupEnd)
    {
        var a8Scale = a8Peak > 0f ? a8Peak / 127f : 1f;
        var vRmsInv = Vector256.Create(rmsInv);
        var vA8Inv = Vector256.Create(1f / a8Scale);
        var vA8Scale = Vector256.Create(a8Scale);
        var low = Vector256.Create(-128f);
        var high = Vector256.Create(127f);
        var raw = stackalloc float[Cq.GroupSize + 16];
        var rotated = (float*)(((nuint)raw + 63) & ~(nuint)63);
        var ones = Vector256.Create((byte)1);
        var interleave = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
        for (var g = groupStart; g < groupEnd; g++)
        {
            var peak = HadamardPeakRmsFakeAvx2(x + g * Cq.GroupSize, rotated, vRmsInv, vA8Inv, vA8Scale, low, high);
            PackGroup(rotated, peak, bits, a, 0, g, ones, interleave);
        }
    }
#endif

    [SkipLocalsInit]
    public static void PrepareRmsFakeA8(float* x, float* scratchNormed, int cols, int bits, in Activations a)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) { NeonPrepare.RmsFakeA8(x, cols, bits, a); return; }
        if (Simd.UseAvx && (void*)Kernels.Active.Prepare == null && (void*)Kernels.Active.RmsFakeA8 == null)
        {
            Ops.RmsStats(x, cols, out var inv, out var rawPeak);
            PrepareFakeGroups(x, rawPeak * inv, inv, bits, a);
            return;
        }
#endif
        Ops.RmsFakeA8(x, cols, scratchNormed);
        Prepare(scratchNormed, 1, cols, bits, a);
    }

    [SkipLocalsInit]
    public static void PrepareGateFakeA8(float* x, float* gate, int cols, int bits, in Activations a)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) { NeonPrepare.GateFakeA8(x, gate, cols, bits, a); return; }
        if (Simd.UseAvx && (void*)Kernels.Active.Prepare == null && (void*)Kernels.Active.GateFakeA8 == null)
        {
            var peak0 = Vector256<float>.Zero;
            var peak1 = Vector256<float>.Zero;
            var i = 0;
            for (; i + 16 <= cols; i += 16)
            {
                var v0 = Avx.Multiply(Avx.LoadVector256(x + i), Ops.Sigmoid(Avx.LoadVector256(gate + i)));
                var v1 = Avx.Multiply(Avx.LoadVector256(x + i + 8), Ops.Sigmoid(Avx.LoadVector256(gate + i + 8)));
                Avx.Store(x + i, v0);
                Avx.Store(x + i + 8, v1);
                peak0 = Avx.Max(peak0, Vector256.Abs(v0));
                peak1 = Avx.Max(peak1, Vector256.Abs(v1));
            }
            for (; i < cols; i += 8)
            {
                var v = Avx.Multiply(Avx.LoadVector256(x + i), Ops.Sigmoid(Avx.LoadVector256(gate + i)));
                Avx.Store(x + i, v);
                peak0 = Avx.Max(peak0, Vector256.Abs(v));
            }
            PrepareFakeGroups(x, Ops.FoldPeak(Avx.Max(peak0, peak1)), 1f, bits, a);
            return;
        }
#endif
        Ops.GateFakeA8(x, gate, cols);
        Prepare(x, 1, cols, bits, a);
    }

    [SkipLocalsInit]
    public static void PrepareCenteredRmsFakeA8(float* x, float* normScale, float* scratchNormed, int cols, int bits, in Activations a)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) { NeonPrepare.CenteredRmsFakeA8(x, normScale, cols, bits, a); return; }
        if (Simd.UseAvx && (void*)Kernels.Active.Prepare == null && (void*)Kernels.Active.CenteredRmsFakeA8 == null)
        {
            Ops.CenteredRmsStats(x, normScale, cols, out var inv, out var rawPeak);
            var a8Peak = rawPeak * inv;
            var a8Scale = a8Peak > 0f ? a8Peak / 127f : 1f;
            var one = Vector256.Create(1f);
            var vRmsInv = Vector256.Create(inv);
            var vA8Inv = Vector256.Create(1f / a8Scale);
            var vA8Scale = Vector256.Create(a8Scale);
            var low = Vector256.Create(-128f);
            var high = Vector256.Create(127f);
            var raw = stackalloc float[Cq.GroupSize + 16];
            var rotated = (float*)(((nuint)raw + 63) & ~(nuint)63);
            var ones = Vector256.Create((byte)1);
            var interleave = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
            for (var g = 0; g < a.Groups; g++)
            {
                var offset = g * Cq.GroupSize;
                var peak = HadamardPeakCenteredRmsFakeAvx2(x + offset, normScale + offset, rotated, one, vRmsInv, vA8Inv, vA8Scale, low, high);
                PackGroup(rotated, peak, bits, a, 0, g, ones, interleave);
            }
            return;
        }
#endif
        Ops.CenteredRmsFakeA8(x, normScale, cols, scratchNormed);
        Prepare(scratchNormed, 1, cols, bits, a);
    }

    public static void Hadamard(float* p) => Hadamard(p, p);

    public static void Hadamard(float* source, float* p)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Hadamard(source, p); return; }
#endif
        if (!Simd.Enabled) { Scalar.Hadamard(source, p); return; }
#if NEEDLE_SIMD
        HadamardPeakAvx2(source, p);
#endif
    }

    struct PrepareJob
    {
        public float* X;
        public int Tokens, Cols, Bits, Tasks;
        public Activations Act;
    }

    public static void Prepare(Pool pool, float* x, int tokens, int cols, int bits, in Activations a)
    {
        if (tokens <= 0 || a.Groups <= 0) return;
        if (pool.Threads <= 1 || (tokens == 1 && a.Groups <= 1))
        {
            Prepare(x, tokens, cols, bits, a);
            return;
        }
        var tasks = tokens == 1 ? Math.Min(pool.Members, a.Groups) : Math.Min(tokens, pool.Members * 4);
        var job = new PrepareJob { X = x, Tokens = tokens, Cols = cols, Bits = bits, Tasks = tasks, Act = a };
        if (tokens == 1) pool.Run(&PrepareGroupsTask, &job, tasks);
        else pool.Run(&PrepareTokensTask, &job, tasks);
    }

    static void PrepareTokensTask(void* context, int task)
    {
        var job = (PrepareJob*)context;
        var start = task * job->Tokens / job->Tasks;
        var end = (task + 1) * job->Tokens / job->Tasks;
        for (var t = start; t < end; t++)
            PrepareRange(job->X + (long)t * job->Cols, 1, job->Cols, job->Bits, job->Act.Slice(t), 0, job->Act.Groups);
    }

    static void PrepareGroupsTask(void* context, int task)
    {
        var job = (PrepareJob*)context;
        var start = task * job->Act.Groups / job->Tasks;
        var count = (task + 1) * job->Act.Groups / job->Tasks - start;
        PrepareRange(job->X, 1, job->Cols, job->Bits, job->Act, start, count);
    }

    public static void Prepare(float* x, int tokens, int cols, int bits, in Activations a) =>
        PrepareRange(x, tokens, cols, bits, a, 0, a.Groups);

    public static void PrepareGroups(float* x, int groupStart, int groupCount, int bits, in Activations a) =>
        PrepareRange(x, 1, a.Cols, bits, a, groupStart, groupCount);

    [SkipLocalsInit]
    static void PrepareRange(float* x, int tokens, int cols, int bits, in Activations a, int groupStart, int groupCount)
    {
        if (groupCount <= 0 || tokens <= 0) return;
        #if NEEDLE_SIMD
        if (Simd.UseNeon)
        {
            if (groupStart == 0 && groupCount == a.Groups) Neon.Prepare(x, tokens, cols, bits, a);
            else if (tokens == 1)
            {
                var slice = new Activations(a.Values + groupStart * Cq.GroupSize, a.Scales + groupStart, a.Offsets + groupStart, groupCount * Cq.GroupSize);
                Neon.Prepare(x + groupStart * Cq.GroupSize, 1, groupCount * Cq.GroupSize, bits, slice);
            }
            return;
        }
#endif
        var kernelFp = Kernels.Active.Prepare;
        if ((void*)kernelFp != null)
        {
            if (groupStart == 0 && groupCount == a.Groups)
                InvokePrepare(kernelFp, x, tokens, cols, bits, a);
            else if (tokens == 1)
            {
                var slice = new Activations(a.Values + groupStart * Cq.GroupSize, a.Scales + groupStart, a.Offsets + groupStart, groupCount * Cq.GroupSize);
                InvokePrepare(kernelFp, x + groupStart * Cq.GroupSize, 1, groupCount * Cq.GroupSize, bits, slice);
            }
            return;
        }
        if (!Simd.Enabled)
        {
            var rotatedScalar = stackalloc float[Cq.GroupSize];
            var quantizedScalar = stackalloc sbyte[Cq.GroupSize];
            var end = groupStart + groupCount;
            for (var t = 0; t < tokens; t++)
                for (var g = groupStart; g < end; g++)
                {
                    var slot = t * a.Stride + g;
                    Scalar.Hadamard(x + (long)t * cols + g * Cq.GroupSize, rotatedScalar);
                    a.Scales[slot] = Scalar.QuantizeInt8(rotatedScalar, Cq.GroupSize, quantizedScalar);
                    Scalar.Planes(quantizedScalar, bits, a.Values + ((long)t * a.Groups + g) * Cq.GroupSize, out var scalarSum);
                    a.Offsets[slot] = Codebook.Bias * (float)scalarSum;
                }
            return;
        }
#if NEEDLE_SIMD
        var raw = stackalloc float[Cq.GroupSize + 16];
        var rotated = (float*)(((nuint)raw + 63) & ~(nuint)63);
        var ones = Vector256.Create((byte)1);
        var interleave = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
        var groupEnd = groupStart + groupCount;
        for (var t = 0; t < tokens; t++)
            for (var g = groupStart; g < groupEnd; g++)
            {
                var peak = HadamardPeakAvx2(x + (long)t * cols + g * Cq.GroupSize, rotated);
                PackGroup(rotated, peak, bits, a, t, g, ones, interleave);
            }
#endif
    }

#if NEEDLE_SIMD
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void Planes2(Vector256<sbyte> q0, Vector256<sbyte> q1, Vector256<sbyte> q2, Vector256<sbyte> q3, sbyte* destination)
    {
        var gather = Vector256.Create((byte)0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        var s0 = Avx2.Shuffle(q0.AsByte(), gather).AsInt32();
        var s1 = Avx2.Shuffle(q1.AsByte(), gather).AsInt32();
        var s2 = Avx2.Shuffle(q2.AsByte(), gather).AsInt32();
        var s3 = Avx2.Shuffle(q3.AsByte(), gather).AsInt32();
        var low01 = Avx2.UnpackLow(s0, s1);
        var high01 = Avx2.UnpackHigh(s0, s1);
        var low23 = Avx2.UnpackLow(s2, s3);
        var high23 = Avx2.UnpackHigh(s2, s3);
        var interleave = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
        Avx.Store(destination, Avx2.PermuteVar8x32(Avx2.UnpackLow(low01.AsInt64(), low23.AsInt64()).AsInt32(), interleave).AsSByte());
        Avx.Store(destination + 32, Avx2.PermuteVar8x32(Avx2.UnpackHigh(low01.AsInt64(), low23.AsInt64()).AsInt32(), interleave).AsSByte());
        Avx.Store(destination + 64, Avx2.PermuteVar8x32(Avx2.UnpackLow(high01.AsInt64(), high23.AsInt64()).AsInt32(), interleave).AsSByte());
        Avx.Store(destination + 96, Avx2.PermuteVar8x32(Avx2.UnpackHigh(high01.AsInt64(), high23.AsInt64()).AsInt32(), interleave).AsSByte());
    }
#endif

#if NEEDLE_SIMD
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void Planes4(Vector256<sbyte> q0, Vector256<sbyte> q1, Vector256<sbyte> q2, Vector256<sbyte> q3, sbyte* destination)
    {
        var gather = Vector256.Create((byte)0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
        var s0 = Avx2.Shuffle(q0.AsByte(), gather).AsInt64();
        var s1 = Avx2.Shuffle(q1.AsByte(), gather).AsInt64();
        var s2 = Avx2.Shuffle(q2.AsByte(), gather).AsInt64();
        var s3 = Avx2.Shuffle(q3.AsByte(), gather).AsInt64();
        Avx.Store(destination, Avx2.Permute4x64(Avx2.UnpackLow(s0, s1), 0b11_01_10_00).AsSByte());
        Avx.Store(destination + 32, Avx2.Permute4x64(Avx2.UnpackHigh(s0, s1), 0b11_01_10_00).AsSByte());
        Avx.Store(destination + 64, Avx2.Permute4x64(Avx2.UnpackLow(s2, s3), 0b11_01_10_00).AsSByte());
        Avx.Store(destination + 96, Avx2.Permute4x64(Avx2.UnpackHigh(s2, s3), 0b11_01_10_00).AsSByte());
    }
#endif

    public static void DecodeRow(in Cq w, int row, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.DecodeRow(w, row, destination); return; }
#endif
        if (!Simd.Enabled) { Scalar.DecodeRow(w, row, destination); return; }
#if NEEDLE_SIMD
        var indices = w.Indices + (long)row * w.RowBytes;
        var norms = w.Norms + (long)row * w.Groups;
        var lut = Avx.LoadVector256(w.Book->Levels);
        if (w.Bits == 2)
            for (var g = 0; g < w.Groups; g++)
                DecodeGroup2(indices + g * 32, Avx.Multiply(lut, Vector256.Create(Scalar.HalfToSingle(norms[g]))), destination + g * Cq.GroupSize);
        else
        {
            var upper = Avx.LoadVector256(w.Book->Levels + 8);
            for (var g = 0; g < w.Groups; g++)
            {
                var norm = Vector256.Create(Scalar.HalfToSingle(norms[g]));
                DecodeGroup4(indices + g * 64, Avx.Multiply(lut, norm), Avx.Multiply(upper, norm), destination + g * Cq.GroupSize);
            }
        }
#endif
    }

    public static void GatherRow(in Cq w, int row, float* destination)
    {
        var kernelFp = Kernels.Active.GatherRow;
        if ((void*)kernelFp != null)
        {
            InvokeGatherRow(kernelFp, w, row, destination);
            return;
        }
        DecodeRow(w, row, destination);
        for (var g = 0; g < w.Groups; g++) Hadamard(destination + g * Cq.GroupSize);
    }

#if NEEDLE_SIMD
    static void DecodeGroup2(byte* packed, Vector256<float> lut, float* destination)
    {
        var lowShift = Vector256.Create(0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u);
        var highShift = Vector256.Create(16u, 18u, 20u, 22u, 24u, 26u, 28u, 30u);
        for (var k = 0; k < 8; k++)
        {
            var word = Vector256.Create(((uint*)packed)[k]);
            Avx.Store(destination + 16 * k, Avx2.PermuteVar8x32(lut, Avx2.ShiftRightLogicalVariable(word, lowShift).AsInt32()));
            Avx.Store(destination + 16 * k + 8, Avx2.PermuteVar8x32(lut, Avx2.ShiftRightLogicalVariable(word, highShift).AsInt32()));
        }
    }
#endif

#if NEEDLE_SIMD
    static void DecodeGroup4(byte* packed, Vector256<float> lower, Vector256<float> upper, float* destination)
    {
        var shift = Vector256.Create(0u, 4u, 8u, 12u, 16u, 20u, 24u, 28u);
        for (var k = 0; k < 16; k++)
        {
            var index = Avx2.ShiftRightLogicalVariable(Vector256.Create(((uint*)packed)[k]), shift);
            var selector = Avx2.ShiftLeftLogical(index, 28).AsSingle();
            var index3 = index.AsInt32();
            Avx.Store(destination + 8 * k, Avx.BlendVariable(Avx2.PermuteVar8x32(lower, index3), Avx2.PermuteVar8x32(upper, index3), selector));
        }
    }
#endif

#if NEEDLE_SIMD
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<int> Accumulate(Vector256<int> sum, Vector256<byte> weights, Vector256<sbyte> activations) =>
        Tiers.Accumulate(sum, weights, activations);
#endif

    public static void MatMul(in Cq w, in Activations a, int tokens, float* y, int ldy) =>
        MatMulRows(w, a, tokens, y, ldy, 0, w.Rows);

    public static void MatMulRows(in Cq w, in Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.MatMulRows(w, a, tokens, y, ldy, rowStart, rowEnd); return; }
#endif
        var kernelFp = Kernels.Active.MatMulRows;
        if ((void*)kernelFp != null)
        {
            InvokeMatMulRows(kernelFp, w, a, tokens, y, ldy, rowStart, rowEnd);
            return;
        }
        if (!Simd.Enabled) { Scalar.MatMulRows(w, a, tokens, y, ldy, rowStart, rowEnd); return; }
#if NEEDLE_SIMD
        var tiledTokens = tokens & ~(TokenTile - 1);
        for (var t = 0; t < tiledTokens; t += TokenTile) Gemv.Tile4(w, a, t, y + (long)t * ldy, ldy, rowStart, rowEnd);
        if (tokens - tiledTokens >= 2) { Gemv.Tile2(w, a, tiledTokens, y + (long)tiledTokens * ldy, ldy, rowStart, rowEnd); tiledTokens += 2; }
        for (var t = tiledTokens; t < tokens; t++) Gemv.Rows(w, a, t, y + (long)t * ldy, rowStart, rowEnd);
#endif
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    static float InvokeFakeA8(nint fp, float* x, int n) =>
        ((delegate* unmanaged[Cdecl]<float*, int, float>)(void*)fp)(x, n);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static float InvokeQuantizeInt8(nint fp, float* x, int n, sbyte* destination) =>
        ((delegate* unmanaged[Cdecl]<float*, int, sbyte*, float>)(void*)fp)(x, n, destination);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokePrepare(nint fp, float* x, int tokens, int cols, int bits, in Activations a)
    {
        fixed (Activations* pa = &a)
            ((delegate* unmanaged[Cdecl]<float*, int, int, int, Activations*, void>)(void*)fp)(x, tokens, cols, bits, pa);
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeGatherRow(nint fp, in Cq w, int row, float* destination)
    {
        fixed (Cq* pw = &w)
            ((delegate* unmanaged[Cdecl]<Cq*, int, float*, void>)(void*)fp)(pw, row, destination);
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeMatMulRows(nint fp, in Cq w, in Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
    {
        fixed (Cq* pw = &w)
        fixed (Activations* pa = &a)
            ((delegate* unmanaged[Cdecl]<Cq*, Activations*, int, float*, int, int, int, void>)(void*)fp)(pw, pa, tokens, y, ldy, rowStart, rowEnd);
    }

}

internal unsafe struct MatMulJob
{
    public Cq W;
    public float* Y;
    public int Ldy;
}

internal unsafe struct MatMulJobs
{
    MatMulJob first, second, third, fourth;

    public MatMulJob* At(int index)
    {
        fixed (MatMulJob* p = &first) return p + index;
    }
}

internal unsafe struct MatMulBatch
{
    public MatMulJobs Jobs;
    public Activations Activations;
    public int Tokens;
    public int Count;
    public int RowsPerTask;

    public MatMulBatch(in Activations activations, int tokens)
    {
        Activations = activations;
        Tokens = tokens;
        Count = 0;
        RowsPerTask = 0;
    }

    public void Add(in Cq w, float* y, int ldy) => *Jobs.At(Count++) = new MatMulJob { W = w, Y = y, Ldy = ldy };

    public static void Run(Pool pool, MatMulBatch* batch)
    {
        var totalBlocks = batch->TotalRowBlocks();
        var members = Math.Min(pool.Members, totalBlocks);
        if (members <= 1)
        {
            RunShard(batch, 0, 1);
            return;
        }
        batch->Members = members;
        pool.Run(&ExecuteRange, batch, members);
    }

    public int TotalRowBlocks()
    {
        var total = 0;
        for (var j = 0; j < Count; j++) total += (Jobs.At(j)->W.Rows + Quant.RowBlock - 1) / Quant.RowBlock;
        return total;
    }

    public static void RunShard(MatMulBatch* batch, int member, int members)
    {
        var totalBlocks = batch->TotalRowBlocks();
        var firstBlock = member * totalBlocks / members;
        var endBlock = (member + 1) * totalBlocks / members;
        RunRange(batch, firstBlock, endBlock);
    }

    public static void RunSteal(MatMulBatch* batch, int* cursor, int member, int members)
    {
        if (members <= 1 || cursor == null)
        {
            RunShard(batch, member, members);
            return;
        }
        var totalBlocks = batch->TotalRowBlocks();
        if (totalBlocks < members * 8)
        {
            RunShard(batch, member, members);
            return;
        }
        var workerInit = batch->Tokens == 1 ? (totalBlocks * 3) / (4 * members) : (totalBlocks * 2) / (3 * members);
        var initTotal = (members - 1) * workerInit;
        if (member < members - 1 && workerInit > 0)
            RunRange(batch, member * workerInit, (member + 1) * workerInit);
        var remain = totalBlocks - initTotal;
        var chunk = batch->Tokens == 1 ? Math.Clamp(remain / (members * 4), 8, 64) : Math.Clamp(remain / (members * 8), 2, 16);
        while (true)
        {
            var taken = Interlocked.Add(ref *cursor, chunk) - chunk;
            var start = initTotal + taken;
            if (start >= totalBlocks) break;
            RunRange(batch, start, Math.Min(totalBlocks, start + chunk));
        }
    }

    static void RunRange(MatMulBatch* batch, int firstBlock, int endBlock)
    {
        if (firstBlock >= endBlock) return;
        var cursor = 0;
        for (var j = 0; j < batch->Count; j++)
        {
            var job = batch->Jobs.At(j);
            var jobBlocks = (job->W.Rows + Quant.RowBlock - 1) / Quant.RowBlock;
            var jobEnd = cursor + jobBlocks;
            if (jobEnd > firstBlock && cursor < endBlock)
            {
                var localStart = Math.Max(0, firstBlock - cursor) * Quant.RowBlock;
                var localEnd = Math.Min(job->W.Rows, (endBlock - cursor) * Quant.RowBlock);
                if (localEnd > localStart)
                    Quant.MatMulRows(job->W, batch->Activations, batch->Tokens, job->Y, job->Ldy, localStart, localEnd);
            }
            cursor = jobEnd;
        }
    }

    public int Members;

    public void Plan(int threads)
    {
        var rows = 0;
        for (var j = 0; j < Count; j++) rows += Jobs.At(j)->W.Rows;
        var perTask = Math.Max(MinimumRowsPerTask, (rows + threads * TasksPerThread - 1) / (threads * TasksPerThread));
        RowsPerTask = (perTask + Quant.RowBlock - 1) / Quant.RowBlock * Quant.RowBlock;
    }

    public int TaskCount
    {
        get
        {
            var tasks = 0;
            for (var j = 0; j < Count; j++) tasks += (Jobs.At(j)->W.Rows + RowsPerTask - 1) / RowsPerTask;
            return tasks;
        }
    }

    public static void RunTask(MatMulBatch* batch, int index)
    {
        for (var j = 0; j < batch->Count; j++)
        {
            var tasks = JobTasks(batch, j);
            if (index >= tasks) { index -= tasks; continue; }
            var job = batch->Jobs.At(j);
            var start = index * batch->RowsPerTask;
            Quant.MatMulRows(job->W, batch->Activations, batch->Tokens, job->Y, job->Ldy, start, Math.Min(job->W.Rows, start + batch->RowsPerTask));
            return;
        }
    }

    static int JobTasks(MatMulBatch* batch, int job) => (batch->Jobs.At(job)->W.Rows + batch->RowsPerTask - 1) / batch->RowsPerTask;

    static void ExecuteRange(void* context, int member)
    {
        var batch = (MatMulBatch*)context;
        RunShard(batch, member, batch->Members);
    }

    const int TasksPerThread = 3;
    const int MinimumRowsPerTask = 8;
}
