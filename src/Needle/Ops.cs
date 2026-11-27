using System;
using System.Runtime.CompilerServices;
#if NEEDLE_SIMD
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace Needle;

internal static unsafe class Ops
{
    const float RmsEpsilon = 1e-6f;

#if NEEDLE_SIMD
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Exp(Vector256<float> x)
    {
        x = Avx.Min(Avx.Max(x, Vector256.Create(-87.3f)), Vector256.Create(88.7f));
        var n = Avx.RoundToNearestInteger(Avx.Multiply(x, Vector256.Create(1.44269504f)));
        var r = Fma.MultiplyAddNegated(n, Vector256.Create(0.693359375f), x);
        r = Fma.MultiplyAddNegated(n, Vector256.Create(-2.12194440e-4f), r);
        var p = Vector256.Create(1.9875691500e-4f);
        p = Fma.MultiplyAdd(p, r, Vector256.Create(1.3981999507e-3f));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(8.3334519073e-3f));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(4.1665795894e-2f));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(1.6666665459e-1f));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(5.0000001201e-1f));
        var y = Fma.MultiplyAdd(p, Avx.Multiply(r, r), Avx.Add(r, Vector256.Create(1f)));
        var scale = Avx2.ShiftLeftLogical(Avx2.Add(Avx.ConvertToVector256Int32(n), Vector256.Create(127)), 23).AsSingle();
        return Avx.Multiply(y, scale);
    }
#endif

#if NEEDLE_SIMD
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Sigmoid(Vector256<float> x)
    {
        var one = Vector256.Create(1f);
        return Avx.Divide(one, Avx.Add(one, Exp(Avx.Subtract(Vector256<float>.Zero, x))));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> FastSigmoid(Vector256<float> x)
    {
        x = Avx.Max(x, Vector256.Create(-87.3f));
        var one = Vector256.Create(1f);
        var denom = Avx.Add(one, Exp(Avx.Subtract(Vector256<float>.Zero, x)));
        var r = Avx.Reciprocal(denom);
        return Avx.Multiply(r, Fma.MultiplyAddNegated(denom, r, Vector256.Create(2f)));
    }

    internal static void ApplyGateAndHeadPeaks(float* output, float* gate, int heads, int vDim, float* peaks)
    {
        for (var h = 0; h < heads; h++)
        {
            var x = output + h * vDim;
            var g = gate + h * vDim;
            var peak0 = Vector256<float>.Zero;
            var peak1 = Vector256<float>.Zero;
            var i = 0;
            for (; i + 16 <= vDim; i += 16)
            {
                var v0 = Avx.Multiply(Avx.LoadVector256(x + i), Sigmoid(Avx.LoadVector256(g + i)));
                var v1 = Avx.Multiply(Avx.LoadVector256(x + i + 8), Sigmoid(Avx.LoadVector256(g + i + 8)));
                Avx.Store(x + i, v0);
                Avx.Store(x + i + 8, v1);
                peak0 = Avx.Max(peak0, Vector256.Abs(v0));
                peak1 = Avx.Max(peak1, Vector256.Abs(v1));
            }
            for (; i + 8 <= vDim; i += 8)
            {
                var v = Avx.Multiply(Avx.LoadVector256(x + i), Sigmoid(Avx.LoadVector256(g + i)));
                Avx.Store(x + i, v);
                peak0 = Avx.Max(peak0, Vector256.Abs(v));
            }
            var headPeak = FoldPeak(Avx.Max(peak0, peak1));
            for (; i < vDim; i++)
            {
                var v = x[i] * Sigmoid(g[i]);
                x[i] = v;
                headPeak = MathF.Max(headPeak, MathF.Abs(v));
            }
            peaks[h * 16] = headPeak;
        }
    }
#endif

    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    public static void Silu(float* x, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Silu(x, n); return; }
#endif
        if (!Simd.Enabled) { Scalar.Silu(x, n); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.Silu(x, n); return; }
        var i = 0;
        for (; i + 8 <= n; i += 8)
        {
            var v = Avx.LoadVector256(x + i);
            Avx.Store(x + i, Avx.Multiply(v, Sigmoid(v)));
        }
        for (; i < n; i++) x[i] *= Sigmoid(x[i]);
#endif
    }

    public static void MultiplyBySigmoid(float* x, float* gate, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.MultiplyBySigmoid(x, gate, n); return; }
#endif
        if (!Simd.Enabled) { Scalar.MultiplyBySigmoid(x, gate, n); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.MultiplyBySigmoid(x, gate, n); return; }
        var i = 0;
        for (; i + 8 <= n; i += 8)
            Avx.Store(x + i, Avx.Multiply(Avx.LoadVector256(x + i), Sigmoid(Avx.LoadVector256(gate + i))));
        for (; i < n; i++) x[i] *= Sigmoid(gate[i]);
#endif
    }

    public static float SumOfSquares(float* x, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { return Neon.SumOfSquares(x, n); }
#endif
        if (!Simd.Enabled) return Scalar.SumOfSquares(x, n);
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) return OpsAvx512.SumOfSquares(x, n);
        var s0 = Vector256<float>.Zero;
        var s1 = Vector256<float>.Zero;
        var s2 = Vector256<float>.Zero;
        var s3 = Vector256<float>.Zero;
        var i = 0;
        var bulk = n & ~31;
        for (; i < bulk; i += 32)
        {
            var a0 = Avx.LoadVector256(x + i);
            var a1 = Avx.LoadVector256(x + i + 8);
            var a2 = Avx.LoadVector256(x + i + 16);
            var a3 = Avx.LoadVector256(x + i + 24);
            s0 = Fma.MultiplyAdd(a0, a0, s0);
            s1 = Fma.MultiplyAdd(a1, a1, s1);
            s2 = Fma.MultiplyAdd(a2, a2, s2);
            s3 = Fma.MultiplyAdd(a3, a3, s3);
        }
        for (; i + 16 <= n; i += 16)
        {
            var a = Avx.LoadVector256(x + i);
            var b = Avx.LoadVector256(x + i + 8);
            s0 = Fma.MultiplyAdd(a, a, s0);
            s1 = Fma.MultiplyAdd(b, b, s1);
        }
        if (i + 8 <= n)
        {
            var a = Avx.LoadVector256(x + i);
            s2 = Fma.MultiplyAdd(a, a, s2);
            i += 8;
        }
        var tail = 0f;
        for (; i < n; i++) tail = MathF.FusedMultiplyAdd(x[i], x[i], tail);
        return Vector256.Sum(Avx.Add(Avx.Add(s0, s1), Avx.Add(s2, s3))) + tail;
#endif
    }

    public static float Dot(float* a, float* b, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { return Neon.Dot(a, b, n); }
#endif
        var kernelFp = Kernels.Active.Dot;
        if ((void*)kernelFp != null)
        {
            return InvokeDot(kernelFp, a, b, n);
        }
        if (!Simd.Enabled) return Scalar.Dot(a, b, n);
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) return OpsAvx512.Dot(a, b, n);
        var s0 = Vector256<float>.Zero;
        var s1 = Vector256<float>.Zero;
        var s2 = Vector256<float>.Zero;
        var s3 = Vector256<float>.Zero;
        var i = 0;
        var bulk = n & ~31;
        for (; i < bulk; i += 32)
        {
            s0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), s0);
            s1 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8), s1);
            s2 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 16), Avx.LoadVector256(b + i + 16), s2);
            s3 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 24), Avx.LoadVector256(b + i + 24), s3);
        }
        for (; i + 16 <= n; i += 16)
        {
            s0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), s0);
            s1 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8), s1);
        }
        if (i + 8 <= n)
        {
            s2 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), s2);
            i += 8;
        }
        var tail = 0f;
        for (; i < n; i++) tail = MathF.FusedMultiplyAdd(a[i], b[i], tail);
        return Vector256.Sum(Avx.Add(Avx.Add(s0, s1), Avx.Add(s2, s3))) + tail;
#endif
    }

    public static float InverseRms(float* x, int n) => 1f / MathF.Sqrt(SumOfSquares(x, n) / n + RmsEpsilon);

    public static void RmsUnit(float* x, int n, float* destination)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.RmsUnit(x, n, destination); return; }
#endif
        var kernelFp = Kernels.Active.RmsUnit;
        if ((void*)kernelFp != null)
        {
            InvokeRmsUnit(kernelFp, x, n, destination);
            return;
        }
        if (!Simd.Enabled) { var scalarInverse = InverseRms(x, n); for (var k = 0; k < n; k++) destination[k] = x[k] * scalarInverse; return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.RmsUnit(x, n, destination); return; }
        var scalarInv = InverseRms(x, n);
        var inverse = Vector256.Create(scalarInv);
        var i = 0;
        for (; i + 8 <= n; i += 8) Avx.Store(destination + i, Avx.Multiply(Avx.LoadVector256(x + i), inverse));
        for (; i < n; i++) destination[i] = x[i] * scalarInv;
#endif
    }

    public static void ZeroCenteredRms(float* x, float* scale, int n, float* destination)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.ZeroCenteredRms(x, scale, n, destination); return; }
#endif
        var kernelFp = Kernels.Active.ZeroCenteredRms;
        if ((void*)kernelFp != null)
        {
            InvokeZeroCenteredRms(kernelFp, x, scale, n, destination);
            return;
        }
        if (!Simd.Enabled) { var scalarInverse = InverseRms(x, n); for (var k = 0; k < n; k++) destination[k] = (1f + scale[k]) * x[k] * scalarInverse; return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.ZeroCenteredRms(x, scale, n, destination); return; }
        var scalarInv = InverseRms(x, n);
        var inverse = Vector256.Create(scalarInv);
        var one = Vector256.Create(1f);
        var i = 0;
        for (; i + 8 <= n; i += 8)
            Avx.Store(destination + i, Avx.Multiply(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i)), Avx.LoadVector256(x + i)), inverse));
        for (; i < n; i++) destination[i] = (1f + scale[i]) * x[i] * scalarInv;
#endif
    }

    public static void Softmax(float* s, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Softmax(s, n); return; }
#endif
        var kernelFp = Kernels.Active.Softmax;
        if ((void*)kernelFp != null)
        {
            InvokeSoftmax(kernelFp, s, n);
            return;
        }
        if (!Simd.Enabled) { Scalar.Softmax(s, n); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.Softmax(s, n); return; }
        if (n == 8)
        {
            var v = Avx.LoadVector256(s);
            var e = Exp(Avx.Subtract(v, Vector256.Create(Max8(v))));
            var total8 = Vector256.Sum(e);
            Avx.Store(s, Avx.Multiply(e, Vector256.Create(1f / total8)));
            return;
        }
        var peakVector = Vector256.Create(float.NegativeInfinity);
        var peakEnd = n & ~7;
        var p = 0;
        for (; p < peakEnd; p += 8) peakVector = Avx.Max(peakVector, Avx.LoadVector256(s + p));
        var peak = Max8(peakVector);
        for (; p < n; p++) peak = MathF.Max(peak, s[p]);
        var shift = Vector256.Create(peak);
        var sum = Vector256<float>.Zero;
        var vectorEnd = n & ~7;
        var i2 = 0;
        for (; i2 < vectorEnd; i2 += 8)
        {
            var e = Exp(Avx.Subtract(Avx.LoadVector256(s + i2), shift));
            Avx.Store(s + i2, e);
            sum = Avx.Add(sum, e);
        }
        var total = Vector256.Sum(sum);
        for (; i2 < n; i2++)
        {
            s[i2] = MathF.Exp(s[i2] - peak);
            total += s[i2];
        }
        var inverse = Vector256.Create(1f / total);
        for (i2 = 0; i2 < vectorEnd; i2 += 8) Avx.Store(s + i2, Avx.Multiply(Avx.LoadVector256(s + i2), inverse));
        for (; i2 < n; i2++) s[i2] /= total;
#endif
    }

    public static void LogSoftmax(float* s, int n)
    {
        if (!Simd.Enabled)
        {
            var scalarPeak = float.NegativeInfinity;
            for (var j = 0; j < n; j++) scalarPeak = MathF.Max(scalarPeak, s[j]);
            double scalarSum = 0;
            for (var j = 0; j < n; j++) scalarSum += Math.Exp(s[j] - scalarPeak);
            var scalarShift = scalarPeak + (float)Math.Log(scalarSum);
            for (var j = 0; j < n; j++) s[j] -= scalarShift;
            return;
        }
#if NEEDLE_SIMD
        var p0 = Vector256.Create(float.NegativeInfinity);
        var p1 = p0;
        var p2 = p0;
        var p3 = p0;
        var i = 0;
        var bulk32 = n & ~31;
        for (; i < bulk32; i += 32)
        {
            p0 = Avx.Max(p0, Avx.LoadVector256(s + i));
            p1 = Avx.Max(p1, Avx.LoadVector256(s + i + 8));
            p2 = Avx.Max(p2, Avx.LoadVector256(s + i + 16));
            p3 = Avx.Max(p3, Avx.LoadVector256(s + i + 24));
        }
        p0 = Avx.Max(Avx.Max(p0, p1), Avx.Max(p2, p3));
        var bulk8 = n & ~7;
        for (; i < bulk8; i += 8) p0 = Avx.Max(p0, Avx.LoadVector256(s + i));
        var peak = Max8(p0);
        for (; i < n; i++) peak = MathF.Max(peak, s[i]);
        var shift = Vector256.Create(peak);
        var s0 = Vector256<float>.Zero;
        var s1 = s0;
        for (i = 0; i < bulk8 - 8; i += 16)
        {
            s0 = Avx.Add(s0, Exp(Avx.Subtract(Avx.LoadVector256(s + i), shift)));
            s1 = Avx.Add(s1, Exp(Avx.Subtract(Avx.LoadVector256(s + i + 8), shift)));
        }
        for (; i < bulk8; i += 8) s0 = Avx.Add(s0, Exp(Avx.Subtract(Avx.LoadVector256(s + i), shift)));
        var total = Vector256.Sum(Avx.Add(s0, s1));
        for (; i < n; i++) total += MathF.Exp(s[i] - peak);
        var logShift = Vector256.Create(peak + MathF.Log(total));
        var scalarLogShift = peak + MathF.Log(total);
        for (i = 0; i < bulk32; i += 32)
        {
            Avx.Store(s + i, Avx.Subtract(Avx.LoadVector256(s + i), logShift));
            Avx.Store(s + i + 8, Avx.Subtract(Avx.LoadVector256(s + i + 8), logShift));
            Avx.Store(s + i + 16, Avx.Subtract(Avx.LoadVector256(s + i + 16), logShift));
            Avx.Store(s + i + 24, Avx.Subtract(Avx.LoadVector256(s + i + 24), logShift));
        }
        for (; i < bulk8; i += 8) Avx.Store(s + i, Avx.Subtract(Avx.LoadVector256(s + i), logShift));
        for (; i < n; i++) s[i] -= scalarLogShift;
#endif
    }

    public static void Rope(float* x, int heads, int headDim, float* cos, float* sin)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Rope(x, heads, headDim, cos, sin); return; }
#endif
        var kernelFp = Kernels.Active.Rope;
        if ((void*)kernelFp != null)
        {
            InvokeRope(kernelFp, x, heads, headDim, cos, sin);
            return;
        }
        if (!Simd.Enabled) { Scalar.Rope(x, heads, headDim, cos, sin); return; }
#if NEEDLE_SIMD
        var half = headDim / 2;
        for (var h = 0; h < heads; h++)
        {
            var lo = x + h * headDim;
            var hi = lo + half;
            for (var i = 0; i < half; i += 8)
            {
                var c = Avx.LoadVector256(cos + i);
                var s = Avx.LoadVector256(sin + i);
                var a = Avx.LoadVector256(lo + i);
                var b = Avx.LoadVector256(hi + i);
                Avx.Store(lo + i, Fma.MultiplyAddNegated(b, s, Avx.Multiply(a, c)));
                Avx.Store(hi + i, Fma.MultiplyAdd(a, s, Avx.Multiply(b, c)));
            }
        }
#endif
    }

    public static void Mix4(float* s, int dim, float w0, float w1, float w2, float w3, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Mix4(s, dim, w0, w1, w2, w3, destination); return; }
#endif
        var kernelFp = Kernels.Active.Mix4;
        if ((void*)kernelFp != null)
        {
            InvokeMix4(kernelFp, s, dim, w0, w1, w2, w3, destination);
            return;
        }
        if (!Simd.Enabled) { Scalar.Mix4(s, dim, w0, w1, w2, w3, destination); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.Mix4(s, dim, w0, w1, w2, w3, destination); return; }
        var v0 = Vector256.Create(w0);
        var v1 = Vector256.Create(w1);
        var v2 = Vector256.Create(w2);
        var v3 = Vector256.Create(w3);
        for (var c = 0; c < dim; c += 8)
        {
            var acc = Avx.Multiply(v0, Avx.LoadVector256(s + c));
            acc = Fma.MultiplyAdd(v1, Avx.LoadVector256(s + dim + c), acc);
            acc = Fma.MultiplyAdd(v2, Avx.LoadVector256(s + 2 * dim + c), acc);
            acc = Fma.MultiplyAdd(v3, Avx.LoadVector256(s + 3 * dim + c), acc);
            Avx.Store(destination + c, acc);
        }
#endif
    }

    public static void Combine4(float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Combine4(s, dim, w0, w1, w2, w3, y, post, destination); return; }
#endif
        var kernelFp = Kernels.Active.Combine4;
        if ((void*)kernelFp != null)
        {
            InvokeCombine4(kernelFp, s, dim, w0, w1, w2, w3, y, post, destination);
            return;
        }
        if (!Simd.Enabled) { Scalar.Combine4(s, dim, w0, w1, w2, w3, y, post, destination); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.Combine4(s, dim, w0, w1, w2, w3, y, post, destination); return; }
        var v0 = Vector256.Create(w0);
        var v1 = Vector256.Create(w1);
        var v2 = Vector256.Create(w2);
        var v3 = Vector256.Create(w3);
        var vp = Vector256.Create(post);
        for (var c = 0; c < dim; c += 8)
        {
            var acc = Avx.Multiply(vp, Avx.LoadVector256(y + c));
            acc = Fma.MultiplyAdd(v0, Avx.LoadVector256(s + c), acc);
            acc = Fma.MultiplyAdd(v1, Avx.LoadVector256(s + dim + c), acc);
            acc = Fma.MultiplyAdd(v2, Avx.LoadVector256(s + 2 * dim + c), acc);
            acc = Fma.MultiplyAdd(v3, Avx.LoadVector256(s + 3 * dim + c), acc);
            Avx.Store(destination + c, acc);
        }
#endif
    }
    public static void Sinkhorn4x4(float* logits, float* result)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Sinkhorn4x4(logits, result); return; }
#endif
        var kernelFp = Kernels.Active.Sinkhorn4x4;
        if ((void*)kernelFp != null)
        {
            InvokeSinkhorn4x4(kernelFp, logits, result);
            return;
        }
        if (!Simd.Enabled) { Scalar.Sinkhorn4x4(logits, result); return; }
#if NEEDLE_SIMD
        var r01 = Avx.LoadVector256(logits);
        var r23 = Avx.LoadVector256(logits + 8);
        var p01 = Avx.Max(r01, Avx.Permute(r01, 0xB1));
        p01 = Avx.Max(p01, Avx.Permute(p01, 0x4E));
        var p23 = Avx.Max(r23, Avx.Permute(r23, 0xB1));
        p23 = Avx.Max(p23, Avx.Permute(p23, 0x4E));
        var expBuf = stackalloc float[16];
        Avx.Store(expBuf, Avx.Subtract(r01, p01));
        Avx.Store(expBuf + 8, Avx.Subtract(r23, p23));
        for (var i = 0; i < 16; i++) expBuf[i] = MathF.Exp(expBuf[i]);
        var a01 = Avx.LoadVector256(expBuf);
        var a23 = Avx.LoadVector256(expBuf + 8);
        var zero = Vector128<float>.Zero;
        for (var iteration = 0; iteration < 20; iteration++)
        {
            var r1 = Avx.Add(a01, Avx.Permute(a01, 0x55));
            var r2 = Avx.Add(r1, Avx.Permute(a01, 0xAA));
            var r3 = Avx.Add(r2, Avx.Permute(a01, 0xFF));
            var rs01 = Avx.Permute(r3, 0x00);
            var s1 = Avx.Add(a23, Avx.Permute(a23, 0x55));
            var s2 = Avx.Add(s1, Avx.Permute(a23, 0xAA));
            var s3 = Avx.Add(s2, Avx.Permute(a23, 0xFF));
            var rs23 = Avx.Permute(s3, 0x00);
            a01 = Avx.Divide(a01, rs01);
            a23 = Avx.Divide(a23, rs23);
            var cs = Sse.Add(Sse.Add(Sse.Add(a01.GetLower(), a01.GetUpper()), a23.GetLower()), a23.GetUpper());
            var cs2 = Vector256.Create(cs, cs);
            var pos = Sse.CompareGreaterThan(cs, zero);
            if (Sse.MoveMask(pos) == 0xF)
            {
                a01 = Avx.Divide(a01, cs2);
                a23 = Avx.Divide(a23, cs2);
            }
            else
            {
                var pos2 = Vector256.Create(pos, pos);
                a01 = Avx.BlendVariable(a01, Avx.Divide(a01, cs2), pos2);
                a23 = Avx.BlendVariable(a23, Avx.Divide(a23, cs2), pos2);
            }
        }
        Avx.Store(result, a01);
        Avx.Store(result + 8, a23);
#endif
    }

#if NEEDLE_SIMD
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void RmsStats(float* x, int n, out float inverse, out float rawPeak)
    {
        if (Simd.UseNeon) { NeonPrepare.RmsStats(x, n, out inverse, out rawPeak); return; }
        var s0 = Vector256<float>.Zero;
        var s1 = Vector256<float>.Zero;
        var s2 = Vector256<float>.Zero;
        var s3 = Vector256<float>.Zero;
        var p0 = Vector256<float>.Zero;
        var p1 = Vector256<float>.Zero;
        var p2 = Vector256<float>.Zero;
        var p3 = Vector256<float>.Zero;
        var i = 0;
        var bulk = n & ~31;
        for (; i < bulk; i += 32)
        {
            var a0 = Avx.LoadVector256(x + i);
            var a1 = Avx.LoadVector256(x + i + 8);
            var a2 = Avx.LoadVector256(x + i + 16);
            var a3 = Avx.LoadVector256(x + i + 24);
            s0 = Fma.MultiplyAdd(a0, a0, s0);
            s1 = Fma.MultiplyAdd(a1, a1, s1);
            s2 = Fma.MultiplyAdd(a2, a2, s2);
            s3 = Fma.MultiplyAdd(a3, a3, s3);
            p0 = Avx.Max(p0, Vector256.Abs(a0));
            p1 = Avx.Max(p1, Vector256.Abs(a1));
            p2 = Avx.Max(p2, Vector256.Abs(a2));
            p3 = Avx.Max(p3, Vector256.Abs(a3));
        }
        for (; i + 16 <= n; i += 16)
        {
            var a = Avx.LoadVector256(x + i);
            var b = Avx.LoadVector256(x + i + 8);
            s0 = Fma.MultiplyAdd(a, a, s0);
            s1 = Fma.MultiplyAdd(b, b, s1);
            p0 = Avx.Max(p0, Vector256.Abs(a));
            p1 = Avx.Max(p1, Vector256.Abs(b));
        }
        if (i + 8 <= n)
        {
            var a = Avx.LoadVector256(x + i);
            s2 = Fma.MultiplyAdd(a, a, s2);
            p2 = Avx.Max(p2, Vector256.Abs(a));
            i += 8;
        }
        var tailSum = 0f;
        var tailPeak = 0f;
        for (; i < n; i++)
        {
            var v = x[i];
            tailSum = MathF.FusedMultiplyAdd(v, v, tailSum);
            tailPeak = MathF.Max(tailPeak, MathF.Abs(v));
        }
        var sum = Vector256.Sum(Avx.Add(Avx.Add(s0, s1), Avx.Add(s2, s3))) + tailSum;
        inverse = 1f / MathF.Sqrt(sum / n + RmsEpsilon);
        rawPeak = MathF.Max(FoldPeak(Avx.Max(Avx.Max(p0, p1), Avx.Max(p2, p3))), tailPeak);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void CenteredRmsStats(float* x, float* scale, int n, out float inverse, out float rawPeak)
    {
        var s0 = Vector256<float>.Zero;
        var s1 = Vector256<float>.Zero;
        var s2 = Vector256<float>.Zero;
        var s3 = Vector256<float>.Zero;
        var p0 = Vector256<float>.Zero;
        var p1 = Vector256<float>.Zero;
        var p2 = Vector256<float>.Zero;
        var p3 = Vector256<float>.Zero;
        var one = Vector256.Create(1f);
        var i = 0;
        var bulk = n & ~31;
        for (; i < bulk; i += 32)
        {
            var a0 = Avx.LoadVector256(x + i);
            var a1 = Avx.LoadVector256(x + i + 8);
            var a2 = Avx.LoadVector256(x + i + 16);
            var a3 = Avx.LoadVector256(x + i + 24);
            s0 = Fma.MultiplyAdd(a0, a0, s0);
            s1 = Fma.MultiplyAdd(a1, a1, s1);
            s2 = Fma.MultiplyAdd(a2, a2, s2);
            s3 = Fma.MultiplyAdd(a3, a3, s3);
            var t0 = Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i)), a0);
            var t1 = Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i + 8)), a1);
            var t2 = Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i + 16)), a2);
            var t3 = Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i + 24)), a3);
            p0 = Avx.Max(p0, Vector256.Abs(t0));
            p1 = Avx.Max(p1, Vector256.Abs(t1));
            p2 = Avx.Max(p2, Vector256.Abs(t2));
            p3 = Avx.Max(p3, Vector256.Abs(t3));
        }
        for (; i + 16 <= n; i += 16)
        {
            var a = Avx.LoadVector256(x + i);
            var b = Avx.LoadVector256(x + i + 8);
            s0 = Fma.MultiplyAdd(a, a, s0);
            s1 = Fma.MultiplyAdd(b, b, s1);
            p0 = Avx.Max(p0, Vector256.Abs(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i)), a)));
            p1 = Avx.Max(p1, Vector256.Abs(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i + 8)), b)));
        }
        if (i + 8 <= n)
        {
            var a = Avx.LoadVector256(x + i);
            s2 = Fma.MultiplyAdd(a, a, s2);
            p2 = Avx.Max(p2, Vector256.Abs(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i)), a)));
            i += 8;
        }
        var tailSum = 0f;
        var tailPeak = 0f;
        for (; i < n; i++)
        {
            var v = x[i];
            tailSum = MathF.FusedMultiplyAdd(v, v, tailSum);
            tailPeak = MathF.Max(tailPeak, MathF.Abs((1f + scale[i]) * v));
        }
        var sum = Vector256.Sum(Avx.Add(Avx.Add(s0, s1), Avx.Add(s2, s3))) + tailSum;
        inverse = 1f / MathF.Sqrt(sum / n + RmsEpsilon);
        rawPeak = MathF.Max(FoldPeak(Avx.Max(Avx.Max(p0, p1), Avx.Max(p2, p3))), tailPeak);
    }
#endif

    public static void RmsFakeA8(float* x, int n, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.RmsFakeA8(x, n, destination); return; }
#endif
        var kernelFp = Kernels.Active.RmsFakeA8;
        if ((void*)kernelFp != null)
        {
            InvokeRmsFakeA8(kernelFp, x, n, destination);
            return;
        }
        if (!Simd.Enabled) { Scalar.RmsFakeA8(x, n, destination); return; }
#if NEEDLE_SIMD
        RmsStats(x, n, out var inv, out var rawPeak);
        var peak = rawPeak * inv;
        var scale = peak > 0f ? peak / 127f : 1f;
        var vRmsInv = Vector256.Create(inv);
        var vA8Inv = Vector256.Create(1f / scale);
        var vA8Scale = Vector256.Create(scale);
        var low = Vector256.Create(-128f);
        var high = Vector256.Create(127f);
        var i = 0;
        for (; i + 16 <= n; i += 16)
        {
            var v0 = Avx.Multiply(Avx.LoadVector256(x + i), vRmsInv);
            var v1 = Avx.Multiply(Avx.LoadVector256(x + i + 8), vRmsInv);
            var q0 = Avx.RoundToNearestInteger(Avx.Multiply(v0, vA8Inv));
            var q1 = Avx.RoundToNearestInteger(Avx.Multiply(v1, vA8Inv));
            Avx.Store(destination + i, Avx.Multiply(Avx.Min(Avx.Max(q0, low), high), vA8Scale));
            Avx.Store(destination + i + 8, Avx.Multiply(Avx.Min(Avx.Max(q1, low), high), vA8Scale));
        }
        for (; i + 8 <= n; i += 8)
        {
            var v = Avx.Multiply(Avx.LoadVector256(x + i), vRmsInv);
            var q = Avx.RoundToNearestInteger(Avx.Multiply(v, vA8Inv));
            Avx.Store(destination + i, Avx.Multiply(Avx.Min(Avx.Max(q, low), high), vA8Scale));
        }
        var sInvA8 = 1f / scale;
        for (; i < n; i++)
        {
            var q = MathF.Round(x[i] * inv * sInvA8);
            destination[i] = MathF.Min(MathF.Max(q, -128f), 127f) * scale;
        }
#endif
    }

    public static void CenteredRmsFakeA8(float* x, float* scale, int n, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.CenteredRmsFakeA8(x, scale, n, destination); return; }
#endif
        var kernelFp = Kernels.Active.CenteredRmsFakeA8;
        if ((void*)kernelFp != null)
        {
            InvokeCenteredRmsFakeA8(kernelFp, x, scale, n, destination);
            return;
        }
        if (!Simd.Enabled) { Scalar.CenteredRmsFakeA8(x, scale, n, destination); return; }
#if NEEDLE_SIMD
        CenteredRmsStats(x, scale, n, out var inv, out var rawPeak);
        var peak = rawPeak * inv;
        var a8Scale = peak > 0f ? peak / 127f : 1f;
        var one = Vector256.Create(1f);
        var vRmsInv = Vector256.Create(inv);
        var vA8Inv = Vector256.Create(1f / a8Scale);
        var vA8Scale = Vector256.Create(a8Scale);
        var low = Vector256.Create(-128f);
        var high = Vector256.Create(127f);
        var i = 0;
        for (; i + 16 <= n; i += 16)
        {
            var v0 = Avx.Multiply(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i)), Avx.LoadVector256(x + i)), vRmsInv);
            var v1 = Avx.Multiply(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i + 8)), Avx.LoadVector256(x + i + 8)), vRmsInv);
            var q0 = Avx.RoundToNearestInteger(Avx.Multiply(v0, vA8Inv));
            var q1 = Avx.RoundToNearestInteger(Avx.Multiply(v1, vA8Inv));
            Avx.Store(destination + i, Avx.Multiply(Avx.Min(Avx.Max(q0, low), high), vA8Scale));
            Avx.Store(destination + i + 8, Avx.Multiply(Avx.Min(Avx.Max(q1, low), high), vA8Scale));
        }
        for (; i + 8 <= n; i += 8)
        {
            var v = Avx.Multiply(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i)), Avx.LoadVector256(x + i)), vRmsInv);
            var q = Avx.RoundToNearestInteger(Avx.Multiply(v, vA8Inv));
            Avx.Store(destination + i, Avx.Multiply(Avx.Min(Avx.Max(q, low), high), vA8Scale));
        }
        var sInvA8 = 1f / a8Scale;
        for (; i < n; i++)
        {
            var q = MathF.Round((1f + scale[i]) * x[i] * inv * sInvA8);
            destination[i] = MathF.Min(MathF.Max(q, -128f), 127f) * a8Scale;
        }
#endif
    }

    public static void GateFakeA8(float* x, float* gate, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.GateFakeA8(x, gate, n); return; }
#endif
        var kernelFp = Kernels.Active.GateFakeA8;
        if ((void*)kernelFp != null)
        {
            InvokeGateFakeA8(kernelFp, x, gate, n);
            return;
        }
        if (!Simd.Enabled) { Scalar.GateFakeA8(x, gate, n); return; }
#if NEEDLE_SIMD
        var peak0 = Vector256<float>.Zero;
        var peak1 = Vector256<float>.Zero;
        var i = 0;
        for (; i + 16 <= n; i += 16)
        {
            var v0 = Avx.Multiply(Avx.LoadVector256(x + i), Sigmoid(Avx.LoadVector256(gate + i)));
            var v1 = Avx.Multiply(Avx.LoadVector256(x + i + 8), Sigmoid(Avx.LoadVector256(gate + i + 8)));
            Avx.Store(x + i, v0);
            Avx.Store(x + i + 8, v1);
            peak0 = Avx.Max(peak0, Vector256.Abs(v0));
            peak1 = Avx.Max(peak1, Vector256.Abs(v1));
        }
        for (; i + 8 <= n; i += 8)
        {
            var v = Avx.Multiply(Avx.LoadVector256(x + i), Sigmoid(Avx.LoadVector256(gate + i)));
            Avx.Store(x + i, v);
            peak0 = Avx.Max(peak0, Vector256.Abs(v));
        }
        var peak = FoldPeak(Avx.Max(peak0, peak1));
        for (; i < n; i++)
        {
            var v = x[i] * Sigmoid(gate[i]);
            x[i] = v;
            peak = MathF.Max(peak, MathF.Abs(v));
        }
        FinishFakeA8(x, n, peak);
#endif
    }

#if NEEDLE_SIMD
    internal static float FoldPeak(Vector256<float> peak)
    {
        var folded = Sse.Max(peak.GetLower(), peak.GetUpper());
        folded = Sse.Max(folded, Sse.Shuffle(folded, folded, 0b01_00_11_10));
        return Sse.Max(folded, Sse.Shuffle(folded, folded, 0b10_11_00_01)).ToScalar();
    }

    static void FinishFakeA8(float* x, int n, float peak)
    {
        var scale = peak > 0f ? peak / 127f : 1f;
        var inverse = Vector256.Create(1f / scale);
        var sc = Vector256.Create(scale);
        var low = Vector256.Create(-128f);
        var high = Vector256.Create(127f);
        var i = 0;
        for (; i + 16 <= n; i += 16)
        {
            var q0 = Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(x + i), inverse));
            var q1 = Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(x + i + 8), inverse));
            Avx.Store(x + i, Avx.Multiply(Avx.Min(Avx.Max(q0, low), high), sc));
            Avx.Store(x + i + 8, Avx.Multiply(Avx.Min(Avx.Max(q1, low), high), sc));
        }
        for (; i + 8 <= n; i += 8)
        {
            var q = Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(x + i), inverse));
            Avx.Store(x + i, Avx.Multiply(Avx.Min(Avx.Max(q, low), high), sc));
        }
        var sInv = 1f / scale;
        for (; i < n; i++) x[i] = MathF.Min(MathF.Max(MathF.Round(x[i] * sInv), -128f), 127f) * scale;
    }
#endif

    public static void CombineDiff4x4(float* s, int dim, float* mix, float* x, float* u, float* postGate, float* destination)
    {
#if NEEDLE_SIMD
        if (!Simd.UseNeon && Simd.Enabled && (void*)Kernels.Active.CombineDiff4 == null)
        {
            CombineDiff4x4Range(s, dim, mix, x, u, postGate, destination, 0, dim);
            return;
        }
#endif
        for (var i = 0; i < 4; i++)
            CombineDiff4(s, dim, mix[i * 4], mix[i * 4 + 1], mix[i * 4 + 2], mix[i * 4 + 3], x, u, postGate[i], destination + (long)i * dim);
    }

#if NEEDLE_SIMD
    internal static void CombineDiff4x4Range(float* s, int dim, float* mix, float* x, float* u, float* postGate, float* destination, int offsetStart, int offsetEnd)
    {
        var s1 = s + dim;
        var s2 = s + 2 * dim;
        var s3 = s + 3 * dim;
        var d1 = destination + dim;
        var d2 = destination + 2 * dim;
        var d3 = destination + 3 * dim;
        var p0 = Avx.BroadcastScalarToVector256(postGate);
        var p1 = Avx.BroadcastScalarToVector256(postGate + 1);
        var p2 = Avx.BroadcastScalarToVector256(postGate + 2);
        var p3 = Avx.BroadcastScalarToVector256(postGate + 3);
        var c = offsetStart;
        for (; c + 8 <= offsetEnd; c += 8)
        {
            var diff = Avx.Subtract(Avx.LoadVector256(x + c), Avx.LoadVector256(u + c));
            var v0 = Avx.LoadVector256(s + c);
            var v1 = Avx.LoadVector256(s1 + c);
            var v2 = Avx.LoadVector256(s2 + c);
            var v3 = Avx.LoadVector256(s3 + c);
            var a0 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix), v0, Avx.Multiply(p0, diff));
            a0 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 1), v1, a0);
            a0 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 2), v2, a0);
            a0 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 3), v3, a0);
            Avx.Store(destination + c, a0);

            var a1 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 4), v0, Avx.Multiply(p1, diff));
            a1 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 5), v1, a1);
            a1 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 6), v2, a1);
            a1 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 7), v3, a1);
            Avx.Store(d1 + c, a1);

            var a2 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 8), v0, Avx.Multiply(p2, diff));
            a2 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 9), v1, a2);
            a2 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 10), v2, a2);
            a2 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 11), v3, a2);
            Avx.Store(d2 + c, a2);

            var a3 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 12), v0, Avx.Multiply(p3, diff));
            a3 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 13), v1, a3);
            a3 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 14), v2, a3);
            a3 = Fma.MultiplyAdd(Avx.BroadcastScalarToVector256(mix + 15), v3, a3);
            Avx.Store(d3 + c, a3);
        }
        for (; c < offsetEnd; c++)
        {
            var diff = x[c] - u[c];
            var v0 = s[c];
            var v1 = s1[c];
            var v2 = s2[c];
            var v3 = s3[c];
            destination[c] = postGate[0] * diff + mix[0] * v0 + mix[1] * v1 + mix[2] * v2 + mix[3] * v3;
            d1[c] = postGate[1] * diff + mix[4] * v0 + mix[5] * v1 + mix[6] * v2 + mix[7] * v3;
            d2[c] = postGate[2] * diff + mix[8] * v0 + mix[9] * v1 + mix[10] * v2 + mix[11] * v3;
            d3[c] = postGate[3] * diff + mix[12] * v0 + mix[13] * v1 + mix[14] * v2 + mix[15] * v3;
        }
    }

#endif

    public static void CombineDiff4(float* s, int dim, float w0, float w1, float w2, float w3, float* x, float* u, float post, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.CombineDiff4(s, dim, w0, w1, w2, w3, x, u, post, destination); return; }
#endif
        var kernelFp = Kernels.Active.CombineDiff4;
        if ((void*)kernelFp != null)
        {
            InvokeCombineDiff4(kernelFp, s, dim, w0, w1, w2, w3, x, u, post, destination);
            return;
        }
        if (!Simd.Enabled) { Scalar.CombineDiff4(s, dim, w0, w1, w2, w3, x, u, post, destination); return; }
#if NEEDLE_SIMD
        var v0 = Vector256.Create(w0);
        var v1 = Vector256.Create(w1);
        var v2 = Vector256.Create(w2);
        var v3 = Vector256.Create(w3);
        var vp = Vector256.Create(post);
        for (var c = 0; c < dim; c += 8)
        {
            var acc = Avx.Multiply(vp, Avx.Subtract(Avx.LoadVector256(x + c), Avx.LoadVector256(u + c)));
            acc = Fma.MultiplyAdd(v0, Avx.LoadVector256(s + c), acc);
            acc = Fma.MultiplyAdd(v1, Avx.LoadVector256(s + dim + c), acc);
            acc = Fma.MultiplyAdd(v2, Avx.LoadVector256(s + 2 * dim + c), acc);
            acc = Fma.MultiplyAdd(v3, Avx.LoadVector256(s + 3 * dim + c), acc);
            Avx.Store(destination + c, acc);
        }
#endif
    }

    public static void PostNormAdd(float* residual, float* x, float* scale, float gate, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.PostNormAdd(residual, x, scale, gate, n); return; }
#endif
        var kernelFp = Kernels.Active.PostNormAdd;
        if ((void*)kernelFp != null)
        {
            InvokePostNormAdd(kernelFp, residual, x, scale, gate, n);
            return;
        }
        if (!Simd.Enabled) { Scalar.PostNormAdd(residual, x, scale, gate, n); return; }
#if NEEDLE_SIMD
        var scalarInv = InverseRms(x, n);
        var inverse = Vector256.Create(scalarInv);
        var one = Vector256.Create(1f);
        var g = Vector256.Create(gate);
        var i = 0;
        for (; i + 8 <= n; i += 8)
            Avx.Store(residual + i, Fma.MultiplyAdd(g,
                Avx.Multiply(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i)), Avx.LoadVector256(x + i)), inverse),
                Avx.LoadVector256(residual + i)));
        for (; i < n; i++) residual[i] += gate * (1f + scale[i]) * x[i] * scalarInv;
#endif
    }

    public static void Subtract(float* a, float* b, int n, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Subtract(a, b, n, destination); return; }
#endif
        if (!Simd.Enabled) { Scalar.Subtract(a, b, n, destination); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.Subtract(a, b, n, destination); return; }
        var i = 0;
        for (; i + 8 <= n; i += 8) Avx.Store(destination + i, Avx.Subtract(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
        for (; i < n; i++) destination[i] = a[i] - b[i];
#endif
    }

    public static void MultiplyAccumulate(float* destination, float* a, float* b, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.MultiplyAccumulate(destination, a, b, n); return; }
#endif
        var kernelFp = Kernels.Active.MultiplyAccumulate;
        if ((void*)kernelFp != null)
        {
            InvokeMultiplyAccumulate(kernelFp, destination, a, b, n);
            return;
        }
        if (!Simd.Enabled) { Scalar.MultiplyAccumulate(destination, a, b, n); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.MultiplyAccumulate(destination, a, b, n); return; }
        var i = 0;
        for (; i + 8 <= n; i += 8)
            Avx.Store(destination + i, Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), Avx.LoadVector256(destination + i)));
        for (; i < n; i++) destination[i] += a[i] * b[i];
#endif
    }

    public static void ScaleAdd(float* destination, float scale, float* x, float xScale, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.ScaleAdd(destination, scale, x, xScale, n); return; }
#endif
        var kernelFp = Kernels.Active.ScaleAdd;
        if ((void*)kernelFp != null)
        {
            InvokeScaleAdd(kernelFp, destination, scale, x, xScale, n);
            return;
        }
        if (!Simd.Enabled) { Scalar.ScaleAdd(destination, scale, x, xScale, n); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.ScaleAdd(destination, scale, x, xScale, n); return; }
        var s = Vector256.Create(scale);
        var xs = Vector256.Create(xScale);
        var i = 0;
        for (; i + 8 <= n; i += 8) Avx.Store(destination + i, Fma.MultiplyAdd(xs, Avx.LoadVector256(x + i), Avx.Multiply(s, Avx.LoadVector256(destination + i))));
        for (; i < n; i++) destination[i] = scale * destination[i] + xScale * x[i];
#endif
    }

    public static void AddScaled(float* destination, float* x, float scale, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.AddScaled(destination, x, scale, n); return; }
        #endif
        var kernelFp = Kernels.Active.AddScaled;
        if ((void*)kernelFp != null)
        {
            InvokeAddScaled(kernelFp, destination, x, scale, n);
            return;
        }
        if (!Simd.Enabled) { Scalar.AddScaled(destination, x, scale, n); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.AddScaled(destination, x, scale, n); return; }
        var s = Vector256.Create(scale);
        var i = 0;
        for (; i + 8 <= n; i += 8) Avx.Store(destination + i, Fma.MultiplyAdd(s, Avx.LoadVector256(x + i), Avx.LoadVector256(destination + i)));
        for (; i < n; i++) destination[i] += scale * x[i];
#endif
    }

    public static void Zero(float* x, int n)
    {
        for (var i = 0; i < n; i++) x[i] = 0f;
    }

    public static void Glu(float* value, float* gate, float* destination, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Glu(value, gate, destination, n); return; }
#endif
        if (!Simd.Enabled) { Scalar.Glu(value, gate, destination, n); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.Glu(value, gate, destination, n); return; }
        var i = 0;
        for (; i + 8 <= n; i += 8)
            Avx.Store(destination + i, Avx.Multiply(Avx.LoadVector256(value + i), Sigmoid(Avx.LoadVector256(gate + i))));
        for (; i < n; i++) destination[i] = value[i] * Sigmoid(gate[i]);
#endif
    }

    public static void Taps3(float* taps, int width, float* current, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Taps3(taps, width, current, destination); return; }
        #endif
        var kernelFp = Kernels.Active.Taps3;
        if ((void*)kernelFp != null)
        {
            InvokeTaps3(kernelFp, taps, width, current, destination);
            return;
        }
        if (!Simd.Enabled) { Scalar.Taps3(taps, width, current, destination); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.Taps3(taps, width, current, destination); return; }
        for (var c = 0; c < width; c += 8)
        {
            var acc = Avx.Multiply(Avx.LoadVector256(taps + c), Avx.LoadVector256(current + c));
            acc = Fma.MultiplyAdd(Avx.LoadVector256(taps + width + c), Avx.LoadVector256(current + c - width), acc);
            acc = Fma.MultiplyAdd(Avx.LoadVector256(taps + 2 * width + c), Avx.LoadVector256(current + c - 2 * width), acc);
            Avx.Store(destination + c, acc);
        }
#endif
    }

    public static int ArgMax(float* x, int n)
    {
#if NEEDLE_SIMD
        if (Simd.Enabled && !Simd.UseNeon && n >= 32)
        {
            var p0 = Vector256.Create(float.NegativeInfinity);
            var p1 = p0;
            var p2 = p0;
            var p3 = p0;
            var i = 0;
            var bulk32 = n & ~31;
            for (; i < bulk32; i += 32)
            {
                p0 = Avx.Max(p0, Avx.LoadVector256(x + i));
                p1 = Avx.Max(p1, Avx.LoadVector256(x + i + 8));
                p2 = Avx.Max(p2, Avx.LoadVector256(x + i + 16));
                p3 = Avx.Max(p3, Avx.LoadVector256(x + i + 24));
            }
            p0 = Avx.Max(Avx.Max(p0, p1), Avx.Max(p2, p3));
            var bulk8 = n & ~7;
            for (; i < bulk8; i += 8) p0 = Avx.Max(p0, Avx.LoadVector256(x + i));
            var peak = Max8(p0);
            for (; i < n; i++) peak = MathF.Max(peak, x[i]);
            if (float.IsFinite(peak))
            {
                var vPeak = Vector256.Create(peak);
                for (i = 0; i < bulk8; i += 8)
                {
                    var mask = Avx.MoveMask(Avx.CompareEqual(Avx.LoadVector256(x + i), vPeak));
                    if (mask != 0) return i + System.Numerics.BitOperations.TrailingZeroCount((uint)mask);
                }
                for (; i < n; i++) if (x[i] == peak) return i;
            }
        }
#endif
        var best = 0;
        for (var i = 1; i < n; i++) if (x[i] > x[best]) best = i;
        return best;
    }

    public static void TopKLogSoftmax(float* x, int n, int k, int* indices, float* logProbs, float* probs, bool maskSpecial023)
    {
        var topVal = stackalloc float[k];
        var topIdx = stackalloc int[k];
        for (var i = 0; i < k; i++) { topVal[i] = float.NegativeInfinity; topIdx[i] = -1; }
        var minVal = float.NegativeInfinity;
        var minIdx = int.MaxValue;
        var start = 0;
        if (maskSpecial023 && n > 1)
        {
            topVal[0] = x[1];
            topIdx[0] = 1;
            if (k == 1) { minVal = topVal[0]; minIdx = 1; }
            start = 4;
        }
#if NEEDLE_SIMD
        if (Simd.Enabled && !Simd.UseNeon && n >= 16)
        {
            var headEnd = Math.Min(8, n);
            for (var i = start; i < headEnd; i++)
            {
                var v = x[i];
                if (v > minVal || (v == minVal && i < minIdx))
                {
                    var pos = k - 1;
                    while (pos > 0 && (v > topVal[pos - 1] || (v == topVal[pos - 1] && i < topIdx[pos - 1])))
                    {
                        topVal[pos] = topVal[pos - 1];
                        topIdx[pos] = topIdx[pos - 1];
                        pos--;
                    }
                    topVal[pos] = v;
                    topIdx[pos] = i;
                    minVal = topVal[k - 1];
                    minIdx = topIdx[k - 1];
                }
            }
            var bulk8 = n & ~7;
            var vMin = Vector256.Create(minVal);
            var idx = 8;
            for (; idx < bulk8; idx += 8)
            {
                var vec = Avx.LoadVector256(x + idx);
                var mask = Avx.MoveMask(Avx.CompareGreaterThan(vec, vMin));
                if (mask == 0) continue;
                for (var b = 0; b < 8; b++)
                {
                    if ((mask & (1 << b)) == 0) continue;
                    var j = idx + b;
                    var v = x[j];
                    if (v > minVal || (v == minVal && j < minIdx))
                    {
                        var pos = k - 1;
                        while (pos > 0 && (v > topVal[pos - 1] || (v == topVal[pos - 1] && j < topIdx[pos - 1])))
                        {
                            topVal[pos] = topVal[pos - 1];
                            topIdx[pos] = topIdx[pos - 1];
                            pos--;
                        }
                        topVal[pos] = v;
                        topIdx[pos] = j;
                        minVal = topVal[k - 1];
                        minIdx = topIdx[k - 1];
                    }
                }
                vMin = Vector256.Create(minVal);
            }
            start = idx;
        }
#endif
        for (var i = start; i < n; i++)
        {
            if (maskSpecial023 && (i == 0 || i == 2 || i == 3)) continue;
            var v = x[i];
            if (v > minVal || (v == minVal && i < minIdx))
            {
                var pos = k - 1;
                while (pos > 0 && (v > topVal[pos - 1] || (v == topVal[pos - 1] && i < topIdx[pos - 1])))
                {
                    topVal[pos] = topVal[pos - 1];
                    topIdx[pos] = topIdx[pos - 1];
                    pos--;
                }
                topVal[pos] = v;
                topIdx[pos] = i;
                minVal = topVal[k - 1];
                minIdx = topIdx[k - 1];
            }
        }
        var peak = topVal[0];
        float logShift;
#if NEEDLE_SIMD
        if (Simd.Enabled && !Simd.UseNeon && n >= 16)
        {
            var shift = Vector256.Create(peak);
            var first = Avx.LoadVector256(x);
            if (maskSpecial023) first = Avx.Blend(first, Vector256.Create(-1e30f), 0x0D);
            var s0 = Exp(Avx.Subtract(first, shift));
            var s1 = Exp(Avx.Subtract(Avx.LoadVector256(x + 8), shift));
            var bulk8 = n & ~7;
            var i = 16;
            for (; i < bulk8 - 8; i += 16)
            {
                s0 = Avx.Add(s0, Exp(Avx.Subtract(Avx.LoadVector256(x + i), shift)));
                s1 = Avx.Add(s1, Exp(Avx.Subtract(Avx.LoadVector256(x + i + 8), shift)));
            }
            for (; i < bulk8; i += 8) s0 = Avx.Add(s0, Exp(Avx.Subtract(Avx.LoadVector256(x + i), shift)));
            var total = Vector256.Sum(Avx.Add(s0, s1));
            for (; i < n; i++) total += MathF.Exp(x[i] - peak);
            logShift = peak + MathF.Log(total);
        }
        else
#endif
        {
            double total = 0;
            for (var i = 0; i < n; i++)
            {
                var v = maskSpecial023 && (i == 0 || i == 2 || i == 3) ? -1e30f : x[i];
                total += Math.Exp(v - peak);
            }
            logShift = peak + (float)Math.Log(total);
        }
        for (var m = 0; m < k; m++)
        {
            indices[m] = topIdx[m];
            var lp = topVal[m] - logShift;
            logProbs[m] = lp;
            if (probs != null) probs[m] = MathF.Exp(lp);
        }
    }

    public static int ArgMaxMasked(float* x, uint* mask, int words, int length)
    {
#if NEEDLE_SIMD
        if (!Simd.UseNeon && Simd.Enabled) return ArgMaxMaskedAvx(x, mask, words, length);
#endif
        var kernelFp = Kernels.Active.ArgMaxMasked;
        if ((void*)kernelFp != null)
        {
            return InvokeArgMaxMasked(kernelFp, x, mask, words, length);
        }
        return ArgMaxMaskedScalar(x, mask, words, length);
    }

    static int ArgMaxMaskedScalar(float* x, uint* mask, int words, int length)
    {
        var best = -1;
        var peak = float.NegativeInfinity;
        for (var w = 0; w < words; w++)
        {
            var bits = mask[w];
            if (bits == 0) continue;
            var bottom = w * 32;
            if (bits == 0xFFFFFFFFu)
            {
                var top = bottom + 32 <= length ? bottom + 32 : length;
                for (var id = bottom; id < top; id++)
                {
                    var value = x[id];
                    if (value > peak) { peak = value; best = id; }
                }
                continue;
            }
            while (bits != 0)
            {
                var id = bottom + Bits.TrailingZeros(bits);
                bits &= bits - 1;
                if (id >= length) continue;
                var value = x[id];
                if (value > peak) { peak = value; best = id; }
            }
        }
        return best;
    }

#if NEEDLE_SIMD
    static int ArgMaxMaskedAvx(float* x, uint* mask, int words, int length)
    {
        var best = -1;
        var peak = float.NegativeInfinity;
        for (var w = 0; w < words; w++)
        {
            var bits = mask[w];
            if (bits == 0) continue;
            var bottom = w * 32;
            if (bits == 0xFFFFFFFFu && bottom + 32 <= length)
            {
                var m0 = Avx.Max(Avx.LoadVector256(x + bottom), Avx.LoadVector256(x + bottom + 8));
                var m1 = Avx.Max(Avx.LoadVector256(x + bottom + 16), Avx.LoadVector256(x + bottom + 24));
                var local = Max8(Avx.Max(m0, m1));
                if (local > peak)
                {
                    peak = local;
                    best = bottom + LaneOfMax32(x + bottom, local);
                }
                continue;
            }
            while (bits != 0)
            {
                var id = bottom + Bits.TrailingZeros(bits);
                bits &= bits - 1;
                if (id >= length) continue;
                var value = x[id];
                if (value > peak) { peak = value; best = id; }
            }
        }
        return best;
    }
#endif

    public static int ArgMaxMaskedWithProbability(float* x, uint* mask, int words, int length, float* scratch, out float probability)
    {
#if NEEDLE_SIMD
        if (Simd.Enabled && !Simd.UseNeon && (void*)Kernels.Active.BestProbability == null)
            return ArgMaxMaskedWithProbabilityAvx(x, mask, words, length, scratch, out probability);
#endif
        var best = -1;
        var count = 0;
        var peak = float.NegativeInfinity;
        for (var w = 0; w < words; w++)
        {
            var bits = mask[w];
            while (bits != 0)
            {
                var id = w * 32 + Bits.TrailingZeros(bits);
                bits &= bits - 1;
                if (id >= length) continue;
                var value = x[id];
                scratch[count++] = value;
                if (value > peak) { peak = value; best = id; }
            }
        }
        if (best < 0)
        {
            probability = 0f;
            return -1;
        }
        probability = BestProbability(scratch, count, peak);
        return best;
    }

#if NEEDLE_SIMD
    static int ArgMaxMaskedWithProbabilityAvx(float* x, uint* mask, int words, int length, float* scratch, out float probability)
    {
        var best = -1;
        var peak = float.NegativeInfinity;
        for (var w = 0; w < words; w++)
        {
            var bits = mask[w];
            if (bits == 0) continue;
            var bottom = w * 32;
            if (bits == 0xFFFFFFFFu && bottom + 32 <= length)
            {
                var m0 = Avx.Max(Avx.LoadVector256(x + bottom), Avx.LoadVector256(x + bottom + 8));
                var m1 = Avx.Max(Avx.LoadVector256(x + bottom + 16), Avx.LoadVector256(x + bottom + 24));
                var local = Max8(Avx.Max(m0, m1));
                if (local > peak)
                {
                    peak = local;
                    best = bottom + LaneOfMax32(x + bottom, local);
                }
                continue;
            }
            while (bits != 0)
            {
                var id = bottom + Bits.TrailingZeros(bits);
                bits &= bits - 1;
                if (id >= length) continue;
                var value = x[id];
                if (value > peak) { peak = value; best = id; }
            }
        }
        if (best < 0)
        {
            probability = 0f;
            return -1;
        }
        var shift = Vector256.Create(peak);
        var floor = Vector256.Create(peak - 32f);
        var sum0 = Vector256<float>.Zero;
        var sum1 = Vector256<float>.Zero;
        var sum2 = Vector256<float>.Zero;
        var sum3 = Vector256<float>.Zero;
        var tailCount = 0;
        for (var w = 0; w < words; w++)
        {
            var bits = mask[w];
            if (bits == 0) continue;
            var bottom = w * 32;
            if (bits == 0xFFFFFFFFu && bottom + 32 <= length)
            {
                var v0 = Avx.LoadVector256(x + bottom);
                var v1 = Avx.LoadVector256(x + bottom + 8);
                var v2 = Avx.LoadVector256(x + bottom + 16);
                var v3 = Avx.LoadVector256(x + bottom + 24);
                var bmax = Avx.Max(Avx.Max(v0, v1), Avx.Max(v2, v3));
                if (Avx.MoveMask(Avx.Compare(bmax, floor, FloatComparisonMode.OrderedGreaterThanOrEqualNonSignaling)) == 0)
                    continue;
                sum0 = Avx.Add(sum0, Exp(Avx.Subtract(v0, shift)));
                sum1 = Avx.Add(sum1, Exp(Avx.Subtract(v1, shift)));
                sum2 = Avx.Add(sum2, Exp(Avx.Subtract(v2, shift)));
                sum3 = Avx.Add(sum3, Exp(Avx.Subtract(v3, shift)));
                continue;
            }
            while (bits != 0)
            {
                var id = bottom + Bits.TrailingZeros(bits);
                bits &= bits - 1;
                if (id >= length) continue;
                scratch[tailCount++] = x[id];
            }
        }
        var tailEnd = tailCount & ~7;
        var i = 0;
        for (; i < tailEnd; i += 8)
            sum0 = Avx.Add(sum0, Exp(Avx.Subtract(Avx.LoadVector256(scratch + i), shift)));
        var total = Vector256.Sum(Avx.Add(Avx.Add(sum0, sum1), Avx.Add(sum2, sum3)));
        for (; i < tailCount; i++) total += MathF.Exp(scratch[i] - peak);
        probability = 1f / total;
        return best;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float Max8(Vector256<float> v)
    {
        var halves = Sse.Max(v.GetLower(), v.GetUpper());
        var pairs = Sse.Max(halves, Sse.Shuffle(halves, halves, 0x4E));
        return Sse.Max(pairs, Sse.Shuffle(pairs, pairs, 0xB1)).ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int LaneOfMax32(float* x, float peak)
    {
        var p = Vector256.Create(peak);
        for (var q = 0; q < 4; q++)
        {
            var m = Avx.MoveMask(Avx.Compare(Avx.LoadVector256(x + q * 8), p, FloatComparisonMode.OrderedEqualNonSignaling));
            if (m != 0) return q * 8 + Bits.TrailingZeros((uint)m);
        }
        return 0;
    }
#endif

    public static float BestProbability(float* values, int count, float best)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) return Neon.BestProbability(values, count, best);
#endif
        var kernelFp = Kernels.Active.BestProbability;
        if ((void*)kernelFp != null)
        {
            return InvokeBestProbability(kernelFp, values, count, best);
        }
        if (!Simd.Enabled) return ScalarBestProbability(values, count, best);
#if NEEDLE_SIMD
        var shift = Vector256.Create(best);
        var sum0 = Vector256<float>.Zero;
        var sum1 = sum0;
        var sum2 = sum0;
        var sum3 = sum0;
        var bulk32 = count & ~31;
        var i = 0;
        for (; i < bulk32; i += 32)
        {
            sum0 = Avx.Add(sum0, Exp(Avx.Subtract(Avx.LoadVector256(values + i), shift)));
            sum1 = Avx.Add(sum1, Exp(Avx.Subtract(Avx.LoadVector256(values + i + 8), shift)));
            sum2 = Avx.Add(sum2, Exp(Avx.Subtract(Avx.LoadVector256(values + i + 16), shift)));
            sum3 = Avx.Add(sum3, Exp(Avx.Subtract(Avx.LoadVector256(values + i + 24), shift)));
        }
        var vectorEnd = count & ~7;
        for (; i < vectorEnd; i += 8)
            sum0 = Avx.Add(sum0, Exp(Avx.Subtract(Avx.LoadVector256(values + i), shift)));
        var total = Vector256.Sum(Avx.Add(Avx.Add(sum0, sum1), Avx.Add(sum2, sum3)));
        for (; i < count; i++) total += MathF.Exp(values[i] - best);
        return 1f / total;
#else
        return ScalarBestProbability(values, count, best);
#endif
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static float InvokeDot(nint fp, float* a, float* b, int n) =>
        ((delegate* unmanaged[Cdecl]<float*, float*, int, float>)(void*)fp)(a, b, n);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeRmsUnit(nint fp, float* x, int n, float* destination) =>
        ((delegate* unmanaged[Cdecl]<float*, int, float*, void>)(void*)fp)(x, n, destination);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeZeroCenteredRms(nint fp, float* x, float* scale, int n, float* destination) =>
        ((delegate* unmanaged[Cdecl]<float*, float*, int, float*, void>)(void*)fp)(x, scale, n, destination);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeSoftmax(nint fp, float* s, int n) =>
        ((delegate* unmanaged[Cdecl]<float*, int, void>)(void*)fp)(s, n);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeRope(nint fp, float* x, int heads, int headDim, float* cos, float* sin) =>
        ((delegate* unmanaged[Cdecl]<float*, int, int, float*, float*, void>)(void*)fp)(x, heads, headDim, cos, sin);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeMix4(nint fp, float* s, int dim, float w0, float w1, float w2, float w3, float* destination) =>
        ((delegate* unmanaged[Cdecl]<float*, int, float, float, float, float, float*, void>)(void*)fp)(s, dim, w0, w1, w2, w3, destination);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeCombine4(nint fp, float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination) =>
        ((delegate* unmanaged[Cdecl]<float*, int, float, float, float, float, float*, float, float*, void>)(void*)fp)(s, dim, w0, w1, w2, w3, y, post, destination);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeSinkhorn4x4(nint fp, float* logits, float* result) =>
        ((delegate* unmanaged[Cdecl]<float*, float*, void>)(void*)fp)(logits, result);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeRmsFakeA8(nint fp, float* x, int n, float* destination) =>
        ((delegate* unmanaged[Cdecl]<float*, int, float*, void>)(void*)fp)(x, n, destination);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeCenteredRmsFakeA8(nint fp, float* x, float* scale, int n, float* destination) =>
        ((delegate* unmanaged[Cdecl]<float*, float*, int, float*, void>)(void*)fp)(x, scale, n, destination);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeGateFakeA8(nint fp, float* x, float* gate, int n) =>
        ((delegate* unmanaged[Cdecl]<float*, float*, int, void>)(void*)fp)(x, gate, n);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeCombineDiff4(nint fp, float* s, int dim, float w0, float w1, float w2, float w3, float* x, float* u, float post, float* destination) =>
        ((delegate* unmanaged[Cdecl]<float*, int, float, float, float, float, float*, float*, float, float*, void>)(void*)fp)(s, dim, w0, w1, w2, w3, x, u, post, destination);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokePostNormAdd(nint fp, float* residual, float* x, float* scale, float gate, int n) =>
        ((delegate* unmanaged[Cdecl]<float*, float*, float*, float, int, void>)(void*)fp)(residual, x, scale, gate, n);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeMultiplyAccumulate(nint fp, float* destination, float* a, float* b, int n) =>
        ((delegate* unmanaged[Cdecl]<float*, float*, float*, int, void>)(void*)fp)(destination, a, b, n);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeScaleAdd(nint fp, float* destination, float scale, float* x, float xScale, int n) =>
        ((delegate* unmanaged[Cdecl]<float*, float, float*, float, int, void>)(void*)fp)(destination, scale, x, xScale, n);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeAddScaled(nint fp, float* destination, float* x, float scale, int n) =>
        ((delegate* unmanaged[Cdecl]<float*, float*, float, int, void>)(void*)fp)(destination, x, scale, n);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void InvokeTaps3(nint fp, float* taps, int width, float* current, float* destination) =>
        ((delegate* unmanaged[Cdecl]<float*, int, float*, float*, void>)(void*)fp)(taps, width, current, destination);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static int InvokeArgMaxMasked(nint fp, float* x, uint* mask, int words, int length) =>
        ((delegate* unmanaged[Cdecl]<float*, uint*, int, int, int>)(void*)fp)(x, mask, words, length);
    [MethodImpl(MethodImplOptions.NoInlining)]
    static float InvokeBestProbability(nint fp, float* values, int count, float best) =>
        ((delegate* unmanaged[Cdecl]<float*, int, float, float>)(void*)fp)(values, count, best);

    static float ScalarBestProbability(float* values, int count, float best)
    {
        var total = 0f;
        for (var i = 0; i < count; i++) total += MathF.Exp(values[i] - best);
        return 1f / total;
    }
}
