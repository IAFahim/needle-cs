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
        for (var i = 0; i < n; i += 8)
        {
            var v = Avx.LoadVector256(x + i);
            Avx.Store(x + i, Avx.Multiply(v, Sigmoid(v)));
        }
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
        for (var i = 0; i < n; i += 8)
            Avx.Store(x + i, Avx.Multiply(Avx.LoadVector256(x + i), Sigmoid(Avx.LoadVector256(gate + i))));
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
        for (var i = 0; i < n; i += 16)
        {
            var a = Avx.LoadVector256(x + i);
            var b = Avx.LoadVector256(x + i + 8);
            s0 = Fma.MultiplyAdd(a, a, s0);
            s1 = Fma.MultiplyAdd(b, b, s1);
        }
        return Vector256.Sum(Avx.Add(s0, s1));
#endif
    }

    public static float Dot(float* a, float* b, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { return Neon.Dot(a, b, n); }
#endif
        if (!Simd.Enabled) return Scalar.Dot(a, b, n);
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) return OpsAvx512.Dot(a, b, n);
        var s0 = Vector256<float>.Zero;
        var s1 = Vector256<float>.Zero;
        for (var i = 0; i < n; i += 16)
        {
            s0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), s0);
            s1 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8), s1);
        }
        return Vector256.Sum(Avx.Add(s0, s1));
#endif
    }

    public static float InverseRms(float* x, int n) => 1f / MathF.Sqrt(SumOfSquares(x, n) / n + RmsEpsilon);

    public static void RmsUnit(float* x, int n, float* destination)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.RmsUnit(x, n, destination); return; }
#endif
        if (!Simd.Enabled) { var scalarInverse = InverseRms(x, n); for (var i = 0; i < n; i++) destination[i] = x[i] * scalarInverse; return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.RmsUnit(x, n, destination); return; }
        var inverse = Vector256.Create(InverseRms(x, n));
        for (var i = 0; i < n; i += 8) Avx.Store(destination + i, Avx.Multiply(Avx.LoadVector256(x + i), inverse));
#endif
    }

    public static void ZeroCenteredRms(float* x, float* scale, int n, float* destination)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.ZeroCenteredRms(x, scale, n, destination); return; }
#endif
        if (!Simd.Enabled) { var scalarInverse = InverseRms(x, n); for (var i = 0; i < n; i++) destination[i] = (1f + scale[i]) * x[i] * scalarInverse; return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.ZeroCenteredRms(x, scale, n, destination); return; }
        var inverse = Vector256.Create(InverseRms(x, n));
        var one = Vector256.Create(1f);
        for (var i = 0; i < n; i += 8)
            Avx.Store(destination + i, Avx.Multiply(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i)), Avx.LoadVector256(x + i)), inverse));
#endif
    }

    public static void Softmax(float* s, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Softmax(s, n); return; }
#endif
        if (!Simd.Enabled) { Scalar.Softmax(s, n); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.Softmax(s, n); return; }
        var peak = float.NegativeInfinity;
        for (var i = 0; i < n; i++) peak = MathF.Max(peak, s[i]);
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

    public static void Rope(float* x, int heads, int headDim, float* cos, float* sin)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Rope(x, heads, headDim, cos, sin); return; }
#endif
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
        if (!Simd.Enabled) { Scalar.Sinkhorn4x4(logits, result); return; }
#if NEEDLE_SIMD
        var scratch = stackalloc float[4];
        for (var r = 0; r < 4; r++)
        {
            var row = Sse.LoadVector128(logits + r * 4);
            Sse.Store(scratch, row);
            var shift = Vector128.Create(MathF.Max(MathF.Max(scratch[0], scratch[1]), MathF.Max(scratch[2], scratch[3])));
            Sse.Store(scratch, Sse.Subtract(row, shift));
            for (var c = 0; c < 4; c++) result[r * 4 + c] = MathF.Exp(scratch[c]);
        }
        for (var iteration = 0; iteration < 20; iteration++)
        {
            NormRow4(result);
            NormRow4(result + 4);
            NormRow4(result + 8);
            NormRow4(result + 12);
            var a0 = Sse.LoadVector128(result);
            var a1 = Sse.LoadVector128(result + 4);
            var a2 = Sse.LoadVector128(result + 8);
            var a3 = Sse.LoadVector128(result + 12);
            var total = Sse.Add(Sse.Add(Sse.Add(a0, a1), a2), a3);
            Sse.Store(scratch, total);
            if (scratch[0] > 0f && scratch[1] > 0f && scratch[2] > 0f && scratch[3] > 0f)
            {
                Sse.Store(result, Sse.Divide(a0, total));
                Sse.Store(result + 4, Sse.Divide(a1, total));
                Sse.Store(result + 8, Sse.Divide(a2, total));
                Sse.Store(result + 12, Sse.Divide(a3, total));
            }
            else
            {
                for (var c = 0; c < 4; c++)
                {
                    var sum = result[c] + result[4 + c] + result[8 + c] + result[12 + c];
                    if (sum > 0f) for (var r = 0; r < 4; r++) result[r * 4 + c] /= sum;
                }
            }
        }
#endif
    }

#if NEEDLE_SIMD
    static void NormRow4(float* row)
    {
        var v = Sse.LoadVector128(row);
        var lanes = stackalloc float[4];
        Sse.Store(lanes, v);
        Sse.Store(row, Sse.Divide(v, Vector128.Create(lanes[0] + lanes[1] + lanes[2] + lanes[3])));
    }
#endif

    public static void RmsFakeA8(float* x, int n, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.RmsFakeA8(x, n, destination); return; }
#endif
        if (!Simd.Enabled) { Scalar.RmsFakeA8(x, n, destination); return; }
#if NEEDLE_SIMD
        var inverse = Vector256.Create(InverseRms(x, n));
        var peak = Vector256<float>.Zero;
        for (var i = 0; i < n; i += 8)
        {
            var v = Avx.Multiply(Avx.LoadVector256(x + i), inverse);
            Avx.Store(destination + i, v);
            peak = Avx.Max(peak, Vector256.Abs(v));
        }
        FinishFakeA8(destination, n, FoldPeak(peak));
#endif
    }

    public static void CenteredRmsFakeA8(float* x, float* scale, int n, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.CenteredRmsFakeA8(x, scale, n, destination); return; }
#endif
        if (!Simd.Enabled) { Scalar.CenteredRmsFakeA8(x, scale, n, destination); return; }
#if NEEDLE_SIMD
        var inverse = Vector256.Create(InverseRms(x, n));
        var one = Vector256.Create(1f);
        var peak = Vector256<float>.Zero;
        for (var i = 0; i < n; i += 8)
        {
            var v = Avx.Multiply(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i)), Avx.LoadVector256(x + i)), inverse);
            Avx.Store(destination + i, v);
            peak = Avx.Max(peak, Vector256.Abs(v));
        }
        FinishFakeA8(destination, n, FoldPeak(peak));
#endif
    }

    public static void GateFakeA8(float* x, float* gate, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.GateFakeA8(x, gate, n); return; }
#endif
        if (!Simd.Enabled) { Scalar.GateFakeA8(x, gate, n); return; }
#if NEEDLE_SIMD
        var peak = Vector256<float>.Zero;
        for (var i = 0; i < n; i += 8)
        {
            var v = Avx.Multiply(Avx.LoadVector256(x + i), Sigmoid(Avx.LoadVector256(gate + i)));
            Avx.Store(x + i, v);
            peak = Avx.Max(peak, Vector256.Abs(v));
        }
        FinishFakeA8(x, n, FoldPeak(peak));
#endif
    }

#if NEEDLE_SIMD
    static float FoldPeak(Vector256<float> peak)
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
        for (var i = 0; i < n; i += 8)
        {
            var q = Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(x + i), inverse));
            Avx.Store(x + i, Avx.Multiply(Avx.Min(Avx.Max(q, low), high), sc));
        }
    }
#endif

    public static void CombineDiff4(float* s, int dim, float w0, float w1, float w2, float w3, float* x, float* u, float post, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.CombineDiff4(s, dim, w0, w1, w2, w3, x, u, post, destination); return; }
#endif
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
        if (!Simd.Enabled) { Scalar.PostNormAdd(residual, x, scale, gate, n); return; }
#if NEEDLE_SIMD
        var inverse = Vector256.Create(InverseRms(x, n));
        var one = Vector256.Create(1f);
        var g = Vector256.Create(gate);
        for (var i = 0; i < n; i += 8)
            Avx.Store(residual + i, Fma.MultiplyAdd(g,
                Avx.Multiply(Avx.Multiply(Avx.Add(one, Avx.LoadVector256(scale + i)), Avx.LoadVector256(x + i)), inverse),
                Avx.LoadVector256(residual + i)));
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
        for (var i = 0; i < n; i += 8) Avx.Store(destination + i, Avx.Subtract(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
#endif
    }

    public static void MultiplyAccumulate(float* destination, float* a, float* b, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.MultiplyAccumulate(destination, a, b, n); return; }
#endif
        if (!Simd.Enabled) { Scalar.MultiplyAccumulate(destination, a, b, n); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.MultiplyAccumulate(destination, a, b, n); return; }
        for (var i = 0; i < n; i += 8)
            Avx.Store(destination + i, Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), Avx.LoadVector256(destination + i)));
#endif
    }

    public static void ScaleAdd(float* destination, float scale, float* x, float xScale, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.ScaleAdd(destination, scale, x, xScale, n); return; }
#endif
        if (!Simd.Enabled) { Scalar.ScaleAdd(destination, scale, x, xScale, n); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.ScaleAdd(destination, scale, x, xScale, n); return; }
        var s = Vector256.Create(scale);
        var xs = Vector256.Create(xScale);
        for (var i = 0; i < n; i += 8) Avx.Store(destination + i, Fma.MultiplyAdd(xs, Avx.LoadVector256(x + i), Avx.Multiply(s, Avx.LoadVector256(destination + i))));
#endif
    }

    public static void AddScaled(float* destination, float* x, float scale, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.AddScaled(destination, x, scale, n); return; }
#endif
        if (!Simd.Enabled) { Scalar.AddScaled(destination, x, scale, n); return; }
#if NEEDLE_SIMD
        if (Tiers.Active == CpuTier.Avx512Bw) { OpsAvx512.AddScaled(destination, x, scale, n); return; }
        var s = Vector256.Create(scale);
        for (var i = 0; i < n; i += 8) Avx.Store(destination + i, Fma.MultiplyAdd(s, Avx.LoadVector256(x + i), Avx.LoadVector256(destination + i)));
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
        for (var i = 0; i < n; i += 8)
            Avx.Store(destination + i, Avx.Multiply(Avx.LoadVector256(value + i), Sigmoid(Avx.LoadVector256(gate + i))));
#endif
    }

    public static void Taps3(float* taps, int width, float* current, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Taps3(taps, width, current, destination); return; }
#endif
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
        var best = 0;
        for (var i = 1; i < n; i++) if (x[i] > x[best]) best = i;
        return best;
    }

    // Masked argmax over a bit mask of allowed ids (`words` 32-bit words, one
    // bit per logit). Zero words are skipped; fully-set words reduce 8 logits
    // at a time with AVX while sparse words visit only their set bits.
    // Returns -1 when no bit is set. `length` bounds the logit array; padding
    // bits past it are never set by our masks and are ignored defensively.
    public static int ArgMaxMasked(float* x, uint* mask, int words, int length)
    {
#if NEEDLE_SIMD
        if (!Simd.UseNeon && Simd.Enabled) return ArgMaxMaskedAvx(x, mask, words, length);
#endif
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
                for (var q = 0; q < 4; q++)
                {
                    var at = x + bottom + q * 8;
                    var local = Max8(Avx.LoadVector256(at));
                    if (local > peak)
                    {
                        peak = local;
                        best = bottom + q * 8 + LaneOfMax(at, local);
                    }
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float Max8(Vector256<float> v)
    {
        var halves = Sse.Max(v.GetLower(), v.GetUpper());
        var pairs = Sse.Max(halves, Sse.Shuffle(halves, halves, 0x4E));
        return Sse.Max(pairs, Sse.Shuffle(pairs, pairs, 0xB1)).ToScalar();
    }

    static int LaneOfMax(float* x, float peak)
    {
        for (var i = 0; i < 8; i++) if (x[i] == peak) return i;
        return 0;
    }
#endif

    // Probability of the best (maximum) entry: 1 / sum(exp(v - best)).
    // The caller passes the maximum of `values[0..count)`; the exp-sum is
    // vectorized and only the best probability is produced, unlike Softmax
    // which normalizes every entry. `count` must be at least 1.
    public static float BestProbability(float* values, int count, float best)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) return Neon.BestProbability(values, count, best);
#endif
        if (!Simd.Enabled) return ScalarBestProbability(values, count, best);
#if NEEDLE_SIMD
        var shift = Vector256.Create(best);
        var sum = Vector256<float>.Zero;
        var vectorEnd = count & ~7;
        var i = 0;
        for (; i < vectorEnd; i += 8)
            sum = Avx.Add(sum, Exp(Avx.Subtract(Avx.LoadVector256(values + i), shift)));
        var total = Vector256.Sum(sum);
        for (; i < count; i++) total += MathF.Exp(values[i] - best);
        return 1f / total;
#else
        return ScalarBestProbability(values, count, best);
#endif
    }

    static float ScalarBestProbability(float* values, int count, float best)
    {
        var total = 0f;
        for (var i = 0; i < count; i++) total += MathF.Exp(values[i] - best);
        return 1f / total;
    }
}
