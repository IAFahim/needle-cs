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

    public static void Subtract(float* a, float* b, int n, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Subtract(a, b, n, destination); return; }
#endif
        if (!Simd.Enabled) { Scalar.Subtract(a, b, n, destination); return; }
#if NEEDLE_SIMD
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
}
