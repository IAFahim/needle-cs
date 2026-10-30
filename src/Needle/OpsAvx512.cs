#if NEEDLE_SIMD
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Needle;

internal static unsafe class OpsAvx512
{
    const float RmsEpsilon = 1e-6f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector512<float> Exp(Vector512<float> x) =>
        Vector512.Create(Ops.Exp(x.GetLower()), Ops.Exp(x.GetUpper()));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector512<float> Sigmoid(Vector512<float> x) =>
        Vector512.Create(Ops.Sigmoid(x.GetLower()), Ops.Sigmoid(x.GetUpper()));

    public static void Silu(float* x, int n)
    {
        var i = 0;
        for (; i + 16 <= n; i += 16)
        {
            var v = Avx512F.LoadVector512(x + i);
            Avx512F.Store(x + i, Avx512F.Multiply(v, Sigmoid(v)));
        }
        for (; i < n; i += 8)
        {
            var v = Avx.LoadVector256(x + i);
            Avx.Store(x + i, Avx.Multiply(v, Ops.Sigmoid(v)));
        }
    }

    public static void MultiplyBySigmoid(float* x, float* gate, int n)
    {
        var i = 0;
        for (; i + 16 <= n; i += 16)
            Avx512F.Store(x + i, Avx512F.Multiply(Avx512F.LoadVector512(x + i), Sigmoid(Avx512F.LoadVector512(gate + i))));
        for (; i < n; i += 8)
            Avx.Store(x + i, Avx.Multiply(Avx.LoadVector256(x + i), Ops.Sigmoid(Avx.LoadVector256(gate + i))));
    }

    public static float SumOfSquares(float* x, int n)
    {
        var s0 = Vector512<float>.Zero;
        var s1 = Vector512<float>.Zero;
        for (var i = 0; i < n; i += 32)
        {
            s0 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(x + i), Avx512F.LoadVector512(x + i), s0);
            s1 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(x + i + 16), Avx512F.LoadVector512(x + i + 16), s1);
        }
        var folded = Avx512F.Add(s0, s1);
        return Vector256.Sum(Avx.Add(folded.GetLower(), folded.GetUpper()));
    }

    public static void RmsUnit(float* x, int n, float* destination)
    {
        var inverse = 1f / MathF.Sqrt(SumOfSquares(x, n) / n + RmsEpsilon);
        var wide = Vector512.Create(inverse);
        var i = 0;
        for (; i + 16 <= n; i += 16)
            Avx512F.Store(destination + i, Avx512F.Multiply(Avx512F.LoadVector512(x + i), wide));
        var tail = Vector256.Create(inverse);
        for (; i < n; i += 8) Avx.Store(destination + i, Avx.Multiply(Avx.LoadVector256(x + i), tail));
    }

    public static float Dot(float* a, float* b, int n)
    {
        var s0 = Vector512<float>.Zero;
        var s1 = Vector512<float>.Zero;
        for (var i = 0; i < n; i += 32)
        {
            s0 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(a + i), Avx512F.LoadVector512(b + i), s0);
            s1 = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(a + i + 16), Avx512F.LoadVector512(b + i + 16), s1);
        }
        var folded = Avx512F.Add(s0, s1);
        return Vector256.Sum(Avx.Add(folded.GetLower(), folded.GetUpper()));
    }

    public static void Softmax(float* s, int n)
    {
        var peak = float.NegativeInfinity;
        for (var i = 0; i < n; i++) peak = MathF.Max(peak, s[i]);
        var shift = Vector512.Create(peak);
        var sum = Vector512<float>.Zero;
        var vectorEnd = n & ~15;
        var i2 = 0;
        for (; i2 < vectorEnd; i2 += 16)
        {
            var e = Exp(Avx512F.Subtract(Avx512F.LoadVector512(s + i2), shift));
            Avx512F.Store(s + i2, e);
            sum = Avx512F.Add(sum, e);
        }
        var total = Vector256.Sum(Avx.Add(sum.GetLower(), sum.GetUpper()));
        for (; i2 < n; i2++)
        {
            s[i2] = MathF.Exp(s[i2] - peak);
            total += s[i2];
        }
        var inverse = Vector512.Create(1f / total);
        for (i2 = 0; i2 < vectorEnd; i2 += 16)
            Avx512F.Store(s + i2, Avx512F.Multiply(Avx512F.LoadVector512(s + i2), inverse));
        for (; i2 < n; i2++) s[i2] /= total;
    }

    public static void Mix4(float* s, int dim, float w0, float w1, float w2, float w3, float* destination)
    {
        var v0 = Vector512.Create(w0);
        var v1 = Vector512.Create(w1);
        var v2 = Vector512.Create(w2);
        var v3 = Vector512.Create(w3);
        var c = 0;
        for (; c + 16 <= dim; c += 16)
        {
            var acc = Avx512F.Multiply(v0, Avx512F.LoadVector512(s + c));
            acc = Avx512F.FusedMultiplyAdd(v1, Avx512F.LoadVector512(s + dim + c), acc);
            acc = Avx512F.FusedMultiplyAdd(v2, Avx512F.LoadVector512(s + 2 * dim + c), acc);
            acc = Avx512F.FusedMultiplyAdd(v3, Avx512F.LoadVector512(s + 3 * dim + c), acc);
            Avx512F.Store(destination + c, acc);
        }
        var t0 = Vector256.Create(w0);
        var t1 = Vector256.Create(w1);
        var t2 = Vector256.Create(w2);
        var t3 = Vector256.Create(w3);
        for (; c < dim; c += 8)
        {
            var acc = Avx.Multiply(t0, Avx.LoadVector256(s + c));
            acc = Fma.MultiplyAdd(t1, Avx.LoadVector256(s + dim + c), acc);
            acc = Fma.MultiplyAdd(t2, Avx.LoadVector256(s + 2 * dim + c), acc);
            acc = Fma.MultiplyAdd(t3, Avx.LoadVector256(s + 3 * dim + c), acc);
            Avx.Store(destination + c, acc);
        }
    }

    public static void ZeroCenteredRms(float* x, float* scale, int n, float* destination)
    {
        var inverse = 1f / MathF.Sqrt(SumOfSquares(x, n) / n + RmsEpsilon);
        var wide = Vector512.Create(inverse);
        var one = Vector512.Create(1f);
        var i = 0;
        for (; i + 16 <= n; i += 16)
            Avx512F.Store(destination + i, Avx512F.Multiply(Avx512F.Multiply(Avx512F.Add(one, Avx512F.LoadVector512(scale + i)), Avx512F.LoadVector512(x + i)), wide));
        var tailInverse = Vector256.Create(inverse);
        var tailOne = Vector256.Create(1f);
        for (; i < n; i += 8)
            Avx.Store(destination + i, Avx.Multiply(Avx.Multiply(Avx.Add(tailOne, Avx.LoadVector256(scale + i)), Avx.LoadVector256(x + i)), tailInverse));
    }

    public static void Combine4(float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination)
    {
        var v0 = Vector512.Create(w0);
        var v1 = Vector512.Create(w1);
        var v2 = Vector512.Create(w2);
        var v3 = Vector512.Create(w3);
        var vp = Vector512.Create(post);
        var c = 0;
        for (; c + 16 <= dim; c += 16)
        {
            var acc = Avx512F.Multiply(vp, Avx512F.LoadVector512(y + c));
            acc = Avx512F.FusedMultiplyAdd(v0, Avx512F.LoadVector512(s + c), acc);
            acc = Avx512F.FusedMultiplyAdd(v1, Avx512F.LoadVector512(s + dim + c), acc);
            acc = Avx512F.FusedMultiplyAdd(v2, Avx512F.LoadVector512(s + 2 * dim + c), acc);
            acc = Avx512F.FusedMultiplyAdd(v3, Avx512F.LoadVector512(s + 3 * dim + c), acc);
            Avx512F.Store(destination + c, acc);
        }
        var t0 = Vector256.Create(w0);
        var t1 = Vector256.Create(w1);
        var t2 = Vector256.Create(w2);
        var t3 = Vector256.Create(w3);
        var tp = Vector256.Create(post);
        for (; c < dim; c += 8)
        {
            var acc = Avx.Multiply(tp, Avx.LoadVector256(y + c));
            acc = Fma.MultiplyAdd(t0, Avx.LoadVector256(s + c), acc);
            acc = Fma.MultiplyAdd(t1, Avx.LoadVector256(s + dim + c), acc);
            acc = Fma.MultiplyAdd(t2, Avx.LoadVector256(s + 2 * dim + c), acc);
            acc = Fma.MultiplyAdd(t3, Avx.LoadVector256(s + 3 * dim + c), acc);
            Avx.Store(destination + c, acc);
        }
    }

    public static void Subtract(float* a, float* b, int n, float* destination)
    {
        var i = 0;
        for (; i + 16 <= n; i += 16)
            Avx512F.Store(destination + i, Avx512F.Subtract(Avx512F.LoadVector512(a + i), Avx512F.LoadVector512(b + i)));
        for (; i < n; i += 8) Avx.Store(destination + i, Avx.Subtract(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
    }

    public static void MultiplyAccumulate(float* destination, float* a, float* b, int n)
    {
        var i = 0;
        for (; i + 16 <= n; i += 16)
            Avx512F.Store(destination + i, Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(a + i), Avx512F.LoadVector512(b + i), Avx512F.LoadVector512(destination + i)));
        for (; i < n; i += 8)
            Avx.Store(destination + i, Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), Avx.LoadVector256(destination + i)));
    }

    public static void ScaleAdd(float* destination, float scale, float* x, float xScale, int n)
    {
        var s = Vector512.Create(scale);
        var xs = Vector512.Create(xScale);
        var i = 0;
        for (; i + 16 <= n; i += 16)
            Avx512F.Store(destination + i, Avx512F.FusedMultiplyAdd(xs, Avx512F.LoadVector512(x + i), Avx512F.Multiply(s, Avx512F.LoadVector512(destination + i))));
        var t = Vector256.Create(scale);
        var tx = Vector256.Create(xScale);
        for (; i < n; i += 8) Avx.Store(destination + i, Fma.MultiplyAdd(tx, Avx.LoadVector256(x + i), Avx.Multiply(t, Avx.LoadVector256(destination + i))));
    }

    public static void AddScaled(float* destination, float* x, float scale, int n)
    {
        var s = Vector512.Create(scale);
        var i = 0;
        for (; i + 16 <= n; i += 16)
            Avx512F.Store(destination + i, Avx512F.FusedMultiplyAdd(s, Avx512F.LoadVector512(x + i), Avx512F.LoadVector512(destination + i)));
        var t = Vector256.Create(scale);
        for (; i < n; i += 8) Avx.Store(destination + i, Fma.MultiplyAdd(t, Avx.LoadVector256(x + i), Avx.LoadVector256(destination + i)));
    }

    public static void Glu(float* value, float* gate, float* destination, int n)
    {
        var i = 0;
        for (; i + 16 <= n; i += 16)
            Avx512F.Store(destination + i, Avx512F.Multiply(Avx512F.LoadVector512(value + i), Sigmoid(Avx512F.LoadVector512(gate + i))));
        for (; i < n; i += 8)
            Avx.Store(destination + i, Avx.Multiply(Avx.LoadVector256(value + i), Ops.Sigmoid(Avx.LoadVector256(gate + i))));
    }

    public static void Taps3(float* taps, int width, float* current, float* destination)
    {
        var c = 0;
        for (; c + 16 <= width; c += 16)
        {
            var acc = Avx512F.Multiply(Avx512F.LoadVector512(taps + c), Avx512F.LoadVector512(current + c));
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(taps + width + c), Avx512F.LoadVector512(current + c - width), acc);
            acc = Avx512F.FusedMultiplyAdd(Avx512F.LoadVector512(taps + 2 * width + c), Avx512F.LoadVector512(current + c - 2 * width), acc);
            Avx512F.Store(destination + c, acc);
        }
        for (; c < width; c += 8)
        {
            var acc = Avx.Multiply(Avx.LoadVector256(taps + c), Avx.LoadVector256(current + c));
            acc = Fma.MultiplyAdd(Avx.LoadVector256(taps + width + c), Avx.LoadVector256(current + c - width), acc);
            acc = Fma.MultiplyAdd(Avx.LoadVector256(taps + 2 * width + c), Avx.LoadVector256(current + c - 2 * width), acc);
            Avx.Store(destination + c, acc);
        }
    }
}
#endif
