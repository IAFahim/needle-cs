#if NEEDLE_SIMD
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Needle;

internal static unsafe class Neon
{
    const float HadamardScale = 0.08838834764831845f;
    const float RmsEpsilon = 1e-6f;
    const float InverseSqrtQkDim = 0.14433756729740643f;

    static readonly bool UseDotProduct = Dp.IsSupported && Environment.GetEnvironmentVariable("NEEDLE_NODP") == null;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Exp(Vector128<float> x)
    {
        x = AdvSimd.Min(AdvSimd.Max(x, Vector128.Create(-87.3f)), Vector128.Create(88.7f));
        var n = AdvSimd.RoundToNearest(AdvSimd.Multiply(x, Vector128.Create(1.44269504f)));
        var r = AdvSimd.FusedMultiplySubtract(x, n, Vector128.Create(0.693359375f));
        r = AdvSimd.FusedMultiplySubtract(r, n, Vector128.Create(-2.12194440e-4f));
        var p = Vector128.Create(1.9875691500e-4f);
        p = AdvSimd.FusedMultiplyAdd(Vector128.Create(1.3981999507e-3f), r, p);
        p = AdvSimd.FusedMultiplyAdd(Vector128.Create(8.3334519073e-3f), r, p);
        p = AdvSimd.FusedMultiplyAdd(Vector128.Create(4.1665795894e-2f), r, p);
        p = AdvSimd.FusedMultiplyAdd(Vector128.Create(1.6666665459e-1f), r, p);
        p = AdvSimd.FusedMultiplyAdd(Vector128.Create(5.0000001201e-1f), r, p);
        var y = AdvSimd.FusedMultiplyAdd(AdvSimd.Add(r, Vector128.Create(1f)), AdvSimd.Multiply(r, r), p);
        var exponent = AdvSimd.ShiftLeftLogical(
            AdvSimd.Add(AdvSimd.ConvertToInt32RoundToEven(n), Vector128.Create(127)).AsUInt32(), 23);
        return AdvSimd.Multiply(y, exponent.AsSingle());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> RoundToNearestOf(Vector128<float> x) => AdvSimd.RoundToNearest(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> FusedMultiplySubtractOf(Vector128<float> a, Vector128<float> b, float c) =>
        AdvSimd.FusedMultiplySubtract(a, b, Vector128.Create(c));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Sigmoid(Vector128<float> x) =>
        AdvSimd.Arm64.Divide(Vector128.Create(1f), AdvSimd.Add(Vector128.Create(1f), Exp(AdvSimd.Negate(x))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HorizontalSum(Vector128<float> v)
    {
        var pairs = AdvSimd.AddPairwise(v.GetLower(), v.GetUpper());
        return AdvSimd.AddPairwise(pairs, pairs).ToScalar();
    }

    public static void Silu(float* x, int n)
    {
        var i = 0;
        for (; i + 4 <= n; i += 4)
        {
            var v = AdvSimd.LoadVector128(x + i);
            AdvSimd.Store(x + i, AdvSimd.Multiply(v, Sigmoid(v)));
        }
        for (; i < n; i++) x[i] *= Ops.Sigmoid(x[i]);
    }

    public static void MultiplyBySigmoid(float* x, float* gate, int n)
    {
        var i = 0;
        for (; i + 4 <= n; i += 4)
            AdvSimd.Store(x + i, AdvSimd.Multiply(AdvSimd.LoadVector128(x + i), Sigmoid(AdvSimd.LoadVector128(gate + i))));
        for (; i < n; i++) x[i] *= Ops.Sigmoid(gate[i]);
    }

    public static float SumOfSquares(float* x, int n)
    {
        var s0 = Vector128<float>.Zero;
        var s1 = Vector128<float>.Zero;
        var i = 0;
        for (; i + 8 <= n; i += 8)
        {
            var a = AdvSimd.LoadVector128(x + i);
            var b = AdvSimd.LoadVector128(x + i + 4);
            s0 = AdvSimd.FusedMultiplyAdd(s0, a, a);
            s1 = AdvSimd.FusedMultiplyAdd(s1, b, b);
        }
        var tail = 0f;
        for (; i < n; i++) tail = MathF.FusedMultiplyAdd(x[i], x[i], tail);
        return HorizontalSum(AdvSimd.Add(s0, s1)) + tail;
    }

    public static float Dot(float* a, float* b, int n)
    {
        var s0 = Vector128<float>.Zero;
        var s1 = Vector128<float>.Zero;
        var i = 0;
        for (; i + 8 <= n; i += 8)
        {
            s0 = AdvSimd.FusedMultiplyAdd(s0, AdvSimd.LoadVector128(a + i), AdvSimd.LoadVector128(b + i));
            s1 = AdvSimd.FusedMultiplyAdd(s1, AdvSimd.LoadVector128(a + i + 4), AdvSimd.LoadVector128(b + i + 4));
        }
        var tail = 0f;
        for (; i < n; i++) tail = MathF.FusedMultiplyAdd(a[i], b[i], tail);
        return HorizontalSum(AdvSimd.Add(s0, s1)) + tail;
    }

    public static void RmsUnit(float* x, int n, float* destination)
    {
        var scalarInv = 1f / MathF.Sqrt(SumOfSquares(x, n) / n + RmsEpsilon);
        var inverse = Vector128.Create(scalarInv);
        var i = 0;
        for (; i + 4 <= n; i += 4) AdvSimd.Store(destination + i, AdvSimd.Multiply(AdvSimd.LoadVector128(x + i), inverse));
        for (; i < n; i++) destination[i] = x[i] * scalarInv;
    }

    public static void ZeroCenteredRms(float* x, float* scale, int n, float* destination)
    {
        var scalarInv = 1f / MathF.Sqrt(SumOfSquares(x, n) / n + RmsEpsilon);
        var inverse = Vector128.Create(scalarInv);
        var one = Vector128.Create(1f);
        var i = 0;
        for (; i + 4 <= n; i += 4)
            AdvSimd.Store(destination + i, AdvSimd.Multiply(AdvSimd.Multiply(AdvSimd.Add(one, AdvSimd.LoadVector128(scale + i)), AdvSimd.LoadVector128(x + i)), inverse));
        for (; i < n; i++) destination[i] = (1f + scale[i]) * x[i] * scalarInv;
    }

    // The largest of n floats. Max is exact and order-free (it propagates NaN and ranks +0 above -0,
    // as MathF.Max does), so the vector scan returns the same value as a scalar one.
    public static float Peak(float* s, int n)
    {
        var p0 = Vector128.Create(float.NegativeInfinity);
        var p1 = p0;
        var i = 0;
        for (; i + 8 <= n; i += 8)
        {
            p0 = AdvSimd.Max(p0, AdvSimd.LoadVector128(s + i));
            p1 = AdvSimd.Max(p1, AdvSimd.LoadVector128(s + i + 4));
        }
        if (i + 4 <= n)
        {
            p0 = AdvSimd.Max(p0, AdvSimd.LoadVector128(s + i));
            i += 4;
        }
        var peak = AdvSimd.Arm64.MaxAcross(AdvSimd.Max(p0, p1)).ToScalar();
        for (; i < n; i++) peak = MathF.Max(peak, s[i]);
        return peak;
    }

    public static void Softmax(float* s, int n)
    {
        var peak = Peak(s, n);
        var shift = Vector128.Create(peak);
        var sum = Vector128<float>.Zero;
        var vectorEnd = n & ~3;
        var i2 = 0;
        for (; i2 < vectorEnd; i2 += 4)
        {
            var e = Exp(AdvSimd.Subtract(AdvSimd.LoadVector128(s + i2), shift));
            AdvSimd.Store(s + i2, e);
            sum = AdvSimd.Add(sum, e);
        }
        var total = HorizontalSum(sum);
        for (; i2 < n; i2++)
        {
            s[i2] = MathF.Exp(s[i2] - peak);
            total += s[i2];
        }
        var inverse = Vector128.Create(1f / total);
        for (i2 = 0; i2 < vectorEnd; i2 += 4) AdvSimd.Store(s + i2, AdvSimd.Multiply(AdvSimd.LoadVector128(s + i2), inverse));
        for (; i2 < n; i2++) s[i2] /= total;
    }

    public static float BestProbability(float* values, int count, float best)
    {
        var shift = Vector128.Create(best);
        var sum = Vector128<float>.Zero;
        var vectorEnd = count & ~3;
        var i = 0;
        for (; i < vectorEnd; i += 4)
            sum = AdvSimd.Add(sum, Exp(AdvSimd.Subtract(AdvSimd.LoadVector128(values + i), shift)));
        var total = HorizontalSum(sum);
        for (; i < count; i++) total += MathF.Exp(values[i] - best);
        return 1f / total;
    }

    public static void Rope(float* x, int heads, int headDim, float* cos, float* sin)
    {
        var half = headDim / 2;
        for (var h = 0; h < heads; h++)
        {
            var lo = x + h * headDim;
            var hi = lo + half;
            for (var i = 0; i < half; i += 4)
            {
                var c = AdvSimd.LoadVector128(cos + i);
                var s = AdvSimd.LoadVector128(sin + i);
                var a = AdvSimd.LoadVector128(lo + i);
                var b = AdvSimd.LoadVector128(hi + i);
                AdvSimd.Store(lo + i, AdvSimd.FusedMultiplySubtract(AdvSimd.Multiply(a, c), b, s));
                AdvSimd.Store(hi + i, AdvSimd.FusedMultiplyAdd(AdvSimd.Multiply(b, c), s, a));
            }
        }
    }

    public static void Mix4(float* s, int dim, float w0, float w1, float w2, float w3, float* destination)
    {
        var v0 = Vector128.Create(w0);
        var v1 = Vector128.Create(w1);
        var v2 = Vector128.Create(w2);
        var v3 = Vector128.Create(w3);
        for (var c = 0; c < dim; c += 4)
        {
            var acc = AdvSimd.Multiply(v0, AdvSimd.LoadVector128(s + c));
            acc = AdvSimd.FusedMultiplyAdd(acc, v1, AdvSimd.LoadVector128(s + dim + c));
            acc = AdvSimd.FusedMultiplyAdd(acc, v2, AdvSimd.LoadVector128(s + 2 * dim + c));
            acc = AdvSimd.FusedMultiplyAdd(acc, v3, AdvSimd.LoadVector128(s + 3 * dim + c));
            AdvSimd.Store(destination + c, acc);
        }
    }

    public static void Combine4(float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination)
    {
        var v0 = Vector128.Create(w0);
        var v1 = Vector128.Create(w1);
        var v2 = Vector128.Create(w2);
        var v3 = Vector128.Create(w3);
        var vp = Vector128.Create(post);
        for (var c = 0; c < dim; c += 4)
        {
            var acc = AdvSimd.Multiply(vp, AdvSimd.LoadVector128(y + c));
            acc = AdvSimd.FusedMultiplyAdd(acc, v0, AdvSimd.LoadVector128(s + c));
            acc = AdvSimd.FusedMultiplyAdd(acc, v1, AdvSimd.LoadVector128(s + dim + c));
            acc = AdvSimd.FusedMultiplyAdd(acc, v2, AdvSimd.LoadVector128(s + 2 * dim + c));
            acc = AdvSimd.FusedMultiplyAdd(acc, v3, AdvSimd.LoadVector128(s + 3 * dim + c));
            AdvSimd.Store(destination + c, acc);
        }
    }

    public static void Subtract(float* a, float* b, int n, float* destination)
    {
        var i = 0;
        for (; i + 4 <= n; i += 4) AdvSimd.Store(destination + i, AdvSimd.Subtract(AdvSimd.LoadVector128(a + i), AdvSimd.LoadVector128(b + i)));
        for (; i < n; i++) destination[i] = a[i] - b[i];
    }

    public static void MultiplyAccumulate(float* destination, float* a, float* b, int n)
    {
        var i = 0;
        for (; i + 4 <= n; i += 4)
            AdvSimd.Store(destination + i, AdvSimd.FusedMultiplyAdd(AdvSimd.LoadVector128(destination + i), AdvSimd.LoadVector128(a + i), AdvSimd.LoadVector128(b + i)));
        for (; i < n; i++) destination[i] += a[i] * b[i];
    }

    public static void ScaleAdd(float* destination, float scale, float* x, float xScale, int n)
    {
        var s = Vector128.Create(scale);
        var xs = Vector128.Create(xScale);
        var i = 0;
        for (; i + 4 <= n; i += 4)
            AdvSimd.Store(destination + i, AdvSimd.FusedMultiplyAdd(AdvSimd.Multiply(s, AdvSimd.LoadVector128(destination + i)), xs, AdvSimd.LoadVector128(x + i)));
        for (; i < n; i++) destination[i] = scale * destination[i] + xScale * x[i];
    }

    public static void AddScaled(float* destination, float* x, float scale, int n)
    {
        var s = Vector128.Create(scale);
        var i = 0;
        for (; i + 4 <= n; i += 4) AdvSimd.Store(destination + i, AdvSimd.FusedMultiplyAdd(AdvSimd.LoadVector128(destination + i), s, AdvSimd.LoadVector128(x + i)));
        for (; i < n; i++) destination[i] += scale * x[i];
    }

    public static void Glu(float* value, float* gate, float* destination, int n)
    {
        var i = 0;
        for (; i + 4 <= n; i += 4)
            AdvSimd.Store(destination + i, AdvSimd.Multiply(AdvSimd.LoadVector128(value + i), Sigmoid(AdvSimd.LoadVector128(gate + i))));
        for (; i < n; i++) destination[i] = value[i] * Ops.Sigmoid(gate[i]);
    }

    public static void Taps3(float* taps, int width, float* current, float* destination)
    {
        for (var c = 0; c < width; c += 4)
        {
            var acc = AdvSimd.Multiply(AdvSimd.LoadVector128(taps + c), AdvSimd.LoadVector128(current + c));
            acc = AdvSimd.FusedMultiplyAdd(acc, AdvSimd.LoadVector128(taps + width + c), AdvSimd.LoadVector128(current + c - width));
            acc = AdvSimd.FusedMultiplyAdd(acc, AdvSimd.LoadVector128(taps + 2 * width + c), AdvSimd.LoadVector128(current + c - 2 * width));
            AdvSimd.Store(destination + c, acc);
        }
    }

    // The decode step's causal conv for one head: Taps3's arithmetic (h2 is the newest row), so a token
    // decoded alone matches the same token inside a prefill chunk bit for bit.
    public static void TapsHead(float* taps, int stride, float* h0, float* h1, float* h2, float* destination, float* live0, float* live1, int n)
    {
        for (var c = 0; c < n; c += 4)
        {
            var x2 = AdvSimd.LoadVector128(h2 + c);
            var x1 = AdvSimd.LoadVector128(h1 + c);
            var acc = AdvSimd.Multiply(AdvSimd.LoadVector128(taps + c), x2);
            acc = AdvSimd.FusedMultiplyAdd(acc, AdvSimd.LoadVector128(taps + stride + c), x1);
            acc = AdvSimd.FusedMultiplyAdd(acc, AdvSimd.LoadVector128(taps + 2 * stride + c), AdvSimd.LoadVector128(h0 + c));
            AdvSimd.Store(destination + c, acc);
            AdvSimd.Store(live0 + c, x1);
            AdvSimd.Store(live1 + c, x2);
        }
    }

    // Per head: output *= sigmoid(gate), and the head's peak |output| at peaks[h * 16]. The products
    // are NeonPrepare.GateFakeA8's, so the max of the head peaks is its peak and the fused attention
    // gate prepares the same activations.
    public static void ApplyGateAndHeadPeaks(float* output, float* gate, int heads, int vDim, float* peaks)
    {
        for (var h = 0; h < heads; h++)
        {
            var x = output + h * vDim;
            var g = gate + h * vDim;
            var peak = Vector128<float>.Zero;
            var peak1 = Vector128<float>.Zero;
            var i = 0;
            for (; i + 8 <= vDim; i += 8)
            {
                var e0 = Exp(AdvSimd.Negate(AdvSimd.LoadVector128(g + i)));
                var e1 = Exp(AdvSimd.Negate(AdvSimd.LoadVector128(g + i + 4)));
                var v0 = AdvSimd.Multiply(AdvSimd.LoadVector128(x + i), AdvSimd.Arm64.Divide(Vector128.Create(1f), AdvSimd.Add(Vector128.Create(1f), e0)));
                var v1 = AdvSimd.Multiply(AdvSimd.LoadVector128(x + i + 4), AdvSimd.Arm64.Divide(Vector128.Create(1f), AdvSimd.Add(Vector128.Create(1f), e1)));
                AdvSimd.Store(x + i, v0);
                AdvSimd.Store(x + i + 4, v1);
                peak = AdvSimd.Max(peak, AdvSimd.Abs(v0));
                peak1 = AdvSimd.Max(peak1, AdvSimd.Abs(v1));
            }
            peak = AdvSimd.Max(peak, peak1);
            for (; i + 4 <= vDim; i += 4)
            {
                var v = AdvSimd.Multiply(AdvSimd.LoadVector128(x + i), Sigmoid(AdvSimd.LoadVector128(g + i)));
                AdvSimd.Store(x + i, v);
                peak = AdvSimd.Max(peak, AdvSimd.Abs(v));
            }
            var headPeak = AdvSimd.Arm64.MaxAcross(peak).ToScalar();
            for (; i < vDim; i++)
            {
                var v = x[i] * Ops.Sigmoid(g[i]);
                x[i] = v;
                headPeak = MathF.Max(headPeak, MathF.Abs(v));
            }
            peaks[h * 16] = headPeak;
        }
    }

    public static float AbsMax(float* x, int n)
    {
        var peak = Vector128<float>.Zero;
        var i = 0;
        for (; i + 4 <= n; i += 4) peak = AdvSimd.Max(peak, AdvSimd.Abs(AdvSimd.LoadVector128(x + i)));
        var best = AdvSimd.Arm64.MaxAcross(peak).ToScalar();
        for (; i < n; i++) best = MathF.Max(best, MathF.Abs(x[i]));
        return best;
    }

    public static void RoundTrip(float* x, int n, float inverse, float scale)
    {
        var inv = Vector128.Create(inverse);
        var sc = Vector128.Create(scale);
        var low = Vector128.Create(-128f);
        var high = Vector128.Create(127f);
        var i = 0;
        for (; i + 4 <= n; i += 4)
        {
            var q = AdvSimd.RoundToNearest(AdvSimd.Multiply(AdvSimd.LoadVector128(x + i), inv));
            AdvSimd.Store(x + i, AdvSimd.Multiply(AdvSimd.Min(AdvSimd.Max(q, low), high), sc));
        }
        for (; i < n; i++) x[i] = MathF.Min(MathF.Max(MathF.Round(x[i] * inverse), -128f), 127f) * scale;
    }

    public static float QuantizeInt8(float* x, int n, sbyte* destination)
    {
        var peak = AbsMax(x, n);
        var scale = peak > 0f ? peak / 127f : 1f;
        var scalarInv = 1f / scale;
        var inv = Vector128.Create(scalarInv);
        var i = 0;
        for (; i + 4 <= n; i += 4)
        {
            var wide = AdvSimd.ConvertToInt32RoundToZero(AdvSimd.Multiply(AdvSimd.LoadVector128(x + i), inv));
            var narrow = AdvSimd.ExtractNarrowingSaturateLower(wide).ToVector128();
            var bytes = AdvSimd.ExtractNarrowingSaturateLower(narrow);
            *(int*)(destination + i) = bytes.AsInt32().ToScalar();
        }
        for (; i < n; i++) destination[i] = (sbyte)MathF.Min(MathF.Max(MathF.Round(x[i] * scalarInv), -128f), 127f);
        return scale;
    }

    public static void Hadamard(float* source, float* p)
    {
        if (source != p)
            for (var i = 0; i < Cq.GroupSize; i += 4) AdvSimd.Store(p + i, AdvSimd.LoadVector128(source + i));
        for (var stride = 64; stride >= 8; stride >>= 1)
            for (var block = 0; block < Cq.GroupSize; block += stride * 2)
                for (var i = block; i < block + stride; i += 4)
                {
                    var a = AdvSimd.LoadVector128(p + i);
                    var b = AdvSimd.LoadVector128(p + i + stride);
                    AdvSimd.Store(p + i, AdvSimd.Add(a, b));
                    AdvSimd.Store(p + i + stride, AdvSimd.Subtract(a, b));
                }
        var scale = Vector128.Create(HadamardScale);
        for (var i = 0; i < Cq.GroupSize; i += 8)
        {
            var a = AdvSimd.LoadVector128(p + i);
            var b = AdvSimd.LoadVector128(p + i + 4);
            var s = AdvSimd.Add(a, b);
            var d = AdvSimd.Subtract(a, b);
            AdvSimd.Store(p + i, AdvSimd.Multiply(Butterfly8(s), scale));
            AdvSimd.Store(p + i + 4, AdvSimd.Multiply(Butterfly8(d), scale));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<float> Butterfly8(Vector128<float> t)
    {
        var rotated = Vector128.Shuffle(t, Vector128.Create(2, 3, 0, 1));
        var even = AdvSimd.Add(t, rotated).GetLower();
        var odd = AdvSimd.Subtract(t, rotated).GetLower();
        var sums = AdvSimd.AddPairwise(even, odd);
        var diffs = AdvSimd.Arm64.ZipLow(
            AdvSimd.Subtract(even, Vector64.Shuffle(even, Vector64.Create(1, 0))),
            AdvSimd.Subtract(odd, Vector64.Shuffle(odd, Vector64.Create(1, 0))));
        return AdvSimd.Arm64.ZipLow(sums.ToVector128(), diffs.ToVector128());
    }

    public static void Prepare(float* x, int tokens, int cols, int bits, in Activations a) =>
        NeonPrepare.Prepare(x, tokens, cols, bits, a);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<int> Dot16(Vector128<int> sum, Vector128<sbyte> weights, Vector128<sbyte> values)
    {
        if (UseDotProduct) return Dp.DotProduct(sum, weights, values);
        var weightLow = AdvSimd.SignExtendWideningLower(weights.GetLower());
        var weightHigh = AdvSimd.SignExtendWideningUpper(weights);
        var valueLow = AdvSimd.SignExtendWideningLower(values.GetLower());
        var valueHigh = AdvSimd.SignExtendWideningUpper(values);
        sum = AdvSimd.MultiplyWideningLowerAndAdd(sum, valueLow.GetLower(), weightLow.GetLower());
        sum = AdvSimd.MultiplyWideningUpperAndAdd(sum, valueLow, weightLow);
        sum = AdvSimd.MultiplyWideningLowerAndAdd(sum, valueHigh.GetLower(), weightHigh.GetLower());
        return AdvSimd.MultiplyWideningUpperAndAdd(sum, valueHigh, weightHigh);
    }

    public static void LoadFactors(ushort* norms, int groups, float scale, float* factors)
    {
        var multiplier = Vector128.Create(scale);
        var g = 0;
        for (; g + 4 <= groups; g += 4)
        {
            var wide = AdvSimd.ZeroExtendWideningLower(AdvSimd.LoadVector64(norms + g));
            var magnitude = AdvSimd.And(wide, Vector128.Create(0x7FFFu));
            var sign = AdvSimd.ShiftLeftLogical(AdvSimd.And(wide, Vector128.Create(0x8000u)), 16);
            var rebased = AdvSimd.Multiply(AdvSimd.ShiftLeftLogical(magnitude, 13).AsSingle(), Vector128.Create(5.192296858534828e33f));
            var single = AdvSimd.Or(rebased.AsUInt32(), sign).AsSingle();
            AdvSimd.Store(factors + g, AdvSimd.Multiply(single, multiplier));
        }
        if (g < groups && groups >= 4) AdvSimd.Store(factors + g, AdvSimd.Multiply(HalfTail(norms, groups), multiplier));
        else for (; g < groups; g++) factors[g] = Scalar.HalfToSingle(norms[g]) * scale;
    }

    // Scalar.HalfToSingle of a row's last groups % 4 halves (groups >= 4, groups % 4 != 0), zero in
    // the lanes after them: one load of the last four halves, shifted down past the ones already done.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<float> HalfTail(ushort* norms, int groups)
    {
        var missing = 4 - (groups & 3);
        var packed = AdvSimd.ShiftLogicalScalar(AdvSimd.LoadVector64((ulong*)(norms + groups - 4)), Vector64.Create(-16L * missing));
        return ExactHalves(AdvSimd.ZeroExtendWideningLower(packed.AsUInt16()));
    }

    // Scalar.HalfToSingle per lane, subnormals and the +0 for a zero magnitude included.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<float> ExactHalves(Vector128<uint> wide)
    {
        var magnitude = AdvSimd.And(wide, Vector128.Create(0x7FFFu));
        var sign = AdvSimd.ShiftLeftLogical(AdvSimd.And(wide, Vector128.Create(0x8000u)), 16);
        var normal = AdvSimd.Or(AdvSimd.Add(AdvSimd.ShiftLeftLogical(magnitude, 13), Vector128.Create(0x38000000u)), sign);
        var subnormal = AdvSimd.Or(AdvSimd.Multiply(AdvSimd.ConvertToSingle(magnitude), Vector128.Create(5.9604645e-8f)).AsUInt32(), sign);
        var value = AdvSimd.BitwiseSelect(AdvSimd.CompareLessThan(magnitude, Vector128.Create(0x400u)), subnormal, normal);
        return AdvSimd.And(value, AdvSimd.CompareGreaterThan(magnitude, Vector128<uint>.Zero)).AsSingle();
    }

    public static float ScaleFactors(float* factors, float* scaled, float* scales, float* offsets, int groups)
    {
        var bias = Vector128<float>.Zero;
        var v = 0;
        for (; v + 4 <= groups; v += 4)
        {
            var f = AdvSimd.Multiply(AdvSimd.LoadVector128(factors + v), AdvSimd.LoadVector128(scales + v));
            AdvSimd.Store(scaled + v, f);
            bias = AdvSimd.FusedMultiplyAdd(bias, f, AdvSimd.LoadVector128(offsets + v));
        }
        if (v < groups)
        {
            var left = groups - v;
            var f = AdvSimd.Multiply(AdvSimd.LoadVector128(factors + v), Partial(scales + v, left));
            AdvSimd.Store(scaled + v, f);
            bias = AdvSimd.FusedMultiplyAdd(bias, f, Partial(offsets + v, left));
        }
        return HorizontalSum(bias);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<float> Partial(float* source, int count) =>
        Vector128.Create(source[0], count > 1 ? source[1] : 0f, count > 2 ? source[2] : 0f, 0f);

    public static void DecodeRow(in Cq w, int row, float* destination)
    {
        var norms = w.Norms + (long)row * w.Groups;
        var indices = w.Indices + (long)row * w.RowBytes;
        var planes = stackalloc float[64];
        if (w.Bits == 2)
        {
            var levelBits = AdvSimd.LoadVector128(w.Book->Levels).AsByte();
            for (var g = 0; g < w.Groups; g++)
            {
                var norm = Vector64.Create(Scalar.HalfToSingle(norms[g]));
                var output = destination + g * Cq.GroupSize;
                EmitLevels2(AdvSimd.LoadVector128(indices + g * 32), levelBits, norm, planes);
                Interleave4(planes, output);
                EmitLevels2(AdvSimd.LoadVector128(indices + g * 32 + 16), levelBits, norm, planes);
                Interleave4(planes, output + 64);
            }
            return;
        }
        var levels = (
            AdvSimd.LoadVector128(w.Book->Levels).AsByte(),
            AdvSimd.LoadVector128(w.Book->Levels + 4).AsByte(),
            AdvSimd.LoadVector128(w.Book->Levels + 8).AsByte(),
            AdvSimd.LoadVector128(w.Book->Levels + 12).AsByte());
        var nibble = Vector128.Create((byte)0x0F);
        for (var g = 0; g < w.Groups; g++)
        {
            var norm = Vector64.Create(Scalar.HalfToSingle(norms[g]));
            var output = destination + g * Cq.GroupSize;
            for (var r = 0; r < 4; r++)
            {
                var source = AdvSimd.LoadVector128(indices + g * 64 + r * 16);
                EmitLevels4(AdvSimd.And(source, nibble), levels, norm, planes);
                EmitLevels4(AdvSimd.And(AdvSimd.ShiftRightLogical(source, 4), nibble), levels, norm, planes + 16);
                Interleave2(planes, output + r * 32);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void EmitLevels2(Vector128<byte> source, Vector128<byte> levelBits, Vector64<float> norm, float* planes)
    {
        EmitLevels2Plane(AdvSimd.And(source, Vector128.Create((byte)3)), levelBits, norm, planes);
        EmitLevels2Plane(AdvSimd.And(AdvSimd.ShiftRightLogical(source, 2), Vector128.Create((byte)3)), levelBits, norm, planes + 16);
        EmitLevels2Plane(AdvSimd.And(AdvSimd.ShiftRightLogical(source, 4), Vector128.Create((byte)3)), levelBits, norm, planes + 32);
        EmitLevels2Plane(AdvSimd.And(AdvSimd.ShiftRightLogical(source, 6), Vector128.Create((byte)3)), levelBits, norm, planes + 48);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void EmitLevels2Plane(Vector128<byte> index, Vector128<byte> levelBits, Vector64<float> norm, float* planes)
    {
        for (var c = 0; c < 8; c++)
        {
            var first = index.GetElement(2 * c) * 4;
            var second = index.GetElement(2 * c + 1) * 4;
            var gather = Vector64.Create((byte)first, (byte)(first + 1), (byte)(first + 2), (byte)(first + 3), (byte)second, (byte)(second + 1), (byte)(second + 2), (byte)(second + 3));
            AdvSimd.Store(planes + c * 2, AdvSimd.Multiply(AdvSimd.VectorTableLookup(levelBits, gather).AsSingle(), norm));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void EmitLevels4(Vector128<byte> index, (Vector128<byte>, Vector128<byte>, Vector128<byte>, Vector128<byte>) levels, Vector64<float> norm, float* planes)
    {
        for (var c = 0; c < 8; c++)
        {
            var first = index.GetElement(2 * c) * 4;
            var second = index.GetElement(2 * c + 1) * 4;
            var gather = Vector64.Create((byte)first, (byte)(first + 1), (byte)(first + 2), (byte)(first + 3), (byte)second, (byte)(second + 1), (byte)(second + 2), (byte)(second + 3));
            AdvSimd.Store(planes + c * 2, AdvSimd.Multiply(AdvSimd.VectorTableLookup(levels, gather).AsSingle(), norm));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void Interleave4(float* planes, float* output)
    {
        for (var q = 0; q < 4; q++)
        {
            var p0 = AdvSimd.LoadVector128(planes + q * 4);
            var p1 = AdvSimd.LoadVector128(planes + 16 + q * 4);
            var p2 = AdvSimd.LoadVector128(planes + 32 + q * 4);
            var p3 = AdvSimd.LoadVector128(planes + 48 + q * 4);
            var evenLow = AdvSimd.Arm64.ZipLow(p0, p2);
            var evenHigh = AdvSimd.Arm64.ZipHigh(p0, p2);
            var oddLow = AdvSimd.Arm64.ZipLow(p1, p3);
            var oddHigh = AdvSimd.Arm64.ZipHigh(p1, p3);
            AdvSimd.Store(output + q * 16, AdvSimd.Arm64.ZipLow(evenLow, oddLow));
            AdvSimd.Store(output + q * 16 + 4, AdvSimd.Arm64.ZipHigh(evenLow, oddLow));
            AdvSimd.Store(output + q * 16 + 8, AdvSimd.Arm64.ZipLow(evenHigh, oddHigh));
            AdvSimd.Store(output + q * 16 + 12, AdvSimd.Arm64.ZipHigh(evenHigh, oddHigh));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void Interleave2(float* planes, float* output)
    {
        for (var q = 0; q < 2; q++)
        {
            var a = AdvSimd.LoadVector128(planes + q * 8);
            var b = AdvSimd.LoadVector128(planes + q * 8 + 4);
            var c = AdvSimd.LoadVector128(planes + 16 + q * 8);
            var d = AdvSimd.LoadVector128(planes + 16 + q * 8 + 4);
            AdvSimd.Store(output + q * 16, AdvSimd.Arm64.ZipLow(a, c));
            AdvSimd.Store(output + q * 16 + 4, AdvSimd.Arm64.ZipHigh(a, c));
            AdvSimd.Store(output + q * 16 + 8, AdvSimd.Arm64.ZipLow(b, d));
            AdvSimd.Store(output + q * 16 + 12, AdvSimd.Arm64.ZipHigh(b, d));
        }
    }

    public static void MatMulRows(in Cq w, in Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd) =>
        NeonGemv.MatMulRows(w, a, tokens, y, ldy, rowStart, rowEnd);


    public static void ConditionWeights(float* condV, float* input, int dim, float* weights)
    {
        var even0 = Vector128<float>.Zero;
        var even1 = Vector128<float>.Zero;
        var odd0 = Vector128<float>.Zero;
        var odd1 = Vector128<float>.Zero;
        for (var i = 0; i < dim; i += 2)
        {
            var first = Vector128.Create(input[i]);
            var second = Vector128.Create(input[i + 1]);
            even0 = AdvSimd.FusedMultiplyAdd(even0, first, AdvSimd.LoadVector128(condV + i * 8));
            even1 = AdvSimd.FusedMultiplyAdd(even1, first, AdvSimd.LoadVector128(condV + i * 8 + 4));
            odd0 = AdvSimd.FusedMultiplyAdd(odd0, second, AdvSimd.LoadVector128(condV + i * 8 + 8));
            odd1 = AdvSimd.FusedMultiplyAdd(odd1, second, AdvSimd.LoadVector128(condV + i * 8 + 12));
        }
        AdvSimd.Store(weights, AdvSimd.Add(even0, odd0));
        AdvSimd.Store(weights + 4, AdvSimd.Add(even1, odd1));
        Softmax(weights, 8);
    }

    public static void ScaleInput(float* input, float* d1, int dim, int padded, float* destination)
    {
        for (var i = 0; i < dim; i += 4) AdvSimd.Store(destination + i, AdvSimd.Multiply(AdvSimd.LoadVector128(input + i), AdvSimd.LoadVector128(d1 + i)));
        for (var i = dim; i < padded; i += 4) AdvSimd.Store(destination + i, Vector128<float>.Zero);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<float> Gather(float* source, int* permutation) =>
        Vector128.Create(source[permutation[0]], source[permutation[1]], source[permutation[2]], source[permutation[3]]);

    // Four independent vectors per step, so the long FMA and exp chains of neighbouring columns
    // overlap; each column's arithmetic is PermuteModulateActivateSingle's.
    public static void PermuteModulateActivate(int* permutation, float* source, float* d2, float* weights, float* condU, float* b2, int width, float* destination)
    {
        var w0 = Vector128.Create(weights[0]);
        var w1 = Vector128.Create(weights[1]);
        var w2 = Vector128.Create(weights[2]);
        var w3 = Vector128.Create(weights[3]);
        var w4 = Vector128.Create(weights[4]);
        var w5 = Vector128.Create(weights[5]);
        var w6 = Vector128.Create(weights[6]);
        var w7 = Vector128.Create(weights[7]);
        var one = Vector128.Create(1f);
        var i = 0;
        for (; i + 16 <= width; i += 16)
        {
            var c0 = AdvSimd.FusedMultiplyAdd(one, w0, AdvSimd.LoadVector128(condU + i));
            var c1 = AdvSimd.FusedMultiplyAdd(one, w0, AdvSimd.LoadVector128(condU + i + 4));
            var c2 = AdvSimd.FusedMultiplyAdd(one, w0, AdvSimd.LoadVector128(condU + i + 8));
            var c3 = AdvSimd.FusedMultiplyAdd(one, w0, AdvSimd.LoadVector128(condU + i + 12));
            c0 = AdvSimd.FusedMultiplyAdd(c0, w1, AdvSimd.LoadVector128(condU + 1 * width + i));
            c1 = AdvSimd.FusedMultiplyAdd(c1, w1, AdvSimd.LoadVector128(condU + 1 * width + i + 4));
            c2 = AdvSimd.FusedMultiplyAdd(c2, w1, AdvSimd.LoadVector128(condU + 1 * width + i + 8));
            c3 = AdvSimd.FusedMultiplyAdd(c3, w1, AdvSimd.LoadVector128(condU + 1 * width + i + 12));
            c0 = AdvSimd.FusedMultiplyAdd(c0, w2, AdvSimd.LoadVector128(condU + 2 * width + i));
            c1 = AdvSimd.FusedMultiplyAdd(c1, w2, AdvSimd.LoadVector128(condU + 2 * width + i + 4));
            c2 = AdvSimd.FusedMultiplyAdd(c2, w2, AdvSimd.LoadVector128(condU + 2 * width + i + 8));
            c3 = AdvSimd.FusedMultiplyAdd(c3, w2, AdvSimd.LoadVector128(condU + 2 * width + i + 12));
            c0 = AdvSimd.FusedMultiplyAdd(c0, w3, AdvSimd.LoadVector128(condU + 3 * width + i));
            c1 = AdvSimd.FusedMultiplyAdd(c1, w3, AdvSimd.LoadVector128(condU + 3 * width + i + 4));
            c2 = AdvSimd.FusedMultiplyAdd(c2, w3, AdvSimd.LoadVector128(condU + 3 * width + i + 8));
            c3 = AdvSimd.FusedMultiplyAdd(c3, w3, AdvSimd.LoadVector128(condU + 3 * width + i + 12));
            c0 = AdvSimd.FusedMultiplyAdd(c0, w4, AdvSimd.LoadVector128(condU + 4 * width + i));
            c1 = AdvSimd.FusedMultiplyAdd(c1, w4, AdvSimd.LoadVector128(condU + 4 * width + i + 4));
            c2 = AdvSimd.FusedMultiplyAdd(c2, w4, AdvSimd.LoadVector128(condU + 4 * width + i + 8));
            c3 = AdvSimd.FusedMultiplyAdd(c3, w4, AdvSimd.LoadVector128(condU + 4 * width + i + 12));
            c0 = AdvSimd.FusedMultiplyAdd(c0, w5, AdvSimd.LoadVector128(condU + 5 * width + i));
            c1 = AdvSimd.FusedMultiplyAdd(c1, w5, AdvSimd.LoadVector128(condU + 5 * width + i + 4));
            c2 = AdvSimd.FusedMultiplyAdd(c2, w5, AdvSimd.LoadVector128(condU + 5 * width + i + 8));
            c3 = AdvSimd.FusedMultiplyAdd(c3, w5, AdvSimd.LoadVector128(condU + 5 * width + i + 12));
            c0 = AdvSimd.FusedMultiplyAdd(c0, w6, AdvSimd.LoadVector128(condU + 6 * width + i));
            c1 = AdvSimd.FusedMultiplyAdd(c1, w6, AdvSimd.LoadVector128(condU + 6 * width + i + 4));
            c2 = AdvSimd.FusedMultiplyAdd(c2, w6, AdvSimd.LoadVector128(condU + 6 * width + i + 8));
            c3 = AdvSimd.FusedMultiplyAdd(c3, w6, AdvSimd.LoadVector128(condU + 6 * width + i + 12));
            c0 = AdvSimd.FusedMultiplyAdd(c0, w7, AdvSimd.LoadVector128(condU + 7 * width + i));
            c1 = AdvSimd.FusedMultiplyAdd(c1, w7, AdvSimd.LoadVector128(condU + 7 * width + i + 4));
            c2 = AdvSimd.FusedMultiplyAdd(c2, w7, AdvSimd.LoadVector128(condU + 7 * width + i + 8));
            c3 = AdvSimd.FusedMultiplyAdd(c3, w7, AdvSimd.LoadVector128(condU + 7 * width + i + 12));
            var v0 = AdvSimd.FusedMultiplyAdd(AdvSimd.LoadVector128(b2 + i), AdvSimd.Multiply(AdvSimd.LoadVector128(d2 + i), c0), Gather(source, permutation + i));
            var v1 = AdvSimd.FusedMultiplyAdd(AdvSimd.LoadVector128(b2 + i + 4), AdvSimd.Multiply(AdvSimd.LoadVector128(d2 + i + 4), c1), Gather(source, permutation + i + 4));
            var v2 = AdvSimd.FusedMultiplyAdd(AdvSimd.LoadVector128(b2 + i + 8), AdvSimd.Multiply(AdvSimd.LoadVector128(d2 + i + 8), c2), Gather(source, permutation + i + 8));
            var v3 = AdvSimd.FusedMultiplyAdd(AdvSimd.LoadVector128(b2 + i + 12), AdvSimd.Multiply(AdvSimd.LoadVector128(d2 + i + 12), c3), Gather(source, permutation + i + 12));
            var e0 = Exp(AdvSimd.Negate(v0));
            var e1 = Exp(AdvSimd.Negate(v1));
            var e2 = Exp(AdvSimd.Negate(v2));
            var e3 = Exp(AdvSimd.Negate(v3));
            AdvSimd.Store(destination + i, AdvSimd.Multiply(v0, AdvSimd.Arm64.Divide(one, AdvSimd.Add(one, e0))));
            AdvSimd.Store(destination + i + 4, AdvSimd.Multiply(v1, AdvSimd.Arm64.Divide(one, AdvSimd.Add(one, e1))));
            AdvSimd.Store(destination + i + 8, AdvSimd.Multiply(v2, AdvSimd.Arm64.Divide(one, AdvSimd.Add(one, e2))));
            AdvSimd.Store(destination + i + 12, AdvSimd.Multiply(v3, AdvSimd.Arm64.Divide(one, AdvSimd.Add(one, e3))));
        }
        for (; i < width; i += 4)
        {
            var condition = AdvSimd.FusedMultiplyAdd(one, w0, AdvSimd.LoadVector128(condU + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w1, AdvSimd.LoadVector128(condU + width + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w2, AdvSimd.LoadVector128(condU + 2 * width + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w3, AdvSimd.LoadVector128(condU + 3 * width + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w4, AdvSimd.LoadVector128(condU + 4 * width + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w5, AdvSimd.LoadVector128(condU + 5 * width + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w6, AdvSimd.LoadVector128(condU + 6 * width + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w7, AdvSimd.LoadVector128(condU + 7 * width + i));
            var v = AdvSimd.FusedMultiplyAdd(
                AdvSimd.LoadVector128(b2 + i),
                AdvSimd.Multiply(AdvSimd.LoadVector128(d2 + i), condition),
                Gather(source, permutation + i));
            AdvSimd.Store(destination + i, AdvSimd.Multiply(v, Sigmoid(v)));
        }
    }

    internal static void PermuteModulateActivateSingle(int* permutation, float* source, float* d2, float* weights, float* condU, float* b2, int width, float* destination)
    {
        var w0 = Vector128.Create(weights[0]);
        var w1 = Vector128.Create(weights[1]);
        var w2 = Vector128.Create(weights[2]);
        var w3 = Vector128.Create(weights[3]);
        var w4 = Vector128.Create(weights[4]);
        var w5 = Vector128.Create(weights[5]);
        var w6 = Vector128.Create(weights[6]);
        var w7 = Vector128.Create(weights[7]);
        var one = Vector128.Create(1f);
        for (var i = 0; i < width; i += 4)
        {
            var condition = AdvSimd.FusedMultiplyAdd(one, w0, AdvSimd.LoadVector128(condU + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w1, AdvSimd.LoadVector128(condU + width + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w2, AdvSimd.LoadVector128(condU + 2 * width + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w3, AdvSimd.LoadVector128(condU + 3 * width + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w4, AdvSimd.LoadVector128(condU + 4 * width + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w5, AdvSimd.LoadVector128(condU + 5 * width + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w6, AdvSimd.LoadVector128(condU + 6 * width + i));
            condition = AdvSimd.FusedMultiplyAdd(condition, w7, AdvSimd.LoadVector128(condU + 7 * width + i));
            var v = AdvSimd.FusedMultiplyAdd(
                AdvSimd.LoadVector128(b2 + i),
                AdvSimd.Multiply(AdvSimd.LoadVector128(d2 + i), condition),
                Gather(source, permutation + i));
            AdvSimd.Store(destination + i, AdvSimd.Multiply(v, Sigmoid(v)));
        }
    }

    public static void PermuteScale(int* permutation, float* source, float* scale, int width, float* destination)
    {
        for (var i = 0; i < width; i += 4)
            AdvSimd.Store(destination + i, AdvSimd.Multiply(Gather(source, permutation + i), AdvSimd.LoadVector128(scale + i)));
    }

    public static void AccumulateOutput(float* residual, float* product, float* d4, float scale, int count)
    {
        var s = Vector128.Create(scale);
        for (var i = 0; i < count; i += 4)
            AdvSimd.Store(residual + i, AdvSimd.FusedMultiplyAdd(
                AdvSimd.LoadVector128(residual + i), s, AdvSimd.Multiply(AdvSimd.LoadVector128(product + i), AdvSimd.LoadVector128(d4 + i))));
    }
    public static void Sinkhorn4x4(float* logits, float* result)
    {
        var centered = stackalloc float[4];
        var sums = stackalloc float[4];
        for (var r = 0; r < 4; r++)
        {
            var row = AdvSimd.LoadVector128(logits + r * 4);
            var shift = Vector128.Create(AdvSimd.Arm64.MaxAcross(row).ToScalar());
            AdvSimd.Store(centered, AdvSimd.Subtract(row, shift));
            for (var c = 0; c < 4; c++) result[r * 4 + c] = MathF.Exp(centered[c]);
        }
        for (var iteration = 0; iteration < 20; iteration++)
        {
            NormRow(result);
            NormRow(result + 4);
            NormRow(result + 8);
            NormRow(result + 12);
            var a0 = AdvSimd.LoadVector128(result);
            var a1 = AdvSimd.LoadVector128(result + 4);
            var a2 = AdvSimd.LoadVector128(result + 8);
            var a3 = AdvSimd.LoadVector128(result + 12);
            AdvSimd.Store(sums, AdvSimd.Add(AdvSimd.Add(AdvSimd.Add(a0, a1), a2), a3));
            if (sums[0] > 0f && sums[1] > 0f && sums[2] > 0f && sums[3] > 0f)
            {
                var total = AdvSimd.LoadVector128(sums);
                AdvSimd.Store(result, AdvSimd.Arm64.Divide(a0, total));
                AdvSimd.Store(result + 4, AdvSimd.Arm64.Divide(a1, total));
                AdvSimd.Store(result + 8, AdvSimd.Arm64.Divide(a2, total));
                AdvSimd.Store(result + 12, AdvSimd.Arm64.Divide(a3, total));
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
    }

    static void NormRow(float* row)
    {
        var v = AdvSimd.LoadVector128(row);
        var lanes = stackalloc float[4];
        AdvSimd.Store(lanes, v);
        AdvSimd.Store(row, AdvSimd.Arm64.Divide(v, Vector128.Create(lanes[0] + lanes[1] + lanes[2] + lanes[3])));
    }

    public static void RmsFakeA8(float* x, int n, float* destination)
    {
        var scalarInv = 1f / MathF.Sqrt(SumOfSquares(x, n) / n + RmsEpsilon);
        var inverse = Vector128.Create(scalarInv);
        var peak = Vector128<float>.Zero;
        var i = 0;
        for (; i + 4 <= n; i += 4)
        {
            var v = AdvSimd.Multiply(AdvSimd.LoadVector128(x + i), inverse);
            AdvSimd.Store(destination + i, v);
            peak = AdvSimd.Max(peak, AdvSimd.Abs(v));
        }
        var folded = AdvSimd.Arm64.MaxAcross(peak).ToScalar();
        for (; i < n; i++)
        {
            var v = x[i] * scalarInv;
            destination[i] = v;
            folded = MathF.Max(folded, MathF.Abs(v));
        }
        FinishFakeA8(destination, n, folded);
    }

    public static void CenteredRmsFakeA8(float* x, float* scale, int n, float* destination)
    {
        var scalarInv = 1f / MathF.Sqrt(SumOfSquares(x, n) / n + RmsEpsilon);
        var inverse = Vector128.Create(scalarInv);
        var one = Vector128.Create(1f);
        var peak = Vector128<float>.Zero;
        var i = 0;
        for (; i + 4 <= n; i += 4)
        {
            var v = AdvSimd.Multiply(AdvSimd.Multiply(AdvSimd.Add(one, AdvSimd.LoadVector128(scale + i)), AdvSimd.LoadVector128(x + i)), inverse);
            AdvSimd.Store(destination + i, v);
            peak = AdvSimd.Max(peak, AdvSimd.Abs(v));
        }
        var folded = AdvSimd.Arm64.MaxAcross(peak).ToScalar();
        for (; i < n; i++)
        {
            var v = (1f + scale[i]) * x[i] * scalarInv;
            destination[i] = v;
            folded = MathF.Max(folded, MathF.Abs(v));
        }
        FinishFakeA8(destination, n, folded);
    }

    public static void GateFakeA8(float* x, float* gate, int n)
    {
        var peak = Vector128<float>.Zero;
        var i = 0;
        for (; i + 4 <= n; i += 4)
        {
            var v = AdvSimd.Multiply(AdvSimd.LoadVector128(x + i), Sigmoid(AdvSimd.LoadVector128(gate + i)));
            AdvSimd.Store(x + i, v);
            peak = AdvSimd.Max(peak, AdvSimd.Abs(v));
        }
        var folded = AdvSimd.Arm64.MaxAcross(peak).ToScalar();
        for (; i < n; i++)
        {
            var v = x[i] * Ops.Sigmoid(gate[i]);
            x[i] = v;
            folded = MathF.Max(folded, MathF.Abs(v));
        }
        FinishFakeA8(x, n, folded);
    }

    static void FinishFakeA8(float* x, int n, float peak)
    {
        var scale = peak > 0f ? peak / 127f : 1f;
        RoundTrip(x, n, 1f / scale, scale);
    }

    public static void CombineDiff4(float* s, int dim, float w0, float w1, float w2, float w3, float* x, float* u, float post, float* destination)
    {
        var v0 = Vector128.Create(w0);
        var v1 = Vector128.Create(w1);
        var v2 = Vector128.Create(w2);
        var v3 = Vector128.Create(w3);
        var vp = Vector128.Create(post);
        for (var c = 0; c < dim; c += 4)
        {
            var acc = AdvSimd.Multiply(vp, AdvSimd.Subtract(AdvSimd.LoadVector128(x + c), AdvSimd.LoadVector128(u + c)));
            acc = AdvSimd.FusedMultiplyAdd(acc, v0, AdvSimd.LoadVector128(s + c));
            acc = AdvSimd.FusedMultiplyAdd(acc, v1, AdvSimd.LoadVector128(s + dim + c));
            acc = AdvSimd.FusedMultiplyAdd(acc, v2, AdvSimd.LoadVector128(s + 2 * dim + c));
            acc = AdvSimd.FusedMultiplyAdd(acc, v3, AdvSimd.LoadVector128(s + 3 * dim + c));
            AdvSimd.Store(destination + c, acc);
        }
    }

    // CombineDiff4 for the four output lanes in one pass over the stream: the same products in the
    // same order, with each lane's weights read by element from mix.
    public static void CombineDiff4x4(float* s, int dim, float* mix, float* x, float* u, float* postGate, float* destination)
    {
        var m0 = AdvSimd.LoadVector128(mix);
        var m1 = AdvSimd.LoadVector128(mix + 4);
        var m2 = AdvSimd.LoadVector128(mix + 8);
        var m3 = AdvSimd.LoadVector128(mix + 12);
        var p = AdvSimd.LoadVector128(postGate);
        var s1 = s + dim;
        var s2 = s + 2 * dim;
        var s3 = s + 3 * dim;
        var d1 = destination + dim;
        var d2 = destination + 2 * dim;
        var d3 = destination + 3 * dim;
        for (var c = 0; c < dim; c += 4)
        {
            var diff = AdvSimd.Subtract(AdvSimd.LoadVector128(x + c), AdvSimd.LoadVector128(u + c));
            var v0 = AdvSimd.LoadVector128(s + c);
            var v1 = AdvSimd.LoadVector128(s1 + c);
            var v2 = AdvSimd.LoadVector128(s2 + c);
            var v3 = AdvSimd.LoadVector128(s3 + c);
            var a0 = AdvSimd.MultiplyBySelectedScalar(diff, p, 0);
            var a1 = AdvSimd.MultiplyBySelectedScalar(diff, p, 1);
            var a2 = AdvSimd.MultiplyBySelectedScalar(diff, p, 2);
            var a3 = AdvSimd.MultiplyBySelectedScalar(diff, p, 3);
            a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0, v0, m0, 0);
            a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1, v0, m1, 0);
            a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2, v0, m2, 0);
            a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3, v0, m3, 0);
            a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0, v1, m0, 1);
            a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1, v1, m1, 1);
            a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2, v1, m2, 1);
            a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3, v1, m3, 1);
            a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0, v2, m0, 2);
            a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1, v2, m1, 2);
            a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2, v2, m2, 2);
            a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3, v2, m3, 2);
            a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0, v3, m0, 3);
            a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1, v3, m1, 3);
            a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2, v3, m2, 3);
            a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3, v3, m3, 3);
            AdvSimd.Store(destination + c, a0);
            AdvSimd.Store(d1 + c, a1);
            AdvSimd.Store(d2 + c, a2);
            AdvSimd.Store(d3 + c, a3);
        }
    }

    public static void PostNormAdd(float* residual, float* x, float* scale, float gate, int n)
    {
        var scalarInv = 1f / MathF.Sqrt(SumOfSquares(x, n) / n + RmsEpsilon);
        var inverse = Vector128.Create(scalarInv);
        var one = Vector128.Create(1f);
        var g = Vector128.Create(gate);
        var i = 0;
        for (; i + 4 <= n; i += 4)
            AdvSimd.Store(residual + i, AdvSimd.FusedMultiplyAdd(AdvSimd.LoadVector128(residual + i), g,
                AdvSimd.Multiply(AdvSimd.Multiply(AdvSimd.Add(one, AdvSimd.LoadVector128(scale + i)), AdvSimd.LoadVector128(x + i)), inverse)));
        for (; i < n; i++) residual[i] += gate * (1f + scale[i]) * x[i] * scalarInv;
    }

    [SkipLocalsInit]
    public static void Kron(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
    {
        var tile = stackalloc float[4 * cols];
        nint stride1 = cols;
        nint stride2 = 2 * cols;
        nint stride3 = 3 * cols;
        for (var k = 0; k < resultRows; k += 4)
        {
            for (var c = 0; c < cols; c += 16)
            {
                var a00 = Vector128<float>.Zero;
                var a01 = Vector128<float>.Zero;
                var a02 = Vector128<float>.Zero;
                var a03 = Vector128<float>.Zero;
                var a10 = Vector128<float>.Zero;
                var a11 = Vector128<float>.Zero;
                var a12 = Vector128<float>.Zero;
                var a13 = Vector128<float>.Zero;
                var a20 = Vector128<float>.Zero;
                var a21 = Vector128<float>.Zero;
                var a22 = Vector128<float>.Zero;
                var a23 = Vector128<float>.Zero;
                var a30 = Vector128<float>.Zero;
                var a31 = Vector128<float>.Zero;
                var a32 = Vector128<float>.Zero;
                var a33 = Vector128<float>.Zero;
                var column = z + c;
                var factor = left + k;
                for (var i = 0; i < activeRows; i++)
                {
                    var z0 = AdvSimd.LoadVector128(column);
                    var z1 = AdvSimd.LoadVector128(column + 4);
                    var z2 = AdvSimd.LoadVector128(column + 8);
                    var z3 = AdvSimd.LoadVector128(column + 12);
                    var l = AdvSimd.LoadVector128(factor);
                    a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, z0, l, 0);
                    a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, z1, l, 0);
                    a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, z2, l, 0);
                    a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, z3, l, 0);
                    a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, z0, l, 1);
                    a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, z1, l, 1);
                    a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, z2, l, 1);
                    a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, z3, l, 1);
                    a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, z0, l, 2);
                    a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, z1, l, 2);
                    a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, z2, l, 2);
                    a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, z3, l, 2);
                    a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, z0, l, 3);
                    a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, z1, l, 3);
                    a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, z2, l, 3);
                    a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, z3, l, 3);
                    column += cols;
                    factor += rows;
                }
                AdvSimd.Store(tile + c, a00);
                AdvSimd.Store(tile + c + 4, a01);
                AdvSimd.Store(tile + c + 8, a02);
                AdvSimd.Store(tile + c + 12, a03);
                AdvSimd.Store(tile + c + stride1, a10);
                AdvSimd.Store(tile + c + stride1 + 4, a11);
                AdvSimd.Store(tile + c + stride1 + 8, a12);
                AdvSimd.Store(tile + c + stride1 + 12, a13);
                AdvSimd.Store(tile + c + stride2, a20);
                AdvSimd.Store(tile + c + stride2 + 4, a21);
                AdvSimd.Store(tile + c + stride2 + 8, a22);
                AdvSimd.Store(tile + c + stride2 + 12, a23);
                AdvSimd.Store(tile + c + stride3, a30);
                AdvSimd.Store(tile + c + stride3 + 4, a31);
                AdvSimd.Store(tile + c + stride3 + 8, a32);
                AdvSimd.Store(tile + c + stride3 + 12, a33);
            }
            for (var c = 0; c < cols; c += 16)
            {
                var a00 = Vector128<float>.Zero;
                var a01 = Vector128<float>.Zero;
                var a02 = Vector128<float>.Zero;
                var a03 = Vector128<float>.Zero;
                var a10 = Vector128<float>.Zero;
                var a11 = Vector128<float>.Zero;
                var a12 = Vector128<float>.Zero;
                var a13 = Vector128<float>.Zero;
                var a20 = Vector128<float>.Zero;
                var a21 = Vector128<float>.Zero;
                var a22 = Vector128<float>.Zero;
                var a23 = Vector128<float>.Zero;
                var a30 = Vector128<float>.Zero;
                var a31 = Vector128<float>.Zero;
                var a32 = Vector128<float>.Zero;
                var a33 = Vector128<float>.Zero;
                var row = right + c;
                var scalars = tile;
                for (var j = 0; j < cols; j += 4)
                {
                    var t0 = AdvSimd.LoadVector128(scalars);
                    var t1 = AdvSimd.LoadVector128(scalars + stride1);
                    var t2 = AdvSimd.LoadVector128(scalars + stride2);
                    var t3 = AdvSimd.LoadVector128(scalars + stride3);
                    {
                        var r0 = AdvSimd.LoadVector128(row);
                        var r1 = AdvSimd.LoadVector128(row + 4);
                        var r2 = AdvSimd.LoadVector128(row + 8);
                        var r3 = AdvSimd.LoadVector128(row + 12);
                        a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, r0, t0, 0);
                        a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, r1, t0, 0);
                        a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, r2, t0, 0);
                        a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, r3, t0, 0);
                        a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, r0, t1, 0);
                        a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, r1, t1, 0);
                        a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, r2, t1, 0);
                        a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, r3, t1, 0);
                        a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, r0, t2, 0);
                        a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, r1, t2, 0);
                        a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, r2, t2, 0);
                        a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, r3, t2, 0);
                        a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, r0, t3, 0);
                        a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, r1, t3, 0);
                        a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, r2, t3, 0);
                        a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, r3, t3, 0);
                        row += cols;
                    }
                    {
                        var r0 = AdvSimd.LoadVector128(row);
                        var r1 = AdvSimd.LoadVector128(row + 4);
                        var r2 = AdvSimd.LoadVector128(row + 8);
                        var r3 = AdvSimd.LoadVector128(row + 12);
                        a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, r0, t0, 1);
                        a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, r1, t0, 1);
                        a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, r2, t0, 1);
                        a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, r3, t0, 1);
                        a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, r0, t1, 1);
                        a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, r1, t1, 1);
                        a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, r2, t1, 1);
                        a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, r3, t1, 1);
                        a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, r0, t2, 1);
                        a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, r1, t2, 1);
                        a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, r2, t2, 1);
                        a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, r3, t2, 1);
                        a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, r0, t3, 1);
                        a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, r1, t3, 1);
                        a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, r2, t3, 1);
                        a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, r3, t3, 1);
                        row += cols;
                    }
                    {
                        var r0 = AdvSimd.LoadVector128(row);
                        var r1 = AdvSimd.LoadVector128(row + 4);
                        var r2 = AdvSimd.LoadVector128(row + 8);
                        var r3 = AdvSimd.LoadVector128(row + 12);
                        a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, r0, t0, 2);
                        a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, r1, t0, 2);
                        a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, r2, t0, 2);
                        a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, r3, t0, 2);
                        a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, r0, t1, 2);
                        a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, r1, t1, 2);
                        a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, r2, t1, 2);
                        a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, r3, t1, 2);
                        a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, r0, t2, 2);
                        a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, r1, t2, 2);
                        a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, r2, t2, 2);
                        a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, r3, t2, 2);
                        a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, r0, t3, 2);
                        a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, r1, t3, 2);
                        a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, r2, t3, 2);
                        a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, r3, t3, 2);
                        row += cols;
                    }
                    {
                        var r0 = AdvSimd.LoadVector128(row);
                        var r1 = AdvSimd.LoadVector128(row + 4);
                        var r2 = AdvSimd.LoadVector128(row + 8);
                        var r3 = AdvSimd.LoadVector128(row + 12);
                        a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, r0, t0, 3);
                        a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, r1, t0, 3);
                        a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, r2, t0, 3);
                        a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, r3, t0, 3);
                        a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, r0, t1, 3);
                        a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, r1, t1, 3);
                        a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, r2, t1, 3);
                        a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, r3, t1, 3);
                        a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, r0, t2, 3);
                        a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, r1, t2, 3);
                        a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, r2, t2, 3);
                        a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, r3, t2, 3);
                        a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, r0, t3, 3);
                        a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, r1, t3, 3);
                        a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, r2, t3, 3);
                        a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, r3, t3, 3);
                        row += cols;
                    }
                    scalars += 4;
                }
                var output = result + k * cols + c;
                AdvSimd.Store(output, a00);
                AdvSimd.Store(output + 4, a01);
                AdvSimd.Store(output + 8, a02);
                AdvSimd.Store(output + 12, a03);
                AdvSimd.Store(output + stride1, a10);
                AdvSimd.Store(output + stride1 + 4, a11);
                AdvSimd.Store(output + stride1 + 8, a12);
                AdvSimd.Store(output + stride1 + 12, a13);
                AdvSimd.Store(output + stride2, a20);
                AdvSimd.Store(output + stride2 + 4, a21);
                AdvSimd.Store(output + stride2 + 8, a22);
                AdvSimd.Store(output + stride2 + 12, a23);
                AdvSimd.Store(output + stride3, a30);
                AdvSimd.Store(output + stride3 + 4, a31);
                AdvSimd.Store(output + stride3 + 8, a32);
                AdvSimd.Store(output + stride3 + 12, a33);
            }
        }
    }

    public static int DotInt8x48(sbyte* a, sbyte* b)
    {
        var sum = Dot16(Vector128<int>.Zero, AdvSimd.LoadVector128(a), AdvSimd.LoadVector128(b));
        sum = Dot16(sum, AdvSimd.LoadVector128(a + 16), AdvSimd.LoadVector128(b + 16));
        sum = Dot16(sum, AdvSimd.LoadVector128(a + 32), AdvSimd.LoadVector128(b + 32));
        return AdvSimd.Arm64.AddAcross(sum).ToScalar();
    }

    public static void AttentionHead(
        sbyte* query, float queryScale, sbyte* keys, float* keyScales, sbyte* values, float* valueScales,
        int keyStride, int valueStride, int scaleStride, int first, int last, float* scores, float* output, float* probabilities)
    {
        var count = last - first + 1;
        var scale = queryScale * InverseSqrtQkDim;
        for (var s = 0; s < count; s++)
            scores[s] = DotInt8x48(query, keys + (long)(first + s) * keyStride) * keyScales[(long)(first + s) * scaleStride] * scale;
        Softmax(scores, count);
        if (probabilities != null) for (var s = 0; s < count; s++) probabilities[first + s] = scores[s];
        var a0 = Vector128<float>.Zero;
        var a1 = Vector128<float>.Zero;
        var a2 = Vector128<float>.Zero;
        var a3 = Vector128<float>.Zero;
        var a4 = Vector128<float>.Zero;
        var a5 = Vector128<float>.Zero;
        var a6 = Vector128<float>.Zero;
        var a7 = Vector128<float>.Zero;
        var a8 = Vector128<float>.Zero;
        var a9 = Vector128<float>.Zero;
        var a10 = Vector128<float>.Zero;
        var a11 = Vector128<float>.Zero;
        var a12 = Vector128<float>.Zero;
        var a13 = Vector128<float>.Zero;
        var a14 = Vector128<float>.Zero;
        var a15 = Vector128<float>.Zero;
        for (var s = 0; s < count; s++)
        {
            var row = values + (long)(first + s) * valueStride;
            var weight = Vector128.Create(scores[s] * valueScales[(long)(first + s) * scaleStride]);
            for (var c = 0; c < 4; c++)
            {
                var bytes = AdvSimd.LoadVector128(row + c * 16);
                var low = AdvSimd.SignExtendWideningLower(bytes.GetLower());
                var high = AdvSimd.SignExtendWideningUpper(bytes);
                var f0 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(low.GetLower()));
                var f1 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(low));
                var f2 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(high.GetLower()));
                var f3 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(high));
                switch (c)
                {
                    case 0:
                        a0 = AdvSimd.FusedMultiplyAdd(a0, weight, f0);
                        a1 = AdvSimd.FusedMultiplyAdd(a1, weight, f1);
                        a2 = AdvSimd.FusedMultiplyAdd(a2, weight, f2);
                        a3 = AdvSimd.FusedMultiplyAdd(a3, weight, f3);
                        break;
                    case 1:
                        a4 = AdvSimd.FusedMultiplyAdd(a4, weight, f0);
                        a5 = AdvSimd.FusedMultiplyAdd(a5, weight, f1);
                        a6 = AdvSimd.FusedMultiplyAdd(a6, weight, f2);
                        a7 = AdvSimd.FusedMultiplyAdd(a7, weight, f3);
                        break;
                    case 2:
                        a8 = AdvSimd.FusedMultiplyAdd(a8, weight, f0);
                        a9 = AdvSimd.FusedMultiplyAdd(a9, weight, f1);
                        a10 = AdvSimd.FusedMultiplyAdd(a10, weight, f2);
                        a11 = AdvSimd.FusedMultiplyAdd(a11, weight, f3);
                        break;
                    default:
                        a12 = AdvSimd.FusedMultiplyAdd(a12, weight, f0);
                        a13 = AdvSimd.FusedMultiplyAdd(a13, weight, f1);
                        a14 = AdvSimd.FusedMultiplyAdd(a14, weight, f2);
                        a15 = AdvSimd.FusedMultiplyAdd(a15, weight, f3);
                        break;
                }
            }
        }
        Store4(output, a0, a1, a2, a3);
        Store4(output + 16, a4, a5, a6, a7);
        Store4(output + 32, a8, a9, a10, a11);
        Store4(output + 48, a12, a13, a14, a15);
    }

    public static void AttentionTile(in AttentionTile tile) => NeonAttention.Tile(tile);

    // The per-head tile NeonAttention replaced, kept as the bit-exact reference for its tests.
    internal static void AttentionTilePerHead(in AttentionTile tile)
    {
        var count = tile.Count;
        for (var h = 0; h < tile.Heads; h++)
        {
            var scores = tile.Scores + h * count;
            var query = tile.Queries + h * 48;
            var scale = tile.QueryScales[h] * InverseSqrtQkDim;
            for (var s = 0; s < count; s++)
                scores[s] = DotInt8x48(query, tile.Keys + (long)(tile.First + s) * tile.KeyStride) * tile.KeyScales[(long)(tile.First + s) * tile.ScaleStride] * scale;
            Softmax(scores, count);
            if (tile.Probabilities != null)
                for (var s = 0; s < count; s++) tile.Probabilities[h * tile.ProbabilityStride + tile.First + s] = scores[s];
        }
        for (var h = 0; h < tile.Heads; h += 2)
            for (var column = 0; column < 64; column += 16)
            {
                if (tile.Heads - h >= 2) AccumulateValuePair(tile, h, column);
                else AccumulateValueSingle(tile, h, column);
            }
    }

    static void AccumulateValuePair(in AttentionTile tile, int head, int column)
    {
        var count = tile.Count;
        var first = tile.Scores + head * count;
        var second = first + count;
        var a0 = Vector128<float>.Zero;
        var a1 = a0;
        var a2 = a0;
        var a3 = a0;
        var b0 = a0;
        var b1 = a0;
        var b2 = a0;
        var b3 = a0;
        for (var s = 0; s < count; s++)
        {
            Widen(AdvSimd.LoadVector128(tile.Values + (long)(tile.First + s) * tile.ValueStride + column), out var f0, out var f1, out var f2, out var f3);
            var valueScale = tile.ValueScales[(long)(tile.First + s) * tile.ScaleStride];
            var wa = Vector128.Create(first[s] * valueScale);
            var wb = Vector128.Create(second[s] * valueScale);
            a0 = AdvSimd.FusedMultiplyAdd(a0, wa, f0);
            a1 = AdvSimd.FusedMultiplyAdd(a1, wa, f1);
            a2 = AdvSimd.FusedMultiplyAdd(a2, wa, f2);
            a3 = AdvSimd.FusedMultiplyAdd(a3, wa, f3);
            b0 = AdvSimd.FusedMultiplyAdd(b0, wb, f0);
            b1 = AdvSimd.FusedMultiplyAdd(b1, wb, f1);
            b2 = AdvSimd.FusedMultiplyAdd(b2, wb, f2);
            b3 = AdvSimd.FusedMultiplyAdd(b3, wb, f3);
        }
        Store4(tile.Output + head * 64 + column, a0, a1, a2, a3);
        Store4(tile.Output + (head + 1) * 64 + column, b0, b1, b2, b3);
    }

    static void AccumulateValueSingle(in AttentionTile tile, int head, int column)
    {
        var count = tile.Count;
        var weights = tile.Scores + head * count;
        var a0 = Vector128<float>.Zero;
        var a1 = a0;
        var a2 = a0;
        var a3 = a0;
        for (var s = 0; s < count; s++)
        {
            Widen(AdvSimd.LoadVector128(tile.Values + (long)(tile.First + s) * tile.ValueStride + column), out var f0, out var f1, out var f2, out var f3);
            var weight = Vector128.Create(weights[s] * tile.ValueScales[(long)(tile.First + s) * tile.ScaleStride]);
            a0 = AdvSimd.FusedMultiplyAdd(a0, weight, f0);
            a1 = AdvSimd.FusedMultiplyAdd(a1, weight, f1);
            a2 = AdvSimd.FusedMultiplyAdd(a2, weight, f2);
            a3 = AdvSimd.FusedMultiplyAdd(a3, weight, f3);
        }
        Store4(tile.Output + head * 64 + column, a0, a1, a2, a3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void Widen(Vector128<sbyte> bytes, out Vector128<float> f0, out Vector128<float> f1, out Vector128<float> f2, out Vector128<float> f3)
    {
        var low = AdvSimd.SignExtendWideningLower(bytes.GetLower());
        var high = AdvSimd.SignExtendWideningUpper(bytes);
        f0 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(low.GetLower()));
        f1 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(low));
        f2 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(high.GetLower()));
        f3 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(high));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void Store4(float* destination, Vector128<float> a, Vector128<float> b, Vector128<float> c, Vector128<float> d)
    {
        AdvSimd.Store(destination, a);
        AdvSimd.Store(destination + 4, b);
        AdvSimd.Store(destination + 8, c);
        AdvSimd.Store(destination + 12, d);
    }
}
#endif
