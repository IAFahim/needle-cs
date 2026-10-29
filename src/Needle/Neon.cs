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
        for (var i = 0; i < n; i += 4)
        {
            var v = AdvSimd.LoadVector128(x + i);
            AdvSimd.Store(x + i, AdvSimd.Multiply(v, Sigmoid(v)));
        }
    }

    public static void MultiplyBySigmoid(float* x, float* gate, int n)
    {
        for (var i = 0; i < n; i += 4)
            AdvSimd.Store(x + i, AdvSimd.Multiply(AdvSimd.LoadVector128(x + i), Sigmoid(AdvSimd.LoadVector128(gate + i))));
    }

    public static float SumOfSquares(float* x, int n)
    {
        var s0 = Vector128<float>.Zero;
        var s1 = Vector128<float>.Zero;
        for (var i = 0; i < n; i += 8)
        {
            var a = AdvSimd.LoadVector128(x + i);
            var b = AdvSimd.LoadVector128(x + i + 4);
            s0 = AdvSimd.FusedMultiplyAdd(s0, a, a);
            s1 = AdvSimd.FusedMultiplyAdd(s1, b, b);
        }
        return HorizontalSum(AdvSimd.Add(s0, s1));
    }

    public static float Dot(float* a, float* b, int n)
    {
        var s0 = Vector128<float>.Zero;
        var s1 = Vector128<float>.Zero;
        for (var i = 0; i < n; i += 8)
        {
            s0 = AdvSimd.FusedMultiplyAdd(s0, AdvSimd.LoadVector128(a + i), AdvSimd.LoadVector128(b + i));
            s1 = AdvSimd.FusedMultiplyAdd(s1, AdvSimd.LoadVector128(a + i + 4), AdvSimd.LoadVector128(b + i + 4));
        }
        return HorizontalSum(AdvSimd.Add(s0, s1));
    }

    public static void RmsUnit(float* x, int n, float* destination)
    {
        var inverse = Vector128.Create(1f / MathF.Sqrt(SumOfSquares(x, n) / n + RmsEpsilon));
        for (var i = 0; i < n; i += 4) AdvSimd.Store(destination + i, AdvSimd.Multiply(AdvSimd.LoadVector128(x + i), inverse));
    }

    public static void ZeroCenteredRms(float* x, float* scale, int n, float* destination)
    {
        var inverse = Vector128.Create(1f / MathF.Sqrt(SumOfSquares(x, n) / n + RmsEpsilon));
        var one = Vector128.Create(1f);
        for (var i = 0; i < n; i += 4)
            AdvSimd.Store(destination + i, AdvSimd.Multiply(AdvSimd.Multiply(AdvSimd.Add(one, AdvSimd.LoadVector128(scale + i)), AdvSimd.LoadVector128(x + i)), inverse));
    }

    public static void Softmax(float* s, int n)
    {
        var peak = float.NegativeInfinity;
        for (var i = 0; i < n; i++) peak = MathF.Max(peak, s[i]);
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
        for (var i = 0; i < n; i += 4) AdvSimd.Store(destination + i, AdvSimd.Subtract(AdvSimd.LoadVector128(a + i), AdvSimd.LoadVector128(b + i)));
    }

    public static void MultiplyAccumulate(float* destination, float* a, float* b, int n)
    {
        for (var i = 0; i < n; i += 4)
            AdvSimd.Store(destination + i, AdvSimd.FusedMultiplyAdd(AdvSimd.LoadVector128(destination + i), AdvSimd.LoadVector128(a + i), AdvSimd.LoadVector128(b + i)));
    }

    public static void ScaleAdd(float* destination, float scale, float* x, float xScale, int n)
    {
        var s = Vector128.Create(scale);
        var xs = Vector128.Create(xScale);
        for (var i = 0; i < n; i += 4)
            AdvSimd.Store(destination + i, AdvSimd.FusedMultiplyAdd(AdvSimd.Multiply(s, AdvSimd.LoadVector128(destination + i)), xs, AdvSimd.LoadVector128(x + i)));
    }

    public static void AddScaled(float* destination, float* x, float scale, int n)
    {
        var s = Vector128.Create(scale);
        for (var i = 0; i < n; i += 4) AdvSimd.Store(destination + i, AdvSimd.FusedMultiplyAdd(AdvSimd.LoadVector128(destination + i), s, AdvSimd.LoadVector128(x + i)));
    }

    public static void Glu(float* value, float* gate, float* destination, int n)
    {
        for (var i = 0; i < n; i += 4)
            AdvSimd.Store(destination + i, AdvSimd.Multiply(AdvSimd.LoadVector128(value + i), Sigmoid(AdvSimd.LoadVector128(gate + i))));
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

    public static float AbsMax(float* x, int n)
    {
        var peak = Vector128<float>.Zero;
        for (var i = 0; i < n; i += 4) peak = AdvSimd.Max(peak, AdvSimd.Abs(AdvSimd.LoadVector128(x + i)));
        return AdvSimd.Arm64.MaxAcross(peak).ToScalar();
    }

    public static void RoundTrip(float* x, int n, float inverse, float scale)
    {
        var inv = Vector128.Create(inverse);
        var sc = Vector128.Create(scale);
        var low = Vector128.Create(-128f);
        var high = Vector128.Create(127f);
        for (var i = 0; i < n; i += 4)
        {
            var q = AdvSimd.RoundToNearest(AdvSimd.Multiply(AdvSimd.LoadVector128(x + i), inv));
            AdvSimd.Store(x + i, AdvSimd.Multiply(AdvSimd.Min(AdvSimd.Max(q, low), high), sc));
        }
    }

    public static float QuantizeInt8(float* x, int n, sbyte* destination)
    {
        var peak = AbsMax(x, n);
        var scale = peak > 0f ? peak / 127f : 1f;
        var inv = Vector128.Create(1f / scale);
        for (var i = 0; i < n; i += 4)
        {
            var wide = AdvSimd.ConvertToInt32RoundToZero(AdvSimd.Multiply(AdvSimd.LoadVector128(x + i), inv));
            var narrow = AdvSimd.ExtractNarrowingSaturateLower(wide).ToVector128();
            var bytes = AdvSimd.ExtractNarrowingSaturateLower(narrow);
            *(int*)(destination + i) = bytes.AsInt32().ToScalar();
        }
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

    public static void Prepare(float* x, int tokens, int cols, int bits, in Activations a)
    {
        var rotated = stackalloc float[Cq.GroupSize];
        var quantized = stackalloc sbyte[Cq.GroupSize];
        for (var t = 0; t < tokens; t++)
            for (var g = 0; g < a.Groups; g++)
            {
                var slot = t * a.Stride + g;
                Hadamard(x + (long)t * cols + g * Cq.GroupSize, rotated);
                a.Scales[slot] = QuantizeInt8(rotated, Cq.GroupSize, quantized);
                Planes(quantized, bits, a.Values + ((long)t * a.Groups + g) * Cq.GroupSize, out var sum);
                a.Offsets[slot] = Codebook.Bias * sum;
            }
    }

    public static void Planes(sbyte* quantized, int bits, sbyte* destination, out int sum)
    {
        sum = 0;
        if (bits == 2)
        {
            for (var half = 0; half < 2; half++)
            {
                var q0 = AdvSimd.LoadVector128(quantized + half * 64);
                var q1 = AdvSimd.LoadVector128(quantized + half * 64 + 16);
                var q2 = AdvSimd.LoadVector128(quantized + half * 64 + 32);
                var q3 = AdvSimd.LoadVector128(quantized + half * 64 + 48);
                StorePlanes2(q0, q1, q2, q3, destination + half * 16);
                sum += SumOf(q0) + SumOf(q1) + SumOf(q2) + SumOf(q3);
            }
        }
        else
        {
            for (var half = 0; half < 2; half++)
            {
                var q0 = AdvSimd.LoadVector128(quantized + half * 64);
                var q1 = AdvSimd.LoadVector128(quantized + half * 64 + 16);
                var q2 = AdvSimd.LoadVector128(quantized + half * 64 + 32);
                var q3 = AdvSimd.LoadVector128(quantized + half * 64 + 48);
                StorePlanes4(q0, q1, q2, q3, destination + half * 64);
                sum += SumOf(q0) + SumOf(q1) + SumOf(q2) + SumOf(q3);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void StorePlanes2(Vector128<sbyte> q0, Vector128<sbyte> q1, Vector128<sbyte> q2, Vector128<sbyte> q3, sbyte* destination)
    {
        var table = (q0.AsByte(), q1.AsByte(), q2.AsByte(), q3.AsByte());
        for (var p = 0; p < 4; p++)
        {
            var baseIndex = (byte)p;
            var first = Vector64.Create(baseIndex, (byte)(baseIndex + 4), (byte)(baseIndex + 8), (byte)(baseIndex + 12), (byte)(baseIndex + 16), (byte)(baseIndex + 20), (byte)(baseIndex + 24), (byte)(baseIndex + 28));
            var second = Vector64.Create((byte)(baseIndex + 32), (byte)(baseIndex + 36), (byte)(baseIndex + 40), (byte)(baseIndex + 44), (byte)(baseIndex + 48), (byte)(baseIndex + 52), (byte)(baseIndex + 56), (byte)(baseIndex + 60));
            AdvSimd.Store(destination + p * 32, AdvSimd.VectorTableLookup(table, first).AsSByte());
            AdvSimd.Store(destination + p * 32 + 8, AdvSimd.VectorTableLookup(table, second).AsSByte());
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void StorePlanes4(Vector128<sbyte> q0, Vector128<sbyte> q1, Vector128<sbyte> q2, Vector128<sbyte> q3, sbyte* destination)
    {
        var table = (q0.AsByte(), q1.AsByte(), q2.AsByte(), q3.AsByte());
        for (var p = 0; p < 2; p++)
        {
            var baseIndex = Vector64.Create((byte)p, (byte)(p + 2), (byte)(p + 4), (byte)(p + 6), (byte)(p + 8), (byte)(p + 10), (byte)(p + 12), (byte)(p + 14));
            for (var quarter = 0; quarter < 4; quarter++)
            {
                var index = AdvSimd.Add(baseIndex, Vector64.Create((byte)(16 * quarter)));
                AdvSimd.Store(destination + p * 32 + quarter * 8, AdvSimd.VectorTableLookup(table, index).AsSByte());
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int SumOf(Vector128<sbyte> q)
    {
        if (UseDotProduct)
        {
            var sums = Dp.DotProduct(Vector128<int>.Zero, Vector128.Create((sbyte)1), q);
            return AdvSimd.Arm64.AddAcross(sums).ToScalar();
        }
        var low = AdvSimd.SignExtendWideningLower(q.GetLower());
        var high = AdvSimd.SignExtendWideningUpper(q);
        var wide = AdvSimd.AddPairwiseWideningAndAdd(Vector128<int>.Zero, low);
        wide = AdvSimd.AddPairwiseWideningAndAdd(wide, high);
        return AdvSimd.Arm64.AddAcross(wide).ToScalar();
    }

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
            var bits = AdvSimd.Add(AdvSimd.ShiftLeftLogical(magnitude, 13), Vector128.Create(0x38000000u));
            var nonZero = AdvSimd.CompareGreaterThan(magnitude.AsInt32(), Vector128<int>.Zero).AsUInt32();
            var single = AdvSimd.Or(AdvSimd.And(bits, nonZero), sign).AsSingle();
            AdvSimd.Store(factors + g, AdvSimd.Multiply(single, multiplier));
        }
        for (; g < groups; g++) factors[g] = Scalar.HalfToSingle(norms[g]) * scale;
    }

    public static float ScaleFactors(float* factors, float* scaled, float* scales, float* offsets, int stride)
    {
        var bias = Vector128<float>.Zero;
        for (var v = 0; v < stride; v += 4)
        {
            var f = AdvSimd.Multiply(AdvSimd.LoadVector128(factors + v), AdvSimd.LoadVector128(scales + v));
            AdvSimd.Store(scaled + v, f);
            bias = AdvSimd.FusedMultiplyAdd(bias, f, AdvSimd.LoadVector128(offsets + v));
        }
        return HorizontalSum(bias);
    }

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

    public static float DotRow(in Cq w, byte* packedRows, in Activations a, int token, float* factors, float* scaled)
    {
        var groups = a.Groups;
        var bias = ScaleFactors(factors, scaled, a.Scales + token * a.Stride, a.Offsets + token * a.Stride, a.Stride);
        var table = AdvSimd.LoadVector128(w.Book->Bytes);
        var mask = Vector128.Create(w.Bits == 2 ? (byte)0x03 : (byte)0x0F);
        var values = a.Values + (long)token * a.Cols;
        var accumulator = 0f;
        for (var g = 0; g < groups; g++)
        {
            var sum = GroupDot(packedRows + g * w.RowBytes / groups, values + g * Cq.GroupSize, mask, table);
            accumulator += HorizontalSum(AdvSimd.ConvertToSingle(sum)) * scaled[g];
        }
        return accumulator - bias;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<int> GroupDot(byte* packed, sbyte* values, Vector128<byte> mask, Vector128<byte> table)
    {
        var two = mask.GetElement(0) == 3;
        var sum = Vector128<int>.Zero;
        if (two)
        {
            for (var r = 0; r < 2; r++)
            {
                var source = AdvSimd.LoadVector128(packed + r * 16);
                var act = values + r * 16;
                sum = Dot16(sum, Weights(table, AdvSimd.And(source, mask)), AdvSimd.LoadVector128(act));
                sum = Dot16(sum, Weights(table, AdvSimd.And(AdvSimd.ShiftRightLogical(source, 2), mask)), AdvSimd.LoadVector128(act + 32));
                sum = Dot16(sum, Weights(table, AdvSimd.And(AdvSimd.ShiftRightLogical(source, 4), mask)), AdvSimd.LoadVector128(act + 64));
                sum = Dot16(sum, Weights(table, AdvSimd.And(AdvSimd.ShiftRightLogical(source, 6), mask)), AdvSimd.LoadVector128(act + 96));
            }
            return sum;
        }
        for (var r = 0; r < 4; r++)
        {
            var source = AdvSimd.LoadVector128(packed + r * 16);
            var act = values + (r >> 1) * 64 + (r & 1) * 16;
            sum = Dot16(sum, Weights(table, AdvSimd.And(source, mask)), AdvSimd.LoadVector128(act));
            sum = Dot16(sum, Weights(table, AdvSimd.And(AdvSimd.ShiftRightLogical(source, 4), mask)), AdvSimd.LoadVector128(act + 32));
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<sbyte> Weights(Vector128<byte> table, Vector128<byte> index)
    {
        var low = AdvSimd.VectorTableLookup(table, index.GetLower());
        var high = AdvSimd.VectorTableLookup(table, index.GetUpper());
        return Vector128.WithUpper(low.AsSByte().ToVector128(), high.AsSByte());
    }

    public static void DotTile(in Cq w, byte* packedRows, in Activations a, int token, float* factors, float* scaled, float* y, int ldy)
    {
        var groups = a.Groups;
        var stride = a.Stride;
        var cols = a.Cols;
        var bias0 = ScaleFactors(factors, scaled, a.Scales + token * stride, a.Offsets + token * stride, stride);
        var bias1 = ScaleFactors(factors, scaled + stride, a.Scales + (token + 1) * stride, a.Offsets + (token + 1) * stride, stride);
        var bias2 = ScaleFactors(factors, scaled + 2 * stride, a.Scales + (token + 2) * stride, a.Offsets + (token + 2) * stride, stride);
        var bias3 = ScaleFactors(factors, scaled + 3 * stride, a.Scales + (token + 3) * stride, a.Offsets + (token + 3) * stride, stride);
        var table = AdvSimd.LoadVector128(w.Book->Bytes);
        var mask = Vector128.Create(w.Bits == 2 ? (byte)0x03 : (byte)0x0F);
        var v0 = a.Values + (long)token * cols;
        var v1 = v0 + cols;
        var v2 = v1 + cols;
        var v3 = v2 + cols;
        var accumulator0 = 0f;
        var accumulator1 = 0f;
        var accumulator2 = 0f;
        var accumulator3 = 0f;
        var packedStep = w.RowBytes / groups;
        var regs = w.Bits == 2 ? 2 : 4;
        for (var g = 0; g < groups; g++)
        {
            var packed = packedRows + g * packedStep;
            var sum0 = Vector128<int>.Zero;
            var sum1 = Vector128<int>.Zero;
            var sum2 = Vector128<int>.Zero;
            var sum3 = Vector128<int>.Zero;
            for (var r = 0; r < regs; r++)
            {
                var source = AdvSimd.LoadVector128(packed + r * 16);
                var act = g * Cq.GroupSize + (w.Bits == 2 ? r * 16 : (r >> 1) * 64 + (r & 1) * 16);
                if (w.Bits == 2)
                {
                    TilePlane(table, mask, source, 0, act, v0, v1, v2, v3, ref sum0, ref sum1, ref sum2, ref sum3);
                    TilePlane(table, mask, AdvSimd.ShiftRightLogical(source, 2), 1, act, v0, v1, v2, v3, ref sum0, ref sum1, ref sum2, ref sum3);
                    TilePlane(table, mask, AdvSimd.ShiftRightLogical(source, 4), 2, act, v0, v1, v2, v3, ref sum0, ref sum1, ref sum2, ref sum3);
                    TilePlane(table, mask, AdvSimd.ShiftRightLogical(source, 6), 3, act, v0, v1, v2, v3, ref sum0, ref sum1, ref sum2, ref sum3);
                }
                else
                {
                    TilePlane(table, mask, source, 0, act, v0, v1, v2, v3, ref sum0, ref sum1, ref sum2, ref sum3);
                    TilePlane(table, mask, AdvSimd.ShiftRightLogical(source, 4), 1, act, v0, v1, v2, v3, ref sum0, ref sum1, ref sum2, ref sum3);
                }
            }
            accumulator0 += HorizontalSum(AdvSimd.ConvertToSingle(sum0)) * scaled[g];
            accumulator1 += HorizontalSum(AdvSimd.ConvertToSingle(sum1)) * scaled[stride + g];
            accumulator2 += HorizontalSum(AdvSimd.ConvertToSingle(sum2)) * scaled[2 * stride + g];
            accumulator3 += HorizontalSum(AdvSimd.ConvertToSingle(sum3)) * scaled[3 * stride + g];
        }
        y[0] = accumulator0 - bias0;
        y[ldy] = accumulator1 - bias1;
        y[2 * ldy] = accumulator2 - bias2;
        y[3 * ldy] = accumulator3 - bias3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void TilePlane(Vector128<byte> table, Vector128<byte> mask, Vector128<byte> index, int plane, int act, sbyte* v0, sbyte* v1, sbyte* v2, sbyte* v3, ref Vector128<int> sum0, ref Vector128<int> sum1, ref Vector128<int> sum2, ref Vector128<int> sum3)
    {
        var offset = act + plane * 32;
        var weights = Weights(table, AdvSimd.And(index, mask));
        sum0 = Dot16(sum0, weights, AdvSimd.LoadVector128(v0 + offset));
        sum1 = Dot16(sum1, weights, AdvSimd.LoadVector128(v1 + offset));
        sum2 = Dot16(sum2, weights, AdvSimd.LoadVector128(v2 + offset));
        sum3 = Dot16(sum3, weights, AdvSimd.LoadVector128(v3 + offset));
    }

    public static void MatMulRows(in Cq w, in Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
    {
        var factors = stackalloc float[a.Stride];
        var scaled = stackalloc float[a.Stride * 4];
        var groups = w.Groups;
        if ((rowEnd - rowStart) * 4 <= tokens)
        {
            for (var t = 0; t + 4 <= tokens; t += 4)
                for (var n = rowStart; n < rowEnd; n++)
                {
                    LoadFactors(w.Norms + (long)n * groups, groups, w.Book->ByteScale, factors);
                    DotTile(w, w.Indices + (long)n * w.RowBytes, a, t, factors, scaled, y + (long)t * ldy + n, ldy);
                }
            for (var t = tokens & ~3; t < tokens; t++)
                for (var n = rowStart; n < rowEnd; n++)
                {
                    LoadFactors(w.Norms + (long)n * groups, groups, w.Book->ByteScale, factors);
                    y[(long)t * ldy + n] = DotRow(w, w.Indices + (long)n * w.RowBytes, a, t, factors, scaled);
                }
            return;
        }
        for (var n = rowStart; n < rowEnd; n++)
        {
            LoadFactors(w.Norms + (long)n * groups, groups, w.Book->ByteScale, factors);
            var t = 0;
            for (; t + 4 <= tokens; t += 4) DotTile(w, w.Indices + (long)n * w.RowBytes, a, t, factors, scaled, y + (long)t * ldy + n, ldy);
            for (; t < tokens; t++) y[(long)t * ldy + n] = DotRow(w, w.Indices + (long)n * w.RowBytes, a, t, factors, scaled);
        }
    }

    public static void ModulateActivate(float* d2, float* condition, float* a, float* b2, int width)
    {
        for (var i = 0; i < width; i += 4)
            AdvSimd.Store(a + i, AdvSimd.FusedMultiplyAdd(
                AdvSimd.LoadVector128(b2 + i),
                AdvSimd.Multiply(AdvSimd.LoadVector128(d2 + i), AdvSimd.LoadVector128(condition + i)),
                AdvSimd.LoadVector128(a + i)));
    }

    public static void Condition(float* condV, float* condU, float* input, int dim, int width, float* condition)
    {
        var logits0 = Vector128<float>.Zero;
        var logits1 = Vector128<float>.Zero;
        for (var i = 0; i < dim; i++)
        {
            var broadcast = Vector128.Create(input[i]);
            logits0 = AdvSimd.FusedMultiplyAdd(logits0, broadcast, AdvSimd.LoadVector128(condV + i * 8));
            logits1 = AdvSimd.FusedMultiplyAdd(logits1, broadcast, AdvSimd.LoadVector128(condV + i * 8 + 4));
        }
        var weights = stackalloc float[8];
        AdvSimd.Store(weights, logits0);
        AdvSimd.Store(weights + 4, logits1);
        Softmax(weights, 8);
        for (var n = 0; n < width; n += 4)
        {
            var acc = Vector128.Create(1f);
            for (var j = 0; j < 8; j++) acc = AdvSimd.FusedMultiplyAdd(acc, Vector128.Create(weights[j]), AdvSimd.LoadVector128(condU + j * width + n));
            AdvSimd.Store(condition + n, acc);
        }
    }

    public static void Kron(float* z, float* left, float* right, int rows, int cols, float* result)
    {
        var scratch = stackalloc float[rows * cols];
        for (var k = 0; k < rows; k++)
            for (var c = 0; c < cols; c += 4)
            {
                var acc = Vector128<float>.Zero;
                for (var i = 0; i < rows; i++) acc = AdvSimd.FusedMultiplyAdd(acc, Vector128.Create(left[i * rows + k]), AdvSimd.LoadVector128(z + i * cols + c));
                AdvSimd.Store(scratch + k * cols + c, acc);
            }
        for (var k = 0; k < rows; k++)
            for (var c = 0; c < cols; c += 4)
            {
                var acc = Vector128<float>.Zero;
                for (var j = 0; j < cols; j++) acc = AdvSimd.FusedMultiplyAdd(acc, Vector128.Create(scratch[k * cols + j]), AdvSimd.LoadVector128(right + j * cols + c));
                AdvSimd.Store(result + k * cols + c, acc);
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

    public static void AttentionTile(in AttentionTile tile)
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
