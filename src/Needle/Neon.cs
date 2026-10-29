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
                    var l0 = AdvSimd.LoadAndReplicateToVector128(factor);
                    a00 = AdvSimd.FusedMultiplyAdd(a00, l0, z0);
                    a01 = AdvSimd.FusedMultiplyAdd(a01, l0, z1);
                    a02 = AdvSimd.FusedMultiplyAdd(a02, l0, z2);
                    a03 = AdvSimd.FusedMultiplyAdd(a03, l0, z3);
                    var l1 = AdvSimd.LoadAndReplicateToVector128(factor + 1);
                    a10 = AdvSimd.FusedMultiplyAdd(a10, l1, z0);
                    a11 = AdvSimd.FusedMultiplyAdd(a11, l1, z1);
                    a12 = AdvSimd.FusedMultiplyAdd(a12, l1, z2);
                    a13 = AdvSimd.FusedMultiplyAdd(a13, l1, z3);
                    var l2 = AdvSimd.LoadAndReplicateToVector128(factor + 2);
                    a20 = AdvSimd.FusedMultiplyAdd(a20, l2, z0);
                    a21 = AdvSimd.FusedMultiplyAdd(a21, l2, z1);
                    a22 = AdvSimd.FusedMultiplyAdd(a22, l2, z2);
                    a23 = AdvSimd.FusedMultiplyAdd(a23, l2, z3);
                    var l3 = AdvSimd.LoadAndReplicateToVector128(factor + 3);
                    a30 = AdvSimd.FusedMultiplyAdd(a30, l3, z0);
                    a31 = AdvSimd.FusedMultiplyAdd(a31, l3, z1);
                    a32 = AdvSimd.FusedMultiplyAdd(a32, l3, z2);
                    a33 = AdvSimd.FusedMultiplyAdd(a33, l3, z3);
                    column += cols;
                    factor += rows;
                }
                AdvSimd.Store(tile + c, a00);
                AdvSimd.Store(tile + c + 4, a01);
                AdvSimd.Store(tile + c + 8, a02);
                AdvSimd.Store(tile + c + 12, a03);
                AdvSimd.Store(tile + stride1 + c, a10);
                AdvSimd.Store(tile + stride1 + c + 4, a11);
                AdvSimd.Store(tile + stride1 + c + 8, a12);
                AdvSimd.Store(tile + stride1 + c + 12, a13);
                AdvSimd.Store(tile + stride2 + c, a20);
                AdvSimd.Store(tile + stride2 + c + 4, a21);
                AdvSimd.Store(tile + stride2 + c + 8, a22);
                AdvSimd.Store(tile + stride2 + c + 12, a23);
                AdvSimd.Store(tile + stride3 + c, a30);
                AdvSimd.Store(tile + stride3 + c + 4, a31);
                AdvSimd.Store(tile + stride3 + c + 8, a32);
                AdvSimd.Store(tile + stride3 + c + 12, a33);
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
                for (var j = 0; j < cols; j++)
                {
                    var r0 = AdvSimd.LoadVector128(row);
                    var r1 = AdvSimd.LoadVector128(row + 4);
                    var r2 = AdvSimd.LoadVector128(row + 8);
                    var r3 = AdvSimd.LoadVector128(row + 12);
                    var s0 = AdvSimd.LoadAndReplicateToVector128(scalars);
                    a00 = AdvSimd.FusedMultiplyAdd(a00, s0, r0);
                    a01 = AdvSimd.FusedMultiplyAdd(a01, s0, r1);
                    a02 = AdvSimd.FusedMultiplyAdd(a02, s0, r2);
                    a03 = AdvSimd.FusedMultiplyAdd(a03, s0, r3);
                    var s1 = AdvSimd.LoadAndReplicateToVector128(scalars + stride1);
                    a10 = AdvSimd.FusedMultiplyAdd(a10, s1, r0);
                    a11 = AdvSimd.FusedMultiplyAdd(a11, s1, r1);
                    a12 = AdvSimd.FusedMultiplyAdd(a12, s1, r2);
                    a13 = AdvSimd.FusedMultiplyAdd(a13, s1, r3);
                    var s2 = AdvSimd.LoadAndReplicateToVector128(scalars + stride2);
                    a20 = AdvSimd.FusedMultiplyAdd(a20, s2, r0);
                    a21 = AdvSimd.FusedMultiplyAdd(a21, s2, r1);
                    a22 = AdvSimd.FusedMultiplyAdd(a22, s2, r2);
                    a23 = AdvSimd.FusedMultiplyAdd(a23, s2, r3);
                    var s3 = AdvSimd.LoadAndReplicateToVector128(scalars + stride3);
                    a30 = AdvSimd.FusedMultiplyAdd(a30, s3, r0);
                    a31 = AdvSimd.FusedMultiplyAdd(a31, s3, r1);
                    a32 = AdvSimd.FusedMultiplyAdd(a32, s3, r2);
                    a33 = AdvSimd.FusedMultiplyAdd(a33, s3, r3);
                    row += cols;
                    scalars++;
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
