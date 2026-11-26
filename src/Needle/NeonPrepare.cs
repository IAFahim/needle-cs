#if NEEDLE_SIMD
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Needle;

// Fused NEON activation preparation: fake-A8 normalisation, the 128-point Hadamard rotation, int8
// quantisation and the plane layout, one group at a time with the group held in L1 and the rotation's
// strided stages held in registers. Every value is produced by the same operations, in the same
// order, as Neon.RmsFakeA8 / CenteredRmsFakeA8 / GateFakeA8 followed by Neon.Prepare, so the
// activations are bit-identical.
[SkipLocalsInit]
internal static unsafe class NeonPrepare
{
    const float HadamardScale = 0.08838834764831845f;
    const float RmsEpsilon = 1e-6f;

    // Neon.RmsFakeA8(x, n, scratch) then Neon.Prepare(scratch, 1, n, bits, a).
    public static void RmsFakeA8(float* x, int n, int bits, in Activations a)
    {
        SumOfSquaresAndPeak(x, null, n, out var sumOfSquares, out var rawPeak);
        var scalarInv = 1f / MathF.Sqrt(sumOfSquares / n + RmsEpsilon);
        // |x * inv| is monotonic in |x|, so the peak of the scaled values is the scaled raw peak.
        var peak = rawPeak * scalarInv;
        var fake = FakeScale(peak);
        var inverse = Vector128.Create(scalarInv);
        var p = stackalloc float[Cq.GroupSize];
        for (var g = 0; g < a.Groups; g++)
        {
            var source = x + g * Cq.GroupSize;
            for (var i = 0; i < Cq.GroupSize; i += 4)
                AdvSimd.Store(p + i, fake.Apply(AdvSimd.Multiply(AdvSimd.LoadVector128(source + i), inverse)));
            Pack(p, p, bits, a, g);
        }
    }

    // Neon.CenteredRmsFakeA8(x, scale, n, scratch) then Neon.Prepare(scratch, 1, n, bits, a).
    public static void CenteredRmsFakeA8(float* x, float* scale, int n, int bits, in Activations a)
    {
        SumOfSquaresAndPeak(x, scale, n, out var sumOfSquares, out var rawPeak);
        var scalarInv = 1f / MathF.Sqrt(sumOfSquares / n + RmsEpsilon);
        var peak = rawPeak * scalarInv;
        var fake = FakeScale(peak);
        var inverse = Vector128.Create(scalarInv);
        var one = Vector128.Create(1f);
        var p = stackalloc float[Cq.GroupSize];
        for (var g = 0; g < a.Groups; g++)
        {
            var offset = g * Cq.GroupSize;
            for (var i = 0; i < Cq.GroupSize; i += 4)
            {
                var v = AdvSimd.Multiply(AdvSimd.Multiply(AdvSimd.Add(one, AdvSimd.LoadVector128(scale + offset + i)), AdvSimd.LoadVector128(x + offset + i)), inverse);
                AdvSimd.Store(p + i, fake.Apply(v));
            }
            Pack(p, p, bits, a, g);
        }
    }

    // Neon.GateFakeA8(x, gate, n) then Neon.Prepare(x, 1, n, bits, a); x ends holding the fake-A8 values.
    public static void GateFakeA8(float* x, float* gate, int n, int bits, in Activations a)
    {
        var peakVector = Vector128<float>.Zero;
        for (var i = 0; i < n; i += 4)
        {
            var v = AdvSimd.Multiply(AdvSimd.LoadVector128(x + i), Neon.Sigmoid(AdvSimd.LoadVector128(gate + i)));
            AdvSimd.Store(x + i, v);
            peakVector = AdvSimd.Max(peakVector, AdvSimd.Abs(v));
        }
        var fake = FakeScale(AdvSimd.Arm64.MaxAcross(peakVector).ToScalar());
        var p = stackalloc float[Cq.GroupSize];
        for (var g = 0; g < a.Groups; g++)
        {
            var source = x + g * Cq.GroupSize;
            for (var i = 0; i < Cq.GroupSize; i += 4)
            {
                var v = fake.Apply(AdvSimd.LoadVector128(source + i));
                AdvSimd.Store(source + i, v);
            }
            Pack(source, p, bits, a, g);
        }
    }

    // Neon.Prepare: rotate, quantise and lay out every group of every token.
    public static void Prepare(float* x, int tokens, int cols, int bits, in Activations a)
    {
        var p = stackalloc float[Cq.GroupSize];
        for (var t = 0; t < tokens; t++)
        {
            var slice = a.Slice(t);
            for (var g = 0; g < a.Groups; g++)
                Pack(x + (long)t * cols + g * Cq.GroupSize, p, bits, slice, g);
        }
    }

    struct Fake
    {
        public Vector128<float> Inverse, Scale, Low, High;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly Vector128<float> Apply(Vector128<float> v) =>
            AdvSimd.Multiply(AdvSimd.Min(AdvSimd.Max(AdvSimd.RoundToNearest(AdvSimd.Multiply(v, Inverse)), Low), High), Scale);
    }

    // FinishFakeA8 / RoundTrip constants for a peak.
    static Fake FakeScale(float peak)
    {
        var scale = peak > 0f ? peak / 127f : 1f;
        Fake f;
        f.Inverse = Vector128.Create(1f / scale);
        f.Scale = Vector128.Create(scale);
        f.Low = Vector128.Create(-128f);
        f.High = Vector128.Create(127f);
        return f;
    }

    // Neon.SumOfSquares(x, n) (n a multiple of 8) plus the peak of |x| or of |(1 + scale) * x|.
    static void SumOfSquaresAndPeak(float* x, float* scale, int n, out float sumOfSquares, out float peak)
    {
        var s0 = Vector128<float>.Zero;
        var s1 = Vector128<float>.Zero;
        var m0 = Vector128<float>.Zero;
        var m1 = Vector128<float>.Zero;
        var one = Vector128.Create(1f);
        for (var i = 0; i < n; i += 8)
        {
            var a = AdvSimd.LoadVector128(x + i);
            var b = AdvSimd.LoadVector128(x + i + 4);
            s0 = AdvSimd.FusedMultiplyAdd(s0, a, a);
            s1 = AdvSimd.FusedMultiplyAdd(s1, b, b);
            if (scale != null)
            {
                a = AdvSimd.Multiply(AdvSimd.Add(one, AdvSimd.LoadVector128(scale + i)), a);
                b = AdvSimd.Multiply(AdvSimd.Add(one, AdvSimd.LoadVector128(scale + i + 4)), b);
            }
            m0 = AdvSimd.Max(m0, AdvSimd.Abs(a));
            m1 = AdvSimd.Max(m1, AdvSimd.Abs(b));
        }
        sumOfSquares = Neon.HorizontalSum(AdvSimd.Add(s0, s1)) + 0f;
        peak = AdvSimd.Arm64.MaxAcross(AdvSimd.Max(m0, m1)).ToScalar();
    }

    // Rotates the group at source into p (source may equal p), then quantises it into slot g of a.
    static void Pack(float* source, float* p, int bits, in Activations a, int g)
    {
        StridedStages(source, p, 0);
        StridedStages(source, p, 1);
        var scale = Vector128.Create(HadamardScale);
        var peak = Vector128<float>.Zero;
        for (var i = 0; i < Cq.GroupSize; i += 8)
        {
            var x = AdvSimd.LoadVector128(p + i);
            var y = AdvSimd.LoadVector128(p + i + 4);
            var r0 = AdvSimd.Multiply(Butterfly8(AdvSimd.Add(x, y)), scale);
            var r1 = AdvSimd.Multiply(Butterfly8(AdvSimd.Subtract(x, y)), scale);
            AdvSimd.Store(p + i, r0);
            AdvSimd.Store(p + i + 4, r1);
            peak = AdvSimd.Max(peak, AdvSimd.Max(AdvSimd.Abs(r0), AdvSimd.Abs(r1)));
        }
        var groupPeak = AdvSimd.Arm64.MaxAcross(peak).ToScalar();
        var groupScale = groupPeak > 0f ? groupPeak / 127f : 1f;
        var inv = Vector128.Create(1f / groupScale);
        var quantized = stackalloc sbyte[Cq.GroupSize];
        var sums = Vector128<short>.Zero;
        for (var i = 0; i < Cq.GroupSize; i += 16)
        {
            var w0 = AdvSimd.ConvertToInt32RoundToZero(AdvSimd.Multiply(AdvSimd.LoadVector128(p + i), inv));
            var w1 = AdvSimd.ConvertToInt32RoundToZero(AdvSimd.Multiply(AdvSimd.LoadVector128(p + i + 4), inv));
            var w2 = AdvSimd.ConvertToInt32RoundToZero(AdvSimd.Multiply(AdvSimd.LoadVector128(p + i + 8), inv));
            var w3 = AdvSimd.ConvertToInt32RoundToZero(AdvSimd.Multiply(AdvSimd.LoadVector128(p + i + 12), inv));
            var h0 = AdvSimd.ExtractNarrowingSaturateUpper(AdvSimd.ExtractNarrowingSaturateLower(w0), w1);
            var h1 = AdvSimd.ExtractNarrowingSaturateUpper(AdvSimd.ExtractNarrowingSaturateLower(w2), w3);
            var q = AdvSimd.ExtractNarrowingSaturateUpper(AdvSimd.ExtractNarrowingSaturateLower(h0), h1);
            AdvSimd.Store(quantized + i, q);
            sums = AdvSimd.AddPairwiseWideningAndAdd(sums, q);
        }
        var slot = g;
        a.Scales[slot] = groupScale;
        a.Offsets[slot] = Codebook.Bias * AdvSimd.Arm64.AddAcrossWidening(sums).ToScalar();
        var destination = a.Values + (long)g * Cq.GroupSize;
        if (bits == 2)
        {
            for (var half = 0; half < 2; half++)
            {
                var (p0, p1, p2, p3) = AdvSimd.Arm64.Load4xVector128AndUnzip(quantized + half * 64);
                AdvSimd.Store(destination + half * 16, p0);
                AdvSimd.Store(destination + half * 16 + 32, p1);
                AdvSimd.Store(destination + half * 16 + 64, p2);
                AdvSimd.Store(destination + half * 16 + 96, p3);
            }
        }
        else
        {
            for (var quarter = 0; quarter < 4; quarter++)
            {
                var (even, odd) = AdvSimd.Arm64.Load2xVector128AndUnzip(quantized + quarter * 32);
                var column = (quarter >> 1) * 64 + (quarter & 1) * 16;
                AdvSimd.Store(destination + column, even);
                AdvSimd.Store(destination + column + 32, odd);
            }
        }
    }

    // The 64/32/16/8-element butterfly stages of Neon.Hadamard for the vectors of one parity
    // (vector j covers elements 4j..4j+3; these stages pair vectors whose indices differ by an even
    // number, so each parity is independent and fits in sixteen registers).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void StridedStages(float* source, float* p, int parity)
    {
        var s = source + parity * 4;
        var v0 = AdvSimd.LoadVector128(s);
        var v1 = AdvSimd.LoadVector128(s + 8);
        var v2 = AdvSimd.LoadVector128(s + 16);
        var v3 = AdvSimd.LoadVector128(s + 24);
        var v4 = AdvSimd.LoadVector128(s + 32);
        var v5 = AdvSimd.LoadVector128(s + 40);
        var v6 = AdvSimd.LoadVector128(s + 48);
        var v7 = AdvSimd.LoadVector128(s + 56);
        var v8 = AdvSimd.LoadVector128(s + 64);
        var v9 = AdvSimd.LoadVector128(s + 72);
        var v10 = AdvSimd.LoadVector128(s + 80);
        var v11 = AdvSimd.LoadVector128(s + 88);
        var v12 = AdvSimd.LoadVector128(s + 96);
        var v13 = AdvSimd.LoadVector128(s + 104);
        var v14 = AdvSimd.LoadVector128(s + 112);
        var v15 = AdvSimd.LoadVector128(s + 120);
        Butterfly(ref v0, ref v8); Butterfly(ref v1, ref v9); Butterfly(ref v2, ref v10); Butterfly(ref v3, ref v11);
        Butterfly(ref v4, ref v12); Butterfly(ref v5, ref v13); Butterfly(ref v6, ref v14); Butterfly(ref v7, ref v15);
        Butterfly(ref v0, ref v4); Butterfly(ref v1, ref v5); Butterfly(ref v2, ref v6); Butterfly(ref v3, ref v7);
        Butterfly(ref v8, ref v12); Butterfly(ref v9, ref v13); Butterfly(ref v10, ref v14); Butterfly(ref v11, ref v15);
        Butterfly(ref v0, ref v2); Butterfly(ref v1, ref v3); Butterfly(ref v4, ref v6); Butterfly(ref v5, ref v7);
        Butterfly(ref v8, ref v10); Butterfly(ref v9, ref v11); Butterfly(ref v12, ref v14); Butterfly(ref v13, ref v15);
        Butterfly(ref v0, ref v1); Butterfly(ref v2, ref v3); Butterfly(ref v4, ref v5); Butterfly(ref v6, ref v7);
        Butterfly(ref v8, ref v9); Butterfly(ref v10, ref v11); Butterfly(ref v12, ref v13); Butterfly(ref v14, ref v15);
        var d = p + parity * 4;
        AdvSimd.Store(d, v0);
        AdvSimd.Store(d + 8, v1);
        AdvSimd.Store(d + 16, v2);
        AdvSimd.Store(d + 24, v3);
        AdvSimd.Store(d + 32, v4);
        AdvSimd.Store(d + 40, v5);
        AdvSimd.Store(d + 48, v6);
        AdvSimd.Store(d + 56, v7);
        AdvSimd.Store(d + 64, v8);
        AdvSimd.Store(d + 72, v9);
        AdvSimd.Store(d + 80, v10);
        AdvSimd.Store(d + 88, v11);
        AdvSimd.Store(d + 96, v12);
        AdvSimd.Store(d + 104, v13);
        AdvSimd.Store(d + 112, v14);
        AdvSimd.Store(d + 120, v15);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void Butterfly(ref Vector128<float> a, ref Vector128<float> b)
    {
        var sum = AdvSimd.Add(a, b);
        b = AdvSimd.Subtract(a, b);
        a = sum;
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
}
#endif
