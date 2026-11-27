#if NEEDLE_SIMD
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Needle;

// Int8 dot product strategy for the NEON codebook GEMV. The choice is a type parameter, so each
// instantiation's inner loops carry no per-call feature test.
internal interface INeonDot
{
    static abstract Vector128<int> Dot(Vector128<int> sum, Vector128<sbyte> weights, Vector128<sbyte> values);
}

internal readonly struct SdotNeonDot : INeonDot
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<int> Dot(Vector128<int> sum, Vector128<sbyte> weights, Vector128<sbyte> values) =>
        Dp.DotProduct(sum, weights, values);
}

internal readonly struct WideningNeonDot : INeonDot
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<int> Dot(Vector128<int> sum, Vector128<sbyte> weights, Vector128<sbyte> values)
    {
        var weightLow = AdvSimd.SignExtendWideningLower(weights.GetLower());
        var weightHigh = AdvSimd.SignExtendWideningUpper(weights);
        var valueLow = AdvSimd.SignExtendWideningLower(values.GetLower());
        var valueHigh = AdvSimd.SignExtendWideningUpper(values);
        sum = AdvSimd.MultiplyWideningLowerAndAdd(sum, valueLow.GetLower(), weightLow.GetLower());
        sum = AdvSimd.MultiplyWideningUpperAndAdd(sum, valueLow, weightLow);
        sum = AdvSimd.MultiplyWideningLowerAndAdd(sum, valueHigh.GetLower(), weightHigh.GetLower());
        return AdvSimd.MultiplyWideningUpperAndAdd(sum, valueHigh, weightHigh);
    }
}

// Codebook width as a type parameter, so the 2-bit and 4-bit loops compile without a per-group test.
internal interface INeonBits
{
    static abstract bool Two { get; }
}

internal readonly struct NeonTwoBits : INeonBits
{
    public static bool Two => true;
}

internal readonly struct NeonFourBits : INeonBits
{
    public static bool Two => false;
}

// Codebook weights as NEON table lookups. A 2-bit row byte holds four plane indices; its low nibble
// (planes 0 and 1) and high nibble (planes 2 and 3) each index a pair table, so one TBL yields
// sixteen weights. A 4-bit row byte holds two plane indices, each one TBL into the 16-entry codebook.
// The per-group integer dot products are exact (|sum| < 2^24), so their reduction order is free and
// every result is bit-identical to the original per-row kernel.
[SkipLocalsInit]
internal static unsafe class NeonGemv
{
    static readonly bool UseDotProduct = Dp.IsSupported && Environment.GetEnvironmentVariable("NEEDLE_NODP") == null;

    public static void MatMulRows(in Cq w, in Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
    {
        if (UseDotProduct)
        {
            if (w.Bits == 2) MatMulRows<SdotNeonDot, NeonTwoBits>(w, a, tokens, y, ldy, rowStart, rowEnd);
            else MatMulRows<SdotNeonDot, NeonFourBits>(w, a, tokens, y, ldy, rowStart, rowEnd);
        }
        else if (w.Bits == 2) MatMulRows<WideningNeonDot, NeonTwoBits>(w, a, tokens, y, ldy, rowStart, rowEnd);
        else MatMulRows<WideningNeonDot, NeonFourBits>(w, a, tokens, y, ldy, rowStart, rowEnd);
    }

    struct Tables
    {
        public Vector128<byte> First, Second, Nibble;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Tables MakeTables(in Cq w)
    {
        var book = AdvSimd.LoadVector128(w.Book->Bytes);
        Tables t;
        t.Nibble = Vector128.Create((byte)0x0F);
        if (w.Bits == 2)
        {
            t.First = AdvSimd.Arm64.VectorTableLookup(book, Vector128.Create((byte)0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3));
            t.Second = AdvSimd.Arm64.VectorTableLookup(book, Vector128.Create((byte)0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3));
        }
        else
        {
            t.First = book;
            t.Second = book;
        }
        return t;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<sbyte> Look(Vector128<byte> table, Vector128<byte> index) =>
        AdvSimd.Arm64.VectorTableLookup(table, index).AsSByte();

    // One row of one 128-column group against activation registers x0..x7 (x_k = values + 16k).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<int> Row2<TDot>(byte* packed, in Tables t,
        Vector128<sbyte> x0, Vector128<sbyte> x1, Vector128<sbyte> x2, Vector128<sbyte> x3,
        Vector128<sbyte> x4, Vector128<sbyte> x5, Vector128<sbyte> x6, Vector128<sbyte> x7)
        where TDot : struct, INeonDot
    {
        var s0 = AdvSimd.LoadVector128(packed);
        var s1 = AdvSimd.LoadVector128(packed + 16);
        var lo0 = AdvSimd.And(s0, t.Nibble);
        var hi0 = AdvSimd.ShiftRightLogical(s0, 4);
        var lo1 = AdvSimd.And(s1, t.Nibble);
        var hi1 = AdvSimd.ShiftRightLogical(s1, 4);
        var sum = TDot.Dot(Vector128<int>.Zero, Look(t.First, lo0), x0);
        var other = TDot.Dot(Vector128<int>.Zero, Look(t.First, lo1), x1);
        sum = TDot.Dot(sum, Look(t.Second, lo0), x2);
        other = TDot.Dot(other, Look(t.Second, lo1), x3);
        sum = TDot.Dot(sum, Look(t.First, hi0), x4);
        other = TDot.Dot(other, Look(t.First, hi1), x5);
        sum = TDot.Dot(sum, Look(t.Second, hi0), x6);
        other = TDot.Dot(other, Look(t.Second, hi1), x7);
        return AdvSimd.Add(sum, other);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<int> Row4<TDot>(byte* packed, in Tables t,
        Vector128<sbyte> x0, Vector128<sbyte> x1, Vector128<sbyte> x2, Vector128<sbyte> x3,
        Vector128<sbyte> x4, Vector128<sbyte> x5, Vector128<sbyte> x6, Vector128<sbyte> x7)
        where TDot : struct, INeonDot
    {
        var s0 = AdvSimd.LoadVector128(packed);
        var s1 = AdvSimd.LoadVector128(packed + 16);
        var s2 = AdvSimd.LoadVector128(packed + 32);
        var s3 = AdvSimd.LoadVector128(packed + 48);
        var sum = TDot.Dot(Vector128<int>.Zero, Look(t.First, AdvSimd.And(s0, t.Nibble)), x0);
        var other = TDot.Dot(Vector128<int>.Zero, Look(t.First, AdvSimd.And(s1, t.Nibble)), x1);
        sum = TDot.Dot(sum, Look(t.First, AdvSimd.ShiftRightLogical(s0, 4)), x2);
        other = TDot.Dot(other, Look(t.First, AdvSimd.ShiftRightLogical(s1, 4)), x3);
        sum = TDot.Dot(sum, Look(t.First, AdvSimd.And(s2, t.Nibble)), x4);
        other = TDot.Dot(other, Look(t.First, AdvSimd.And(s3, t.Nibble)), x5);
        sum = TDot.Dot(sum, Look(t.First, AdvSimd.ShiftRightLogical(s2, 4)), x6);
        other = TDot.Dot(other, Look(t.First, AdvSimd.ShiftRightLogical(s3, 4)), x7);
        return AdvSimd.Add(sum, other);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<int> Row<TDot, TBits>(byte* packed, in Tables t,
        Vector128<sbyte> x0, Vector128<sbyte> x1, Vector128<sbyte> x2, Vector128<sbyte> x3,
        Vector128<sbyte> x4, Vector128<sbyte> x5, Vector128<sbyte> x6, Vector128<sbyte> x7)
        where TDot : struct, INeonDot
        where TBits : struct, INeonBits =>
        TBits.Two ? Row2<TDot>(packed, t, x0, x1, x2, x3, x4, x5, x6, x7) : Row4<TDot>(packed, t, x0, x1, x2, x3, x4, x5, x6, x7);

    // Lane i of the result is the total of lane-vector si: four exact integer sums, one per input.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<float> Totals(Vector128<int> s0, Vector128<int> s1, Vector128<int> s2, Vector128<int> s3) =>
        AdvSimd.ConvertToSingle(AdvSimd.Arm64.AddPairwise(AdvSimd.Arm64.AddPairwise(s0, s1), AdvSimd.Arm64.AddPairwise(s2, s3)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float Total(Vector128<int> s) => AdvSimd.Arm64.AddAcross(s).ToScalar();

    // Transposes four rows of per-group factors (row r at source + r * stride) into group-major quads.
    static void Interleave4(float* source, int stride, float* destination)
    {
        for (var g = 0; g < stride; g += 4)
        {
            var r0 = AdvSimd.LoadVector128(source + g);
            var r1 = AdvSimd.LoadVector128(source + stride + g);
            var r2 = AdvSimd.LoadVector128(source + 2 * stride + g);
            var r3 = AdvSimd.LoadVector128(source + 3 * stride + g);
            var t0 = AdvSimd.Arm64.ZipLow(r0, r1);
            var t1 = AdvSimd.Arm64.ZipHigh(r0, r1);
            var t2 = AdvSimd.Arm64.ZipLow(r2, r3);
            var t3 = AdvSimd.Arm64.ZipHigh(r2, r3);
            AdvSimd.Store(destination + g * 4, AdvSimd.Arm64.ZipLow(t0.AsDouble(), t2.AsDouble()).AsSingle());
            AdvSimd.Store(destination + g * 4 + 4, AdvSimd.Arm64.ZipHigh(t0.AsDouble(), t2.AsDouble()).AsSingle());
            AdvSimd.Store(destination + g * 4 + 8, AdvSimd.Arm64.ZipLow(t1.AsDouble(), t3.AsDouble()).AsSingle());
            AdvSimd.Store(destination + g * 4 + 12, AdvSimd.Arm64.ZipHigh(t1.AsDouble(), t3.AsDouble()).AsSingle());
        }
    }

    // Neon.LoadFactors then Neon.ScaleFactors for four consecutive rows against one token, with the
    // scaled factors written group-major (quads[g * 4 + r]); returns the four biases. Same operations
    // per element as the two-pass version: the vector half conversion for the groups LoadFactors
    // converts four at a time, Scalar.HalfToSingle for its tail, zero past the last group.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<float> Factors4(ushort* norms, int groups, int stride, float byteScale, float* scales, float* offsets, float* quads)
    {
        var multiplier = Vector128.Create(byteScale);
        var vectorGroups = groups & ~3;
        var b0 = Vector128<float>.Zero;
        var b1 = b0;
        var b2 = b0;
        var b3 = b0;
        for (var v = 0; v < stride; v += 4)
        {
            var f0 = RowFactor(norms, v, groups, vectorGroups, multiplier, byteScale);
            var f1 = RowFactor(norms + groups, v, groups, vectorGroups, multiplier, byteScale);
            var f2 = RowFactor(norms + 2 * groups, v, groups, vectorGroups, multiplier, byteScale);
            var f3 = RowFactor(norms + 3 * groups, v, groups, vectorGroups, multiplier, byteScale);
            var left = groups - v;
            var sc = left >= 4 ? AdvSimd.LoadVector128(scales + v) : left > 0 ? Neon.Partial(scales + v, left) : Vector128<float>.Zero;
            var of = left >= 4 ? AdvSimd.LoadVector128(offsets + v) : left > 0 ? Neon.Partial(offsets + v, left) : Vector128<float>.Zero;
            f0 = AdvSimd.Multiply(f0, sc);
            f1 = AdvSimd.Multiply(f1, sc);
            f2 = AdvSimd.Multiply(f2, sc);
            f3 = AdvSimd.Multiply(f3, sc);
            b0 = AdvSimd.FusedMultiplyAdd(b0, f0, of);
            b1 = AdvSimd.FusedMultiplyAdd(b1, f1, of);
            b2 = AdvSimd.FusedMultiplyAdd(b2, f2, of);
            b3 = AdvSimd.FusedMultiplyAdd(b3, f3, of);
            var t0 = AdvSimd.Arm64.ZipLow(f0, f1);
            var t1 = AdvSimd.Arm64.ZipHigh(f0, f1);
            var t2 = AdvSimd.Arm64.ZipLow(f2, f3);
            var t3 = AdvSimd.Arm64.ZipHigh(f2, f3);
            var d = quads + v * 4;
            AdvSimd.Store(d, AdvSimd.Arm64.ZipLow(t0.AsDouble(), t2.AsDouble()).AsSingle());
            AdvSimd.Store(d + 4, AdvSimd.Arm64.ZipHigh(t0.AsDouble(), t2.AsDouble()).AsSingle());
            AdvSimd.Store(d + 8, AdvSimd.Arm64.ZipLow(t1.AsDouble(), t3.AsDouble()).AsSingle());
            AdvSimd.Store(d + 12, AdvSimd.Arm64.ZipHigh(t1.AsDouble(), t3.AsDouble()).AsSingle());
        }
        return Vector128.Create(Neon.HorizontalSum(b0), Neon.HorizontalSum(b1), Neon.HorizontalSum(b2), Neon.HorizontalSum(b3));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<float> RowFactor(ushort* norms, int v, int groups, int vectorGroups, Vector128<float> multiplier, float byteScale)
    {
        if (v < vectorGroups)
        {
            var wide = AdvSimd.ZeroExtendWideningLower(AdvSimd.LoadVector64(norms + v));
            var magnitude = AdvSimd.And(wide, Vector128.Create(0x7FFFu));
            var sign = AdvSimd.ShiftLeftLogical(AdvSimd.And(wide, Vector128.Create(0x8000u)), 16);
            var rebased = AdvSimd.Multiply(AdvSimd.ShiftLeftLogical(magnitude, 13).AsSingle(), Vector128.Create(5.192296858534828e33f));
            var single = AdvSimd.Or(rebased.AsUInt32(), sign).AsSingle();
            return AdvSimd.Multiply(single, multiplier);
        }
        if (v < groups && groups >= 4) return AdvSimd.Multiply(Neon.HalfTail(norms, groups), multiplier);
        return Vector128.Create(
            v < groups ? Scalar.HalfToSingle(norms[v]) * byteScale : 0f,
            v + 1 < groups ? Scalar.HalfToSingle(norms[v + 1]) * byteScale : 0f,
            v + 2 < groups ? Scalar.HalfToSingle(norms[v + 2]) * byteScale : 0f,
            v + 3 < groups ? Scalar.HalfToSingle(norms[v + 3]) * byteScale : 0f);
    }

    static void MatMulRows<TDot, TBits>(in Cq w, in Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
        where TDot : struct, INeonDot
        where TBits : struct, INeonBits
    {
        var stride = a.Stride;
        var rowFactors = stackalloc float[stride * 4];
        var scaled = stackalloc float[stride * 8];
        var quads = stackalloc float[stride * 8];
        new Span<float>(rowFactors, stride * 4).Clear();
        var t = MakeTables(w);
        var full = tokens & ~3;
        if (full > 0)
        {
            var n = rowStart;
            for (; n + 2 <= rowEnd; n += 2)
                for (var token = 0; token < full; token += 4)
                    Pair<TDot, TBits>(w, n, a, token, rowFactors, scaled, quads, y + (long)token * ldy + n, ldy, t);
            for (; n < rowEnd; n++)
                for (var token = 0; token < full; token += 4)
                    Tile<TDot, TBits>(w, n, a, token, rowFactors, scaled, quads, y + (long)token * ldy + n, ldy, t);
        }
        for (var token = full; token < tokens; token++)
        {
            var yToken = y + (long)token * ldy;
            var n = rowStart;
            for (; n + 4 <= rowEnd; n += 4)
                Block<TDot, TBits>(w, n, a, token, rowFactors, scaled, quads, yToken + n, t);
            for (; n < rowEnd; n++)
                yToken[n] = Single<TDot, TBits>(w, n, a, token, rowFactors, scaled, t);
        }
    }

    // Four rows by one token: the decode GEMV.
    static void Block<TDot, TBits>(in Cq w, int row, in Activations a, int token, float* rowFactors, float* scaled, float* quads, float* y, in Tables t)
        where TDot : struct, INeonDot
        where TBits : struct, INeonBits
    {
        var groups = w.Groups;
        var stride = a.Stride;
        var byteScale = w.Book->ByteScale;
        var norms = w.Norms + (long)row * groups;
        var scales = a.Scales + token * stride;
        var offsets = a.Offsets + token * stride;
        var bias = Factors4(norms, groups, stride, byteScale, scales, offsets, quads);
        var rowBytes = w.RowBytes;
        var groupBytes = rowBytes / groups;
        var p0 = w.Indices + (long)row * rowBytes;
        var p1 = p0 + rowBytes;
        var p2 = p1 + rowBytes;
        var p3 = p2 + rowBytes;
        var values = a.Values + (long)token * a.Cols;
        var accumulator = Vector128<float>.Zero;
        for (var g = 0; g < groups; g++)
        {
            var x = values + g * Cq.GroupSize;
            var x0 = AdvSimd.LoadVector128(x);
            var x1 = AdvSimd.LoadVector128(x + 16);
            var x2 = AdvSimd.LoadVector128(x + 32);
            var x3 = AdvSimd.LoadVector128(x + 48);
            var x4 = AdvSimd.LoadVector128(x + 64);
            var x5 = AdvSimd.LoadVector128(x + 80);
            var x6 = AdvSimd.LoadVector128(x + 96);
            var x7 = AdvSimd.LoadVector128(x + 112);
            var offset = g * groupBytes;
            var s0 = Row<TDot, TBits>(p0 + offset, t, x0, x1, x2, x3, x4, x5, x6, x7);
            var s1 = Row<TDot, TBits>(p1 + offset, t, x0, x1, x2, x3, x4, x5, x6, x7);
            var s2 = Row<TDot, TBits>(p2 + offset, t, x0, x1, x2, x3, x4, x5, x6, x7);
            var s3 = Row<TDot, TBits>(p3 + offset, t, x0, x1, x2, x3, x4, x5, x6, x7);
            accumulator = AdvSimd.Add(accumulator, AdvSimd.Multiply(Totals(s0, s1, s2, s3), AdvSimd.LoadVector128(quads + g * 4)));
        }
        AdvSimd.Store(y, AdvSimd.Subtract(accumulator, bias));
    }

    static float Single<TDot, TBits>(in Cq w, int row, in Activations a, int token, float* rowFactors, float* scaled, in Tables t)
        where TDot : struct, INeonDot
        where TBits : struct, INeonBits
    {
        var groups = w.Groups;
        var stride = a.Stride;
        Neon.LoadFactors(w.Norms + (long)row * groups, groups, w.Book->ByteScale, rowFactors);
        var bias = Neon.ScaleFactors(rowFactors, scaled, a.Scales + token * stride, a.Offsets + token * stride, a.Groups);
        var rowBytes = w.RowBytes;
        var groupBytes = rowBytes / groups;
        var packed = w.Indices + (long)row * rowBytes;
        var values = a.Values + (long)token * a.Cols;
        var accumulator = 0f;
        for (var g = 0; g < groups; g++)
        {
            var x = values + g * Cq.GroupSize;
            var sum = Row<TDot, TBits>(packed + g * groupBytes, t,
                AdvSimd.LoadVector128(x), AdvSimd.LoadVector128(x + 16), AdvSimd.LoadVector128(x + 32), AdvSimd.LoadVector128(x + 48),
                AdvSimd.LoadVector128(x + 64), AdvSimd.LoadVector128(x + 80), AdvSimd.LoadVector128(x + 96), AdvSimd.LoadVector128(x + 112));
            accumulator += (float)Total(sum) * scaled[g];
        }
        return accumulator - bias;
    }

    // Two rows by four tokens: the prefill tile. Each decoded weight vector feeds four tokens.
    static void Pair<TDot, TBits>(in Cq w, int row, in Activations a, int token, float* rowFactors, float* scaled, float* quads, float* y, int ldy, in Tables t)
        where TDot : struct, INeonDot
        where TBits : struct, INeonBits
    {
        var groups = w.Groups;
        var stride = a.Stride;
        var cols = a.Cols;
        var byteScale = w.Book->ByteScale;
        var norms = w.Norms + (long)row * groups;
        Neon.LoadFactors(norms, groups, byteScale, rowFactors);
        Neon.LoadFactors(norms + groups, groups, byteScale, rowFactors + stride);
        var scales = a.Scales + token * stride;
        var offsets = a.Offsets + token * stride;
        // scaled row-major: [r * 4 + k] * stride for row r and token k
        var bias0 = Vector128.Create(
            Neon.ScaleFactors(rowFactors, scaled, scales, offsets, a.Groups),
            Neon.ScaleFactors(rowFactors, scaled + stride, scales + stride, offsets + stride, a.Groups),
            Neon.ScaleFactors(rowFactors, scaled + 2 * stride, scales + 2 * stride, offsets + 2 * stride, a.Groups),
            Neon.ScaleFactors(rowFactors, scaled + 3 * stride, scales + 3 * stride, offsets + 3 * stride, a.Groups));
        var bias1 = Vector128.Create(
            Neon.ScaleFactors(rowFactors + stride, scaled + 4 * stride, scales, offsets, a.Groups),
            Neon.ScaleFactors(rowFactors + stride, scaled + 5 * stride, scales + stride, offsets + stride, a.Groups),
            Neon.ScaleFactors(rowFactors + stride, scaled + 6 * stride, scales + 2 * stride, offsets + 2 * stride, a.Groups),
            Neon.ScaleFactors(rowFactors + stride, scaled + 7 * stride, scales + 3 * stride, offsets + 3 * stride, a.Groups));
        Interleave4(scaled, stride, quads);
        Interleave4(scaled + 4 * stride, stride, quads + 4 * stride);
        var rowBytes = w.RowBytes;
        var groupBytes = rowBytes / groups;
        var r0 = w.Indices + (long)row * rowBytes;
        var r1 = r0 + rowBytes;
        var v0 = a.Values + (long)token * cols;
        var v1 = v0 + cols;
        var v2 = v1 + cols;
        var v3 = v2 + cols;
        var two = TBits.Two;
        var acc0 = Vector128<float>.Zero;
        var acc1 = Vector128<float>.Zero;
        for (var g = 0; g < groups; g++)
        {
            var s00 = Vector128<int>.Zero; var s01 = s00; var s02 = s00; var s03 = s00;
            var s10 = s00; var s11 = s00; var s12 = s00; var s13 = s00;
            var offset = g * groupBytes;
            var col = g * Cq.GroupSize;
            if (two)
            {
                for (var half = 0; half < 2; half++)
                {
                    var a0 = AdvSimd.LoadVector128(r0 + offset + half * 16);
                    var a1 = AdvSimd.LoadVector128(r1 + offset + half * 16);
                    var lo0 = AdvSimd.And(a0, t.Nibble);
                    var lo1 = AdvSimd.And(a1, t.Nibble);
                    var hi0 = AdvSimd.ShiftRightLogical(a0, 4);
                    var hi1 = AdvSimd.ShiftRightLogical(a1, 4);
                    var c = col + half * 16;
                    Quad<TDot>(Look(t.First, lo0), Look(t.First, lo1), v0 + c, v1 + c, v2 + c, v3 + c, ref s00, ref s01, ref s02, ref s03, ref s10, ref s11, ref s12, ref s13);
                    Quad<TDot>(Look(t.Second, lo0), Look(t.Second, lo1), v0 + c + 32, v1 + c + 32, v2 + c + 32, v3 + c + 32, ref s00, ref s01, ref s02, ref s03, ref s10, ref s11, ref s12, ref s13);
                    Quad<TDot>(Look(t.First, hi0), Look(t.First, hi1), v0 + c + 64, v1 + c + 64, v2 + c + 64, v3 + c + 64, ref s00, ref s01, ref s02, ref s03, ref s10, ref s11, ref s12, ref s13);
                    Quad<TDot>(Look(t.Second, hi0), Look(t.Second, hi1), v0 + c + 96, v1 + c + 96, v2 + c + 96, v3 + c + 96, ref s00, ref s01, ref s02, ref s03, ref s10, ref s11, ref s12, ref s13);
                }
            }
            else
            {
                for (var q = 0; q < 4; q++)
                {
                    var a0 = AdvSimd.LoadVector128(r0 + offset + q * 16);
                    var a1 = AdvSimd.LoadVector128(r1 + offset + q * 16);
                    var c = col + (q >> 1) * 64 + (q & 1) * 16;
                    Quad<TDot>(Look(t.First, AdvSimd.And(a0, t.Nibble)), Look(t.First, AdvSimd.And(a1, t.Nibble)), v0 + c, v1 + c, v2 + c, v3 + c, ref s00, ref s01, ref s02, ref s03, ref s10, ref s11, ref s12, ref s13);
                    Quad<TDot>(Look(t.First, AdvSimd.ShiftRightLogical(a0, 4)), Look(t.First, AdvSimd.ShiftRightLogical(a1, 4)), v0 + c + 32, v1 + c + 32, v2 + c + 32, v3 + c + 32, ref s00, ref s01, ref s02, ref s03, ref s10, ref s11, ref s12, ref s13);
                }
            }
            acc0 = AdvSimd.Add(acc0, AdvSimd.Multiply(Totals(s00, s01, s02, s03), AdvSimd.LoadVector128(quads + g * 4)));
            acc1 = AdvSimd.Add(acc1, AdvSimd.Multiply(Totals(s10, s11, s12, s13), AdvSimd.LoadVector128(quads + 4 * stride + g * 4)));
        }
        var out0 = AdvSimd.Subtract(acc0, bias0);
        var out1 = AdvSimd.Subtract(acc1, bias1);
        y[0] = out0.GetElement(0);
        y[ldy] = out0.GetElement(1);
        y[2 * ldy] = out0.GetElement(2);
        y[3 * ldy] = out0.GetElement(3);
        y[1] = out1.GetElement(0);
        y[ldy + 1] = out1.GetElement(1);
        y[2 * ldy + 1] = out1.GetElement(2);
        y[3 * ldy + 1] = out1.GetElement(3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void Quad<TDot>(Vector128<sbyte> w0, Vector128<sbyte> w1, sbyte* x0, sbyte* x1, sbyte* x2, sbyte* x3,
        ref Vector128<int> s00, ref Vector128<int> s01, ref Vector128<int> s02, ref Vector128<int> s03,
        ref Vector128<int> s10, ref Vector128<int> s11, ref Vector128<int> s12, ref Vector128<int> s13)
        where TDot : struct, INeonDot
    {
        var a0 = AdvSimd.LoadVector128(x0);
        var a1 = AdvSimd.LoadVector128(x1);
        var a2 = AdvSimd.LoadVector128(x2);
        var a3 = AdvSimd.LoadVector128(x3);
        s00 = TDot.Dot(s00, w0, a0);
        s01 = TDot.Dot(s01, w0, a1);
        s02 = TDot.Dot(s02, w0, a2);
        s03 = TDot.Dot(s03, w0, a3);
        s10 = TDot.Dot(s10, w1, a0);
        s11 = TDot.Dot(s11, w1, a1);
        s12 = TDot.Dot(s12, w1, a2);
        s13 = TDot.Dot(s13, w1, a3);
    }

    // One row by four tokens: the odd row left over by the prefill pairs.
    static void Tile<TDot, TBits>(in Cq w, int row, in Activations a, int token, float* rowFactors, float* scaled, float* quads, float* y, int ldy, in Tables t)
        where TDot : struct, INeonDot
        where TBits : struct, INeonBits
    {
        var groups = w.Groups;
        var stride = a.Stride;
        var cols = a.Cols;
        Neon.LoadFactors(w.Norms + (long)row * groups, groups, w.Book->ByteScale, rowFactors);
        var scales = a.Scales + token * stride;
        var offsets = a.Offsets + token * stride;
        var bias = Vector128.Create(
            Neon.ScaleFactors(rowFactors, scaled, scales, offsets, a.Groups),
            Neon.ScaleFactors(rowFactors, scaled + stride, scales + stride, offsets + stride, a.Groups),
            Neon.ScaleFactors(rowFactors, scaled + 2 * stride, scales + 2 * stride, offsets + 2 * stride, a.Groups),
            Neon.ScaleFactors(rowFactors, scaled + 3 * stride, scales + 3 * stride, offsets + 3 * stride, a.Groups));
        Interleave4(scaled, stride, quads);
        var rowBytes = w.RowBytes;
        var groupBytes = rowBytes / groups;
        var packed = w.Indices + (long)row * rowBytes;
        var v0 = a.Values + (long)token * cols;
        var v1 = v0 + cols;
        var v2 = v1 + cols;
        var v3 = v2 + cols;
        var two = TBits.Two;
        var acc = Vector128<float>.Zero;
        for (var g = 0; g < groups; g++)
        {
            var s0 = Vector128<int>.Zero; var s1 = s0; var s2 = s0; var s3 = s0;
            var offset = g * groupBytes;
            var col = g * Cq.GroupSize;
            if (two)
            {
                for (var half = 0; half < 2; half++)
                {
                    var source = AdvSimd.LoadVector128(packed + offset + half * 16);
                    var lo = AdvSimd.And(source, t.Nibble);
                    var hi = AdvSimd.ShiftRightLogical(source, 4);
                    var c = col + half * 16;
                    Four<TDot>(Look(t.First, lo), v0 + c, v1 + c, v2 + c, v3 + c, ref s0, ref s1, ref s2, ref s3);
                    Four<TDot>(Look(t.Second, lo), v0 + c + 32, v1 + c + 32, v2 + c + 32, v3 + c + 32, ref s0, ref s1, ref s2, ref s3);
                    Four<TDot>(Look(t.First, hi), v0 + c + 64, v1 + c + 64, v2 + c + 64, v3 + c + 64, ref s0, ref s1, ref s2, ref s3);
                    Four<TDot>(Look(t.Second, hi), v0 + c + 96, v1 + c + 96, v2 + c + 96, v3 + c + 96, ref s0, ref s1, ref s2, ref s3);
                }
            }
            else
            {
                for (var q = 0; q < 4; q++)
                {
                    var source = AdvSimd.LoadVector128(packed + offset + q * 16);
                    var c = col + (q >> 1) * 64 + (q & 1) * 16;
                    Four<TDot>(Look(t.First, AdvSimd.And(source, t.Nibble)), v0 + c, v1 + c, v2 + c, v3 + c, ref s0, ref s1, ref s2, ref s3);
                    Four<TDot>(Look(t.First, AdvSimd.ShiftRightLogical(source, 4)), v0 + c + 32, v1 + c + 32, v2 + c + 32, v3 + c + 32, ref s0, ref s1, ref s2, ref s3);
                }
            }
            acc = AdvSimd.Add(acc, AdvSimd.Multiply(Totals(s0, s1, s2, s3), AdvSimd.LoadVector128(quads + g * 4)));
        }
        var result = AdvSimd.Subtract(acc, bias);
        y[0] = result.GetElement(0);
        y[ldy] = result.GetElement(1);
        y[2 * ldy] = result.GetElement(2);
        y[3 * ldy] = result.GetElement(3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void Four<TDot>(Vector128<sbyte> weights, sbyte* x0, sbyte* x1, sbyte* x2, sbyte* x3,
        ref Vector128<int> s0, ref Vector128<int> s1, ref Vector128<int> s2, ref Vector128<int> s3)
        where TDot : struct, INeonDot
    {
        s0 = TDot.Dot(s0, weights, AdvSimd.LoadVector128(x0));
        s1 = TDot.Dot(s1, weights, AdvSimd.LoadVector128(x1));
        s2 = TDot.Dot(s2, weights, AdvSimd.LoadVector128(x2));
        s3 = TDot.Dot(s3, weights, AdvSimd.LoadVector128(x3));
    }
}
#endif
