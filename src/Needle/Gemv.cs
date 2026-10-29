#if NEEDLE_SIMD
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Needle;

internal readonly struct PlaneTables
{
    public readonly Vector256<byte> First, Second, Mask;

    public PlaneTables(Vector256<byte> first, Vector256<byte> second, Vector256<byte> mask)
    {
        First = first;
        Second = second;
        Mask = mask;
    }
}

internal interface IPlaneLayout
{
    static abstract int GroupBytes { get; }
    static abstract PlaneTables Tables(Vector256<byte> codebook);
    static abstract unsafe void Expand(byte* packed, in PlaneTables tables, out Vector256<byte> p0, out Vector256<byte> p1, out Vector256<byte> p2, out Vector256<byte> p3);
}

internal readonly struct TwoBitPlanes : IPlaneLayout
{
    public static int GroupBytes => 32;

    public static PlaneTables Tables(Vector256<byte> codebook) => new(
        Avx2.Shuffle(codebook, Vector256.Create((byte)0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3)),
        Avx2.Shuffle(codebook, Vector256.Create((byte)0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)),
        Vector256.Create((byte)0x0F));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Expand(byte* packed, in PlaneTables tables, out Vector256<byte> p0, out Vector256<byte> p1, out Vector256<byte> p2, out Vector256<byte> p3)
    {
        var v = Avx.LoadVector256(packed).AsUInt16();
        var lower = Avx2.And(v.AsByte(), tables.Mask);
        var upper = Avx2.And(Avx2.ShiftRightLogical(v, 4).AsByte(), tables.Mask);
        p0 = Avx2.Shuffle(tables.First, lower);
        p1 = Avx2.Shuffle(tables.Second, lower);
        p2 = Avx2.Shuffle(tables.First, upper);
        p3 = Avx2.Shuffle(tables.Second, upper);
    }
}

internal readonly struct PermuteTwoBitPlanes : IPlaneLayout
{
    public static int GroupBytes => TwoBitPlanes.GroupBytes;

    public static PlaneTables Tables(Vector256<byte> codebook) => TwoBitPlanes.Tables(codebook);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Expand(byte* packed, in PlaneTables tables, out Vector256<byte> p0, out Vector256<byte> p1, out Vector256<byte> p2, out Vector256<byte> p3)
    {
        var v = Avx.LoadVector256(packed).AsUInt16();
        var lower = v.AsByte();
        var upper = Avx2.ShiftRightLogical(v, 4).AsByte();
        p0 = Avx512Vbmi.VL.PermuteVar32x8(tables.First, lower);
        p1 = Avx512Vbmi.VL.PermuteVar32x8(tables.Second, lower);
        p2 = Avx512Vbmi.VL.PermuteVar32x8(tables.First, upper);
        p3 = Avx512Vbmi.VL.PermuteVar32x8(tables.Second, upper);
    }
}

internal readonly struct FourBitPlanes : IPlaneLayout
{
    public static int GroupBytes => 64;

    public static PlaneTables Tables(Vector256<byte> codebook) => new(codebook, codebook, Vector256.Create((byte)0x0F));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Expand(byte* packed, in PlaneTables tables, out Vector256<byte> p0, out Vector256<byte> p1, out Vector256<byte> p2, out Vector256<byte> p3)
    {
        var lo = Avx.LoadVector256(packed).AsUInt16();
        var hi = Avx.LoadVector256(packed + 32).AsUInt16();
        p0 = Avx2.Shuffle(tables.First, Avx2.And(lo.AsByte(), tables.Mask));
        p1 = Avx2.Shuffle(tables.First, Avx2.And(Avx2.ShiftRightLogical(lo, 4).AsByte(), tables.Mask));
        p2 = Avx2.Shuffle(tables.First, Avx2.And(hi.AsByte(), tables.Mask));
        p3 = Avx2.Shuffle(tables.First, Avx2.And(Avx2.ShiftRightLogical(hi, 4).AsByte(), tables.Mask));
    }
}

internal readonly struct PermuteFourBitPlanes : IPlaneLayout
{
    public static int GroupBytes => FourBitPlanes.GroupBytes;

    public static PlaneTables Tables(Vector256<byte> codebook) => FourBitPlanes.Tables(codebook);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Expand(byte* packed, in PlaneTables tables, out Vector256<byte> p0, out Vector256<byte> p1, out Vector256<byte> p2, out Vector256<byte> p3)
    {
        var lo = Avx.LoadVector256(packed).AsUInt16();
        var hi = Avx.LoadVector256(packed + 32).AsUInt16();
        p0 = Avx512Vbmi.VL.PermuteVar32x8(tables.First, lo.AsByte());
        p1 = Avx512Vbmi.VL.PermuteVar32x8(tables.First, Avx2.ShiftRightLogical(lo, 4).AsByte());
        p2 = Avx512Vbmi.VL.PermuteVar32x8(tables.First, hi.AsByte());
        p3 = Avx512Vbmi.VL.PermuteVar32x8(tables.First, Avx2.ShiftRightLogical(hi, 4).AsByte());
    }
}

internal static unsafe class Gemv
{
    const int RowBlock = Quant.RowBlock;
    const int TileTokens = 4;
    const int PrefetchBlocks = 4;
    const int CacheLine = 64;
    const int HalvesPerVector = 8;
    static readonly bool UsePermute = Simd.UseAvx && AvxVnni.IsSupported && Avx512Vbmi.VL.IsSupported && Environment.GetEnvironmentVariable("NEEDLE_NOAVX512") == null;
    const float HalfExponentRebias = 5.192296858534828e33f;

    public static void Rows(in Cq w, in Activations a, int token, float* y, int rowStart, int rowEnd)
    {
        if (UsePermute)
        {
            if (w.Bits == 2) Rows<PermuteTwoBitPlanes>(w, a, token, y, rowStart, rowEnd);
            else Rows<PermuteFourBitPlanes>(w, a, token, y, rowStart, rowEnd);
        }
        else if (w.Bits == 2) Rows<TwoBitPlanes>(w, a, token, y, rowStart, rowEnd);
        else Rows<FourBitPlanes>(w, a, token, y, rowStart, rowEnd);
    }

    static void Rows<TPlanes>(in Cq w, in Activations a, int token, float* y, int rowStart, int rowEnd) where TPlanes : struct, IPlaneLayout
    {
        var groups = w.Groups;
        var stride = a.Stride;
        var tables = TPlanes.Tables(Avx.LoadVector256(w.Book->Bytes));
        var weightedScales = stackalloc float[stride];
        var scaled = stackalloc float[RowBlock * stride];
        var activationScales = a.Scales + token * stride;
        var byteScale = Vector256.Create(w.Book->ByteScale * HalfExponentRebias);
        for (var v = 0; v < stride; v += HalvesPerVector)
            Avx.Store(weightedScales + v, Avx.Multiply(Avx.LoadVector256(activationScales + v), byteScale));
        var offsets = a.Offsets + token * stride;
        var values = a.Values + (long)token * a.Cols;
        var rowBytes = w.RowBytes;
        var paddedGroups = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
        var convertibleRows = (long)w.Rows * groups < paddedGroups ? 0 : (int)(((long)w.Rows * groups - paddedGroups) / groups) + 1;
        var n = rowStart;
        for (; n + RowBlock <= rowEnd && n + RowBlock <= convertibleRows; n += RowBlock)
            Block<TPlanes>(w.Indices + (long)n * rowBytes, rowBytes, w.Norms + (long)n * groups, groups, stride, tables, weightedScales, offsets, values, scaled, y + n);
        for (; n < rowEnd; n++)
            y[n] = Single<TPlanes>(w.Indices + (long)n * rowBytes, w.Norms + (long)n * groups, groups, stride, n < convertibleRows, tables, weightedScales, offsets, values, scaled);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> Corrections(ushort* norms, int stride, float* weightedScales, float* offsets, float* scaled)
    {
        var correction = Vector256<float>.Zero;
        for (var v = 0; v < stride; v += HalvesPerVector)
        {
            var factors = Avx.Multiply(UnbiasedHalfToSingle(Avx2.ConvertToVector256Int32(norms + v)), Avx.LoadVector256(weightedScales + v));
            Avx.Store(scaled + v, factors);
            correction = Fma.MultiplyAddNegated(factors, Avx.LoadVector256(offsets + v), correction);
        }
        return correction;
    }

    static Vector256<float> CorrectionsAtEdge(ushort* norms, int groups, int stride, float* weightedScales, float* offsets, float* scaled)
    {
        for (var i = 0; i < stride; i++) scaled[i] = i < groups ? BitConverter.Int32BitsToSingle(norms[i] << 13) : 0f;
        var correction = Vector256<float>.Zero;
        for (var v = 0; v < stride; v += HalvesPerVector)
        {
            var factors = Avx.Multiply(Avx.LoadVector256(scaled + v), Avx.LoadVector256(weightedScales + v));
            Avx.Store(scaled + v, factors);
            correction = Fma.MultiplyAddNegated(factors, Avx.LoadVector256(offsets + v), correction);
        }
        return correction;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> UnbiasedHalfToSingle(Vector256<int> half) => Avx2.ShiftLeftLogical(half, 13).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<int> Dot(Vector256<byte> p0, Vector256<byte> p1, Vector256<byte> p2, Vector256<byte> p3, Vector256<sbyte> x0, Vector256<sbyte> x1, Vector256<sbyte> x2, Vector256<sbyte> x3)
    {
        var sum = Quant.Accumulate(Vector256<int>.Zero, p0, x0);
        sum = Quant.Accumulate(sum, p1, x1);
        sum = Quant.Accumulate(sum, p2, x2);
        return Quant.Accumulate(sum, p3, x3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<int> Dot(Vector256<byte> p0, Vector256<byte> p1, Vector256<byte> p2, Vector256<byte> p3, sbyte* x) =>
        Dot(p0, p1, p2, p3, Avx.LoadVector256(x), Avx.LoadVector256(x + 32), Avx.LoadVector256(x + 64), Avx.LoadVector256(x + 96));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> Group<TPlanes>(byte* packed, in PlaneTables tables, Vector256<sbyte> x0, Vector256<sbyte> x1, Vector256<sbyte> x2, Vector256<sbyte> x3, float* scale, Vector256<float> accumulator) where TPlanes : struct, IPlaneLayout
    {
        TPlanes.Expand(packed, tables, out var p0, out var p1, out var p2, out var p3);
        return Fma.MultiplyAdd(Avx.ConvertToVector256Single(Dot(p0, p1, p2, p3, x0, x1, x2, x3)), Avx.BroadcastScalarToVector256(scale), accumulator);
    }

    static void Block<TPlanes>(byte* packed, int rowBytes, ushort* norms, int groups, int stride, in PlaneTables tables, float* weightedScales, float* offsets, sbyte* values, float* scaled, float* y) where TPlanes : struct, IPlaneLayout
    {
        var start0 = Corrections(norms, stride, weightedScales, offsets, scaled);
        var start1 = Corrections(norms + groups, stride, weightedScales, offsets, scaled + stride);
        var start2 = Corrections(norms + 2 * groups, stride, weightedScales, offsets, scaled + 2 * stride);
        var start3 = Corrections(norms + 3 * groups, stride, weightedScales, offsets, scaled + 3 * stride);
        var row1 = packed + rowBytes;
        var row2 = packed + 2 * rowBytes;
        var row3 = packed + 3 * rowBytes;
        var accumulator0 = start0;
        var accumulator1 = start1;
        var accumulator2 = start2;
        var accumulator3 = start3;
        var groupBytes = TPlanes.GroupBytes;
        var prefetchDistance = (long)PrefetchBlocks * RowBlock * rowBytes;
        for (var g = 0; g < groups; g++)
        {
            if ((g * groupBytes & CacheLine - 1) == 0)
            {
                var ahead = g * groupBytes + prefetchDistance;
                Sse.Prefetch0(packed + ahead);
                Sse.Prefetch0(row1 + ahead);
                Sse.Prefetch0(row2 + ahead);
                Sse.Prefetch0(row3 + ahead);
            }
            var x = values + g * Cq.GroupSize;
            var x0 = Avx.LoadVector256(x);
            var x1 = Avx.LoadVector256(x + 32);
            var x2 = Avx.LoadVector256(x + 64);
            var x3 = Avx.LoadVector256(x + 96);
            var offset = g * groupBytes;
            accumulator0 = Group<TPlanes>(packed + offset, tables, x0, x1, x2, x3, scaled + g, accumulator0);
            accumulator1 = Group<TPlanes>(row1 + offset, tables, x0, x1, x2, x3, scaled + stride + g, accumulator1);
            accumulator2 = Group<TPlanes>(row2 + offset, tables, x0, x1, x2, x3, scaled + 2 * stride + g, accumulator2);
            accumulator3 = Group<TPlanes>(row3 + offset, tables, x0, x1, x2, x3, scaled + 3 * stride + g, accumulator3);
        }
        var pairs01 = Avx.HorizontalAdd(accumulator0, accumulator1);
        var pairs23 = Avx.HorizontalAdd(accumulator2, accumulator3);
        var quads = Avx.HorizontalAdd(pairs01, pairs23);
        Sse.Store(y, Sse.Add(quads.GetLower(), quads.GetUpper()));
    }

    static float Single<TPlanes>(byte* packed, ushort* norms, int groups, int stride, bool convertible, in PlaneTables tables, float* weightedScales, float* offsets, sbyte* values, float* scaled) where TPlanes : struct, IPlaneLayout
    {
        var accumulator = convertible
            ? Corrections(norms, stride, weightedScales, offsets, scaled)
            : CorrectionsAtEdge(norms, groups, stride, weightedScales, offsets, scaled);
        var groupBytes = TPlanes.GroupBytes;
        for (var g = 0; g < groups; g++)
        {
            var x = values + g * Cq.GroupSize;
            accumulator = Group<TPlanes>(packed + g * groupBytes, tables, Avx.LoadVector256(x), Avx.LoadVector256(x + 32), Avx.LoadVector256(x + 64), Avx.LoadVector256(x + 96), scaled + g, accumulator);
        }
        return Vector256.Sum(accumulator);
    }

    public static void Tile(in Cq w, in Activations a, int token, float* y, int ldy, int rowStart, int rowEnd)
    {
        if (UsePermute)
        {
            if (w.Bits == 2) Tile<PermuteTwoBitPlanes>(w, a, token, y, ldy, rowStart, rowEnd);
            else Tile<PermuteFourBitPlanes>(w, a, token, y, ldy, rowStart, rowEnd);
        }
        else if (w.Bits == 2) Tile<TwoBitPlanes>(w, a, token, y, ldy, rowStart, rowEnd);
        else Tile<FourBitPlanes>(w, a, token, y, ldy, rowStart, rowEnd);
    }

    static void Tile<TPlanes>(in Cq w, in Activations a, int token, float* y, int ldy, int rowStart, int rowEnd) where TPlanes : struct, IPlaneLayout
    {
        var groups = w.Groups;
        var stride = a.Stride;
        var cols = a.Cols;
        var tables = TPlanes.Tables(Avx.LoadVector256(w.Book->Bytes));
        var weightedScales = stackalloc float[TileTokens * stride];
        var scaled = stackalloc float[TileTokens * stride];
        var rowFactors = stackalloc float[stride];
        var byteScale = Vector256.Create(w.Book->ByteScale * HalfExponentRebias);
        for (var t = 0; t < TileTokens; t++)
            for (var v = 0; v < stride; v += HalvesPerVector)
                Avx.Store(weightedScales + t * stride + v, Avx.Multiply(Avx.LoadVector256(a.Scales + (token + t) * stride + v), byteScale));
        var offsets = a.Offsets + token * stride;
        var values = a.Values + (long)token * cols;
        var paddedGroups = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
        var convertibleRows = (long)w.Rows * groups < paddedGroups ? 0 : (int)(((long)w.Rows * groups - paddedGroups) / groups) + 1;
        var groupBytes = TPlanes.GroupBytes;
        for (var n = rowStart; n < rowEnd; n++)
        {
            var norms = w.Norms + (long)n * groups;
            if (n < convertibleRows)
                for (var v = 0; v < stride; v += HalvesPerVector) Avx.Store(rowFactors + v, UnbiasedHalfToSingle(Avx2.ConvertToVector256Int32(norms + v)));
            else
                for (var i = 0; i < stride; i++) rowFactors[i] = i < groups ? BitConverter.Int32BitsToSingle(norms[i] << 13) : 0f;
            var accumulator0 = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
            var accumulator1 = TileCorrections(rowFactors, weightedScales + stride, offsets + stride, scaled + stride, stride);
            var accumulator2 = TileCorrections(rowFactors, weightedScales + 2 * stride, offsets + 2 * stride, scaled + 2 * stride, stride);
            var accumulator3 = TileCorrections(rowFactors, weightedScales + 3 * stride, offsets + 3 * stride, scaled + 3 * stride, stride);
            var packed = w.Indices + (long)n * w.RowBytes;
            for (var g = 0; g < groups; g++)
            {
                TPlanes.Expand(packed + g * groupBytes, tables, out var p0, out var p1, out var p2, out var p3);
                var x = values + g * Cq.GroupSize;
                accumulator0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Dot(p0, p1, p2, p3, x)), Avx.BroadcastScalarToVector256(scaled + g), accumulator0);
                accumulator1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Dot(p0, p1, p2, p3, x + cols)), Avx.BroadcastScalarToVector256(scaled + stride + g), accumulator1);
                accumulator2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Dot(p0, p1, p2, p3, x + 2 * cols)), Avx.BroadcastScalarToVector256(scaled + 2 * stride + g), accumulator2);
                accumulator3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Dot(p0, p1, p2, p3, x + 3 * cols)), Avx.BroadcastScalarToVector256(scaled + 3 * stride + g), accumulator3);
            }
            var quads = Avx.HorizontalAdd(Avx.HorizontalAdd(accumulator0, accumulator1), Avx.HorizontalAdd(accumulator2, accumulator3));
            var sums = Sse.Add(quads.GetLower(), quads.GetUpper());
            y[n] = sums.ToScalar();
            y[ldy + n] = Sse41.Extract(sums, 1);
            y[2 * ldy + n] = Sse41.Extract(sums, 2);
            y[3 * ldy + n] = Sse41.Extract(sums, 3);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> TileCorrections(float* rowFactors, float* weightedScales, float* offsets, float* scaled, int stride)
    {
        var correction = Vector256<float>.Zero;
        for (var v = 0; v < stride; v += HalvesPerVector)
        {
            var factors = Avx.Multiply(Avx.LoadVector256(rowFactors + v), Avx.LoadVector256(weightedScales + v));
            Avx.Store(scaled + v, factors);
            correction = Fma.MultiplyAddNegated(factors, Avx.LoadVector256(offsets + v), correction);
        }
        return correction;
    }
}
#endif
