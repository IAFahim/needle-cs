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
            if (w.Bits == 2) RowsDot<PermuteTwoBitPlanes>(w, a, token, y, rowStart, rowEnd);
            else RowsDot<PermuteFourBitPlanes>(w, a, token, y, rowStart, rowEnd);
        }
        else if (w.Bits == 2) RowsDot<TwoBitPlanes>(w, a, token, y, rowStart, rowEnd);
        else RowsDot<FourBitPlanes>(w, a, token, y, rowStart, rowEnd);
    }

    static void RowsDot<TPlanes>(in Cq w, in Activations a, int token, float* y, int rowStart, int rowEnd) where TPlanes : struct, IPlaneLayout
    {
        if (Tiers.Active == CpuTier.AvxVnniInt8) Rows<TPlanes, VnniInt8Dot>(w, a, token, y, rowStart, rowEnd);
        else if (Tiers.Active == CpuTier.Avx512Bw) Rows<TPlanes, BwDot>(w, a, token, y, rowStart, rowEnd);
        else if (Tiers.Active == CpuTier.AvxVnni) Rows<TPlanes, VnniDot>(w, a, token, y, rowStart, rowEnd);
        else Rows<TPlanes, Avx2Dot>(w, a, token, y, rowStart, rowEnd);
    }

    static void Rows<TPlanes, TDot>(in Cq w, in Activations a, int token, float* y, int rowStart, int rowEnd)
        where TPlanes : struct, IPlaneLayout
        where TDot : struct, IInt8Dot
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
            Block<TPlanes, TDot>(w.Indices + (long)n * rowBytes, rowBytes, w.Norms + (long)n * groups, groups, stride, tables, weightedScales, offsets, values, scaled, y + n);
        for (; n < rowEnd; n++)
            y[n] = Single<TPlanes, TDot>(w.Indices + (long)n * rowBytes, w.Norms + (long)n * groups, groups, stride, n < convertibleRows, tables, weightedScales, offsets, values, scaled);
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
    static Vector256<int> Dot<TDot>(Vector256<byte> p0, Vector256<byte> p1, Vector256<byte> p2, Vector256<byte> p3, Vector256<sbyte> x0, Vector256<sbyte> x1, Vector256<sbyte> x2, Vector256<sbyte> x3) where TDot : struct, IInt8Dot =>
        TDot.Dot(p0, p1, p2, p3, x0, x1, x2, x3);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<int> Dot<TDot>(Vector256<byte> p0, Vector256<byte> p1, Vector256<byte> p2, Vector256<byte> p3, sbyte* x) where TDot : struct, IInt8Dot =>
        Dot<TDot>(p0, p1, p2, p3, Avx.LoadVector256(x), Avx.LoadVector256(x + 32), Avx.LoadVector256(x + 64), Avx.LoadVector256(x + 96));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> Group<TPlanes, TDot>(byte* packed, in PlaneTables tables, Vector256<sbyte> x0, Vector256<sbyte> x1, Vector256<sbyte> x2, Vector256<sbyte> x3, float* scale, Vector256<float> accumulator)
        where TPlanes : struct, IPlaneLayout
        where TDot : struct, IInt8Dot
    {
        TPlanes.Expand(packed, tables, out var p0, out var p1, out var p2, out var p3);
        return Fma.MultiplyAdd(Avx.ConvertToVector256Single(Dot<TDot>(p0, p1, p2, p3, x0, x1, x2, x3)), Avx.BroadcastScalarToVector256(scale), accumulator);
    }

    static void Block<TPlanes, TDot>(byte* packed, int rowBytes, ushort* norms, int groups, int stride, in PlaneTables tables, float* weightedScales, float* offsets, sbyte* values, float* scaled, float* y)
        where TPlanes : struct, IPlaneLayout
        where TDot : struct, IInt8Dot
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
            accumulator0 = Group<TPlanes, TDot>(packed + offset, tables, x0, x1, x2, x3, scaled + g, accumulator0);
            accumulator1 = Group<TPlanes, TDot>(row1 + offset, tables, x0, x1, x2, x3, scaled + stride + g, accumulator1);
            accumulator2 = Group<TPlanes, TDot>(row2 + offset, tables, x0, x1, x2, x3, scaled + 2 * stride + g, accumulator2);
            accumulator3 = Group<TPlanes, TDot>(row3 + offset, tables, x0, x1, x2, x3, scaled + 3 * stride + g, accumulator3);
        }
        var pairs01 = Avx.HorizontalAdd(accumulator0, accumulator1);
        var pairs23 = Avx.HorizontalAdd(accumulator2, accumulator3);
        var quads = Avx.HorizontalAdd(pairs01, pairs23);
        Sse.Store(y, Sse.Add(quads.GetLower(), quads.GetUpper()));
    }

    static float Single<TPlanes, TDot>(byte* packed, ushort* norms, int groups, int stride, bool convertible, in PlaneTables tables, float* weightedScales, float* offsets, sbyte* values, float* scaled)
        where TPlanes : struct, IPlaneLayout
        where TDot : struct, IInt8Dot
    {
        var accumulator = convertible
            ? Corrections(norms, stride, weightedScales, offsets, scaled)
            : CorrectionsAtEdge(norms, groups, stride, weightedScales, offsets, scaled);
        var groupBytes = TPlanes.GroupBytes;
        for (var g = 0; g < groups; g++)
        {
            var x = values + g * Cq.GroupSize;
            accumulator = Group<TPlanes, TDot>(packed + g * groupBytes, tables, Avx.LoadVector256(x), Avx.LoadVector256(x + 32), Avx.LoadVector256(x + 64), Avx.LoadVector256(x + 96), scaled + g, accumulator);
        }
        return Vector256.Sum(accumulator);
    }

    public static void Tile4(in Cq w, in Activations a, int token, float* y, int ldy, int rowStart, int rowEnd)
    {
        if (UsePermute)
        {
            if (w.Bits == 2) Tile4Dot<PermuteTwoBitPlanes>(w, a, token, y, ldy, rowStart, rowEnd);
            else Tile4Dot<PermuteFourBitPlanes>(w, a, token, y, ldy, rowStart, rowEnd);
        }
        else if (w.Bits == 2) Tile4Dot<TwoBitPlanes>(w, a, token, y, ldy, rowStart, rowEnd);
        else Tile4Dot<FourBitPlanes>(w, a, token, y, ldy, rowStart, rowEnd);
    }

    static void Tile4Dot<TPlanes>(in Cq w, in Activations a, int token, float* y, int ldy, int rowStart, int rowEnd) where TPlanes : struct, IPlaneLayout
    {
        if (Tiers.Active == CpuTier.AvxVnniInt8) Tile4<TPlanes, VnniInt8Dot>(w, a, token, y, ldy, rowStart, rowEnd);
        else if (Tiers.Active == CpuTier.Avx512Bw) Tile4<TPlanes, BwDot>(w, a, token, y, ldy, rowStart, rowEnd);
        else if (Tiers.Active == CpuTier.AvxVnni) Tile4<TPlanes, VnniDot>(w, a, token, y, ldy, rowStart, rowEnd);
        else Tile4<TPlanes, Avx2Dot>(w, a, token, y, ldy, rowStart, rowEnd);
    }

    public static void Tile2(in Cq w, in Activations a, int token, float* y, int ldy, int rowStart, int rowEnd)
    {
        if (UsePermute)
        {
            if (w.Bits == 2) Tile2Dot<PermuteTwoBitPlanes>(w, a, token, y, ldy, rowStart, rowEnd);
            else Tile2Dot<PermuteFourBitPlanes>(w, a, token, y, ldy, rowStart, rowEnd);
        }
        else if (w.Bits == 2) Tile2Dot<TwoBitPlanes>(w, a, token, y, ldy, rowStart, rowEnd);
        else Tile2Dot<FourBitPlanes>(w, a, token, y, ldy, rowStart, rowEnd);
    }

    static void Tile2Dot<TPlanes>(in Cq w, in Activations a, int token, float* y, int ldy, int rowStart, int rowEnd) where TPlanes : struct, IPlaneLayout
    {
        if (Tiers.Active == CpuTier.AvxVnniInt8) Tile2<TPlanes, VnniInt8Dot>(w, a, token, y, ldy, rowStart, rowEnd);
        else if (Tiers.Active == CpuTier.Avx512Bw) Tile2<TPlanes, BwDot>(w, a, token, y, ldy, rowStart, rowEnd);
        else if (Tiers.Active == CpuTier.AvxVnni) Tile2<TPlanes, VnniDot>(w, a, token, y, ldy, rowStart, rowEnd);
        else Tile2<TPlanes, Avx2Dot>(w, a, token, y, ldy, rowStart, rowEnd);
    }

    static void Tile4<TPlanes, TDot>(in Cq w, in Activations a, int token, float* y, int ldy, int rowStart, int rowEnd)
        where TPlanes : struct, IPlaneLayout
        where TDot : struct, IInt8Dot
    {
        var groups = w.Groups;
        var stride = a.Stride;
        var cols = a.Cols;
        var tables = TPlanes.Tables(Avx.LoadVector256(w.Book->Bytes));
        var weightedScales = stackalloc float[TileTokens * stride];
        var scaled = stackalloc float[2 * TileTokens * stride];
        var rowFactors = stackalloc float[2 * stride];
        var byteScale = Vector256.Create(w.Book->ByteScale * HalfExponentRebias);
        for (var t = 0; t < TileTokens; t++)
            for (var v = 0; v < stride; v += HalvesPerVector)
                Avx.Store(weightedScales + t * stride + v, Avx.Multiply(Avx.LoadVector256(a.Scales + (token + t) * stride + v), byteScale));
        var offsets = a.Offsets + token * stride;
        var values = a.Values + (long)token * cols;
        var rowBytes = w.RowBytes;
        var groupBytes = TPlanes.GroupBytes;
        var paddedGroups = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
        var convertibleRows = (long)w.Rows * groups < paddedGroups ? 0 : (int)(((long)w.Rows * groups - paddedGroups) / groups) + 1;
        var prefetchDistance = (long)PrefetchBlocks * 2 * rowBytes;
        var n = rowStart;
        for (; n + 2 <= rowEnd && n + 2 <= convertibleRows; n += 2)
            Pair4<TPlanes, TDot>(w.Indices + (long)n * rowBytes, rowBytes, w.Norms + (long)n * groups, groups, stride, cols, tables, weightedScales, offsets, values, scaled, rowFactors, groupBytes, prefetchDistance, y + n, ldy);
        for (; n < rowEnd; n++)
            Row4<TPlanes, TDot>(w.Indices + (long)n * rowBytes, w.Norms + (long)n * groups, n < convertibleRows, groups, stride, cols, tables, weightedScales, offsets, values, scaled, rowFactors, groupBytes, prefetchDistance, y + n, ldy);
    }

    static void Tile2<TPlanes, TDot>(in Cq w, in Activations a, int token, float* y, int ldy, int rowStart, int rowEnd)
        where TPlanes : struct, IPlaneLayout
        where TDot : struct, IInt8Dot
    {
        const int Tokens = 2;
        var groups = w.Groups;
        var stride = a.Stride;
        var cols = a.Cols;
        var tables = TPlanes.Tables(Avx.LoadVector256(w.Book->Bytes));
        var weightedScales = stackalloc float[Tokens * stride];
        var scaled = stackalloc float[2 * Tokens * stride];
        var rowFactors = stackalloc float[2 * stride];
        var byteScale = Vector256.Create(w.Book->ByteScale * HalfExponentRebias);
        for (var t = 0; t < Tokens; t++)
            for (var v = 0; v < stride; v += HalvesPerVector)
                Avx.Store(weightedScales + t * stride + v, Avx.Multiply(Avx.LoadVector256(a.Scales + (token + t) * stride + v), byteScale));
        var offsets = a.Offsets + token * stride;
        var values = a.Values + (long)token * cols;
        var rowBytes = w.RowBytes;
        var groupBytes = TPlanes.GroupBytes;
        var paddedGroups = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
        var convertibleRows = (long)w.Rows * groups < paddedGroups ? 0 : (int)(((long)w.Rows * groups - paddedGroups) / groups) + 1;
        var prefetchDistance = (long)PrefetchBlocks * 2 * rowBytes;
        var n = rowStart;
        for (; n + 2 <= rowEnd && n + 2 <= convertibleRows; n += 2)
            Pair2<TPlanes, TDot>(w.Indices + (long)n * rowBytes, rowBytes, w.Norms + (long)n * groups, groups, stride, cols, tables, weightedScales, offsets, values, scaled, rowFactors, groupBytes, prefetchDistance, y + n, ldy);
        for (; n < rowEnd; n++)
            Row2<TPlanes, TDot>(w.Indices + (long)n * rowBytes, w.Norms + (long)n * groups, n < convertibleRows, groups, stride, cols, tables, weightedScales, offsets, values, scaled, rowFactors, groupBytes, prefetchDistance, y + n, ldy);
    }

    static void Pair4<TPlanes, TDot>(byte* packed, int rowBytes, ushort* norms, int groups, int stride, int cols, in PlaneTables tables, float* weightedScales, float* offsets, sbyte* values, float* scaled, float* rowFactors, int groupBytes, long prefetchDistance, float* y, int ldy)
        where TPlanes : struct, IPlaneLayout
        where TDot : struct, IInt8Dot
    {
        for (var v = 0; v < stride; v += HalvesPerVector)
        {
            Avx.Store(rowFactors + v, UnbiasedHalfToSingle(Avx2.ConvertToVector256Int32(norms + v)));
            Avx.Store(rowFactors + stride + v, UnbiasedHalfToSingle(Avx2.ConvertToVector256Int32(norms + groups + v)));
        }
        var accumulator00 = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
        var accumulator01 = TileCorrections(rowFactors, weightedScales + stride, offsets + stride, scaled + stride, stride);
        var accumulator02 = TileCorrections(rowFactors, weightedScales + 2 * stride, offsets + 2 * stride, scaled + 2 * stride, stride);
        var accumulator03 = TileCorrections(rowFactors, weightedScales + 3 * stride, offsets + 3 * stride, scaled + 3 * stride, stride);
        var accumulator10 = TileCorrections(rowFactors + stride, weightedScales, offsets, scaled + 4 * stride, stride);
        var accumulator11 = TileCorrections(rowFactors + stride, weightedScales + stride, offsets + stride, scaled + 5 * stride, stride);
        var accumulator12 = TileCorrections(rowFactors + stride, weightedScales + 2 * stride, offsets + 2 * stride, scaled + 6 * stride, stride);
        var accumulator13 = TileCorrections(rowFactors + stride, weightedScales + 3 * stride, offsets + 3 * stride, scaled + 7 * stride, stride);
        var row1 = packed + rowBytes;
        for (var g = 0; g < groups; g++)
        {
            if ((g * groupBytes & CacheLine - 1) == 0)
            {
                var ahead = g * groupBytes + prefetchDistance;
                Sse.Prefetch0(packed + ahead);
                Sse.Prefetch0(row1 + ahead);
            }
            TPlanes.Expand(packed + g * groupBytes, tables, out var p0, out var p1, out var p2, out var p3);
            TPlanes.Expand(row1 + g * groupBytes, tables, out var q0, out var q1, out var q2, out var q3);
            var x = values + g * Cq.GroupSize;
            var x1 = x + cols;
            var x2 = x + 2 * cols;
            var x3 = x + 3 * cols;
            accumulator00 = ApplyPlanes<TDot>(p0, p1, p2, p3, x, scaled + g, accumulator00);
            accumulator10 = ApplyPlanes<TDot>(q0, q1, q2, q3, x, scaled + 4 * stride + g, accumulator10);
            accumulator01 = ApplyPlanes<TDot>(p0, p1, p2, p3, x1, scaled + stride + g, accumulator01);
            accumulator11 = ApplyPlanes<TDot>(q0, q1, q2, q3, x1, scaled + 5 * stride + g, accumulator11);
            accumulator02 = ApplyPlanes<TDot>(p0, p1, p2, p3, x2, scaled + 2 * stride + g, accumulator02);
            accumulator12 = ApplyPlanes<TDot>(q0, q1, q2, q3, x2, scaled + 6 * stride + g, accumulator12);
            accumulator03 = ApplyPlanes<TDot>(p0, p1, p2, p3, x3, scaled + 3 * stride + g, accumulator03);
            accumulator13 = ApplyPlanes<TDot>(q0, q1, q2, q3, x3, scaled + 7 * stride + g, accumulator13);
        }
        var quads0 = Avx.HorizontalAdd(Avx.HorizontalAdd(accumulator00, accumulator01), Avx.HorizontalAdd(accumulator02, accumulator03));
        var sums0 = Sse.Add(quads0.GetLower(), quads0.GetUpper());
        y[0] = sums0.ToScalar();
        y[ldy] = Sse41.Extract(sums0, 1);
        y[2 * ldy] = Sse41.Extract(sums0, 2);
        y[3 * ldy] = Sse41.Extract(sums0, 3);
        var quads1 = Avx.HorizontalAdd(Avx.HorizontalAdd(accumulator10, accumulator11), Avx.HorizontalAdd(accumulator12, accumulator13));
        var sums1 = Sse.Add(quads1.GetLower(), quads1.GetUpper());
        y[1] = sums1.ToScalar();
        y[ldy + 1] = Sse41.Extract(sums1, 1);
        y[2 * ldy + 1] = Sse41.Extract(sums1, 2);
        y[3 * ldy + 1] = Sse41.Extract(sums1, 3);
    }

    static void Row4<TPlanes, TDot>(byte* packed, ushort* norms, bool convertible, int groups, int stride, int cols, in PlaneTables tables, float* weightedScales, float* offsets, sbyte* values, float* scaled, float* rowFactors, int groupBytes, long prefetchDistance, float* y, int ldy)
        where TPlanes : struct, IPlaneLayout
        where TDot : struct, IInt8Dot
    {
        if (convertible)
            for (var v = 0; v < stride; v += HalvesPerVector) Avx.Store(rowFactors + v, UnbiasedHalfToSingle(Avx2.ConvertToVector256Int32(norms + v)));
        else
            for (var i = 0; i < stride; i++) rowFactors[i] = i < groups ? BitConverter.Int32BitsToSingle(norms[i] << 13) : 0f;
        var accumulator0 = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
        var accumulator1 = TileCorrections(rowFactors, weightedScales + stride, offsets + stride, scaled + stride, stride);
        var accumulator2 = TileCorrections(rowFactors, weightedScales + 2 * stride, offsets + 2 * stride, scaled + 2 * stride, stride);
        var accumulator3 = TileCorrections(rowFactors, weightedScales + 3 * stride, offsets + 3 * stride, scaled + 3 * stride, stride);
        for (var g = 0; g < groups; g++)
        {
            if ((g * groupBytes & CacheLine - 1) == 0) Sse.Prefetch0(packed + g * groupBytes + prefetchDistance);
            TPlanes.Expand(packed + g * groupBytes, tables, out var p0, out var p1, out var p2, out var p3);
            var x = values + g * Cq.GroupSize;
            accumulator0 = ApplyPlanes<TDot>(p0, p1, p2, p3, x, scaled + g, accumulator0);
            accumulator1 = ApplyPlanes<TDot>(p0, p1, p2, p3, x + cols, scaled + stride + g, accumulator1);
            accumulator2 = ApplyPlanes<TDot>(p0, p1, p2, p3, x + 2 * cols, scaled + 2 * stride + g, accumulator2);
            accumulator3 = ApplyPlanes<TDot>(p0, p1, p2, p3, x + 3 * cols, scaled + 3 * stride + g, accumulator3);
        }
        var quads = Avx.HorizontalAdd(Avx.HorizontalAdd(accumulator0, accumulator1), Avx.HorizontalAdd(accumulator2, accumulator3));
        var sums = Sse.Add(quads.GetLower(), quads.GetUpper());
        y[0] = sums.ToScalar();
        y[ldy] = Sse41.Extract(sums, 1);
        y[2 * ldy] = Sse41.Extract(sums, 2);
        y[3 * ldy] = Sse41.Extract(sums, 3);
    }

    static void Pair2<TPlanes, TDot>(byte* packed, int rowBytes, ushort* norms, int groups, int stride, int cols, in PlaneTables tables, float* weightedScales, float* offsets, sbyte* values, float* scaled, float* rowFactors, int groupBytes, long prefetchDistance, float* y, int ldy)
        where TPlanes : struct, IPlaneLayout
        where TDot : struct, IInt8Dot
    {
        for (var v = 0; v < stride; v += HalvesPerVector)
        {
            Avx.Store(rowFactors + v, UnbiasedHalfToSingle(Avx2.ConvertToVector256Int32(norms + v)));
            Avx.Store(rowFactors + stride + v, UnbiasedHalfToSingle(Avx2.ConvertToVector256Int32(norms + groups + v)));
        }
        var accumulator00 = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
        var accumulator01 = TileCorrections(rowFactors, weightedScales + stride, offsets + stride, scaled + stride, stride);
        var accumulator10 = TileCorrections(rowFactors + stride, weightedScales, offsets, scaled + 2 * stride, stride);
        var accumulator11 = TileCorrections(rowFactors + stride, weightedScales + stride, offsets + stride, scaled + 3 * stride, stride);
        var row1 = packed + rowBytes;
        for (var g = 0; g < groups; g++)
        {
            if ((g * groupBytes & CacheLine - 1) == 0)
            {
                var ahead = g * groupBytes + prefetchDistance;
                Sse.Prefetch0(packed + ahead);
                Sse.Prefetch0(row1 + ahead);
            }
            TPlanes.Expand(packed + g * groupBytes, tables, out var p0, out var p1, out var p2, out var p3);
            TPlanes.Expand(row1 + g * groupBytes, tables, out var q0, out var q1, out var q2, out var q3);
            var x = values + g * Cq.GroupSize;
            var x1 = x + cols;
            accumulator00 = ApplyPlanes<TDot>(p0, p1, p2, p3, x, scaled + g, accumulator00);
            accumulator10 = ApplyPlanes<TDot>(q0, q1, q2, q3, x, scaled + 2 * stride + g, accumulator10);
            accumulator01 = ApplyPlanes<TDot>(p0, p1, p2, p3, x1, scaled + stride + g, accumulator01);
            accumulator11 = ApplyPlanes<TDot>(q0, q1, q2, q3, x1, scaled + 3 * stride + g, accumulator11);
        }
        y[0] = Vector256.Sum(accumulator00);
        y[ldy] = Vector256.Sum(accumulator01);
        y[1] = Vector256.Sum(accumulator10);
        y[ldy + 1] = Vector256.Sum(accumulator11);
    }

    static void Row2<TPlanes, TDot>(byte* packed, ushort* norms, bool convertible, int groups, int stride, int cols, in PlaneTables tables, float* weightedScales, float* offsets, sbyte* values, float* scaled, float* rowFactors, int groupBytes, long prefetchDistance, float* y, int ldy)
        where TPlanes : struct, IPlaneLayout
        where TDot : struct, IInt8Dot
    {
        if (convertible)
            for (var v = 0; v < stride; v += HalvesPerVector) Avx.Store(rowFactors + v, UnbiasedHalfToSingle(Avx2.ConvertToVector256Int32(norms + v)));
        else
            for (var i = 0; i < stride; i++) rowFactors[i] = i < groups ? BitConverter.Int32BitsToSingle(norms[i] << 13) : 0f;
        var accumulator0 = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
        var accumulator1 = TileCorrections(rowFactors, weightedScales + stride, offsets + stride, scaled + stride, stride);
        for (var g = 0; g < groups; g++)
        {
            if ((g * groupBytes & CacheLine - 1) == 0) Sse.Prefetch0(packed + g * groupBytes + prefetchDistance);
            TPlanes.Expand(packed + g * groupBytes, tables, out var p0, out var p1, out var p2, out var p3);
            var x = values + g * Cq.GroupSize;
            accumulator0 = ApplyPlanes<TDot>(p0, p1, p2, p3, x, scaled + g, accumulator0);
            accumulator1 = ApplyPlanes<TDot>(p0, p1, p2, p3, x + cols, scaled + stride + g, accumulator1);
        }
        y[0] = Vector256.Sum(accumulator0);
        y[ldy] = Vector256.Sum(accumulator1);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> ApplyPlanes<TDot>(Vector256<byte> p0, Vector256<byte> p1, Vector256<byte> p2, Vector256<byte> p3, sbyte* x, float* scale, Vector256<float> accumulator) where TDot : struct, IInt8Dot =>
        Fma.MultiplyAdd(Avx.ConvertToVector256Single(Dot<TDot>(p0, p1, p2, p3, x)), Avx.BroadcastScalarToVector256(scale), accumulator);

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
