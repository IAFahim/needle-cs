#if NEEDLE_SIMD
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
    static abstract unsafe Vector256<int> Dot(byte* packed, in PlaneTables tables, Vector256<sbyte> x0, Vector256<sbyte> x1, Vector256<sbyte> x2, Vector256<sbyte> x3);
}

internal readonly struct TwoBitPlanes : IPlaneLayout
{
    public static int GroupBytes => 32;

    public static PlaneTables Tables(Vector256<byte> codebook) => new(
        Avx2.Shuffle(codebook, Vector256.Create((byte)0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3)),
        Avx2.Shuffle(codebook, Vector256.Create((byte)0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)),
        Vector256.Create((byte)0x0F));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe Vector256<int> Dot(byte* packed, in PlaneTables tables, Vector256<sbyte> x0, Vector256<sbyte> x1, Vector256<sbyte> x2, Vector256<sbyte> x3)
    {
        var v = Avx.LoadVector256(packed).AsUInt16();
        var lower = Avx2.And(v.AsByte(), tables.Mask);
        var upper = Avx2.And(Avx2.ShiftRightLogical(v, 4).AsByte(), tables.Mask);
        var sum = Quant.Accumulate(Vector256<int>.Zero, Avx2.Shuffle(tables.First, lower), x0);
        sum = Quant.Accumulate(sum, Avx2.Shuffle(tables.Second, lower), x1);
        sum = Quant.Accumulate(sum, Avx2.Shuffle(tables.First, upper), x2);
        return Quant.Accumulate(sum, Avx2.Shuffle(tables.Second, upper), x3);
    }
}

internal readonly struct FourBitPlanes : IPlaneLayout
{
    public static int GroupBytes => 64;

    public static PlaneTables Tables(Vector256<byte> codebook) => new(codebook, codebook, Vector256.Create((byte)0x0F));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe Vector256<int> Dot(byte* packed, in PlaneTables tables, Vector256<sbyte> x0, Vector256<sbyte> x1, Vector256<sbyte> x2, Vector256<sbyte> x3)
    {
        var lo = Avx.LoadVector256(packed).AsUInt16();
        var hi = Avx.LoadVector256(packed + 32).AsUInt16();
        var sum = Quant.Accumulate(Vector256<int>.Zero, Avx2.Shuffle(tables.First, Avx2.And(lo.AsByte(), tables.Mask)), x0);
        sum = Quant.Accumulate(sum, Avx2.Shuffle(tables.First, Avx2.And(Avx2.ShiftRightLogical(lo, 4).AsByte(), tables.Mask)), x1);
        sum = Quant.Accumulate(sum, Avx2.Shuffle(tables.First, Avx2.And(hi.AsByte(), tables.Mask)), x2);
        return Quant.Accumulate(sum, Avx2.Shuffle(tables.First, Avx2.And(Avx2.ShiftRightLogical(hi, 4).AsByte(), tables.Mask)), x3);
    }
}

internal static unsafe class Gemv
{
    const int RowBlock = 4;
    const int HalvesPerVector = 8;

    public static void Rows(in Cq w, in Activations a, int token, float* y, int rowStart, int rowEnd)
    {
        if (w.Bits == 2) Rows<TwoBitPlanes>(w, a, token, y, rowStart, rowEnd);
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
        var byteScale = Vector256.Create(w.Book->ByteScale);
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
    static Vector256<float> Factors(ushort* norms, int stride, float* weightedScales, float* offsets, float* scaled)
    {
        var bias = Vector256<float>.Zero;
        for (var v = 0; v < stride; v += HalvesPerVector)
        {
            var factors = Avx.Multiply(HalfToSingle(Avx2.ConvertToVector256Int32(norms + v)), Avx.LoadVector256(weightedScales + v));
            Avx.Store(scaled + v, factors);
            bias = Fma.MultiplyAdd(factors, Avx.LoadVector256(offsets + v), bias);
        }
        return bias;
    }

    static Vector256<float> FactorsAtEdge(ushort* norms, int groups, int stride, float* weightedScales, float* offsets, float* scaled)
    {
        for (var i = 0; i < stride; i++) scaled[i] = i < groups ? Scalar.HalfToSingle(norms[i]) : 0f;
        var bias = Vector256<float>.Zero;
        for (var v = 0; v < stride; v += HalvesPerVector)
        {
            var factors = Avx.Multiply(Avx.LoadVector256(scaled + v), Avx.LoadVector256(weightedScales + v));
            Avx.Store(scaled + v, factors);
            bias = Fma.MultiplyAdd(factors, Avx.LoadVector256(offsets + v), bias);
        }
        return bias;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> HalfToSingle(Vector256<int> half)
    {
        var magnitude = Avx2.And(half, Vector256.Create(0x7FFF));
        var sign = Avx2.ShiftLeftLogical(Avx2.And(half, Vector256.Create(0x8000)), 16);
        var bits = Avx2.Add(Avx2.ShiftLeftLogical(magnitude, 13), Vector256.Create(0x38000000));
        var nonZero = Avx2.CompareGreaterThan(magnitude, Vector256<int>.Zero);
        return Avx2.Or(Avx2.And(bits, nonZero), sign).AsSingle();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> Group<TPlanes>(byte* packed, in PlaneTables tables, Vector256<sbyte> x0, Vector256<sbyte> x1, Vector256<sbyte> x2, Vector256<sbyte> x3, float scale, Vector256<float> accumulator) where TPlanes : struct, IPlaneLayout =>
        Fma.MultiplyAdd(Avx.ConvertToVector256Single(TPlanes.Dot(packed, tables, x0, x1, x2, x3)), Vector256.Create(scale), accumulator);

    static void Block<TPlanes>(byte* packed, int rowBytes, ushort* norms, int groups, int stride, in PlaneTables tables, float* weightedScales, float* offsets, sbyte* values, float* scaled, float* y) where TPlanes : struct, IPlaneLayout
    {
        var bias0 = Factors(norms, stride, weightedScales, offsets, scaled);
        var bias1 = Factors(norms + groups, stride, weightedScales, offsets, scaled + stride);
        var bias2 = Factors(norms + 2 * groups, stride, weightedScales, offsets, scaled + 2 * stride);
        var bias3 = Factors(norms + 3 * groups, stride, weightedScales, offsets, scaled + 3 * stride);
        var row1 = packed + rowBytes;
        var row2 = packed + 2 * rowBytes;
        var row3 = packed + 3 * rowBytes;
        var accumulator0 = Vector256<float>.Zero;
        var accumulator1 = Vector256<float>.Zero;
        var accumulator2 = Vector256<float>.Zero;
        var accumulator3 = Vector256<float>.Zero;
        var groupBytes = TPlanes.GroupBytes;
        for (var g = 0; g < groups; g++)
        {
            var x = values + g * Cq.GroupSize;
            var x0 = Avx.LoadVector256(x);
            var x1 = Avx.LoadVector256(x + 32);
            var x2 = Avx.LoadVector256(x + 64);
            var x3 = Avx.LoadVector256(x + 96);
            var offset = g * groupBytes;
            accumulator0 = Group<TPlanes>(packed + offset, tables, x0, x1, x2, x3, scaled[g], accumulator0);
            accumulator1 = Group<TPlanes>(row1 + offset, tables, x0, x1, x2, x3, scaled[stride + g], accumulator1);
            accumulator2 = Group<TPlanes>(row2 + offset, tables, x0, x1, x2, x3, scaled[2 * stride + g], accumulator2);
            accumulator3 = Group<TPlanes>(row3 + offset, tables, x0, x1, x2, x3, scaled[3 * stride + g], accumulator3);
        }
        var pairs01 = Avx.HorizontalAdd(Avx.Subtract(accumulator0, bias0), Avx.Subtract(accumulator1, bias1));
        var pairs23 = Avx.HorizontalAdd(Avx.Subtract(accumulator2, bias2), Avx.Subtract(accumulator3, bias3));
        var quads = Avx.HorizontalAdd(pairs01, pairs23);
        Sse.Store(y, Sse.Add(quads.GetLower(), quads.GetUpper()));
    }

    static float Single<TPlanes>(byte* packed, ushort* norms, int groups, int stride, bool convertible, in PlaneTables tables, float* weightedScales, float* offsets, sbyte* values, float* scaled) where TPlanes : struct, IPlaneLayout
    {
        var bias = convertible
            ? Factors(norms, stride, weightedScales, offsets, scaled)
            : FactorsAtEdge(norms, groups, stride, weightedScales, offsets, scaled);
        var accumulator = Vector256<float>.Zero;
        var groupBytes = TPlanes.GroupBytes;
        for (var g = 0; g < groups; g++)
        {
            var x = values + g * Cq.GroupSize;
            accumulator = Group<TPlanes>(packed + g * groupBytes, tables, Avx.LoadVector256(x), Avx.LoadVector256(x + 32), Avx.LoadVector256(x + 64), Avx.LoadVector256(x + 96), scaled[g], accumulator);
        }
        return Vector256.Sum(Avx.Subtract(accumulator, bias));
    }
}
#endif
