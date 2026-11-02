#if NEEDLE_SIMD
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Needle;

internal static unsafe class Gemv512Exp
{
    public static bool IsSupported =>
        Avx512F.IsSupported && Avx512BW.IsSupported && Avx512Vbmi.IsSupported &&
        Avx512Vbmi.VL.IsSupported && Vector512.IsHardwareAccelerated && Simd.UseAvx;

    public static Vector512<byte> Widen(Vector256<byte> lane) => Vector512.Create(lane, lane);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Expand2(
        byte* packed, Vector512<byte> first, Vector512<byte> second,
        out Vector512<byte> p0, out Vector512<byte> p1,
        out Vector512<byte> p2, out Vector512<byte> p3)
    {
        var words = Avx512F.LoadVector512((ushort*)packed);
        var lower = words.AsByte();
        var upper = Avx512BW.ShiftRightLogical(words, 4).AsByte();
        p0 = Avx512Vbmi.PermuteVar64x8(first, lower);
        p1 = Avx512Vbmi.PermuteVar64x8(second, lower);
        p2 = Avx512Vbmi.PermuteVar64x8(first, upper);
        p3 = Avx512Vbmi.PermuteVar64x8(second, upper);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Expand4(
        byte* packed, Vector512<byte> code,
        out Vector512<byte> low, out Vector512<byte> high)
    {
        var words = Avx512F.LoadVector512((ushort*)packed);
        low = Avx512Vbmi.PermuteVar64x8(code, words.AsByte());
        high = Avx512Vbmi.PermuteVar64x8(code, Avx512BW.ShiftRightLogical(words, 4).AsByte());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<int> Dot2(
        Vector512<byte> p0, Vector512<byte> p1,
        Vector512<byte> p2, Vector512<byte> p3, sbyte* x)
    {
        var x0 = Vector512.Create(Avx.LoadVector256(x), Avx.LoadVector256(x + 128));
        var x1 = Vector512.Create(Avx.LoadVector256(x + 32), Avx.LoadVector256(x + 160));
        var x2 = Vector512.Create(Avx.LoadVector256(x + 64), Avx.LoadVector256(x + 192));
        var x3 = Vector512.Create(Avx.LoadVector256(x + 96), Avx.LoadVector256(x + 224));
        var ones = Vector512.Create((short)1);
        var s0 = Avx512BW.MultiplyAddAdjacent(p0, x0);
        var s1 = Avx512BW.MultiplyAddAdjacent(p1, x1);
        var s2 = Avx512BW.MultiplyAddAdjacent(p2, x2);
        var s3 = Avx512BW.MultiplyAddAdjacent(p3, x3);
        var i01 = Avx512F.Add(
            Avx512BW.MultiplyAddAdjacent(s0, ones),
            Avx512BW.MultiplyAddAdjacent(s1, ones));
        var i23 = Avx512F.Add(
            Avx512BW.MultiplyAddAdjacent(s2, ones),
            Avx512BW.MultiplyAddAdjacent(s3, ones));
        return Avx512F.Add(i01, i23);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<int> Dot4(
        Vector512<byte> low, Vector512<byte> high, sbyte* x)
    {
        var ones = Vector512.Create((short)1);
        var lo = Avx512BW.MultiplyAddAdjacent(
            low, Vector512.Create(Avx.LoadVector256(x), Avx.LoadVector256(x + 64)));
        var hi = Avx512BW.MultiplyAddAdjacent(
            high, Vector512.Create(Avx.LoadVector256(x + 32), Avx.LoadVector256(x + 96)));
        return Avx512F.Add(
            Avx512BW.MultiplyAddAdjacent(lo, ones),
            Avx512BW.MultiplyAddAdjacent(hi, ones));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Accumulate(
        Vector512<float> accumulator, Vector512<int> dot, float scale) =>
        Avx512F.FusedMultiplyAdd(
            Avx512F.ConvertToVector512Single(dot),
            Vector512.Create(scale), accumulator);
}
#endif
