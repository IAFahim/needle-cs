using System;
#if NEEDLE_SIMD
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
#endif

namespace Needle;

internal enum CpuTier : byte
{
    Scalar,
    Avx2,
    AvxVnni,
    AvxVnniInt8,
    Avx512Bw,
    Neon,
    NeonDot,
}

#if NEEDLE_SIMD
internal interface IInt8Dot
{
    static abstract Vector256<int> Dot(
        Vector256<byte> p0, Vector256<byte> p1, Vector256<byte> p2, Vector256<byte> p3,
        Vector256<sbyte> x0, Vector256<sbyte> x1, Vector256<sbyte> x2, Vector256<sbyte> x3);
}

internal readonly struct Avx2Dot : IInt8Dot
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<int> Dot(
        Vector256<byte> p0, Vector256<byte> p1, Vector256<byte> p2, Vector256<byte> p3,
        Vector256<sbyte> x0, Vector256<sbyte> x1, Vector256<sbyte> x2, Vector256<sbyte> x3)
    {
        var ones = Vector256.Create((short)1);
        var sum = Avx2.Add(Vector256<int>.Zero, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(p0, x0), ones));
        sum = Avx2.Add(sum, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(p1, x1), ones));
        sum = Avx2.Add(sum, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(p2, x2), ones));
        return Avx2.Add(sum, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(p3, x3), ones));
    }
}

internal readonly struct VnniDot : IInt8Dot
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<int> Dot(
        Vector256<byte> p0, Vector256<byte> p1, Vector256<byte> p2, Vector256<byte> p3,
        Vector256<sbyte> x0, Vector256<sbyte> x1, Vector256<sbyte> x2, Vector256<sbyte> x3)
    {
        var sum = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, p0, x0);
        sum = AvxVnni.MultiplyWideningAndAdd(sum, p1, x1);
        sum = AvxVnni.MultiplyWideningAndAdd(sum, p2, x2);
        return AvxVnni.MultiplyWideningAndAdd(sum, p3, x3);
    }
}

internal readonly struct VnniInt8Dot : IInt8Dot
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<int> Dot(
        Vector256<byte> p0, Vector256<byte> p1, Vector256<byte> p2, Vector256<byte> p3,
        Vector256<sbyte> x0, Vector256<sbyte> x1, Vector256<sbyte> x2, Vector256<sbyte> x3)
    {
        var sum = AvxVnniInt8.MultiplyWideningAndAdd(Vector256<int>.Zero, p0.AsSByte(), x0);
        sum = AvxVnniInt8.MultiplyWideningAndAdd(sum, p1.AsSByte(), x1);
        sum = AvxVnniInt8.MultiplyWideningAndAdd(sum, p2.AsSByte(), x2);
        return AvxVnniInt8.MultiplyWideningAndAdd(sum, p3.AsSByte(), x3);
    }
}

internal readonly struct BwDot : IInt8Dot
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<int> Dot(
        Vector256<byte> p0, Vector256<byte> p1, Vector256<byte> p2, Vector256<byte> p3,
        Vector256<sbyte> x0, Vector256<sbyte> x1, Vector256<sbyte> x2, Vector256<sbyte> x3)
    {
        var low = Avx512BW.MultiplyAddAdjacent(Vector512.Create(p0, p1), Vector512.Create(x0, x1));
        var high = Avx512BW.MultiplyAddAdjacent(Vector512.Create(p2, p3), Vector512.Create(x2, x3));
        var ones = Vector512.Create((short)1);
        var sums = Avx512F.Add(Avx512BW.MultiplyAddAdjacent(low, ones), Avx512BW.MultiplyAddAdjacent(high, ones));
        return Avx2.Add(sums.GetLower(), sums.GetUpper());
    }
}
#endif

internal static class Tiers
{
#if NEEDLE_SIMD
    internal static readonly CpuTier Active = Select();

    internal static bool HasAvx => Avx2.IsSupported && Fma.IsSupported && Sse41.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<int> Accumulate(Vector256<int> sum, Vector256<byte> weights, Vector256<sbyte> activations)
    {
        if (Active == CpuTier.AvxVnniInt8) return AvxVnniInt8.MultiplyWideningAndAdd(sum, weights.AsSByte(), activations);
        if (Active == CpuTier.AvxVnni) return AvxVnni.MultiplyWideningAndAdd(sum, weights, activations);
        return Avx2.Add(sum, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), Vector256.Create((short)1)));
    }

    static CpuTier Select()
    {
        if (TryParseTier(Environment.GetEnvironmentVariable("NEEDLE_TIER"), out var forced) && Supported(forced))
            return forced;
        if (Environment.GetEnvironmentVariable("NEEDLE_SCALAR") != null)
            return CpuTier.Scalar;
        return Best();
    }

    static CpuTier Best()
    {
        if (AdvSimd.IsSupported)
            return Dp.IsSupported && Environment.GetEnvironmentVariable("NEEDLE_NODP") == null ? CpuTier.NeonDot : CpuTier.Neon;
        if (Environment.GetEnvironmentVariable("NEEDLE_NOVNNI") == null)
        {
            if (HasAvx && AvxVnniInt8.IsSupported) return CpuTier.AvxVnniInt8;
            if (HasAvx && AvxVnni.IsSupported) return CpuTier.AvxVnni;
        }
        if (HasAvx && Avx512F.IsSupported && Avx512BW.IsSupported && Vector512.IsHardwareAccelerated
            && Environment.GetEnvironmentVariable("NEEDLE_NOAVX512") == null)
            return CpuTier.Avx512Bw;
        if (HasAvx) return CpuTier.Avx2;
        return CpuTier.Scalar;
    }

    static bool Supported(CpuTier tier) => tier switch
    {
        CpuTier.Scalar => true,
        CpuTier.Avx2 => HasAvx,
        CpuTier.AvxVnni => HasAvx && AvxVnni.IsSupported,
        CpuTier.AvxVnniInt8 => HasAvx && AvxVnniInt8.IsSupported,
        CpuTier.Avx512Bw => HasAvx && Avx512F.IsSupported && Avx512BW.IsSupported && Vector512.IsHardwareAccelerated,
        CpuTier.Neon => AdvSimd.IsSupported,
        CpuTier.NeonDot => AdvSimd.IsSupported && Dp.IsSupported,
        _ => false,
    };
#else
    internal const CpuTier Active = CpuTier.Scalar;
#endif

    internal static bool TryParseTier(string? text, out CpuTier tier)
    {
        if (!string.IsNullOrWhiteSpace(text))
        {
            switch (text.Trim().ToLowerInvariant())
            {
                case "scalar": tier = CpuTier.Scalar; return true;
                case "avx2": tier = CpuTier.Avx2; return true;
                case "avxvnni": tier = CpuTier.AvxVnni; return true;
                case "avxvnniint8": tier = CpuTier.AvxVnniInt8; return true;
                case "avx512":
                case "avx512bw": tier = CpuTier.Avx512Bw; return true;
                case "neon": tier = CpuTier.Neon; return true;
                case "neondot": tier = CpuTier.NeonDot; return true;
            }
        }
        tier = CpuTier.Scalar;
        return false;
    }
}
