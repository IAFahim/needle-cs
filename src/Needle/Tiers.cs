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
    Avx512Vnni,
}

// What the dispatcher needs to know about a CPU, as plain flags so tests can describe any machine.
internal readonly struct CpuFeatures
{
    public bool Avx2 { get; init; }
    public bool AvxVnni { get; init; }
    public bool AvxVnniInt8 { get; init; }
    public bool Avx512 { get; init; }
    public bool Avx512Vnni { get; init; }
    public bool AdvSimd { get; init; }
    public bool Dp { get; init; }

    public static CpuFeatures Zen4 => new() { Avx2 = true, Avx512 = true, Avx512Vnni = true };
    public static CpuFeatures AlderLake => new() { Avx2 = true, AvxVnni = true };
    public static CpuFeatures SapphireRapids => new() { Avx2 = true, AvxVnni = true, Avx512 = true, Avx512Vnni = true };
    public static CpuFeatures PlainAvx2 => new() { Avx2 = true };

    public override string ToString() =>
        $"Avx2 {Avx2}, AvxVnni {AvxVnni}, AvxVnniInt8 {AvxVnniInt8}, Avx512 {Avx512}, Avx512Vnni {Avx512Vnni}, AdvSimd {AdvSimd}, Dp {Dp}";
}

// Environment switches that steer the auto pick (NEEDLE_SCALAR, NEEDLE_NOVNNI, NEEDLE_NODP) and
// NEEDLE_NOEVEX, which keeps the AVX512-VNNI machine code off even when NEEDLE_TIER asks for it.
internal readonly struct TierSwitches
{
    public bool Scalar { get; init; }
    public bool NoVnni { get; init; }
    public bool NoDp { get; init; }
    public bool NoEvex { get; init; }
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
    internal static readonly CpuFeatures Detected = Detect();

    internal static readonly CpuTier Active = Select();

    internal static bool HasAvx => Avx2.IsSupported && Fma.IsSupported && Sse41.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<int> Accumulate(Vector256<int> sum, Vector256<byte> weights, Vector256<sbyte> activations)
    {
        if (Active == CpuTier.AvxVnniInt8) return AvxVnniInt8.MultiplyWideningAndAdd(sum, weights.AsSByte(), activations);
        if ((Active is CpuTier.AvxVnni or CpuTier.Avx512Vnni) && AvxVnni.IsSupported) return AvxVnni.MultiplyWideningAndAdd(sum, weights, activations);
        return Avx2.Add(sum, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), Vector256.Create((short)1)));
    }

    static CpuFeatures Detect() => new()
    {
        Avx2 = HasAvx,
        AvxVnni = HasAvx && AvxVnni.IsSupported,
        AvxVnniInt8 = HasAvx && AvxVnniInt8.IsSupported,
        Avx512 = HasAvx && Avx512F.IsSupported && Avx512F.VL.IsSupported && Avx512BW.IsSupported && Vector512.IsHardwareAccelerated,
        Avx512Vnni = HasAvx && EvexVnni.Ready,
        AdvSimd = AdvSimd.IsSupported,
        Dp = AdvSimd.IsSupported && Dp.IsSupported,
    };

    static TierSwitches Switches() => new()
    {
        Scalar = Environment.GetEnvironmentVariable("NEEDLE_SCALAR") != null,
        NoVnni = Environment.GetEnvironmentVariable("NEEDLE_NOVNNI") != null,
        NoDp = Environment.GetEnvironmentVariable("NEEDLE_NODP") != null,
        NoEvex = Environment.GetEnvironmentVariable("NEEDLE_NOEVEX") != null,
    };

    static CpuTier Select() => Resolve(Environment.GetEnvironmentVariable("NEEDLE_TIER"));

    internal static CpuTier Resolve(string? raw) => Resolve(raw, Detected, Switches());
#else
    internal const CpuTier Active = CpuTier.Scalar;
#endif

    // Pure dispatch: the tier for a CPU, the NEEDLE_TIER override and the environment switches.
    // Ranking (fastest first) is measured, see plan.md "x86 tier selection (Avx512Vnni)".
    internal static CpuTier Resolve(string? raw, CpuFeatures cpu, in TierSwitches switches)
    {
        if (switches.NoEvex) cpu = cpu with { Avx512Vnni = false };
        if (string.IsNullOrWhiteSpace(raw))
            return Auto(cpu, switches);
        if (TryParseTier(raw, out var forced) && Supported(forced, cpu))
            return forced;
        var requested = TryParseTier(raw, out var named) ? named.ToString() : "unknown tier '" + raw + "'";
        throw new InvalidOperationException(
            $"NEEDLE_TIER={raw} requests {requested}, which is unsupported on this CPU (auto-pick is {Auto(cpu, switches)}). Unset NEEDLE_TIER to run the auto-picked tier.");
    }

    internal static CpuTier Auto(in CpuFeatures cpu, in TierSwitches switches)
    {
        if (switches.Scalar) return CpuTier.Scalar;
        if (cpu.AdvSimd) return cpu.Dp && !switches.NoDp ? CpuTier.NeonDot : CpuTier.Neon;
        if (!cpu.Avx2) return CpuTier.Scalar;
        if (!switches.NoVnni)
        {
            if (cpu.AvxVnniInt8) return CpuTier.AvxVnniInt8;
            if (cpu.Avx512Vnni) return CpuTier.Avx512Vnni;
            if (cpu.AvxVnni) return CpuTier.AvxVnni;
        }
        return CpuTier.Avx2;
    }

    internal static bool Supported(CpuTier tier, in CpuFeatures cpu) => tier switch
    {
        CpuTier.Scalar => true,
        CpuTier.Avx2 => cpu.Avx2,
        CpuTier.AvxVnni => cpu.Avx2 && cpu.AvxVnni,
        CpuTier.AvxVnniInt8 => cpu.Avx2 && cpu.AvxVnniInt8,
        CpuTier.Avx512Bw => cpu.Avx2 && cpu.Avx512,
        CpuTier.Avx512Vnni => cpu.Avx2 && cpu.Avx512 && cpu.Avx512Vnni,
        CpuTier.Neon => cpu.AdvSimd,
        CpuTier.NeonDot => cpu.AdvSimd && cpu.Dp,
        _ => false,
    };


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
                case "avx512vnni": tier = CpuTier.Avx512Vnni; return true;
                case "neon": tier = CpuTier.Neon; return true;
                case "neondot": tier = CpuTier.NeonDot; return true;
            }
        }
        tier = CpuTier.Scalar;
        return false;
    }
}
