using System;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using Needle;

namespace Needle.Cli;

/// <summary>`needle cpu`: which kernel tier this process picked and why, for bug reports and CI logs.</summary>
internal static class CpuInfo
{
    public static int Run()
    {
        Console.WriteLine($"arch        : {RuntimeInformation.ProcessArchitecture} on {RuntimeInformation.OSDescription}");
        Console.WriteLine($"processors  : {Environment.ProcessorCount}");
        Console.WriteLine($"active tier : {Tiers.Active}");
        Console.WriteLine($"vector512   : {Vector512.IsHardwareAccelerated} (Avx512F {Avx512F.IsSupported}, Avx512BW {Avx512BW.IsSupported}, Avx512Vbmi {Avx512Vbmi.IsSupported})");
        Console.WriteLine($"x86 int8 dot: Avx2 {Avx2.IsSupported}, Fma {Fma.IsSupported}, AvxVnni {AvxVnni.IsSupported}, AvxVnniInt8 {AvxVnniInt8.IsSupported}");
        Console.WriteLine($"arm         : AdvSimd {AdvSimd.IsSupported}, Dp {Dp.IsSupported}");
        foreach (var name in new[] { "NEEDLE_TIER", "NEEDLE_SCALAR", "NEEDLE_NOVNNI", "NEEDLE_NOAVX512", "NEEDLE_NODP", "DOTNET_EnableAVX512F", "DOTNET_EnableAVX2", "DOTNET_PROCESSOR_COUNT" })
            if (Environment.GetEnvironmentVariable(name) is { } value) Console.WriteLine($"env         : {name}={value}");
        return 0;
    }
}
