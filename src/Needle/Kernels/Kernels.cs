using System;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Needle;

public static class Kernels
{
#if !NEEDLE_SIMD
    static readonly object InitGate = new();
    static volatile bool defaultResolved;
#endif

    internal static nint SidecarDecodeTeam;
    internal static nint SidecarPrefillTeam;

    public static KernelTable Active;

    public static string? Provider { get; private set; }

    public static bool IsInstalled => Provider != null;

    public static void Install(in KernelTable table, string? provider = null)
    {
        Active = table;
        Provider = provider ?? "custom";
        if (Provider != "net10-unmanaged")
        {
            SidecarDecodeTeam = 0;
            SidecarPrefillTeam = 0;
        }
    }

    public static void Reset()
    {
        Active = default;
        Provider = null;
        SidecarDecodeTeam = 0;
        SidecarPrefillTeam = 0;
#if !NEEDLE_SIMD
        defaultResolved = false;
#endif
    }

    public static bool TryInstallSidecar()
    {
#if NEEDLE_SIMD
        return false;
#else
        if (Environment.GetEnvironmentVariable("NEEDLE_FORCE_BURST") != null) return false;
        if (Provider == "portable") Reset();
        if (IsInstalled || !Active.IsEmpty) return Provider == "net10-unmanaged";
        lock (InitGate)
        {
            if (Provider == "portable") Reset();
            if (IsInstalled || !Active.IsEmpty) return Provider == "net10-unmanaged";
            if (TryInstallNet10Sidecar())
            {
                defaultResolved = true;
                return true;
            }
            return false;
        }
#endif
    }

    internal static unsafe void EnsureDefault()
    {
#if !NEEDLE_SIMD
        if (defaultResolved || !Active.IsEmpty || IsInstalled) return;
        lock (InitGate)
        {
            if (defaultResolved || !Active.IsEmpty || IsInstalled) return;
            defaultResolved = true;
            if (Environment.GetEnvironmentVariable("NEEDLE_SCALAR") != null) return;
            if (TryInstallNet10Sidecar()) return;
            PortableKernels.TryInstall();
        }
#endif
    }

#if !NEEDLE_SIMD
    static unsafe bool TryInstallNet10Sidecar()
    {
        if (Environment.Version.Major < 8) return false;
        if (Environment.GetEnvironmentVariable("NEEDLE_FORCE_BURST") != null) return false;
        try
        {
            Assembly? asm = null;
            foreach (var loaded in AppDomain.CurrentDomain.GetAssemblies())
            {
                if (string.Equals(loaded.GetName().Name, "Needle.Inference.Net10", StringComparison.OrdinalIgnoreCase))
                {
                    asm = loaded;
                    break;
                }
            }
            if (asm == null)
            {
                var location = typeof(Kernels).Assembly.Location;
                var asmDir = string.IsNullOrEmpty(location) ? null : Path.GetDirectoryName(location);
                var baseDir = AppContext.BaseDirectory;
                string?[] candidates =
                {
                    asmDir != null ? Path.Combine(asmDir, "Needle.Inference.Net10.dll") : null,
                    !string.IsNullOrEmpty(baseDir) ? Path.Combine(baseDir, "Needle.Inference.Net10.dll") : null,
                    asmDir != null ? Path.Combine(asmDir, "..", "net10.0", "Needle.Inference.Net10.dll") : null,
                    Path.GetFullPath(Path.Combine("Packages", "com.needle.inference", "Runtime", "Plugins", "net10.0", "Needle.Inference.Net10.dll")),
                };
                for (var i = 0; i < candidates.Length; i++)
                {
                    var path = candidates[i];
                    if (string.IsNullOrEmpty(path) || !File.Exists(path)) continue;
                    asm = Assembly.LoadFrom(path);
                    break;
                }
            }
            if (asm == null) return false;
            var exports = asm.GetType("Needle.Net10KernelExports", false);
            var fill = exports?.GetMethod("Fill", BindingFlags.Public | BindingFlags.Static);
            if (fill == null) return false;
            KernelTable table = default;
            fill.Invoke(null, new object[] { (nint)(&table), KernelTable.FieldCount });
            if (table.IsEmpty) return false;
            Install(table, "net10-unmanaged");
            if (Environment.GetEnvironmentVariable("NEEDLE_NO_COARSE_SIDECAR") == null)
            {
                var fillTeam = exports?.GetMethod("FillTeam", BindingFlags.Public | BindingFlags.Static);
                if (fillTeam != null)
                {
                    var slots = stackalloc nint[2];
                    slots[0] = 0;
                    slots[1] = 0;
                    fillTeam.Invoke(null, new object[] { (nint)slots, Unsafe.SizeOf<EngineCore<NoClock>.DecodeJob>() });
                    SidecarDecodeTeam = slots[0];
                    SidecarPrefillTeam = slots[1];
                }
            }
            return true;
        }
        catch
        {
            return false;
        }
    }
#endif
}
