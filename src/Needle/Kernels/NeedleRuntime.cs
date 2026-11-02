using System;
using System.Runtime.InteropServices;

namespace Needle;

// N30: one-line runtime report for startup logs and bug reports, mirroring
// `needle cpu` (CpuInfo). Shows the SIMD tier pick plus which kernel provider
// claimed the table, so a Unity player log line is enough to tell whether the
// fast path is active.
public static class NeedleRuntime
{
    public static string Describe()
    {
        var provider = Kernels.Provider ?? "none";
        var table = Kernels.Active.IsEmpty ? "empty" : "installed";
        return $"tier={Tiers.Active} provider={provider} table={table} arch={RuntimeInformation.ProcessArchitecture} threads={Environment.ProcessorCount}";
    }
}
