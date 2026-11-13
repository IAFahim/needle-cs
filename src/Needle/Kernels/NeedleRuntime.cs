using System;
using System.Runtime.InteropServices;

namespace Needle;

public static class NeedleRuntime
{
    public static string Describe()
    {
        var provider = Kernels.Provider ?? "none";
        var table = Kernels.Active.IsEmpty ? "empty" : "installed";
        return $"tier={Tiers.Active} provider={provider} table={table} arch={RuntimeInformation.ProcessArchitecture} threads={Environment.ProcessorCount}";
    }
}
