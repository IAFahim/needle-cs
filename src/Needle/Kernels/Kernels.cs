namespace Needle;

// N30: kernel table installation point (plan-netstandard.md section 2).
//
// Provider selection happens once at startup, before Model.Open:
//   1. Explicit Kernels.Install(...) (intrinsics on .NET 8+, Burst, tests).
//   2. Portable provider inside the core (a later N5x item).
//   3. Reference: the table stays empty and every site runs today's path.
//
// NOTE for provider authors (N31 etc.): this class IS `Needle.Kernels`, so
// provider source must not declare `namespace Needle.Kernels` (the type and
// the namespace would collide for consumers referencing both assemblies).
// Name provider namespaces after the provider, e.g. `Needle.BurstKernels`.
// The table is read on hot paths, so Active is a public field: reading one
// entry compiles to a single static load with no struct copy. Install once;
// entries must be installed before worker threads start (assignment is not
// otherwise synchronized, matching "installed once" in the plan).
public static class Kernels
{
    public static KernelTable Active;

    // Name recorded by Install for NeedleRuntime.Describe logs. Null means no
    // provider claimed the table (reference path).
    public static string? Provider { get; private set; }

    public static bool IsInstalled => Provider != null;

    public static void Install(in KernelTable table, string? provider = null)
    {
        Active = table;
        Provider = provider ?? "custom";
    }

    // Tests and provider swap-out. Restores exactly today's behavior.
    public static void Reset()
    {
        Active = default;
        Provider = null;
    }
}
