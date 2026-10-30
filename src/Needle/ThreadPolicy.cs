using System;

namespace Needle;

public enum WorkerPriority : byte
{
    Normal,
    BelowNormal,
    Lowest,
    AboveNormal,
}

public enum CorePlacement : byte
{
    Any,
    PerformanceCores,
    EfficiencyCores,
    Explicit,
}

public enum CallerRole : byte
{
    Participates,
    Waits,
}

public readonly struct ThreadPolicy
{
    public const int DefaultSpinMicroseconds = 300;
    public const int GameSpinMicroseconds = 50;
    public const int MaxExplicitCpu = 63;

    readonly int threadCount;

    public int Threads => Math.Max(1, threadCount);
    public int SpinMicroseconds { get; }
    public WorkerPriority Priority { get; }
    public CorePlacement Placement { get; }
    public CallerRole Caller { get; }
    public ulong ExplicitCpus { get; }

    public ThreadPolicy(int threads, int spinMicroseconds, WorkerPriority priority, CorePlacement placement, CallerRole caller, ulong explicitCpus)
    {
        threadCount = Math.Max(1, threads);
        SpinMicroseconds = Math.Max(0, spinMicroseconds);
        Priority = priority;
        Placement = placement == CorePlacement.Explicit && explicitCpus == 0 ? CorePlacement.Any : placement;
        Caller = caller;
        ExplicitCpus = explicitCpus;
    }

    public static ThreadPolicy Default => new(1, DefaultSpinMicroseconds, WorkerPriority.Normal, CorePlacement.PerformanceCores, CallerRole.Participates, 0);

    public static ThreadPolicy Game => new(1, GameSpinMicroseconds, WorkerPriority.BelowNormal, CorePlacement.Any, CallerRole.Participates, 0);

    public int Workers => Caller == CallerRole.Participates ? Math.Max(0, Threads - 1) : Threads;

    public ThreadPolicy WithThreads(int threads) => new(threads, SpinMicroseconds, Priority, Placement, Caller, ExplicitCpus);

    public ThreadPolicy WithSpinMicroseconds(int spinMicroseconds) => new(Threads, spinMicroseconds, Priority, Placement, Caller, ExplicitCpus);

    public ThreadPolicy WithPriority(WorkerPriority priority) => new(Threads, SpinMicroseconds, priority, Placement, Caller, ExplicitCpus);

    public ThreadPolicy WithPlacement(CorePlacement placement) => new(Threads, SpinMicroseconds, Priority, placement, Caller, ExplicitCpus);

    public ThreadPolicy WithCaller(CallerRole caller) => new(Threads, SpinMicroseconds, Priority, Placement, caller, ExplicitCpus);

    public ThreadPolicy WithCpus(params int[] cpus)
    {
        var mask = 0UL;
        foreach (var cpu in cpus)
            if ((uint)cpu <= MaxExplicitCpu) mask |= 1UL << cpu;
        return new ThreadPolicy(Threads, SpinMicroseconds, Priority, CorePlacement.Explicit, Caller, mask);
    }
}

public readonly struct FrameBudget
{
    public int MaxTokensPerSlice { get; }
    public double MaxMillisecondsPerSlice { get; }
    public bool IsUnlimited => MaxTokensPerSlice <= 0 && MaxMillisecondsPerSlice <= 0;
    public static FrameBudget Unlimited => default;
    public FrameBudget(int maxTokensPerSlice, double maxMillisecondsPerSlice)
    {
        MaxTokensPerSlice = Math.Max(0, maxTokensPerSlice);
        MaxMillisecondsPerSlice = Math.Max(0, maxMillisecondsPerSlice);
    }
    public static FrameBudget FromTokens(int maxTokensPerSlice) => new(maxTokensPerSlice, 0);
    public static FrameBudget FromMilliseconds(double maxMillisecondsPerSlice) => new(0, maxMillisecondsPerSlice);
    public FrameBudget WithMaxTokensPerSlice(int maxTokensPerSlice) => new(maxTokensPerSlice, MaxMillisecondsPerSlice);
    public FrameBudget WithMaxMillisecondsPerSlice(double maxMillisecondsPerSlice) => new(MaxTokensPerSlice, maxMillisecondsPerSlice);
}
