using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Needle;

internal static unsafe class OsThreads
{
    const int CpuMaskBits = 1024;
    const int MaskWords = CpuMaskBits / (8 * sizeof(ulong));
    const int LinuxProcessScope = 0;
    const int WindowsPriorityLowest = -2;
    const int WindowsPriorityBelowNormal = -1;
    const int WindowsPriorityAboveNormal = 1;
    const int WindowsPowerThrottlingClass = 4;
    const uint WindowsExecutionSpeedThrottling = 1;
    const uint MacQosUserInitiated = 0x19;
    const uint MacQosUtility = 0x11;
    const uint MacQosBackground = 0x09;

    public static readonly bool IsLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux);
    public static readonly bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
    public static readonly bool IsMac = RuntimeInformation.IsOSPlatform(OSPlatform.OSX);

    [DllImport("libc", EntryPoint = "sched_setaffinity")]
    static extern int SchedSetAffinity(int thread, nuint bytes, ulong* mask);

    [DllImport("libc", EntryPoint = "sched_getaffinity")]
    static extern int SchedGetAffinity(int thread, nuint bytes, ulong* mask);

    [DllImport("libc", EntryPoint = "gettid")]
    static extern int LinuxThreadId();

    [DllImport("libc", EntryPoint = "setpriority")]
    static extern int LinuxSetPriority(int scope, uint thread, int nice);

    [DllImport("kernel32", EntryPoint = "GetCurrentThread")]
    static extern IntPtr WindowsCurrentThread();

    [DllImport("kernel32", EntryPoint = "SetThreadPriority")]
    static extern int WindowsSetThreadPriority(IntPtr thread, int priority);

    [DllImport("kernel32", EntryPoint = "SetThreadSelectedCpuSets")]
    static extern int WindowsSetSelectedCpuSets(IntPtr thread, uint* ids, uint count);

    [DllImport("kernel32", EntryPoint = "SetThreadInformation")]
    static extern int WindowsSetThreadInformation(IntPtr thread, int informationClass, uint* information, uint bytes);

    [DllImport("kernel32", EntryPoint = "GetSystemCpuSetInformation")]
    static extern int WindowsGetCpuSetInformation(byte* buffer, uint bytes, uint* returned, IntPtr process, uint flags);

    [DllImport("libSystem", EntryPoint = "pthread_set_qos_class_self_np")]
    static extern int MacSetQosClass(uint qosClass, int relativePriority);

    public static int CurrentThreadId()
    {
        try { return IsLinux ? LinuxThreadId() : Environment.CurrentManagedThreadId; }
        catch (Exception) { return Environment.CurrentManagedThreadId; }
    }

    public static int[]? AllowedLinuxCpus()
    {
        if (!IsLinux) return null;
        try
        {
            var mask = stackalloc ulong[MaskWords];
            if (SchedGetAffinity(0, CpuMaskBits / 8, mask) != 0) return null;
            var cpus = new List<int>();
            for (var cpu = 0; cpu < CpuMaskBits; cpu++)
                if ((mask[cpu / 64] >> (cpu % 64) & 1) != 0) cpus.Add(cpu);
            return cpus.ToArray();
        }
        catch (Exception) { return null; }
    }

    public static bool TryReadWindowsCpuSets(out byte[] buffer)
    {
        buffer = Array.Empty<byte>();
        if (!IsWindows) return false;
        try
        {
            uint needed = 0;
            WindowsGetCpuSetInformation(null, 0, &needed, IntPtr.Zero, 0);
            if (needed == 0) return false;
            var bytes = new byte[needed];
            fixed (byte* pinned = bytes)
                if (WindowsGetCpuSetInformation(pinned, needed, &needed, IntPtr.Zero, 0) == 0) return false;
            buffer = bytes;
            return true;
        }
        catch (Exception) { return false; }
    }

    public static void ConfigureCurrentThread(WorkerPriority priority, CorePlacement placement, int cpu)
    {
        try
        {
            if (IsLinux) ConfigureLinux(priority, cpu);
            else if (IsWindows) ConfigureWindows(priority, placement, cpu);
            else if (IsMac) ConfigureMac(priority, placement);
        }
        catch (Exception) { }
    }

    static void ConfigureLinux(WorkerPriority priority, int cpu)
    {
        if (cpu >= 0 && cpu < CpuMaskBits)
        {
            var mask = stackalloc ulong[MaskWords];
            for (var i = 0; i < MaskWords; i++) mask[i] = 0;
            mask[cpu / 64] = 1UL << (cpu % 64);
            SchedSetAffinity(0, CpuMaskBits / 8, mask);
        }
        var nice = priority switch
        {
            WorkerPriority.AboveNormal => -5,
            WorkerPriority.BelowNormal => 5,
            WorkerPriority.Lowest => 19,
            _ => 0,
        };
        if (nice != 0) LinuxSetPriority(LinuxProcessScope, (uint)LinuxThreadId(), nice);
    }

    static void ConfigureWindows(WorkerPriority priority, CorePlacement placement, int cpuSetId)
    {
        var thread = WindowsCurrentThread();
        if (cpuSetId >= 0)
        {
            var id = (uint)cpuSetId;
            WindowsSetSelectedCpuSets(thread, &id, 1);
        }
        var level = priority switch
        {
            WorkerPriority.AboveNormal => WindowsPriorityAboveNormal,
            WorkerPriority.BelowNormal => WindowsPriorityBelowNormal,
            WorkerPriority.Lowest => WindowsPriorityLowest,
            _ => 0,
        };
        if (level != 0) WindowsSetThreadPriority(thread, level);
        if (placement != CorePlacement.EfficiencyCores && priority is WorkerPriority.Normal or WorkerPriority.AboveNormal)
        {
            var state = stackalloc uint[3];
            state[0] = 1;
            state[1] = WindowsExecutionSpeedThrottling;
            state[2] = 0;
            WindowsSetThreadInformation(thread, WindowsPowerThrottlingClass, state, 3 * sizeof(uint));
        }
    }

    static void ConfigureMac(WorkerPriority priority, CorePlacement placement)
    {
        var qos = priority switch
        {
            WorkerPriority.AboveNormal => MacQosUserInitiated,
            WorkerPriority.BelowNormal => MacQosUtility,
            WorkerPriority.Lowest => MacQosBackground,
            _ => placement == CorePlacement.PerformanceCores ? MacQosUserInitiated : 0u,
        };
        if (placement == CorePlacement.EfficiencyCores) qos = MacQosBackground;
        if (qos != 0) MacSetQosClass(qos, 0);
    }
}
