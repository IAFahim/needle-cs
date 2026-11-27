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

    [DllImport("kernel32", EntryPoint = "GetSystemTimes")]
    static extern int WindowsGetSystemTimes(long* idle, long* kernel, long* user);

    [DllImport("kernel32", EntryPoint = "GetThreadTimes")]
    static extern int WindowsGetThreadTimes(IntPtr thread, long* creation, long* exit, long* kernel, long* user);

    [DllImport("libc", EntryPoint = "clock_gettime")]
    static extern int LinuxClockGetTime(int clock, long* timespec);

    [DllImport("libc", EntryPoint = "open")]
    static extern int LinuxOpen(byte* path, int flags);

    [DllImport("libc", EntryPoint = "read")]
    static extern nint LinuxRead(int fd, byte* buffer, nint count);

    [DllImport("libc", EntryPoint = "close")]
    static extern int LinuxClose(int fd);

    [DllImport("libSystem", EntryPoint = "pthread_set_qos_class_self_np")]
    static extern int MacSetQosClass(uint qosClass, int relativePriority);

    public static int CurrentThreadId()
    {
        try { return IsLinux ? LinuxThreadId() : Environment.CurrentManagedThreadId; }
        catch (Exception) { return Environment.CurrentManagedThreadId; }
    }

    /// <summary>
    /// Idle CPU time of the whole machine, summed over every logical CPU, in seconds since boot
    /// (Windows: GetSystemTimes; Linux: /proc/stat idle plus iowait). False where the OS does not
    /// say (macOS, or a sandbox that hides /proc/stat). No allocation.
    /// </summary>
    public static bool TrySystemIdleSeconds(out double seconds)
    {
        seconds = 0;
        try
        {
            if (IsWindows)
            {
                long idle, kernel, user;
                if (WindowsGetSystemTimes(&idle, &kernel, &user) == 0) return false;
                seconds = idle * 1e-7;
                return true;
            }
            if (IsLinux) return TryLinuxIdle(out seconds);
        }
        catch (Exception) { }
        return false;
    }

    /// <summary>CPU time the calling thread has used, in 100 ns units; false where not known. No allocation.</summary>
    public static bool TryCurrentThreadCpu(out long ticks)
    {
        ticks = 0;
        try
        {
            if (IsWindows)
            {
                long creation, exit, kernel, user;
                if (WindowsGetThreadTimes(WindowsCurrentThread(), &creation, &exit, &kernel, &user) == 0) return false;
                ticks = kernel + user;
                return true;
            }
            if (IsLinux)
            {
                const int ClockThreadCpuTime = 3;
                var timespec = stackalloc long[2];
                if (LinuxClockGetTime(ClockThreadCpuTime, timespec) != 0) return false;
                ticks = timespec[0] * 10_000_000 + timespec[1] / 100;
                return true;
            }
        }
        catch (Exception) { }
        return false;
    }

    static bool TryLinuxIdle(out double seconds)
    {
        seconds = 0;
        var path = stackalloc byte[] { (byte)'/', (byte)'p', (byte)'r', (byte)'o', (byte)'c', (byte)'/', (byte)'s', (byte)'t', (byte)'a', (byte)'t', 0 };
        var fd = LinuxOpen(path, 0);
        if (fd < 0) return false;
        const int Size = 256;
        var buffer = stackalloc byte[Size];
        var length = LinuxRead(fd, buffer, Size);
        LinuxClose(fd);
        // First line: "cpu  user nice system idle iowait ..." in clock ticks (USER_HZ, 100 per second).
        if (length < 8 || buffer[0] != (byte)'c' || buffer[1] != (byte)'p' || buffer[2] != (byte)'u' || buffer[3] != (byte)' ') return false;
        var field = 0;
        long idle = 0;
        var i = 4;
        while (i < length && buffer[i] != (byte)'\n' && field < 5)
        {
            while (i < length && buffer[i] == (byte)' ') i++;
            long value = 0;
            var digits = 0;
            while (i < length && buffer[i] >= (byte)'0' && buffer[i] <= (byte)'9') { value = value * 10 + (buffer[i] - (byte)'0'); i++; digits++; }
            if (digits == 0) return false;
            if (field == 3 || field == 4) idle += value;
            field++;
        }
        if (field < 5) return false;
        seconds = idle * 0.01;
        return true;
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
