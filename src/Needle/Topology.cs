using System;
using System.IO;
using System.Runtime.InteropServices;

namespace Needle;

#if NETSTANDARD2_1
internal static unsafe partial class Libc
{
    public static int[] FastCores(int count) => Array.Empty<int>();

    public static bool PinToCpu(int cpu) => false;
}
#else
internal static unsafe partial class Libc
{
    const int MaxCpus = 1024;

    [LibraryImport("libc", EntryPoint = "sched_setaffinity", SetLastError = true)]
    private static partial int SchedSetAffinity(int pid, nuint bytes, nuint* mask);

    static int[]? fastCores;

    public static int[] FastCores(int count)
    {
        if (Environment.GetEnvironmentVariable("NEEDLE_PIN") == "0" || count <= 0) return Array.Empty<int>();
        if (fastCores == null) fastCores = DiscoverCores();
        var cores = fastCores;
        if (cores.Length == 0) return Array.Empty<int>();
        if (count >= cores.Length) return cores;
        var picked = new int[count];
        Array.Copy(cores, picked, count);
        return picked;
    }

    public static bool PinToCpu(int cpu)
    {
        if ((uint)cpu >= MaxCpus || !OperatingSystem.IsLinux()) return false;
        var mask = stackalloc nuint[MaxCpus / (8 * sizeof(nuint))];
        mask[cpu / (8 * sizeof(nuint))] = (nuint)1 << (cpu % (8 * sizeof(nuint)));
        return SchedSetAffinity(0, MaxCpus / 8, mask) == 0;
    }

    static int[] DiscoverCores()
    {
        if (!OperatingSystem.IsLinux()) return Array.Empty<int>();
        int[] cpus;
        try
        {
            var online = File.ReadAllText("/sys/devices/system/cpu/online");
            cpus = ParseList(online);
        }
        catch { return Array.Empty<int>(); }
        if (cpus.Length == 0 || cpus.Length > MaxCpus) return Array.Empty<int>();
        var order = new (int Cpu, uint Freq, int Rank)[cpus.Length];
        for (var i = 0; i < cpus.Length; i++)
        {
            var cpu = cpus[i];
            var sys = $"/sys/devices/system/cpu/cpu{cpu}";
            var freq = ReadUint($"{sys}/cpufreq/cpuinfo_max_freq");
            var rank = 0;
            try
            {
                var siblings = File.ReadAllText($"{sys}/topology/thread_siblings_list");
                foreach (var member in ParseList(siblings))
                    if (member == cpu) break;
                    else rank++;
            }
            catch { }
            order[i] = (cpu, freq, rank);
        }
        Array.Sort(order, (a, b) =>
        {
            if (a.Rank != b.Rank) return a.Rank - b.Rank;
            var byFreq = (a.Freq > b.Freq) ? -1 : (a.Freq < b.Freq) ? 1 : 0;
            return byFreq != 0 ? byFreq : a.Cpu - b.Cpu;
        });
        var result = new int[order.Length];
        for (var i = 0; i < order.Length; i++) result[i] = order[i].Cpu;
        return result;
    }

    static uint ReadUint(string path)
    {
        try { return uint.Parse(File.ReadAllText(path).AsSpan().TrimEnd()); }
        catch { return 0; }
    }

    static int[] ParseList(string text)
    {
        var values = new System.Collections.Generic.List<int>();
        foreach (var part in text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var range = part.Split('-');
            if (range.Length == 2 && int.TryParse(range[0], out var from) && int.TryParse(range[1], out var to))
                for (var c = from; c <= to && values.Count < MaxCpus; c++) values.Add(c);
            else if (int.TryParse(part, out var single)) values.Add(single);
        }
        return values.ToArray();
    }
}
#endif
