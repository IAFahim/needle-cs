using System;
using System.Collections.Generic;
using System.IO;

namespace Needle;

internal readonly struct CoreInfo
{
    public readonly int Id;
    public readonly int SmtRank;
    public readonly uint Strength;
    public readonly bool Performance;

    public CoreInfo(int id, int smtRank, uint strength, bool performance)
    {
        Id = id;
        SmtRank = smtRank;
        Strength = strength;
        Performance = performance;
    }
}

internal static class Topology
{
    const int MaxCpus = 1024;
    const double HybridGapRatio = 1.15;
    const int WindowsCpuSetRecordType = 0;
    const int WindowsCpuSetIdOffset = 8;
    const int WindowsCoreIndexOffset = 15;
    const int WindowsEfficiencyClassOffset = 18;
    const int WindowsMinimumRecordSize = 24;

    static CoreInfo[]? discovered;

    public static int[] Select(in ThreadPolicy policy)
    {
        if (Environment.GetEnvironmentVariable("NEEDLE_PIN") == "0") return Array.Empty<int>();
        return policy.Placement switch
        {
            CorePlacement.PerformanceCores => TakeBest(Allowed(Cores(), onlyPerformance: true), policy.Threads),
            CorePlacement.EfficiencyCores => TakeBest(Allowed(Cores(), onlyPerformance: false), policy.Threads),
            CorePlacement.Explicit => ExplicitIds(policy.ExplicitCpus),
            _ => Array.Empty<int>(),
        };
    }

    static CoreInfo[] Cores()
    {
        discovered ??= Discover();
        return discovered;
    }

    static CoreInfo[] Discover()
    {
        if (OsThreads.IsLinux) return DiscoverLinux();
        if (OsThreads.IsWindows) return DiscoverWindows();
        return Array.Empty<CoreInfo>();
    }

    static List<CoreInfo> Allowed(CoreInfo[] cores, bool onlyPerformance)
    {
        var allowed = OsThreads.AllowedLinuxCpus();
        var picked = new List<CoreInfo>(cores.Length);
        foreach (var core in cores)
        {
            if (core.Performance != onlyPerformance) continue;
            if (allowed != null && Array.IndexOf(allowed, core.Id) < 0) continue;
            picked.Add(core);
        }
        picked.Sort(Compare);
        return picked;
    }

    static int Compare(CoreInfo a, CoreInfo b)
    {
        if (a.SmtRank != b.SmtRank) return a.SmtRank - b.SmtRank;
        if (a.Strength != b.Strength) return a.Strength > b.Strength ? -1 : 1;
        return a.Id - b.Id;
    }

    static int[] TakeBest(List<CoreInfo> ordered, int count)
    {
        var ids = new int[Math.Min(count, ordered.Count)];
        for (var i = 0; i < ids.Length; i++) ids[i] = ordered[i].Id;
        return ids;
    }

    static int[] ExplicitIds(ulong mask)
    {
        var allowed = OsThreads.AllowedLinuxCpus();
        var ids = new List<int>();
        for (var cpu = 0; cpu <= ThreadPolicy.MaxExplicitCpu; cpu++)
        {
            if ((mask >> cpu & 1) == 0) continue;
            if (allowed != null && Array.IndexOf(allowed, cpu) < 0) continue;
            ids.Add(cpu);
        }
        return ids.ToArray();
    }

    static CoreInfo[] DiscoverLinux()
    {
        int[] cpus;
        try { cpus = ParseList(File.ReadAllText("/sys/devices/system/cpu/online")); }
        catch (Exception) { return Array.Empty<CoreInfo>(); }
        if (cpus.Length == 0 || cpus.Length > MaxCpus) return Array.Empty<CoreInfo>();
        var strengths = new uint[cpus.Length];
        var ranks = new int[cpus.Length];
        for (var i = 0; i < cpus.Length; i++)
        {
            var sys = $"/sys/devices/system/cpu/cpu{cpus[i]}";
            var capacity = ReadUint($"{sys}/cpu_capacity");
            strengths[i] = capacity != 0 ? capacity : ReadUint($"{sys}/cpufreq/cpuinfo_max_freq");
            ranks[i] = ReadSmtRank(sys, cpus[i]);
        }
        var floor = PerformanceFloor(strengths);
        var cores = new CoreInfo[cpus.Length];
        for (var i = 0; i < cpus.Length; i++) cores[i] = new CoreInfo(cpus[i], ranks[i], strengths[i], strengths[i] >= floor);
        return cores;
    }

    static int ReadSmtRank(string sys, int cpu)
    {
        try
        {
            var rank = 0;
            foreach (var member in ParseList(File.ReadAllText($"{sys}/topology/thread_siblings_list")))
                if (member == cpu) break;
                else rank++;
            return rank;
        }
        catch (Exception) { return 0; }
    }

    internal static uint PerformanceFloor(uint[] strengths)
    {
        var distinct = new List<uint>();
        foreach (var strength in strengths)
            if (!distinct.Contains(strength)) distinct.Add(strength);
        distinct.Sort();
        var floor = 0u;
        var widest = HybridGapRatio;
        for (var i = 1; i < distinct.Count; i++)
        {
            if (distinct[i - 1] == 0) continue;
            var ratio = (double)distinct[i] / distinct[i - 1];
            if (ratio < widest) continue;
            widest = ratio;
            floor = distinct[i];
        }
        return floor;
    }

    static CoreInfo[] DiscoverWindows()
    {
        if (!OsThreads.TryReadWindowsCpuSets(out var buffer)) return Array.Empty<CoreInfo>();
        var records = new List<(int Id, int Core, byte Efficiency)>();
        for (var offset = 0; offset + WindowsMinimumRecordSize <= buffer.Length;)
        {
            var size = BitConverter.ToInt32(buffer, offset);
            if (size < WindowsMinimumRecordSize || offset + size > buffer.Length) break;
            if (BitConverter.ToInt32(buffer, offset + 4) == WindowsCpuSetRecordType)
                records.Add((BitConverter.ToInt32(buffer, offset + WindowsCpuSetIdOffset), buffer[offset + WindowsCoreIndexOffset], buffer[offset + WindowsEfficiencyClassOffset]));
            offset += size;
        }
        var best = 0;
        foreach (var record in records) best = Math.Max(best, record.Efficiency);
        var seen = new Dictionary<int, int>();
        var cores = new CoreInfo[records.Count];
        for (var i = 0; i < records.Count; i++)
        {
            seen.TryGetValue(records[i].Core, out var rank);
            seen[records[i].Core] = rank + 1;
            cores[i] = new CoreInfo(records[i].Id, rank, records[i].Efficiency, records[i].Efficiency == best);
        }
        return cores;
    }

    static uint ReadUint(string path)
    {
        try { return uint.Parse(File.ReadAllText(path).Trim()); }
        catch (Exception) { return 0; }
    }

    internal static int[] ParseList(string text)
    {
        var values = new List<int>();
        foreach (var raw in text.Split(','))
        {
            var part = raw.Trim();
            if (part.Length == 0) continue;
            var range = part.Split('-');
            if (range.Length == 2 && int.TryParse(range[0], out var from) && int.TryParse(range[1], out var to))
                for (var c = from; c <= to && values.Count < MaxCpus; c++) values.Add(c);
            else if (int.TryParse(part, out var single)) values.Add(single);
        }
        return values.ToArray();
    }
}
