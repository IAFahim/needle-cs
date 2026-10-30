using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

namespace Needle;

internal sealed unsafe class Pool : IDisposable
{
    const int CacheLine = 64;

    [StructLayout(LayoutKind.Explicit, Size = 4 * CacheLine)]
    struct Control
    {
        [FieldOffset(0)] public long Ticket;
        [FieldOffset(CacheLine)] public int Done;
        [FieldOffset(2 * CacheLine)] public delegate*<void*, int, void> Function;
        [FieldOffset(2 * CacheLine + 8)] public void* Context;
        [FieldOffset(2 * CacheLine + 16)] public int Count;
        [FieldOffset(3 * CacheLine)] public int Sleeping;
        [FieldOffset(3 * CacheLine + 4)] public int Stopping;
        [FieldOffset(3 * CacheLine + 8)] public int Paused;
    }

    readonly Control* control;
    readonly Thread[] workers;
    readonly ThreadPolicy policy;
    readonly int[] nativeWorkerIds;
    readonly long spinTicks;
    readonly object gate = new();

    public int Threads => policy.Threads;

    public int[] NativeWorkerIds => nativeWorkerIds;

    public bool IsPaused => Volatile.Read(ref control->Paused) != 0;

    public Pool(int threads) : this(ThreadPolicy.Default.WithThreads(threads))
    {
    }

    public Pool(ThreadPolicy requested)
    {
        policy = WithDiagnosticOverrides(requested);
        spinTicks = policy.SpinMicroseconds * Stopwatch.Frequency / 1_000_000;
        control = (Control*)Mem.AlignedAlloc((nuint)sizeof(Control), CacheLine);
        Mem.Clear(control, (nuint)sizeof(Control));
        var cores = Topology.Select(policy);
        var callerSlots = policy.Caller == CallerRole.Participates && policy.Placement != CorePlacement.Explicit ? 1 : 0;
        workers = new Thread[policy.Workers];
        nativeWorkerIds = new int[workers.Length];
        for (var i = 0; i < workers.Length; i++)
        {
            var index = i;
            var cpu = cores.Length > 0 ? cores[(i + callerSlots) % cores.Length] : -1;
            workers[i] = new Thread(() => Work(index, cpu)) { IsBackground = true, Name = $"needle-{i}" };
            workers[i].Start();
        }
    }

    static ThreadPolicy WithDiagnosticOverrides(ThreadPolicy requested)
    {
        var spin = Environment.GetEnvironmentVariable("NEEDLE_SPIN_US");
        return spin != null && int.TryParse(spin, out var microseconds) ? requested.WithSpinMicroseconds(microseconds) : requested;
    }

    public void Pause() => Volatile.Write(ref control->Paused, 1);

    public void Resume()
    {
        Volatile.Write(ref control->Paused, 0);
        WakeAll();
    }

    public void Run(delegate*<void*, int, void> function, void* context, int count)
    {
        var runsOnCaller = policy.Caller == CallerRole.Participates && count <= 1;
        if (workers.Length == 0 || runsOnCaller || IsPaused)
        {
            for (var i = 0; i < count; i++) function(context, i);
            return;
        }
        var generation = (Volatile.Read(ref control->Ticket) >> 32) + 1;
        Interlocked.Exchange(ref control->Ticket, (generation << 32) | int.MaxValue);
        control->Function = function;
        control->Context = context;
        control->Count = count;
        Volatile.Write(ref control->Done, 0);
        Interlocked.Exchange(ref control->Ticket, generation << 32);
        if (Volatile.Read(ref control->Sleeping) > 0) WakeAll();
        if (policy.Caller == CallerRole.Participates) Drain();
        var spin = new SpinWait();
        while (Volatile.Read(ref control->Done) < count)
#if NETSTANDARD2_1
            spin.SpinOnce();
#else
            spin.SpinOnce(-1);
#endif
    }

    void WakeAll()
    {
        lock (gate) Monitor.PulseAll(gate);
    }

    bool Drain()
    {
        var claimed = false;
        for (;;)
        {
            var ticket = Volatile.Read(ref control->Ticket);
            var index = (int)ticket;
            if (index >= control->Count) return claimed;
            if (Interlocked.CompareExchange(ref control->Ticket, ticket + 1, ticket) != ticket) continue;
            control->Function(control->Context, index);
            Interlocked.Increment(ref control->Done);
            claimed = true;
        }
    }

    bool HasWork() => (int)Volatile.Read(ref control->Ticket) < control->Count;

    void Park()
    {
        lock (gate)
        {
            Interlocked.Increment(ref control->Sleeping);
            while (Volatile.Read(ref control->Stopping) == 0 && (IsPaused || !HasWork())) Monitor.Wait(gate);
            Interlocked.Decrement(ref control->Sleeping);
        }
    }

    void Work(int index, int cpu)
    {
        Volatile.Write(ref nativeWorkerIds[index], OsThreads.CurrentThreadId());
        OsThreads.ConfigureCurrentThread(policy.Priority, policy.Placement, cpu);
        var deadline = Stopwatch.GetTimestamp() + spinTicks;
        while (Volatile.Read(ref control->Stopping) == 0)
        {
            var paused = IsPaused;
            if (!paused && Drain()) { deadline = Stopwatch.GetTimestamp() + spinTicks; continue; }
            if (!paused && Stopwatch.GetTimestamp() < deadline) { Thread.SpinWait(4); continue; }
            Park();
            deadline = Stopwatch.GetTimestamp() + spinTicks;
        }
    }

    public void Dispose()
    {
        Volatile.Write(ref control->Stopping, 1);
        WakeAll();
        foreach (var worker in workers) worker.Join();
        Mem.AlignedFree(control);
    }
}
