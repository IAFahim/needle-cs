using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

namespace Needle;

[StructLayout(LayoutKind.Explicit, Size = 16)]
internal struct TeamBarrier
{
    [FieldOffset(0)] public long Phase;
    [FieldOffset(8)] public int Arrived;
    [FieldOffset(12)] public int Participants;
}

internal sealed unsafe class Pool : IDisposable
{
    const int CacheLine = 64;

    [StructLayout(LayoutKind.Explicit, Size = 6 * CacheLine)]
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
        [FieldOffset(4 * CacheLine)] public long TeamTicket;
        [FieldOffset(4 * CacheLine + 8)] public delegate*<void*, int, void> TeamFunction;
        [FieldOffset(4 * CacheLine + 16)] public void* TeamContext;
        [FieldOffset(4 * CacheLine + 24)] public int TeamDone;
        [FieldOffset(4 * CacheLine + 28)] public int TeamActive;
    }

    readonly Control* control;
    readonly Thread[] workers;
    readonly ThreadPolicy policy;
    readonly int[] nativeWorkerIds;
    readonly long spinTicks;
    readonly object gate = new();

    public int Threads => policy.Threads;

    public int Members => workers.Length + (policy.Caller == CallerRole.Participates ? 1 : 0);

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

    public void RunTeam(delegate*<void*, int, void> function, void* context, TeamBarrier* barrier)
    {
        var members = Members;
        if (members <= 1 || workers.Length == 0 || IsPaused)
        {
            barrier->Participants = 1;
            for (var m = 0; m < Math.Max(1, members); m++) function(context, m);
            return;
        }
        barrier->Participants = members;
        var generation = Volatile.Read(ref control->TeamTicket) + 1;
        control->TeamFunction = function;
        control->TeamContext = context;
        Volatile.Write(ref control->TeamDone, 0);
        Volatile.Write(ref control->TeamActive, 1);
        Volatile.Write(ref control->TeamTicket, generation);
        if (Volatile.Read(ref control->Sleeping) > 0) WakeAll();
        if (policy.Caller == CallerRole.Participates)
        {
            function(context, workers.Length);
            Interlocked.Increment(ref control->TeamDone);
        }
        var spin = new SpinWait();
        while (Volatile.Read(ref control->TeamDone) < members)
#if NETSTANDARD2_1
            spin.SpinOnce();
#else
            spin.SpinOnce(-1);
#endif
        Volatile.Write(ref control->TeamActive, 0);
    }

    public static void Barrier(TeamBarrier* barrier)
    {
        var participants = Volatile.Read(ref barrier->Participants);
        if (participants <= 1) return;
        var phase = Volatile.Read(ref barrier->Phase);
        if (Interlocked.Increment(ref barrier->Arrived) == participants)
        {
            Volatile.Write(ref barrier->Arrived, 0);
            Volatile.Write(ref barrier->Phase, phase + 1);
        }
        else
        {
            var spin = new SpinWait();
            while (Volatile.Read(ref barrier->Phase) == phase)
#if NETSTANDARD2_1
                spin.SpinOnce();
#else
                spin.SpinOnce(-1);
#endif
        }
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

    bool HasTeamWork() => Volatile.Read(ref control->TeamActive) != 0;

    void Park()
    {
        lock (gate)
        {
            Interlocked.Increment(ref control->Sleeping);
            while (Volatile.Read(ref control->Stopping) == 0 && (IsPaused || (!HasWork() && !HasTeamWork()))) Monitor.Wait(gate);
            Interlocked.Decrement(ref control->Sleeping);
        }
    }

    void Work(int index, int cpu)
    {
        Volatile.Write(ref nativeWorkerIds[index], OsThreads.CurrentThreadId());
        OsThreads.ConfigureCurrentThread(policy.Priority, policy.Placement, cpu);
        var deadline = Stopwatch.GetTimestamp() + spinTicks;
        var teamSeen = 0L;
        while (Volatile.Read(ref control->Stopping) == 0)
        {
            var paused = IsPaused;
            var teamTicket = Volatile.Read(ref control->TeamTicket);
            if (!paused && teamTicket != teamSeen)
            {
                teamSeen = teamTicket;
                var teamFunction = control->TeamFunction;
                var teamContext = control->TeamContext;
                teamFunction(teamContext, index);
                Interlocked.Increment(ref control->TeamDone);
                deadline = Stopwatch.GetTimestamp() + spinTicks;
                continue;
            }
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
