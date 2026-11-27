using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

namespace Needle;

[StructLayout(LayoutKind.Explicit, Size = 128)]
internal struct TeamBarrier
{
    [FieldOffset(0)] public int Arrived;
    [FieldOffset(4)] public int Participants;
    [FieldOffset(64)] public long Phase;
}

#if NETSTANDARD2_1
internal struct Spinner
{
    int _count;

    public void SpinOnce()
    {
        if (_count < 10)
        {
            Thread.SpinWait(4 << _count);
            _count++;
        }
        else
        {
            Thread.Yield();
        }
    }
}
#endif

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
    }

    readonly Control* control;
    readonly Thread[] workers;
    readonly ThreadPolicy policy;
    readonly int[] nativeWorkerIds;
    readonly long spinTicks;
    readonly object gate = new();
    int disposed;

    public int Threads => policy.Threads;

    public int Members => workers.Length + (policy.Caller == CallerRole.Participates ? 1 : 0);

    public int[] NativeWorkerIds => nativeWorkerIds;

    public bool IsPaused => Volatile.Read(ref control->Paused) != 0;

    public int TeamSize => IsPaused ? 1 : Members;

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
        var spins = 0;
        while (Volatile.Read(ref control->Done) < count)
        {
            if ((++spins & 4095) == 0) Thread.Yield();
#if NET8_0_OR_GREATER
            else if (System.Runtime.Intrinsics.X86.X86Base.IsSupported) System.Runtime.Intrinsics.X86.X86Base.Pause();
#endif
            else Thread.SpinWait(1);
        }
    }

    public void RunTeam(delegate*<void*, int, void> function, void* context, TeamBarrier* barrier, int members)
    {
        if (members <= 1 || workers.Length == 0)
        {
            barrier->Participants = 1;
            function(context, 0);
            return;
        }
        if (members != Members) throw new ArgumentOutOfRangeException(nameof(members), $"a team has 1 or {Members} members");
        barrier->Participants = members;
        var generation = Volatile.Read(ref control->TeamTicket) + 1;
        control->TeamFunction = function;
        control->TeamContext = context;
        Volatile.Write(ref control->TeamDone, 0);
        Interlocked.Exchange(ref control->TeamTicket, generation);
        if (Volatile.Read(ref control->Sleeping) > 0) WakeAll();
        if (policy.Caller == CallerRole.Participates)
        {
            function(context, workers.Length);
            Interlocked.Increment(ref control->TeamDone);
        }
        var spins = 0;
        while (Volatile.Read(ref control->TeamDone) < members)
        {
            if ((++spins & 4095) == 0) Thread.Yield();
#if NET8_0_OR_GREATER
            else if (System.Runtime.Intrinsics.X86.X86Base.IsSupported) System.Runtime.Intrinsics.X86.X86Base.Pause();
#endif
            else Thread.SpinWait(1);
        }
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
            var spins = 0;
            while (Volatile.Read(ref barrier->Phase) == phase)
            {
                if ((++spins & 4095) == 0) Thread.Yield();
#if NET8_0_OR_GREATER
                else if (System.Runtime.Intrinsics.X86.X86Base.IsSupported) System.Runtime.Intrinsics.X86.X86Base.Pause();
#endif
                else Thread.SpinWait(1);
            }
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

    void Park(long teamSeen)
    {
        lock (gate)
        {
            Interlocked.Increment(ref control->Sleeping);
            while (Volatile.Read(ref control->Stopping) == 0 && Volatile.Read(ref control->TeamTicket) == teamSeen && !HasWork()) Monitor.Wait(gate);
            Interlocked.Decrement(ref control->Sleeping);
        }
    }

    void Work(int index, int cpu)
    {
        Volatile.Write(ref nativeWorkerIds[index], OsThreads.CurrentThreadId());
        OsThreads.ConfigureCurrentThread(policy.Priority, policy.Placement, cpu);
        var deadline = Stopwatch.GetTimestamp() + spinTicks;
        var teamSeen = 0L;
        var spins = 0;
        while (Volatile.Read(ref control->Stopping) == 0)
        {
            var teamTicket = Volatile.Read(ref control->TeamTicket);
            if (teamTicket != teamSeen)
            {
                teamSeen = teamTicket;
                var teamFunction = control->TeamFunction;
                var teamContext = control->TeamContext;
                teamFunction(teamContext, index);
                Interlocked.Increment(ref control->TeamDone);
                deadline = Stopwatch.GetTimestamp() + spinTicks;
                spins = 0;
                continue;
            }
            if (Drain()) { deadline = Stopwatch.GetTimestamp() + spinTicks; spins = 0; continue; }
            if (!IsPaused && (((++spins & 63) != 0 && spinTicks > 0) || Stopwatch.GetTimestamp() < deadline))
            {
#if NET8_0_OR_GREATER
                if (System.Runtime.Intrinsics.X86.X86Base.IsSupported) System.Runtime.Intrinsics.X86.X86Base.Pause();
                else Thread.SpinWait(4);
#else
                Thread.SpinWait(4);
#endif
                continue;
            }
            Park(teamSeen);
            deadline = Stopwatch.GetTimestamp() + spinTicks;
            spins = 0;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0) return;
        Volatile.Write(ref control->Stopping, 1);
        WakeAll();
        foreach (var worker in workers) worker.Join();
        Mem.AlignedFree(control);
    }
}
