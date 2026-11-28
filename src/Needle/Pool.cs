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
    [FieldOffset(8)] public long SpinTicks;
    [FieldOffset(64)] public long Phase;
    [FieldOffset(72)] public int Sleepers;
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
        [FieldOffset(2 * CacheLine + 20)] public int RunWorkers;
        [FieldOffset(3 * CacheLine)] public int Sleeping;
        [FieldOffset(3 * CacheLine + 4)] public int Stopping;
        [FieldOffset(3 * CacheLine + 8)] public int Paused;
        [FieldOffset(4 * CacheLine)] public long TeamTicket;
        [FieldOffset(4 * CacheLine + 8)] public delegate*<void*, int, void> TeamFunction;
        [FieldOffset(4 * CacheLine + 16)] public void* TeamContext;
        [FieldOffset(4 * CacheLine + 24)] public int TeamDone;
        [FieldOffset(4 * CacheLine + 28)] public int TeamActive;
        [FieldOffset(4 * CacheLine + 32)] public int TeamMembers;
        [FieldOffset(5 * CacheLine)] public int CallerSleeping;
        [FieldOffset(5 * CacheLine + 8)] public long PingTicket;
    }

    readonly Control* control;
    readonly Thread[] workers;
    readonly ThreadPolicy policy;
    readonly int[] nativeWorkerIds;
    readonly long spinTicks;
    readonly long waitTicks;
    readonly object gate = new();
    static readonly object BarrierGate = new();
    const int ProbeMinRuns = 16;
    const int ProbeMaxRuns = 1024;
    static readonly long StallTicks = 2 * Stopwatch.Frequency / 1000;
    static readonly long IdleWindowTicks = Stopwatch.Frequency / 10;
    static readonly long FitWindowTicks = Stopwatch.Frequency / 20;
    const int MinWaitMicroseconds = 20;
    const int MaxWaitMicroseconds = 200;
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
        waitTicks = Math.Clamp(policy.SpinMicroseconds, MinWaitMicroseconds, MaxWaitMicroseconds) * Stopwatch.Frequency / 1_000_000;
        control = (Control*)Mem.AlignedAlloc((nuint)sizeof(Control), CacheLine);
        Mem.Clear(control, (nuint)sizeof(Control));
        var cores = Topology.Select(policy);
        var callerSlots = policy.Caller == CallerRole.Participates && policy.Placement != CorePlacement.Explicit ? 1 : 0;
        workers = new Thread[policy.Workers];
        nativeWorkerIds = new int[workers.Length];
        teamSize = Members;
        teamCost = new double[Members + 1];
        pongs = new long[workers.Length];
        var sizes = 1;
        for (var size = Members; size > 1; size = Smaller(size)) sizes++;
        chain = new int[sizes];
        for (int size = Members, i = 0; i < sizes; size = Smaller(size), i++) chain[i] = size;
        workerCpu = new long[workers.Length];
        workerCpuSeen = new long[workers.Length];
        SampleIdle(0);
        probeEvery = ProbeMinRuns;
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
        // Only the workers of the settled team size take tasks: when a busy machine has shrunk the team,
        // a task claimed by a worker that then waits for a core would hold up the whole run. Tasks are
        // independent, so who runs them never changes the result.
        var participates = policy.Caller == CallerRole.Participates;
        var settled = SettledTeamMembers;
        var runsOnCaller = participates && (count <= 1 || settled <= 1);
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
        control->RunWorkers = Math.Max(1, participates ? settled - 1 : settled);
        Volatile.Write(ref control->Done, 0);
        Interlocked.Exchange(ref control->Ticket, generation << 32);
        if (Volatile.Read(ref control->Sleeping) > 0) WakeAll();
        if (policy.Caller == CallerRole.Participates) Drain();
        WaitForWorkers(ref control->Done, count);
    }

    /// <summary>
    /// The members the next adaptive team run should use. The pool keeps a running mean of the wall
    /// time of a run at each size on the halving chain (Members, Members/2, ..., 1), stays on the
    /// cheapest, and now and then times a neighbouring size: a busy machine (other threads on every
    /// core, or workers starved by their lower priority) makes barriers stall and a smaller team wins;
    /// once the cores free up the full team wins again. A run far slower than usual probes a smaller
    /// team at once. Probes back off exponentially while they keep losing. A size that needs more
    /// workers than the machine has idle cores (read from the OS every 50 to 100 ms) is cut down before
    /// the run starts, and a probe that would add workers waits for idle cores.
    /// </summary>
    public int TeamMembers
    {
        get
        {
            if (IsPaused || workers.Length == 0) return 1;
            if (pinnedTeam > 0) return Math.Min(pinnedTeam, Members);
            return Fit(probeSize > 0 ? probeSize : teamSize);
        }
    }

    /// <summary>The adaptive team size without a pending probe, for runs that should not be timed. Like
    /// <see cref="TeamMembers"/> it shrinks before a run when the machine has fewer idle cores than workers.</summary>
    public int SettledTeamMembers
    {
        get
        {
            if (IsPaused || workers.Length == 0) return 1;
            if (pinnedTeam > 0) return Math.Min(pinnedTeam, Members);
            return Fit(teamSize, adopt: false);
        }
    }

    /// <summary>Fixes the adaptive team size (0 restores adaptation). Diagnostics and tests.</summary>
    public void PinTeamMembers(int members) => pinnedTeam = Math.Max(0, members);

    int teamSize;
    int pinnedTeam;
    int probeSize;
    int probeEvery;
    int sinceProbe;
    int probeCursor;
    int pingTarget;
    long pingAt;
    readonly long[] pongs;
    static readonly long PingTicks = 25 * Stopwatch.Frequency / 1000;
    readonly int[] chain;
    double calm;
    long idleAt;
    double idleSeconds;
    double freeCores = double.PositiveInfinity;
    double workerCores;
    readonly long[] workerCpu;
    readonly long[] workerCpuSeen;
    bool idleKnown = true;
    readonly double[] teamCost;

    public void RunTeam(delegate*<void*, int, void> function, void* context, TeamBarrier* barrier) =>
        RunTeam(function, context, barrier, Members, false);

    /// <summary>Runs a team of <paramref name="members"/> without timing it.</summary>
    public void RunTeam(delegate*<void*, int, void> function, void* context, TeamBarrier* barrier, int members) =>
        RunTeam(function, context, barrier, members, false);

    /// <summary>Runs a team of <paramref name="members"/> (from <see cref="TeamMembers"/>) and adapts the next size.</summary>
    public void RunAdaptiveTeam(delegate*<void*, int, void> function, void* context, TeamBarrier* barrier, int members) =>
        RunTeam(function, context, barrier, members, true);

    void RunTeam(delegate*<void*, int, void> function, void* context, TeamBarrier* barrier, int members, bool adapt)
    {
        members = Math.Min(members, Members);
        // A paused pool still runs a team on its workers: running the members one after another on the
        // caller would break a team function whose phases meet at barriers.
        if (members <= 1 || workers.Length == 0)
        {
            barrier->Participants = 1;
            var inline = Stopwatch.GetTimestamp();
            for (var m = 0; m < Math.Max(1, members); m++) function(context, m);
            if (adapt && members == 1 && workers.Length > 0 && !IsPaused) Adapt(1, Stopwatch.GetTimestamp() - inline);
            return;
        }
        var participates = policy.Caller == CallerRole.Participates;
        barrier->Participants = members;
        barrier->SpinTicks = waitTicks;
        barrier->Sleepers = 0;
        var started = Stopwatch.GetTimestamp();
        var generation = Volatile.Read(ref control->TeamTicket) + 1;
        control->TeamFunction = function;
        control->TeamContext = context;
        control->TeamMembers = participates ? members - 1 : members;
        Volatile.Write(ref control->TeamDone, 0);
        Interlocked.Exchange(ref control->TeamTicket, generation);
        if (Volatile.Read(ref control->Sleeping) > 0) WakeAll();
        if (participates)
        {
            function(context, members - 1);
            Interlocked.Increment(ref control->TeamDone);
        }
        WaitForWorkers(ref control->TeamDone, members);
        Volatile.Write(ref control->TeamActive, 0);
        if (adapt && !IsPaused) Adapt(members, Stopwatch.GetTimestamp() - started);
    }

    void Adapt(int members, long ticks)
    {
        if (pinnedTeam > 0 || ticks <= 0) return;
        var cost = teamCost;
        SampleIdle(IdleWindowTicks);
        CheckPing();
        if (members == probeSize)
        {
            probeSize = 0;
            sinceProbe = 0;
            // One fresh sample: an older mean at this size describes a machine that may have changed.
            cost[members] = ticks;
            if (ticks < cost[teamSize] * 0.85)
            {
                teamSize = members;
                calm = cost[members];
                probeEvery = ProbeMinRuns;
            }
            else probeEvery = Math.Min(probeEvery * 2, ProbeMaxRuns);
            return;
        }
        if (members != teamSize) return;
        var mean = cost[members];
        if (mean > 0 && ticks > 4 * mean + StallTicks && teamSize > 1)
        {
            // The caller alone never waits on another thread: the safe size to compare against.
            cost[members] = mean + (ticks - mean) / 4;
            probeSize = 1;
            pingAt = 0;
            return;
        }
        cost[members] = mean <= 0 ? ticks : mean + (ticks - mean) / 8;
        calm = calm <= 0 ? cost[members] : Math.Min(calm, cost[members]);
        if (cost[members] > calm * 1.5)
        {
            calm = cost[members];
            probeEvery = ProbeMinRuns;
        }
        if (pingAt != 0 || probeSize != 0 || ++sinceProbe < probeEvery) return;
        sinceProbe = 0;
        // Round robin over the other sizes on the chain, so a size two steps away is tried too
        // (workers starved by priority make every size with workers stall, not just the largest).
        for (var tries = 0; tries < chain.Length; tries++)
        {
            probeCursor = (probeCursor + 1) % chain.Length;
            if (chain[probeCursor] == teamSize) continue;
            Propose(chain[probeCursor]);
            return;
        }
    }

    // A probe that needs workers first pings them: a run with a starved worker can take seconds (a
    // BelowNormal thread on a busy Windows machine waits for the starvation boost), but an unanswered
    // ping costs nothing. The probe runs only once every worker it needs answered within PingTicks (25 ms: far above a
    // busy Linux scheduler's wake latency, far below the seconds a starved Windows thread waits).
    void Propose(int size)
    {
        if (size == 1)
        {
            probeSize = 1;
            return;
        }
        // A probe that adds workers needs idle cores for them. With every core busy (other programs, or a
        // game's own threads) the added workers would be time-sliced against them, and on Windows a
        // run that waits on a preempted worker waits a whole scheduler quantum at every barrier: seconds
        // per token. A busy machine counts as a lost probe, so the next try backs off.
        if (size > teamSize && freeCores + workerCores < (size - 1) * 0.6)
        {
            probeEvery = Math.Min(probeEvery * 2, ProbeMaxRuns);
            return;
        }
        pingTarget = size;
        pingAt = Stopwatch.GetTimestamp();
        Interlocked.Exchange(ref control->PingTicket, pingAt);
        if (Volatile.Read(ref control->Sleeping) > 0) WakeAll();
    }

    void CheckPing()
    {
        if (pingAt == 0) return;
        var needed = Math.Min(workers.Length, policy.Caller == CallerRole.Participates ? pingTarget - 1 : pingTarget);
        var answered = true;
        for (var i = 0; i < needed; i++)
        {
            var pong = Volatile.Read(ref pongs[i]);
            if (pong < pingAt || pong - pingAt > PingTicks) answered = false;
        }
        if (!answered && Stopwatch.GetTimestamp() - pingAt <= PingTicks) return;
        pingAt = 0;
        if (answered) probeSize = pingTarget;
        else probeEvery = Math.Min(probeEvery * 2, ProbeMaxRuns);
    }

    // Idle cores over the last window (at least minWindow long), from the OS's machine-wide idle time,
    // and the cores this pool's workers used in it, from their own thread CPU time; infinite when the
    // OS does not say. One read per window, none per run. Idle plus our workers' share is what the
    // workers could have: on a saturated machine both are near zero, even while a team waits on them.
    void SampleIdle(long minWindow)
    {
        if (!idleKnown) return;
        var now = Stopwatch.GetTimestamp();
        if (idleAt != 0 && now - idleAt < minWindow) return;
        if (!OsThreads.TrySystemIdleSeconds(out var idle))
        {
            idleKnown = false;
            freeCores = double.PositiveInfinity;
            return;
        }
        long used = 0;
        for (var i = 0; i < workerCpu.Length; i++)
        {
            var cpu = Volatile.Read(ref workerCpu[i]);
            used += cpu - workerCpuSeen[i];
            workerCpuSeen[i] = cpu;
        }
        if (idleAt != 0)
        {
            var seconds = (double)(now - idleAt) / Stopwatch.Frequency;
            freeCores = (idle - idleSeconds) / seconds;
            workerCores = used * 1e-7 / seconds;
        }
        idleAt = now;
        idleSeconds = idle;
    }

    // Shrinks a run that needs more workers than the cores they could get, before it starts. Without
    // this the first runs on a saturated machine (a prefill, before any decode has been timed) wait on
    // workers that never get a core; with BelowNormal workers on Windows that is seconds per barrier.
    // adopt: a smaller fit becomes the adaptive size, which only timed (adaptive) runs grow back. Runs that
    // are not timed (prefill, batched layers) must not adopt it, or one busy moment would pin them small.
    int Fit(int members, bool adopt = true)
    {
        if (members <= 1 || pinnedTeam > 0 || IsPaused) return members;
        SampleIdle(FitWindowTicks);
        var available = freeCores + workerCores;
        if (available >= (members - 1) * 0.6) return members;
        var fit = 1;
        foreach (var size in chain)
            if (size <= members && (size - 1) * 0.6 <= available) { fit = size; break; }
        if (adopt && fit < teamSize)
        {
            teamSize = fit;
            probeSize = 0;
            pingAt = 0;
            sinceProbe = 0;
        }
        return fit;
    }

    static int Smaller(int size) => Math.Max(1, size / 2);

    public static void Barrier(TeamBarrier* barrier)
    {
        var participants = Volatile.Read(ref barrier->Participants);
        if (participants <= 1) return;
        var phase = Volatile.Read(ref barrier->Phase);
        if (Interlocked.Increment(ref barrier->Arrived) == participants)
        {
            Volatile.Write(ref barrier->Arrived, 0);
            Interlocked.Exchange(ref barrier->Phase, phase + 1);
            if (Volatile.Read(ref barrier->Sleepers) > 0)
                lock (BarrierGate) Monitor.PulseAll(BarrierGate);
            return;
        }
        var budget = Volatile.Read(ref barrier->SpinTicks);
        if (budget <= 0) budget = MinWaitMicroseconds * Stopwatch.Frequency / 1_000_000;
        var start = Stopwatch.GetTimestamp();
        var spins = 0;
        while (Volatile.Read(ref barrier->Phase) == phase)
        {
            if ((++spins & 63) != 0)
            {
                CpuRelax();
                continue;
            }
            if (Stopwatch.GetTimestamp() - start < budget)
            {
                if ((spins & 1023) == 0) Thread.Yield();
                continue;
            }
            lock (BarrierGate)
            {
                Interlocked.Increment(ref barrier->Sleepers);
                while (Volatile.Read(ref barrier->Phase) == phase) Monitor.Wait(BarrierGate);
                Interlocked.Decrement(ref barrier->Sleepers);
            }
            return;
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    static void CpuRelax()
    {
#if NET8_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.X86Base.IsSupported) System.Runtime.Intrinsics.X86.X86Base.Pause();
        else Thread.SpinWait(1);
#else
        Thread.SpinWait(1);
#endif
    }

    void WaitForWorkers(ref int done, int target)
    {
        var start = Stopwatch.GetTimestamp();
        var spins = 0;
        while (Volatile.Read(ref done) < target)
        {
            if ((++spins & 63) != 0)
            {
                CpuRelax();
                continue;
            }
            if (Stopwatch.GetTimestamp() - start < waitTicks)
            {
                if ((spins & 1023) == 0) Thread.Yield();
                continue;
            }
            lock (gate)
            {
                Volatile.Write(ref control->CallerSleeping, 1);
                Interlocked.MemoryBarrier();
                while (Volatile.Read(ref done) < target) Monitor.Wait(gate);
                Volatile.Write(ref control->CallerSleeping, 0);
            }
            return;
        }
    }

    void Finished(ref int done)
    {
        Interlocked.Increment(ref done);
        if (Volatile.Read(ref control->CallerSleeping) != 0)
            lock (gate) Monitor.PulseAll(gate);
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
            Finished(ref control->Done);
            claimed = true;
        }
    }

    void NoteCpu(int index)
    {
        if (idleKnown && OsThreads.TryCurrentThreadCpu(out var cpu)) Volatile.Write(ref workerCpu[index], cpu);
    }

    bool HasWork(int index) => index < Volatile.Read(ref control->RunWorkers) && (int)Volatile.Read(ref control->Ticket) < control->Count;

    void Park(int index, long teamSeen, long pingSeen)
    {
        lock (gate)
        {
            Interlocked.Increment(ref control->Sleeping);
            while (Volatile.Read(ref control->Stopping) == 0 && Volatile.Read(ref control->TeamTicket) == teamSeen && (IsPaused || (!HasWork(index) && Volatile.Read(ref control->PingTicket) == pingSeen))) Monitor.Wait(gate);
            Interlocked.Decrement(ref control->Sleeping);
        }
    }

    void Work(int index, int cpu)
    {
        Volatile.Write(ref nativeWorkerIds[index], OsThreads.CurrentThreadId());
        OsThreads.ConfigureCurrentThread(policy.Priority, policy.Placement, cpu);
        var deadline = Stopwatch.GetTimestamp() + spinTicks;
        var teamSeen = 0L;
        var pingSeen = 0L;
        var spins = 0;
        while (Volatile.Read(ref control->Stopping) == 0)
        {
            var ping = Volatile.Read(ref control->PingTicket);
            if (ping != pingSeen)
            {
                pingSeen = ping;
                Volatile.Write(ref pongs[index], Stopwatch.GetTimestamp());
            }
            var paused = IsPaused;
            var teamTicket = Volatile.Read(ref control->TeamTicket);
            if (teamTicket != teamSeen)
            {
                teamSeen = teamTicket;
                if (index >= control->TeamMembers) continue;
                var teamFunction = control->TeamFunction;
                var teamContext = control->TeamContext;
                teamFunction(teamContext, index);
                Finished(ref control->TeamDone);
                NoteCpu(index);
                deadline = Stopwatch.GetTimestamp() + spinTicks;
                spins = 0;
                continue;
            }
            if (!paused && index < Volatile.Read(ref control->RunWorkers) && Drain()) { NoteCpu(index); deadline = Stopwatch.GetTimestamp() + spinTicks; spins = 0; continue; }
            if (!paused && (((++spins & 63) != 0 && spinTicks > 0) || Stopwatch.GetTimestamp() < deadline))
            {
#if NET8_0_OR_GREATER
                if (System.Runtime.Intrinsics.X86.X86Base.IsSupported) System.Runtime.Intrinsics.X86.X86Base.Pause();
                else Thread.SpinWait(4);
#else
                Thread.SpinWait(4);
#endif
                continue;
            }
            Park(index, teamSeen, pingSeen);
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
