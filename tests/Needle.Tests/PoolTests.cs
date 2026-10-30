using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using Xunit;

namespace Needle.Tests;

public sealed unsafe class PoolTests
{
    const int Slots = 128;
    const int AffinityWords = 16;
    const long IdleBudgetNanoseconds = 3_000_000;
    const int IdleAttempts = 5;

    [DllImport("libc", EntryPoint = "sched_getaffinity")]
    static extern int SchedGetAffinity(int thread, nuint bytes, ulong* mask);

    [DllImport("libc", EntryPoint = "getpriority")]
    static extern int GetPriority(int scope, uint thread);

    [DllImport("libc", EntryPoint = "gettid")]
    static extern int ThreadId();

    static void Increment(void* context, int index) => ((int*)context)[index] += 1;

    static void RecordThread(void* context, int index)
    {
        ((int*)context)[index] = Environment.CurrentManagedThreadId;
        Thread.SpinWait(2000);
    }

    static ulong[] CallerAffinity()
    {
        var words = new ulong[AffinityWords];
        fixed (ulong* mask = words) Assert.Equal(0, SchedGetAffinity(0, AffinityWords * sizeof(ulong), mask));
        return words;
    }

    static int CallerNice() => GetPriority(0, (uint)ThreadId());

    static long WorkerCpuNanoseconds(Pool pool)
    {
        var total = 0L;
        foreach (var id in pool.NativeWorkerIds)
            total += long.Parse(File.ReadAllText($"/proc/self/task/{id}/schedstat").Split(' ')[0]);
        return total;
    }

    static void WaitForWorkers(Pool pool)
    {
        var deadline = Environment.TickCount64 + 5000;
        while (Array.IndexOf(pool.NativeWorkerIds, 0) >= 0 && Environment.TickCount64 < deadline) Thread.Sleep(1);
    }

    static void Burst(Pool pool, int jobs)
    {
        var counts = stackalloc int[Slots];
        for (var i = 0; i < Slots; i++) counts[i] = 0;
        for (var job = 0; job < jobs; job++) pool.Run(&Increment, counts, 1 + job % 7);
    }

    [Fact]
    public void EveryTaskRunsExactlyOnceAcrossManyBackToBackJobs()
    {
        using var pool = new Pool(8);
        var counts = (int*)NativeMemory.AllocZeroed(Slots * sizeof(int));
        var expected = new int[Slots];
        var mismatches = 0;
        try
        {
            for (var job = 0; job < 50_000; job++)
            {
                var tasks = 1 + job % 97;
                pool.Run(&Increment, counts, tasks);
                for (var i = 0; i < tasks; i++) expected[i]++;
                for (var i = 0; i < Slots; i++) if (counts[i] != expected[i]) mismatches++;
            }
        }
        finally { NativeMemory.Free(counts); }
        Assert.Equal(0, mismatches);
    }

    [Fact]
    public void ZeroSpinBudgetStillRunsEveryTaskExactlyOnce()
    {
        using var pool = new Pool(ThreadPolicy.Default.WithThreads(4).WithSpinMicroseconds(0));
        var counts = (int*)NativeMemory.AllocZeroed(Slots * sizeof(int));
        var expected = new int[Slots];
        var mismatches = 0;
        try
        {
            for (var job = 0; job < 5_000; job++)
            {
                var tasks = 1 + job % 97;
                pool.Run(&Increment, counts, tasks);
                for (var i = 0; i < tasks; i++) expected[i]++;
                for (var i = 0; i < Slots; i++) if (counts[i] != expected[i]) mismatches++;
            }
        }
        finally { NativeMemory.Free(counts); }
        Assert.Equal(0, mismatches);
    }

    [Fact]
    public void ConstructingPoolsLeavesCallerAffinityAndPriorityUnchanged()
    {
        if (!OperatingSystem.IsLinux()) return;
        var affinity = CallerAffinity();
        var nice = CallerNice();
        var policies = new[]
        {
            ThreadPolicy.Default.WithThreads(4),
            ThreadPolicy.Game.WithThreads(4),
            ThreadPolicy.Default.WithThreads(4).WithPlacement(CorePlacement.EfficiencyCores),
            ThreadPolicy.Default.WithThreads(4).WithCpus(0),
            ThreadPolicy.Default.WithThreads(4).WithCaller(CallerRole.Waits).WithPriority(WorkerPriority.Lowest),
        };
        foreach (var policy in policies)
        {
            using var pool = new Pool(policy);
            WaitForWorkers(pool);
            Burst(pool, 200);
            Assert.Equal(affinity, CallerAffinity());
            Assert.Equal(nice, CallerNice());
        }
        using (new Pool(4)) Assert.Equal(affinity, CallerAffinity());
    }

    [Fact]
    public void ConstructingEngineLeavesCallerAffinityUnchanged()
    {
        if (!OperatingSystem.IsLinux()) return;
        Assets.Require();
        using var model = Model.Open(Assets.Model);
        var affinity = CallerAffinity();
        var nice = CallerNice();
        using (var engine = new Engine(model, 256, 64, 4))
        {
            engine.Feed(new[] { 1, 2, 3, 4 });
            Assert.Equal(affinity, CallerAffinity());
        }
        using (var engine = new Engine(model, ThreadPolicy.Game.WithThreads(4), 256, 64))
        {
            engine.Feed(new[] { 1, 2, 3, 4 });
            Assert.Equal(affinity, CallerAffinity());
        }
        Assert.Equal(nice, CallerNice());
    }

    static long IdleCpuNanoseconds(ThreadPolicy policy)
    {
        var quietest = long.MaxValue;
        for (var attempt = 0; attempt < IdleAttempts; attempt++)
        {
            using var pool = new Pool(policy);
            WaitForWorkers(pool);
            Burst(pool, 2_000);
            var before = WorkerCpuNanoseconds(pool);
            Thread.Sleep(100);
            quietest = Math.Min(quietest, WorkerCpuNanoseconds(pool) - before);
        }
        return quietest;
    }

    static bool HasSchedulerStatistics() => OperatingSystem.IsLinux() && File.Exists($"/proc/self/task/{ThreadId()}/schedstat");

    [Fact]
    public void DefaultWorkersStopConsumingCpuWithinTheSpinBudget()
    {
        if (!HasSchedulerStatistics()) return;
        var consumed = IdleCpuNanoseconds(ThreadPolicy.Default.WithThreads(4));
        Assert.True(consumed < IdleBudgetNanoseconds, $"workers burned {consumed / 1000} us over 100 ms of idle");
    }

    [Fact]
    public void GameWorkersBurnUnderOneMillisecondAfterABurst()
    {
        if (!HasSchedulerStatistics()) return;
        var consumed = IdleCpuNanoseconds(ThreadPolicy.Game.WithThreads(4));
        Assert.True(consumed < 1_000_000, $"workers burned {consumed / 1000} us over 100 ms of idle");
    }

    [Fact]
    public void ParkedWorkersWakeForTheNextJob()
    {
        using var pool = new Pool(ThreadPolicy.Default.WithThreads(4).WithSpinMicroseconds(10));
        var counts = (int*)NativeMemory.AllocZeroed(Slots * sizeof(int));
        try
        {
            for (var round = 1; round <= 20; round++)
            {
                Thread.Sleep(5);
                pool.Run(&Increment, counts, 64);
                for (var i = 0; i < 64; i++) Assert.Equal(round, counts[i]);
            }
        }
        finally { NativeMemory.Free(counts); }
    }

    [Fact]
    public void PausedPoolRunsEverythingOnTheCallerAndStaysIdle()
    {
        using var pool = new Pool(ThreadPolicy.Default.WithThreads(4));
        var ids = (int*)NativeMemory.AllocZeroed(Slots * sizeof(int));
        try
        {
            WaitForWorkers(pool);
            pool.Pause();
            Assert.True(pool.IsPaused);
            pool.Run(&RecordThread, ids, 64);
            var caller = Environment.CurrentManagedThreadId;
            for (var i = 0; i < 64; i++) Assert.Equal(caller, ids[i]);
            pool.Resume();
            Assert.False(pool.IsPaused);
            var sawWorker = false;
            for (var attempt = 0; attempt < 200 && !sawWorker; attempt++)
            {
                pool.Run(&RecordThread, ids, 64);
                for (var i = 0; i < 64; i++) sawWorker |= ids[i] != caller;
            }
            Assert.True(sawWorker);
        }
        finally { NativeMemory.Free(ids); }
    }

    [Fact]
    public void PausedWorkersConsumeNoCpu()
    {
        if (!HasSchedulerStatistics()) return;
        using var pool = new Pool(ThreadPolicy.Default.WithThreads(4).WithSpinMicroseconds(5_000));
        WaitForWorkers(pool);
        Burst(pool, 500);
        pool.Pause();
        Thread.Sleep(20);
        var before = WorkerCpuNanoseconds(pool);
        Burst(pool, 500);
        Thread.Sleep(50);
        Assert.True(WorkerCpuNanoseconds(pool) - before < IdleBudgetNanoseconds);
        pool.Resume();
        Burst(pool, 500);
    }

    [Fact]
    public void WaitingCallerNeverRunsTasks()
    {
        using var pool = new Pool(ThreadPolicy.Default.WithThreads(3).WithCaller(CallerRole.Waits));
        var ids = (int*)NativeMemory.AllocZeroed(Slots * sizeof(int));
        try
        {
            var caller = Environment.CurrentManagedThreadId;
            foreach (var tasks in new[] { 1, 2, 17, 64 })
            {
                pool.Run(&RecordThread, ids, tasks);
                for (var i = 0; i < tasks; i++) Assert.NotEqual(caller, ids[i]);
                for (var i = 0; i < tasks; i++) Assert.NotEqual(0, ids[i]);
            }
        }
        finally { NativeMemory.Free(ids); }
    }

    [Fact]
    public void PolicyIsTotalAndResolvesWorkerCounts()
    {
        Assert.Equal(1, default(ThreadPolicy).Threads);
        Assert.Equal(0, default(ThreadPolicy).Workers);
        Assert.Equal(3, ThreadPolicy.Default.WithThreads(4).Workers);
        Assert.Equal(4, ThreadPolicy.Default.WithThreads(4).WithCaller(CallerRole.Waits).Workers);
        Assert.Equal(1, ThreadPolicy.Default.WithThreads(-5).Threads);
        Assert.Equal(0, ThreadPolicy.Default.WithSpinMicroseconds(-1).SpinMicroseconds);
        Assert.Equal(ThreadPolicy.DefaultSpinMicroseconds, ThreadPolicy.Default.SpinMicroseconds);
        Assert.Equal(CorePlacement.Any, new ThreadPolicy(2, 50, WorkerPriority.Normal, CorePlacement.Explicit, CallerRole.Participates, 0).Placement);
        Assert.Equal(0b101UL, ThreadPolicy.Default.WithCpus(0, 2, 99, -1).ExplicitCpus);
    }

    [Fact]
    public void TopologyParsesCpuListsAndSplitsHybridCores()
    {
        Assert.Equal(new[] { 0, 1, 2, 5 }, Topology.ParseList("0-2,5\n"));
        Assert.Empty(Topology.ParseList("\n"));
        Assert.Equal(0u, Topology.PerformanceFloor(new uint[] { 4000, 4000, 4100 }));
        Assert.Equal(5400u, Topology.PerformanceFloor(new uint[] { 3800, 3800, 5400, 5400 }));
        Assert.Equal(800u, Topology.PerformanceFloor(new uint[] { 400, 800, 1024 }));
    }

    [Fact]
    public void SpinBudgetsStayAtTheirMeasuredDefaults()
    {
        Assert.Equal(300, ThreadPolicy.DefaultSpinMicroseconds);
        Assert.Equal(50, ThreadPolicy.GameSpinMicroseconds);
        Assert.Equal(ThreadPolicy.DefaultSpinMicroseconds, ThreadPolicy.Default.SpinMicroseconds);
        Assert.Equal(ThreadPolicy.GameSpinMicroseconds, ThreadPolicy.Game.SpinMicroseconds);
    }

    [Fact]
    public void FrameBudgetsDefaultToUnlimitedAndClampNegativeInputs()
    {
        Assert.True(default(FrameBudget).IsUnlimited);
        Assert.True(FrameBudget.Unlimited.IsUnlimited);
        Assert.True(new NeedleOptions().FrameBudget.IsUnlimited);
        Assert.True(new FrameBudget(-3, -1.5).IsUnlimited);
        var tokens = FrameBudget.FromTokens(8);
        Assert.Equal(8, tokens.MaxTokensPerSlice);
        Assert.Equal(0, tokens.MaxMillisecondsPerSlice);
        Assert.False(tokens.IsUnlimited);
        var both = FrameBudget.FromMilliseconds(2.5).WithMaxTokensPerSlice(8);
        Assert.Equal(8, both.MaxTokensPerSlice);
        Assert.Equal(2.5, both.MaxMillisecondsPerSlice);
        Assert.False(both.IsUnlimited);
        Assert.True(both.WithMaxTokensPerSlice(0).WithMaxMillisecondsPerSlice(0).IsUnlimited);
    }
}

public sealed unsafe class TeamBarrierTests
{
    const int Phases = 4;

    struct TeamContext
    {
        public int* Seen;
        public TeamBarrier* Barrier;
        public int Members;
    }

    static void Rendezvous(void* context, int member)
    {
        var state = (TeamContext*)context;
        for (var phase = 0; phase < Phases; phase++)
        {
            state->Seen[member * Phases + phase] = phase + 1;
            Pool.Barrier(state->Barrier);
            for (var m = 0; m < state->Members; m++)
                if (state->Seen[m * Phases + phase] != phase + 1)
                    state->Seen[member * Phases + phase] = -1;
            Pool.Barrier(state->Barrier);
        }
    }

    static void RecordMember(void* context, int member) => ((int*)context)[member] = Environment.CurrentManagedThreadId;

    [Fact]
    public void TeamDispatchRendezvousEveryMemberAtEveryBarrier()
    {
        using var pool = new Pool(4);
        var members = pool.Members;
        Assert.Equal(4, members);
        var seen = stackalloc int[members * Phases];
        var barrier = stackalloc TeamBarrier[1];
        for (var round = 0; round < 50; round++)
        {
            for (var i = 0; i < members * Phases; i++) seen[i] = 0;
            var context = new TeamContext { Seen = seen, Barrier = barrier, Members = members };
            pool.RunTeam(&Rendezvous, &context, barrier);
            for (var i = 0; i < members * Phases; i++) Assert.Equal((i % Phases) + 1, seen[i]);
        }
    }

    [Fact]
    public void PausedTeamRunsEveryMemberOnTheCaller()
    {
        using var pool = new Pool(4);
        pool.Pause();
        var members = pool.Members;
        var ids = stackalloc int[members];
        for (var i = 0; i < members; i++) ids[i] = 0;
        var barrier = stackalloc TeamBarrier[1];
        pool.RunTeam(&RecordMember, ids, barrier);
        var caller = Environment.CurrentManagedThreadId;
        for (var i = 0; i < members; i++) Assert.Equal(caller, ids[i]);
        pool.Resume();
    }

    [Fact]
    public void SingleThreadTeamRunsInlineWithoutWorkers()
    {
        using var pool = new Pool(1);
        Assert.Equal(1, pool.Members);
        var ran = 0;
        var barrier = stackalloc TeamBarrier[1];
        pool.RunTeam(&RecordMember, &ran, barrier);
        Assert.Equal(Environment.CurrentManagedThreadId, ran);
    }
}
