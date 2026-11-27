using System;
using System.Diagnostics;
using System.Threading;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

public class AgentPostTests
{
    readonly ITestOutputHelper output;

    public AgentPostTests(ITestOutputHelper output) => this.output = output;

    static NeedleAgent Open(NeedleModel model, int threads = 1) =>
        model.Agent(SessionCases.DefaultTools(), new AgentOptions { Threads = Assets.ThreadsFor(threads), AutoDate = Toggle.Off, Stateless = true });

    static bool Poll(NeedleAgent agent, out int calls, out string first)
    {
        var clock = Stopwatch.StartNew();
        while (clock.Elapsed < TimeSpan.FromMinutes(2))
        {
            if (agent.TryTake(out var answer))
            {
                calls = answer.Calls.Count;
                first = calls > 0 ? answer.Calls[0].ToString() : answer.ToString();
                return true;
            }
            Thread.Sleep(1);
        }
        calls = 0;
        first = "";
        return false;
    }

    [Fact]
    public void PostedAnswersMatchAsking()
    {
        Assets.Require();
        using var model = NeedleModel.Open(Assets.Model);
        using var agent = Open(model);
        var direct = agent.Ask("set a timer for 12 minutes and play some Miles Davis");
        var expectedCount = direct.Calls.Count;
        var expectedFirst = direct.Calls[0].ToString();

        Assert.True(agent.TryPost("set a timer for 12 minutes and play some Miles Davis"));
        Assert.False(agent.TryPost("another request while busy"));
        Assert.True(Poll(agent, out var count, out var first));
        Assert.Equal(expectedCount, count);
        Assert.Equal(expectedFirst, first);
        Assert.False(agent.IsBusy);
        Assert.False(agent.TryTake(out _));
    }

    [Fact]
    public void AskingWhileBusyThrows()
    {
        Assets.Require();
        using var model = NeedleModel.Open(Assets.Model);
        using var agent = Open(model);
        agent.Post("set a timer for 12 minutes");
        if (agent.IsBusy) Assert.Throws<InvalidOperationException>(() => agent.Ask("hello"));
        var answer = agent.Take();
        Assert.True(answer.Ok, answer.ToString());
    }

    [Fact]
    public void DisposingWhileRunningDoesNotHang()
    {
        Assets.Require();
        using var model = NeedleModel.Open(Assets.Model);
        var agent = Open(model);
        agent.Post("set a timer for 12 minutes and play some Miles Davis");
        var clock = Stopwatch.StartNew();
        agent.Dispose();
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30));
        Assert.False(agent.IsOpen);
    }

    [Fact]
    public void PostingDoesNotAllocate()
    {
        Assets.Require();
        using var model = NeedleModel.Open(Assets.Model);
        using var agent = Open(model);
        for (var i = 0; i < 3; i++) { agent.Post("set a timer for 12 minutes"); agent.Take(); }
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 6; i++)
        {
            agent.Post("set a timer for 12 minutes");
            while (!agent.TryTake(out var answer)) Thread.SpinWait(64);
        }
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    [Fact]
    public void PostingCostsLittleOverAskingUnderLoad()
    {
        Assets.Require();
        using var model = NeedleModel.Open(Assets.Model);
        using var agent = Open(model, 2);
        const string query = "set a timer for 12 minutes";
        agent.Ask(query);
        agent.Post(query);
        agent.Take();

        var stop = 0;
        var hogs = new Thread[Environment.ProcessorCount];
        for (var i = 0; i < hogs.Length; i++)
        {
            hogs[i] = new Thread(() => { var x = 0L; while (Volatile.Read(ref stop) == 0) x++; }) { IsBackground = true, Priority = ThreadPriority.BelowNormal };
            hogs[i].Start();
        }
        try
        {
            var direct = new double[7];
            var posted = new double[7];
            for (var i = 0; i < direct.Length; i++)
            {
                var clock = Stopwatch.StartNew();
                agent.Ask(query);
                direct[i] = clock.Elapsed.TotalMilliseconds;
                clock.Restart();
                agent.Post(query);
                while (!agent.TryTake(out _)) Thread.Sleep(0);
                posted[i] = clock.Elapsed.TotalMilliseconds;
            }
            Array.Sort(direct);
            Array.Sort(posted);
            output.WriteLine($"under load ({hogs.Length} busy threads): ask median {direct[3]:0.0} ms, post+take median {posted[3]:0.0} ms");
            Assert.True(posted[3] <= direct[3] * 2 + 20, $"posting took {posted[3]:0.0} ms against {direct[3]:0.0} ms asking directly");
        }
        finally
        {
            Volatile.Write(ref stop, 1);
            foreach (var hog in hogs) hog.Join();
        }
    }
}
