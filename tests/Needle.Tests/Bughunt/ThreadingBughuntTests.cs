using System;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace Needle.Tests.Bughunt;

public sealed unsafe class ThreadingBughuntTests
{
    struct SumContext
    {
        public int* Written;
        public int* Seen;
        public TeamBarrier* Barrier;
        public int Members;
    }

    static void WriteThenSum(void* context, int member)
    {
        var state = (SumContext*)context;
        state->Written[member] = member + 1;
        Pool.Barrier(state->Barrier);
        var sum = 0;
        for (var i = 0; i < state->Members; i++) sum += Volatile.Read(ref state->Written[i]);
        state->Seen[member] = sum;
    }

    [Fact]
    public void PausedTeamStillRendezvousEveryMemberAtTheBarrier()
    {
        using var pool = new Pool(4);
        var members = pool.Members;
        var written = stackalloc int[members];
        var seen = stackalloc int[members];
        var barrier = stackalloc TeamBarrier[1];
        pool.Pause();
        var context = new SumContext { Written = written, Seen = seen, Barrier = barrier, Members = members };
        pool.RunTeam(&WriteThenSum, &context, barrier, members);
        pool.Resume();
        var expected = members * (members + 1) / 2;
        for (var m = 0; m < members; m++)
            Assert.True(seen[m] == expected, $"member {m} saw {seen[m]} after the barrier, every member's write sums to {expected}");
    }

    [Fact]
    public void PausedPoolOffersASingleMemberTeamThatRunsOnTheCaller()
    {
        using var pool = new Pool(4);
        pool.Pause();
        Assert.Equal(1, pool.TeamSize);
        var written = stackalloc int[1];
        var seen = stackalloc int[1];
        var barrier = stackalloc TeamBarrier[1];
        var context = new SumContext { Written = written, Seen = seen, Barrier = barrier, Members = 1 };
        pool.RunTeam(&WriteThenSum, &context, barrier, pool.TeamSize);
        Assert.Equal(1, seen[0]);
        pool.Resume();
        Assert.Equal(pool.Members, pool.TeamSize);
    }

    static Pool? pausing;

    static void PauseThenRendezvous(void* context, int member)
    {
        var state = (SumContext*)context;
        if (member == state->Members - 1) pausing!.Pause();
        state->Written[member] = member + 1;
        Pool.Barrier(state->Barrier);
        state->Seen[member] = 1;
    }

    [Fact]
    public void PauseWhileATeamIsInFlightLetsTheTeamFinish()
    {
        using var pool = new Pool(ThreadPolicy.Default.WithThreads(4).WithSpinMicroseconds(0));
        var members = pool.Members;
        var written = (int*)NativeMemory.AllocZeroed((nuint)(members * sizeof(int)));
        var seen = (int*)NativeMemory.AllocZeroed((nuint)(members * sizeof(int)));
        var barrier = (TeamBarrier*)NativeMemory.AlignedAlloc((nuint)sizeof(TeamBarrier), 64);
        *barrier = default;
        pausing = pool;
        var hung = -1;
        try
        {
            for (var round = 0; round < 20 && hung < 0; round++)
            {
                Thread.Sleep(5);
                pool.Resume();
                Thread.Sleep(5);
                var context = new SumContext { Written = written, Seen = seen, Barrier = barrier, Members = members };
                var contextAddress = (nint)(&context);
                var team = new Thread(() => pool.RunTeam(&PauseThenRendezvous, (void*)contextAddress, barrier, members)) { IsBackground = true };
                team.Start();
                if (!team.Join(TimeSpan.FromSeconds(3)))
                {
                    hung = round;
                    pool.Resume();
                    team.Join(TimeSpan.FromSeconds(10));
                }
            }
        }
        finally
        {
            pool.Resume();
            pausing = null;
        }
        NativeMemory.Free(written);
        NativeMemory.Free(seen);
        NativeMemory.AlignedFree(barrier);
        Assert.True(hung < 0, $"a team that was already dispatched hung at its barrier once Pause() landed (round {hung}); its members must finish it");
    }

    [Fact]
    public void PoolDisposeIsIdempotent()
    {
        var pool = new Pool(2);
        pool.Dispose();
        pool.Dispose();
    }

    [Fact]
    public void NeedleServiceDisposeTwiceDoesNotThrow()
    {
        var service = new NeedleService();
        service.Dispose();
        var second = Xunit.Record.Exception(service.Dispose);
        Assert.Null(second);
    }

    [Fact]
    public void SpeechServiceDisposeTwiceDoesNotThrow()
    {
        var service = new SpeechService();
        service.Dispose();
        var second = Xunit.Record.Exception(service.Dispose);
        Assert.Null(second);
    }

    sealed class FakeHost : IDisposable
    {
        static readonly BindingFlags Any = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
        public readonly NeedleHost Host = new();
        public readonly NeedleChannel Channel;
        static int delayMicroseconds;

        public FakeHost(int requestCapacity, int replyCapacity, int runDelayMicroseconds = 0)
        {
            delayMicroseconds = runDelayMicroseconds;
            var type = typeof(NeedleHost);
            Channel = NeedleChannel.Allocate(requestCapacity, replyCapacity);
            var runnerType = type.GetNestedType("Runner", BindingFlags.NonPublic)!;
            var run = typeof(FakeHost).GetMethod(nameof(Answer), BindingFlags.Static | BindingFlags.NonPublic)!;
            type.GetField("runner", Any)!.SetValue(Host, Delegate.CreateDelegate(runnerType, run));
            type.GetField("typedHost", Any)!.SetValue(Host, true);
            type.GetField("maxBatch", Any)!.SetValue(Host, 16);
            type.GetField("started", Any)!.SetValue(Host, 1);
            type.GetField("channel", Any)!.SetValue(Host, Channel);
            var scratch = (nint)NativeMemory.AlignedAlloc((nuint)(16 * (ChannelLimits.CallsCapacity + 2 * ChannelLimits.TextCapacity)), 64);
            type.GetField("scratch", Any)!.SetValue(Host, Pointer.Box((void*)scratch, typeof(byte*)));
            var self = GCHandle.Alloc(Host);
            type.GetField("self", Any)!.SetValue(Host, self);
            Channel.SetWake(&Wake, (void*)GCHandle.ToIntPtr(self));
            var loop = type.GetMethod("Loop", Any)!;
            var worker = new Thread(() => loop.Invoke(Host, null)) { IsBackground = true, Name = "fake-needle-host" };
            type.GetField("worker", Any)!.SetValue(Host, worker);
            worker.Start();
        }

        static void Wake(void* state) =>
            typeof(NeedleHost).GetMethod("Pulse", Any)!.Invoke(GCHandle.FromIntPtr((nint)state).Target, null);

        static Reply Answer(ref ChannelRequest request, byte* bottom)
        {
            if (delayMicroseconds > 0) Thread.Sleep(TimeSpan.FromTicks(delayMicroseconds * 10L));
            return new Reply(ReplyStatus.Responded, FailureReason.None, ReplyFlags.None, 1f, 1f, 1f, 0, 0, 0, 0, request.Utf8Length, 0, 0, 0, 0);
        }

        public void Dispose() => Host.Dispose();
    }

    [Fact]
    public void HostServesEveryQueryWithoutLosingAWakeup()
    {
        using var fake = new FakeHost(8, 8);
        var channel = fake.Channel;
        var query = "q"u8.ToArray();
        var lost = -1;
        var deadline = Environment.TickCount64 + 8000;
        var round = 0;
        for (; Environment.TickCount64 < deadline && lost < 0; round++)
        {
            Assert.True(channel.TrySubmitQuery(query, out var ticket));
            if (!SharedChannel.WaitForReply(in channel, out var reply, 2000))
            {
                lost = round;
                channel.Wake();
                Assert.True(SharedChannel.WaitForReply(in channel, out reply, 2000));
            }
            Assert.Equal(ticket, reply.Ticket);
            if ((round & 1) == 0) Thread.SpinWait(50 + round % 200);
        }
        Assert.True(lost < 0, $"query {lost} of {round} sat in the ring with the host asleep; the submit and the host's park missed each other");
    }

    [Fact]
    public void HostDisposeReturnsWhileRepliesAreUncollected()
    {
        var fake = new FakeHost(8, 2);
        var channel = fake.Channel;
        var query = "q"u8.ToArray();
        for (var i = 0; i < 5; i++) Assert.True(channel.TrySubmitQuery(query, out _));
        var spin = Environment.TickCount64 + 5000;
        while (channel.HasPendingRequest && Environment.TickCount64 < spin) Thread.Sleep(1);
        Thread.Sleep(50);
        var dispose = Task.Run(fake.Dispose);
        var finished = dispose.Wait(TimeSpan.FromSeconds(3));
        if (!finished)
        {
            var drain = Environment.TickCount64 + 5000;
            while (!dispose.IsCompleted && Environment.TickCount64 < drain) channel.TryReceive(out _);
            dispose.Wait(TimeSpan.FromSeconds(5));
        }
        Assert.True(finished, "NeedleHost.Dispose must return even when the client stopped collecting replies");
    }

    [Fact]
    public void ChannelDeliversEveryTicketExactlyOnceUnderContention()
    {
        const int Producers = 3;
        const int Consumers = 3;
        const int PerProducer = 40_000;
        var channel = NeedleChannel.Allocate(16, 16);
        try
        {
            var seen = new int[Producers * PerProducer];
            var bad = 0;
            var consumed = 0;
            var producers = new Thread[Producers];
            var consumers = new Thread[Consumers];
            for (var p = 0; p < Producers; p++)
            {
                var id = p;
                producers[p] = new Thread(() =>
                {
                    var payload = new byte[8];
                    for (var i = 0; i < PerProducer; i++)
                    {
                        var value = id * PerProducer + i;
                        BitConverter.TryWriteBytes(payload, value);
                        BitConverter.TryWriteBytes(payload.AsSpan(4), ~value);
                        while (!channel.TrySubmitQuery(payload, out _)) Thread.SpinWait(8);
                    }
                }) { IsBackground = true };
            }
            for (var c = 0; c < Consumers; c++)
            {
                consumers[c] = new Thread(() =>
                {
                    while (Volatile.Read(ref consumed) < Producers * PerProducer)
                    {
                        if (!channel.TryClaim(out var request, out _)) { Thread.SpinWait(8); continue; }
                        var bytes = Utf8.AsSpan(ref request.Utf8, 8);
                        var value = BitConverter.ToInt32(bytes);
                        if (request.Utf8Length != 8 || BitConverter.ToInt32(bytes.Slice(4)) != ~value || (uint)value >= (uint)seen.Length)
                            Interlocked.Increment(ref bad);
                        else Interlocked.Increment(ref seen[value]);
                        Interlocked.Increment(ref consumed);
                    }
                }) { IsBackground = true };
            }
            foreach (var t in consumers) t.Start();
            foreach (var t in producers) t.Start();
            foreach (var t in producers) Assert.True(t.Join(TimeSpan.FromSeconds(60)));
            foreach (var t in consumers) Assert.True(t.Join(TimeSpan.FromSeconds(60)));
            Assert.Equal(0, bad);
            for (var i = 0; i < seen.Length; i++) Assert.True(seen[i] == 1, $"ticket payload {i} arrived {seen[i]} times");
        }
        finally { channel.Free(); }
    }

    static int woke;

    static void MarkWoken(void* state) => Volatile.Write(ref woke, 1);

    [Fact]
    public void ChannelParkHandshakeNeverMissesASubmit()
    {
        const int Rounds = 400_000;
        var channel = NeedleChannel.Allocate(4, 4);
        var payload = new byte[1];
        var go = 0;
        var done = 0;
        var parkerSawWork = new bool[Rounds];
        var missed = 0;
        try
        {
            channel.SetWake(&MarkWoken, null);
            var parker = new Thread(() =>
            {
                for (var r = 0; r < Rounds; r++)
                {
                    while (Volatile.Read(ref go) <= r) { }
                    channel.SetSleeping(1);
                    parkerSawWork[r] = channel.HasPendingRequest;
                    Interlocked.Increment(ref done);
                    while (Volatile.Read(ref done) < 2 * (r + 1)) { }
                }
            }) { IsBackground = true };
            parker.Start();
            var deadline = Environment.TickCount64 + 20_000;
            for (var r = 0; r < Rounds && Environment.TickCount64 < deadline; r++)
            {
                Volatile.Write(ref woke, 0);
                channel.SetSleeping(0);
                Interlocked.Increment(ref go);
                Assert.True(channel.TrySubmitQuery(payload, out _));
                while (Volatile.Read(ref done) < 2 * r + 1) { }
                if (!parkerSawWork[r] && Volatile.Read(ref woke) == 0) missed++;
                Assert.True(channel.TryClaim(out _, out _));
                Interlocked.Increment(ref done);
            }
            Interlocked.Exchange(ref go, int.MaxValue);
            Interlocked.Exchange(ref done, int.MaxValue);
            parker.Join(TimeSpan.FromSeconds(5));
        }
        finally { channel.Free(); }
        Assert.True(missed == 0, $"{missed} submits found the consumer awake while the consumer, having just announced it was going to sleep, found the ring empty: a lost wakeup each time");
    }
}
