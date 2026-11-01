using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using Xunit;

namespace Needle.Tests;

public sealed class ChannelTests : IDisposable
{
    const int Capacity = 4;

    readonly NeedleChannel channel;

    public ChannelTests() => channel = NeedleChannel.Allocate(Capacity, Capacity);

    public void Dispose() => channel.Free();

    readonly struct DisposableChannel : IDisposable
    {
        public readonly NeedleChannel Channel;
        public DisposableChannel(NeedleChannel channel) => Channel = channel;
        public void Dispose() => Channel.Free();
    }

    [Fact]
    public void SubmitThenClaimAndReplyRoundTripsThroughTheRing()
    {
        Assert.True(channel.TrySubmitQuery("what's it like in Lagos right now?"u8, out var ticket));
        Assert.True(channel.TrySubmitToolResult("{\"city\":\"Lagos\"}"u8, out var toolTicket));
        Assert.True(channel.HasPendingRequest);
        Assert.False(channel.HasCompletedReply);

        Assert.True(channel.TryClaim(out var request, out var claimedTicket));
        Assert.Equal(ticket, claimedTicket);
        Assert.Equal(ChannelRequestKind.Query, request.Kind);
        Assert.True("what's it like in Lagos right now?"u8.SequenceEqual(Utf8.AsSpan(ref request.Utf8, request.Utf8Length)));

        Assert.True(channel.TryClaim(out request, out claimedTicket));
        Assert.Equal(toolTicket, claimedTicket);
        Assert.Equal(ChannelRequestKind.ToolResult, request.Kind);

        ChannelReply reply = default;
        reply.Ticket = ticket;
        reply.Reply = reply.Reply with { Utf8Length = 7 };
        "{\"x\":1}"u8.CopyTo(Utf8.AsSpan(ref reply.Calls, reply.Reply.Utf8Length));
        Assert.True(channel.TryPublishReply(in reply));

        Assert.True(channel.TryReceive(out var received));
        Assert.Equal(ticket, received.Ticket);
        Assert.True("{\"x\":1}"u8.SequenceEqual(Utf8.AsSpan(ref received.Calls, received.Reply.Utf8Length)));
        Assert.False(channel.HasPendingRequest);
        Assert.False(channel.HasCompletedReply);
        Assert.False(channel.TryReceive(out _));
    }

    [Fact]
    public void RingIsBoundedAndRejectsWhenFull()
    {
        Span<byte> payload = stackalloc byte[1];
        for (var i = 0; i < Capacity; i++)
        {
            payload[0] = (byte)"abcdefgh"[i];
            Assert.True(channel.TrySubmitQuery(payload, out _));
        }
        Assert.False(channel.TrySubmitQuery("overflow"u8, out _));

        for (var i = 0; i < Capacity; i++) Assert.True(channel.TryClaim(out _, out _));
        Assert.False(channel.TryClaim(out _, out _));

        Assert.True(channel.TrySubmitQuery("fits again"u8, out _));
    }

    [Fact]
    public void RingPositionsWrapWithoutCorruption()
    {
        var roundTrips = 10_000;
        Span<byte> payload = stackalloc byte[8];
        for (var i = 0; i < roundTrips; i++)
        {
            BinaryPrimitives.WriteInt64LittleEndian(payload, i);
            Assert.True(channel.TrySubmitQuery(payload, out var ticket));
            Assert.True(channel.TryClaim(out var request, out _));
            Assert.Equal(i, BinaryPrimitives.ReadInt64LittleEndian(Utf8.AsSpan(ref request.Utf8, request.Utf8Length)));

            ChannelReply reply = default;
            reply.Ticket = ticket;
            Assert.True(channel.TryPublishReply(in reply));
            Assert.True(channel.TryReceive(out var received));
            Assert.Equal((ulong)i, received.Ticket);
        }
    }

    [Fact]
    public void TicketsAreMonotonicAndConsumedInSubmissionOrder()
    {
        using var big = new DisposableChannel(NeedleChannel.Allocate(1024, 1024));
        var channel = big.Channel;
        Span<byte> payload = stackalloc byte[1];
        for (var i = 0; i < 1_000; i++)
        {
            payload[0] = (byte)i;
            Assert.True(channel.TrySubmitQuery(payload, out var ticket));
        }
        for (var i = 0; i < 1_000; i++)
        {
            Assert.True(channel.TryClaim(out _, out var ticket));
            Assert.Equal((ulong)i, ticket);
        }
    }

    [Fact]
    public void MultipleProducersAndConsumersLoseNoTickets()
    {
        const int producers = 4, consumers = 4, perProducer = 262_144;
        var stress = NeedleChannel.Allocate(256, 256);
        try
        {
            var remaining = producers * perProducer;
            var seen = new int[producers * perProducer];
            string? failure = null;
            var producerThreads = new Thread[producers];
            var consumerThreads = new Thread[consumers];

            for (var p = 0; p < producers; p++)
            {
                var producer = p;
                producerThreads[p] = new Thread(() =>
                {
                    try
                    {
                        var spin = new SpinWait();
                        Span<byte> payload = stackalloc byte[12];
                        for (long seq = 1; seq <= perProducer; seq++)
                        {
                            BinaryPrimitives.WriteInt32LittleEndian(payload, producer);
                            BinaryPrimitives.WriteInt64LittleEndian(payload.Slice(4), seq);
                            while (!stress.TrySubmitQuery(payload, out _)) spin.SpinOnce();
                        }
                    }
                    catch (Exception e)
                    {
                        Interlocked.CompareExchange(ref failure, e.ToString(), null);
                        Volatile.Write(ref remaining, 0);
                    }
                }) { IsBackground = true };
            }

            for (var c = 0; c < consumers; c++)
            {
                consumerThreads[c] = new Thread(() =>
                {
                    try
                    {
                        var spin = new SpinWait();
                        while (true)
                        {
                            if (stress.TryClaim(out var request, out _))
                            {
                                var payload = Utf8.AsSpan(ref request.Utf8, request.Utf8Length);
                                var producer = BinaryPrimitives.ReadInt32LittleEndian(payload);
                                var seq = BinaryPrimitives.ReadInt64LittleEndian(payload.Slice(4));
                                var index = producer * perProducer + (int)(seq - 1);
                                if (Interlocked.Exchange(ref seen[index], 1) != 0)
                                {
                                    Interlocked.CompareExchange(ref failure, $"producer {producer} seq {seq} delivered twice", null);
                                    Volatile.Write(ref remaining, 0);
                                    return;
                                }
                                if (Interlocked.Decrement(ref remaining) == 0) return;
                            }
                            else if (Volatile.Read(ref remaining) == 0) return;
                            else spin.SpinOnce();
                        }
                    }
                    catch (Exception e)
                    {
                        Interlocked.CompareExchange(ref failure, e.ToString(), null);
                        Volatile.Write(ref remaining, 0);
                    }
                }) { IsBackground = true };
            }

            foreach (var thread in consumerThreads) thread.Start();
            foreach (var thread in producerThreads) thread.Start();
            foreach (var thread in producerThreads) Assert.True(thread.Join(60_000), "producer did not finish");
            foreach (var thread in consumerThreads) Assert.True(thread.Join(60_000), "consumer did not finish");

            Assert.Null(failure);
            Assert.Equal(0, Volatile.Read(ref remaining));
        }
        finally
        {
            stress.Free();
        }
    }

    [Fact]
    public void CancellationFlagsAreVisiblePerTicket()
    {
        Assert.True(channel.TrySubmitQuery("one"u8, out var first));
        Assert.True(channel.TrySubmitQuery("two"u8, out var second));
        Assert.True(channel.TrySubmitQuery("three"u8, out var third));

        channel.Cancel(second);
        channel.Cancel(second);

        Assert.False(channel.IsCancelled(first));
        Assert.True(channel.IsCancelled(second));
        Assert.False(channel.IsCancelled(third));
    }

    [Fact]
    public void PayloadLongerThanCapacityIsRejected()
    {
        Assert.False(channel.TrySubmitQuery(new byte[ChannelLimits.QueryCapacity + 1], out var ticket));
        Assert.Equal(0UL, ticket);
    }

    [Fact]
    public unsafe void SubmitClaimPublishAndReceiveAllocateNothing()
    {
        for (var i = 0; i < 1_000; i++) Assert.True(RoundTrip(i));

        var before = GC.GetAllocatedBytesForCurrentThread();
        var ok = true;
        for (var i = 0; i < 10_000; i++) ok &= RoundTrip(i);
        Assert.Equal(before, GC.GetAllocatedBytesForCurrentThread());
        Assert.True(ok);
    }

    bool RoundTrip(int sequence)
    {
        Span<byte> payload = stackalloc byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(payload, sequence);
        if (!channel.TrySubmitQuery(payload, out _)) return false;
        if (!channel.TryClaim(out var request, out var ticket)) return false;
        if (sequence != BinaryPrimitives.ReadInt64LittleEndian(Utf8.AsSpan(ref request.Utf8, request.Utf8Length))) return false;

        ChannelReply reply = default;
        reply.Ticket = ticket;
        if (!channel.TryPublishReply(in reply)) return false;
        if (!channel.TryReceive(out var received)) return false;
        return ticket == received.Ticket;
    }

    [Fact]
    public void ChannelStructsCarryNoManagedReferences()
    {
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<ChannelRequest>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<ChannelReply>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<NeedleChannel>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<Reply>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<QueryUtf8>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<CallsUtf8>());
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<TextUtf8>());

        var visited = new HashSet<Type>();
        foreach (var type in new[] { typeof(ChannelRequest), typeof(ChannelReply), typeof(NeedleChannel), typeof(Reply), typeof(QueryUtf8), typeof(CallsUtf8), typeof(TextUtf8) })
            AssertUnmanaged(type, visited);
    }

    static void AssertUnmanaged(Type type, HashSet<Type> visited)
    {
        if (!visited.Add(type)) return;
        Assert.True(type.IsValueType || type.IsPointer, $"{type.Name} must be unmanaged");
        if (type.IsPrimitive || type.IsEnum || type.IsPointer) return;
        foreach (var field in type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
            AssertUnmanaged(field.FieldType, visited);
    }

    [Fact]
    public void HostServesRealToolCallsThroughTheChannelAndShutsDownWithPendingWork()
    {
        Assets.Require();
        using var host = new NeedleHost();
        var channel = host.Start(new HostOptions
        {
            ModelPath = Assets.Model,
            ToolsJson = File.ReadAllText(Assets.Tools),
            Session = new NeedleOptions { Threads = 2 },
        });

        Assert.True(channel.TrySubmitQuery("what's it like in Lagos right now?"u8, out var queryTicket));
        Assert.True(WaitForReply(channel, queryTicket, out var call));
        Assert.Equal(ReplyStatus.Called, call.Reply.Status);
        Assert.Equal(1, call.Reply.CallCount);
        var calls = Encoding.UTF8.GetString(Utf8.AsSpan(ref call.Calls, call.Reply.Utf8Length));
        Assert.Contains("get_weather", calls);

        Assert.True(channel.TrySubmitToolResult("{\"city\":\"Lagos\",\"temp_c\":27,\"sky\":\"clear\"}"u8, out var resultTicket));
        Assert.True(WaitForReply(channel, resultTicket, out var answered));
        Assert.Equal(ReplyStatus.Responded, answered.Reply.Status);
        Assert.Equal(0, answered.Reply.CallCount);
        Assert.True(answered.Reply.ResponseLength > 0);
        var response = Encoding.UTF8.GetString(Utf8.AsSpan(ref answered.Response, answered.Reply.ResponseLength));
        Assert.Contains("27", response);

        Assert.True(channel.TrySubmitQuery("set a timer for 12 minutes"u8, out _));
        host.Cancel(queryTicket);
        host.Dispose();
        Assert.Throws<InvalidOperationException>(() => _ = host.Start(default));
    }

    [Fact]
    public void HostSubmitWakeAndReceiveAllocateNothingOnTheCallerThread()
    {
        Assets.Require();
        using var host = new NeedleHost();
        var channel = host.Start(new HostOptions { ModelPath = Assets.Model, ToolsJson = File.ReadAllText(Assets.Tools) });
        Thread.Sleep(200);

        var requests = 16;
        Span<ulong> tickets = stackalloc ulong[requests];
        var before = GC.GetAllocatedBytesForCurrentThread();
        var submitted = true;
        for (var i = 0; i < requests; i++)
        {
            submitted &= channel.TrySubmitQuery("ping"u8, out tickets[i]);
            host.Cancel(tickets[i]);
        }
        Assert.Equal(before, GC.GetAllocatedBytesForCurrentThread());
        Assert.True(submitted);

        var spin = new SpinWait();
        var received = 0;
        var ordered = true;
        var statuses = true;
        var receiveBefore = GC.GetAllocatedBytesForCurrentThread();
        while (received < requests)
        {
            if (channel.TryReceive(out var reply))
            {
                ordered &= tickets[received] == reply.Ticket;
                statuses &= reply.Reply.Status is ReplyStatus.Cancelled or ReplyStatus.Called;
                received++;
            }
            else spin.SpinOnce();
        }
        Assert.Equal(receiveBefore, GC.GetAllocatedBytesForCurrentThread());
        Assert.True(ordered);
        Assert.True(statuses);
    }

    static bool WaitForReply(NeedleChannel channel, ulong ticket, out ChannelReply reply)
    {
        var deadline = Environment.TickCount64 + 120_000;
        var spin = new SpinWait();
        while (Environment.TickCount64 < deadline)
        {
            if (channel.TryReceive(out reply) && reply.Ticket == ticket) return true;
            spin.SpinOnce();
        }
        reply = default;
        return false;
    }
}
