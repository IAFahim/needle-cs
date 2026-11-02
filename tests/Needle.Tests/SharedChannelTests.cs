using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Xunit;
using Xunit.Abstractions;

namespace Needle.Tests;

// N71: shared-memory NeedleChannel. Two mappings of one backing file act as
// two processes (each view rebuilds ring pointers over its own base
// address); the ping-pong test measures the zero-copy round trip.
public sealed unsafe class SharedChannelTests
{
    readonly ITestOutputHelper output;

    public SharedChannelTests(ITestOutputHelper output) => this.output = output;

    [Fact]
    public void TwoViewsOfOneMappingRoundTripRequestsAndReplies()
    {
        var path = SharedChannel.NewPath();
        try
        {
            using var server = SharedChannel.Create(path, 16, 16);
            using var client = SharedChannel.Open(path);
            Assert.True(server.Channel.IsShared);
            Assert.True(client.Channel.IsShared);
            output.WriteLine($"server base=0x{(ulong)server.Channel.Identity:X} client base=0x{(ulong)client.Channel.Identity:X}");

            Assert.True(client.Channel.TrySubmitQuery("shared hello"u8, out var ticket));
            Assert.True(SharedChannel.WaitForRequest(server.Channel, 5000));

            Assert.True(server.Channel.TryClaim(out var request, out var claimed));
            Assert.Equal(ticket, claimed);
            Assert.Equal(ChannelRequestKind.Query, request.Kind);
            Assert.True("shared hello"u8.SequenceEqual(Utf8.AsSpan(ref request.Utf8, request.Utf8Length)));

            ChannelReply reply = default;
            reply.Ticket = claimed;
            Assert.True(server.Channel.TryPublishReply(in reply));
            Assert.True(SharedChannel.WaitForReply(client.Channel, out var received, 5000));
            Assert.Equal(ticket, received.Ticket);
        }
        finally
        {
            DeleteQuiet(path);
        }
    }

    [Fact]
    public void SharedSubmitIgnoresStaleWakePointer()
    {
        var path = SharedChannel.NewPath();
        try
        {
            using var server = SharedChannel.Create(path, 4, 4);
            using var client = SharedChannel.Open(path);
            // Simulate the foreign address space: the wake slots hold garbage
            // that must never be called. (On a local channel this would crash.)
            client.Channel.SetWake((delegate*<void*, void>)unchecked((nint)0xDEADBEEFu), (void*)0x1234);
            Assert.True(client.Channel.TrySubmitQuery("no wake"u8, out _));
            Assert.True(server.Channel.TryClaim(out _, out _));
        }
        finally
        {
            DeleteQuiet(path);
        }
    }

    [Fact]
    public void CancelPropagatesAcrossViews()
    {
        var path = SharedChannel.NewPath();
        try
        {
            using var server = SharedChannel.Create(path, 4, 4);
            using var client = SharedChannel.Open(path);
            Assert.True(client.Channel.TrySubmitQuery("cancel me"u8, out var ticket));
            client.Channel.Cancel(ticket);
            Assert.True(server.Channel.IsCancelled(ticket));
        }
        finally
        {
            DeleteQuiet(path);
        }
    }

    [Fact]
    public void OpenMissingPathThrows()
    {
        Assert.Throws<FileNotFoundException>(() => SharedChannel.Open(SharedChannel.NewPath("needle-nope")));
    }

    [Fact]
    public void PingPongRoundTripStaysUnderTwentyMicroseconds()
    {
        var path = SharedChannel.NewPath();
        try
        {
            using var server = SharedChannel.Create(path, 64, 64);
            using var client = SharedChannel.Open(path);
            var serverChannel = server.Channel;
            var clientChannel = client.Channel;
            using var stop = new ManualResetEventSlim(false);
            var serverError = null as Exception;
            var serverThread = new Thread(() =>
            {
                try
                {
                    while (!stop.IsSet)
                    {
                        if (!serverChannel.TryClaim(out _, out var claimed)) { Thread.Yield(); continue; }
                        ChannelReply reply = default;
                        reply.Ticket = claimed;
                        while (!serverChannel.TryPublishReply(in reply)) Thread.Yield();
                    }
                }
                catch (Exception e) { serverError = e; }
            })
            { IsBackground = true };
            serverThread.Start();

            const int warmup = 1000;
            const int measured = 20000;
            var payload = new byte[] { (byte)'p', (byte)'i', (byte)'n', (byte)'g' };
            for (var i = 0; i < warmup; i++)
            {
                Assert.True(clientChannel.TrySubmitQuery(payload, out _));
                Assert.True(SharedChannel.WaitForReply(clientChannel, out _, 5000));
            }
            var elapsed = new long[measured];
            var frequency = (double)Stopwatch.Frequency;
            for (var i = 0; i < measured; i++)
            {
                var start = Stopwatch.GetTimestamp();
                Assert.True(clientChannel.TrySubmitQuery(payload, out _));
                Assert.True(SharedChannel.WaitForReply(clientChannel, out _, 5000));
                elapsed[i] = Stopwatch.GetTimestamp() - start;
            }
            stop.Set();
            serverThread.Join();
            Assert.Null(serverError);

            Array.Sort(elapsed);
            var medianUs = elapsed[measured / 2] * 1_000_000.0 / frequency;
            var p99Us = elapsed[(int)(measured * 0.99)] * 1_000_000.0 / frequency;
            var maxUs = elapsed[measured - 1] * 1_000_000.0 / frequency;
            output.WriteLine($"shared ping-pong: n={measured} median={medianUs:F2}us p99={p99Us:F2}us max={maxUs:F2}us (Stopwatch {frequency:F0}Hz, {Environment.ProcessorCount} cores)");
            Assert.True(medianUs <= 20.0, $"median round trip {medianUs:F2}us exceeds 20us");
        }
        finally
        {
            DeleteQuiet(path);
        }
    }

    static void DeleteQuiet(string path)
    {
        try { File.Delete(path); } catch { }
    }
}
