using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Xunit;

namespace Needle.Tests;

public sealed unsafe class TypedHostTests
{
    static bool WaitForReply(NeedleChannel channel, ulong ticket, out ChannelReply reply)
    {
        var start = Environment.TickCount;
        while (Environment.TickCount - start < 120000)
        {
            while (channel.TryReceive(out var candidate))
                if (candidate.Ticket == ticket)
                {
                    reply = candidate;
                    return true;
                }
            Thread.Sleep(1);
        }
        reply = default;
        return false;
    }

    [Fact]
    public void TypedHostRoundTripsCommandBytesThroughTheChannel()
    {
        Assets.Require();
        using var host = new NeedleHost();
        var channel = host.StartTyped<GameSink, GameCommand>(new HostOptions
        {
            ModelPath = Assets.Model,
            ToolsJson = File.ReadAllText(Assets.Tools),
            Session = new NeedleOptions { Threads = 2 },
        });

        Assert.True(channel.TrySubmitQuery("what's it like in Lagos right now?"u8, out var ticket));
        Assert.True(WaitForReply(channel, ticket, out var reply));
        Assert.Equal(ticket, reply.Ticket);
        Assert.Equal(ReplyStatus.Called, reply.Reply.Status);
        Assert.Equal(1, reply.Reply.CallCount);
        Assert.Equal(Marshal.SizeOf<GameCommand>(), reply.Reply.Utf8Length);
        var bytes = MemoryMarshal.CreateSpan(ref reply.Calls[0], reply.Reply.Utf8Length);
        var command = MemoryMarshal.Read<GameCommand>(bytes);
        Assert.Equal(GameCommandKind.GetWeather, command.Kind);
    }

    [Fact]
    public void StatelessQueriesAnswerEveryOrderAsIfItWereTheFirst()
    {
        Assets.Require();
        using var host = new NeedleHost();
        var channel = host.StartTyped<GameSink, GameCommand>(new HostOptions
        {
            ModelPath = Assets.Model,
            ToolsJson = File.ReadAllText(Assets.Tools),
            Session = new NeedleOptions { Threads = 2 },
            StatelessQueries = true,
        });

        var firstConfidence = 0f;
        var firstCommand = default(GameCommand);
        // An unrelated order in between must not leak into the repeat of the first one.
        foreach (var query in new[] { "what's it like in Lagos right now?"u8.ToArray(), "set a timer for 12 minutes"u8.ToArray(), "what's it like in Lagos right now?"u8.ToArray() })
        {
            Assert.True(channel.TrySubmitQuery(query, out var ticket));
            Assert.True(WaitForReply(channel, ticket, out var reply));
            Assert.Equal(ReplyStatus.Called, reply.Reply.Status);
            var command = MemoryMarshal.Read<GameCommand>(MemoryMarshal.CreateSpan(ref reply.Calls[0], reply.Reply.Utf8Length));
            if (firstConfidence == 0f) { firstConfidence = reply.Reply.Confidence; firstCommand = command; continue; }
            if (query.Length == "set a timer for 12 minutes".Length) continue;
            Assert.Equal(firstConfidence, reply.Reply.Confidence);
            Assert.Equal(firstCommand.Kind, command.Kind);
        }
    }

    [Fact]
    public void TypedHostReportsZeroCommandBytesOnEmptyInput()
    {
        Assets.Require();
        using var host = new NeedleHost();
        var channel = host.StartTyped<DeviceSink, DeviceCommand>(new HostOptions
        {
            ModelPath = Assets.Model,
            ToolsJson = Encoding.UTF8.GetString(DeviceCommands.Schema.ToArray()),
            Session = new NeedleOptions { Threads = 1 },
        });

        Assert.True(channel.TrySubmitQuery(ReadOnlySpan<byte>.Empty, out var ticket));
        Assert.True(WaitForReply(channel, ticket, out var reply));
        Assert.Equal(ticket, reply.Ticket);
        Assert.Equal(ReplyStatus.Failed, reply.Reply.Status);
        Assert.Equal(FailureReason.EmptyInput, reply.Reply.Failure);
        Assert.Equal(0, reply.Reply.CallCount);
        Assert.Equal(0, reply.Reply.Utf8Length);
    }
}
