using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using Needle.Cli;
using Xunit;

namespace Needle.Tests.Bughunt;

public sealed class HostBughuntTests
{
    const int Rounds = 3;

    static NeedleOptions RetrievingGameOptions => new() { Threads = 4, RetrievalTopK = 5, AutoDate = false, Stateless = true };

    static IEnumerable<string> GameAsks()
    {
        for (var round = 0; round < Rounds; round++)
            foreach (var prompt in BenchPrompts.Game)
                yield return prompt;
    }

    [Fact]
    public void StatelessRetrievingSessionNeverRejectsATokenItsMaskAllowed()
    {
        Assets.Require();
        using var model = Model.Open(Assets.Model);
        using var session = new NeedleSession(model, BenchPrompts.GameToolsJson, RetrievingGameOptions);
        var calls = new byte[8192];
        var failures = new List<string>();
        foreach (var prompt in GameAsks())
        {
            var reply = session.Complete(Encoding.UTF8.GetBytes(prompt), calls);
            if (reply.Failure is FailureReason.GrammarRejectedToken or FailureReason.GrammarDeadEnd)
                failures.Add($"{prompt}: {reply.Failure}");
        }
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    [Fact]
    public void StatelessRetrievingHostNeverRejectsATokenItsMaskAllowed()
    {
        Assets.Require();
        using var host = new NeedleHost();
        var channel = host.Start(new HostOptions
        {
            ModelPath = Assets.Model,
            ToolsJson = BenchPrompts.GameToolsJson,
            Session = RetrievingGameOptions,
            StatelessQueries = true,
            MaxBatchSize = 1,
        });
        var failures = new List<string>();
        foreach (var prompt in GameAsks())
        {
            Assert.True(channel.TrySubmitQuery(Encoding.UTF8.GetBytes(prompt), out var ticket));
            Assert.True(WaitForReply(channel, ticket, out var reply), $"{prompt}: no reply");
            if (reply.Reply.Failure is FailureReason.GrammarRejectedToken or FailureReason.GrammarDeadEnd)
                failures.Add($"{prompt}: {reply.Reply.Failure}");
        }
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

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
}
