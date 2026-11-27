using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Needle.Unity.PlayMode.Tests
{
    // The setup that showed GrammarRejectedToken in the Editor (issue #74): a host with stateless
    // queries, the Game thread policy and tools/game_tools.json, whose seven tools are more than
    // RetrievalTopK, so the grammar is rebuilt whenever retrieval picks a different subset. A rebuilt
    // grammar that lands on a freed one's address must not get the freed one's cached token masks.
    // Whether the allocator reuses an address varies from run to run, so the test asks every game
    // query NEEDLE_REPRO_ROUNDS times (default 3) and fails if any answer was a grammar failure.
    public sealed class GrammarRegressionTests
    {
        static readonly string[] Queries = new string[]
        {
            "move to 120, 45",
            "attack the orc chieftain",
            "give Lyra 3 potions",
            "switch to wedge formation and follow the captain",
            "stop everything",
            "give 50 gold to Borin then move to 10, 20",
            "follow Mira",
            "form a circle",
        };

        NeedleHost host;

        [TearDown]
        public void TearDown()
        {
            host?.Dispose();
            host = null;
        }

        [UnityTest]
        public IEnumerator StatelessRetrievingHostNeverRejectsATokenItsMaskAllowed()
        {
            string model = PlayModeAssets.Require(PlayModeAssets.FromEnvOrRepo("NEEDLE_MODEL", "ref/hf/needle3.cact"), "Needle 3 model (needle3.cact)");
            string tools = File.ReadAllText(PlayModeAssets.Require(PlayModeAssets.FromEnvOrRepo("NEEDLE_GAME_TOOLS", "tools/game_tools.json"), "Game tool set (tools/game_tools.json)"));
            int rounds = 3;
            int parsed;
            if (int.TryParse(Environment.GetEnvironmentVariable("NEEDLE_REPRO_ROUNDS"), out parsed) && parsed > 0) rounds = parsed;

            host = new NeedleHost();
            NeedleChannel channel = host.Start(new HostOptions
            {
                ModelPath = model,
                ToolsJson = tools,
                StatelessQueries = true,
                Session = new NeedleOptions { Threading = ThreadPolicy.Game.WithThreads(PlayModeAssets.Threads), RetrievalTopK = 5 },
            });

            int asks = 0, grammarFailures = 0, otherFailures = 0;
            var failures = new StringBuilder();
            for (int round = 0; round < rounds; round++)
            {
                for (int q = 0; q < Queries.Length; q++)
                {
                    Assert.IsTrue(channel.TrySubmitQuery(Encoding.UTF8.GetBytes(Queries[q]), out ulong ticket), "the request ring refused " + Queries[q]);
                    var wall = Stopwatch.StartNew();
                    ChannelReply reply = default;
                    while (!channel.TryReceive(out reply))
                    {
                        if (wall.Elapsed > TimeSpan.FromMinutes(2)) Assert.Fail("no reply to '" + Queries[q] + "' within two minutes");
                        yield return null;
                    }
                    Assert.AreEqual(ticket, reply.Ticket);
                    asks++;
                    FailureReason why = reply.Reply.Failure;
                    if (why == FailureReason.GrammarRejectedToken || why == FailureReason.GrammarDeadEnd)
                    {
                        grammarFailures++;
                        failures.Append("round ").Append(round + 1).Append(", '").Append(Queries[q]).Append("': ").Append(why).Append("; ");
                    }
                    else if (reply.Reply.Status == ReplyStatus.Failed) otherFailures++;
                }
            }

            UnityEngine.Debug.Log("[needle-repro] grammar failures " + grammarFailures + " of " + asks + " asks (" + otherFailures + " other failures): " + failures);
            Assert.AreEqual(0, grammarFailures, "grammar failures in " + asks + " asks: " + failures);
        }
    }
}
