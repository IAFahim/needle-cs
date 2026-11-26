using System;
using System.Diagnostics;
using System.Text;
using NUnit.Framework;
using Unity.Collections;
using Unity.Entities;

namespace Needle.Unity.Tests
{
    // Drives the Burst NeedleSystem in a private World. The first tests play the worker side
    // themselves, so they need no model; the last one runs a real NeedleHost.
    public sealed unsafe class NeedleSystemTests
    {
        World world;
        NeedleChannel channel;
        bool ownsChannel;
        NeedleHost host;

        [SetUp]
        public void SetUp()
        {
            world = new World("Needle test world");
            DefaultWorldInitialization.AddSystemsToRootLevelSystemGroups(world,
                typeof(NeedleSystem), typeof(EndSimulationEntityCommandBufferSystem));
        }

        [TearDown]
        public void TearDown()
        {
            if (world != null && world.IsCreated)
            {
                world.EntityManager.CompleteAllTrackedJobs();
                world.Dispose();
            }
            world = null;
            host?.Dispose();
            host = null;
            if (ownsChannel) channel.Free();
            ownsChannel = false;
        }

        EntityManager Manager => world.EntityManager;

        void BindChannel(NeedleChannel bound)
        {
            channel = bound;
            Manager.CreateSingleton(new NeedleChannelHandle { Channel = bound, Identity = bound.Identity });
        }

        Entity CreateAsker(string query)
        {
            Entity entity = Manager.CreateEntity(typeof(NeedlePending));
            Manager.SetComponentEnabled<NeedlePending>(entity, false);
            Manager.AddComponentData(entity, new NeedleAsk { Query = new FixedString512Bytes(query) });
            return entity;
        }

        void Tick()
        {
            world.Update();
            Manager.CompleteAllTrackedJobs();
        }

        [Test]
        public void SubmitsAsksAndMarksThemPending()
        {
            BindChannel(NeedleChannel.Allocate(8, 8));
            ownsChannel = true;
            Entity asker = CreateAsker("attack the watchtower near the river");

            Tick();

            Assert.IsTrue(channel.HasPendingRequest, "NeedleSystem did not submit the ask");
            Assert.IsTrue(Manager.IsComponentEnabled<NeedlePending>(asker));
            Assert.IsTrue(channel.TryClaim(out ChannelRequest request, out _));
            Assert.AreEqual("attack the watchtower near the river", ChannelTests.QueryText(in request));

            Tick();
            Assert.IsFalse(channel.HasPendingRequest, "a pending ask was submitted twice");
        }

        [Test]
        public void DeliversRepliesAsOutcomes()
        {
            BindChannel(NeedleChannel.Allocate(8, 8));
            ownsChannel = true;
            Entity first = CreateAsker("move to the gate");
            Entity second = CreateAsker("stop");
            Tick();

            int claimed = 0;
            while (channel.TryClaim(out ChannelRequest request, out ulong ticket))
            {
                string query = ChannelTests.QueryText(in request);
                string call = query == "stop" ? "{\"name\":\"stop\",\"arguments\":{}}" : "{\"name\":\"move_to\",\"arguments\":{\"x\":1,\"y\":2}}";
                ChannelTests.Publish(channel, ticket, ReplyStatus.Called, call, query == "stop" ? 0.75f : 0.95f);
                claimed++;
            }
            Assert.AreEqual(2, claimed);

            Tick();

            foreach (var entity in new[] { first, second })
            {
                Assert.IsTrue(Manager.HasComponent<NeedleOutcome>(entity), "no NeedleOutcome after the reply arrived");
                Assert.IsFalse(Manager.HasComponent<NeedleAsk>(entity), "NeedleAsk was not removed");
                Assert.IsFalse(Manager.IsComponentEnabled<NeedlePending>(entity), "NeedlePending stayed enabled");
                var outcome = Manager.GetComponentData<NeedleOutcome>(entity);
                Assert.AreEqual(ReplyStatus.Called, outcome.Status);
                Assert.AreEqual(1, outcome.CallCount);
            }
            Assert.AreEqual(0.75f, Manager.GetComponentData<NeedleOutcome>(second).Confidence, 1e-6f);
            var calls = Manager.GetComponentData<NeedleOutcome>(first).Calls;
            StringAssert.Contains("move_to", Encoding.UTF8.GetString(Utf8.AsSpan(ref calls, 64)));
        }

        [Test]
        public void IgnoresRepliesForUnknownTickets()
        {
            BindChannel(NeedleChannel.Allocate(8, 8));
            ownsChannel = true;
            Entity asker = CreateAsker("hold position");
            Tick();
            Assert.IsTrue(channel.TryClaim(out _, out ulong ticket));
            ChannelTests.Publish(channel, ticket + 1000, ReplyStatus.Called, "{\"name\":\"stop\",\"arguments\":{}}");

            Tick();

            Assert.IsFalse(Manager.HasComponent<NeedleOutcome>(asker));
            Assert.IsTrue(Manager.IsComponentEnabled<NeedlePending>(asker));
        }

        [Test]
        public void DoesNothingWithoutABoundChannel()
        {
            Manager.CreateSingleton(default(NeedleChannelHandle));
            Entity asker = CreateAsker("hello");
            Tick();
            Assert.IsFalse(Manager.IsComponentEnabled<NeedlePending>(asker));
            Assert.IsTrue(Manager.HasComponent<NeedleAsk>(asker));
        }

        [Test]
        public void RealHostAnswersThroughTheSystem()
        {
            string model = NeedleTestAssets.RequireModel();
            string tools = System.IO.File.ReadAllText(NeedleTestAssets.RequireTools());
            host = new NeedleHost();
            BindChannel(host.Start(new HostOptions
            {
                ModelPath = model,
                ToolsJson = tools,
                StatelessQueries = true,
                Session = new NeedleOptions { Threading = ThreadPolicy.Game.WithThreads(NeedleTestAssets.Threads) },
            }));
            Entity asker = CreateAsker("what's it like in Lagos right now?");

            var clock = Stopwatch.StartNew();
            while (!Manager.HasComponent<NeedleOutcome>(asker))
            {
                if (clock.Elapsed > TimeSpan.FromMinutes(2)) Assert.Fail("no reply within two minutes");
                channel.Wake();
                Tick();
                System.Threading.Thread.Sleep(5);
            }

            var outcome = Manager.GetComponentData<NeedleOutcome>(asker);
            var calls = outcome.Calls;
            string text = Encoding.UTF8.GetString(Utf8.AsSpan(ref calls, ChannelLimits.CallsCapacity)).TrimEnd('\0');
            UnityEngine.Debug.Log("[needle] system reply " + outcome.Status + " confidence=" + outcome.Confidence + " calls=" + text + " after " + clock.ElapsedMilliseconds + " ms");
            Assert.AreEqual(ReplyStatus.Called, outcome.Status, text);
            StringAssert.Contains("get_weather", text);
        }
    }
}
