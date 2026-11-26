using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using Needle.BurstKernels;
using NUnit.Framework;
using UnityEngine;

namespace Needle.Unity.Tests
{
    // End-to-end checks with the real weights, inside the Unity Editor's runtime. They are Ignored
    // when the files are missing, unless NEEDLE_REQUIRE_MODEL=1 (see NeedleTestAssets).
    public sealed class ModelTests
    {
        [OneTimeSetUp]
        public void InstallKernels()
        {
            BurstKernelInstaller.EnsureInstalled();
            UnityEngine.Debug.Log("[needle] " + NeedleRuntime.Describe());
        }

        [Test]
        public void SessionCallsTheWeatherTool()
        {
            string tools = File.ReadAllText(NeedleTestAssets.RequireTools());
            using (var model = Model.Open(NeedleTestAssets.RequireModel()))
            using (var session = new NeedleSession(model, tools, new NeedleOptions { Threads = NeedleTestAssets.Threads }))
            {
                var clock = Stopwatch.StartNew();
                var result = session.Complete("what's it like in Lagos right now?");
                UnityEngine.Debug.Log("[needle] session " + clock.Elapsed.TotalMilliseconds.ToString("F1") + " ms: " + result.ToJson());
                Assert.AreEqual(1, result.FunctionCalls.Count, result.ToJson());
                Assert.AreEqual("get_weather", result.FunctionCalls[0].Name);
                StringAssert.Contains("Lagos", result.FunctionCalls[0].ArgumentsJson);
            }
        }

        [Test]
        public void SessionCallsTwoToolsInOneTurn()
        {
            string tools = File.ReadAllText(NeedleTestAssets.RequireTools());
            using (var model = Model.Open(NeedleTestAssets.RequireModel()))
            using (var session = new NeedleSession(model, tools, new NeedleOptions { Threads = NeedleTestAssets.Threads }))
            {
                var result = session.Complete("set a timer for 12 minutes and play some Miles Davis");
                var names = new StringBuilder();
                foreach (var call in result.FunctionCalls) names.Append(call.Name).Append(' ');
                StringAssert.Contains("set_timer", names.ToString(), result.ToJson());
                StringAssert.Contains("play_music", names.ToString(), result.ToJson());
            }
        }

        [Test]
        public void SpanApiDoesNotAllocateOnTheCallingThread()
        {
            string tools = File.ReadAllText(NeedleTestAssets.RequireTools());
            using (var model = Model.Open(NeedleTestAssets.RequireModel()))
            using (var session = new NeedleSession(model, tools, new NeedleOptions { Threads = NeedleTestAssets.Threads }))
            {
                byte[] query = Encoding.UTF8.GetBytes("what's it like in Lagos right now?");
                byte[] calls = new byte[ChannelLimits.CallsCapacity];
                session.Complete(query, calls);
                session.Reset();
                long before = GC.GetAllocatedBytesForCurrentThread();
                Reply reply = session.Complete(query, calls);
                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                UnityEngine.Debug.Log("[needle] span call " + reply.Status + ", " + allocated + " bytes allocated on the calling thread");
                Assert.AreEqual(ReplyStatus.Called, reply.Status);
                Assert.AreEqual(0, allocated, "the span API allocated on the calling thread");
            }
        }

        [Test]
        public void HostRepliesThroughTheChannel()
        {
            string tools = File.ReadAllText(NeedleTestAssets.RequireTools());
            using (var host = new NeedleHost())
            {
                NeedleChannel channel = host.Start(new HostOptions
                {
                    ModelPath = NeedleTestAssets.RequireModel(),
                    ToolsJson = tools,
                    StatelessQueries = true,
                    Session = new NeedleOptions { Threading = ThreadPolicy.Game.WithThreads(NeedleTestAssets.Threads) },
                });
                Assert.IsTrue(channel.TrySubmitQuery(Encoding.UTF8.GetBytes("set a timer for 12 minutes"), out ulong ticket));
                var clock = Stopwatch.StartNew();
                ChannelReply reply;
                while (!channel.TryReceive(out reply))
                {
                    if (clock.Elapsed > TimeSpan.FromMinutes(2)) Assert.Fail("no reply within two minutes");
                    System.Threading.Thread.Sleep(1);
                }
                string calls = Encoding.UTF8.GetString(Utf8.AsSpan(ref reply.Calls, reply.Reply.Utf8Length));
                UnityEngine.Debug.Log("[needle] host reply " + reply.Reply.Status + " in " + clock.ElapsedMilliseconds + " ms: " + calls);
                Assert.AreEqual(ticket, reply.Ticket);
                Assert.AreEqual(ReplyStatus.Called, reply.Reply.Status, calls);
                StringAssert.Contains("set_timer", calls);
            }
        }

        [Test]
        public void WhistleTranscribesTheRecordClip()
        {
            string whistlePath = NeedleTestAssets.RequireWhistle();
            float[] pcm = Wav.Read(NeedleTestAssets.RequireAudio("record.wav"));
            using (var speech = Model.Open(whistlePath))
            using (var whistle = new Whistle(speech, NeedleTestAssets.Threads))
            {
                var clock = Stopwatch.StartNew();
                var result = whistle.Transcribe(pcm);
                UnityEngine.Debug.Log("[needle] whistle " + clock.Elapsed.TotalMilliseconds.ToString("F1") + " ms: " + result.Text);
                Assert.AreEqual("en", result.Language);
                StringAssert.Contains("speed of sound at sea level", result.Text);
            }
        }
    }
}
