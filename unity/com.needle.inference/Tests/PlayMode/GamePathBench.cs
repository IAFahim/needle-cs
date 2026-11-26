using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Needle.Unity.PlayMode.Tests
{
    // Measures what a game pays for inference, against the Unity targets of the 4.x plan
    // (docs/design/native-api-4x.md, section 10): main-thread cost per ask, managed allocation on
    // the main thread, frame-time spikes while the model runs, and engine latency per ask.
    //
    // Each frame runs a fixed piece of main-thread work (a stand-in for game code) and times it.
    // Frames with no request in flight are the baseline; frames with one in flight show how much the
    // inference threads slow the game thread down by competing for cores. The report goes to
    // NEEDLE_GAMEPATH_OUT (JSON) and the log. The test fails only on what must always hold: every
    // ask answered, no managed allocation by submit or poll, and no single submit or poll over
    // NEEDLE_FRAME_BUDGET_MS. The 20 us per ask and the frame-spike targets are reported, not
    // enforced, until they are met on the CI machine.
    public sealed class GamePathBench
    {
        const int IdleFrames = 240;
        const int WorkIterations = 20000;
        const double MainThreadTargetUs = 20.0;

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

        static float sink;
        NeedleHost host;

        [TearDown]
        public void TearDown()
        {
            host?.Dispose();
            host = null;
        }

        [UnityTest]
        public IEnumerator GamePathStaysOffTheMainThread()
        {
            string model = PlayModeAssets.Require(PlayModeAssets.FromEnvOrRepo("NEEDLE_MODEL", "ref/hf/needle3.cact"), "Needle 3 model (needle3.cact)");
            string tools = File.ReadAllText(PlayModeAssets.Require(PlayModeAssets.FromEnvOrRepo("NEEDLE_GAME_TOOLS", "tools/game_tools.json"), "Game tool set (tools/game_tools.json)"));
            int threads = PlayModeAssets.Threads;
            int rounds = Math.Max(1, PlayModeAssets.Reps);

            var encoded = new byte[Queries.Length][];
            for (int i = 0; i < Queries.Length; i++) encoded[i] = Encoding.UTF8.GetBytes(Queries[i]);

            host = new NeedleHost();
            NeedleChannel channel = host.Start(new HostOptions
            {
                ModelPath = model,
                ToolsJson = tools,
                StatelessQueries = true,
                Session = new NeedleOptions { Threading = ThreadPolicy.Game.WithThreads(threads) },
            });

            // Warm-up ask: the first one pays for JIT and page faults on the worker side.
            ChannelReply reply = default;
            Assert.IsTrue(channel.TrySubmitQuery(encoded[0], out ulong warmTicket), "the request ring refused the warm-up query");
            var deadline = Stopwatch.StartNew();
            while (!channel.TryReceive(out reply))
            {
                if (deadline.Elapsed > TimeSpan.FromMinutes(2)) Assert.Fail("no warm-up reply within two minutes");
                yield return null;
            }
            Assert.AreEqual(warmTicket, reply.Ticket);

            var idleWork = new List<double>(IdleFrames);
            for (int f = 0; f < IdleFrames; f++)
            {
                idleWork.Add(FrameWorkMs());
                yield return null;
            }

            var busyWork = new List<double>();
            var submitUs = new List<double>();
            var pollUs = new List<double>();
            var askMainUs = new List<double>();
            var engineMs = new List<double>();
            var askFrames = new List<double>();
            long allocatedBytes = 0;
            int failed = 0;
            var failures = new StringBuilder();
            double tickUs = 1e6 / Stopwatch.Frequency;

            for (int round = 0; round < rounds; round++)
            {
                for (int q = 0; q < encoded.Length; q++)
                {
                    long before = GC.GetAllocatedBytesForCurrentThread();
                    long t0 = Stopwatch.GetTimestamp();
                    bool submitted = channel.TrySubmitQuery(encoded[q], out ulong ticket);
                    long t1 = Stopwatch.GetTimestamp();
                    allocatedBytes += GC.GetAllocatedBytesForCurrentThread() - before;
                    Assert.IsTrue(submitted, "the request ring refused " + Queries[q]);
                    double askUs = (t1 - t0) * tickUs;
                    submitUs.Add(askUs);

                    int frames = 0;
                    var wall = Stopwatch.StartNew();
                    bool received = false;
                    while (!received)
                    {
                        if (wall.Elapsed > TimeSpan.FromMinutes(2)) Assert.Fail("no reply to '" + Queries[q] + "' within two minutes");
                        busyWork.Add(FrameWorkMs());
                        yield return null;
                        frames++;
                        before = GC.GetAllocatedBytesForCurrentThread();
                        t0 = Stopwatch.GetTimestamp();
                        received = channel.TryReceive(out reply);
                        t1 = Stopwatch.GetTimestamp();
                        allocatedBytes += GC.GetAllocatedBytesForCurrentThread() - before;
                        double poll = (t1 - t0) * tickUs;
                        pollUs.Add(poll);
                        askUs += poll;
                    }

                    askMainUs.Add(askUs);
                    askFrames.Add(frames);
                    engineMs.Add(reply.Reply.ElapsedTicks * 1000.0 / Stopwatch.Frequency);
                    if (reply.Ticket != ticket || reply.Reply.Status == ReplyStatus.Failed || reply.Reply.Status == ReplyStatus.Cancelled)
                    {
                        failed++;
                        failures.Append(Queries[q]).Append(": ").Append(reply.Reply.Status).Append(' ').Append(reply.Reply.Failure).Append("; ");
                    }
                }
            }

            string json = Report(threads, rounds, idleWork, busyWork, submitUs, pollUs, askMainUs, engineMs, askFrames, allocatedBytes, failed);
            UnityEngine.Debug.Log("[needle-gamepath] " + json);
            string outPath = Environment.GetEnvironmentVariable("NEEDLE_GAMEPATH_OUT");
            if (!string.IsNullOrEmpty(outPath))
            {
                string dir = Path.GetDirectoryName(outPath);
                if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                File.WriteAllText(outPath, json);
            }

            Assert.AreEqual(0, failed, "asks that failed: " + failures);
            Assert.AreEqual(0L, allocatedBytes, "submitting and polling allocated managed memory on the main thread");
            Assert.Less(Max(submitUs) / 1000.0, PlayModeAssets.FrameBudgetMs, "one submit cost the main thread more than the frame budget");
            Assert.Less(Max(pollUs) / 1000.0, PlayModeAssets.FrameBudgetMs, "one poll cost the main thread more than the frame budget");
        }

        // A fixed piece of main-thread work, timed. It does not allocate.
        static double FrameWorkMs()
        {
            long start = Stopwatch.GetTimestamp();
            float x = sink + 1.0f;
            for (int i = 0; i < WorkIterations; i++) x = x * 1.0000001f + 0.5f / (1.0f + (i & 7));
            sink = x;
            return (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
        }

        static string Report(int threads, int rounds, List<double> idle, List<double> busy, List<double> submitUs, List<double> pollUs,
            List<double> askMainUs, List<double> engineMs, List<double> askFrames, long allocatedBytes, int failed)
        {
            double idleP50 = Percentile(idle, 50), idleP99 = Percentile(idle, 99);
            double busyP50 = Percentile(busy, 50), busyP99 = Percentile(busy, 99);
            double askP50 = Percentile(askMainUs, 50), askP99 = Percentile(askMainUs, 99);
            var b = new StringBuilder(1024);
            b.Append("{\"schema\":\"needle.unity-gamepath/1\"");
            b.Append(",\"threads\":").Append(threads.ToString(CultureInfo.InvariantCulture));
            b.Append(",\"asks\":").Append(askMainUs.Count.ToString(CultureInfo.InvariantCulture));
            b.Append(",\"rounds\":").Append(rounds.ToString(CultureInfo.InvariantCulture));
            b.Append(",\"failed\":").Append(failed.ToString(CultureInfo.InvariantCulture));
            b.Append(",\"runtime\":\"").Append(Escape(System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription)).Append('"');
            b.Append(",\"kernels\":\"").Append(Escape(Needle.NeedleRuntime.Describe())).Append('"');
            b.Append(",\"main_thread_alloc_bytes\":").Append(allocatedBytes.ToString(CultureInfo.InvariantCulture));
            Stats(b, "submit_us", submitUs);
            Stats(b, "poll_us", pollUs);
            Stats(b, "main_thread_us_per_ask", askMainUs);
            Stats(b, "frames_per_ask", askFrames);
            Stats(b, "engine_ms", engineMs);
            Stats(b, "frame_work_idle_ms", idle);
            Stats(b, "frame_work_busy_ms", busy);
            b.Append(",\"targets\":{");
            b.Append("\"main_thread_us_per_ask_p99\":{\"target\":").Append(Num(MainThreadTargetUs)).Append(",\"measured\":").Append(Num(askP99))
                .Append(",\"met\":").Append(askP99 <= MainThreadTargetUs ? "true" : "false").Append('}');
            b.Append(",\"zero_main_thread_alloc\":{\"met\":").Append(allocatedBytes == 0 ? "true" : "false").Append('}');
            double spike = idleP99 > 0 ? busyP99 / idleP99 : 0;
            b.Append(",\"frame_work_p99_busy_over_idle\":{\"target\":1.1,\"measured\":").Append(Num(spike))
                .Append(",\"met\":").Append(spike > 0 && spike <= 1.1 ? "true" : "false").Append('}');
            b.Append('}');
            b.Append(",\"summary\":\"main thread ").Append(Num(askP50)).Append("/").Append(Num(askP99)).Append(" us per ask (p50/p99), frame work ")
                .Append(Num(idleP50)).Append("/").Append(Num(idleP99)).Append(" ms idle vs ").Append(Num(busyP50)).Append("/").Append(Num(busyP99))
                .Append(" ms busy, engine ").Append(Num(Percentile(engineMs, 50))).Append(" ms per ask\"");
            b.Append('}');
            return b.ToString();
        }

        static void Stats(StringBuilder b, string name, List<double> values)
        {
            b.Append(",\"").Append(name).Append("\":{\"p50\":").Append(Num(Percentile(values, 50)))
                .Append(",\"p99\":").Append(Num(Percentile(values, 99)))
                .Append(",\"max\":").Append(Num(Max(values)))
                .Append(",\"n\":").Append(values.Count.ToString(CultureInfo.InvariantCulture)).Append('}');
        }

        static double Percentile(List<double> values, double p)
        {
            if (values.Count == 0) return 0;
            var sorted = new List<double>(values);
            sorted.Sort();
            int index = (int)Math.Ceiling(p / 100.0 * sorted.Count) - 1;
            return sorted[Math.Max(0, Math.Min(sorted.Count - 1, index))];
        }

        static double Max(List<double> values)
        {
            double max = 0;
            for (int i = 0; i < values.Count; i++) max = Math.Max(max, values[i]);
            return max;
        }

        static string Num(double value) => value.ToString("0.###", CultureInfo.InvariantCulture);

        static string Escape(string text) => (text ?? string.Empty).Replace("\\", "\\\\").Replace("\"", "\\\"");
    }
}
