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
    // The game-loop case: frames keep ticking while the model answers on its own threads, and the
    // main thread only pays for submitting the query and polling the reply ring. Ignored without
    // the model, unless NEEDLE_REQUIRE_MODEL=1.
    public sealed class HostPlayModeTests
    {
        NeedleHost host;

        [TearDown]
        public void TearDown()
        {
            host?.Dispose();
            host = null;
        }

        [UnityTest]
        public IEnumerator HostRepliesWhileFramesTick()
        {
            string model = PlayModeAssets.Require(PlayModeAssets.FromEnvOrRepo("NEEDLE_MODEL", "ref/hf/needle3.cact"), "Needle 3 model (needle3.cact)");
            string tools = File.ReadAllText(PlayModeAssets.Require(PlayModeAssets.FromEnvOrRepo("NEEDLE_TOOLS", "tools/tools.json"), "Tool set (tools/tools.json)"));

            host = new NeedleHost();
            NeedleChannel channel = host.Start(new HostOptions
            {
                ModelPath = model,
                ToolsJson = tools,
                StatelessQueries = true,
                Session = new NeedleOptions { Threading = ThreadPolicy.Game.WithThreads(PlayModeAssets.Threads) },
            });

            byte[] query = Encoding.UTF8.GetBytes("set a timer for 12 minutes");
            var mainThread = new Stopwatch();
            mainThread.Start();
            bool submitted = channel.TrySubmitQuery(query, out ulong ticket);
            mainThread.Stop();
            Assert.IsTrue(submitted, "the request ring refused the query");
            double worstFrameMs = mainThread.Elapsed.TotalMilliseconds;

            var wall = Stopwatch.StartNew();
            int frames = 0;
            ChannelReply reply = default;
            bool received = false;
            while (!received)
            {
                if (wall.Elapsed > TimeSpan.FromMinutes(2)) Assert.Fail("no reply within two minutes (" + frames + " frames)");
                yield return null;
                frames++;
                mainThread.Restart();
                received = channel.TryReceive(out reply);
                mainThread.Stop();
                worstFrameMs = Math.Max(worstFrameMs, mainThread.Elapsed.TotalMilliseconds);
            }

            string calls = Encoding.UTF8.GetString(Utf8.AsSpan(ref reply.Calls, reply.Reply.Utf8Length));
            UnityEngine.Debug.Log("[needle] play mode reply " + reply.Reply.Status + " after " + wall.ElapsedMilliseconds + " ms and " + frames +
                " frames, worst main-thread cost " + worstFrameMs.ToString("F3") + " ms: " + calls);
            Assert.AreEqual(ticket, reply.Ticket);
            Assert.AreEqual(ReplyStatus.Called, reply.Reply.Status, calls);
            StringAssert.Contains("set_timer", calls);
            Assert.Greater(frames, 1, "the reply arrived without the frame loop running; the host blocked the main thread");
            Assert.Less(worstFrameMs, PlayModeAssets.FrameBudgetMs,
                "submitting or polling cost the main thread more than " + PlayModeAssets.FrameBudgetMs + " ms in one frame");
        }
    }

    // A copy of the EditMode NeedleTestAssets lookup: that assembly is Editor-only, and this one
    // also builds into players.
    static class PlayModeAssets
    {
        public static int Threads
        {
            get
            {
                int wanted;
                string text = Environment.GetEnvironmentVariable("NEEDLE_THREADS");
                if (string.IsNullOrEmpty(text) || !int.TryParse(text.Split(',')[0].Trim(), out wanted)) wanted = 4;
                return Math.Max(1, Math.Min(wanted, Environment.ProcessorCount));
            }
        }

        public static int Reps
        {
            get
            {
                int reps;
                string text = Environment.GetEnvironmentVariable("NEEDLE_REPS");
                return !string.IsNullOrEmpty(text) && int.TryParse(text.Trim(), out reps) && reps > 0 ? reps : 2;
            }
        }

        public static double FrameBudgetMs
        {
            get
            {
                double budget;
                string text = Environment.GetEnvironmentVariable("NEEDLE_FRAME_BUDGET_MS");
                return !string.IsNullOrEmpty(text) && double.TryParse(text, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out budget) ? budget : 1.0;
            }
        }

        public static string Require(string path, string what)
        {
            if (!string.IsNullOrEmpty(path) && File.Exists(path)) return path;
            string message = what + " not found at '" + path + "'. Run tools/fetch.sh in the needle-cs checkout, or set the environment variable for it.";
            if (Environment.GetEnvironmentVariable("NEEDLE_REQUIRE_MODEL") == "1") Assert.Fail(message);
            Assert.Ignore(message);
            return path;
        }

        public static string FromEnvOrRepo(string variable, string relative)
        {
            string fromEnv = Environment.GetEnvironmentVariable(variable);
            if (!string.IsNullOrEmpty(fromEnv)) return fromEnv;
            string root = Environment.GetEnvironmentVariable("NEEDLE_REPO");
            if (string.IsNullOrEmpty(root)) root = FindRepoRoot(Directory.GetCurrentDirectory()) ?? FindRepoRoot(Path.GetDirectoryName(Application.dataPath));
            return string.IsNullOrEmpty(root) ? relative : Path.Combine(root, relative);
        }

        static string FindRepoRoot(string start)
        {
            for (var dir = string.IsNullOrEmpty(start) ? null : new DirectoryInfo(start); dir != null; dir = dir.Parent)
                if (File.Exists(Path.Combine(dir.FullName, "tools", "ref.lock")) && File.Exists(Path.Combine(dir.FullName, "Directory.Build.props")))
                    return dir.FullName;
            return null;
        }
    }
}
