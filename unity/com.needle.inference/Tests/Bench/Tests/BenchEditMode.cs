// N11c: EditMode bench (CoreCLR Editor + ns2.1 DLL). Reads the model and tools
// paths from the environment (set by tools/bench-ns21.sh editmode) so the
// test stays hermetic: NEEDLE_MODEL, NEEDLE_TOOLS, NEEDLE_GAME_TOOLS
// (optional), NEEDLE_THREADS (default "4"), NEEDLE_REPS (default 2),
// NEEDLE_WARMUPS (default 1), NEEDLE_SETS (default "golden").
using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace Needle.Bench.Tests
{
    public sealed class BenchEditMode
    {
        [Test]
        public void BenchGoldenReproducesCallsAndEmitsBenchE2EJson()
        {
            string model = Environment.GetEnvironmentVariable("NEEDLE_MODEL") ?? "";
            string tools = Environment.GetEnvironmentVariable("NEEDLE_TOOLS") ?? "";
            string gameTools = Environment.GetEnvironmentVariable("NEEDLE_GAME_TOOLS") ?? "";
            if (string.IsNullOrEmpty(model) || !File.Exists(model))
                Assert.Fail("NEEDLE_MODEL is missing or unreadable: '" + model + "'. Run tools/bench-ns21.sh editmode.");
            if (string.IsNullOrEmpty(tools) || !File.Exists(tools))
                Assert.Fail("NEEDLE_TOOLS is missing or unreadable: '" + tools + "'. Run tools/bench-ns21.sh editmode.");

            int[] threads = ParseThreads(Environment.GetEnvironmentVariable("NEEDLE_THREADS") ?? "4");
            int reps = ParseInt(Environment.GetEnvironmentVariable("NEEDLE_REPS"), 2);
            int warmups = ParseInt(Environment.GetEnvironmentVariable("NEEDLE_WARMUPS"), 1);
            string sets = Environment.GetEnvironmentVariable("NEEDLE_SETS") ?? "golden";
            string gameToolsJson = !string.IsNullOrEmpty(gameTools) && File.Exists(gameTools)
                ? File.ReadAllText(gameTools)
                : null;

            string json;
            int failures = BenchDriver.Run(
                delegate (string line) { Debug.Log("[bench-ns21] " + line); },
                model,
                File.ReadAllText(tools),
                gameToolsJson,
                threads,
                reps,
                warmups,
                sets,
                out json);

            TestContext.Progress.WriteLine(json);
            Assert.That(json, Does.Contain("needle.bench-e2e/1"), "runner must print needle.bench-e2e/1 JSON");
            Assert.That(failures, Is.EqualTo(0), "golden-set call outputs did not reproduce");
        }

        static int[] ParseThreads(string text)
        {
            string[] parts = text.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            var counts = new int[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                counts[i] = int.Parse(parts[i].Trim());
            return counts;
        }

        static int ParseInt(string text, int fallback)
        {
            int value;
            return !string.IsNullOrEmpty(text) && int.TryParse(text.Trim(), out value) ? value : fallback;
        }
    }
}
