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
            string model = FromEnvOrRepo("NEEDLE_MODEL", "ref/hf/needle3.cact");
            string tools = FromEnvOrRepo("NEEDLE_TOOLS", "tools/tools.json");
            string gameTools = FromEnvOrRepo("NEEDLE_GAME_TOOLS", "tools/game_tools.json");
            Require(model, "NEEDLE_MODEL (needle3.cact)");
            Require(tools, "NEEDLE_TOOLS (tools/tools.json)");

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

            // TestContext.Progress reaches neither the -logFile nor the NUnit XML of a batch run.
            Debug.Log("[bench-ns21] " + json);
            TestContext.Progress.WriteLine(json);
            string output = Environment.GetEnvironmentVariable("NEEDLE_BENCH_OUT");
            if (!string.IsNullOrEmpty(output))
            {
                string directory = Path.GetDirectoryName(Path.GetFullPath(output));
                if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
                File.WriteAllText(output, json);
            }
            Assert.That(json, Does.Contain("needle.bench-e2e/1"), "runner must print needle.bench-e2e/1 JSON");
            Assert.That(failures, Is.EqualTo(0), "golden-set call outputs did not reproduce");
        }

        // The bench needs the weights. Without them it is Ignored, so projects that list the package
        // in "testables" still pass; CI sets NEEDLE_REQUIRE_MODEL=1 to make a missing file a failure.
        static void Require(string path, string what)
        {
            if (!string.IsNullOrEmpty(path) && File.Exists(path)) return;
            string message = what + " is missing or unreadable: '" + path + "'. Run tools/fetch.sh, or tools/bench-ns21.sh editmode.";
            if (Environment.GetEnvironmentVariable("NEEDLE_REQUIRE_MODEL") == "1") Assert.Fail(message);
            Assert.Ignore(message);
        }

        static string FromEnvOrRepo(string variable, string relative)
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
