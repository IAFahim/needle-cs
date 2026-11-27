using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace Needle.Unity.Tests
{
    // Finds the model files the end-to-end tests need. Every path can be set with an environment
    // variable; otherwise the tests look for a needle-cs checkout above the Unity project (the
    // case for unity/NeedleDemo) and use its ref/ folder, which tools/fetch.sh fills.
    //
    // A missing file makes the test Ignored, so projects that list the package in "testables"
    // without the weights still pass. CI sets NEEDLE_REQUIRE_MODEL=1, which turns a missing file
    // into a failure.
    public static class NeedleTestAssets
    {
        public static string RepoRoot
        {
            get
            {
                string fromEnv = Environment.GetEnvironmentVariable("NEEDLE_REPO");
                if (!string.IsNullOrEmpty(fromEnv)) return fromEnv;
                string found = FindRepoRoot(Directory.GetCurrentDirectory());
                if (found == null) found = FindRepoRoot(Path.GetDirectoryName(Application.dataPath));
                return found ?? string.Empty;
            }
        }

        public static string ModelPath => FromEnvOrRepo("NEEDLE_MODEL", "ref/hf/needle3.cact");
        public static string WhistlePath => FromEnvOrRepo("NEEDLE_WHISTLE", "ref/hf/whistle/whistle.cact");
        public static string ToolsPath => FromEnvOrRepo("NEEDLE_TOOLS", "tools/tools.json");
        public static string AudioPath(string clip)
        {
            string directory = Environment.GetEnvironmentVariable("NEEDLE_AUDIO_DIR");
            return !string.IsNullOrEmpty(directory) ? Path.Combine(directory, clip) : FromEnvOrRepo(null, Path.Combine("ref", "audio", clip));
        }

        public static bool ModelsRequired => Environment.GetEnvironmentVariable("NEEDLE_REQUIRE_MODEL") == "1";

        public static string Require(string path, string what)
        {
            if (!string.IsNullOrEmpty(path) && File.Exists(path)) return path;
            string message = what + " not found at '" + path + "'. Run tools/fetch.sh in the needle-cs checkout, or set the environment variable for it (see Tests/Runtime/NeedleTestAssets.cs).";
            if (ModelsRequired) Assert.Fail(message);
            Assert.Ignore(message);
            return path;
        }

        public static string RequireModel() => Require(ModelPath, "Needle 3 model (needle3.cact)");
        public static string RequireWhistle() => Require(WhistlePath, "Whistle model (whistle.cact)");
        public static string RequireTools() => Require(ToolsPath, "Tool set (tools/tools.json)");
        public static string RequireAudio(string clip) => Require(AudioPath(clip), "Audio clip " + clip);

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

        static string FromEnvOrRepo(string variable, string relative)
        {
            string fromEnv = variable == null ? null : Environment.GetEnvironmentVariable(variable);
            if (!string.IsNullOrEmpty(fromEnv)) return fromEnv;
            string root = RepoRoot;
            return string.IsNullOrEmpty(root) ? relative : Path.Combine(root, relative);
        }

        static string FindRepoRoot(string start)
        {
            for (var dir = string.IsNullOrEmpty(start) ? null : new DirectoryInfo(start); dir != null; dir = dir.Parent)
            {
                if (File.Exists(Path.Combine(dir.FullName, "tools", "ref.lock")) &&
                    File.Exists(Path.Combine(dir.FullName, "Directory.Build.props")))
                    return dir.FullName;
            }
            return null;
        }
    }
}
