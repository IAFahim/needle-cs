// N11d: headless IL2CPP player build entry point for the bench scene.
// Invoked by tools/bench-ns21.sh player via:
//   unity build --execute-method Needle.Bench.Tests.BenchBuild.PerformLinuxIl2cpp
// Env: NEEDLE_PLAYER_OUT (default out/bench-ns21/BenchNs21).
#if UNITY_EDITOR
using System;
using UnityEditor;
using UnityEditor.Build.Reporting;

namespace Needle.Bench.Tests
{
    public static class BenchBuild
    {
        public static void PerformLinuxIl2cpp()
        {
            string scene = "Packages/com.needle.inference/Tests/Bench/BenchPlayerScene.unity";
            string output = Environment.GetEnvironmentVariable("NEEDLE_PLAYER_OUT");
            if (string.IsNullOrEmpty(output))
                output = "out/bench-ns21/BenchNs21";
            EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, BuildTarget.StandaloneLinux64);
            PlayerSettings.SetScriptingBackend(BuildTargetGroup.Standalone, ScriptingImplementation.IL2CPP);
            var options = new BuildPlayerOptions();
            options.scenes = new string[] { scene };
            options.locationPathName = output;
            options.target = BuildTarget.StandaloneLinux64;
            options.options = BuildOptions.None;
            BuildReport report = BuildPipeline.BuildPlayer(options);
            if (report.summary.result != BuildResult.Succeeded)
                throw new Exception("bench player build failed: " + report.summary.result);
        }

        // N01: CoreCLR twin of PerformLinuxIl2cpp (same scene). The player
        // runs the ns2.1 reference plus the net10 sidecar (Editor/Standalone
        // compatible, never a compile reference); the bench JSON's simd and
        // runtime fields report which tier actually ran.
        public static void PerformLinuxCoreClr()
        {
            string scene = "Packages/com.needle.inference/Tests/Bench/BenchPlayerScene.unity";
            string output = Environment.GetEnvironmentVariable("NEEDLE_PLAYER_OUT");
            if (string.IsNullOrEmpty(output))
                output = "out/bench-ns21/BenchNs21CoreClr";
            EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, BuildTarget.StandaloneLinux64);
            PlayerSettings.SetScriptingBackend(BuildTargetGroup.Standalone, ScriptingImplementation.CoreCLR);
            var options = new BuildPlayerOptions();
            options.scenes = new string[] { scene };
            options.locationPathName = output;
            options.target = BuildTarget.StandaloneLinux64;
            options.options = BuildOptions.None;
            BuildReport report = BuildPipeline.BuildPlayer(options);
            if (report.summary.result != BuildResult.Succeeded)
                throw new Exception("bench player build failed: " + report.summary.result);
        }
    }
}
#endif
