// N11d: IL2CPP player bench scene behaviour. Lives on a GameObject in
// BenchPlayerScene.unity; on Start it runs BenchDriver with the paths from
// the player command line, logs the needle.bench-e2e/1 JSON, writes it to
// --out (and always to persistentDataPath), then quits.
using System;
using System.Reflection;
using System.Text;
using System.IO;
using UnityEngine;

namespace Needle.Bench
{
    public sealed class BenchPlayer : MonoBehaviour
    {
        [Header("Defaults; every field is overridable from the command line.")]
        public string ModelPath = "";
        public string ToolsPath = "";
        public string GameToolsPath = "";
        public string OutPath = "";
        public string Threads = "4";
        public int Reps = 2;
        public int Warmups = 1;
        public string Sets = "golden";

        void Start()
        {
            int failures = 1;
            string json = "";
            try
            {
                // N41b: hard reference keeping Needle.Kernels.Burst in the
                // player link. An asmdef-only reference is pruned when no
                // Bench source names a Burst type, so IL2CPP drops the
                // assembly and its BeforeSceneLoad AutoInstall roots fire
                // nowhere (provider "none", silent). Naming the installers
                // here forces the link and logs the compiled lane — the
                // only truthful lane source — into the player log.
                Needle.BurstKernels.BurstKernelInstaller.EnsureInstalled();
                Needle.BurstKernels.BurstMlpInstaller.EnsureInstalled();
                Needle.BurstKernels.BurstAttention.Install();
                Needle.BurstKernels.BurstGemvNeon.Install();
                Debug.Log("[needle] " + Needle.BurstKernels.BurstKernelInstaller.Describe());
            }
            catch (Exception e)
            {
                Debug.LogWarning("[needle] burst explicit install failed: " + e.Message);
            }
            try
            {
                ReadCommandLine();
                string toolsJson = File.ReadAllText(ToolsPath);
                string gameToolsJson = string.IsNullOrEmpty(GameToolsPath) ? null : File.ReadAllText(GameToolsPath);
                failures = BenchDriver.Run(
                    delegate (string line) { Debug.Log("[bench-ns21] " + line); },
                    ModelPath,
                    toolsJson,
                    gameToolsJson,
                    ParseThreads(Threads),
                    Reps,
                    Warmups,
                    Sets,
                    out json);
                Debug.Log("[bench-ns21] failures=" + failures + "\n" + json);
                string outPath = string.IsNullOrEmpty(OutPath)
                    ? Path.Combine(Application.persistentDataPath, "bench-ns21.json")
                    : OutPath;
                try
                {
                    File.WriteAllText(outPath, json);
                    Debug.Log("[bench-ns21] wrote " + outPath);
                }
                catch (Exception e)
                {
                    Debug.LogError("[bench-ns21] could not write " + outPath + ": " + e.Message);
                }
                WriteProbe(outPath);
            }
            catch (Exception e)
            {
                Debug.LogError("[bench-ns21] " + e.GetType().Name + ": " + e.Message);
            }
            Debug.Log("[bench-ns21] BENCH_EXIT:" + failures);
            Application.Quit();
        }

        void ReadCommandLine()
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--model" && i + 1 < args.Length) ModelPath = args[++i];
                else if (args[i] == "--tools" && i + 1 < args.Length) ToolsPath = args[++i];
                else if (args[i] == "--game-tools" && i + 1 < args.Length) GameToolsPath = args[++i];
                else if (args[i] == "--out" && i + 1 < args.Length) OutPath = args[++i];
                else if (args[i] == "--threads" && i + 1 < args.Length) Threads = args[++i];
                else if (args[i] == "--reps" && i + 1 < args.Length) Reps = int.Parse(args[++i]);
                else if (args[i] == "--warmups" && i + 1 < args.Length) Warmups = int.Parse(args[++i]);
                else if (args[i] == "--sets" && i + 1 < args.Length) Sets = args[++i];
            }
            if (string.IsNullOrEmpty(ModelPath))
                ModelPath = Path.Combine(Application.streamingAssetsPath, "needle3.cact");
        }

        // N41: self-contained needle.probe/1 dump (mirrors Samples~/Probe and
        // Tests/Probe field for field) so the IL2CPP bench player settles H4
        // without a second player build. Reflection-only for intrinsics/Burst
        // lookups; never throws.
        void WriteProbe(string benchOutPath)
        {
            try
            {
                string json = ProbeJson();
                Debug.Log("[needle.probe/1] " + json);
                string dir = benchOutPath;
                try { dir = Path.GetDirectoryName(benchOutPath); } catch { }
                if (string.IsNullOrEmpty(dir)) dir = Application.persistentDataPath;
                try
                {
                    string probePath = Path.Combine(dir, "needle-probe.json");
                    File.WriteAllText(probePath, json);
                    Debug.Log("[bench-ns21] wrote " + probePath);
                }
                catch (Exception e)
                {
                    Debug.LogError("[bench-ns21] could not write probe: " + e.Message);
                }
                try { File.WriteAllText(Path.Combine(Application.persistentDataPath, "needle-probe.json"), json); } catch { }
            }
            catch (Exception e)
            {
                Debug.LogError("[bench-ns21] probe failed: " + e.Message);
            }
        }

        static string ProbeJson()
        {
            bool vectorHw = false;
            int vectorBytes = 0;
            try { vectorHw = System.Numerics.Vector.IsHardwareAccelerated; } catch { }
            try { vectorBytes = System.Numerics.Vector<byte>.Count; } catch { }
            string provider = "none";
            try
            {
                Type kernels = FindProbeType("Needle.Kernels");
                if (kernels != null)
                {
                    PropertyInfo prop = kernels.GetProperty("Provider", BindingFlags.Public | BindingFlags.Static);
                    if (prop != null)
                    {
                        object value = prop.GetValue(null, null);
                        if (value != null) provider = value.ToString();
                    }
                }
            }
            catch { }
            var sb = new StringBuilder(512);
            sb.Append("{\"schema\":\"needle.probe/1\"");
            AppendProbeString(sb, "frameworkDescription", FrameworkDescription());
            AppendProbeString(sb, "processArchitecture", Architecture());
            sb.Append(",\"vectorIsHardwareAccelerated\":").Append(vectorHw ? "true" : "false");
            sb.Append(",\"vectorByteCount\":").Append(vectorBytes);
            sb.Append(",\"avx2\":").Append(IntrinsicIsSupported("System.Runtime.Intrinsics.X86.Avx2") ? "true" : "false");
            sb.Append(",\"avxVnni\":").Append(IntrinsicIsSupported("System.Runtime.Intrinsics.X86.AvxVnni") ? "true" : "false");
            sb.Append(",\"advSimd\":").Append(IntrinsicIsSupported("System.Runtime.Intrinsics.Arm.AdvSimd") ? "true" : "false");
            sb.Append(",\"dp\":").Append(IntrinsicIsSupported("System.Runtime.Intrinsics.Arm.Dp") ? "true" : "false");
            string burstVersion;
            int burstEnabled;
            bool burstPresent = BurstInfo(out burstVersion, out burstEnabled);
            sb.Append(",\"burstPresent\":").Append(burstPresent ? "true" : "false");
            AppendProbeString(sb, "burstVersion", burstVersion);
            sb.Append(",\"burstEnabled\":").Append(burstEnabled);
            sb.Append(",\"processorCount\":").Append(Environment.ProcessorCount);
            AppendProbeString(sb, "simd", BenchDriver.SimdTier());
            AppendProbeString(sb, "provider", provider);
            sb.Append('}');
            return sb.ToString();
        }

        static string FrameworkDescription()
        {
            try { return System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription ?? ""; } catch { return ""; }
        }

        static string Architecture()
        {
            try { return System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(); } catch { return ""; }
        }

        static bool BurstInfo(out string version, out int enabled)
        {
            version = "";
            enabled = -1;
            try
            {
                Type compiler = FindProbeType("Unity.Burst.BurstCompiler");
                if (compiler == null) return false;
                try
                {
                    Version v = compiler.Assembly.GetName().Version;
                    if (v != null) version = v.ToString();
                }
                catch { }
                try
                {
                    PropertyInfo prop = compiler.GetProperty("IsEnabled", BindingFlags.Public | BindingFlags.Static);
                    if (prop != null && prop.PropertyType == typeof(bool))
                        enabled = ((bool)prop.GetValue(null, null)) ? 1 : 0;
                }
                catch { }
                return true;
            }
            catch { return false; }
        }

        static bool IntrinsicIsSupported(string fullName)
        {
            try
            {
                Type type = FindProbeType(fullName);
                if (type == null) return false;
                PropertyInfo property = type.GetProperty("IsSupported", BindingFlags.Public | BindingFlags.Static);
                if (property == null || property.PropertyType != typeof(bool)) return false;
                return (bool)property.GetValue(null, null);
            }
            catch { return false; }
        }

        static Type FindProbeType(string fullName)
        {
            try
            {
                Type type = Type.GetType(fullName + ", System.Private.CoreLib");
                if (type != null) return type;
                type = Type.GetType(fullName + ", System.Runtime.Intrinsics");
                if (type != null) return type;
                type = Type.GetType(fullName + ", Needle.Inference");
                if (type != null) return type;
                type = Type.GetType(fullName + ", Unity.Burst");
                if (type != null) return type;
                type = Type.GetType(fullName);
                if (type != null) return type;
            }
            catch { }
            try
            {
                Assembly[] assemblies = AppDomain.CurrentDomain.GetAssemblies();
                for (int i = 0; i < assemblies.Length; i++)
                {
                    Type type = null;
                    try { type = assemblies[i].GetType(fullName); } catch { }
                    if (type != null) return type;
                }
            }
            catch { }
            return null;
        }

        static void AppendProbeString(StringBuilder sb, string name, string value)
        {
            sb.Append(",\"").Append(name).Append("\":\"");
            if (value != null)
            {
                foreach (char c in value)
                {
                    if (c == '"' || c == '\\') sb.Append('\\');
                    sb.Append(c);
                }
            }
            sb.Append('"');
        }

        static int[] ParseThreads(string text)
        {
            string[] parts = text.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            var counts = new int[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                counts[i] = int.Parse(parts[i].Trim());
            return counts;
        }
    }
}
