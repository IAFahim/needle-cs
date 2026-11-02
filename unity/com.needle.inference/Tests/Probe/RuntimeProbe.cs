// N10 runtime capability probe: reports the managed runtime, Vector<T>
// width, hardware-intrinsic availability (via reflection so this file
// compiles against the netstandard facade), Burst presence, and thread
// counts. The player scene (Samples~/Probe) carries a copy of this logic so
// the sample stays importable without the test assembly.
// C# 9 safe: block namespaces, no target-typed new, no u8 literals.
using System;
using System.Reflection;
using System.Text;
using UnityEngine;

namespace Needle.Probe
{
    /// <summary>A needle.probe/1 report. See <see cref="ToJson"/> for the exact JSON schema.</summary>
    [Serializable]
    public sealed class RuntimeProbeReport
    {
        public string schema = "needle.probe/1";
        public string frameworkDescription = string.Empty;
        public string processArchitecture = string.Empty;
        public bool vectorIsHardwareAccelerated;
        public int vectorByteCount;
        public bool avx2;
        public bool avxVnni;
        public bool advSimd;
        public bool dp;
        public bool burstPresent;
        public string burstVersion = string.Empty;
        public int burstEnabled = -1;
        public int processorCount;
        public int systemProcessorCount;
        public int systemProcessorFrequencyMHz;
        public string processorType = string.Empty;
        public string operatingSystem = string.Empty;
        public int jobWorkerCount = -1;
        public long stopwatchFrequency;
        public string unityVersion = string.Empty;
        public string platform = string.Empty;

        public string ToJson()
        {
            StringBuilder sb = new StringBuilder(512);
            sb.Append("{\"schema\":\"needle.probe/1\"");
            AppendString(sb, "frameworkDescription", frameworkDescription);
            AppendString(sb, "processArchitecture", processArchitecture);
            AppendBool(sb, "vectorIsHardwareAccelerated", vectorIsHardwareAccelerated);
            AppendInt(sb, "vectorByteCount", vectorByteCount);
            AppendBool(sb, "avx2", avx2);
            AppendBool(sb, "avxVnni", avxVnni);
            AppendBool(sb, "advSimd", advSimd);
            AppendBool(sb, "dp", dp);
            AppendBool(sb, "burstPresent", burstPresent);
            AppendString(sb, "burstVersion", burstVersion);
            AppendInt(sb, "burstEnabled", burstEnabled);
            AppendInt(sb, "processorCount", processorCount);
            AppendInt(sb, "systemProcessorCount", systemProcessorCount);
            AppendInt(sb, "systemProcessorFrequencyMHz", systemProcessorFrequencyMHz);
            AppendString(sb, "processorType", processorType);
            AppendString(sb, "operatingSystem", operatingSystem);
            AppendInt(sb, "jobWorkerCount", jobWorkerCount);
            AppendLong(sb, "stopwatchFrequency", stopwatchFrequency);
            AppendString(sb, "unityVersion", unityVersion);
            AppendString(sb, "platform", platform);
            sb.Append('}');
            return sb.ToString();
        }

        static void AppendString(StringBuilder sb, string name, string value)
        {
            sb.Append(",\"").Append(name).Append("\":\"").Append(Escape(value)).Append('"');
        }

        static void AppendBool(StringBuilder sb, string name, bool value)
        {
            sb.Append(",\"").Append(name).Append("\":").Append(value ? "true" : "false");
        }

        static void AppendInt(StringBuilder sb, string name, int value)
        {
            sb.Append(",\"").Append(name).Append("\":").Append(value);
        }

        static void AppendLong(StringBuilder sb, string name, long value)
        {
            sb.Append(",\"").Append(name).Append("\":").Append(value);
        }

        static string Escape(string value)
        {
            if (string.IsNullOrEmpty(value)) return string.Empty;
            StringBuilder sb = new StringBuilder(value.Length + 8);
            for (int i = 0; i < value.Length; i++)
            {
                char c = value[i];
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c < 0x20)
                        {
                            sb.Append("\\u");
                            sb.Append(((int)c).ToString("x4"));
                        }
                        else sb.Append(c);
                        break;
                }
            }
            return sb.ToString();
        }
    }

    public static class RuntimeProbe
    {
        public static RuntimeProbeReport Collect()
        {
            RuntimeProbeReport report = new RuntimeProbeReport();
            try { report.frameworkDescription = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription; }
            catch { report.frameworkDescription = string.Empty; }
            try { report.processArchitecture = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(); }
            catch { report.processArchitecture = string.Empty; }
            try { report.vectorIsHardwareAccelerated = System.Numerics.Vector.IsHardwareAccelerated; }
            catch { report.vectorIsHardwareAccelerated = false; }
            try { report.vectorByteCount = System.Numerics.Vector<byte>.Count; }
            catch { report.vectorByteCount = 0; }
            report.avx2 = IntrinsicIsSupported("System.Runtime.Intrinsics.X86.Avx2");
            report.avxVnni = IntrinsicIsSupported("System.Runtime.Intrinsics.X86.AvxVnni");
            report.advSimd = IntrinsicIsSupported("System.Runtime.Intrinsics.Arm.AdvSimd");
            report.dp = IntrinsicIsSupported("System.Runtime.Intrinsics.Arm.Dp");
            CollectBurst(report);
            try { report.processorCount = Environment.ProcessorCount; }
            catch { report.processorCount = 0; }
            try
            {
                report.systemProcessorCount = SystemInfo.processorCount;
                report.systemProcessorFrequencyMHz = SystemInfo.processorFrequency;
                report.processorType = SystemInfo.processorType ?? string.Empty;
                report.operatingSystem = SystemInfo.operatingSystem ?? string.Empty;
            }
            catch
            {
                report.processorType = string.Empty;
                report.operatingSystem = string.Empty;
            }
            report.jobWorkerCount = GetJobWorkerCount();
            try { report.stopwatchFrequency = System.Diagnostics.Stopwatch.Frequency; }
            catch { report.stopwatchFrequency = 0; }
            try { report.unityVersion = Application.unityVersion ?? string.Empty; }
            catch { report.unityVersion = string.Empty; }
            try { report.platform = Application.platform.ToString(); }
            catch { report.platform = string.Empty; }
            return report;
        }

        public static bool IntrinsicIsSupported(string fullName)
        {
            try
            {
                Type type = FindType(fullName);
                if (type == null) return false;
                PropertyInfo property = type.GetProperty("IsSupported", BindingFlags.Public | BindingFlags.Static);
                if (property == null || property.PropertyType != typeof(bool)) return false;
                return (bool)property.GetValue(null, null);
            }
            catch { return false; }
        }

        static void CollectBurst(RuntimeProbeReport report)
        {
            report.burstPresent = false;
            report.burstVersion = string.Empty;
            report.burstEnabled = -1;
            try
            {
                Type compiler = Type.GetType("Unity.Burst.BurstCompiler, Unity.Burst");
                if (compiler == null) compiler = FindType("Unity.Burst.BurstCompiler");
                if (compiler == null) return;
                report.burstPresent = true;
                try
                {
                    Version version = compiler.Assembly.GetName().Version;
                    report.burstVersion = version != null ? version.ToString() : string.Empty;
                }
                catch { report.burstVersion = string.Empty; }
                try
                {
                    PropertyInfo enabled = compiler.GetProperty("IsEnabled", BindingFlags.Public | BindingFlags.Static);
                    if (enabled != null && enabled.PropertyType == typeof(bool))
                        report.burstEnabled = ((bool)enabled.GetValue(null, null)) ? 1 : 0;
                }
                catch { report.burstEnabled = -1; }
            }
            catch { }
        }

        static int GetJobWorkerCount()
        {
            try
            {
                Type util = Type.GetType("Unity.Jobs.LowLevel.Unsafe.JobsUtility, Unity.Jobs");
                if (util == null) util = FindType("Unity.Jobs.LowLevel.Unsafe.JobsUtility");
                if (util == null) return -1;
                PropertyInfo property = util.GetProperty("JobWorkerCount", BindingFlags.Public | BindingFlags.Static);
                if (property == null) return -1;
                return (int)property.GetValue(null, null);
            }
            catch { return -1; }
        }

        static Type FindType(string fullName)
        {
            try
            {
                Type type = Type.GetType(fullName + ", System.Private.CoreLib");
                if (type != null) return type;
                type = Type.GetType(fullName + ", System.Runtime.Intrinsics");
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
                    try
                    {
                        Type type = assemblies[i].GetType(fullName);
                        if (type != null) return type;
                    }
                    catch { }
                }
            }
            catch { }
            return null;
        }
    }
}
