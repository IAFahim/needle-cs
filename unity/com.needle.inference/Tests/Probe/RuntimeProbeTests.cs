using NUnit.Framework;
using UnityEngine;

namespace Needle.Probe.Tests
{
    public sealed class RuntimeProbeTests
    {
        [Test]
        public void CollectReturnsPopulatedReport()
        {
            RuntimeProbeReport report = RuntimeProbe.Collect();
            Assert.IsNotNull(report);
            Assert.AreEqual("needle.probe/1", report.schema);
            Assert.IsFalse(string.IsNullOrEmpty(report.frameworkDescription), "FrameworkDescription must be reported");
            Assert.Greater(report.vectorByteCount, 0, "Vector<byte>.Count must be positive");
            Assert.Greater(report.processorCount, 0, "ProcessorCount must be positive");
            Assert.Greater(report.stopwatchFrequency, 0L, "Stopwatch.Frequency must be positive");
            string json = report.ToJson();
            string jsonAgain = report.ToJson();
            Assert.IsFalse(string.IsNullOrEmpty(json));
            Assert.AreEqual(json, jsonAgain, "ToJson must be deterministic");
            Assert.IsTrue(json.Contains("\"schema\":\"needle.probe/1\""), "JSON must carry the schema marker");
            Debug.Log("[needle.probe/1] " + json);
        }

        [Test]
        public void IntrinsicProbeNeverThrows()
        {
            Assert.IsFalse(RuntimeProbe.IntrinsicIsSupported("No.Such.Type, No.Such.Assembly"));
            bool avx2 = RuntimeProbe.IntrinsicIsSupported("System.Runtime.Intrinsics.X86.Avx2");
            bool avxVnni = RuntimeProbe.IntrinsicIsSupported("System.Runtime.Intrinsics.X86.AvxVnni");
            bool advSimd = RuntimeProbe.IntrinsicIsSupported("System.Runtime.Intrinsics.Arm.AdvSimd");
            bool dp = RuntimeProbe.IntrinsicIsSupported("System.Runtime.Intrinsics.Arm.Dp");
            Debug.Log("[needle.probe/1] intrinsics avx2=" + avx2 + " avxVnni=" + avxVnni + " advSimd=" + advSimd + " dp=" + dp);
        }
    }
}
