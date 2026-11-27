using System;
using System.Reflection;
using Needle.BurstKernels;
using NUnit.Framework;
using UnityEngine;

namespace Needle.Unity.Tests
{
    // Checks the two plugin DLLs the package ships and which kernel provider the Editor ends up
    // with. None of these need a model file.
    public sealed class PackageTests
    {
        const string SidecarName = "Needle.Inference.Net10";

        static string StampOf(Assembly assembly)
        {
            foreach (var attribute in assembly.GetCustomAttributes<AssemblyMetadataAttribute>())
                if (attribute.Key == "Needle.SourceStamp") return attribute.Value;
            return null;
        }

        static Assembly LoadedSidecar()
        {
            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
                if (string.Equals(assembly.GetName().Name, SidecarName, StringComparison.OrdinalIgnoreCase))
                    return assembly;
            return null;
        }

        static bool IsCoreClrDesktopEditor =>
            Environment.Version.Major >= 8 &&
            (Application.platform == RuntimePlatform.LinuxEditor ||
             Application.platform == RuntimePlatform.WindowsEditor ||
             Application.platform == RuntimePlatform.OSXEditor);

        [Test]
        public void CoreDllIsStampedWithItsSources()
        {
            string stamp = StampOf(typeof(Model).Assembly);
            Assert.IsNotNull(stamp, "Needle.Inference.dll carries no Needle.SourceStamp; rebuild it with tools/pack-unity.sh.");
            StringAssert.StartsWith("src-sha256:", stamp);
            Debug.Log("[needle] core " + typeof(Model).Assembly.GetName().Name + " " + stamp);
        }

        [Test]
        public void SidecarAndCoreComeFromTheSameSources()
        {
            BurstKernelInstaller.EnsureInstalled();
            Assembly sidecar = LoadedSidecar();
            if (sidecar == null)
            {
                if (IsCoreClrDesktopEditor && Environment.GetEnvironmentVariable("NEEDLE_FORCE_BURST") == null)
                    Assert.Fail(SidecarName + ".dll is not loaded in a CoreCLR Editor; the package's Runtime/Plugins/net10.0 folder did not reach the Editor.");
                Assert.Ignore(SidecarName + " is not loaded on this runtime (" + Environment.Version + ", " + Application.platform + ").");
            }
            Assert.AreEqual(StampOf(typeof(Model).Assembly), StampOf(sidecar),
                "The netstandard2.1 core and the net10.0 sidecar were built from different sources; run tools/pack-unity.sh and commit both DLLs.");
        }

        [Test]
        public void CoreClrEditorRunsTheNet10Kernels()
        {
            if (!IsCoreClrDesktopEditor)
                Assert.Ignore("Only a CoreCLR desktop Editor loads the net10.0 sidecar (this one: .NET " + Environment.Version + ", " + Application.platform + ").");
            if (Environment.GetEnvironmentVariable("NEEDLE_FORCE_BURST") != null)
                Assert.Ignore("NEEDLE_FORCE_BURST is set, so the Burst provider is expected instead.");
            BurstKernelInstaller.EnsureInstalled();
            string describe = NeedleRuntime.Describe();
            Debug.Log("[needle] " + describe);
            StringAssert.StartsWith("net10-unmanaged", Kernels.Provider,
                "Expected the net10.0 sidecar to provide the kernels in a CoreCLR Editor. " + describe);
        }

        [Test]
        public void KernelTableIsInstalledOnEveryRuntime()
        {
            BurstKernelInstaller.EnsureInstalled();
            Assert.IsTrue(Kernels.IsInstalled, "No kernel provider installed. " + NeedleRuntime.Describe());
            Assert.IsTrue(Kernels.Active.MatMulRows != 0, "KernelTable.MatMulRows is empty. " + NeedleRuntime.Describe());
            Assert.IsTrue(Kernels.Active.Softmax != 0, "KernelTable.Softmax is empty. " + NeedleRuntime.Describe());
        }
    }
}
