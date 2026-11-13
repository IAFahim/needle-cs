using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Unity.Burst;
using UnityEngine;

namespace Needle.BurstKernels
{
    public static unsafe class BurstKernelInstaller
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void SoftmaxFn(float* s, int n);

        static bool installAttempted;

        static nint laneFp;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int LaneFn();

        public static bool IsInstalled => Kernels.Active.Softmax != 0 && Kernels.Active.MatMulRows != 0;

        public static void EnsureInstalled()
        {
            if (Kernels.Provider == "portable")
                Kernels.Reset();
            Kernels.TryInstallSidecar();
            if (Kernels.Active.Softmax == 0)
            {
                FunctionPointer<SoftmaxFn> compiled =
                    BurstCompiler.CompileFunctionPointer<SoftmaxFn>(BurstSoftmax.Invoke);
                if (!compiled.IsCreated)
                    throw new InvalidOperationException(
                        "N31: Burst refused to compile BurstSoftmax.Invoke. " +
                        "The kernel must live in package source (H6); " +
                        "a precompiled DLL entry cannot be compiled.");
                KernelTable table = Kernels.Active;
                table.Softmax = (nint)compiled.Value;
                Kernels.Install(table, Kernels.IsInstalled ? Kernels.Provider + "+burst" : "burst");
            }
            if (RuntimeInformation.ProcessArchitecture == Architecture.Arm64)
                BurstGemvNeon.Install();
            else
                BurstGemvTileX86.Install();
            BurstPrepare.Install();
            BurstRowOps.Install();
            BurstAttention.Install();
            BurstMlpInstaller.EnsureInstalled();
            BurstMisc.Install();
            if (laneFp == 0)
            {
                FunctionPointer<LaneFn> lane =
                    BurstCompiler.CompileFunctionPointer<LaneFn>(BurstSoftmax.Lane);
                if (!lane.IsCreated)
                    throw new InvalidOperationException("N31: Burst refused to compile BurstSoftmax.Lane.");
                laneFp = (nint)lane.Value;
            }
            installAttempted = true;
        }

        public static int CompiledLane()
        {
            EnsureInstalled();
            if (laneFp == 0) return -1;
            delegate* unmanaged[Cdecl]<int> fp = (delegate* unmanaged[Cdecl]<int>)(void*)laneFp;
            return fp();
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void AutoInstall()
        {
            if (installAttempted) return;
            try
            {
                EnsureInstalled();
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning("[needle] Burst kernel install failed, using Scalar: " + e.Message);
            }
        }

        public static string Describe()
        {
            int compiled = CompiledLane();
            string lane = compiled == 2 ? "x86-avx2-fma" : compiled == 1 ? "arm-neon" : compiled == 0 ? "scalar" : "unknown";
            return NeedleRuntime.Describe() + " burstLane=" + lane
                + " burstEnabled=" + (BurstCompiler.IsEnabled ? "1" : "0");
        }

        public static double MeasureSoftmaxCallOverheadNs(int n = 8, int calls = 200000)
        {
            if (n < 1 || n > 64) throw new ArgumentOutOfRangeException("n");
            EnsureInstalled();
            delegate* unmanaged[Cdecl]<float*, int, void> fp =
                (delegate* unmanaged[Cdecl]<float*, int, void>)(void*)Kernels.Active.Softmax;
            float* buf = stackalloc float[64];
            for (int i = 0; i < 64; i++) buf[i] = (i % 7) * 0.25f - 0.5f;
            for (int w = 0; w < 1000; w++) fp(buf, n);
            Stopwatch stopwatch = Stopwatch.StartNew();
            for (int c = 0; c < calls; c++) fp(buf, n);
            stopwatch.Stop();
            double nsPerCall = (double)stopwatch.ElapsedTicks * 1000000000.0 / Stopwatch.Frequency / calls;
            return nsPerCall;
        }
    }
}
