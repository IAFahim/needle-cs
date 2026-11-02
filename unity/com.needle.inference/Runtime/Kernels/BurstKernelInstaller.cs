// N31: Burst provider installer + H5/H6/H8 evidence.
//
// Installs the Burst-compiled Softmax entry into Needle.Kernels.Active so
// every Ops.Softmax dispatch on an ns2.1 build (CoreCLR Editor, IL2CPP and
// Mono players) runs the Burst lane instead of Scalar. Net10 builds keep
// their direct-SIMD branches ahead of the table, so this is a no-op there
// beyond the Describe string: provider selection (plan §2) prefers the
// fastest installed provider, and an explicit Kernels.Install always wins
// because EnsureInstalled never overwrites a filled Softmax slot.
//
// H6 (Burst compiles package source, not the precompiled DLL): the kernel
// lives in this source asmdef, so Burst's IL post-processing sees it and
// CompileFunctionPointer returns a created pointer. A [BurstCompile]
// method inside the precompiled Needle.Inference.dll could not be compiled
// this way (Burst only rewrites source assemblies). The EditMode test
// asserts IsCreated plus golden outputs, which settles H6 positively: the
// pointer under test provably came from package source.
//
// H8 (no AVX-VNNI / AVX-512 in the installed Burst): settled by reflecting
// the Unity.Burst assembly API — X86 nested types are exactly Avx, Avx2,
// Bmi1, Bmi2, F16C, Fma, Popcnt, Sse, Sse2, Sse3, Sse4_1, Sse4_2, Ssse3.
// Softmax needs neither extension, so N31 is unaffected; the integer
// matmul ports (N32+) must use the Avx2Dot sequence per the plan.
//
// C# 9 safe: block namespace, no target-typed new.
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Unity.Burst;
using UnityEngine;

namespace Needle.BurstKernels
{
    public static unsafe class BurstKernelInstaller
    {
        // Cdecl delegate for the table slot: void(float* s, int n).
        // The UnmanagedFunctionPointer attribute pins the calling
        // convention so the compiled pointer matches the KernelTable ABI
        // (delegate* unmanaged[Cdecl]<float*, int, void>).
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void SoftmaxFn(float* s, int n);

        static bool installAttempted;

        static nint laneFp;

        // Cdecl delegate for the lane reporter: int(void).
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int LaneFn();

        public static bool IsInstalled => Kernels.Active.Softmax != 0;

        // Idempotent: safe from the auto-install hooks below, from the
        // Editor bootstrap, and from tests. Never overwrites an explicit
        // or foreign provider's Softmax entry (plan §2 selection order).
        public static void EnsureInstalled()
        {
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

        // Which lane the COMPILED code actually runs (2/1/0, -1 unknown).
        // A managed IsSupported read always says False (probed on host
        // CoreCLR), so this calli into the compiled reporter is the only
        // truthful source; Describe() below uses it.
        public static int CompiledLane()
        {
            EnsureInstalled();
            if (laneFp == 0) return -1;
            delegate* unmanaged[Cdecl]<int> fp = (delegate* unmanaged[Cdecl]<int>)(void*)laneFp;
            return fp();
        }


        // Player + play-mode entry: BeforeSceneLoad so the table is ready
        // before Model.Open runs. Guarded: inference must survive a
        // missing Burst toolchain (falls back to Scalar, logs once).
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

        // One-line startup report: tier/provider plus the lane the installed
        // binary actually compiled for. Uses CompiledLane(), never a managed
        // IsSupported read (managed evaluation always says False, probed).
        public static string Describe()
        {
            int compiled = CompiledLane();
            string lane = compiled == 2 ? "x86-avx2-fma" : compiled == 1 ? "arm-neon" : compiled == 0 ? "scalar" : "unknown";
            return NeedleRuntime.Describe() + " burstLane=" + lane
                + " burstEnabled=" + (BurstCompiler.IsEnabled ? "1" : "0");
        }

        // H5 microbench: time the installed Softmax function pointer from
        // the calling thread (EditMode test thread or any pool worker; a
        // raw calli needs no Unity thread state). n is small on purpose:
        // the per-call work is one vector, so ns/call is an upper bound
        // on the function-pointer call overhead the plan budgets at 100ns.
        // Returns nanoseconds per call.
        public static double MeasureSoftmaxCallOverheadNs(int n = 8, int calls = 200000)
        {
            if (n < 1 || n > 64) throw new ArgumentOutOfRangeException("n");
            EnsureInstalled();
            delegate* unmanaged[Cdecl]<float*, int, void> fp =
                (delegate* unmanaged[Cdecl]<float*, int, void>)(void*)Kernels.Active.Softmax;
            float* buf = stackalloc float[64];
            for (int i = 0; i < 64; i++) buf[i] = (i % 7) * 0.25f - 0.5f;
            // Warmup: trigger Burst JIT/AOT page-in off the clock.
            for (int w = 0; w < 1000; w++) fp(buf, n);
            Stopwatch stopwatch = Stopwatch.StartNew();
            for (int c = 0; c < calls; c++) fp(buf, n);
            stopwatch.Stop();
            double nsPerCall = (double)stopwatch.ElapsedTicks * 1000000000.0 / Stopwatch.Frequency / calls;
            return nsPerCall;
        }
    }
}
