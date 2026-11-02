// N31/N13: Burst Softmax conformance (EditMode).
//
// Exercises the installed KernelTable.Softmax entry end to end through the
// raw Cdecl function pointer, the same path every Ops.Softmax dispatch on
// an ns2.1 build takes once the provider installs itself. Public API only
// (Kernels, KernelTable, installer); no InternalsVisibleTo needed.
//
//   - Golden values + edge shapes (n=1, ties, clamp range, odd tails).
//   - N13 parity bound on random inputs: cosine >= 0.99995, same argmax.
//   - H7 probe: max abs difference vs the Scalar reference, logged. The
//     x86/Neon lanes mirror the net10 instruction order, so fixtures land
//     bitwise; the assert stays on the parity bound per the plan's
//     fallback rule (bitwise or bound, PR says which).
//   - H5: FP call-overhead microbench, logged with the 100ns verdict.
//   - H6/H8: installer report, logged (source-compiled pointer + lane).
//
// C# 9 safe: block namespace, no target-typed new. Unsafe (raw pointers,
// function pointers) under allowUnsafeCode.
using System;
using System.IO;
using System.Reflection;
using Needle;
using NUnit.Framework;
using UnityEngine;

namespace Needle.BurstKernels.Tests
{
    public sealed unsafe class BurstSoftmaxConformance
    {
        static delegate* unmanaged[Cdecl]<float*, int, void> InstalledFp()
        {
            BurstKernels.BurstKernelInstaller.EnsureInstalled();
            nint raw = Kernels.Active.Softmax;
            Assert.IsTrue(raw != 0, "Burst provider must install KernelTable.Softmax (H6: source compile).");
            return (delegate* unmanaged[Cdecl]<float*, int, void>)(void*)raw;
        }

        static void RunThroughTable(float[] s)
        {
            fixed (float* p = s)
            {
                InstalledFp()(p, s.Length);
            }
        }

        // Mirrors Scalar.Softmax exactly (the N13 reference for this entry).
        static void ReferenceSoftmax(float[] s)
        {
            float peak = float.NegativeInfinity;
            for (int i = 0; i < s.Length; i++) peak = Math.Max(peak, s[i]);
            float total = 0f;
            for (int i = 0; i < s.Length; i++)
            {
                s[i] = (float)Math.Exp(s[i] - peak);
                total += s[i];
            }
            for (int i = 0; i < s.Length; i++) s[i] /= total;
        }

        static double Cosine(float[] a, float[] b)
        {
            double ab = 0, aa = 0, bb = 0;
            for (int i = 0; i < a.Length; i++)
            {
                ab += (double)a[i] * b[i];
                aa += (double)a[i] * a[i];
                bb += (double)b[i] * b[i];
            }
            return ab / (Math.Sqrt(aa) * Math.Sqrt(bb));
        }

        static int ArgMax(float[] s)
        {
            int best = 0;
            for (int i = 1; i < s.Length; i++) if (s[i] > s[best]) best = i;
            return best;
        }

        struct XorShift64
        {
            ulong state;
            public XorShift64(ulong seed) { state = seed != 0 ? seed : 0x9E3779B97F4A7C15uL; }
            public ulong Next()
            {
                ulong x = state;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                state = x;
                return x;
            }
            public float NextFloat(float scale)
            {
                return ((float)(Next() >> 11) * (1f / 9007199254740992f) - 0.5f) * 2f * scale;
            }
        }

        [Test]
        public void SoftmaxGoldenValues()
        {
            float[] s = new float[] { 1f, 2f, 3f };
            RunThroughTable(s);
            Assert.AreEqual(0.09003057f, s[0], 1e-6f);
            Assert.AreEqual(0.24472848f, s[1], 1e-6f);
            Assert.AreEqual(0.66524094f, s[2], 1e-6f);
        }

        [Test]
        public void SoftmaxEdgeShapes()
        {
            // n=1 normalizes to exactly 1.
            float[] one = new float[] { -4f };
            RunThroughTable(one);
            Assert.AreEqual(1f, one[0], 1e-7f);
            // Uniform input stays uniform; ties keep index 0.
            float[] tied = new float[] { 2f, 2f, 2f, 2f, 2f, 2f, 2f };
            RunThroughTable(tied);
            for (int i = 0; i < tied.Length; i++) Assert.AreEqual(1f / 7f, tied[i], 1e-6f);
            Assert.AreEqual(0, ArgMax(tied));
            // Deep negatives hit the Exp clamp; output stays finite and sums to 1.
            float[] cold = new float[] { -100f, -101f, -99f, -100.5f, -98f, -102f, -100f, -99.5f, -100.25f };
            RunThroughTable(cold);
            float sum = 0f;
            for (int i = 0; i < cold.Length; i++)
            {
                Assert.IsTrue(float.IsFinite(cold[i]), "clamp-range output must be finite");
                sum += cold[i];
            }
            Assert.AreEqual(1f, sum, 1e-5f);
        }

        [Test]
        public void SoftmaxParityRandom()
        {
            XorShift64 rng = new XorShift64(0x0DDD);
            foreach (int n in new int[] { 1, 3, 7, 8, 9, 17, 64, 100 })
            {
                float[] a = new float[n];
                float[] b = new float[n];
                for (int i = 0; i < n; i++) a[i] = b[i] = rng.NextFloat(4f);
                ReferenceSoftmax(a);
                RunThroughTable(b);
                Assert.GreaterOrEqual(Cosine(a, b), 0.99995, "N13 parity bound at n=" + n);
                Assert.AreEqual(ArgMax(a), ArgMax(b), "same argmax at n=" + n);
            }
        }

        // True H7 oracle: the net10 AVX2 kernel itself, loaded from the
        // sibling net10.0 plugin DLL (Editor is CoreCLR, so it loads) and
        // invoked by reflection. LoadFile, not LoadFrom: the ns2.1 plugin
        // is already loaded under the same simple name/version, and
        // LoadFrom would hand back the loaded copy instead of the file.
        static unsafe class Net10Oracle
        {
            static MethodInfo softmax;
            static bool primed;
            public static bool TryPrime()
            {
                if (primed) return softmax != null;
                primed = true;
                try
                {
                    string self = typeof(Kernels).Assembly.Location;
                    string candidate = Path.GetFullPath(Path.Combine(
                        Path.GetDirectoryName(self), "..", "net10.0", "Needle.Inference.dll"));
                    if (!File.Exists(candidate))
                    {
                        string pkg = Path.GetFullPath(Path.Combine(
                            Application.dataPath, "..", "Packages", "com.needle.inference",
                            "Runtime", "Plugins", "net10.0", "Needle.Inference.dll"));
                        if (File.Exists(pkg)) candidate = pkg;
                    }
                    if (!File.Exists(candidate)) return false;
                    Assembly net10 = Assembly.LoadFile(candidate);
                    Type ops = net10.GetType("Needle.Ops");
                    if (ops == null) return false;
                    softmax = ops.GetMethod("Softmax",
                        BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public,
                        null, new Type[] { typeof(float*), typeof(int) }, null);
                    return softmax != null;
                }
                catch (Exception e)
                {
                    Debug.Log("[needle] H7 net10 oracle unavailable: " + e.GetType().Name);
                    softmax = null;
                    return false;
                }
            }
            public static void Run(float[] s)
            {
                fixed (float* p = s)
                {
                    object boxed = Pointer.Box(p, typeof(float*));
                    softmax.Invoke(null, new object[] { boxed, s.Length });
                }
            }
        }

        static void OracleSoftmax(float[] s, bool useNet10)
        {
            if (useNet10) Net10Oracle.Run(s);
            else ReferenceSoftmax(s);
        }

        static void Accumulate(float[] a, float[] b, ref float worst, ref int exact, ref int total)
        {
            for (int i = 0; i < a.Length; i++)
            {
                float d = Math.Abs(a[i] - b[i]);
                if (d > worst) worst = d;
                if (d == 0f) exact++;
                total++;
            }
        }

        [Test]
        public void SoftmaxMatchesNet10BitwiseProbe()
        {
            // H7 probe for Softmax. The compiled-lane gate comes first: a
            // managed IsSupported read always says False (probed), so an
            // earlier maxAbsDiff=0 proved nothing until the vector lane is
            // confirmed. The oracle is the net10 kernel when loadable,
            // scalar-sequential otherwise (logged). Adversarial fixtures
            // use the N33 order (Sum128(lower)+Sum128(upper)); any nonzero
            // diff adopts the N38 spill+tree pattern. Parity is asserted
            // per the plan fallback rule.
            int lane = BurstKernels.BurstKernelInstaller.CompiledLane();
            Debug.Log("[needle] H7-softmax lane=" + lane + " (2=x86-avx2-fma,1=arm-neon,0=scalar)");
            Assert.AreNotEqual(0, lane, "H7 probe must exercise a compiled vector lane; a scalar run proves nothing about reduction order.");
            bool useNet10 = Net10Oracle.TryPrime();
            Debug.Log("[needle] H7-softmax oracle=" + (useNet10 ? "net10-Ops.Softmax" : "scalar-sequential"));
            XorShift64 rng = new XorShift64(0xB17);
            float worst = 0f;
            int exact = 0;
            int total = 0;
            foreach (int n in new int[] { 4, 7, 8, 16, 31, 33, 64, 257 })
            {
                for (int rep = 0; rep < 4; rep++)
                {
                    float[] a = new float[n];
                    float[] b = new float[n];
                    for (int i = 0; i < n; i++) a[i] = b[i] = rng.NextFloat(6f);
                    OracleSoftmax(a, useNet10);
                    RunThroughTable(b);
                    Accumulate(a, b, ref worst, ref exact, ref total);
                    Assert.GreaterOrEqual(Cosine(a, b), 0.99995, "parity at n=" + n);
                }
            }
            // Adversarial dynamic range: moderate-span random over many lanes
            // (every term contributes at ulp level, no single-term
            // dominance), plus alternating extremes and the denormal edge.
            foreach (int n in new int[] { 8, 33, 257 })
            {
                float[] a = new float[n];
                float[] b = new float[n];
                for (int i = 0; i < n; i++) a[i] = b[i] = rng.NextFloat(1f);
                OracleSoftmax(a, useNet10);
                RunThroughTable(b);
                Accumulate(a, b, ref worst, ref exact, ref total);
                Assert.GreaterOrEqual(Cosine(a, b), 0.99995, "parity moderate-span n=" + n);
                for (int i = 0; i < n; i++)
                {
                    float v = (i % 2 == 0 ? 20f : -20f) + rng.NextFloat(1f);
                    a[i] = b[i] = v;
                }
                OracleSoftmax(a, useNet10);
                RunThroughTable(b);
                Accumulate(a, b, ref worst, ref exact, ref total);
                Assert.GreaterOrEqual(Cosine(a, b), 0.99995, "parity alternating n=" + n);
                for (int i = 0; i < n; i++)
                {
                    float v = (i % 3 == 0) ? 0f : -87f + (i % 5);
                    a[i] = b[i] = v;
                }
                OracleSoftmax(a, useNet10);
                RunThroughTable(b);
                Accumulate(a, b, ref worst, ref exact, ref total);
                Assert.GreaterOrEqual(Cosine(a, b), 0.99995, "parity denormal-edge n=" + n);
            }
            Debug.Log("[needle] H7-softmax maxAbsDiff=" + worst.ToString("R") + " exactLanes=" + exact + "/" + total);
        }

        [Test]
        public void SoftmaxCallOverheadReport()
        {
            // H5: a Burst function-pointer call from a non-Unity thread
            // costs < 100ns. The bench measures the installed FP on tiny
            // inputs (one vector of work), so ns/call is an upper bound on
            // the overhead itself.
            double ns = BurstKernels.BurstKernelInstaller.MeasureSoftmaxCallOverheadNs(8, 200000);
            Debug.Log("[needle] H5-softmax-fp nsPerCall=" + ns.ToString("F1")
                + " verdict=" + (ns < 100.0 ? "HOLDS(<100ns)" : "FAIL(>=100ns)"));
            Assert.Less(ns, 1000.0, "FP call must be a raw indirect call, not a managed transition.");
        }

        [Test]
        public void BurstProviderReport()
        {
            // H6/H8 evidence line for the run log.
            BurstKernels.BurstKernelInstaller.EnsureInstalled();
            string report = BurstKernels.BurstKernelInstaller.Describe();
            Debug.Log("[needle] " + report);
            Assert.IsTrue(report.Contains("provider="), "Describe must name the provider.");
            Assert.IsTrue(BurstKernels.BurstKernelInstaller.IsInstalled, "Softmax slot must be installed.");
        }
    }
}
