// N38: Burst row ops (§2.1 RowOps list except Softmax, which N31 owns).
//
// Ports Ops/Quant row ops 1:1 to Unity.Burst.Intrinsics: the x86 lane mirrors
// the net10 AVX2+FMA path (same constants, same FMA order, same max/sum
// reductions), the Neon lane mirrors Neon.cs (same polynomial/order, same
// pairwise folds). Anything without AVX2+FMA or Neon falls back to a scalar
// loop that mirrors Scalar.cs. Every entry does >= ~1 us of work; arguments
// are blittable only; no allocation, no static mutable state, safe from any
// thread (pool workers call via the table).
//
// Exactness (N13 conformance + H7 bitwise probes):
//   - Bitwise vs the matching net10 lane (x86 vs AVX2 tier, Neon vs Neon.cs):
//     RmsUnit, ZeroCenteredRms, RmsFakeA8, CenteredRmsFakeA8, GateFakeA8,
//     PostNormAdd, Mix4, Combine4, CombineDiff4, Taps3, Rope, Dot,
//     AddScaled, ScaleAdd, MultiplyAccumulate, QuantizeInt8, FakeA8.
//     The FMA chains keep net10 order (including the Add+Mul+Mul inner forms
//     of ZeroCenteredRms/PostNormAdd, SIMD-05); the max reductions are exact
//     (no rounding); the final 8-wide sum reproduces the BCL Vector256.Sum
//     pairwise tree via a spill plus scalar adds (a hadd/hadd guess matches
//     the BCL only ~40%: probed 200/200, see Sum256X86). QuantizeInt8 keeps
//     the lane's rounding (x86 round-to-nearest-even via cvtps, Neon truncate
//     via vcvt); FakeA8/RoundTrip agree bit-for-bit on finite inputs (|x| <=
//     128 < 2^22, so the magic-constant RoundEven and hardware vrnd agree);
//     NaN inputs are implementation-defined on every lane, as in net10.
//   - Parity bound vs Scalar (cosine >= 0.99995, same argmax): everywhere.
//   - AVX-512 tier: net10 uses OpsAvx512 there (2x512-bit accumulators), so
//     on Avx512Bw machines this file's x86 lane meets the parity bound, not
//     H7 bitwise. H7 bitwise holds against the AVX2 tier.
//   - Denormals: same FTZ caveat as N31. Exp is only used by GateFakeA8's
//     sigmoid (copied bit-for-bit from BurstSoftmax) and Sinkhorn (scalar
//     MathF.Exp, same call order as net10).
//
// H8 note: this file needs no AVX-VNNI and no AVX-512. Integer narrowing on
// x86 uses packs (packssdw/packsswb), the Avx2Dot-sequence era is N32+.
//
// C# 9 safe: block namespace, no target-typed new, no static lambdas.
using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;

namespace Needle.BurstKernels
{
    public static unsafe class BurstRowOps
    {
        const float RmsEpsilon = 1e-6f;

        // ---- entry points (installed into KernelTable; Cdecl, see delegates) ----

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeRmsFakeA8(float* x, int n, float* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { RmsFakeA8X86(x, n, destination); return; }
            if (Arm.Neon.IsNeonSupported) { RmsFakeA8Neon(x, n, destination); return; }
            RmsFakeA8Scalar(x, n, destination);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeCenteredRmsFakeA8(float* x, float* scale, int n, float* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { CenteredRmsFakeA8X86(x, scale, n, destination); return; }
            if (Arm.Neon.IsNeonSupported) { CenteredRmsFakeA8Neon(x, scale, n, destination); return; }
            CenteredRmsFakeA8Scalar(x, scale, n, destination);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeGateFakeA8(float* x, float* gate, int n)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { GateFakeA8X86(x, gate, n); return; }
            if (Arm.Neon.IsNeonSupported) { GateFakeA8Neon(x, gate, n); return; }
            GateFakeA8Scalar(x, gate, n);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokePostNormAdd(float* residual, float* x, float* scale, float gate, int n)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { PostNormAddX86(residual, x, scale, gate, n); return; }
            if (Arm.Neon.IsNeonSupported) { PostNormAddNeon(residual, x, scale, gate, n); return; }
            PostNormAddScalar(residual, x, scale, gate, n);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeCombineDiff4(float* s, int dim, float w0, float w1, float w2, float w3, float* x, float* u, float post, float* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { CombineDiff4X86(s, dim, w0, w1, w2, w3, x, u, post, destination); return; }
            if (Arm.Neon.IsNeonSupported) { CombineDiff4Neon(s, dim, w0, w1, w2, w3, x, u, post, destination); return; }
            CombineDiff4Scalar(s, dim, w0, w1, w2, w3, x, u, post, destination);
        }
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeMix4(float* s, int dim, float w0, float w1, float w2, float w3, float* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { Mix4X86(s, dim, w0, w1, w2, w3, destination); return; }
            if (Arm.Neon.IsNeonSupported) { Mix4Neon(s, dim, w0, w1, w2, w3, destination); return; }
            Mix4Scalar(s, dim, w0, w1, w2, w3, destination);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeCombine4(float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { Combine4X86(s, dim, w0, w1, w2, w3, y, post, destination); return; }
            if (Arm.Neon.IsNeonSupported) { Combine4Neon(s, dim, w0, w1, w2, w3, y, post, destination); return; }
            Combine4Scalar(s, dim, w0, w1, w2, w3, y, post, destination);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeTaps3(float* taps, int width, float* current, float* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { Taps3X86(taps, width, current, destination); return; }
            if (Arm.Neon.IsNeonSupported) { Taps3Neon(taps, width, current, destination); return; }
            Taps3Scalar(taps, width, current, destination);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeRope(float* x, int heads, int headDim, float* cos, float* sin)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { RopeX86(x, heads, headDim, cos, sin); return; }
            if (Arm.Neon.IsNeonSupported) { RopeNeon(x, heads, headDim, cos, sin); return; }
            RopeScalar(x, heads, headDim, cos, sin);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeZeroCenteredRms(float* x, float* scale, int n, float* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { ZeroCenteredRmsX86(x, scale, n, destination); return; }
            if (Arm.Neon.IsNeonSupported) { ZeroCenteredRmsNeon(x, scale, n, destination); return; }
            ZeroCenteredRmsScalar(x, scale, n, destination);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeRmsUnit(float* x, int n, float* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { RmsUnitX86(x, n, destination); return; }
            if (Arm.Neon.IsNeonSupported) { RmsUnitNeon(x, n, destination); return; }
            RmsUnitScalar(x, n, destination);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static float InvokeDot(float* a, float* b, int n)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) return DotX86(a, b, n);
            if (Arm.Neon.IsNeonSupported) return DotNeon(a, b, n);
            return DotScalar(a, b, n);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeAddScaled(float* destination, float* x, float scale, int n)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { AddScaledX86(destination, x, scale, n); return; }
            if (Arm.Neon.IsNeonSupported) { AddScaledNeon(destination, x, scale, n); return; }
            AddScaledScalar(destination, x, scale, n);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeScaleAdd(float* destination, float scale, float* x, float xScale, int n)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { ScaleAddX86(destination, scale, x, xScale, n); return; }
            if (Arm.Neon.IsNeonSupported) { ScaleAddNeon(destination, scale, x, xScale, n); return; }
            ScaleAddScalar(destination, scale, x, xScale, n);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeMultiplyAccumulate(float* destination, float* a, float* b, int n)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { MultiplyAccumulateX86(destination, a, b, n); return; }
            if (Arm.Neon.IsNeonSupported) { MultiplyAccumulateNeon(destination, a, b, n); return; }
            MultiplyAccumulateScalar(destination, a, b, n);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static float InvokeQuantizeInt8(float* x, int n, sbyte* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) return QuantizeInt8X86(x, n, destination);
            if (Arm.Neon.IsNeonSupported) return QuantizeInt8Neon(x, n, destination);
            return QuantizeInt8Scalar(x, n, destination);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static float InvokeFakeA8(float* x, int n)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) return FakeA8X86(x, n);
            if (Arm.Neon.IsNeonSupported) return FakeA8Neon(x, n);
            return FakeA8Scalar(x, n);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeSinkhorn4x4(float* logits, float* result)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { SinkhornX86(logits, result); return; }
            if (Arm.Neon.IsNeonSupported) { SinkhornNeon(logits, result); return; }
            SinkhornScalar(logits, result);
        }

        // ---- x86 shared helpers (mirror Ops.Exp / Max8 / Vector256.Sum) ----

        // Mirrors Ops.Exp bit for bit (copied from BurstSoftmax): clamp,
        // round, two fnmadd reductions, six-term FMA polynomial, 2^n scale.
        static v256 ExpX86(v256 x)
        {
            x = X86.Avx.mm256_min_ps(
                X86.Avx.mm256_max_ps(x, X86.Avx.mm256_set1_ps(-87.3f)),
                X86.Avx.mm256_set1_ps(88.7f));
            v256 n = X86.Avx.mm256_round_ps(
                X86.Avx.mm256_mul_ps(x, X86.Avx.mm256_set1_ps(1.44269504f)), 8);
            v256 r = X86.Fma.mm256_fnmadd_ps(n, X86.Avx.mm256_set1_ps(0.693359375f), x);
            r = X86.Fma.mm256_fnmadd_ps(n, X86.Avx.mm256_set1_ps(-2.12194440e-4f), r);
            v256 p = X86.Avx.mm256_set1_ps(1.9875691500e-4f);
            p = X86.Fma.mm256_fmadd_ps(p, r, X86.Avx.mm256_set1_ps(1.3981999507e-3f));
            p = X86.Fma.mm256_fmadd_ps(p, r, X86.Avx.mm256_set1_ps(8.3334519073e-3f));
            p = X86.Fma.mm256_fmadd_ps(p, r, X86.Avx.mm256_set1_ps(4.1665795894e-2f));
            p = X86.Fma.mm256_fmadd_ps(p, r, X86.Avx.mm256_set1_ps(1.6666665459e-1f));
            p = X86.Fma.mm256_fmadd_ps(p, r, X86.Avx.mm256_set1_ps(5.0000001201e-1f));
            v256 y = X86.Fma.mm256_fmadd_ps(
                p,
                X86.Avx.mm256_mul_ps(r, r),
                X86.Avx.mm256_add_ps(r, X86.Avx.mm256_set1_ps(1f)));
            v256 scaled = X86.Avx.mm256_castsi256_ps(
                X86.Avx2.mm256_slli_epi32(
                    X86.Avx2.mm256_add_epi32(X86.Avx.mm256_cvtps_epi32(n), new v256(127)), 23));
            return X86.Avx.mm256_mul_ps(y, scaled);
        }

        // Mirrors Ops.Sigmoid: 1 / (1 + exp(-x)).
        static v256 SigmoidX86(v256 x)
        {
            v256 one = X86.Avx.mm256_set1_ps(1f);
            return X86.Avx.mm256_div_ps(one, X86.Avx.mm256_add_ps(one, ExpX86(X86.Avx.mm256_sub_ps(X86.Avx.mm256_set1_ps(0f), x))));
        }

        // Mirrors Ops.Max8: max(lower, upper), then the 0x4E and 0xB1
        // shuffles, lane 0 out via cvtss (a pure lane read, no rounding).
        static float Max8X86(v256 v)
        {
            v128 halves = X86.Sse.max_ps(
                X86.Avx.mm256_castps256_ps128(v),
                X86.Avx.mm256_extractf128_ps(v, 1));
            v128 pairs = X86.Sse.max_ps(halves, X86.Sse.shuffle_ps(halves, halves, 0x4E));
            v128 quad = X86.Sse.max_ps(pairs, X86.Sse.shuffle_ps(pairs, pairs, 0xB1));
            return X86.Sse.cvtss_f32(quad);
        }

        // Mirrors the BCL Vector256.Sum order used by Ops: Sum128(lower) +
        // Sum128(upper), where Sum128 is hadd,hadd lane 0 (probed 200/200 on
        // AVX2 hardware; a single add(low,high)+hadd+hadd folds (l+h) lanes
        // first and matches the BCL only ~40%, off by 1 ulp). A stack spill
        // plus scalar adds is exactly that tree under FloatMode.Strict.
        static float Sum256X86(v256 v)
        {
            float* lanes = stackalloc float[8];
            X86.Avx.mm256_storeu_ps(lanes, v);
            float lo = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
            float hi = (lanes[4] + lanes[5]) + (lanes[6] + lanes[7]);
            return lo + hi;
        }

        // Mirrors Ops.SumOfSquares: 4 FMA accumulators, 32 bulk, 16 tail,
        // 8 tail, scalar FMA tail. Vector256.Abs is andnot with -0.0.
        static float SumOfSquaresX86(float* x, int n)
        {
            v256 s0 = X86.Avx.mm256_setzero_ps();
            v256 s1 = X86.Avx.mm256_setzero_ps();
            v256 s2 = X86.Avx.mm256_setzero_ps();
            v256 s3 = X86.Avx.mm256_setzero_ps();
            int i = 0;
            int bulk = n & ~31;
            for (; i < bulk; i += 32)
            {
                v256 a0 = X86.Avx.mm256_loadu_ps(x + i);
                v256 a1 = X86.Avx.mm256_loadu_ps(x + i + 8);
                v256 a2 = X86.Avx.mm256_loadu_ps(x + i + 16);
                v256 a3 = X86.Avx.mm256_loadu_ps(x + i + 24);
                s0 = X86.Fma.mm256_fmadd_ps(a0, a0, s0);
                s1 = X86.Fma.mm256_fmadd_ps(a1, a1, s1);
                s2 = X86.Fma.mm256_fmadd_ps(a2, a2, s2);
                s3 = X86.Fma.mm256_fmadd_ps(a3, a3, s3);
            }
            for (; i + 16 <= n; i += 16)
            {
                v256 a = X86.Avx.mm256_loadu_ps(x + i);
                v256 b = X86.Avx.mm256_loadu_ps(x + i + 8);
                s0 = X86.Fma.mm256_fmadd_ps(a, a, s0);
                s1 = X86.Fma.mm256_fmadd_ps(b, b, s1);
            }
            if (i + 8 <= n)
            {
                v256 a = X86.Avx.mm256_loadu_ps(x + i);
                s2 = X86.Fma.mm256_fmadd_ps(a, a, s2);
                i += 8;
            }
            float tail = 0f;
            for (; i < n; i++) tail = ((x[i]) * (x[i]) + (tail));
            return Sum256X86(X86.Avx.mm256_add_ps(X86.Avx.mm256_add_ps(s0, s1), X86.Avx.mm256_add_ps(s2, s3))) + tail;
        }

        // Mirrors Ops.Dot: same 4-accumulator shape as SumOfSquares.
        static float DotX86(float* a, float* b, int n)
        {
            v256 s0 = X86.Avx.mm256_setzero_ps();
            v256 s1 = X86.Avx.mm256_setzero_ps();
            v256 s2 = X86.Avx.mm256_setzero_ps();
            v256 s3 = X86.Avx.mm256_setzero_ps();
            int i = 0;
            int bulk = n & ~31;
            for (; i < bulk; i += 32)
            {
                s0 = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_loadu_ps(a + i), X86.Avx.mm256_loadu_ps(b + i), s0);
                s1 = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_loadu_ps(a + i + 8), X86.Avx.mm256_loadu_ps(b + i + 8), s1);
                s2 = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_loadu_ps(a + i + 16), X86.Avx.mm256_loadu_ps(b + i + 16), s2);
                s3 = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_loadu_ps(a + i + 24), X86.Avx.mm256_loadu_ps(b + i + 24), s3);
            }
            for (; i + 16 <= n; i += 16)
            {
                s0 = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_loadu_ps(a + i), X86.Avx.mm256_loadu_ps(b + i), s0);
                s1 = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_loadu_ps(a + i + 8), X86.Avx.mm256_loadu_ps(b + i + 8), s1);
            }
            if (i + 8 <= n)
            {
                s2 = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_loadu_ps(a + i), X86.Avx.mm256_loadu_ps(b + i), s2);
                i += 8;
            }
            float tail = 0f;
            for (; i < n; i++) tail = ((a[i]) * (b[i]) + (tail));
            return Sum256X86(X86.Avx.mm256_add_ps(X86.Avx.mm256_add_ps(s0, s1), X86.Avx.mm256_add_ps(s2, s3))) + tail;
        }

        static float InverseRmsX86(float* x, int n)
        {
            return 1f / MathF.Sqrt(SumOfSquaresX86(x, n) / n + RmsEpsilon);
        }

        // Mirrors Ops.AbsMax (no scalar tail in net10: n is a multiple of 8).
        static float AbsMaxX86(float* x, int n)
        {
            v256 sign = X86.Avx.mm256_set1_ps(-0.0f);
            v256 peak = X86.Avx.mm256_setzero_ps();
            for (int i = 0; i < n; i += 8)
                peak = X86.Avx.mm256_max_ps(peak, X86.Avx.mm256_andnot_ps(sign, X86.Avx.mm256_loadu_ps(x + i)));
            return Max8X86(peak);
        }

        // Mirrors Ops.RoundTrip via Quant.RoundTrip: round-half-even (mode 8),
        // clamp, rescale.
        static void RoundTripX86(float* x, int n, float inverse, float scale)
        {
            v256 inv = X86.Avx.mm256_set1_ps(inverse);
            v256 sc = X86.Avx.mm256_set1_ps(scale);
            v256 low = X86.Avx.mm256_set1_ps(-128f);
            v256 high = X86.Avx.mm256_set1_ps(127f);
            for (int i = 0; i < n; i += 8)
            {
                v256 q = X86.Avx.mm256_round_ps(X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(x + i), inv), 8);
                X86.Avx.mm256_storeu_ps(x + i, X86.Avx.mm256_mul_ps(X86.Avx.mm256_min_ps(X86.Avx.mm256_max_ps(q, low), high), sc));
            }
        }

        static float FoldPeakX86(v256 peak)
        {
            return Max8X86(peak);
        }

        // Mirrors Ops.FinishFakeA8.
        static void FinishFakeA8X86(float* x, int n, float peak)
        {
            float scale = peak > 0f ? peak / 127f : 1f;
            RoundTripX86(x, n, 1f / scale, scale);
        }

        static void RmsUnitX86(float* x, int n, float* destination)
        {
            v256 inverse = X86.Avx.mm256_set1_ps(InverseRmsX86(x, n));
            for (int i = 0; i < n; i += 8)
                X86.Avx.mm256_storeu_ps(destination + i, X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(x + i), inverse));
        }

        static void ZeroCenteredRmsX86(float* x, float* scale, int n, float* destination)
        {
            // Add + Mul + Mul form, matching Ops (SIMD-05 reverted: the FMA
            // affine is faster but breaks exact-bits tests on other tiers).
            v256 inverse = X86.Avx.mm256_set1_ps(InverseRmsX86(x, n));
            v256 one = X86.Avx.mm256_set1_ps(1f);
            for (int i = 0; i < n; i += 8)
                X86.Avx.mm256_storeu_ps(destination + i, X86.Avx.mm256_mul_ps(X86.Avx.mm256_mul_ps(X86.Avx.mm256_add_ps(one, X86.Avx.mm256_loadu_ps(scale + i)), X86.Avx.mm256_loadu_ps(x + i)), inverse));
        }

        static void RmsFakeA8X86(float* x, int n, float* destination)
        {
            v256 inverse = X86.Avx.mm256_set1_ps(InverseRmsX86(x, n));
            v256 sign = X86.Avx.mm256_set1_ps(-0.0f);
            v256 peak = X86.Avx.mm256_setzero_ps();
            for (int i = 0; i < n; i += 8)
            {
                v256 v = X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(x + i), inverse);
                X86.Avx.mm256_storeu_ps(destination + i, v);
                peak = X86.Avx.mm256_max_ps(peak, X86.Avx.mm256_andnot_ps(sign, v));
            }
            FinishFakeA8X86(destination, n, FoldPeakX86(peak));
        }

        static void CenteredRmsFakeA8X86(float* x, float* scale, int n, float* destination)
        {
            v256 inverse = X86.Avx.mm256_set1_ps(InverseRmsX86(x, n));
            v256 one = X86.Avx.mm256_set1_ps(1f);
            v256 sign = X86.Avx.mm256_set1_ps(-0.0f);
            v256 peak = X86.Avx.mm256_setzero_ps();
            for (int i = 0; i < n; i += 8)
            {
                v256 v = X86.Avx.mm256_mul_ps(X86.Avx.mm256_mul_ps(X86.Avx.mm256_add_ps(one, X86.Avx.mm256_loadu_ps(scale + i)), X86.Avx.mm256_loadu_ps(x + i)), inverse);
                X86.Avx.mm256_storeu_ps(destination + i, v);
                peak = X86.Avx.mm256_max_ps(peak, X86.Avx.mm256_andnot_ps(sign, v));
            }
            FinishFakeA8X86(destination, n, FoldPeakX86(peak));
        }

        static void GateFakeA8X86(float* x, float* gate, int n)
        {
            v256 sign = X86.Avx.mm256_set1_ps(-0.0f);
            v256 peak = X86.Avx.mm256_setzero_ps();
            for (int i = 0; i < n; i += 8)
            {
                v256 v = X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(x + i), SigmoidX86(X86.Avx.mm256_loadu_ps(gate + i)));
                X86.Avx.mm256_storeu_ps(x + i, v);
                peak = X86.Avx.mm256_max_ps(peak, X86.Avx.mm256_andnot_ps(sign, v));
            }
            FinishFakeA8X86(x, n, FoldPeakX86(peak));
        }

        // Mirrors Ops.PostNormAdd (SIMD-05 inner form: Add + Mul + Mul, then
        // one outer FMA).
        static void PostNormAddX86(float* residual, float* x, float* scale, float gate, int n)
        {
            v256 inverse = X86.Avx.mm256_set1_ps(InverseRmsX86(x, n));
            v256 one = X86.Avx.mm256_set1_ps(1f);
            v256 g = X86.Avx.mm256_set1_ps(gate);
            for (int i = 0; i < n; i += 8)
                X86.Avx.mm256_storeu_ps(residual + i, X86.Fma.mm256_fmadd_ps(g,
                    X86.Avx.mm256_mul_ps(X86.Avx.mm256_mul_ps(X86.Avx.mm256_add_ps(one, X86.Avx.mm256_loadu_ps(scale + i)), X86.Avx.mm256_loadu_ps(x + i)), inverse),
                    X86.Avx.mm256_loadu_ps(residual + i)));
        }

        static void RopeX86(float* x, int heads, int headDim, float* cos, float* sin)
        {
            int half = headDim / 2;
            for (int h = 0; h < heads; h++)
            {
                float* lo = x + h * headDim;
                float* hi = lo + half;
                for (int i = 0; i < half; i += 8)
                {
                    v256 c = X86.Avx.mm256_loadu_ps(cos + i);
                    v256 s = X86.Avx.mm256_loadu_ps(sin + i);
                    v256 a = X86.Avx.mm256_loadu_ps(lo + i);
                    v256 b = X86.Avx.mm256_loadu_ps(hi + i);
                    X86.Avx.mm256_storeu_ps(lo + i, X86.Fma.mm256_fnmadd_ps(b, s, X86.Avx.mm256_mul_ps(a, c)));
                    X86.Avx.mm256_storeu_ps(hi + i, X86.Fma.mm256_fmadd_ps(a, s, X86.Avx.mm256_mul_ps(b, c)));
                }
            }
        }

        static void Mix4X86(float* s, int dim, float w0, float w1, float w2, float w3, float* destination)
        {
            v256 v0 = X86.Avx.mm256_set1_ps(w0);
            v256 v1 = X86.Avx.mm256_set1_ps(w1);
            v256 v2 = X86.Avx.mm256_set1_ps(w2);
            v256 v3 = X86.Avx.mm256_set1_ps(w3);
            for (int c = 0; c < dim; c += 8)
            {
                v256 acc = X86.Avx.mm256_mul_ps(v0, X86.Avx.mm256_loadu_ps(s + c));
                acc = X86.Fma.mm256_fmadd_ps(v1, X86.Avx.mm256_loadu_ps(s + dim + c), acc);
                acc = X86.Fma.mm256_fmadd_ps(v2, X86.Avx.mm256_loadu_ps(s + 2 * dim + c), acc);
                acc = X86.Fma.mm256_fmadd_ps(v3, X86.Avx.mm256_loadu_ps(s + 3 * dim + c), acc);
                X86.Avx.mm256_storeu_ps(destination + c, acc);
            }
        }

        static void Combine4X86(float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination)
        {
            v256 v0 = X86.Avx.mm256_set1_ps(w0);
            v256 v1 = X86.Avx.mm256_set1_ps(w1);
            v256 v2 = X86.Avx.mm256_set1_ps(w2);
            v256 v3 = X86.Avx.mm256_set1_ps(w3);
            v256 vp = X86.Avx.mm256_set1_ps(post);
            for (int c = 0; c < dim; c += 8)
            {
                v256 acc = X86.Avx.mm256_mul_ps(vp, X86.Avx.mm256_loadu_ps(y + c));
                acc = X86.Fma.mm256_fmadd_ps(v0, X86.Avx.mm256_loadu_ps(s + c), acc);
                acc = X86.Fma.mm256_fmadd_ps(v1, X86.Avx.mm256_loadu_ps(s + dim + c), acc);
                acc = X86.Fma.mm256_fmadd_ps(v2, X86.Avx.mm256_loadu_ps(s + 2 * dim + c), acc);
                acc = X86.Fma.mm256_fmadd_ps(v3, X86.Avx.mm256_loadu_ps(s + 3 * dim + c), acc);
                X86.Avx.mm256_storeu_ps(destination + c, acc);
            }
        }

        static void CombineDiff4X86(float* s, int dim, float w0, float w1, float w2, float w3, float* x, float* u, float post, float* destination)
        {
            v256 v0 = X86.Avx.mm256_set1_ps(w0);
            v256 v1 = X86.Avx.mm256_set1_ps(w1);
            v256 v2 = X86.Avx.mm256_set1_ps(w2);
            v256 v3 = X86.Avx.mm256_set1_ps(w3);
            v256 vp = X86.Avx.mm256_set1_ps(post);
            for (int c = 0; c < dim; c += 8)
            {
                v256 acc = X86.Avx.mm256_mul_ps(vp, X86.Avx.mm256_sub_ps(X86.Avx.mm256_loadu_ps(x + c), X86.Avx.mm256_loadu_ps(u + c)));
                acc = X86.Fma.mm256_fmadd_ps(v0, X86.Avx.mm256_loadu_ps(s + c), acc);
                acc = X86.Fma.mm256_fmadd_ps(v1, X86.Avx.mm256_loadu_ps(s + dim + c), acc);
                acc = X86.Fma.mm256_fmadd_ps(v2, X86.Avx.mm256_loadu_ps(s + 2 * dim + c), acc);
                acc = X86.Fma.mm256_fmadd_ps(v3, X86.Avx.mm256_loadu_ps(s + 3 * dim + c), acc);
                X86.Avx.mm256_storeu_ps(destination + c, acc);
            }
        }

        static void Taps3X86(float* taps, int width, float* current, float* destination)
        {
            for (int c = 0; c < width; c += 8)
            {
                v256 acc = X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(taps + c), X86.Avx.mm256_loadu_ps(current + c));
                acc = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_loadu_ps(taps + width + c), X86.Avx.mm256_loadu_ps(current + c - width), acc);
                acc = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_loadu_ps(taps + 2 * width + c), X86.Avx.mm256_loadu_ps(current + c - 2 * width), acc);
                X86.Avx.mm256_storeu_ps(destination + c, acc);
            }
        }

        static void MultiplyAccumulateX86(float* destination, float* a, float* b, int n)
        {
            for (int i = 0; i < n; i += 8)
                X86.Avx.mm256_storeu_ps(destination + i, X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_loadu_ps(a + i), X86.Avx.mm256_loadu_ps(b + i), X86.Avx.mm256_loadu_ps(destination + i)));
        }

        static void ScaleAddX86(float* destination, float scale, float* x, float xScale, int n)
        {
            v256 s = X86.Avx.mm256_set1_ps(scale);
            v256 xs = X86.Avx.mm256_set1_ps(xScale);
            for (int i = 0; i < n; i += 8)
                X86.Avx.mm256_storeu_ps(destination + i, X86.Fma.mm256_fmadd_ps(xs, X86.Avx.mm256_loadu_ps(x + i), X86.Avx.mm256_mul_ps(s, X86.Avx.mm256_loadu_ps(destination + i))));
        }

        static void AddScaledX86(float* destination, float* x, float scale, int n)
        {
            v256 s = X86.Avx.mm256_set1_ps(scale);
            for (int i = 0; i < n; i += 8)
                X86.Avx.mm256_storeu_ps(destination + i, X86.Fma.mm256_fmadd_ps(s, X86.Avx.mm256_loadu_ps(x + i), X86.Avx.mm256_loadu_ps(destination + i)));
        }

        // Mirrors Quant.QuantizeInt8: cvtps (round-to-nearest-even, same as
        // BCL ConvertToVector256Int32), packssdw, packsswb, low 8 bytes out.
        static float QuantizeInt8X86(float* x, int n, sbyte* destination)
        {
            float peak = AbsMaxX86(x, n);
            float scale = peak > 0f ? peak / 127f : 1f;
            v256 inv = X86.Avx.mm256_set1_ps(1f / scale);
            for (int i = 0; i < n; i += 8)
            {
                v256 q = X86.Avx.mm256_cvtps_epi32(X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(x + i), inv));
                v128 packed = X86.Sse2.packs_epi32(
                    X86.Avx.mm256_castsi256_si128(q),
                    X86.Avx.mm256_extractf128_si256(q, 1));
                packed = X86.Sse2.packs_epi16(packed, packed);
                *(long*)(destination + i) = X86.Sse2.cvtsi128_si64(packed);
            }
            return scale;
        }

        // Mirrors Quant.FakeA8: AbsMax + RoundTrip (same split as net10, so
        // the rounding bits match the x86 RoundTrip lane above).
        static float FakeA8X86(float* x, int n)
        {
            float peak = AbsMaxX86(x, n);
            float scale = peak > 0f ? peak / 127f : 1f;
            RoundTripX86(x, n, 1f / scale, scale);
            return scale;
        }

        // Mirrors Ops.Sinkhorn4x4. net10 uses aligned Sse loads here; loadu
        // produces identical bits and is safe for any alignment Burst sees.
        static void SinkhornX86(float* logits, float* result)
        {
            float* scratch = stackalloc float[4];
            for (int r = 0; r < 4; r++)
            {
                v128 row = X86.Sse.loadu_ps(logits + r * 4);
                X86.Sse.storeu_ps(scratch, row);
                float shift = MathF.Max(MathF.Max(scratch[0], scratch[1]), MathF.Max(scratch[2], scratch[3]));
                X86.Sse.storeu_ps(scratch, X86.Sse.sub_ps(row, X86.Sse.set1_ps(shift)));
                for (int c = 0; c < 4; c++) result[r * 4 + c] = MathF.Exp(scratch[c]);
            }
            for (int iteration = 0; iteration < 20; iteration++)
            {
                NormRow4X86(result);
                NormRow4X86(result + 4);
                NormRow4X86(result + 8);
                NormRow4X86(result + 12);
                v128 a0 = X86.Sse.loadu_ps(result);
                v128 a1 = X86.Sse.loadu_ps(result + 4);
                v128 a2 = X86.Sse.loadu_ps(result + 8);
                v128 a3 = X86.Sse.loadu_ps(result + 12);
                v128 total = X86.Sse.add_ps(X86.Sse.add_ps(X86.Sse.add_ps(a0, a1), a2), a3);
                X86.Sse.storeu_ps(scratch, total);
                if (scratch[0] > 0f && scratch[1] > 0f && scratch[2] > 0f && scratch[3] > 0f)
                {
                    X86.Sse.storeu_ps(result, X86.Sse.div_ps(a0, total));
                    X86.Sse.storeu_ps(result + 4, X86.Sse.div_ps(a1, total));
                    X86.Sse.storeu_ps(result + 8, X86.Sse.div_ps(a2, total));
                    X86.Sse.storeu_ps(result + 12, X86.Sse.div_ps(a3, total));
                }
                else
                {
                    for (int c = 0; c < 4; c++)
                    {
                        float sum = result[c] + result[4 + c] + result[8 + c] + result[12 + c];
                        if (sum > 0f) for (int rr = 0; rr < 4; rr++) result[rr * 4 + c] /= sum;
                    }
                }
            }
        }

        static void NormRow4X86(float* row)
        {
            v128 v = X86.Sse.loadu_ps(row);
            float* lanes = stackalloc float[4];
            X86.Sse.storeu_ps(lanes, v);
            X86.Sse.storeu_ps(row, X86.Sse.div_ps(v, X86.Sse.set1_ps(lanes[0] + lanes[1] + lanes[2] + lanes[3])));
        }

        // ---- Neon shared helpers (mirror Neon.Exp / Neon.HorizontalSum) ----

        // Mirrors Neon.Exp (copied from BurstSoftmax): same clamp, FRINTN
        // round, same polynomial with explicit vfmaq. The two
        // FusedMultiplySubtract steps become vfmaq with negated constants:
        // negation is exact and fma rounds once, so x + n*(-C) is bitwise
        // identical to x - n*C. The exponent moves through a 16-byte stack
        // spill (Neon has no bit-reinterpret intrinsic).
        static v128 ExpNeon(v128 x)
        {
            x = Arm.Neon.vminq_f32(
                Arm.Neon.vmaxq_f32(x, Arm.Neon.vdupq_n_f32(-87.3f)),
                Arm.Neon.vdupq_n_f32(88.7f));
            v128 n = Arm.Neon.vrndnq_f32(Arm.Neon.vmulq_f32(x, Arm.Neon.vdupq_n_f32(1.44269504f)));
            v128 r = Arm.Neon.vfmaq_f32(x, n, Arm.Neon.vdupq_n_f32(-0.693359375f));
            r = Arm.Neon.vfmaq_f32(r, n, Arm.Neon.vdupq_n_f32(2.12194440e-4f));
            v128 p = Arm.Neon.vdupq_n_f32(1.9875691500e-4f);
            p = Arm.Neon.vfmaq_f32(p, Arm.Neon.vdupq_n_f32(1.3981999507e-3f), r);
            p = Arm.Neon.vfmaq_f32(p, Arm.Neon.vdupq_n_f32(8.3334519073e-3f), r);
            p = Arm.Neon.vfmaq_f32(p, Arm.Neon.vdupq_n_f32(4.1665795894e-2f), r);
            p = Arm.Neon.vfmaq_f32(p, Arm.Neon.vdupq_n_f32(1.6666665459e-1f), r);
            p = Arm.Neon.vfmaq_f32(p, Arm.Neon.vdupq_n_f32(5.0000001201e-1f), r);
            v128 y = Arm.Neon.vfmaq_f32(
                p,
                Arm.Neon.vaddq_f32(r, Arm.Neon.vdupq_n_f32(1f)),
                Arm.Neon.vmulq_f32(r, r));
            v128 bits = Arm.Neon.vshlq_n_u32(
                Arm.Neon.vaddq_u32(
                    Arm.Neon.vcvtnq_u32_f32(n),
                    Arm.Neon.vdupq_n_u32(127u)), 23);
            uint* spill = stackalloc uint[4];
            Arm.Neon.vst1q_u32(spill, bits);
            v128 scale = Arm.Neon.vld1q_f32((float*)spill);
            return Arm.Neon.vmulq_f32(y, scale);
        }

        static v128 SigmoidNeon(v128 x)
        {
            return Arm.Neon.vdivq_f32(Arm.Neon.vdupq_n_f32(1f), Arm.Neon.vaddq_f32(Arm.Neon.vdupq_n_f32(1f), ExpNeon(Arm.Neon.vnegq_f32(x))));
        }

        // Mirrors Neon.HorizontalSum: two pairwise adds, lane 0 out.
        static float HorizontalSumNeon(v128 v)
        {
            v128 pairs = Arm.Neon.vpaddq_f32(v, v);
            v128 quad = Arm.Neon.vpaddq_f32(pairs, pairs);
            return Arm.Neon.vgetq_lane_f32(quad, 0);
        }

        static float SumOfSquaresNeon(float* x, int n)
        {
            v128 s0 = Arm.Neon.vdupq_n_f32(0f);
            v128 s1 = Arm.Neon.vdupq_n_f32(0f);
            for (int i = 0; i < n; i += 8)
            {
                v128 a = Arm.Neon.vld1q_f32(x + i);
                v128 b = Arm.Neon.vld1q_f32(x + i + 4);
                s0 = Arm.Neon.vfmaq_f32(s0, a, a);
                s1 = Arm.Neon.vfmaq_f32(s1, b, b);
            }
            return HorizontalSumNeon(Arm.Neon.vaddq_f32(s0, s1));
        }

        static float DotNeon(float* a, float* b, int n)
        {
            v128 s0 = Arm.Neon.vdupq_n_f32(0f);
            v128 s1 = Arm.Neon.vdupq_n_f32(0f);
            for (int i = 0; i < n; i += 8)
            {
                s0 = Arm.Neon.vfmaq_f32(s0, Arm.Neon.vld1q_f32(a + i), Arm.Neon.vld1q_f32(b + i));
                s1 = Arm.Neon.vfmaq_f32(s1, Arm.Neon.vld1q_f32(a + i + 4), Arm.Neon.vld1q_f32(b + i + 4));
            }
            return HorizontalSumNeon(Arm.Neon.vaddq_f32(s0, s1));
        }

        static float InverseRmsNeon(float* x, int n)
        {
            return 1f / MathF.Sqrt(SumOfSquaresNeon(x, n) / n + RmsEpsilon);
        }

        static void RmsUnitNeon(float* x, int n, float* destination)
        {
            v128 inverse = Arm.Neon.vdupq_n_f32(InverseRmsNeon(x, n));
            for (int i = 0; i < n; i += 4)
                Arm.Neon.vst1q_f32(destination + i, Arm.Neon.vmulq_f32(Arm.Neon.vld1q_f32(x + i), inverse));
        }

        static void ZeroCenteredRmsNeon(float* x, float* scale, int n, float* destination)
        {
            v128 inverse = Arm.Neon.vdupq_n_f32(InverseRmsNeon(x, n));
            v128 one = Arm.Neon.vdupq_n_f32(1f);
            for (int i = 0; i < n; i += 4)
                Arm.Neon.vst1q_f32(destination + i, Arm.Neon.vmulq_f32(Arm.Neon.vmulq_f32(Arm.Neon.vaddq_f32(one, Arm.Neon.vld1q_f32(scale + i)), Arm.Neon.vld1q_f32(x + i)), inverse));
        }

        static float AbsMaxNeon(float* x, int n)
        {
            v128 peak = Arm.Neon.vdupq_n_f32(0f);
            for (int i = 0; i < n; i += 4)
                peak = Arm.Neon.vmaxq_f32(peak, Arm.Neon.vabsq_f32(Arm.Neon.vld1q_f32(x + i)));
            return Arm.Neon.vmaxvq_f32(peak);
        }

        static void RoundTripNeon(float* x, int n, float inverse, float scale)
        {
            v128 inv = Arm.Neon.vdupq_n_f32(inverse);
            v128 sc = Arm.Neon.vdupq_n_f32(scale);
            v128 low = Arm.Neon.vdupq_n_f32(-128f);
            v128 high = Arm.Neon.vdupq_n_f32(127f);
            for (int i = 0; i < n; i += 4)
            {
                v128 q = Arm.Neon.vrndnq_f32(Arm.Neon.vmulq_f32(Arm.Neon.vld1q_f32(x + i), inv));
                Arm.Neon.vst1q_f32(x + i, Arm.Neon.vmulq_f32(Arm.Neon.vminq_f32(Arm.Neon.vmaxq_f32(q, low), high), sc));
            }
        }

        static void FinishFakeA8Neon(float* x, int n, float peak)
        {
            float scale = peak > 0f ? peak / 127f : 1f;
            RoundTripNeon(x, n, 1f / scale, scale);
        }

        static void RmsFakeA8Neon(float* x, int n, float* destination)
        {
            v128 inverse = Arm.Neon.vdupq_n_f32(InverseRmsNeon(x, n));
            v128 peak = Arm.Neon.vdupq_n_f32(0f);
            for (int i = 0; i < n; i += 4)
            {
                v128 v = Arm.Neon.vmulq_f32(Arm.Neon.vld1q_f32(x + i), inverse);
                Arm.Neon.vst1q_f32(destination + i, v);
                peak = Arm.Neon.vmaxq_f32(peak, Arm.Neon.vabsq_f32(v));
            }
            FinishFakeA8Neon(destination, n, Arm.Neon.vmaxvq_f32(peak));
        }

        static void CenteredRmsFakeA8Neon(float* x, float* scale, int n, float* destination)
        {
            v128 inverse = Arm.Neon.vdupq_n_f32(InverseRmsNeon(x, n));
            v128 one = Arm.Neon.vdupq_n_f32(1f);
            v128 peak = Arm.Neon.vdupq_n_f32(0f);
            for (int i = 0; i < n; i += 4)
            {
                v128 v = Arm.Neon.vmulq_f32(Arm.Neon.vmulq_f32(Arm.Neon.vaddq_f32(one, Arm.Neon.vld1q_f32(scale + i)), Arm.Neon.vld1q_f32(x + i)), inverse);
                Arm.Neon.vst1q_f32(destination + i, v);
                peak = Arm.Neon.vmaxq_f32(peak, Arm.Neon.vabsq_f32(v));
            }
            FinishFakeA8Neon(destination, n, Arm.Neon.vmaxvq_f32(peak));
        }

        static void GateFakeA8Neon(float* x, float* gate, int n)
        {
            v128 peak = Arm.Neon.vdupq_n_f32(0f);
            for (int i = 0; i < n; i += 4)
            {
                v128 v = Arm.Neon.vmulq_f32(Arm.Neon.vld1q_f32(x + i), SigmoidNeon(Arm.Neon.vld1q_f32(gate + i)));
                Arm.Neon.vst1q_f32(x + i, v);
                peak = Arm.Neon.vmaxq_f32(peak, Arm.Neon.vabsq_f32(v));
            }
            FinishFakeA8Neon(x, n, Arm.Neon.vmaxvq_f32(peak));
        }

        static void PostNormAddNeon(float* residual, float* x, float* scale, float gate, int n)
        {
            v128 inverse = Arm.Neon.vdupq_n_f32(InverseRmsNeon(x, n));
            v128 one = Arm.Neon.vdupq_n_f32(1f);
            v128 g = Arm.Neon.vdupq_n_f32(gate);
            for (int i = 0; i < n; i += 4)
                Arm.Neon.vst1q_f32(residual + i, Arm.Neon.vfmaq_f32(Arm.Neon.vld1q_f32(residual + i), g,
                    Arm.Neon.vmulq_f32(Arm.Neon.vmulq_f32(Arm.Neon.vaddq_f32(one, Arm.Neon.vld1q_f32(scale + i)), Arm.Neon.vld1q_f32(x + i)), inverse)));
        }

        static void RopeNeon(float* x, int heads, int headDim, float* cos, float* sin)
        {
            int half = headDim / 2;
            for (int h = 0; h < heads; h++)
            {
                float* lo = x + h * headDim;
                float* hi = lo + half;
                for (int i = 0; i < half; i += 4)
                {
                    v128 c = Arm.Neon.vld1q_f32(cos + i);
                    v128 s = Arm.Neon.vld1q_f32(sin + i);
                    v128 a = Arm.Neon.vld1q_f32(lo + i);
                    v128 b = Arm.Neon.vld1q_f32(hi + i);
                    // FusedMultiplySubtract(a*c, b, s) is a*c - b*s; negation
                    // is exact and fma rounds once, so vfmaq with -s matches.
                    Arm.Neon.vst1q_f32(lo + i, Arm.Neon.vfmaq_f32(Arm.Neon.vmulq_f32(a, c), b, Arm.Neon.vnegq_f32(s)));
                    Arm.Neon.vst1q_f32(hi + i, Arm.Neon.vfmaq_f32(Arm.Neon.vmulq_f32(b, c), s, a));
                }
            }
        }

        static void Mix4Neon(float* s, int dim, float w0, float w1, float w2, float w3, float* destination)
        {
            v128 v0 = Arm.Neon.vdupq_n_f32(w0);
            v128 v1 = Arm.Neon.vdupq_n_f32(w1);
            v128 v2 = Arm.Neon.vdupq_n_f32(w2);
            v128 v3 = Arm.Neon.vdupq_n_f32(w3);
            for (int c = 0; c < dim; c += 4)
            {
                v128 acc = Arm.Neon.vmulq_f32(v0, Arm.Neon.vld1q_f32(s + c));
                acc = Arm.Neon.vfmaq_f32(acc, v1, Arm.Neon.vld1q_f32(s + dim + c));
                acc = Arm.Neon.vfmaq_f32(acc, v2, Arm.Neon.vld1q_f32(s + 2 * dim + c));
                acc = Arm.Neon.vfmaq_f32(acc, v3, Arm.Neon.vld1q_f32(s + 3 * dim + c));
                Arm.Neon.vst1q_f32(destination + c, acc);
            }
        }

        static void Combine4Neon(float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination)
        {
            v128 v0 = Arm.Neon.vdupq_n_f32(w0);
            v128 v1 = Arm.Neon.vdupq_n_f32(w1);
            v128 v2 = Arm.Neon.vdupq_n_f32(w2);
            v128 v3 = Arm.Neon.vdupq_n_f32(w3);
            v128 vp = Arm.Neon.vdupq_n_f32(post);
            for (int c = 0; c < dim; c += 4)
            {
                v128 acc = Arm.Neon.vmulq_f32(vp, Arm.Neon.vld1q_f32(y + c));
                acc = Arm.Neon.vfmaq_f32(acc, v0, Arm.Neon.vld1q_f32(s + c));
                acc = Arm.Neon.vfmaq_f32(acc, v1, Arm.Neon.vld1q_f32(s + dim + c));
                acc = Arm.Neon.vfmaq_f32(acc, v2, Arm.Neon.vld1q_f32(s + 2 * dim + c));
                acc = Arm.Neon.vfmaq_f32(acc, v3, Arm.Neon.vld1q_f32(s + 3 * dim + c));
                Arm.Neon.vst1q_f32(destination + c, acc);
            }
        }

        static void CombineDiff4Neon(float* s, int dim, float w0, float w1, float w2, float w3, float* x, float* u, float post, float* destination)
        {
            v128 v0 = Arm.Neon.vdupq_n_f32(w0);
            v128 v1 = Arm.Neon.vdupq_n_f32(w1);
            v128 v2 = Arm.Neon.vdupq_n_f32(w2);
            v128 v3 = Arm.Neon.vdupq_n_f32(w3);
            v128 vp = Arm.Neon.vdupq_n_f32(post);
            for (int c = 0; c < dim; c += 4)
            {
                v128 acc = Arm.Neon.vmulq_f32(vp, Arm.Neon.vsubq_f32(Arm.Neon.vld1q_f32(x + c), Arm.Neon.vld1q_f32(u + c)));
                acc = Arm.Neon.vfmaq_f32(acc, v0, Arm.Neon.vld1q_f32(s + c));
                acc = Arm.Neon.vfmaq_f32(acc, v1, Arm.Neon.vld1q_f32(s + dim + c));
                acc = Arm.Neon.vfmaq_f32(acc, v2, Arm.Neon.vld1q_f32(s + 2 * dim + c));
                acc = Arm.Neon.vfmaq_f32(acc, v3, Arm.Neon.vld1q_f32(s + 3 * dim + c));
                Arm.Neon.vst1q_f32(destination + c, acc);
            }
        }

        static void Taps3Neon(float* taps, int width, float* current, float* destination)
        {
            for (int c = 0; c < width; c += 4)
            {
                v128 acc = Arm.Neon.vmulq_f32(Arm.Neon.vld1q_f32(taps + c), Arm.Neon.vld1q_f32(current + c));
                acc = Arm.Neon.vfmaq_f32(acc, Arm.Neon.vld1q_f32(taps + width + c), Arm.Neon.vld1q_f32(current + c - width));
                acc = Arm.Neon.vfmaq_f32(acc, Arm.Neon.vld1q_f32(taps + 2 * width + c), Arm.Neon.vld1q_f32(current + c - 2 * width));
                Arm.Neon.vst1q_f32(destination + c, acc);
            }
        }

        static void MultiplyAccumulateNeon(float* destination, float* a, float* b, int n)
        {
            for (int i = 0; i < n; i += 4)
                Arm.Neon.vst1q_f32(destination + i, Arm.Neon.vfmaq_f32(Arm.Neon.vld1q_f32(destination + i), Arm.Neon.vld1q_f32(a + i), Arm.Neon.vld1q_f32(b + i)));
        }

        static void ScaleAddNeon(float* destination, float scale, float* x, float xScale, int n)
        {
            v128 s = Arm.Neon.vdupq_n_f32(scale);
            v128 xs = Arm.Neon.vdupq_n_f32(xScale);
            for (int i = 0; i < n; i += 4)
                Arm.Neon.vst1q_f32(destination + i, Arm.Neon.vfmaq_f32(Arm.Neon.vmulq_f32(s, Arm.Neon.vld1q_f32(destination + i)), xs, Arm.Neon.vld1q_f32(x + i)));
        }

        static void AddScaledNeon(float* destination, float* x, float scale, int n)
        {
            v128 s = Arm.Neon.vdupq_n_f32(scale);
            for (int i = 0; i < n; i += 4)
                Arm.Neon.vst1q_f32(destination + i, Arm.Neon.vfmaq_f32(Arm.Neon.vld1q_f32(destination + i), s, Arm.Neon.vld1q_f32(x + i)));
        }

        // Mirrors Neon.QuantizeInt8: truncate (vcvt, same as
        // ConvertToInt32RoundToZero), two saturating narrows, low 4 bytes.
        // The high half of the recombined vector is don't-care: only the low
        // 32 bits are stored, so it is filled with the low half again.
        static float QuantizeInt8Neon(float* x, int n, sbyte* destination)
        {
            float peak = AbsMaxNeon(x, n);
            float scale = peak > 0f ? peak / 127f : 1f;
            v128 inv = Arm.Neon.vdupq_n_f32(1f / scale);
            sbyte* spill = stackalloc sbyte[8];
            for (int i = 0; i < n; i += 4)
            {
                v128 wide = Arm.Neon.vcvtq_s32_f32(Arm.Neon.vmulq_f32(Arm.Neon.vld1q_f32(x + i), inv));
                v64 narrow = Arm.Neon.vqmovn_s32(wide);
                v64 bytes = Arm.Neon.vqmovn_s16(Arm.Neon.vcombine_s16(narrow, narrow));
                Arm.Neon.vst1_s8(spill, bytes);
                *(int*)(destination + i) = *(int*)spill;
            }
            return scale;
        }

        // Mirrors Quant.FakeA8: AbsMax + RoundTrip.
        static float FakeA8Neon(float* x, int n)
        {
            float peak = AbsMaxNeon(x, n);
            float scale = peak > 0f ? peak / 127f : 1f;
            RoundTripNeon(x, n, 1f / scale, scale);
            return scale;
        }

        // Mirrors Neon.Sinkhorn4x4.
        static void SinkhornNeon(float* logits, float* result)
        {
            float* centered = stackalloc float[4];
            float* sums = stackalloc float[4];
            for (int r = 0; r < 4; r++)
            {
                v128 row = Arm.Neon.vld1q_f32(logits + r * 4);
                v128 shift = Arm.Neon.vdupq_n_f32(Arm.Neon.vmaxvq_f32(row));
                Arm.Neon.vst1q_f32(centered, Arm.Neon.vsubq_f32(row, shift));
                for (int c = 0; c < 4; c++) result[r * 4 + c] = MathF.Exp(centered[c]);
            }
            for (int iteration = 0; iteration < 20; iteration++)
            {
                NormRowNeon(result);
                NormRowNeon(result + 4);
                NormRowNeon(result + 8);
                NormRowNeon(result + 12);
                v128 a0 = Arm.Neon.vld1q_f32(result);
                v128 a1 = Arm.Neon.vld1q_f32(result + 4);
                v128 a2 = Arm.Neon.vld1q_f32(result + 8);
                v128 a3 = Arm.Neon.vld1q_f32(result + 12);
                Arm.Neon.vst1q_f32(sums, Arm.Neon.vaddq_f32(Arm.Neon.vaddq_f32(Arm.Neon.vaddq_f32(a0, a1), a2), a3));
                if (sums[0] > 0f && sums[1] > 0f && sums[2] > 0f && sums[3] > 0f)
                {
                    v128 total = Arm.Neon.vld1q_f32(sums);
                    Arm.Neon.vst1q_f32(result, Arm.Neon.vdivq_f32(a0, total));
                    Arm.Neon.vst1q_f32(result + 4, Arm.Neon.vdivq_f32(a1, total));
                    Arm.Neon.vst1q_f32(result + 8, Arm.Neon.vdivq_f32(a2, total));
                    Arm.Neon.vst1q_f32(result + 12, Arm.Neon.vdivq_f32(a3, total));
                }
                else
                {
                    for (int c = 0; c < 4; c++)
                    {
                        float sum = result[c] + result[4 + c] + result[8 + c] + result[12 + c];
                        if (sum > 0f) for (int rr = 0; rr < 4; rr++) result[rr * 4 + c] /= sum;
                    }
                }
            }
        }

        static void NormRowNeon(float* row)
        {
            v128 v = Arm.Neon.vld1q_f32(row);
            float* lanes = stackalloc float[4];
            Arm.Neon.vst1q_f32(lanes, v);
            Arm.Neon.vst1q_f32(row, Arm.Neon.vdivq_f32(v, Arm.Neon.vdupq_n_f32(lanes[0] + lanes[1] + lanes[2] + lanes[3])));
        }

        // ---- scalar lanes (mirror Scalar.cs; also the no-SIMD fallback) ----

        const float RoundBias = 12582912f; // 1.5 * (1 << 23)

        static float RoundEven(float x)
        {
            return (x + RoundBias) - RoundBias;
        }

        static float AbsMaxScalar(float* x, int n)
        {
            float peak = 0f;
            for (int i = 0; i < n; i++) peak = MathF.Max(peak, MathF.Abs(x[i]));
            return peak;
        }

        static void RoundTripScalar(float* x, int n, float inverse, float scale)
        {
            for (int i = 0; i < n; i++) x[i] = MathF.Min(MathF.Max(RoundEven(x[i] * inverse), -128f), 127f) * scale;
        }

        static float SumOfSquaresScalar(float* x, int n)
        {
            float sum = 0f;
            for (int i = 0; i < n; i++) sum += x[i] * x[i];
            return sum;
        }

        static float InverseRmsScalar(float* x, int n)
        {
            return 1f / MathF.Sqrt(SumOfSquaresScalar(x, n) / n + RmsEpsilon);
        }

        static float SigmoidScalar(float x)
        {
            return 1f / (1f + MathF.Exp(-x));
        }

        static float DotScalar(float* a, float* b, int n)
        {
            float sum = 0f;
            for (int i = 0; i < n; i++) sum += a[i] * b[i];
            return sum;
        }

        static void RmsUnitScalar(float* x, int n, float* destination)
        {
            float inverse = InverseRmsScalar(x, n);
            for (int i = 0; i < n; i++) destination[i] = x[i] * inverse;
        }

        static void ZeroCenteredRmsScalar(float* x, float* scale, int n, float* destination)
        {
            float inverse = InverseRmsScalar(x, n);
            for (int i = 0; i < n; i++) destination[i] = (1f + scale[i]) * x[i] * inverse;
        }

        static void RmsFakeA8Scalar(float* x, int n, float* destination)
        {
            float inverse = InverseRmsScalar(x, n);
            float peak = 0f;
            for (int i = 0; i < n; i++)
            {
                destination[i] = x[i] * inverse;
                peak = MathF.Max(peak, MathF.Abs(destination[i]));
            }
            float scale = peak > 0f ? peak / 127f : 1f;
            float round = 1f / scale;
            for (int i = 0; i < n; i++) destination[i] = MathF.Min(MathF.Max(RoundEven(destination[i] * round), -128f), 127f) * scale;
        }

        static void CenteredRmsFakeA8Scalar(float* x, float* scale, int n, float* destination)
        {
            float inverse = InverseRmsScalar(x, n);
            float peak = 0f;
            for (int i = 0; i < n; i++)
            {
                destination[i] = (1f + scale[i]) * x[i] * inverse;
                peak = MathF.Max(peak, MathF.Abs(destination[i]));
            }
            float factor = peak > 0f ? peak / 127f : 1f;
            float round = 1f / factor;
            for (int i = 0; i < n; i++) destination[i] = MathF.Min(MathF.Max(RoundEven(destination[i] * round), -128f), 127f) * factor;
        }

        static void GateFakeA8Scalar(float* x, float* gate, int n)
        {
            float peak = 0f;
            for (int i = 0; i < n; i++)
            {
                x[i] *= SigmoidScalar(gate[i]);
                peak = MathF.Max(peak, MathF.Abs(x[i]));
            }
            float scale = peak > 0f ? peak / 127f : 1f;
            float round = 1f / scale;
            for (int i = 0; i < n; i++) x[i] = MathF.Min(MathF.Max(RoundEven(x[i] * round), -128f), 127f) * scale;
        }

        static void PostNormAddScalar(float* residual, float* x, float* scale, float gate, int n)
        {
            float inverse = InverseRmsScalar(x, n);
            for (int i = 0; i < n; i++) residual[i] += gate * ((1f + scale[i]) * x[i] * inverse);
        }

        static void RopeScalar(float* x, int heads, int headDim, float* cos, float* sin)
        {
            int half = headDim / 2;
            for (int h = 0; h < heads; h++)
            {
                float* lo = x + h * headDim;
                float* hi = lo + half;
                for (int i = 0; i < half; i++)
                {
                    float a = lo[i];
                    float b = hi[i];
                    lo[i] = a * cos[i] - b * sin[i];
                    hi[i] = b * cos[i] + a * sin[i];
                }
            }
        }

        static void Mix4Scalar(float* s, int dim, float w0, float w1, float w2, float w3, float* destination)
        {
            for (int c = 0; c < dim; c++) destination[c] = w0 * s[c] + w1 * s[dim + c] + w2 * s[2 * dim + c] + w3 * s[3 * dim + c];
        }

        static void Combine4Scalar(float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination)
        {
            for (int c = 0; c < dim; c++) destination[c] = post * y[c] + w0 * s[c] + w1 * s[dim + c] + w2 * s[2 * dim + c] + w3 * s[3 * dim + c];
        }

        static void CombineDiff4Scalar(float* s, int dim, float w0, float w1, float w2, float w3, float* x, float* u, float post, float* destination)
        {
            for (int c = 0; c < dim; c++) destination[c] = post * (x[c] - u[c]) + w0 * s[c] + w1 * s[dim + c] + w2 * s[2 * dim + c] + w3 * s[3 * dim + c];
        }

        static void Taps3Scalar(float* taps, int width, float* current, float* destination)
        {
            for (int c = 0; c < width; c++) destination[c] = taps[c] * current[c] + taps[width + c] * current[c - width] + taps[2 * width + c] * current[c - 2 * width];
        }

        static void MultiplyAccumulateScalar(float* destination, float* a, float* b, int n)
        {
            for (int i = 0; i < n; i++) destination[i] += a[i] * b[i];
        }

        static void ScaleAddScalar(float* destination, float scale, float* x, float xScale, int n)
        {
            for (int i = 0; i < n; i++) destination[i] = scale * destination[i] + xScale * x[i];
        }

        static void AddScaledScalar(float* destination, float* x, float scale, int n)
        {
            for (int i = 0; i < n; i++) destination[i] += scale * x[i];
        }

        static float QuantizeInt8Scalar(float* x, int n, sbyte* destination)
        {
            float peak = AbsMaxScalar(x, n);
            float scale = peak > 0f ? peak / 127f : 1f;
            float inverse = 1f / scale;
            for (int i = 0; i < n; i++) destination[i] = (sbyte)MathF.Min(MathF.Max(RoundEven(x[i] * inverse), -128f), 127f);
            return scale;
        }

        static float FakeA8Scalar(float* x, int n)
        {
            float peak = AbsMaxScalar(x, n);
            float scale = peak > 0f ? peak / 127f : 1f;
            RoundTripScalar(x, n, 1f / scale, scale);
            return scale;
        }

        static void SinkhornScalar(float* logits, float* result)
        {
            for (int r = 0; r < 4; r++)
            {
                float peak = MathF.Max(MathF.Max(logits[r * 4], logits[r * 4 + 1]), MathF.Max(logits[r * 4 + 2], logits[r * 4 + 3]));
                for (int c = 0; c < 4; c++) result[r * 4 + c] = MathF.Exp(logits[r * 4 + c] - peak);
            }
            for (int iteration = 0; iteration < 20; iteration++)
            {
                for (int r = 0; r < 4; r++)
                {
                    float sum = result[r * 4] + result[r * 4 + 1] + result[r * 4 + 2] + result[r * 4 + 3];
                    for (int c = 0; c < 4; c++) result[r * 4 + c] /= sum;
                }
                for (int c = 0; c < 4; c++)
                {
                    float sum = result[c] + result[4 + c] + result[8 + c] + result[12 + c];
                    if (sum > 0f) for (int r = 0; r < 4; r++) result[r * 4 + c] /= sum;
                }
            }
        }

        // ---- table delegates (Cdecl ABI, one per §2.1 slot, Softmax is N31) ----

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void RmsFakeA8Fn(float* x, int n, float* destination);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void CenteredRmsFakeA8Fn(float* x, float* scale, int n, float* destination);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void GateFakeA8Fn(float* x, float* gate, int n);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void PostNormAddFn(float* residual, float* x, float* scale, float gate, int n);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void CombineDiff4Fn(float* s, int dim, float w0, float w1, float w2, float w3, float* x, float* u, float post, float* destination);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void Mix4Fn(float* s, int dim, float w0, float w1, float w2, float w3, float* destination);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void Combine4Fn(float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void Taps3Fn(float* taps, int width, float* current, float* destination);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void RopeFn(float* x, int heads, int headDim, float* cos, float* sin);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ZeroCenteredRmsFn(float* x, float* scale, int n, float* destination);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void RmsUnitFn(float* x, int n, float* destination);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate float DotFn(float* a, float* b, int n);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void AddScaledFn(float* destination, float* x, float scale, int n);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ScaleAddFn(float* destination, float scale, float* x, float xScale, int n);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void MultiplyAccumulateFn(float* destination, float* a, float* b, int n);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate float QuantizeInt8Fn(float* x, int n, sbyte* destination);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate float FakeA8Fn(float* x, int n);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void Sinkhorn4x4Fn(float* logits, float* result);

        public static bool IsInstalled =>
            Kernels.Active.RmsFakeA8 != 0 && Kernels.Active.CenteredRmsFakeA8 != 0 &&
            Kernels.Active.GateFakeA8 != 0 && Kernels.Active.PostNormAdd != 0 &&
            Kernels.Active.CombineDiff4 != 0 && Kernels.Active.Mix4 != 0 &&
            Kernels.Active.Combine4 != 0 && Kernels.Active.Taps3 != 0 &&
            Kernels.Active.Rope != 0 && Kernels.Active.ZeroCenteredRms != 0 &&
            Kernels.Active.RmsUnit != 0 && Kernels.Active.Dot != 0 &&
            Kernels.Active.AddScaled != 0 && Kernels.Active.ScaleAdd != 0 &&
            Kernels.Active.MultiplyAccumulate != 0 && Kernels.Active.QuantizeInt8 != 0 &&
            Kernels.Active.FakeA8 != 0 && Kernels.Active.Sinkhorn4x4 != 0;

        // Idempotent: fills every row-op slot that is still zero, never
        // overwrites an explicit or foreign provider's entry (plan §2
        // selection order). Throws with the op name when Burst refuses to
        // compile an entry; slots filled before the failure stay installed.
        public static void Install()
        {
            KernelTable table = Kernels.Active;
            if (table.RmsFakeA8 == 0) table.RmsFakeA8 = Compile<RmsFakeA8Fn>(InvokeRmsFakeA8, "RmsFakeA8");
            if (table.CenteredRmsFakeA8 == 0) table.CenteredRmsFakeA8 = Compile<CenteredRmsFakeA8Fn>(InvokeCenteredRmsFakeA8, "CenteredRmsFakeA8");
            if (table.GateFakeA8 == 0) table.GateFakeA8 = Compile<GateFakeA8Fn>(InvokeGateFakeA8, "GateFakeA8");
            if (table.PostNormAdd == 0) table.PostNormAdd = Compile<PostNormAddFn>(InvokePostNormAdd, "PostNormAdd");
            if (table.CombineDiff4 == 0) table.CombineDiff4 = Compile<CombineDiff4Fn>(InvokeCombineDiff4, "CombineDiff4");
            if (table.Mix4 == 0) table.Mix4 = Compile<Mix4Fn>(InvokeMix4, "Mix4");
            if (table.Combine4 == 0) table.Combine4 = Compile<Combine4Fn>(InvokeCombine4, "Combine4");
            if (table.Taps3 == 0) table.Taps3 = Compile<Taps3Fn>(InvokeTaps3, "Taps3");
            if (table.Rope == 0) table.Rope = Compile<RopeFn>(InvokeRope, "Rope");
            if (table.ZeroCenteredRms == 0) table.ZeroCenteredRms = Compile<ZeroCenteredRmsFn>(InvokeZeroCenteredRms, "ZeroCenteredRms");
            if (table.RmsUnit == 0) table.RmsUnit = Compile<RmsUnitFn>(InvokeRmsUnit, "RmsUnit");
            if (table.Dot == 0) table.Dot = Compile<DotFn>(InvokeDot, "Dot");
            if (table.AddScaled == 0) table.AddScaled = Compile<AddScaledFn>(InvokeAddScaled, "AddScaled");
            if (table.ScaleAdd == 0) table.ScaleAdd = Compile<ScaleAddFn>(InvokeScaleAdd, "ScaleAdd");
            if (table.MultiplyAccumulate == 0) table.MultiplyAccumulate = Compile<MultiplyAccumulateFn>(InvokeMultiplyAccumulate, "MultiplyAccumulate");
            if (table.QuantizeInt8 == 0) table.QuantizeInt8 = Compile<QuantizeInt8Fn>(InvokeQuantizeInt8, "QuantizeInt8");
            if (table.FakeA8 == 0) table.FakeA8 = Compile<FakeA8Fn>(InvokeFakeA8, "FakeA8");
            if (table.Sinkhorn4x4 == 0) table.Sinkhorn4x4 = Compile<Sinkhorn4x4Fn>(InvokeSinkhorn4x4, "Sinkhorn4x4");
            Kernels.Install(table, Kernels.IsInstalled ? Kernels.Provider + "+rowops" : "burst-rowops");
        }

        static nint Compile<T>(T entry, string name) where T : Delegate
        {
            FunctionPointer<T> compiled = BurstCompiler.CompileFunctionPointer<T>(entry);
            if (!compiled.IsCreated)
                throw new InvalidOperationException(
                    "N38: Burst refused to compile BurstRowOps." + name + ". " +
                    "The kernel must live in package source (H6); " +
                    "a precompiled DLL entry cannot be compiled.");
            return (nint)compiled.Value;
        }
    }
}
