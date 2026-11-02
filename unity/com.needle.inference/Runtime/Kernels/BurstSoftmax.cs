// N31: Burst Softmax kernel, the first §2.1 entry end to end.
//
// Ports Ops.Softmax 1:1 to Unity.Burst.Intrinsics: the x86 path mirrors the
// net10 AVX2+FMA lane (same constants, same FMA order, same max/sum
// reductions), the Neon path mirrors Neon.Softmax (same polynomial, same
// pairwise order). Anything else falls back to a scalar loop so WebGL-era
// targets still run.
//
// Burst rules (plan §2.1 cookbook) this file follows:
//   - FloatMode.Strict on the entry point, never Fast: no reassociation.
//   - Explicit FMA intrinsics (mm256_fmadd_ps / mm256_fnmadd_ps, vfmaq_f32);
//     no reliance on contraction.
//   - No [NoAlias]: Softmax is in-place (source and destination are the
//     same pointer), so an aliasing promise would be false.
//   - Blittable arguments only (float*, int); no allocation, no static
//     mutable state; safe from any thread (pool workers call via the table).
//
// H7 note: the x86 and Neon lanes use the same instruction order as the
// net10 kernels, so fixtures with finite inputs in the clamp range compare
// bitwise (Neon.HorizontalSum order and the BCL Vector256.Sum hadd order
// are reproduced lane for lane). Known 1-ulp-scale risks, all outside the
// conformance fixtures: the scalar tail uses MathF.Exp exactly like net10
// (Burst lowers System.MathF.Exp); a Flush-To-Zero difference on
// denormal exp(-87.3)-scale outputs cannot be ruled out from docs alone.
// If the bitwise probe in Tests/ ever diverges, this kernel still meets
// the N13 parity bound (cosine >= 0.99995, same argmax) by construction.
//
// H8 note: this file needs no AVX-VNNI and no AVX-512. Verified against
// the Unity.Burst assembly API: X86 nested types are exactly Avx, Avx2,
// Bmi1, Bmi2, F16C, Fma, Popcnt, Sse, Sse2, Sse3, Sse4_1, Sse4_2, Ssse3.
// Integer matmul ports (N32+) must use the Avx2Dot sequence per the plan.
//
// C# 9 safe: block namespace, no target-typed new, no static lambdas.
using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;

namespace Needle.BurstKernels
{
    public static unsafe class BurstSoftmax
    {
        // Entry point installed into KernelTable.Softmax:
        // void(float* s, int n), Cdecl, in-place normalize.
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void Invoke(float* s, int n)
        {
            // Branches fold at Burst compile time: exactly one lane survives
            // per target (x64 with AVX2+FMA, arm64 Neon, scalar elsewhere).
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { SoftmaxX86(s, n); return; }
            if (Arm.Neon.IsNeonSupported) { SoftmaxNeon(s, n); return; }
            SoftmaxScalar(s, n);
        }

        // Compiled-lane reporter: 2 = x86 AVX2+FMA, 1 = arm Neon, 0 = scalar.
        // Managed evaluation of the IsSupported flags below always reads
        // False (probed on host CoreCLR: real CPUID is a Burst-compile-time
        // fold, not a managed query), so only compiled code can report which
        // lane survived. The installer exposes this as CompiledLane(); never
        // trust a managed IsSupported read for lane questions.
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static int Lane()
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) return 2;
            if (Arm.Neon.IsNeonSupported) return 1;
            return 0;
        }

        // Mirrors Ops.Exp bit for bit: clamp, round, two fnmadd reductions,
        // six-term FMA polynomial, 2^n scale from integer bit tricks.
        static v256 ExpX86(v256 x)
        {
            x = X86.Avx.mm256_min_ps(
                X86.Avx.mm256_max_ps(x, X86.Avx.mm256_set1_ps(-87.3f)),
                X86.Avx.mm256_set1_ps(88.7f));
            // .NET Avx.RoundToNearestInteger is VROUNDPS with
            // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC = 8.
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
            // _mm256_cvtps_epi32 is VCVTPS2DQ (round to nearest even, same as
            // the BCL ConvertToVector256Int32); broadcast 127 via the v256
            // Int32 lane ctor, matching Vector256.Create(127).
            v256 scaled = X86.Avx.mm256_castsi256_ps(
                X86.Avx2.mm256_slli_epi32(
                    X86.Avx2.mm256_add_epi32(X86.Avx.mm256_cvtps_epi32(n), new v256(127)), 23));
            return X86.Avx.mm256_mul_ps(y, scaled);
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

        // Mirrors the BCL Vector256.Sum(float) order per the .NET 10
        // reference source (N33 cross-check): Sum128(lower) + Sum128(upper),
        // where Sum128 is two hadd passes, lane 0 out. NOT add(low,high)
        // followed by hadd+hadd: the two groupings differ by 1 ulp
        // (hardware probe: BCL=33554436 vs hadd-order=33554432).
        static float Sum128X86(v128 v)
        {
            v128 t = X86.Sse3.hadd_ps(v, v);
            t = X86.Sse3.hadd_ps(t, t);
            return X86.Sse.cvtss_f32(t);
        }

        static float Sum256X86(v256 v)
        {
            return Sum128X86(X86.Avx.mm256_castps256_ps128(v))
                + Sum128X86(X86.Avx.mm256_extractf128_ps(v, 1));
        }

        // Mirrors Ops.Softmax exactly: vector peak, scalar tail peak,
        // vector exp-sum, scalar tail exp-sum, broadcast normalize.
        static void SoftmaxX86(float* s, int n)
        {
            v256 peakVector = X86.Avx.mm256_set1_ps(float.NegativeInfinity);
            int peakEnd = n & ~7;
            int p = 0;
            for (; p < peakEnd; p += 8)
                peakVector = X86.Avx.mm256_max_ps(peakVector, X86.Avx.mm256_loadu_ps(s + p));
            float peak = Max8X86(peakVector);
            for (; p < n; p++) peak = MathF.Max(peak, s[p]);
            v256 shift = X86.Avx.mm256_set1_ps(peak);
            v256 sum = X86.Avx.mm256_set1_ps(0f);
            int vectorEnd = n & ~7;
            int i = 0;
            for (; i < vectorEnd; i += 8)
            {
                v256 e = ExpX86(X86.Avx.mm256_sub_ps(X86.Avx.mm256_loadu_ps(s + i), shift));
                X86.Avx.mm256_storeu_ps(s + i, e);
                sum = X86.Avx.mm256_add_ps(sum, e);
            }
            float total = Sum256X86(sum);
            for (; i < n; i++)
            {
                s[i] = MathF.Exp(s[i] - peak);
                total += s[i];
            }
            v256 inverse = X86.Avx.mm256_set1_ps(1f / total);
            for (i = 0; i < vectorEnd; i += 8)
                X86.Avx.mm256_storeu_ps(s + i, X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(s + i), inverse));
            for (; i < n; i++) s[i] /= total;
        }

        // Mirrors Neon.Exp: same clamp, FRINTN round, same polynomial with
        // explicit vfmaq_f32. The two FusedMultiplySubtract steps become
        // vfmaq with negated constants: negation is exact and fma rounds
        // once, so x + n*(-C) is bitwise identical to x - n*C.
        // Operand order note: Neon.Exp writes FusedMultiplyAdd(c, r, p) =
        // c + r*p; this file writes vfmaq(p, c, r) = p + c*r (likewise the
        // y-line). IEEE add/mul commute bitwise on finite inputs, so the
        // two forms are the same instruction sequence with the same
        // rounding — ascending Horner either way, no transposition.
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
            // Neon has no bit-reinterpret intrinsic, so the (n+127)<<23
            // exponent moves through a 16-byte stack spill: store as u32,
            // reload as f32. Same bits as AdvSimd .AsSingle().
            // Signed convert/add/shift (mirrors ConvertToInt32RoundToEven):
            // the unsigned convert saturates negatives to 0, which would
            // pin the scale to 1.0 for every softmax input (s-peak <= 0).
            v128 bits = Arm.Neon.vshlq_n_s32(
                Arm.Neon.vaddq_s32(
                    Arm.Neon.vcvtnq_s32_f32(n),
                    Arm.Neon.vdupq_n_s32(127)), 23);
            uint* spill = stackalloc uint[4];
            Arm.Neon.vst1q_u32(spill, bits);
            v128 scale = Arm.Neon.vld1q_f32((float*)spill);
            return Arm.Neon.vmulq_f32(y, scale);
        }

        // Mirrors Neon.HorizontalSum: two pairwise adds, lane 0 out.
        // vpaddq_f32(v,v) = [l0+l1, u0+u1, ...] matches AddPairwise lane for
        // lane, so the second fold lands on the same sum in the same order.
        static float HorizontalSumNeon(v128 v)
        {
            v128 pairs = Arm.Neon.vpaddq_f32(v, v);
            v128 quad = Arm.Neon.vpaddq_f32(pairs, pairs);
            return Arm.Neon.vgetq_lane_f32(quad, 0);
        }

        // Mirrors Neon.Softmax exactly.
        static void SoftmaxNeon(float* s, int n)
        {
            float peak = float.NegativeInfinity;
            for (int i = 0; i < n; i++) peak = MathF.Max(peak, s[i]);
            v128 shift = Arm.Neon.vdupq_n_f32(peak);
            v128 sum = Arm.Neon.vdupq_n_f32(0f);
            int vectorEnd = n & ~3;
            int i2 = 0;
            for (; i2 < vectorEnd; i2 += 4)
            {
                v128 e = ExpNeon(Arm.Neon.vsubq_f32(Arm.Neon.vld1q_f32(s + i2), shift));
                Arm.Neon.vst1q_f32(s + i2, e);
                sum = Arm.Neon.vaddq_f32(sum, e);
            }
            float total = HorizontalSumNeon(sum);
            for (; i2 < n; i2++)
            {
                s[i2] = MathF.Exp(s[i2] - peak);
                total += s[i2];
            }
            v128 inverse = Arm.Neon.vdupq_n_f32(1f / total);
            for (i2 = 0; i2 < vectorEnd; i2 += 4)
                Arm.Neon.vst1q_f32(s + i2, Arm.Neon.vmulq_f32(Arm.Neon.vld1q_f32(s + i2), inverse));
            for (; i2 < n; i2++) s[i2] /= total;
        }

        // Scalar lane for targets with neither AVX2+FMA nor Neon (and the
        // shape every lane must agree with): mirrors Scalar.Softmax.
        static void SoftmaxScalar(float* s, int n)
        {
            float peak = float.NegativeInfinity;
            for (int i = 0; i < n; i++) peak = MathF.Max(peak, s[i]);
            float total = 0f;
            for (int i = 0; i < n; i++)
            {
                s[i] = MathF.Exp(s[i] - peak);
                total += s[i];
            }
            for (int i = 0; i < n; i++) s[i] /= total;
        }
    }
}
