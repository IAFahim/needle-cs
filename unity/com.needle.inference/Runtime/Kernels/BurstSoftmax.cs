using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;

namespace Needle.BurstKernels
{
    [BurstCompile]
    public static unsafe class BurstSoftmax
    {
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void Invoke(float* s, int n)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { SoftmaxX86(s, n); return; }
            if (Arm.Neon.IsNeonSupported) { SoftmaxNeon(s, n); return; }
            SoftmaxScalar(s, n);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static int Lane()
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) return 2;
            if (Arm.Neon.IsNeonSupported) return 1;
            return 0;
        }

        static v256 ExpX86(v256 x)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
            return default;
        }

        static float Max8X86(v256 v)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v128 halves = X86.Sse.max_ps(
                    X86.Avx.mm256_castps256_ps128(v),
                    X86.Avx.mm256_extractf128_ps(v, 1));
                v128 pairs = X86.Sse.max_ps(halves, X86.Sse.shuffle_ps(halves, halves, 0x4E));
                v128 quad = X86.Sse.max_ps(pairs, X86.Sse.shuffle_ps(pairs, pairs, 0xB1));
                return X86.Sse.cvtss_f32(quad);
            }
            return 0f;
        }

        static float Sum128X86(v128 v)
        {
            if (X86.Sse3.IsSse3Supported)
            {
                v128 t = X86.Sse3.hadd_ps(v, v);
                t = X86.Sse3.hadd_ps(t, t);
                return X86.Sse.cvtss_f32(t);
            }
            return 0f;
        }

        static float Sum256X86(v256 v)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                return Sum128X86(X86.Avx.mm256_castps256_ps128(v))
                    + Sum128X86(X86.Avx.mm256_extractf128_ps(v, 1));
            }
            return 0f;
        }

        static void SoftmaxX86(float* s, int n)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
        }

        static v128 ExpNeon(v128 x)
        {
            if (!Arm.Neon.IsNeonSupported) return x;
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
            v128 bits = Arm.Neon.vshlq_n_s32(
                Arm.Neon.vaddq_s32(
                    Arm.Neon.vcvtnq_s32_f32(n),
                    Arm.Neon.vdupq_n_s32(127)), 23);
            uint* spill = stackalloc uint[4];
            Arm.Neon.vst1q_u32(spill, bits);
            v128 scale = Arm.Neon.vld1q_f32((float*)spill);
            return Arm.Neon.vmulq_f32(y, scale);
        }

        static float HorizontalSumNeon(v128 v)
        {
            if (!Arm.Neon.IsNeonSupported) return 0f;
            v128 pairs = Arm.Neon.vpaddq_f32(v, v);
            v128 quad = Arm.Neon.vpaddq_f32(pairs, pairs);
            return Arm.Neon.vgetq_lane_f32(quad, 0);
        }

        static void SoftmaxNeon(float* s, int n)
        {
            if (!Arm.Neon.IsNeonSupported) return;
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
