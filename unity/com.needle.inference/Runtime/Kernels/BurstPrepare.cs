using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;

namespace Needle.BurstKernels
{
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct BurstActivations
    {
        public sbyte* Values;
        public float* Scales;
        public float* Offsets;
        public int Cols;
        public int Groups;
        public int Stride;
    }

    [BurstCompile]
    public static unsafe class BurstPrepare
    {
        const int GroupSize = 128;
        const float HadamardScale = 0.08838834764831845f;
        const float CodebookBias = 63f;
        const float RoundBias = 12582912f;

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokePrepare(float* x, int tokens, int cols, int bits, BurstActivations* a)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { PrepareX86(x, tokens, cols, bits, a); return; }
            if (Arm.Neon.IsNeonSupported) { PrepareNeon(x, tokens, cols, bits, a); return; }
            PrepareScalar(x, tokens, cols, bits, a);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static float InvokeQuantizeInt8(float* x, int n, sbyte* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { return QuantizeInt8X86(x, n, destination); }
            if (Arm.Neon.IsNeonSupported) { return QuantizeInt8Neon(x, n, destination); }
            return QuantizeInt8Scalar(x, n, destination);
        }

        static float AbsMaxX86(float* x, int n)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 sign = X86.Avx.mm256_set1_ps(-0.0f);
                v256 peak = X86.Avx.mm256_set1_ps(0f);
                int i = 0;
                for (; i + 8 <= n; i += 8)
                    peak = X86.Avx.mm256_max_ps(peak, X86.Avx.mm256_andnot_ps(sign, X86.Avx.mm256_loadu_ps(x + i)));
                v128 halves = X86.Sse.max_ps(X86.Avx.mm256_castps256_ps128(peak), X86.Avx.mm256_extractf128_ps(peak, 1));
                v128 pairs = X86.Sse.max_ps(halves, X86.Sse.shuffle_ps(halves, halves, 0x4E));
                v128 quad = X86.Sse.max_ps(pairs, X86.Sse.shuffle_ps(pairs, pairs, 0xB1));
                float best = X86.Sse.cvtss_f32(quad);
                for (; i < n; i++) best = MathF.Max(best, MathF.Abs(x[i]));
                return best;
            }
            return 0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static v256 Pack32X86(float* source, v256 inverse, v256 interleave)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 c0 = X86.Avx.mm256_cvtps_epi32(X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(source), inverse));
                v256 c1 = X86.Avx.mm256_cvtps_epi32(X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(source + 8), inverse));
                v256 c2 = X86.Avx.mm256_cvtps_epi32(X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(source + 16), inverse));
                v256 c3 = X86.Avx.mm256_cvtps_epi32(X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(source + 24), inverse));
                v256 p01 = X86.Avx2.mm256_packs_epi32(c0, c1);
                v256 p23 = X86.Avx2.mm256_packs_epi32(c2, c3);
                v256 packed = X86.Avx2.mm256_packs_epi16(p01, p23);
                return X86.Avx2.mm256_permutevar8x32_epi32(packed, interleave);
            }
            return default;
        }

        static float QuantizeInt8X86(float* x, int n, sbyte* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                float peak = AbsMaxX86(x, n);
                float scale = peak > 0f ? peak / 127f : 1f;
                float scalarInv = 1f / scale;
                v256 inverse = X86.Avx.mm256_set1_ps(scalarInv);
                v256 interleave = new v256(0, 4, 1, 5, 2, 6, 3, 7);
                int i = 0;
                for (; i + 32 <= n; i += 32)
                    X86.Avx.mm256_storeu_si256((v256*)(destination + i), Pack32X86(x + i, inverse, interleave));
                for (; i + 8 <= n; i += 8)
                {
                    v256 q = X86.Avx.mm256_cvtps_epi32(X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(x + i), inverse));
                    v128 packed = X86.Sse2.packs_epi32(
                        X86.Avx.mm256_castsi256_si128(q),
                        X86.Avx.mm256_extractf128_si256(q, 1));
                    packed = X86.Sse2.packs_epi16(packed, packed);
                    *(long*)(destination + i) = X86.Sse2.cvtsi128_si64(packed);
                }
                for (; i < n; i++) destination[i] = (sbyte)MathF.Min(MathF.Max(RoundEvenScalar(x[i] * scalarInv), -128f), 127f);
                return scale;
            }
            return 0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static v256 Intra8ScaleX86(v256 v, v256 pairSign, v256 quadSign, v256 halfSign, v256 scale)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v = X86.Fma.mm256_fmadd_ps(v, pairSign, X86.Avx.mm256_permute_ps(v, 0xB1));
                v = X86.Fma.mm256_fmadd_ps(v, quadSign, X86.Avx.mm256_permute_ps(v, 0x4E));
                v = X86.Fma.mm256_fmadd_ps(v, halfSign, X86.Avx.mm256_permute2f128_ps(v, v, 0x01));
                return X86.Avx.mm256_mul_ps(v, scale);
            }
            return default;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float HadamardPeakX86(float* source, float* p)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 pairSign = new v256(1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f);
                v256 quadSign = new v256(1f, 1f, -1f, -1f, 1f, 1f, -1f, -1f);
                v256 halfSign = new v256(1f, 1f, 1f, 1f, -1f, -1f, -1f, -1f);
                v256 scale = X86.Avx.mm256_set1_ps(HadamardScale);
                v256 signMask = X86.Avx.mm256_set1_ps(-0.0f);
                for (int i = 0; i < 32; i += 8)
                {
                    v256 a = X86.Avx.mm256_loadu_ps(source + i);
                    v256 b = X86.Avx.mm256_loadu_ps(source + i + 32);
                    v256 c = X86.Avx.mm256_loadu_ps(source + i + 64);
                    v256 d = X86.Avx.mm256_loadu_ps(source + i + 96);
                    v256 ab0 = X86.Avx.mm256_add_ps(a, c);
                    v256 ab1 = X86.Avx.mm256_sub_ps(a, c);
                    v256 cd0 = X86.Avx.mm256_add_ps(b, d);
                    v256 cd1 = X86.Avx.mm256_sub_ps(b, d);
                    X86.Avx.mm256_storeu_ps(p + i, X86.Avx.mm256_add_ps(ab0, cd0));
                    X86.Avx.mm256_storeu_ps(p + i + 32, X86.Avx.mm256_sub_ps(ab0, cd0));
                    X86.Avx.mm256_storeu_ps(p + i + 64, X86.Avx.mm256_add_ps(ab1, cd1));
                    X86.Avx.mm256_storeu_ps(p + i + 96, X86.Avx.mm256_sub_ps(ab1, cd1));
                }
                v256 peak = X86.Avx.mm256_set1_ps(0f);
                for (int b = 0; b < GroupSize; b += 32)
                {
                    v256 r0 = X86.Avx.mm256_loadu_ps(p + b);
                    v256 r1 = X86.Avx.mm256_loadu_ps(p + b + 8);
                    v256 r2 = X86.Avx.mm256_loadu_ps(p + b + 16);
                    v256 r3 = X86.Avx.mm256_loadu_ps(p + b + 24);
                    v256 p0 = X86.Avx.mm256_add_ps(r0, r2);
                    v256 p1 = X86.Avx.mm256_sub_ps(r0, r2);
                    v256 p2 = X86.Avx.mm256_add_ps(r1, r3);
                    v256 p3 = X86.Avx.mm256_sub_ps(r1, r3);
                    v256 v0 = Intra8ScaleX86(X86.Avx.mm256_add_ps(p0, p2), pairSign, quadSign, halfSign, scale);
                    v256 v1 = Intra8ScaleX86(X86.Avx.mm256_sub_ps(p0, p2), pairSign, quadSign, halfSign, scale);
                    v256 v2 = Intra8ScaleX86(X86.Avx.mm256_add_ps(p1, p3), pairSign, quadSign, halfSign, scale);
                    v256 v3 = Intra8ScaleX86(X86.Avx.mm256_sub_ps(p1, p3), pairSign, quadSign, halfSign, scale);
                    X86.Avx.mm256_storeu_ps(p + b, v0);
                    X86.Avx.mm256_storeu_ps(p + b + 8, v1);
                    X86.Avx.mm256_storeu_ps(p + b + 16, v2);
                    X86.Avx.mm256_storeu_ps(p + b + 24, v3);
                    v256 m01 = X86.Avx.mm256_max_ps(X86.Avx.mm256_andnot_ps(signMask, v0), X86.Avx.mm256_andnot_ps(signMask, v1));
                    v256 m23 = X86.Avx.mm256_max_ps(X86.Avx.mm256_andnot_ps(signMask, v2), X86.Avx.mm256_andnot_ps(signMask, v3));
                    peak = X86.Avx.mm256_max_ps(peak, X86.Avx.mm256_max_ps(m01, m23));
                }
                v128 halves = X86.Sse.max_ps(X86.Avx.mm256_castps256_ps128(peak), X86.Avx.mm256_extractf128_ps(peak, 1));
                v128 pairs = X86.Sse.max_ps(halves, X86.Sse.shuffle_ps(halves, halves, 0x4E));
                v128 quad = X86.Sse.max_ps(pairs, X86.Sse.shuffle_ps(pairs, pairs, 0xB1));
                return X86.Sse.cvtss_f32(quad);
            }
            return 0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void Planes2X86(v256 q0, v256 q1, v256 q2, v256 q3, v256 gather, v256 interleave, sbyte* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 s0 = X86.Avx2.mm256_shuffle_epi8(q0, gather);
                v256 s1 = X86.Avx2.mm256_shuffle_epi8(q1, gather);
                v256 s2 = X86.Avx2.mm256_shuffle_epi8(q2, gather);
                v256 s3 = X86.Avx2.mm256_shuffle_epi8(q3, gather);
                v256 low01 = X86.Avx2.mm256_unpacklo_epi32(s0, s1);
                v256 high01 = X86.Avx2.mm256_unpackhi_epi32(s0, s1);
                v256 low23 = X86.Avx2.mm256_unpacklo_epi32(s2, s3);
                v256 high23 = X86.Avx2.mm256_unpackhi_epi32(s2, s3);
                X86.Avx.mm256_storeu_si256((v256*)(destination), X86.Avx2.mm256_permutevar8x32_epi32(X86.Avx2.mm256_unpacklo_epi64(low01, low23), interleave));
                X86.Avx.mm256_storeu_si256((v256*)(destination + 32), X86.Avx2.mm256_permutevar8x32_epi32(X86.Avx2.mm256_unpackhi_epi64(low01, low23), interleave));
                X86.Avx.mm256_storeu_si256((v256*)(destination + 64), X86.Avx2.mm256_permutevar8x32_epi32(X86.Avx2.mm256_unpacklo_epi64(high01, high23), interleave));
                X86.Avx.mm256_storeu_si256((v256*)(destination + 96), X86.Avx2.mm256_permutevar8x32_epi32(X86.Avx2.mm256_unpackhi_epi64(high01, high23), interleave));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void Planes4X86(v256 q0, v256 q1, v256 q2, v256 q3, v256 gather, sbyte* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 s0 = X86.Avx2.mm256_shuffle_epi8(q0, gather);
                v256 s1 = X86.Avx2.mm256_shuffle_epi8(q1, gather);
                v256 s2 = X86.Avx2.mm256_shuffle_epi8(q2, gather);
                v256 s3 = X86.Avx2.mm256_shuffle_epi8(q3, gather);
                X86.Avx.mm256_storeu_si256((v256*)(destination), X86.Avx2.mm256_permute4x64_epi64(X86.Avx2.mm256_unpacklo_epi64(s0, s1), 0xD8));
                X86.Avx.mm256_storeu_si256((v256*)(destination + 32), X86.Avx2.mm256_permute4x64_epi64(X86.Avx2.mm256_unpackhi_epi64(s0, s1), 0xD8));
                X86.Avx.mm256_storeu_si256((v256*)(destination + 64), X86.Avx2.mm256_permute4x64_epi64(X86.Avx2.mm256_unpacklo_epi64(s2, s3), 0xD8));
                X86.Avx.mm256_storeu_si256((v256*)(destination + 96), X86.Avx2.mm256_permute4x64_epi64(X86.Avx2.mm256_unpackhi_epi64(s2, s3), 0xD8));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int Sum8IntX86(v256 v)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v128 s4 = X86.Sse2.add_epi32(
                    X86.Avx.mm256_castsi256_si128(v),
                    X86.Avx.mm256_extractf128_si256(v, 1));
                v128 s2 = X86.Sse2.add_epi32(s4, X86.Sse2.shuffle_epi32(s4, 0x4E));
                v128 s1 = X86.Sse2.add_epi32(s2, X86.Sse2.shuffle_epi32(s2, 0xB1));
                return X86.Sse2.cvtsi128_si32(s1);
            }
            return 0;
        }

        static void PrepareX86(float* x, int tokens, int cols, int bits, BurstActivations* a)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                float* rotated = stackalloc float[GroupSize];
                for (int t = 0; t < tokens; t++)
                    for (int g = 0; g < a->Groups; g++)
                    {
                        int slot = t * a->Stride + g;
                        float peak = HadamardPeakX86(x + (long)t * cols + g * GroupSize, rotated);
                        float scale = peak > 0f ? peak / 127f : 1f;
                        a->Scales[slot] = scale;
                        v256 inverse = X86.Avx.mm256_set1_ps(1f / scale);
                        v256 interleave = new v256(0, 4, 1, 5, 2, 6, 3, 7);
                        v256 q0 = Pack32X86(rotated, inverse, interleave);
                        v256 q1 = Pack32X86(rotated + 32, inverse, interleave);
                        v256 q2 = Pack32X86(rotated + 64, inverse, interleave);
                        v256 q3 = Pack32X86(rotated + 96, inverse, interleave);
                        v256 onesByte = X86.Avx.mm256_set1_epi8((byte)1);
                        v256 onesShort = X86.Avx.mm256_set1_epi16((short)1);
                        v256 sum = X86.Avx2.mm256_add_epi32(
                            X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(onesByte, q0), onesShort),
                            X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(onesByte, q1), onesShort));
                        sum = X86.Avx2.mm256_add_epi32(sum, X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(onesByte, q2), onesShort));
                        sum = X86.Avx2.mm256_add_epi32(sum, X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(onesByte, q3), onesShort));
                        a->Offsets[slot] = CodebookBias * (float)Sum8IntX86(sum);
                        sbyte* destination = a->Values + ((long)t * a->Groups + g) * GroupSize;
                        if (bits == 2)
                        {
                            v256 gather2 = new v256(
                                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                            Planes2X86(q0, q1, q2, q3, gather2, interleave, destination);
                        }
                        else
                        {
                            v256 gather4 = new v256(
                                0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
                                0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
                            Planes4X86(q0, q1, q2, q3, gather4, destination);
                        }
                    }
            }
        }
        static float AbsMaxNeon(float* x, int n)
        {
            if (!Arm.Neon.IsNeonSupported) return 0f;
            v128 peak = Arm.Neon.vdupq_n_f32(0f);
            for (int i = 0; i < n; i += 4)
                peak = Arm.Neon.vmaxq_f32(peak, Arm.Neon.vabsq_f32(Arm.Neon.vld1q_f32(x + i)));
            return Arm.Neon.vmaxvq_f32(peak);
        }

        static float QuantizeInt8Neon(float* x, int n, sbyte* destination)
        {
            if (!Arm.Neon.IsNeonSupported) return 0f;
            float peak = AbsMaxNeon(x, n);
            float scale = peak > 0f ? peak / 127f : 1f;
            v128 inverse = Arm.Neon.vdupq_n_f32(1f / scale);
            int* spill = stackalloc int[4];
            for (int i = 0; i < n; i += 4)
            {
                Arm.Neon.vst1q_s32(spill, Arm.Neon.vcvtq_s32_f32(Arm.Neon.vmulq_f32(Arm.Neon.vld1q_f32(x + i), inverse)));
                for (int k = 0; k < 4; k++)
                {
                    int v = spill[k];
                    if (v > 127) v = 127;
                    if (v < -128) v = -128;
                    destination[i + k] = (sbyte)v;
                }
            }
            return scale;
        }

        static void Butterfly8Neon(float t0, float t1, float t2, float t3, float scale, float* destination)
        {
            float e0 = t0 + t2;
            float e1 = t1 + t3;
            float o0 = t0 - t2;
            float o1 = t1 - t3;
            destination[0] = (e0 + e1) * scale;
            destination[1] = (e0 - e1) * scale;
            destination[2] = (o0 + o1) * scale;
            destination[3] = (o0 - o1) * scale;
        }

        static void HadamardNeon(float* source, float* p)
        {
            if (!Arm.Neon.IsNeonSupported) return;
            if (source != p)
                for (int i = 0; i < GroupSize; i += 4)
                    Arm.Neon.vst1q_f32(p + i, Arm.Neon.vld1q_f32(source + i));
            for (int stride = 64; stride >= 8; stride >>= 1)
                for (int block = 0; block < GroupSize; block += stride * 2)
                    for (int i = block; i < block + stride; i += 4)
                    {
                        v128 a = Arm.Neon.vld1q_f32(p + i);
                        v128 b = Arm.Neon.vld1q_f32(p + i + stride);
                        Arm.Neon.vst1q_f32(p + i, Arm.Neon.vaddq_f32(a, b));
                        Arm.Neon.vst1q_f32(p + i + stride, Arm.Neon.vsubq_f32(a, b));
                    }
            for (int i = 0; i < GroupSize; i += 8)
            {
                Butterfly8Neon(p[i], p[i + 1], p[i + 2], p[i + 3], HadamardScale, p + i);
                Butterfly8Neon(p[i + 4], p[i + 5], p[i + 6], p[i + 7], HadamardScale, p + i + 4);
            }
        }

        static v128 Table64Neon(v128 t0, v128 t1, v128 t2, v128 t3, byte* index)
        {
            if (!Arm.Neon.IsNeonSupported) return default;
            v128 idx = Arm.Neon.vld1q_u8(index);
            v128 r0 = Arm.Neon.vqtbl1q_u8(t0, idx);
            v128 r1 = Arm.Neon.vqtbl1q_u8(t1, Arm.Neon.vsubq_u8(idx, Arm.Neon.vdupq_n_u8(16)));
            v128 r2 = Arm.Neon.vqtbl1q_u8(t2, Arm.Neon.vsubq_u8(idx, Arm.Neon.vdupq_n_u8(32)));
            v128 r3 = Arm.Neon.vqtbl1q_u8(t3, Arm.Neon.vsubq_u8(idx, Arm.Neon.vdupq_n_u8(48)));
            return Arm.Neon.vorrq_u8(Arm.Neon.vorrq_u8(r0, r1), Arm.Neon.vorrq_u8(r2, r3));
        }

        static int PlanesNeon(sbyte* quantized, int bits, sbyte* destination)
        {
            if (!Arm.Neon.IsNeonSupported) return 0;
            int total = 0;
            for (int i = 0; i < GroupSize; i++) total += quantized[i];
            byte* index = stackalloc byte[16];
            for (int half = 0; half < 2; half++)
            {
                sbyte* q = quantized + half * 64;
                v128 t0 = Arm.Neon.vld1q_u8((byte*)(q));
                v128 t1 = Arm.Neon.vld1q_u8((byte*)(q + 16));
                v128 t2 = Arm.Neon.vld1q_u8((byte*)(q + 32));
                v128 t3 = Arm.Neon.vld1q_u8((byte*)(q + 48));
                if (bits == 2)
                {
                    sbyte* d = destination + half * 16;
                    for (int pl = 0; pl < 4; pl++)
                    {
                        for (int k = 0; k < 8; k++)
                        {
                            index[k] = (byte)(pl + 4 * k);
                            index[8 + k] = (byte)(pl + 32 + 4 * k);
                        }
                        Arm.Neon.vst1q_u8((byte*)(d + pl * 32), Table64Neon(t0, t1, t2, t3, index));
                    }
                }
                else
                {
                    sbyte* d = destination + half * 64;
                    for (int pl = 0; pl < 2; pl++)
                        for (int pair = 0; pair < 2; pair++)
                        {
                            for (int k = 0; k < 8; k++)
                            {
                                index[k] = (byte)(pl + 2 * k + 16 * (2 * pair));
                                index[8 + k] = (byte)(pl + 2 * k + 16 * (2 * pair + 1));
                            }
                            Arm.Neon.vst1q_u8((byte*)(d + pl * 32 + pair * 16), Table64Neon(t0, t1, t2, t3, index));
                        }
                }
            }
            return total;
        }

        static void PrepareNeon(float* x, int tokens, int cols, int bits, BurstActivations* a)
        {
            float* rotated = stackalloc float[GroupSize];
            sbyte* quantized = stackalloc sbyte[GroupSize];
            for (int t = 0; t < tokens; t++)
                for (int g = 0; g < a->Groups; g++)
                {
                    int slot = t * a->Stride + g;
                    HadamardNeon(x + (long)t * cols + g * GroupSize, rotated);
                    a->Scales[slot] = QuantizeInt8Neon(rotated, GroupSize, quantized);
                    int sum = PlanesNeon(quantized, bits, a->Values + ((long)t * a->Groups + g) * GroupSize);
                    a->Offsets[slot] = CodebookBias * (float)sum;
                }
        }

        static float RoundEvenScalar(float x) => (x + RoundBias) - RoundBias;

        static float AbsMaxScalar(float* x, int n)
        {
            float peak = 0f;
            for (int i = 0; i < n; i++) peak = MathF.Max(peak, MathF.Abs(x[i]));
            return peak;
        }

        static float QuantizeInt8Scalar(float* x, int n, sbyte* destination)
        {
            float peak = AbsMaxScalar(x, n);
            float scale = peak > 0f ? peak / 127f : 1f;
            float inverse = 1f / scale;
            for (int i = 0; i < n; i++) destination[i] = (sbyte)MathF.Min(MathF.Max(RoundEvenScalar(x[i] * inverse), -128f), 127f);
            return scale;
        }

        static void HadamardScalar(float* source, float* p)
        {
            if (source != p) for (int i = 0; i < GroupSize; i++) p[i] = source[i];
            for (int stride = 1; stride < GroupSize; stride <<= 1)
                for (int block = 0; block < GroupSize; block += stride * 2)
                    for (int i = block; i < block + stride; i++)
                    {
                        float a = p[i];
                        float b = p[i + stride];
                        p[i] = a + b;
                        p[i + stride] = a - b;
                    }
            for (int i = 0; i < GroupSize; i++) p[i] *= HadamardScale;
        }

        static int PlanesScalar(sbyte* quantized, int bits, sbyte* destination)
        {
            int sum = 0;
            if (bits == 2)
                for (int j = 0; j < 32; j++)
                    for (int pl = 0; pl < 4; pl++)
                    {
                        sbyte q = quantized[4 * j + pl];
                        destination[pl * 32 + j] = q;
                        sum += q;
                    }
            else
                for (int half = 0; half < 2; half++)
                    for (int j = 0; j < 32; j++)
                        for (int pl = 0; pl < 2; pl++)
                        {
                            sbyte q = quantized[half * 64 + 2 * j + pl];
                            destination[half * 64 + pl * 32 + j] = q;
                            sum += q;
                        }
            return sum;
        }

        static void PrepareScalar(float* x, int tokens, int cols, int bits, BurstActivations* a)
        {
            float* rotated = stackalloc float[GroupSize];
            sbyte* quantized = stackalloc sbyte[GroupSize];
            for (int t = 0; t < tokens; t++)
                for (int g = 0; g < a->Groups; g++)
                {
                    int slot = t * a->Stride + g;
                    HadamardScalar(x + (long)t * cols + g * GroupSize, rotated);
                    a->Scales[slot] = QuantizeInt8Scalar(rotated, GroupSize, quantized);
                    int sum = PlanesScalar(quantized, bits, a->Values + ((long)t * a->Groups + g) * GroupSize);
                    a->Offsets[slot] = CodebookBias * (float)sum;
                }
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void PrepareFn(float* x, int tokens, int cols, int bits, BurstActivations* a);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate float QuantizeInt8Fn(float* x, int n, sbyte* destination);

        public static bool IsInstalled => Kernels.Active.Prepare != 0 && Kernels.Active.QuantizeInt8 != 0;

        public static void Install()
        {
            if (IsInstalled)
            {
                return;
            }
            FunctionPointer<PrepareFn> prepared =
                BurstCompiler.CompileFunctionPointer<PrepareFn>(InvokePrepare);
            FunctionPointer<QuantizeInt8Fn> quantized =
                BurstCompiler.CompileFunctionPointer<QuantizeInt8Fn>(InvokeQuantizeInt8);
            if (!prepared.IsCreated || !quantized.IsCreated)
                throw new InvalidOperationException(
                    "N35: Burst refused to compile BurstPrepare.InvokePrepare/InvokeQuantizeInt8. " +
                    "The kernels must live in package source (H6); " +
                    "a precompiled DLL entry cannot be compiled.");
            KernelTable table = Kernels.Active;
            if (table.Prepare == 0) table.Prepare = (nint)prepared.Value;
            if (table.QuantizeInt8 == 0) table.QuantizeInt8 = (nint)quantized.Value;
            Kernels.Install(table, Kernels.IsInstalled ? Kernels.Provider + "+burst-prepare" : "burst-prepare");
        }

        public static double MeasurePrepareCallOverheadNs(int tokens = 2, int cols = 256, int calls = 20000)
        {
            if (tokens < 1 || cols < 128 || calls < 1) throw new ArgumentOutOfRangeException();
            Install();
            delegate* unmanaged[Cdecl]<float*, int, int, int, BurstActivations*, void> fp =
                (delegate* unmanaged[Cdecl]<float*, int, int, int, BurstActivations*, void>)(void*)Kernels.Active.Prepare;
            int groups = cols / GroupSize;
            int stride = (groups + 7) & ~7;
            float* x = stackalloc float[tokens * cols];
            sbyte* values = stackalloc sbyte[tokens * groups * GroupSize];
            float* scales = stackalloc float[tokens * stride];
            float* offsets = stackalloc float[tokens * stride];
            for (int i = 0; i < tokens * cols; i++) x[i] = (i % 9) * 0.11f - 0.4f;
            BurstActivations a = new BurstActivations();
            a.Values = values;
            a.Scales = scales;
            a.Offsets = offsets;
            a.Cols = cols;
            a.Groups = groups;
            a.Stride = stride;
            for (int w = 0; w < 100; w++) fp(x, tokens, cols, 2, &a);
            Stopwatch stopwatch = Stopwatch.StartNew();
            for (int c = 0; c < calls; c++) fp(x, tokens, cols, 2, &a);
            stopwatch.Stop();
            return (double)stopwatch.ElapsedTicks * 1000000000.0 / Stopwatch.Frequency / calls;
        }
    }
}
