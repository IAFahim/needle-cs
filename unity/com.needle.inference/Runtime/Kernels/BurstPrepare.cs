// N35: Burst Prepare + Hadamard + QuantizeInt8 + Planes2/4, x86 and Neon.
//
// Ports Quant.Prepare 1:1 to Unity.Burst.Intrinsics: the x86 lane mirrors the
// net10 AVX2+FMA path (same op order, same constants, same FMA use), the Neon
// lane mirrors Neon.Prepare (same op order, same truncation/narrowing, same
// table permutations). Anything else falls back to a scalar lane that mirrors
// Scalar.* exactly. All three lanes are exact-ops kernels (integer sums,
// byte permutes, round/clamp), so fixtures compare bitwise (H7); there is no
// parity fallback in this file.
//
// Burst rules (plan §2.1 cookbook) this file follows:
//   - FloatMode.Strict on both entry points, never Fast: no reassociation.
//   - Explicit FMA only on x86 (mm256_fmadd_ps, matching Fma.MultiplyAdd);
//     the Neon Hadamard has no FMA in the reference either.
//   - Blittable arguments only; no allocation, no static mutable state; safe
//     to call from any thread (pool workers call via the table).
//   - H8: no AVX-VNNI anywhere. The Prepare offset sum uses the Avx2Dot
//     sequence (maddubs + madd + add, Tiers.Avx2Dot) exactly like the
//     non-VNNI tier of Tiers.Accumulate.
//
// Exactness arguments (each is a bit-identity, not a bound):
//   - QuantizeInt8 x86: the net10 packs_epi32/packs_epi16 pair saturates each
//     int32 lane to sbyte range. The Burst lane spills the cvtps output and
//     clamps each lane in scalar: identical bytes for every int32 input,
//     including the 0x80000000 out-of-range sentinel (clamps to -128, which
//     is what the saturating pack produces). Rounding itself stays vector
//     (mm256_cvtps_epi32 = VCVTPS2DQ, round-to-nearest-even, same as BCL).
//   - QuantizeInt8 Neon: vcvtq_s32_f32 is FCVTZS (round toward zero), then
//     scalar clamp mirrors the double narrowing saturate lane for lane.
//   - Prepare offset sum: integer addition is associative, so the spill +
//     scalar add of the 8 int32 lanes equals the net10 Vector256.Sum, and
//     the scalar 128-byte sum equals Neon.SumOf (dotprod or widening), bit
//     for bit. The 63x scale-up is one exact float multiply either way.
//   - Planes x86: same shuffle/unpack/permute instruction order as
//     Quant.Planes2/Planes4 (gather/interleave constants rebuilt identically).
//   - Planes Neon: Burst exposes only single-register TBL (vqtbl1q_u8), so
//     each 16-byte group is looked up in the four 16-byte sub-tables with
//     (idx - 16k) and ORed. TBL zeroes out-of-range lanes, so exactly one
//     lookup hits per lane and the OR equals the 64-byte table result.
//   - Hadamard Neon tail: the scalar Butterfly8 below reproduces the
//     Neon.Butterfly8 dataflow op for op (AddPairwise/ZipLow unrolled, same
//     association), verified against a lane simulation (see N13 fixtures).
//
// C# 9 safe: block namespace, no target-typed new, no static lambdas.
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;

namespace Needle.BurstKernels
{
    // Layout mirror of the internal Needle.Activations (Model-adjacent struct
    // in Quant.cs: sbyte* Values, float* Scales, float* Offsets, int Cols,
    // int Groups, int Stride). The table ABI passes a bare pointer, so an
    // identical layout reads the caller's struct exactly.
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

    public static unsafe class BurstPrepare
    {
        // Cq.GroupSize = 128 (Model.cs); Codebook.Bias = 63 (Model.cs).
        const int GroupSize = 128;
        const float HadamardScale = 0.08838834764831845f;
        const float CodebookBias = 63f;
        const float RoundBias = 12582912f; // 1.5 * (1 << 23); Scalar.RoundEven

        // Table entry: void(float* x, int tokens, int cols, int bits, Activations*).
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokePrepare(float* x, int tokens, int cols, int bits, BurstActivations* a)
        {
            // Branches fold at Burst compile time: exactly one lane survives
            // per target (x64 with AVX2+FMA, arm64 Neon, scalar elsewhere).
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { PrepareX86(x, tokens, cols, bits, a); return; }
            if (Arm.Neon.IsNeonSupported) { PrepareNeon(x, tokens, cols, bits, a); return; }
            PrepareScalar(x, tokens, cols, bits, a);
        }

        // Table entry: float(float* x, int n, sbyte* destination).
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static float InvokeQuantizeInt8(float* x, int n, sbyte* destination)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { return QuantizeInt8X86(x, n, destination); }
            if (Arm.Neon.IsNeonSupported) { return QuantizeInt8Neon(x, n, destination); }
            return QuantizeInt8Scalar(x, n, destination);
        }

        // Mirrors Quant.AbsMax: peak starts at zero, abs is ANDNOT -0.0 (same
        // as Vector256.Abs), then the Max(lower, upper) / 0x4E / 0xB1 fold.
        static float AbsMaxX86(float* x, int n)
        {
            v256 sign = X86.Avx.mm256_set1_ps(-0.0f);
            v256 peak = X86.Avx.mm256_set1_ps(0f);
            for (int i = 0; i < n; i += 8)
                peak = X86.Avx.mm256_max_ps(peak, X86.Avx.mm256_andnot_ps(sign, X86.Avx.mm256_loadu_ps(x + i)));
            v128 halves = X86.Sse.max_ps(X86.Avx.mm256_castps256_ps128(peak), X86.Avx.mm256_extractf128_ps(peak, 1));
            v128 pairs = X86.Sse.max_ps(halves, X86.Sse.shuffle_ps(halves, halves, 0x4E));
            v128 quad = X86.Sse.max_ps(pairs, X86.Sse.shuffle_ps(pairs, pairs, 0xB1));
            return X86.Sse.cvtss_f32(quad);
        }

        // Mirrors Quant.QuantizeInt8: same peak/scale/inverse, same VCVTPS2DQ
        // rounding; narrowing is a per-lane clamp over the cvt spill (bit-
        // identical to the packs_epi32/packs_epi16 pair, see file header).
        static float QuantizeInt8X86(float* x, int n, sbyte* destination)
        {
            float peak = AbsMaxX86(x, n);
            float scale = peak > 0f ? peak / 127f : 1f;
            v256 inverse = X86.Avx.mm256_set1_ps(1f / scale);
            int* spill = stackalloc int[8];
            for (int i = 0; i < n; i += 8)
            {
                v256 q = X86.Avx.mm256_cvtps_epi32(X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(x + i), inverse));
                X86.Avx.mm256_storeu_si256((v256*)spill, q);
                for (int k = 0; k < 8; k++)
                {
                    int v = spill[k];
                    if (v > 127) v = 127;
                    if (v < -128) v = -128;
                    destination[i + k] = (sbyte)v;
                }
            }
            return scale;
        }

        // Mirrors Quant.Hadamard exactly: the 64-wide interleave stage, the
        // 32..8 stride stages, then the FMA sign triple with the same
        // 0xB1/0x4E/0x01 permutes and the same scale. In-place safe (each
        // second-half lane is read before it is written, same iteration).
        static void HadamardX86(float* source, float* p)
        {
            for (int i = 0; i < 64; i += 8)
            {
                v256 a = X86.Avx.mm256_loadu_ps(source + i);
                v256 b = X86.Avx.mm256_loadu_ps(source + i + 64);
                X86.Avx.mm256_storeu_ps(p + i, X86.Avx.mm256_add_ps(a, b));
                X86.Avx.mm256_storeu_ps(p + i + 64, X86.Avx.mm256_sub_ps(a, b));
            }
            for (int stride = 32; stride >= 8; stride >>= 1)
                for (int block = 0; block < GroupSize; block += stride * 2)
                    for (int i = block; i < block + stride; i += 8)
                    {
                        v256 a = X86.Avx.mm256_loadu_ps(p + i);
                        v256 b = X86.Avx.mm256_loadu_ps(p + i + stride);
                        X86.Avx.mm256_storeu_ps(p + i, X86.Avx.mm256_add_ps(a, b));
                        X86.Avx.mm256_storeu_ps(p + i + stride, X86.Avx.mm256_sub_ps(a, b));
                    }
            float* signs = stackalloc float[24];
            for (int k = 0; k < 8; k++)
            {
                signs[k] = (k & 1) == 0 ? 1f : -1f;
                signs[8 + k] = (k & 2) == 0 ? 1f : -1f;
                signs[16 + k] = k < 4 ? 1f : -1f;
            }
            v256 pairSign = X86.Avx.mm256_loadu_ps(signs);
            v256 quadSign = X86.Avx.mm256_loadu_ps(signs + 8);
            v256 halfSign = X86.Avx.mm256_loadu_ps(signs + 16);
            v256 scale = X86.Avx.mm256_set1_ps(HadamardScale);
            for (int i = 0; i < GroupSize; i += 8)
            {
                v256 v = X86.Avx.mm256_loadu_ps(p + i);
                v = X86.Fma.mm256_fmadd_ps(v, pairSign, X86.Avx.mm256_permute_ps(v, 0xB1));
                v = X86.Fma.mm256_fmadd_ps(v, quadSign, X86.Avx.mm256_permute_ps(v, 0x4E));
                v = X86.Fma.mm256_fmadd_ps(v, halfSign, X86.Avx.mm256_permute2f128_ps(v, v, 0x01));
                X86.Avx.mm256_storeu_ps(p + i, X86.Avx.mm256_mul_ps(v, scale));
            }
        }

        // Mirrors Quant.Planes2: same gather constant, same
        // shuffle/unpack/permute order, same 0xB1-style interleave.
        static void Planes2X86(v256 q0, v256 q1, v256 q2, v256 q3, sbyte* destination)
        {
            byte* g = stackalloc byte[32];
            for (int k = 0; k < 16; k++)
            {
                byte v = (byte)(4 * (k % 4) + k / 4);
                g[k] = v;
                g[k + 16] = v;
            }
            v256 gather = X86.Avx.mm256_loadu_si256((v256*)g);
            v256 s0 = X86.Avx2.mm256_shuffle_epi8(q0, gather);
            v256 s1 = X86.Avx2.mm256_shuffle_epi8(q1, gather);
            v256 s2 = X86.Avx2.mm256_shuffle_epi8(q2, gather);
            v256 s3 = X86.Avx2.mm256_shuffle_epi8(q3, gather);
            v256 low01 = X86.Avx2.mm256_unpacklo_epi32(s0, s1);
            v256 high01 = X86.Avx2.mm256_unpackhi_epi32(s0, s1);
            v256 low23 = X86.Avx2.mm256_unpacklo_epi32(s2, s3);
            v256 high23 = X86.Avx2.mm256_unpackhi_epi32(s2, s3);
            int* ix = stackalloc int[8];
            for (int k = 0; k < 8; k++) ix[k] = k / 2 + (k % 2) * 4;
            v256 interleave = X86.Avx.mm256_loadu_si256((v256*)ix);
            X86.Avx.mm256_storeu_si256((v256*)(destination), X86.Avx2.mm256_permutevar8x32_epi32(X86.Avx2.mm256_unpacklo_epi64(low01, low23), interleave));
            X86.Avx.mm256_storeu_si256((v256*)(destination + 32), X86.Avx2.mm256_permutevar8x32_epi32(X86.Avx2.mm256_unpackhi_epi64(low01, low23), interleave));
            X86.Avx.mm256_storeu_si256((v256*)(destination + 64), X86.Avx2.mm256_permutevar8x32_epi32(X86.Avx2.mm256_unpacklo_epi64(high01, high23), interleave));
            X86.Avx.mm256_storeu_si256((v256*)(destination + 96), X86.Avx2.mm256_permutevar8x32_epi32(X86.Avx2.mm256_unpackhi_epi64(high01, high23), interleave));
        }

        // Mirrors Quant.Planes4: same gather constant, same
        // shuffle/unpack/permute order (0b11_01_10_00 = 0xD8).
        static void Planes4X86(v256 q0, v256 q1, v256 q2, v256 q3, sbyte* destination)
        {
            byte* g = stackalloc byte[32];
            for (int k = 0; k < 16; k++)
            {
                byte v = (byte)(2 * (k % 8) + k / 8);
                g[k] = v;
                g[k + 16] = v;
            }
            v256 gather = X86.Avx.mm256_loadu_si256((v256*)g);
            v256 s0 = X86.Avx2.mm256_shuffle_epi8(q0, gather);
            v256 s1 = X86.Avx2.mm256_shuffle_epi8(q1, gather);
            v256 s2 = X86.Avx2.mm256_shuffle_epi8(q2, gather);
            v256 s3 = X86.Avx2.mm256_shuffle_epi8(q3, gather);
            X86.Avx.mm256_storeu_si256((v256*)(destination), X86.Avx2.mm256_permute4x64_epi64(X86.Avx2.mm256_unpacklo_epi64(s0, s1), 0xD8));
            X86.Avx.mm256_storeu_si256((v256*)(destination + 32), X86.Avx2.mm256_permute4x64_epi64(X86.Avx2.mm256_unpackhi_epi64(s0, s1), 0xD8));
            X86.Avx.mm256_storeu_si256((v256*)(destination + 64), X86.Avx2.mm256_permute4x64_epi64(X86.Avx2.mm256_unpacklo_epi64(s2, s3), 0xD8));
            X86.Avx.mm256_storeu_si256((v256*)(destination + 96), X86.Avx2.mm256_permute4x64_epi64(X86.Avx2.mm256_unpackhi_epi64(s2, s3), 0xD8));
        }

        // Mirrors Quant.Prepare on the AVX2 tier: same per-group Hadamard,
        // quantize, Avx2Dot offset sum (H8: the non-VNNI Tiers.Accumulate
        // sequence) and planes dispatch.
        static void PrepareX86(float* x, int tokens, int cols, int bits, BurstActivations* a)
        {
            float* rotated = stackalloc float[GroupSize];
            sbyte* quantized = stackalloc sbyte[GroupSize];
            v256 onesByte = X86.Avx.mm256_set1_epi8((byte)1);
            v256 onesShort = X86.Avx.mm256_set1_epi16((short)1);
            int* accSpill = stackalloc int[8];
            for (int t = 0; t < tokens; t++)
                for (int g = 0; g < a->Groups; g++)
                {
                    int slot = t * a->Stride + g;
                    HadamardX86(x + (long)t * cols + g * GroupSize, rotated);
                    a->Scales[slot] = QuantizeInt8X86(rotated, GroupSize, quantized);
                    v256 q0 = X86.Avx.mm256_loadu_si256((v256*)(quantized));
                    v256 q1 = X86.Avx.mm256_loadu_si256((v256*)(quantized + 32));
                    v256 q2 = X86.Avx.mm256_loadu_si256((v256*)(quantized + 64));
                    v256 q3 = X86.Avx.mm256_loadu_si256((v256*)(quantized + 96));
                    v256 sum = X86.Avx2.mm256_add_epi32(
                        X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(onesByte, q0), onesShort),
                        X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(onesByte, q1), onesShort));
                    sum = X86.Avx2.mm256_add_epi32(sum, X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(onesByte, q2), onesShort));
                    sum = X86.Avx2.mm256_add_epi32(sum, X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(onesByte, q3), onesShort));
                    X86.Avx.mm256_storeu_si256((v256*)accSpill, sum);
                    int total = 0;
                    for (int k = 0; k < 8; k++) total += accSpill[k];
                    a->Offsets[slot] = CodebookBias * (float)total;
                    sbyte* destination = a->Values + ((long)t * a->Groups + g) * GroupSize;
                    if (bits == 2) Planes2X86(q0, q1, q2, q3, destination);
                    else Planes4X86(q0, q1, q2, q3, destination);
                }
        }
        // Mirrors Neon.AbsMax: vector max of |x|, then max-across.
        static float AbsMaxNeon(float* x, int n)
        {
            v128 peak = Arm.Neon.vdupq_n_f32(0f);
            for (int i = 0; i < n; i += 4)
                peak = Arm.Neon.vmaxq_f32(peak, Arm.Neon.vabsq_f32(Arm.Neon.vld1q_f32(x + i)));
            return Arm.Neon.vmaxvq_f32(peak);
        }

        // Mirrors Neon.QuantizeInt8: same peak/scale/inverse, FCVTZS
        // truncation, then a per-lane clamp over the convert spill
        // (bit-identical to the double narrowing saturate).
        static float QuantizeInt8Neon(float* x, int n, sbyte* destination)
        {
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

        // Closed-form Neon.Butterfly8 (same association, op for op):
        // rotated = [t2,t3,t0,t1]; even = low(t+rot) = [t0+t2,t1+t3];
        // odd = low(t-rot) = [t0-t2,t1-t3]; out = [e0+e1, e0-e1, o0+o1, o0-o1].
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

        // Mirrors Neon.Hadamard exactly: the conditional copy, the 64..8
        // stride stages, then Butterfly8 over each consecutive 8 scaled once.
        static void HadamardNeon(float* source, float* p)
        {
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

        // One 16-byte group through the four 16-byte sub-tables. TBL zeroes
        // out-of-range lanes, so exactly one lookup hits per lane and the OR
        // equals the 64-byte table result (bit-identical to vqtbl1 over the
        // (q0,q1,q2,q3) tuple in Neon.StorePlanes2/4).
        static v128 Table64Neon(v128 t0, v128 t1, v128 t2, v128 t3, byte* index)
        {
            v128 idx = Arm.Neon.vld1q_u8(index);
            v128 r0 = Arm.Neon.vqtbl1q_u8(t0, idx);
            v128 r1 = Arm.Neon.vqtbl1q_u8(t1, Arm.Neon.vsubq_u8(idx, Arm.Neon.vdupq_n_u8(16)));
            v128 r2 = Arm.Neon.vqtbl1q_u8(t2, Arm.Neon.vsubq_u8(idx, Arm.Neon.vdupq_n_u8(32)));
            v128 r3 = Arm.Neon.vqtbl1q_u8(t3, Arm.Neon.vsubq_u8(idx, Arm.Neon.vdupq_n_u8(48)));
            return Arm.Neon.vorrq_u8(Arm.Neon.vorrq_u8(r0, r1), Arm.Neon.vorrq_u8(r2, r3));
        }

        // Mirrors Neon.Planes bit for bit: same input/output permutation for
        // both bit depths (the index math matches StorePlanes2/StorePlanes4
        // lane for lane); the offset sum is a scalar 128-byte add, which is
        // bit-identical to Neon.SumOf because integer addition is associative.
        static int PlanesNeon(sbyte* quantized, int bits, sbyte* destination)
        {
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

        // Mirrors Neon.Prepare: same per-group Hadamard, quantize, planes and
        // Bias-scaled offset.
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

        // Scalar lane: mirrors Scalar.* exactly (the !Simd.Enabled path in
        // Quant), for targets with neither AVX2+FMA nor Neon.
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

        // Cdecl delegates for the table slots (KernelTable ABI):
        // Prepare is void(float*, int, int, int, Activations*),
        // QuantizeInt8 is float(float*, int, sbyte*).
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void PrepareFn(float* x, int tokens, int cols, int bits, BurstActivations* a);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate float QuantizeInt8Fn(float* x, int n, sbyte* destination);

        public static bool IsInstalled => Kernels.Active.Prepare != 0 && Kernels.Active.QuantizeInt8 != 0;

        // Idempotent: never overwrites an explicit or foreign provider's
        // entries (plan §2 selection order).
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

        // H5-style microbench for the Prepare entry: one warmed call shape
        // (tokens=2, cols=256, bits=2), timed through the installed Cdecl
        // pointer. Returns nanoseconds per call.
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
