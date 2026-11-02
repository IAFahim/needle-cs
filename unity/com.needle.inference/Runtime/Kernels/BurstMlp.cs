// N37: Burst Hadamard MLP kernel (plan-netstandard.md Track B).
//
// Ports HadamardMlp.Apply 1:1 to Unity.Burst.Intrinsics: the x86 path mirrors
// the net10 AVX2+FMA lane (KronAvx2 register blocking, ConditionWeights,
// PermuteModulateActivate, PermuteScale, AccumulateOutput, same FMA order,
// same reductions), the Neon path mirrors Neon.Kron/ConditionWeights/
// PermuteModulateActivate/PermuteScale/AccumulateOutput, and a scalar lane
// mirrors Scalar.* for targets with neither (WebGL-era fallbacks).
//
// Lane mapping (net10 -> Burst):
//   Avx.LoadVector256 / Store            -> mm256_loadu_ps / mm256_storeu_ps
//   Avx.BroadcastScalarToVector256(float*) -> mm256_set1_ps(*p): same lane
//     bits as VBROADCASTSS (a broadcast is a broadcast), so H7 is unaffected
//     while avoiding an API name the installed Burst may spell differently.
//   Fma.MultiplyAdd / MultiplyAddNegated -> mm256_fmadd_ps / mm256_fnmadd_ps
//   Avx.ConvertToVector256Int32 / Avx2 shifts / casts -> cvtps_epi32,
//     add_epi32, slli_epi32, castsi256_ps (Exp bit tricks, as in BurstSoftmax)
//   AdvSimd.LoadVector128 / Store        -> vld1q_f32 / vst1q_f32
//   AdvSimd.LoadAndReplicateToVector128(float*) -> vdupq_n_f32(*p): same bits.
//   AdvSimd.FusedMultiplyAdd(acc, x, y)  -> vfmaq_f32(acc, x, y)
//   Gather (permuted, unaligned)         -> scalar spill + one vector load:
//     the permuted lanes are gathered to a small stack buffer first, then
//     loaded as a vector. Same lane values as Vector128/256.Create(...),
//     using only load/store APIs every Burst carries.
//   Neon Sigmoid divide                  -> spill + scalar 1/d per lane:
//     IEEE fdiv is correctly rounded per lane either way, so this is bitwise
//     identical to a vector divide with no vdivq dependency.
//   Neon Exp 2^n scale                   -> SIGNED convert/add/shift/spill,
//     matching Neon.Exp (ConvertToInt32RoundToEven, +127, <<23, reinterpret).
//
// Burst rules (plan section 2.1 cookbook), as in BurstSoftmax:
//   - FloatMode.Strict on the entry point, never Fast: no reassociation.
//   - Explicit FMA intrinsics; no reliance on contraction.
//   - No [NoAlias]: residual aliases the norm input, front/back alias through
//     the phase chain, so no aliasing promise is made.
//   - Blittable arguments only (BurstMlpBatch* mirrors the internal MlpBatch
//     layout field for field; float*/int* buffers); no allocation on the hot
//     path (one tile stackalloc per Kron call, same as net10), no static
//     mutable state, safe from any thread.
//   - Needs no AVX-VNNI and no AVX-512 (H8): the x86 Kron is the Avx2Dot-style
//     FMA sequence, and KronAvx2 vs KronAvx512 only differ in tile granularity
//     (per-element i/j sum order is identical), so one x86 lane matches both.
//
// Geometry gates mirror net10 exactly: the vector Krons run only when
// rows % 4 == 0 && cols % 16 == 0 (HadamardMlp.Blocked); anything ragged
// takes the scalar Kron loops. dim is assumed a multiple of 8 on the x86 norm
// path and even on the Neon condition path, exactly as the net10 lanes assume
// (their vector loops have no tails there either).
//
// H7 note: the x86 lane keeps the net10 instruction and FMA order, so finite
// fixtures in range compare bitwise (same rounding at every step). The scalar
// lane mirrors Scalar.* op for op. Parity bound (cosine >= 0.99995, same
// argmax) holds by construction in every lane.
//
// C# 9 safe: block namespace, no target-typed new, no static lambdas.
using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using UnityEngine;

namespace Needle.BurstKernels
{
    // Blittable mirror of the internal MlpWeights (Audio.cs): 14 float* in the
    // same declaration order (D1, D2, B2, D3, D4, W1A, W1B, W2A, W2B, W3A, W3B,
    // CondV, CondU). Sequential layout makes this pointer-interchangeable with
    // MlpWeights* (both are 14 native pointers, 112 bytes on 64-bit).
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct BurstMlpWeights
    {
        public float* D1;
        public float* D2;
        public float* B2;
        public float* D3;
        public float* D4;
        public float* W1A;
        public float* W1B;
        public float* W2A;
        public float* W2B;
        public float* W3A;
        public float* W3B;
        public float* CondV;
        public float* CondU;
    }

    // Blittable mirror of the internal MlpBatch (Blocks.cs): Weights, PreNorm,
    // PermA, PermB, Residual, Dim, Rows, Cols, Scale in order (Width is a
    // computed property on the original, not a field).
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct BurstMlpBatch
    {
        public BurstMlpWeights* Weights;
        public float* PreNorm;
        public int* PermA;
        public int* PermB;
        public float* Residual;
        public int Dim;
        public int Rows;
        public int Cols;
        public float Scale;
    }

    public static unsafe class BurstMlp
    {
        const int Conditions = 8;
        const int TileRows = 4;
        const int TileColumns = 16;

        // Entry point installed into KernelTable.HadamardMlp:
        // void(MlpBatch*, float* residual, float* scratch), Cdecl.
        // scratch must hold HadamardMlp.ScratchFloats(dim, width) floats.
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void Invoke(BurstMlpBatch* b, float* residual, float* scratch)
        {
            // Branches fold at Burst compile time: exactly one lane survives
            // per target (x64 with AVX2+FMA, arm64 Neon, scalar elsewhere).
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported) { ApplyX86(b, residual, scratch); return; }
            if (Arm.Neon.IsNeonSupported) { ApplyNeon(b, residual, scratch); return; }
            ApplyScalar(b, residual, scratch);
        }

        // Mirrors HadamardMlp.Apply: partition layout, phase order, alignment.
        internal static void ApplyX86(BurstMlpBatch* b, float* residual, float* scratch)
        {
            BurstMlpWeights* w = b->Weights;
            int dim = b->Dim;
            int rows = b->Rows;
            int cols = b->Cols;
            int width = rows * cols;
            int inputRows = (dim + cols - 1) / cols;
            int outputRows = rows < (inputRows + TileRows - 1) / TileRows * TileRows
                ? rows : (inputRows + TileRows - 1) / TileRows * TileRows;
            float* aligned = (float*)(((nuint)scratch + 31) & ~(nuint)31);
            float* weights = aligned;
            float* normalized = weights + Conditions;
            float* front = normalized + dim;
            float* back = front + width;

            ZeroCenteredRmsX86(residual, b->PreNorm, dim, normalized);
            ConditionWeightsX86(w->CondV, normalized, dim, weights);
            ScaleInputX86(normalized, w->D1, dim, inputRows * cols, front);
            KronX86(front, inputRows, w->W1A, w->W1B, rows, cols, rows, back);
            PermuteModulateActivateX86(b->PermA, back, w->D2, weights, w->CondU, w->B2, width, front);
            KronX86(front, rows, w->W2A, w->W2B, rows, cols, rows, back);
            PermuteScaleX86(b->PermB, back, w->D3, width, front);
            KronX86(front, rows, w->W3A, w->W3B, rows, cols, outputRows, back);
            AccumulateOutputX86(residual, back, w->D4, b->Scale, dim);
        }

        internal static void ApplyNeon(BurstMlpBatch* b, float* residual, float* scratch)
        {
            BurstMlpWeights* w = b->Weights;
            int dim = b->Dim;
            int rows = b->Rows;
            int cols = b->Cols;
            int width = rows * cols;
            int inputRows = (dim + cols - 1) / cols;
            int outputRows = rows < (inputRows + TileRows - 1) / TileRows * TileRows
                ? rows : (inputRows + TileRows - 1) / TileRows * TileRows;
            float* aligned = (float*)(((nuint)scratch + 31) & ~(nuint)31);
            float* weights = aligned;
            float* normalized = weights + Conditions;
            float* front = normalized + dim;
            float* back = front + width;

            ZeroCenteredRmsNeon(residual, b->PreNorm, dim, normalized);
            ConditionWeightsNeon(w->CondV, normalized, dim, weights);
            ScaleInputNeon(normalized, w->D1, dim, inputRows * cols, front);
            KronNeon(front, inputRows, w->W1A, w->W1B, rows, cols, rows, back);
            PermuteModulateActivateNeon(b->PermA, back, w->D2, weights, w->CondU, w->B2, width, front);
            KronNeon(front, rows, w->W2A, w->W2B, rows, cols, rows, back);
            PermuteScaleNeon(b->PermB, back, w->D3, width, front);
            KronNeon(front, rows, w->W3A, w->W3B, rows, cols, outputRows, back);
            AccumulateOutputNeon(residual, back, w->D4, b->Scale, dim);
        }

        internal static void ApplyScalar(BurstMlpBatch* b, float* residual, float* scratch)
        {
            BurstMlpWeights* w = b->Weights;
            int dim = b->Dim;
            int rows = b->Rows;
            int cols = b->Cols;
            int width = rows * cols;
            int inputRows = (dim + cols - 1) / cols;
            int outputRows = rows < (inputRows + TileRows - 1) / TileRows * TileRows
                ? rows : (inputRows + TileRows - 1) / TileRows * TileRows;
            float* aligned = (float*)(((nuint)scratch + 31) & ~(nuint)31);
            float* weights = aligned;
            float* normalized = weights + Conditions;
            float* front = normalized + dim;
            float* back = front + width;

            ZeroCenteredRmsScalar(residual, b->PreNorm, dim, normalized);
            ConditionWeightsScalar(w->CondV, normalized, dim, weights);
            ScaleInputScalar(normalized, w->D1, dim, inputRows * cols, front);
            KronScalar(front, inputRows, w->W1A, w->W1B, rows, cols, rows, back);
            PermuteModulateActivateScalar(b->PermA, back, w->D2, weights, w->CondU, w->B2, width, front);
            KronScalar(front, rows, w->W2A, w->W2B, rows, cols, rows, back);
            PermuteScaleScalar(b->PermB, back, w->D3, width, front);
            KronScalar(front, rows, w->W3A, w->W3B, rows, cols, outputRows, back);
            AccumulateOutputScalar(residual, back, w->D4, b->Scale, dim);
        }

        // ---- x86 lane (mirrors the net10 AVX2+FMA path) ----

        // Mirrors Ops.SumOfSquares exactly: 4 FMA accumulators over 32-wide
        // bulk, 16-wide pair step, one 8-wide step, scalar FMA tail, hadd-sum
        // plus tail (same add order as Vector256.Sum + tail).
        internal static float SumOfSquaresX86(float* x, int n)
        {
            v256 s0 = X86.Avx.mm256_set1_ps(0f);
            v256 s1 = X86.Avx.mm256_set1_ps(0f);
            v256 s2 = X86.Avx.mm256_set1_ps(0f);
            v256 s3 = X86.Avx.mm256_set1_ps(0f);
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
            for (; i < n; i++) tail = (x[i] * x[i] + tail);
            v256 folded = X86.Avx.mm256_add_ps(
                X86.Avx.mm256_add_ps(s0, s1),
                X86.Avx.mm256_add_ps(s2, s3));
            return Sum256X86(folded) + tail;
        }

        internal static float InverseRmsX86(float* x, int n)
        {
            return 1f / MathF.Sqrt(SumOfSquaresX86(x, n) / n + 1e-6f);
        }

        // Mirrors the Ops.ZeroCenteredRms AVX2 form: Add + Mul + Mul (the
        // SIMD-05 Mul+FMA affine is deliberately NOT used; see Ops.cs).
        internal static void ZeroCenteredRmsX86(float* x, float* scale, int n, float* destination)
        {
            v256 inverse = X86.Avx.mm256_set1_ps(InverseRmsX86(x, n));
            v256 one = X86.Avx.mm256_set1_ps(1f);
            for (int i = 0; i < n; i += 8)
                X86.Avx.mm256_storeu_ps(destination + i, X86.Avx.mm256_mul_ps(
                    X86.Avx.mm256_mul_ps(
                        X86.Avx.mm256_add_ps(one, X86.Avx.mm256_loadu_ps(scale + i)),
                        X86.Avx.mm256_loadu_ps(x + i)),
                    inverse));
        }

        // Mirrors Ops.Exp bit for bit (same as BurstSoftmax.ExpX86, duplicated
        // so this file is self-contained): clamp, round, two fnmadd reductions,
        // six-term FMA polynomial, 2^n scale from integer bit tricks.
        internal static v256 ExpX86(v256 x)
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

        internal static v256 SigmoidX86(v256 x)
        {
            v256 one = X86.Avx.mm256_set1_ps(1f);
            v256 zero = X86.Avx.mm256_set1_ps(0f);
            return X86.Avx.mm256_div_ps(one, X86.Avx.mm256_add_ps(one, ExpX86(X86.Avx.mm256_sub_ps(zero, x))));
        }

        internal static float Max8X86(v256 v)
        {
            v128 halves = X86.Sse.max_ps(
                X86.Avx.mm256_castps256_ps128(v),
                X86.Avx.mm256_extractf128_ps(v, 1));
            v128 pairs = X86.Sse.max_ps(halves, X86.Sse.shuffle_ps(halves, halves, 0x4E));
            v128 quad = X86.Sse.max_ps(pairs, X86.Sse.shuffle_ps(pairs, pairs, 0xB1));
            return X86.Sse.cvtss_f32(quad);
        }

        internal static float Sum256X86(v256 v)
        {
            v128 s = X86.Sse.add_ps(
                X86.Avx.mm256_castps256_ps128(v),
                X86.Avx.mm256_extractf128_ps(v, 1));
            v128 t = X86.Sse3.hadd_ps(s, s);
            t = X86.Sse3.hadd_ps(t, t);
            return X86.Sse.cvtss_f32(t);
        }

        // Mirrors Ops.Softmax exactly (only ever called with n == 8 here, but
        // general like the original: vector peak, scalar tail peak, vector
        // exp-sum, scalar tail exp-sum, broadcast normalize).
        internal static void SoftmaxX86(float* s, int n)
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

        // Mirrors HadamardMlp.ConditionWeights (AVX2 branch): 4 FMA chains, the
        // scalar tail folded into l0, Add(Add(l0,l1),Add(l2,l3)) store, Softmax.
        internal static void ConditionWeightsX86(float* condV, float* input, int dim, float* weights)
        {
            v256 l0 = X86.Avx.mm256_set1_ps(0f);
            v256 l1 = X86.Avx.mm256_set1_ps(0f);
            v256 l2 = X86.Avx.mm256_set1_ps(0f);
            v256 l3 = X86.Avx.mm256_set1_ps(0f);
            int i = 0;
            for (; i + 4 <= dim; i += 4)
            {
                l0 = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_set1_ps(input[i]), X86.Avx.mm256_loadu_ps(condV + i * Conditions), l0);
                l1 = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_set1_ps(input[i + 1]), X86.Avx.mm256_loadu_ps(condV + (i + 1) * Conditions), l1);
                l2 = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_set1_ps(input[i + 2]), X86.Avx.mm256_loadu_ps(condV + (i + 2) * Conditions), l2);
                l3 = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_set1_ps(input[i + 3]), X86.Avx.mm256_loadu_ps(condV + (i + 3) * Conditions), l3);
            }
            for (; i < dim; i++)
                l0 = X86.Fma.mm256_fmadd_ps(X86.Avx.mm256_set1_ps(input[i]), X86.Avx.mm256_loadu_ps(condV + i * Conditions), l0);
            X86.Avx.mm256_storeu_ps(weights, X86.Avx.mm256_add_ps(
                X86.Avx.mm256_add_ps(l0, l1), X86.Avx.mm256_add_ps(l2, l3)));
            SoftmaxX86(weights, Conditions);
        }

        internal static void ScaleInputX86(float* input, float* d1, int dim, int padded, float* destination)
        {
            for (int i = 0; i < dim; i += 8)
                X86.Avx.mm256_storeu_ps(destination + i, X86.Avx.mm256_mul_ps(
                    X86.Avx.mm256_loadu_ps(input + i), X86.Avx.mm256_loadu_ps(d1 + i)));
            for (int i = dim; i < padded; i += 8)
                X86.Avx.mm256_storeu_ps(destination + i, X86.Avx.mm256_set1_ps(0f));
        }

        // Gather through a spill: same lane values as Vector256.Create over 8
        // permuted scalars, using only load/store.
        internal static v256 GatherX86(float* source, int* permutation, float* spill)
        {
            spill[0] = source[permutation[0]];
            spill[1] = source[permutation[1]];
            spill[2] = source[permutation[2]];
            spill[3] = source[permutation[3]];
            spill[4] = source[permutation[4]];
            spill[5] = source[permutation[5]];
            spill[6] = source[permutation[6]];
            spill[7] = source[permutation[7]];
            return X86.Avx.mm256_loadu_ps(spill);
        }

        // Mirrors HadamardMlp.PermuteModulateActivate (AVX2 branch): 8-way
        // condition FMA chain, d2*condition*gather + b2, SiLU.
        internal static void PermuteModulateActivateX86(int* permutation, float* source, float* d2, float* weights, float* condU, float* b2, int width, float* destination)
        {
            v256 w0 = X86.Avx.mm256_set1_ps(weights[0]);
            v256 w1 = X86.Avx.mm256_set1_ps(weights[1]);
            v256 w2 = X86.Avx.mm256_set1_ps(weights[2]);
            v256 w3 = X86.Avx.mm256_set1_ps(weights[3]);
            v256 w4 = X86.Avx.mm256_set1_ps(weights[4]);
            v256 w5 = X86.Avx.mm256_set1_ps(weights[5]);
            v256 w6 = X86.Avx.mm256_set1_ps(weights[6]);
            v256 w7 = X86.Avx.mm256_set1_ps(weights[7]);
            v256 one = X86.Avx.mm256_set1_ps(1f);
            float* u1 = condU + width;
            float* u2 = u1 + width;
            float* u3 = u2 + width;
            float* u4 = u3 + width;
            float* u5 = u4 + width;
            float* u6 = u5 + width;
            float* u7 = u6 + width;
            float* spill = stackalloc float[8];
            for (int i = 0; i < width; i += 8)
            {
                v256 condition = X86.Fma.mm256_fmadd_ps(w0, X86.Avx.mm256_loadu_ps(condU + i), one);
                condition = X86.Fma.mm256_fmadd_ps(w1, X86.Avx.mm256_loadu_ps(u1 + i), condition);
                condition = X86.Fma.mm256_fmadd_ps(w2, X86.Avx.mm256_loadu_ps(u2 + i), condition);
                condition = X86.Fma.mm256_fmadd_ps(w3, X86.Avx.mm256_loadu_ps(u3 + i), condition);
                condition = X86.Fma.mm256_fmadd_ps(w4, X86.Avx.mm256_loadu_ps(u4 + i), condition);
                condition = X86.Fma.mm256_fmadd_ps(w5, X86.Avx.mm256_loadu_ps(u5 + i), condition);
                condition = X86.Fma.mm256_fmadd_ps(w6, X86.Avx.mm256_loadu_ps(u6 + i), condition);
                condition = X86.Fma.mm256_fmadd_ps(w7, X86.Avx.mm256_loadu_ps(u7 + i), condition);
                v256 v = X86.Fma.mm256_fmadd_ps(
                    X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(d2 + i), condition),
                    GatherX86(source, permutation + i, spill),
                    X86.Avx.mm256_loadu_ps(b2 + i));
                X86.Avx.mm256_storeu_ps(destination + i, X86.Avx.mm256_mul_ps(v, SigmoidX86(v)));
            }
        }

        internal static void PermuteScaleX86(int* permutation, float* source, float* scale, int width, float* destination)
        {
            float* spill = stackalloc float[8];
            for (int i = 0; i < width; i += 8)
                X86.Avx.mm256_storeu_ps(destination + i, X86.Avx.mm256_mul_ps(
                    GatherX86(source, permutation + i, spill),
                    X86.Avx.mm256_loadu_ps(scale + i)));
        }

        internal static void AccumulateOutputX86(float* residual, float* product, float* d4, float scale, int count)
        {
            v256 s = X86.Avx.mm256_set1_ps(scale);
            for (int i = 0; i < count; i += 8)
                X86.Avx.mm256_storeu_ps(residual + i, X86.Fma.mm256_fmadd_ps(s,
                    X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(product + i), X86.Avx.mm256_loadu_ps(d4 + i)),
                    X86.Avx.mm256_loadu_ps(residual + i)));
        }

        // Mirrors HadamardMlp.KronAvx2 exactly (4 rows x 2 vectors of
        // accumulators, two phases through a 4 x cols tile, columns unrolled
        // by TileColumns): set1 carries the same broadcast bits as
        // BroadcastScalarToVector256, so every FMA sees identical inputs.
        internal static void KronBlockedX86(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
        {
            float* raw = stackalloc float[TileRows * cols + 8];
            float* tile = (float*)(((nuint)raw + 31) & ~(nuint)31);
            nint stride1 = cols;
            nint stride2 = 2 * cols;
            nint stride3 = 3 * cols;
            for (int k = 0; k < resultRows; k += TileRows)
            {
                for (int c = 0; c < cols; c += TileColumns)
                {
                    v256 a00 = X86.Avx.mm256_set1_ps(0f);
                    v256 a01 = X86.Avx.mm256_set1_ps(0f);
                    v256 a10 = X86.Avx.mm256_set1_ps(0f);
                    v256 a11 = X86.Avx.mm256_set1_ps(0f);
                    v256 a20 = X86.Avx.mm256_set1_ps(0f);
                    v256 a21 = X86.Avx.mm256_set1_ps(0f);
                    v256 a30 = X86.Avx.mm256_set1_ps(0f);
                    v256 a31 = X86.Avx.mm256_set1_ps(0f);
                    float* column = z + c;
                    float* factor = left + k;
                    for (int i = 0; i < activeRows; i++)
                    {
                        v256 z0 = X86.Avx.mm256_loadu_ps(column);
                        v256 z1 = X86.Avx.mm256_loadu_ps(column + 8);
                        v256 l0 = X86.Avx.mm256_set1_ps(factor[0]);
                        a00 = X86.Fma.mm256_fmadd_ps(l0, z0, a00);
                        a01 = X86.Fma.mm256_fmadd_ps(l0, z1, a01);
                        v256 l1 = X86.Avx.mm256_set1_ps(factor[1]);
                        a10 = X86.Fma.mm256_fmadd_ps(l1, z0, a10);
                        a11 = X86.Fma.mm256_fmadd_ps(l1, z1, a11);
                        v256 l2 = X86.Avx.mm256_set1_ps(factor[2]);
                        a20 = X86.Fma.mm256_fmadd_ps(l2, z0, a20);
                        a21 = X86.Fma.mm256_fmadd_ps(l2, z1, a21);
                        v256 l3 = X86.Avx.mm256_set1_ps(factor[3]);
                        a30 = X86.Fma.mm256_fmadd_ps(l3, z0, a30);
                        a31 = X86.Fma.mm256_fmadd_ps(l3, z1, a31);
                        column += cols;
                        factor += rows;
                    }
                    X86.Avx.mm256_storeu_ps(tile + c, a00);
                    X86.Avx.mm256_storeu_ps(tile + c + 8, a01);
                    X86.Avx.mm256_storeu_ps(tile + stride1 + c, a10);
                    X86.Avx.mm256_storeu_ps(tile + stride1 + c + 8, a11);
                    X86.Avx.mm256_storeu_ps(tile + stride2 + c, a20);
                    X86.Avx.mm256_storeu_ps(tile + stride2 + c + 8, a21);
                    X86.Avx.mm256_storeu_ps(tile + stride3 + c, a30);
                    X86.Avx.mm256_storeu_ps(tile + stride3 + c + 8, a31);
                }
                for (int c = 0; c < cols; c += TileColumns)
                {
                    v256 a00 = X86.Avx.mm256_set1_ps(0f);
                    v256 a01 = X86.Avx.mm256_set1_ps(0f);
                    v256 a10 = X86.Avx.mm256_set1_ps(0f);
                    v256 a11 = X86.Avx.mm256_set1_ps(0f);
                    v256 a20 = X86.Avx.mm256_set1_ps(0f);
                    v256 a21 = X86.Avx.mm256_set1_ps(0f);
                    v256 a30 = X86.Avx.mm256_set1_ps(0f);
                    v256 a31 = X86.Avx.mm256_set1_ps(0f);
                    float* row = right + c;
                    float* scalars = tile;
                    for (int j = 0; j < cols; j++)
                    {
                        v256 r0 = X86.Avx.mm256_loadu_ps(row);
                        v256 r1 = X86.Avx.mm256_loadu_ps(row + 8);
                        v256 s0 = X86.Avx.mm256_set1_ps(scalars[0]);
                        a00 = X86.Fma.mm256_fmadd_ps(s0, r0, a00);
                        a01 = X86.Fma.mm256_fmadd_ps(s0, r1, a01);
                        v256 s1 = X86.Avx.mm256_set1_ps(scalars[stride1]);
                        a10 = X86.Fma.mm256_fmadd_ps(s1, r0, a10);
                        a11 = X86.Fma.mm256_fmadd_ps(s1, r1, a11);
                        v256 s2 = X86.Avx.mm256_set1_ps(scalars[stride2]);
                        a20 = X86.Fma.mm256_fmadd_ps(s2, r0, a20);
                        a21 = X86.Fma.mm256_fmadd_ps(s2, r1, a21);
                        v256 s3 = X86.Avx.mm256_set1_ps(scalars[stride3]);
                        a30 = X86.Fma.mm256_fmadd_ps(s3, r0, a30);
                        a31 = X86.Fma.mm256_fmadd_ps(s3, r1, a31);
                        row += cols;
                        scalars++;
                    }
                    float* output = result + k * cols + c;
                    X86.Avx.mm256_storeu_ps(output, a00);
                    X86.Avx.mm256_storeu_ps(output + 8, a01);
                    X86.Avx.mm256_storeu_ps(output + stride1, a10);
                    X86.Avx.mm256_storeu_ps(output + stride1 + 8, a11);
                    X86.Avx.mm256_storeu_ps(output + stride2, a20);
                    X86.Avx.mm256_storeu_ps(output + stride2 + 8, a21);
                    X86.Avx.mm256_storeu_ps(output + stride3, a30);
                    X86.Avx.mm256_storeu_ps(output + stride3 + 8, a31);
                }
            }
        }

        internal static void KronX86(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
        {
            if (rows % TileRows != 0 || cols % TileColumns != 0)
            {
                KronScalar(z, activeRows, left, right, rows, cols, resultRows, result);
                return;
            }
            KronBlockedX86(z, activeRows, left, right, rows, cols, resultRows, result);
        }

        // ---- Neon lane (mirrors the net10 Neon path) ----

        internal static float SumOfSquaresNeon(float* x, int n)
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

        internal static float HorizontalSumNeon(v128 v)
        {
            v128 pairs = Arm.Neon.vpaddq_f32(v, v);
            v128 quad = Arm.Neon.vpaddq_f32(pairs, pairs);
            return Arm.Neon.vgetq_lane_f32(quad, 0);
        }

        // Mirrors Neon.Exp: same clamp, FRINTN round, same polynomial with
        // explicit vfmaq. The two FusedMultiplySubtract steps become vfmaq
        // with negated constants: negation is exact and fma rounds once, so
        // x + n*(-C) is bitwise identical to x - n*C (same trick as
        // BurstSoftmax.ExpNeon). The 2^n scale uses the SIGNED convert/add/
        // shift sequence exactly as Neon.Exp does (ConvertToInt32RoundToEven,
        // +127, <<23, reinterpret through a stack spill).
        internal static v128 ExpNeon(v128 x)
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
            v128 bits = Arm.Neon.vshlq_n_s32(
                Arm.Neon.vaddq_s32(
                    Arm.Neon.vcvtnq_s32_f32(n),
                    Arm.Neon.vdupq_n_s32(127)), 23);
            int* spill = stackalloc int[4];
            Arm.Neon.vst1q_s32(spill, bits);
            v128 scale = Arm.Neon.vld1q_f32((float*)spill);
            return Arm.Neon.vmulq_f32(y, scale);
        }

        // Mirrors Neon.Sigmoid: 1 / (1 + Exp(-x)). The final reciprocal runs
        // per lane through a spill (IEEE fdiv is correctly rounded per lane
        // either way, so this is bitwise identical to a vector divide).
        internal static v128 SigmoidNeon(v128 x)
        {
            v128 neg = Arm.Neon.vsubq_f32(Arm.Neon.vdupq_n_f32(0f), x);
            v128 d = Arm.Neon.vaddq_f32(Arm.Neon.vdupq_n_f32(1f), ExpNeon(neg));
            float* spill = stackalloc float[4];
            Arm.Neon.vst1q_f32(spill, d);
            spill[0] = 1f / spill[0];
            spill[1] = 1f / spill[1];
            spill[2] = 1f / spill[2];
            spill[3] = 1f / spill[3];
            return Arm.Neon.vld1q_f32(spill);
        }

        internal static void SoftmaxNeon(float* s, int n)
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

        internal static void ZeroCenteredRmsNeon(float* x, float* scale, int n, float* destination)
        {
            v128 inverse = Arm.Neon.vdupq_n_f32(1f / MathF.Sqrt(SumOfSquaresNeon(x, n) / n + 1e-6f));
            v128 one = Arm.Neon.vdupq_n_f32(1f);
            for (int i = 0; i < n; i += 4)
                Arm.Neon.vst1q_f32(destination + i, Arm.Neon.vmulq_f32(
                    Arm.Neon.vmulq_f32(
                        Arm.Neon.vaddq_f32(one, Arm.Neon.vld1q_f32(scale + i)),
                        Arm.Neon.vld1q_f32(x + i)),
                    inverse));
        }

        // Mirrors Neon.ConditionWeights: pairwise i += 2 with even/odd FMA
        // chains, Add(even0,odd0) / Add(even1,odd1) stores, Softmax.
        internal static void ConditionWeightsNeon(float* condV, float* input, int dim, float* weights)
        {
            v128 even0 = Arm.Neon.vdupq_n_f32(0f);
            v128 even1 = Arm.Neon.vdupq_n_f32(0f);
            v128 odd0 = Arm.Neon.vdupq_n_f32(0f);
            v128 odd1 = Arm.Neon.vdupq_n_f32(0f);
            for (int i = 0; i < dim; i += 2)
            {
                v128 first = Arm.Neon.vdupq_n_f32(input[i]);
                v128 second = Arm.Neon.vdupq_n_f32(input[i + 1]);
                even0 = Arm.Neon.vfmaq_f32(even0, first, Arm.Neon.vld1q_f32(condV + i * 8));
                even1 = Arm.Neon.vfmaq_f32(even1, first, Arm.Neon.vld1q_f32(condV + i * 8 + 4));
                odd0 = Arm.Neon.vfmaq_f32(odd0, second, Arm.Neon.vld1q_f32(condV + i * 8 + 8));
                odd1 = Arm.Neon.vfmaq_f32(odd1, second, Arm.Neon.vld1q_f32(condV + i * 8 + 12));
            }
            Arm.Neon.vst1q_f32(weights, Arm.Neon.vaddq_f32(even0, odd0));
            Arm.Neon.vst1q_f32(weights + 4, Arm.Neon.vaddq_f32(even1, odd1));
            SoftmaxNeon(weights, Conditions);
        }

        internal static void ScaleInputNeon(float* input, float* d1, int dim, int padded, float* destination)
        {
            for (int i = 0; i < dim; i += 4)
                Arm.Neon.vst1q_f32(destination + i, Arm.Neon.vmulq_f32(
                    Arm.Neon.vld1q_f32(input + i), Arm.Neon.vld1q_f32(d1 + i)));
            for (int i = dim; i < padded; i += 4)
                Arm.Neon.vst1q_f32(destination + i, Arm.Neon.vdupq_n_f32(0f));
        }

        internal static v128 GatherNeon(float* source, int* permutation, float* spill)
        {
            spill[0] = source[permutation[0]];
            spill[1] = source[permutation[1]];
            spill[2] = source[permutation[2]];
            spill[3] = source[permutation[3]];
            return Arm.Neon.vld1q_f32(spill);
        }

        // Mirrors Neon.PermuteModulateActivate.
        internal static void PermuteModulateActivateNeon(int* permutation, float* source, float* d2, float* weights, float* condU, float* b2, int width, float* destination)
        {
            v128 w0 = Arm.Neon.vdupq_n_f32(weights[0]);
            v128 w1 = Arm.Neon.vdupq_n_f32(weights[1]);
            v128 w2 = Arm.Neon.vdupq_n_f32(weights[2]);
            v128 w3 = Arm.Neon.vdupq_n_f32(weights[3]);
            v128 w4 = Arm.Neon.vdupq_n_f32(weights[4]);
            v128 w5 = Arm.Neon.vdupq_n_f32(weights[5]);
            v128 w6 = Arm.Neon.vdupq_n_f32(weights[6]);
            v128 w7 = Arm.Neon.vdupq_n_f32(weights[7]);
            v128 one = Arm.Neon.vdupq_n_f32(1f);
            float* spill = stackalloc float[4];
            for (int i = 0; i < width; i += 4)
            {
                v128 condition = Arm.Neon.vfmaq_f32(one, w0, Arm.Neon.vld1q_f32(condU + i));
                condition = Arm.Neon.vfmaq_f32(condition, w1, Arm.Neon.vld1q_f32(condU + width + i));
                condition = Arm.Neon.vfmaq_f32(condition, w2, Arm.Neon.vld1q_f32(condU + 2 * width + i));
                condition = Arm.Neon.vfmaq_f32(condition, w3, Arm.Neon.vld1q_f32(condU + 3 * width + i));
                condition = Arm.Neon.vfmaq_f32(condition, w4, Arm.Neon.vld1q_f32(condU + 4 * width + i));
                condition = Arm.Neon.vfmaq_f32(condition, w5, Arm.Neon.vld1q_f32(condU + 5 * width + i));
                condition = Arm.Neon.vfmaq_f32(condition, w6, Arm.Neon.vld1q_f32(condU + 6 * width + i));
                condition = Arm.Neon.vfmaq_f32(condition, w7, Arm.Neon.vld1q_f32(condU + 7 * width + i));
                v128 v = Arm.Neon.vfmaq_f32(
                    Arm.Neon.vld1q_f32(b2 + i),
                    Arm.Neon.vmulq_f32(Arm.Neon.vld1q_f32(d2 + i), condition),
                    GatherNeon(source, permutation + i, spill));
                Arm.Neon.vst1q_f32(destination + i, Arm.Neon.vmulq_f32(v, SigmoidNeon(v)));
            }
        }

        internal static void PermuteScaleNeon(int* permutation, float* source, float* scale, int width, float* destination)
        {
            float* spill = stackalloc float[4];
            for (int i = 0; i < width; i += 4)
                Arm.Neon.vst1q_f32(destination + i, Arm.Neon.vmulq_f32(
                    GatherNeon(source, permutation + i, spill),
                    Arm.Neon.vld1q_f32(scale + i)));
        }

        internal static void AccumulateOutputNeon(float* residual, float* product, float* d4, float scale, int count)
        {
            v128 s = Arm.Neon.vdupq_n_f32(scale);
            for (int i = 0; i < count; i += 4)
                Arm.Neon.vst1q_f32(residual + i, Arm.Neon.vfmaq_f32(
                    Arm.Neon.vld1q_f32(residual + i), s,
                    Arm.Neon.vmulq_f32(Arm.Neon.vld1q_f32(product + i), Arm.Neon.vld1q_f32(d4 + i))));
        }

        // Mirrors Neon.Kron (4 rows x 4 vectors of accumulators, two phases
        // through a 4 x cols tile, columns unrolled by 16): vdup carries the
        // same broadcast bits as LoadAndReplicateToVector128.
        internal static void KronBlockedNeon(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
        {
            float* tile = stackalloc float[4 * cols];
            nint stride1 = cols;
            nint stride2 = 2 * cols;
            nint stride3 = 3 * cols;
            for (int k = 0; k < resultRows; k += 4)
            {
                for (int c = 0; c < cols; c += 16)
                {
                    v128 a00 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a01 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a02 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a03 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a10 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a11 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a12 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a13 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a20 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a21 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a22 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a23 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a30 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a31 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a32 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a33 = Arm.Neon.vdupq_n_f32(0f);
                    float* column = z + c;
                    float* factor = left + k;
                    for (int i = 0; i < activeRows; i++)
                    {
                        v128 z0 = Arm.Neon.vld1q_f32(column);
                        v128 z1 = Arm.Neon.vld1q_f32(column + 4);
                        v128 z2 = Arm.Neon.vld1q_f32(column + 8);
                        v128 z3 = Arm.Neon.vld1q_f32(column + 12);
                        v128 l0 = Arm.Neon.vdupq_n_f32(factor[0]);
                        a00 = Arm.Neon.vfmaq_f32(a00, l0, z0);
                        a01 = Arm.Neon.vfmaq_f32(a01, l0, z1);
                        a02 = Arm.Neon.vfmaq_f32(a02, l0, z2);
                        a03 = Arm.Neon.vfmaq_f32(a03, l0, z3);
                        v128 l1 = Arm.Neon.vdupq_n_f32(factor[1]);
                        a10 = Arm.Neon.vfmaq_f32(a10, l1, z0);
                        a11 = Arm.Neon.vfmaq_f32(a11, l1, z1);
                        a12 = Arm.Neon.vfmaq_f32(a12, l1, z2);
                        a13 = Arm.Neon.vfmaq_f32(a13, l1, z3);
                        v128 l2 = Arm.Neon.vdupq_n_f32(factor[2]);
                        a20 = Arm.Neon.vfmaq_f32(a20, l2, z0);
                        a21 = Arm.Neon.vfmaq_f32(a21, l2, z1);
                        a22 = Arm.Neon.vfmaq_f32(a22, l2, z2);
                        a23 = Arm.Neon.vfmaq_f32(a23, l2, z3);
                        v128 l3 = Arm.Neon.vdupq_n_f32(factor[3]);
                        a30 = Arm.Neon.vfmaq_f32(a30, l3, z0);
                        a31 = Arm.Neon.vfmaq_f32(a31, l3, z1);
                        a32 = Arm.Neon.vfmaq_f32(a32, l3, z2);
                        a33 = Arm.Neon.vfmaq_f32(a33, l3, z3);
                        column += cols;
                        factor += rows;
                    }
                    Arm.Neon.vst1q_f32(tile + c, a00);
                    Arm.Neon.vst1q_f32(tile + c + 4, a01);
                    Arm.Neon.vst1q_f32(tile + c + 8, a02);
                    Arm.Neon.vst1q_f32(tile + c + 12, a03);
                    Arm.Neon.vst1q_f32(tile + stride1 + c, a10);
                    Arm.Neon.vst1q_f32(tile + stride1 + c + 4, a11);
                    Arm.Neon.vst1q_f32(tile + stride1 + c + 8, a12);
                    Arm.Neon.vst1q_f32(tile + stride1 + c + 12, a13);
                    Arm.Neon.vst1q_f32(tile + stride2 + c, a20);
                    Arm.Neon.vst1q_f32(tile + stride2 + c + 4, a21);
                    Arm.Neon.vst1q_f32(tile + stride2 + c + 8, a22);
                    Arm.Neon.vst1q_f32(tile + stride2 + c + 12, a23);
                    Arm.Neon.vst1q_f32(tile + stride3 + c, a30);
                    Arm.Neon.vst1q_f32(tile + stride3 + c + 4, a31);
                    Arm.Neon.vst1q_f32(tile + stride3 + c + 8, a32);
                    Arm.Neon.vst1q_f32(tile + stride3 + c + 12, a33);
                }
                for (int c = 0; c < cols; c += 16)
                {
                    v128 a00 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a01 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a02 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a03 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a10 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a11 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a12 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a13 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a20 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a21 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a22 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a23 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a30 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a31 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a32 = Arm.Neon.vdupq_n_f32(0f);
                    v128 a33 = Arm.Neon.vdupq_n_f32(0f);
                    float* row = right + c;
                    float* scalars = tile;
                    for (int j = 0; j < cols; j++)
                    {
                        v128 r0 = Arm.Neon.vld1q_f32(row);
                        v128 r1 = Arm.Neon.vld1q_f32(row + 4);
                        v128 r2 = Arm.Neon.vld1q_f32(row + 8);
                        v128 r3 = Arm.Neon.vld1q_f32(row + 12);
                        v128 s0 = Arm.Neon.vdupq_n_f32(scalars[0]);
                        a00 = Arm.Neon.vfmaq_f32(a00, s0, r0);
                        a01 = Arm.Neon.vfmaq_f32(a01, s0, r1);
                        a02 = Arm.Neon.vfmaq_f32(a02, s0, r2);
                        a03 = Arm.Neon.vfmaq_f32(a03, s0, r3);
                        v128 s1 = Arm.Neon.vdupq_n_f32(scalars[stride1]);
                        a10 = Arm.Neon.vfmaq_f32(a10, s1, r0);
                        a11 = Arm.Neon.vfmaq_f32(a11, s1, r1);
                        a12 = Arm.Neon.vfmaq_f32(a12, s1, r2);
                        a13 = Arm.Neon.vfmaq_f32(a13, s1, r3);
                        v128 s2 = Arm.Neon.vdupq_n_f32(scalars[stride2]);
                        a20 = Arm.Neon.vfmaq_f32(a20, s2, r0);
                        a21 = Arm.Neon.vfmaq_f32(a21, s2, r1);
                        a22 = Arm.Neon.vfmaq_f32(a22, s2, r2);
                        a23 = Arm.Neon.vfmaq_f32(a23, s2, r3);
                        v128 s3 = Arm.Neon.vdupq_n_f32(scalars[stride3]);
                        a30 = Arm.Neon.vfmaq_f32(a30, s3, r0);
                        a31 = Arm.Neon.vfmaq_f32(a31, s3, r1);
                        a32 = Arm.Neon.vfmaq_f32(a32, s3, r2);
                        a33 = Arm.Neon.vfmaq_f32(a33, s3, r3);
                        row += cols;
                        scalars++;
                    }
                    float* output = result + k * cols + c;
                    Arm.Neon.vst1q_f32(output, a00);
                    Arm.Neon.vst1q_f32(output + 4, a01);
                    Arm.Neon.vst1q_f32(output + 8, a02);
                    Arm.Neon.vst1q_f32(output + 12, a03);
                    Arm.Neon.vst1q_f32(output + stride1, a10);
                    Arm.Neon.vst1q_f32(output + stride1 + 4, a11);
                    Arm.Neon.vst1q_f32(output + stride1 + 8, a12);
                    Arm.Neon.vst1q_f32(output + stride1 + 12, a13);
                    Arm.Neon.vst1q_f32(output + stride2, a20);
                    Arm.Neon.vst1q_f32(output + stride2 + 4, a21);
                    Arm.Neon.vst1q_f32(output + stride2 + 8, a22);
                    Arm.Neon.vst1q_f32(output + stride2 + 12, a23);
                    Arm.Neon.vst1q_f32(output + stride3, a30);
                    Arm.Neon.vst1q_f32(output + stride3 + 4, a31);
                    Arm.Neon.vst1q_f32(output + stride3 + 8, a32);
                    Arm.Neon.vst1q_f32(output + stride3 + 12, a33);
                }
            }
        }

        internal static void KronNeon(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
        {
            if (rows % TileRows != 0 || cols % TileColumns != 0)
            {
                KronScalar(z, activeRows, left, right, rows, cols, resultRows, result);
                return;
            }
            KronBlockedNeon(z, activeRows, left, right, rows, cols, resultRows, result);
        }

        // ---- Scalar lane (mirrors Scalar.* op for op) ----

        internal static float SigmoidScalar(float x)
        {
            return 1f / (1f + MathF.Exp(-x));
        }

        internal static void SoftmaxScalar(float* s, int n)
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

        internal static void ZeroCenteredRmsScalar(float* x, float* scale, int n, float* destination)
        {
            float sum = 0f;
            for (int i = 0; i < n; i++) sum += x[i] * x[i];
            float inverse = 1f / MathF.Sqrt(sum / n + 1e-6f);
            for (int i = 0; i < n; i++) destination[i] = (1f + scale[i]) * x[i] * inverse;
        }

        internal static void ConditionWeightsScalar(float* condV, float* input, int dim, float* weights)
        {
            for (int j = 0; j < Conditions; j++)
            {
                float sum = 0f;
                for (int i = 0; i < dim; i++) sum += input[i] * condV[i * Conditions + j];
                weights[j] = sum;
            }
            SoftmaxScalar(weights, Conditions);
        }

        internal static void ScaleInputScalar(float* input, float* d1, int dim, int padded, float* destination)
        {
            for (int i = 0; i < dim; i++) destination[i] = input[i] * d1[i];
            for (int i = dim; i < padded; i++) destination[i] = 0f;
        }

        internal static void PermuteModulateActivateScalar(int* permutation, float* source, float* d2, float* weights, float* condU, float* b2, int width, float* destination)
        {
            for (int n = 0; n < width; n++)
            {
                float condition = 1f;
                for (int j = 0; j < Conditions; j++) condition += weights[j] * condU[j * width + n];
                float v = d2[n] * condition * source[permutation[n]] + b2[n];
                destination[n] = v * SigmoidScalar(v);
            }
        }

        internal static void PermuteScaleScalar(int* permutation, float* source, float* scale, int width, float* destination)
        {
            for (int n = 0; n < width; n++) destination[n] = source[permutation[n]] * scale[n];
        }

        internal static void AccumulateOutputScalar(float* residual, float* product, float* d4, float scale, int count)
        {
            for (int i = 0; i < count; i++) residual[i] += scale * (product[i] * d4[i]);
        }

        // Mirrors Scalar.Kron exactly (scratch per k row, i-order then j-order
        // sums); also the ragged-geometry fallback for the vector Krons above.
        internal static void KronScalar(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
        {
            float* scratch = stackalloc float[cols];
            for (int k = 0; k < resultRows; k++)
            {
                for (int c = 0; c < cols; c++)
                {
                    float sum = 0f;
                    for (int i = 0; i < activeRows; i++) sum += left[i * rows + k] * z[i * cols + c];
                    scratch[c] = sum;
                }
                for (int c = 0; c < cols; c++)
                {
                    float sum = 0f;
                    for (int j = 0; j < cols; j++) sum += scratch[j] * right[j * cols + c];
                    result[k * cols + c] = sum;
                }
            }
        }
    }

    // N37 provider: installs the Burst-compiled HadamardMlp entry into
    // Needle.Kernels.Active so every HadamardMlp.Apply dispatch on an ns2.1
    // build (CoreCLR Editor, IL2CPP and Mono players) runs the Burst lane
    // instead of Scalar. Net10 builds keep their direct-SIMD branches ahead of
    // the table, so this is a no-op there beyond the Describe string.
    // Never overwrites an explicitly installed HadamardMlp entry (plan section
    // 2 selection order: explicit wins).
    //
    // C# 9 safe: block namespace, no target-typed new.
    public static unsafe class BurstMlpInstaller
    {
        // Cdecl delegate for the table slot: void(MlpBatch*, float*, float*).
        // The UnmanagedFunctionPointer attribute pins the calling convention
        // so the compiled pointer matches the KernelTable ABI
        // (delegate* unmanaged[Cdecl]<MlpBatch*, float*, float*, void>).
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void HadamardMlpFn(BurstMlpBatch* b, float* residual, float* scratch);

        static bool installAttempted;

        public static bool IsInstalled => Needle.Kernels.Active.HadamardMlp != 0;

        // Explicit entry point for the N37 acceptance ("Expose Install()").
        public static void Install()
        {
            EnsureInstalled();
        }

        // Idempotent: safe from the auto-install hook below, from the Editor
        // bootstrap, and from tests. Never overwrites an explicit or foreign
        // provider's HadamardMlp entry (plan section 2 selection order).
        public static void EnsureInstalled()
        {
            if (Needle.Kernels.Active.HadamardMlp != 0)
            {
                installAttempted = true;
                return;
            }
            FunctionPointer<HadamardMlpFn> compiled =
                BurstCompiler.CompileFunctionPointer<HadamardMlpFn>(BurstMlp.Invoke);
            if (!compiled.IsCreated)
                throw new InvalidOperationException(
                    "N37: Burst refused to compile BurstMlp.Invoke. " +
                    "The kernel must live in package source (H6); " +
                    "a precompiled DLL entry cannot be compiled.");
            KernelTable table = Needle.Kernels.Active;
            table.HadamardMlp = (nint)compiled.Value;
            Needle.Kernels.Install(table, Needle.Kernels.IsInstalled ? Needle.Kernels.Provider + "+burst-mlp" : "burst-mlp");
            installAttempted = true;
        }

        // Player + play-mode entry: BeforeSceneLoad so the table is ready
        // before Model.Open runs. Guarded: inference must survive a missing
        // Burst toolchain (falls back to Scalar, logs once).
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
                Debug.LogWarning("[needle] Burst MLP kernel install failed, using Scalar: " + e.Message);
            }
        }

        // One-line startup report: tier/provider plus the Burst lane flags
        // the installed binary actually compiled for.
        public static string Describe()
        {
            string lane = Unity.Burst.Intrinsics.X86.Avx2.IsAvx2Supported &&
                Unity.Burst.Intrinsics.X86.Fma.IsFmaSupported ? "x86-avx2-fma"
                : Unity.Burst.Intrinsics.Arm.Neon.IsNeonSupported ? "arm-neon" : "scalar";
            return NeedleRuntime.Describe() + " burstMlpLane=" + lane
                + " burstMlpInstalled=" + (IsInstalled ? "1" : "0");
        }

        // N37 microbench for the acceptance row (µs per HadamardMlp.Apply):
        // times the installed function pointer on synthetic weights of the
        // given geometry. Call from an EditMode test or the Probe scene on the
        // target device; compare against the net10 bench-e2e Apply number.
        // Returns microseconds per Apply.
        public static double MeasureApplyMicroseconds(int dim, int rows, int cols, int calls = 200)
        {
            if (dim < 8 || rows < 1 || cols < 16 || calls < 1)
                throw new ArgumentOutOfRangeException("geometry");
            EnsureInstalled();
            delegate* unmanaged[Cdecl]<BurstMlpBatch*, float*, float*, void> fp =
                (delegate* unmanaged[Cdecl]<BurstMlpBatch*, float*, float*, void>)(void*)Needle.Kernels.Active.HadamardMlp;
            int width = rows * cols;
            int inputRows = (dim + cols - 1) / cols;
            int scratchFloats = ConditionsMirror + dim + 2 * width + 8;
            float[] residual = new float[dim];
            float[] scratch = new float[scratchFloats];
            float[][] held = new float[14][];
            BurstMlpWeights weights = default;
            float** slots = stackalloc float*[14];
            uint state = 0x9E3779B9u;
            for (int s = 0; s < 14; s++)
            {
                int count = s == 0 ? dim : s <= 4 ? (s == 4 ? dim : width) : s <= 10 ? (s % 2 == 1 ? rows * rows : cols * cols) : s == 11 ? dim * 8 : 8 * width;
                if (s == 5 || s == 7 || s == 9) count = System.Math.Max(count, inputRows * rows);
                held[s] = new float[count];
                for (int i = 0; i < count; i++)
                {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    held[s][i] = ((state >> 8) & 0xFFFFFF) / (float)0xFFFFFF * 0.6f - 0.3f + (s <= 4 ? 1f : 0f);
                }
            }
            fixed (float* p0 = held[0])
            fixed (float* p1 = held[1])
            fixed (float* p2 = held[2])
            fixed (float* p3 = held[3])
            fixed (float* p4 = held[4])
            fixed (float* p5 = held[5])
            fixed (float* p6 = held[6])
            fixed (float* p7 = held[7])
            fixed (float* p8 = held[8])
            fixed (float* p9 = held[9])
            fixed (float* p10 = held[10])
            fixed (float* p11 = held[11])
            fixed (float* p12 = held[12])
            fixed (float* p13 = held[13])
            {
                slots[0] = p0; slots[1] = p1; slots[2] = p2; slots[3] = p3; slots[4] = p4;
                slots[5] = p5; slots[6] = p6; slots[7] = p7; slots[8] = p8; slots[9] = p9;
                slots[10] = p10; slots[11] = p11; slots[12] = p12; slots[13] = p13;
                weights.D1 = slots[0];
                weights.D2 = slots[1];
                weights.B2 = slots[2];
                weights.D3 = slots[3];
                weights.D4 = slots[4];
                weights.W1A = slots[5];
                weights.W1B = slots[6];
                weights.W2A = slots[7];
                weights.W2B = slots[8];
                weights.W3A = slots[9];
                weights.W3B = slots[10];
                weights.CondV = slots[11];
                weights.CondU = slots[12];
                int[] permA = new int[width];
                int[] permB = new int[width];
                for (int i = 0; i < width; i++) { permA[i] = i; permB[i] = i; }
                for (int i = width - 1; i > 0; i--)
                {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    int j = (int)(state % (uint)(i + 1));
                    int t = permA[i]; permA[i] = permA[j]; permA[j] = t;
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    j = (int)(state % (uint)(i + 1));
                    t = permB[i]; permB[i] = permB[j]; permB[j] = t;
                }
                float[] preNorm = new float[dim];
                for (int i = 0; i < dim; i++) preNorm[i] = 0.05f * ((i % 7) - 3);
                for (int i = 0; i < dim; i++) residual[i] = 0.1f * ((i % 11) - 5);
                fixed (float* pres = residual)
                fixed (float* scr = scratch)
                fixed (int* pa = permA)
                fixed (int* pb = permB)
                fixed (float* pn = preNorm)
                {
                    BurstMlpBatch batch = new BurstMlpBatch
                    {
                        Weights = &weights,
                        PreNorm = pn,
                        PermA = pa,
                        PermB = pb,
                        Residual = pres,
                        Dim = dim,
                        Rows = rows,
                        Cols = cols,
                        Scale = 1f,
                    };
                    // Warmup: trigger Burst JIT/AOT page-in off the clock.
                    for (int w = 0; w < 10; w++) fp(&batch, pres, scr);
                    System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
                    for (int c = 0; c < calls; c++) fp(&batch, pres, scr);
                    stopwatch.Stop();
                    double usPerCall = (double)stopwatch.ElapsedTicks * 1000000.0 / System.Diagnostics.Stopwatch.Frequency / calls;
                    return usPerCall;
                }
            }
        }

        // Local alias so the microbench does not depend on the private
        // Conditions constant of BurstMlp.
        const int ConditionsMirror = 8;
    }
}
