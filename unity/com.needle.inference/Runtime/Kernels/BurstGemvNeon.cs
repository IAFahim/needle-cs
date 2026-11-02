// N34: Burst port of Neon.MatMulRows, the Arm64 integer-dot GEMV path.
//
// Ports src/Needle/Neon.cs (DotBlock, DotPair, DotTile, DotRow, GroupDot,
// TilePlane, BlockPlane, PairPlane, BlockToken, MatMulRows, plus the shared
// Dot16, LoadFactors, ScaleFactors and HorizontalSum helpers) 1:1 to
// Unity.Burst.Intrinsics. The x86 counterpart strategy (N32/N33) is the same:
// keep the instruction order identical to the net10 kernel; that is what
// makes H7 (bitwise parity) achievable.
//
// Plan cookbook mapping (net10 -> Burst), every name verified against the
// Burst 1.8.25 intrinsics sources (the API surface the installed Burst 2.0
// keeps for core Neon; the N31 kernel already proves the v128/v64,
// vld1q/vst1q, vfmaq_f32, vpaddq_f32 and vshlq_n_u32 members below):
//   AdvSimd.LoadVector128/Store (u8/s8/f32) -> vld1q_u8/vld1q_s8/vld1q_f32,
//     vst1q_f32; LoadVector64(u16) -> vld1_u16.
//   Dp.DotProduct(acc, a, b) -> Arm.Neon.vdotq_s32(acc, a, b), guarded by
//     Arm.Neon.IsNeonDotProdSupported (both paths in this file; the guard
//     folds at Burst compile time per target, so a dotprod player runs the
//     SDOT lane and a baseline-arm64 player runs the widening lane).
//   Widening fallback: SignExtendWideningLower(v64) -> vmovl_s8(vget_low_s8),
//     SignExtendWideningUpper(v128) -> vmovl_high_s8, MultiplyWideningLowerAndAdd
//     -> vmlal_s16(sum, vget_low_s16, vget_low_s16), MultiplyWideningUpperAndAdd
//     -> vmlal_high_s16. Integer products/accumulation are exact mod 2^32, so
//     the fallback and the SDOT lane agree bit for bit (same as net10).
//   VectorTableLookup single-table + v64 index, twice (low/high halves) ->
//     one vqtbl1q_u8(table, index): one TBL.16B over the same 16-byte table
//     with the same 16 indices is lane-for-lane identical to two TBL.8B
//     lookups; the And(mask) stays at the call sites exactly as in net10.
//   AdvSimd.And/ShiftRightLogical on bytes -> vandq_u8/vshrq_n_u8.
//   LoadFactors u32 bit rebuild (And/Shl/Add/CompareGreaterThan/Or/AsSingle)
//     -> vandq_u32/vshlq_n_u32/vaddq_u32/vcgtq_s32/vorrq_u32 plus a 16-byte
//     stack spill for the u32->f32 reinterpret (there is no bit-reinterpret
//     intrinsic; N31 uses the same spill in ExpNeon and it preserves bits
//     exactly). The vector path intentionally mirrors Neon's subnormal-blind
//     rebuild, and the g-tail calls HalfToSingle, exactly like net10.
//   AdvSimd.ConvertToSingle -> vcvtq_f32_s32; AddPairwise float fold ->
//     two vpaddq_f32 + vgetq_lane_f32(..., 0) (same (l0+l1)+(l2+l3) order as
//     Neon.HorizontalSum; N31 proves the mapping).
//   AdvSimd.FusedMultiplyAdd(acc, a, b) -> vfmaq_f32(acc, a, b);
//     Multiply -> vmulq_f32; ZeroExtendWideningLower(u16) -> vmovl_u16.
//
// H7 note: the Neon lane issues the same operations in the same order as
// Neon.MatMulRows, so fixtures compare bitwise on device. Residual,
// explicitly out-of-scope risks: none known inside this port — every float
// op is an f32 add/mul/fma in net10 order under FloatMode.Strict, every int
// op is exact, and NaN payloads travel the same pairwise tree on the same
// architecture. The scalar lane mirrors Scalar.MatMulRows for targets
// without Neon (and doubles as the host-readable reference for N13-style
// edge review: tokens/rows tails, 2-bit vs 4-bit, subnormal norms).
//
// H8 note: this file needs no AVX-VNNI and no AVX-512; it is Neon-only plus
// a scalar tail, like the N31 kernel.
//
// Layout contract: Cq/Activations/Codebook below are field-for-field,
// order-for-order mirrors of the internal Needle.Cq, Needle.Activations and
// Needle.Codebook structs in the precompiled Needle.Inference.dll (see
// src/Needle/Model.cs and src/Needle/Quant.cs). C# structs default to
// sequential layout, so the mirrors are ABI-identical and the installed
// KernelTable.MatMulRows function pointer can be called with the core's
// (Cq*, Activations*, tokens, y, ldy, rowStart, rowEnd) signature directly.
// Only valid where pointers are 8 bytes, which holds everywhere this Neon
// lane can execute (the entry dispatches to scalar off Neon).
//
// Burst rules followed: FloatMode.Strict, explicit FMA, no [NoAlias] claims,
// blittable arguments only, no allocation (rowFactors/scaled are caller-sized
// stackallocs exactly as in net10), no static mutable state, safe from any
// thread (pool workers call via the table).
//
// C# 9 safe: block namespace, no target-typed new, no static lambdas.
using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using UnityEngine;

namespace Needle.BurstKernels
{
    public static unsafe class BurstGemvNeon
    {
        // Field-for-field mirror of Needle.Codebook (src/Needle/Model.cs):
        // fixed float Levels[16] @0, fixed byte Bytes[32] @64, float @96.
        // Plain (non-readonly) struct: C# forbids readonly fixed buffers,
        // and readonly structs require all-readonly fields. Only ever used
        // through Codebook*, never copied by value.
        public unsafe struct Codebook
        {
            public fixed float Levels[16];
            public fixed byte Bytes[32];
            public float ByteScale;
        }

        // Field-for-field mirror of Needle.Cq (src/Needle/Model.cs).
        public readonly unsafe struct Cq
        {
            public const int GroupSize = 128;

            public readonly byte* Indices;
            public readonly ushort* Norms;
            public readonly Codebook* Book;
            public readonly int Rows;
            public readonly int Cols;
            public readonly int Bits;

            public Cq(byte* indices, ushort* norms, Codebook* book, int rows, int cols, int bits)
            {
                Indices = indices;
                Norms = norms;
                Book = book;
                Rows = rows;
                Cols = cols;
                Bits = bits;
            }

            public int Groups => Cols / GroupSize;
            public int RowBytes => Cols * Bits / 8;
        }

        // Field-for-field mirror of Needle.Activations (src/Needle/Quant.cs).
        public readonly unsafe struct Activations
        {
            public readonly sbyte* Values;
            public readonly float* Scales;
            public readonly float* Offsets;
            public readonly int Cols;
            public readonly int Groups;
            public readonly int Stride;

            public Activations(sbyte* values, float* scales, float* offsets, int cols, int groups, int stride)
            {
                Values = values;
                Scales = scales;
                Offsets = offsets;
                Cols = cols;
                Groups = groups;
                Stride = stride;
            }
        }

        // Cdecl delegate for the table slot, mirroring KernelTable.MatMulRows:
        // void(Cq*, Activations*, int tokens, float* y, int ldy, int rowStart, int rowEnd).
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void MatMulRowsFn(Cq* w, Activations* a, int tokens, float* y, int ldy, int rowStart, int rowEnd);

        // Entry point installed into KernelTable.MatMulRows.
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void Invoke(Cq* w, Activations* a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
        {
            // Branches fold at Burst compile time: exactly one lane survives
            // per target (arm64 Neon, scalar elsewhere), as in BurstSoftmax.
            if (Arm.Neon.IsNeonSupported) { MatMulRows(*w, *a, tokens, y, ldy, rowStart, rowEnd); return; }
            ScalarMatMulRows(*w, *a, tokens, y, ldy, rowStart, rowEnd);
        }

        // Idempotent: safe from the auto-install hook below and from tests.
        // Never overwrites an explicit or foreign provider's MatMulRows entry
        // (plan §2 selection order: explicit install always wins).
        public static void Install()
        {
            if (Kernels.Active.MatMulRows != 0) return;
            FunctionPointer<MatMulRowsFn> compiled =
                BurstCompiler.CompileFunctionPointer<MatMulRowsFn>(Invoke);
            if (!compiled.IsCreated)
                throw new InvalidOperationException(
                    "N34: Burst refused to compile BurstGemvNeon.Invoke. " +
                    "The kernel must live in package source; " +
                    "a precompiled DLL entry cannot be compiled.");
            KernelTable table = Kernels.Active;
            table.MatMulRows = (nint)compiled.Value;
            Kernels.Install(table, Kernels.IsInstalled ? Kernels.Provider + "+burst" : "burst");
        }

        // Player + play-mode entry: BeforeSceneLoad so the table is ready
        // before Model.Open runs. Guarded: inference must survive a missing
        // Burst toolchain (falls back to Scalar, logs once).
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void AutoInstall()
        {
            try
            {
                Install();
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning("[needle] Burst GEMV-Neon install failed, using Scalar: " + e.Message);
            }
        }

        // Mirrors Neon.HorizontalSum: two pairwise adds, lane 0 out.
        // vpaddq_f32(v,v) = [l0+l1, l2+l3, l0+l1, l2+l3] matches AddPairwise
        // lane for lane, so the second fold lands on (l0+l1)+(l2+l3), the
        // same sum in the same order.
        static float HorizontalSumNeon(v128 v)
        {
            v128 pairs = Arm.Neon.vpaddq_f32(v, v);
            v128 quad = Arm.Neon.vpaddq_f32(pairs, pairs);
            return Arm.Neon.vgetq_lane_f32(quad, 0);
        }

        // Bit-identical copy of Scalar.HalfToSingle (src/Needle/Scalar.cs),
        // used by the LoadFactors group tail exactly as in Neon.LoadFactors.
        // The u32->f32 reinterpret goes through a 4-byte stack spill (the
        // N31 ExpNeon pattern): a memory round-trip preserves bits exactly.
        static float HalfToSingle(ushort half)
        {
            var magnitude = half & 0x7FFF;
            if (magnitude == 0) return 0f;
            if (magnitude < 0x0400) return (half & 0x8000) != 0 ? -magnitude * 5.9604645e-8f : magnitude * 5.9604645e-8f;
            uint bits = ((uint)magnitude << 13) + 0x38000000u | (uint)(half & 0x8000) << 16;
            uint* spill = stackalloc uint[1];
            *spill = bits;
            return *(float*)spill;
        }

        // Mirrors Neon.Dot16, both paths: SDOT when the target has dotprod,
        // the vmovl/vmlal widening sequence otherwise. Integer arithmetic is
        // exact mod 2^32, so both lanes agree bit for bit (as in net10).
        static v128 Dot16(v128 sum, v128 weights, v128 values)
        {
            if (Arm.Neon.IsNeonDotProdSupported) return Arm.Neon.vdotq_s32(sum, weights, values);
            v128 weightLow = Arm.Neon.vmovl_s8(Arm.Neon.vget_low_s8(weights));
            v128 weightHigh = Arm.Neon.vmovl_high_s8(weights);
            v128 valueLow = Arm.Neon.vmovl_s8(Arm.Neon.vget_low_s8(values));
            v128 valueHigh = Arm.Neon.vmovl_high_s8(values);
            sum = Arm.Neon.vmlal_s16(sum, Arm.Neon.vget_low_s16(valueLow), Arm.Neon.vget_low_s16(weightLow));
            sum = Arm.Neon.vmlal_high_s16(sum, valueLow, weightLow);
            sum = Arm.Neon.vmlal_s16(sum, Arm.Neon.vget_low_s16(valueHigh), Arm.Neon.vget_low_s16(weightHigh));
            return Arm.Neon.vmlal_high_s16(sum, valueHigh, weightHigh);
        }

        // Mirrors Neon.LoadFactors: half-norm vector path (u32 bit rebuild,
        // subnormal-blind exactly like net10) plus the HalfToSingle tail.
        static void LoadFactors(ushort* norms, int groups, float scale, float* factors)
        {
            v128 multiplier = Arm.Neon.vdupq_n_f32(scale);
            var g = 0;
            for (; g + 4 <= groups; g += 4)
            {
                v128 wide = Arm.Neon.vmovl_u16(Arm.Neon.vld1_u16(norms + g));
                v128 magnitude = Arm.Neon.vandq_u32(wide, Arm.Neon.vdupq_n_u32(0x7FFFu));
                v128 sign = Arm.Neon.vshlq_n_u32(Arm.Neon.vandq_u32(wide, Arm.Neon.vdupq_n_u32(0x8000u)), 16);
                v128 bits = Arm.Neon.vaddq_u32(Arm.Neon.vshlq_n_u32(magnitude, 13), Arm.Neon.vdupq_n_u32(0x38000000u));
                v128 nonZero = Arm.Neon.vcgtq_s32(magnitude, Arm.Neon.vdupq_n_s32(0));
                v128 combined = Arm.Neon.vorrq_u32(Arm.Neon.vandq_u32(bits, nonZero), sign);
                uint* spill = stackalloc uint[4];
                Arm.Neon.vst1q_u32(spill, combined);
                v128 single = Arm.Neon.vld1q_f32((float*)spill);
                Arm.Neon.vst1q_f32(factors + g, Arm.Neon.vmulq_f32(single, multiplier));
            }
            for (; g < groups; g++) factors[g] = HalfToSingle(norms[g]) * scale;
        }

        // Mirrors Neon.ScaleFactors: scaled[v] = factors[v]*scales[v],
        // bias = Σ scaled[v]*offsets[v] via explicit vfmaq (Strict: no
        // reassociation), folded with HorizontalSumNeon.
        static float ScaleFactors(float* factors, float* scaled, float* scales, float* offsets, int stride)
        {
            v128 bias = Arm.Neon.vdupq_n_f32(0f);
            for (var v = 0; v < stride; v += 4)
            {
                v128 f = Arm.Neon.vmulq_f32(Arm.Neon.vld1q_f32(factors + v), Arm.Neon.vld1q_f32(scales + v));
                Arm.Neon.vst1q_f32(scaled + v, f);
                bias = Arm.Neon.vfmaq_f32(bias, f, Arm.Neon.vld1q_f32(offsets + v));
            }
            return HorizontalSumNeon(bias);
        }

        // Mirrors Neon.Weights: one TBL.16B over the same 16-byte codebook
        // with the same 16 indices. net10 issues two TBL.8B half-lookups and
        // recombines; the single full-width lookup is lane-for-lane
        // identical (out-of-range lanes cannot occur: indices are 2/4-bit).
        static v128 Weights(v128 table, v128 index)
        {
            return Arm.Neon.vqtbl1q_u8(table, index);
        }

        // Mirrors Neon.GroupDot (single-row × single-token dot over groups).
        // net10 derives `two` from mask lane 0; mask itself derives from
        // w.Bits, so branching on w.Bits directly is the same decision.
        static v128 GroupDot(byte* packed, sbyte* values, v128 mask, v128 table, int bits)
        {
            v128 sum = Arm.Neon.vdupq_n_s32(0);
            if (bits == 2)
            {
                for (var r = 0; r < 2; r++)
                {
                    v128 source = Arm.Neon.vld1q_u8(packed + r * 16);
                    sbyte* act = values + r * 16;
                    sum = Dot16(sum, Weights(table, Arm.Neon.vandq_u8(source, mask)), Arm.Neon.vld1q_s8(act));
                    sum = Dot16(sum, Weights(table, Arm.Neon.vandq_u8(Arm.Neon.vshrq_n_u8(source, 2), mask)), Arm.Neon.vld1q_s8(act + 32));
                    sum = Dot16(sum, Weights(table, Arm.Neon.vandq_u8(Arm.Neon.vshrq_n_u8(source, 4), mask)), Arm.Neon.vld1q_s8(act + 64));
                    sum = Dot16(sum, Weights(table, Arm.Neon.vandq_u8(Arm.Neon.vshrq_n_u8(source, 6), mask)), Arm.Neon.vld1q_s8(act + 96));
                }
                return sum;
            }
            for (var r = 0; r < 4; r++)
            {
                v128 source = Arm.Neon.vld1q_u8(packed + r * 16);
                sbyte* act = values + (r >> 1) * 64 + (r & 1) * 16;
                sum = Dot16(sum, Weights(table, Arm.Neon.vandq_u8(source, mask)), Arm.Neon.vld1q_s8(act));
                sum = Dot16(sum, Weights(table, Arm.Neon.vandq_u8(Arm.Neon.vshrq_n_u8(source, 4), mask)), Arm.Neon.vld1q_s8(act + 32));
            }
            return sum;
        }

        // Mirrors Neon.DotRow.
        static float DotRow(Cq w, byte* packedRows, Activations a, int token, float* factors, float* scaled)
        {
            var groups = a.Groups;
            float bias = ScaleFactors(factors, scaled, a.Scales + token * a.Stride, a.Offsets + token * a.Stride, a.Stride);
            v128 table = Arm.Neon.vld1q_u8(w.Book->Bytes);
            v128 mask = w.Bits == 2 ? Arm.Neon.vdupq_n_u8(3) : Arm.Neon.vdupq_n_u8(15);
            sbyte* values = a.Values + (long)token * a.Cols;
            var accumulator = 0f;
            for (var g = 0; g < groups; g++)
            {
                v128 sum = GroupDot(packedRows + g * w.RowBytes / groups, values + g * Cq.GroupSize, mask, table, w.Bits);
                accumulator += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum)) * scaled[g];
            }
            return accumulator - bias;
        }

        // Mirrors Neon.TilePlane.
        static void TilePlane(v128 table, v128 mask, v128 index, int plane, int act, sbyte* v0, sbyte* v1, sbyte* v2, sbyte* v3, ref v128 sum0, ref v128 sum1, ref v128 sum2, ref v128 sum3)
        {
            var offset = act + plane * 32;
            v128 weights = Weights(table, Arm.Neon.vandq_u8(index, mask));
            sum0 = Dot16(sum0, weights, Arm.Neon.vld1q_s8(v0 + offset));
            sum1 = Dot16(sum1, weights, Arm.Neon.vld1q_s8(v1 + offset));
            sum2 = Dot16(sum2, weights, Arm.Neon.vld1q_s8(v2 + offset));
            sum3 = Dot16(sum3, weights, Arm.Neon.vld1q_s8(v3 + offset));
        }

        // Mirrors Neon.DotTile: one row × four tokens.
        static void DotTile(Cq w, byte* packedRows, Activations a, int token, float* factors, float* scaled, float* y, int ldy)
        {
            var groups = a.Groups;
            var stride = a.Stride;
            var cols = a.Cols;
            float bias0 = ScaleFactors(factors, scaled, a.Scales + token * stride, a.Offsets + token * stride, stride);
            float bias1 = ScaleFactors(factors, scaled + stride, a.Scales + (token + 1) * stride, a.Offsets + (token + 1) * stride, stride);
            float bias2 = ScaleFactors(factors, scaled + 2 * stride, a.Scales + (token + 2) * stride, a.Offsets + (token + 2) * stride, stride);
            float bias3 = ScaleFactors(factors, scaled + 3 * stride, a.Scales + (token + 3) * stride, a.Offsets + (token + 3) * stride, stride);
            v128 table = Arm.Neon.vld1q_u8(w.Book->Bytes);
            v128 mask = w.Bits == 2 ? Arm.Neon.vdupq_n_u8(3) : Arm.Neon.vdupq_n_u8(15);
            sbyte* v0 = a.Values + (long)token * cols;
            sbyte* v1 = v0 + cols;
            sbyte* v2 = v1 + cols;
            sbyte* v3 = v2 + cols;
            var accumulator0 = 0f;
            var accumulator1 = 0f;
            var accumulator2 = 0f;
            var accumulator3 = 0f;
            var packedStep = w.RowBytes / groups;
            var regs = w.Bits == 2 ? 2 : 4;
            for (var g = 0; g < groups; g++)
            {
                byte* packed = packedRows + g * packedStep;
                v128 sum0 = Arm.Neon.vdupq_n_s32(0);
                v128 sum1 = Arm.Neon.vdupq_n_s32(0);
                v128 sum2 = Arm.Neon.vdupq_n_s32(0);
                v128 sum3 = Arm.Neon.vdupq_n_s32(0);
                for (var r = 0; r < regs; r++)
                {
                    v128 source = Arm.Neon.vld1q_u8(packed + r * 16);
                    var act = g * Cq.GroupSize + (w.Bits == 2 ? r * 16 : (r >> 1) * 64 + (r & 1) * 16);
                    if (w.Bits == 2)
                    {
                        TilePlane(table, mask, source, 0, act, v0, v1, v2, v3, ref sum0, ref sum1, ref sum2, ref sum3);
                        TilePlane(table, mask, Arm.Neon.vshrq_n_u8(source, 2), 1, act, v0, v1, v2, v3, ref sum0, ref sum1, ref sum2, ref sum3);
                        TilePlane(table, mask, Arm.Neon.vshrq_n_u8(source, 4), 2, act, v0, v1, v2, v3, ref sum0, ref sum1, ref sum2, ref sum3);
                        TilePlane(table, mask, Arm.Neon.vshrq_n_u8(source, 6), 3, act, v0, v1, v2, v3, ref sum0, ref sum1, ref sum2, ref sum3);
                    }
                    else
                    {
                        TilePlane(table, mask, source, 0, act, v0, v1, v2, v3, ref sum0, ref sum1, ref sum2, ref sum3);
                        TilePlane(table, mask, Arm.Neon.vshrq_n_u8(source, 4), 1, act, v0, v1, v2, v3, ref sum0, ref sum1, ref sum2, ref sum3);
                    }
                }
                accumulator0 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum0)) * scaled[g];
                accumulator1 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum1)) * scaled[stride + g];
                accumulator2 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum2)) * scaled[2 * stride + g];
                accumulator3 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum3)) * scaled[3 * stride + g];
            }
            y[0] = accumulator0 - bias0;
            y[ldy] = accumulator1 - bias1;
            y[2 * ldy] = accumulator2 - bias2;
            y[3 * ldy] = accumulator3 - bias3;
        }

        // Mirrors Neon.BlockPlane.
        static void BlockPlane(v128 table, v128 mask, v128 i0, v128 i1, v128 i2, v128 i3, int plane, int act, sbyte* values, ref v128 sum0, ref v128 sum1, ref v128 sum2, ref v128 sum3)
        {
            v128 activation = Arm.Neon.vld1q_s8(values + act + plane * 32);
            sum0 = Dot16(sum0, Weights(table, Arm.Neon.vandq_u8(i0, mask)), activation);
            sum1 = Dot16(sum1, Weights(table, Arm.Neon.vandq_u8(i1, mask)), activation);
            sum2 = Dot16(sum2, Weights(table, Arm.Neon.vandq_u8(i2, mask)), activation);
            sum3 = Dot16(sum3, Weights(table, Arm.Neon.vandq_u8(i3, mask)), activation);
        }

        // Mirrors Neon.DotBlock: 4 rows × 1 token (the x64 Gemv.Block
        // counterpart). Activation vectors load once per plane and are shared
        // across four rows.
        static void DotBlock(Cq w, byte* packed, int rowBytes, ushort* norms, int groups, Activations a, int token, float* rowFactors, float* scaled, float* y, v128 table, v128 mask)
        {
            var stride = a.Stride;
            for (var r = 0; r < 4; r++)
                LoadFactors(norms + (long)r * groups, groups, w.Book->ByteScale, rowFactors + r * stride);
            float bias0 = ScaleFactors(rowFactors, scaled, a.Scales + token * stride, a.Offsets + token * stride, stride);
            float bias1 = ScaleFactors(rowFactors + stride, scaled + stride, a.Scales + token * stride, a.Offsets + token * stride, stride);
            float bias2 = ScaleFactors(rowFactors + 2 * stride, scaled + 2 * stride, a.Scales + token * stride, a.Offsets + token * stride, stride);
            float bias3 = ScaleFactors(rowFactors + 3 * stride, scaled + 3 * stride, a.Scales + token * stride, a.Offsets + token * stride, stride);
            sbyte* values = a.Values + (long)token * a.Cols;
            var accumulator0 = 0f;
            var accumulator1 = 0f;
            var accumulator2 = 0f;
            var accumulator3 = 0f;
            var packedStep = w.RowBytes / groups;
            var regs = w.Bits == 2 ? 2 : 4;
            byte* row1 = packed + rowBytes;
            byte* row2 = packed + 2 * rowBytes;
            byte* row3 = packed + 3 * rowBytes;
            for (var g = 0; g < groups; g++)
            {
                v128 sum0 = Arm.Neon.vdupq_n_s32(0);
                v128 sum1 = Arm.Neon.vdupq_n_s32(0);
                v128 sum2 = Arm.Neon.vdupq_n_s32(0);
                v128 sum3 = Arm.Neon.vdupq_n_s32(0);
                var offset = g * packedStep;
                for (var r = 0; r < regs; r++)
                {
                    v128 s0 = Arm.Neon.vld1q_u8(packed + offset + r * 16);
                    v128 s1 = Arm.Neon.vld1q_u8(row1 + offset + r * 16);
                    v128 s2 = Arm.Neon.vld1q_u8(row2 + offset + r * 16);
                    v128 s3 = Arm.Neon.vld1q_u8(row3 + offset + r * 16);
                    var act = g * Cq.GroupSize + (w.Bits == 2 ? r * 16 : (r >> 1) * 64 + (r & 1) * 16);
                    if (w.Bits == 2)
                    {
                        BlockPlane(table, mask, s0, s1, s2, s3, 0, act, values, ref sum0, ref sum1, ref sum2, ref sum3);
                        BlockPlane(table, mask, Arm.Neon.vshrq_n_u8(s0, 2), Arm.Neon.vshrq_n_u8(s1, 2), Arm.Neon.vshrq_n_u8(s2, 2), Arm.Neon.vshrq_n_u8(s3, 2), 1, act, values, ref sum0, ref sum1, ref sum2, ref sum3);
                        BlockPlane(table, mask, Arm.Neon.vshrq_n_u8(s0, 4), Arm.Neon.vshrq_n_u8(s1, 4), Arm.Neon.vshrq_n_u8(s2, 4), Arm.Neon.vshrq_n_u8(s3, 4), 2, act, values, ref sum0, ref sum1, ref sum2, ref sum3);
                        BlockPlane(table, mask, Arm.Neon.vshrq_n_u8(s0, 6), Arm.Neon.vshrq_n_u8(s1, 6), Arm.Neon.vshrq_n_u8(s2, 6), Arm.Neon.vshrq_n_u8(s3, 6), 3, act, values, ref sum0, ref sum1, ref sum2, ref sum3);
                    }
                    else
                    {
                        BlockPlane(table, mask, s0, s1, s2, s3, 0, act, values, ref sum0, ref sum1, ref sum2, ref sum3);
                        BlockPlane(table, mask, Arm.Neon.vshrq_n_u8(s0, 4), Arm.Neon.vshrq_n_u8(s1, 4), Arm.Neon.vshrq_n_u8(s2, 4), Arm.Neon.vshrq_n_u8(s3, 4), 1, act, values, ref sum0, ref sum1, ref sum2, ref sum3);
                    }
                }
                accumulator0 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum0)) * scaled[g];
                accumulator1 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum1)) * scaled[stride + g];
                accumulator2 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum2)) * scaled[2 * stride + g];
                accumulator3 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum3)) * scaled[3 * stride + g];
            }
            y[0] = accumulator0 - bias0;
            y[1] = accumulator1 - bias1;
            y[2] = accumulator2 - bias2;
            y[3] = accumulator3 - bias3;
        }

        // Mirrors Neon.PairPlane.
        static void PairPlane(v128 table, v128 mask, v128 i0, v128 i1, int plane, int act, sbyte* v0, sbyte* v1, sbyte* v2, sbyte* v3, ref v128 sum00, ref v128 sum01, ref v128 sum02, ref v128 sum03, ref v128 sum10, ref v128 sum11, ref v128 sum12, ref v128 sum13)
        {
            v128 w0 = Weights(table, Arm.Neon.vandq_u8(i0, mask));
            v128 w1 = Weights(table, Arm.Neon.vandq_u8(i1, mask));
            var offset = act + plane * 32;
            v128 a0 = Arm.Neon.vld1q_s8(v0 + offset);
            v128 a1 = Arm.Neon.vld1q_s8(v1 + offset);
            v128 a2 = Arm.Neon.vld1q_s8(v2 + offset);
            v128 a3 = Arm.Neon.vld1q_s8(v3 + offset);
            sum00 = Dot16(sum00, w0, a0);
            sum01 = Dot16(sum01, w0, a1);
            sum02 = Dot16(sum02, w0, a2);
            sum03 = Dot16(sum03, w0, a3);
            sum10 = Dot16(sum10, w1, a0);
            sum11 = Dot16(sum11, w1, a1);
            sum12 = Dot16(sum12, w1, a2);
            sum13 = Dot16(sum13, w1, a3);
        }

        // Mirrors Neon.DotPair: 2 rows × 4 tokens (the x64 Gemv.Pair4
        // counterpart). Each weight plane expands once and dots against all
        // four token activation vectors.
        static void DotPair(Cq w, byte* packed, int rowBytes, ushort* norms, int groups, Activations a, int token, float* rowFactors, float* scaled, float* y, int ldy, v128 table, v128 mask)
        {
            var stride = a.Stride;
            var cols = a.Cols;
            LoadFactors(norms, groups, w.Book->ByteScale, rowFactors);
            LoadFactors(norms + groups, groups, w.Book->ByteScale, rowFactors + stride);
            float bias00 = ScaleFactors(rowFactors, scaled, a.Scales + token * stride, a.Offsets + token * stride, stride);
            float bias01 = ScaleFactors(rowFactors, scaled + stride, a.Scales + (token + 1) * stride, a.Offsets + (token + 1) * stride, stride);
            float bias02 = ScaleFactors(rowFactors, scaled + 2 * stride, a.Scales + (token + 2) * stride, a.Offsets + (token + 2) * stride, stride);
            float bias03 = ScaleFactors(rowFactors, scaled + 3 * stride, a.Scales + (token + 3) * stride, a.Offsets + (token + 3) * stride, stride);
            float bias10 = ScaleFactors(rowFactors + stride, scaled + 4 * stride, a.Scales + token * stride, a.Offsets + token * stride, stride);
            float bias11 = ScaleFactors(rowFactors + stride, scaled + 5 * stride, a.Scales + (token + 1) * stride, a.Offsets + (token + 1) * stride, stride);
            float bias12 = ScaleFactors(rowFactors + stride, scaled + 6 * stride, a.Scales + (token + 2) * stride, a.Offsets + (token + 2) * stride, stride);
            float bias13 = ScaleFactors(rowFactors + stride, scaled + 7 * stride, a.Scales + (token + 3) * stride, a.Offsets + (token + 3) * stride, stride);
            sbyte* v0 = a.Values + (long)token * cols;
            sbyte* v1 = v0 + cols;
            sbyte* v2 = v1 + cols;
            sbyte* v3 = v2 + cols;
            var accumulator00 = 0f;
            var accumulator01 = 0f;
            var accumulator02 = 0f;
            var accumulator03 = 0f;
            var accumulator10 = 0f;
            var accumulator11 = 0f;
            var accumulator12 = 0f;
            var accumulator13 = 0f;
            var packedStep = w.RowBytes / groups;
            var regs = w.Bits == 2 ? 2 : 4;
            byte* row1 = packed + rowBytes;
            for (var g = 0; g < groups; g++)
            {
                v128 sum00 = Arm.Neon.vdupq_n_s32(0);
                v128 sum01 = Arm.Neon.vdupq_n_s32(0);
                v128 sum02 = Arm.Neon.vdupq_n_s32(0);
                v128 sum03 = Arm.Neon.vdupq_n_s32(0);
                v128 sum10 = Arm.Neon.vdupq_n_s32(0);
                v128 sum11 = Arm.Neon.vdupq_n_s32(0);
                v128 sum12 = Arm.Neon.vdupq_n_s32(0);
                v128 sum13 = Arm.Neon.vdupq_n_s32(0);
                var offset = g * packedStep;
                for (var r = 0; r < regs; r++)
                {
                    v128 s0 = Arm.Neon.vld1q_u8(packed + offset + r * 16);
                    v128 s1 = Arm.Neon.vld1q_u8(row1 + offset + r * 16);
                    var act = g * Cq.GroupSize + (w.Bits == 2 ? r * 16 : (r >> 1) * 64 + (r & 1) * 16);
                    if (w.Bits == 2)
                    {
                        PairPlane(table, mask, s0, s1, 0, act, v0, v1, v2, v3, ref sum00, ref sum01, ref sum02, ref sum03, ref sum10, ref sum11, ref sum12, ref sum13);
                        PairPlane(table, mask, Arm.Neon.vshrq_n_u8(s0, 2), Arm.Neon.vshrq_n_u8(s1, 2), 1, act, v0, v1, v2, v3, ref sum00, ref sum01, ref sum02, ref sum03, ref sum10, ref sum11, ref sum12, ref sum13);
                        PairPlane(table, mask, Arm.Neon.vshrq_n_u8(s0, 4), Arm.Neon.vshrq_n_u8(s1, 4), 2, act, v0, v1, v2, v3, ref sum00, ref sum01, ref sum02, ref sum03, ref sum10, ref sum11, ref sum12, ref sum13);
                        PairPlane(table, mask, Arm.Neon.vshrq_n_u8(s0, 6), Arm.Neon.vshrq_n_u8(s1, 6), 3, act, v0, v1, v2, v3, ref sum00, ref sum01, ref sum02, ref sum03, ref sum10, ref sum11, ref sum12, ref sum13);
                    }
                    else
                    {
                        PairPlane(table, mask, s0, s1, 0, act, v0, v1, v2, v3, ref sum00, ref sum01, ref sum02, ref sum03, ref sum10, ref sum11, ref sum12, ref sum13);
                        PairPlane(table, mask, Arm.Neon.vshrq_n_u8(s0, 4), Arm.Neon.vshrq_n_u8(s1, 4), 1, act, v0, v1, v2, v3, ref sum00, ref sum01, ref sum02, ref sum03, ref sum10, ref sum11, ref sum12, ref sum13);
                    }
                }
                accumulator00 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum00)) * scaled[g];
                accumulator01 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum01)) * scaled[stride + g];
                accumulator02 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum02)) * scaled[2 * stride + g];
                accumulator03 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum03)) * scaled[3 * stride + g];
                accumulator10 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum10)) * scaled[4 * stride + g];
                accumulator11 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum11)) * scaled[5 * stride + g];
                accumulator12 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum12)) * scaled[6 * stride + g];
                accumulator13 += HorizontalSumNeon(Arm.Neon.vcvtq_f32_s32(sum13)) * scaled[7 * stride + g];
            }
            y[0] = accumulator00 - bias00;
            y[ldy] = accumulator01 - bias01;
            y[2 * ldy] = accumulator02 - bias02;
            y[3 * ldy] = accumulator03 - bias03;
            y[1] = accumulator10 - bias10;
            y[ldy + 1] = accumulator11 - bias11;
            y[2 * ldy + 1] = accumulator12 - bias12;
            y[3 * ldy + 1] = accumulator13 - bias13;
        }

        // Mirrors Neon.BlockToken: 4-row DotBlock slices with a single-row
        // DotRow tail for one decode token across a row range.
        static void BlockToken(Cq w, Activations a, int token, float* yToken, int rowStart, int rowEnd, float* rowFactors, float* scaled, v128 table, v128 mask)
        {
            var groups = w.Groups;
            var n = rowStart;
            for (; n + 4 <= rowEnd; n += 4)
                DotBlock(w, w.Indices + (long)n * w.RowBytes, w.RowBytes, w.Norms + (long)n * groups, groups, a, token, rowFactors, scaled, yToken + n, table, mask);
            for (; n < rowEnd; n++)
            {
                LoadFactors(w.Norms + (long)n * groups, groups, w.Book->ByteScale, rowFactors);
                yToken[n] = DotRow(w, w.Indices + (long)n * w.RowBytes, a, token, rowFactors, scaled);
            }
        }

        // Mirrors Neon.MatMulRows: prefill-leaning token-major pairs when
        // (rowEnd - rowStart) * 4 <= tokens, decode-leaning row-major full
        // tiles plus blocked tail tokens otherwise. Row factors for up to a
        // 4-row decode block; scaled for up to a 2-row × 4-token pair tile
        // (DotBlock uses the first four slots).
        static void MatMulRows(Cq w, Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
        {
            float* rowFactors = stackalloc float[a.Stride * 4];
            float* scaled = stackalloc float[a.Stride * 8];
            var groups = w.Groups;
            v128 table = Arm.Neon.vld1q_u8(w.Book->Bytes);
            v128 mask = Arm.Neon.vdupq_n_u8(w.Bits == 2 ? (byte)3 : (byte)15);
            if ((rowEnd - rowStart) * 4 <= tokens)
            {
                // Prefill-leaning: token-major; row pairs share weight-plane expansion.
                for (var t = 0; t + 4 <= tokens; t += 4)
                {
                    var n = rowStart;
                    for (; n + 2 <= rowEnd; n += 2)
                        DotPair(w, w.Indices + (long)n * w.RowBytes, w.RowBytes, w.Norms + (long)n * groups, groups, a, t, rowFactors, scaled, y + (long)t * ldy + n, ldy, table, mask);
                    for (; n < rowEnd; n++)
                    {
                        LoadFactors(w.Norms + (long)n * groups, groups, w.Book->ByteScale, rowFactors);
                        DotTile(w, w.Indices + (long)n * w.RowBytes, a, t, rowFactors, scaled, y + (long)t * ldy + n, ldy);
                    }
                }
                for (var t = tokens & ~3; t < tokens; t++)
                    BlockToken(w, a, t, y + (long)t * ldy, rowStart, rowEnd, rowFactors, scaled, table, mask);
                return;
            }
            // Decode-leaning: row-major full tiles, then blocked tail tokens.
            var full = tokens & ~3;
            for (var n = rowStart; n < rowEnd; n++)
            {
                LoadFactors(w.Norms + (long)n * groups, groups, w.Book->ByteScale, rowFactors);
                for (var t = 0; t < full; t += 4)
                    DotTile(w, w.Indices + (long)n * w.RowBytes, a, t, rowFactors, scaled, y + (long)t * ldy + n, ldy);
            }
            for (var t = full; t < tokens; t++)
                BlockToken(w, a, t, y + (long)t * ldy, rowStart, rowEnd, rowFactors, scaled, table, mask);
        }

        // Scalar lane for targets without Neon (and the host-readable shape
        // every lane must agree with): mirrors Scalar.MatMulRows exactly —
        // per (row, token, group) integer dot, scaled = HalfToSingle(norm) *
        // byteScale * scale, accumulator += scaled * dot,
        // bias += scaled * offset, y = accumulator - bias.
        static void ScalarMatMulRows(Cq w, Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
        {
            var groups = w.Groups;
            byte* table = w.Book->Bytes;
            float byteScale = w.Book->ByteScale;
            for (var t = 0; t < tokens; t++)
            {
                sbyte* values = a.Values + (long)t * a.Cols;
                float* scales = a.Scales + t * a.Stride;
                float* offsets = a.Offsets + t * a.Stride;
                for (var n = rowStart; n < rowEnd; n++)
                {
                    byte* packed = w.Indices + (long)n * w.RowBytes;
                    ushort* norms = w.Norms + (long)n * groups;
                    var accumulator = 0f;
                    var bias = 0f;
                    for (var g = 0; g < groups; g++)
                    {
                        sbyte* x = values + g * Cq.GroupSize;
                        var dot = 0;
                        if (w.Bits == 2)
                        {
                            byte* p = packed + g * 32;
                            for (var j = 0; j < 32; j++)
                            {
                                int b = p[j];
                                dot += table[b & 3] * x[j] + table[b >> 2 & 3] * x[32 + j] + table[b >> 4 & 3] * x[64 + j] + table[b >> 6 & 3] * x[96 + j];
                            }
                        }
                        else
                        {
                            byte* p = packed + g * 64;
                            for (var half = 0; half < 2; half++)
                                for (var j = 0; j < 32; j++)
                                {
                                    int b = p[half * 32 + j];
                                    dot += table[b & 15] * x[2 * half * 32 + j] + table[b >> 4] * x[(2 * half + 1) * 32 + j];
                                }
                        }
                        float scaled = HalfToSingle(norms[g]) * byteScale * scales[g];
                        accumulator += scaled * dot;
                        bias += scaled * offsets[g];
                    }
                    y[(long)t * ldy + n] = accumulator - bias;
                }
            }
        }
    }
}
