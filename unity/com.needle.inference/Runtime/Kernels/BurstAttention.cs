// N36: Burst attention kernels (Track B, second §2.1 entry after N31 Softmax).
//
// Ports to Unity.Burst.Intrinsics, one lane per target, all behind two table
// entries (KernelTable.AttentionTile, KernelTable.MergePartials):
//   - Head:               Attention.Head (single-head flat path) + Ops.Softmax.
//   - GroupAvx2:          Attention.ScoresAvx2 + ExponentiateAcrossPositions
//                         (+ Normalizers/CaptureProbabilities) + ValuesFloat
//                         (AccumulateFloatSix + the 1/2/3-head chunked path).
//   - MergePartials:      Attention.MergePartials (count == 2 fast path +
//                         generic loop), shared by every lane.
//   - Neon AttentionTile: Neon.AttentionHead/DotInt8x48/Softmax structure
//                         (flat per-head scores, pair/single value accumulate)
//                         plus Peaks/Sums support so the chunked contract holds
//                         on every lane (net10 Neon never sees Peaks because
//                         CanSplitPositions is AVX-only; the Burst entry must
//                         still be correct for all table inputs).
// Skipped per H8: GroupVnni / ValuesInt8 (Burst exposes no AVX-VNNI; the
// integer matmul ports (N32+) use the Avx2Dot sequence per the plan).
//
// Layout contract (matches Blocks.Attention exactly):
//   - x86 lanes keep the GroupAvx2 interleaved scores layout
//     (scores[s * Lanes + h]); the buffer must hold ScoreFloats(count) floats.
//     The single-head entry keeps Head's flat layout (scores[s]).
//   - The Neon lane keeps Neon's flat per-head layout (scores[h * count + s]);
//     ScoreFloats(count) = (count + 4) * 8 >= 6 * count always fits.
//   - Output is heads * 64 floats; Probabilities (nullable) is addressed as
//     [h * ProbabilityStride + First + s]; Peaks/Sums (nullable, chunked path)
//     hold 8 floats each (only lanes [0, heads) are meaningful).
//   - No [NoAlias]: AttentionTile.Output may alias the partials slot in the
//     chunked path (ExecuteChunk). Nothing here assumes otherwise.
//
// Numerical contract (plan §3): float tails meet the N13 parity bound
// (cosine >= 0.99995 vs Scalar, same argmax); the int8 QK dots are bit-exact
// (integer addition of exact products commutes, so any unroll order matches
// the naive Scalar loop bit for bit). FloatMode.Strict, never Fast; explicit
// FMA intrinsics (mm256_fmadd_ps / vfmaq_f32 / MathF.FusedMultiplyAdd in the
// scalar lane), no reliance on contraction.
//
// H7 note: the x86 lane keeps the net10 instruction order (same Exp
// polynomial, same FMA order, same max/sum reductions), so finite inputs in
// the clamp range compare bitwise vs the AVX2 kernels up to two deliberate
// zero-risk deviations: (1) weight broadcasts use set1 (same bits as
// vbroadcastss); (2) the per-lane 1/sum divide in Normalizers goes through a
// scalar spill (bitwise identical IEEE division per lane). The Neon lane keeps
// Neon's op order (scalar peak, vector exp, pairwise horizontal sum). If the
// bitwise probe in Tests/ ever diverges, these kernels still meet the parity
// bound by construction (same fallback rule as N31).
//
// H8 note: no AVX-VNNI, no AVX-512. Verified against the Unity.Burst assembly
// API per N31 (X86 nested types are exactly Avx, Avx2, Bmi1, Bmi2, F16C, Fma,
// Popcnt, Sse, Sse2, Sse3, Sse4_1, Sse4_2, Ssse3).
//
// Intrinsic name provenance (reviewers check against the plan cookbook):
//   [N31] name already used by BurstSoftmax.cs (compile-proven in package).
//   [book] plan §2.1 cookbook mapping (N31-confirmed against installed Burst).
//   [fam] same struct family as an [N31] name (e.g. vdupq_n_s32 next to the
//         proven vdupq_n_f32/vdupq_n_u32); single-line fix if Burst renames.
//   x86: mm256_loadu/storeu_ps, set1_ps, min/max/mul/add/sub_ps, round_ps,
//        fmadd/fnmadd_ps, cvtps_epi32, castsi256_ps, castps256_ps128,
//        extractf128_ps, add_ps/shuffle_ps/cvtss_f32 (Sse), hadd_ps (Sse3) [N31];
//        loadu/storeu_si256 (Avx), loadu_si128/loadl_epi64 (Sse2),
//        madd_epi16, add_epi32, cvtepi8_epi16, cvtepi8_epi32, cvtepi32_ps [book].
//        Integer si loads take vector-pointer operands; call sites cast
//        explicitly ((v256*)/(v128*)), which also compiles if the binding
//        uses void*. Int broadcast/zero use `new v256(x)` [N31].
//   Neon: vld1q/vst1q_f32, vdupq_n_f32, vsub/vadd/vmul/vfmaq/min/max/rndn,
//         vshlq_n_s32, vaddq_s32, vcvtnq_s32_f32, vdupq_n_s32, vst1q_u32,
//         vpaddq_f32, vgetq_lane_f32 [N31]; vld1q_s8, vdotq_s32
//         (+IsNeonDotProdSupported), vaddvq_s32 [book].
//   Scalar lane: plain C# + MathF (Exp/Max/FusedMultiplyAdd), mirrors Scalar.
//
// C# 9 safe: block namespace, no target-typed new, no static lambdas.
using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using UnityEngine;

namespace Needle.BurstKernels
{
    // Managed mirror of the internal AttentionTile (src/Needle/Blocks.cs).
    // Field order is the layout contract: the table passes AttentionTile* and
    // this struct must overlay it exactly (Sequential, no packing games).
    // Install() checks the size loudly (128 bytes on 64-bit, 72 on 32-bit).
    [StructLayout(LayoutKind.Sequential)]
    public struct BurstAttentionTile
    {
        public unsafe sbyte* Queries;
        public unsafe float* QueryScales;
        public int Heads;
        public unsafe sbyte* Keys;
        public unsafe float* KeyScales;
        public unsafe sbyte* Values;
        public unsafe float* ValueScales;
        public int KeyStride, ValueStride, ScaleStride, First, Count;
        public unsafe float* Scores;
        public unsafe float* Output;
        public unsafe float* Probabilities;
        public int ProbabilityStride;
        public unsafe float* Peaks;
        public unsafe float* Sums;
    }

    public static unsafe class BurstAttention
    {
        const int QkDim = 48;
        const int VDim = 64;
        const int Lanes = 8;
        const int PositionBlock = 4;
        const int MaxTileHeads = 6;
        const int PartialFloats = MaxTileHeads * VDim + 16; // 400
        const int PeaksOffset = MaxTileHeads * VDim;         // 384
        const int SumsOffset = MaxTileHeads * VDim + Lanes;  // 392
        const float InverseSqrtQkDim = 0.14433756729740643f;

        // Cdecl delegates for the table slots (void(AttentionTile*) and
        // void(float*, int, int, float*)); match KernelTable ABI comments.
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void TileFn(BurstAttentionTile* tile);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void MergeFn(float* partials, int count, int heads, float* output);

        static FunctionPointer<TileFn> compiledTile;
        static bool tileReady;
        static FunctionPointer<MergeFn> compiledMerge;
        static bool mergeReady;
        static bool installAttempted;

        public static bool IsInstalled => Kernels.Active.AttentionTile != 0 && Kernels.Active.MergePartials != 0;

        // Active SIMD lane (for logs; branches fold at Burst compile time).
        public static string Lane =>
            Unity.Burst.Intrinsics.X86.Avx2.IsAvx2Supported && Unity.Burst.Intrinsics.X86.Fma.IsFmaSupported ? "x86-avx2-fma"
            : Unity.Burst.Intrinsics.Arm.Neon.IsNeonSupported ? "arm-neon" : "scalar";

        // Idempotent: fills only empty slots, never overwrites an explicit or
        // foreign provider (plan §2 selection order). Cached function
        // pointers, so Kernels.Reset() in tests re-installs without leaking a
        // fresh Burst compile. Returns true when it claimed at least one slot.
        public static bool Install(string provider = "burst")
        {
            var expected = IntPtr.Size == 8 ? 128 : 72;
            if (Marshal.SizeOf<BurstAttentionTile>() != expected)
                throw new InvalidOperationException(
                    "N36: BurstAttentionTile layout (" + Marshal.SizeOf<BurstAttentionTile>() +
                    ") does not overlay AttentionTile (" + expected + ").");
            var table = Kernels.Active;
            var claimed = false;
            if (table.AttentionTile == 0)
            {
                EnsureTileCompiled();
                table.AttentionTile = (nint)compiledTile.Value;
                claimed = true;
            }
            if (table.MergePartials == 0)
            {
                EnsureMergeCompiled();
                table.MergePartials = (nint)compiledMerge.Value;
                claimed = true;
            }
            if (!claimed)
                return false;
            var current = Kernels.Provider;
            Kernels.Install(table, string.IsNullOrEmpty(current) ? provider
                : (current.Contains("burst") ? current : current + "+" + provider));
            return true;
        }

        static void EnsureTileCompiled()
        {
            if (tileReady)
                return;
            compiledTile = BurstCompiler.CompileFunctionPointer<TileFn>(Tile);
            if (!compiledTile.IsCreated)
                throw new InvalidOperationException(
                    "N36: Burst refused to compile BurstAttention.Tile. " +
                    "The kernel must live in package source (H6).");
            tileReady = true;
        }

        static void EnsureMergeCompiled()
        {
            if (mergeReady)
                return;
            compiledMerge = BurstCompiler.CompileFunctionPointer<MergeFn>(Merge);
            if (!compiledMerge.IsCreated)
                throw new InvalidOperationException(
                    "N36: Burst refused to compile BurstAttention.Merge. " +
                    "The kernel must live in package source (H6).");
            mergeReady = true;
        }

        // Player + play-mode entry: BeforeSceneLoad so the table is ready
        // before Model.Open runs. Guarded: inference must survive a missing
        // Burst toolchain (falls back to Scalar, logs once).
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void AutoInstall()
        {
            if (installAttempted)
                return;
            installAttempted = true;
            try
            {
                Install();
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning("[needle] Burst attention install failed, using Scalar: " + e.Message);
            }
        }

        // Table entry: void(AttentionTile*). Dispatch mirrors Attention.Run
        // minus the pathways Burst cannot take: no Neon-direct branch (the
        // table owns the tile on ns2.1), no Vnni (H8), single head via Head.
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void Tile(BurstAttentionTile* tile)
        {
            if (Unity.Burst.Intrinsics.X86.Avx2.IsAvx2Supported && Unity.Burst.Intrinsics.X86.Fma.IsFmaSupported)
            {
                if (tile->Heads == 1)
                    HeadX86(tile);
                else
                    GroupX86(tile);
                return;
            }
            if (Unity.Burst.Intrinsics.Arm.Neon.IsNeonSupported)
            {
                TileNeon(tile);
                return;
            }
            TileScalar(tile);
        }

        // Table entry: void(float*, int, int, float*).
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void Merge(float* partials, int count, int heads, float* output)
        {
            if (Unity.Burst.Intrinsics.X86.Avx2.IsAvx2Supported && Unity.Burst.Intrinsics.X86.Fma.IsFmaSupported)
            {
                MergeX86(partials, count, heads, output);
                return;
            }
            if (Unity.Burst.Intrinsics.Arm.Neon.IsNeonSupported)
            {
                MergeNeon(partials, count, heads, output);
                return;
            }
            MergeScalar(partials, count, heads, output);
        }

        // Integer QK dot, exact (mirrors PortableAttention.Dot48): the
        // products are exact for sbyte inputs and integer addition commutes,
        // so any unroll order matches the naive Scalar loop bit for bit.
        // Internal (not private) so conformance benches can spot-check it
        // against a naive loop; live on the Neon scalar-fallback dot path.
        internal static int Dot48(sbyte* a, sbyte* b)
        {
            int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (var i = 0; i < QkDim; i += 4)
            {
                s0 += a[i] * b[i];
                s1 += a[i + 1] * b[i + 1];
                s2 += a[i + 2] * b[i + 2];
                s3 += a[i + 3] * b[i + 3];
            }
            return (s0 + s1) + (s2 + s3);
        }

        // ---- x86 single head (mirrors Attention.Head + Ops.Softmax) ----

        static void HeadX86(BurstAttentionTile* tile)
        {
            var count = tile->Count;
            var queryScale = tile->QueryScales[0] * InverseSqrtQkDim;
            var query = tile->Queries;
            var keys = tile->Keys + (long)tile->First * tile->KeyStride;
            var keyScales = tile->KeyScales + (long)tile->First * tile->ScaleStride;
            var scores = tile->Scores;
            for (var s = 0; s < count; s++)
                scores[s] = DotVec48X86(query, keys + (long)s * tile->KeyStride) * keyScales[(long)s * tile->ScaleStride] * queryScale;
            SoftmaxFlatX86(scores, count);
            if (tile->Probabilities != null)
                for (var s = 0; s < count; s++) tile->Probabilities[tile->First + s] = scores[s];
            var a0 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(0f);
            var a1 = a0;
            var a2 = a0;
            var a3 = a0;
            var a4 = a0;
            var a5 = a0;
            var a6 = a0;
            var a7 = a0;
            var values = tile->Values + (long)tile->First * tile->ValueStride;
            var valueScales = tile->ValueScales + (long)tile->First * tile->ScaleStride;
            for (var s = 0; s < count; s++)
            {
                var row = values + (long)s * tile->ValueStride;
                var w = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(scores[s] * valueScales[(long)s * tile->ScaleStride]);
                a0 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w, Widen8X86(row), a0);
                a1 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w, Widen8X86(row + 8), a1);
                a2 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w, Widen8X86(row + 16), a2);
                a3 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w, Widen8X86(row + 24), a3);
                a4 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w, Widen8X86(row + 32), a4);
                a5 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w, Widen8X86(row + 40), a5);
                a6 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w, Widen8X86(row + 48), a6);
                a7 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w, Widen8X86(row + 56), a7);
            }
            var output = tile->Output;
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output, a0);
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output + 8, a1);
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output + 16, a2);
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output + 24, a3);
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output + 32, a4);
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output + 40, a5);
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output + 48, a6);
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output + 56, a7);
        }

        // Mirrors Attention.DotInt8x48: 3 x madd(madd) thirds, scalar spill
        // sum (integer-exact in any reduction order).
        static float DotVec48X86(sbyte* a, sbyte* b)
        {
            var sum = new Unity.Burst.Intrinsics.v256(0);
            sum = Unity.Burst.Intrinsics.X86.Avx2.mm256_add_epi32(sum, Unity.Burst.Intrinsics.X86.Avx2.mm256_madd_epi16(
                Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi16(Unity.Burst.Intrinsics.X86.Sse2.loadu_si128((Unity.Burst.Intrinsics.v128*)a)),
                Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi16(Unity.Burst.Intrinsics.X86.Sse2.loadu_si128((Unity.Burst.Intrinsics.v128*)b))));
            sum = Unity.Burst.Intrinsics.X86.Avx2.mm256_add_epi32(sum, Unity.Burst.Intrinsics.X86.Avx2.mm256_madd_epi16(
                Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi16(Unity.Burst.Intrinsics.X86.Sse2.loadu_si128((Unity.Burst.Intrinsics.v128*)(a + 16))),
                Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi16(Unity.Burst.Intrinsics.X86.Sse2.loadu_si128((Unity.Burst.Intrinsics.v128*)(b + 16)))));
            sum = Unity.Burst.Intrinsics.X86.Avx2.mm256_add_epi32(sum, Unity.Burst.Intrinsics.X86.Avx2.mm256_madd_epi16(
                Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi16(Unity.Burst.Intrinsics.X86.Sse2.loadu_si128((Unity.Burst.Intrinsics.v128*)(a + 32))),
                Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi16(Unity.Burst.Intrinsics.X86.Sse2.loadu_si128((Unity.Burst.Intrinsics.v128*)(b + 32)))));
            int* spill = stackalloc int[8];
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_si256((Unity.Burst.Intrinsics.v256*)spill, sum);
            var total = 0;
            for (var i = 0; i < 8; i++) total += spill[i];
            return total;
        }

        // 8 int8 -> 8 float (mirrors Avx.ConvertToVector256Single(
        // Avx2.ConvertToVector256Int32(row))).
        static Unity.Burst.Intrinsics.v256 Widen8X86(sbyte* row)
        {
            return Unity.Burst.Intrinsics.X86.Avx.mm256_cvtepi32_ps(
                Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi32(
                    Unity.Burst.Intrinsics.X86.Sse2.cvtsi64_si128(*(long*)(row))));
        }

        // Mirrors the BCL Vector256.Sum order (per the N33 hardware probe:
        // Sum128(lower) + Sum128(upper), each hadd/hadd/lane0). The earlier
        // add(low, high) + hadd + hadd reading differs by 1 ulp in the total.
        static float Sum256X86(Unity.Burst.Intrinsics.v256 v)
        {
            var lo = Unity.Burst.Intrinsics.X86.Avx.mm256_castps256_ps128(v);
            var t0 = Unity.Burst.Intrinsics.X86.Sse3.hadd_ps(lo, lo);
            t0 = Unity.Burst.Intrinsics.X86.Sse3.hadd_ps(t0, t0);
            var hi = Unity.Burst.Intrinsics.X86.Avx.mm256_extractf128_ps(v, 1);
            var t1 = Unity.Burst.Intrinsics.X86.Sse3.hadd_ps(hi, hi);
            t1 = Unity.Burst.Intrinsics.X86.Sse3.hadd_ps(t1, t1);
            return Unity.Burst.Intrinsics.X86.Sse.cvtss_f32(t0) + Unity.Burst.Intrinsics.X86.Sse.cvtss_f32(t1);
        }
        // Mirrors Ops.Softmax exactly (flat buffer): vector peak, scalar tail
        // peak, vector exp-sum, scalar tail exp-sum, broadcast normalize.
        static void SoftmaxFlatX86(float* s, int n)
        {
            var peakVector = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(float.NegativeInfinity);
            int peakEnd = n & ~7;
            int p = 0;
            for (; p < peakEnd; p += 8)
                peakVector = Unity.Burst.Intrinsics.X86.Avx.mm256_max_ps(peakVector, Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(s + p));
            float peak = Max8X86(peakVector);
            for (; p < n; p++) peak = MathF.Max(peak, s[p]);
            var shift = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(peak);
            var sum = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(0f);
            int vectorEnd = n & ~7;
            int i = 0;
            for (; i < vectorEnd; i += 8)
            {
                var e = ExpX86(Unity.Burst.Intrinsics.X86.Avx.mm256_sub_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(s + i), shift));
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(s + i, e);
                sum = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(sum, e);
            }
            float total = Sum256X86(sum);
            for (; i < n; i++)
            {
                s[i] = MathF.Exp(s[i] - peak);
                total += s[i];
            }
            var inverse = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1f / total);
            for (i = 0; i < vectorEnd; i += 8)
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(s + i, Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(s + i), inverse));
            for (; i < n; i++) s[i] /= total;
        }

        // Mirrors Ops.Exp bit for bit (same as BurstSoftmax.ExpX86).
        static Unity.Burst.Intrinsics.v256 ExpX86(Unity.Burst.Intrinsics.v256 x)
        {
            x = Unity.Burst.Intrinsics.X86.Avx.mm256_min_ps(
                Unity.Burst.Intrinsics.X86.Avx.mm256_max_ps(x, Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(-87.3f)),
                Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(88.7f));
            var n = Unity.Burst.Intrinsics.X86.Avx.mm256_round_ps(
                Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(x, Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1.44269504f)), 8);
            var r = Unity.Burst.Intrinsics.X86.Fma.mm256_fnmadd_ps(n, Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(0.693359375f), x);
            r = Unity.Burst.Intrinsics.X86.Fma.mm256_fnmadd_ps(n, Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(-2.12194440e-4f), r);
            var poly = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1.9875691500e-4f);
            poly = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(poly, r, Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1.3981999507e-3f));
            poly = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(poly, r, Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(8.3334519073e-3f));
            poly = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(poly, r, Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(4.1665795894e-2f));
            poly = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(poly, r, Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1.6666665459e-1f));
            poly = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(poly, r, Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(5.0000001201e-1f));
            var y = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(
                poly,
                Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(r, r),
                Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(r, Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1f)));
            var scaled = Unity.Burst.Intrinsics.X86.Avx.mm256_castsi256_ps(
                Unity.Burst.Intrinsics.X86.Avx2.mm256_slli_epi32(
                    Unity.Burst.Intrinsics.X86.Avx2.mm256_add_epi32(Unity.Burst.Intrinsics.X86.Avx.mm256_cvtps_epi32(n), new Unity.Burst.Intrinsics.v256(127)), 23));
            return Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(y, scaled);
        }

        static float Max8X86(Unity.Burst.Intrinsics.v256 v)
        {
            var halves = Unity.Burst.Intrinsics.X86.Sse.max_ps(
                Unity.Burst.Intrinsics.X86.Avx.mm256_castps256_ps128(v),
                Unity.Burst.Intrinsics.X86.Avx.mm256_extractf128_ps(v, 1));
            var pairs = Unity.Burst.Intrinsics.X86.Sse.max_ps(halves, Unity.Burst.Intrinsics.X86.Sse.shuffle_ps(halves, halves, 0x4E));
            var quad = Unity.Burst.Intrinsics.X86.Sse.max_ps(pairs, Unity.Burst.Intrinsics.X86.Sse.shuffle_ps(pairs, pairs, 0xB1));
            return Unity.Burst.Intrinsics.X86.Sse.cvtss_f32(quad);
        }

        // ---- x86 group path (mirrors Attention.GroupAvx2) ----

        static void GroupX86(BurstAttentionTile* tile)
        {
            ScoresAvx2X86(tile);
            var inverse = stackalloc float[Lanes];
            ExponentiateX86(tile->Scores, tile->Count, tile->Peaks, tile->Sums, inverse);
            if (tile->Probabilities != null)
                for (var h = 0; h < tile->Heads; h++)
                    for (var s = 0; s < tile->Count; s++)
                        tile->Probabilities[(long)h * tile->ProbabilityStride + tile->First + s] =
                            tile->Scores[(long)s * Lanes + h] * inverse[h];
            ValuesFloatX86(tile, inverse);
        }

        static void ScoresAvx2X86(BurstAttentionTile* tile)
        {
            int* packed = stackalloc int[QkDim / 2 * Lanes];
            for (var j = 0; j < QkDim / 2; j++)
                for (var h = 0; h < Lanes; h++)
                {
                    if (h >= tile->Heads) { packed[j * Lanes + h] = 0; continue; }
                    var q = tile->Queries + h * QkDim + 2 * j;
                    packed[j * Lanes + h] = (int)(ushort)(short)q[0] | q[1] << 16;
                }
            short* converted = stackalloc short[PositionBlock * QkDim];
            float* lanes = stackalloc float[Lanes];
            var scales = tile->QueryScales;
            var heads = tile->Heads;
            lanes[0] = scales[0] * InverseSqrtQkDim;
            lanes[1] = heads > 1 ? scales[1] * InverseSqrtQkDim : 0f;
            lanes[2] = heads > 2 ? scales[2] * InverseSqrtQkDim : 0f;
            lanes[3] = heads > 3 ? scales[3] * InverseSqrtQkDim : 0f;
            lanes[4] = heads > 4 ? scales[4] * InverseSqrtQkDim : 0f;
            lanes[5] = heads > 5 ? scales[5] * InverseSqrtQkDim : 0f;
            lanes[6] = 0f;
            lanes[7] = 0f;
            var scale = Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(lanes);
            var count = tile->Count;
            var last = count - 1;
            var keys = tile->Keys + (long)tile->First * tile->KeyStride;
            var keyStride = tile->KeyStride;
            var keyScales = tile->KeyScales + (long)tile->First * tile->ScaleStride;
            var scaleStride = tile->ScaleStride;
            var scores = tile->Scores;
            int* key1 = (int*)(converted + QkDim);
            int* key2 = (int*)(converted + 2 * QkDim);
            int* key3 = (int*)(converted + 3 * QkDim);
            for (var s = 0; s < count; s += PositionBlock)
            {
                for (var p = 0; p < PositionBlock; p++)
                {
                    var row = keys + (long)Math.Min(s + p, last) * keyStride;
                    Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_si256((Unity.Burst.Intrinsics.v256*)(converted + p * QkDim),
                        Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi16(Unity.Burst.Intrinsics.X86.Sse2.loadu_si128((Unity.Burst.Intrinsics.v128*)row)));
                    Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_si256((Unity.Burst.Intrinsics.v256*)(converted + p * QkDim + 16),
                        Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi16(Unity.Burst.Intrinsics.X86.Sse2.loadu_si128((Unity.Burst.Intrinsics.v128*)(row + 16))));
                    Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_si256((Unity.Burst.Intrinsics.v256*)(converted + p * QkDim + 32),
                        Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi16(Unity.Burst.Intrinsics.X86.Sse2.loadu_si128((Unity.Burst.Intrinsics.v128*)(row + 32))));
                }
                var sum0 = new Unity.Burst.Intrinsics.v256(0);
                var sum1 = sum0;
                var sum2 = sum0;
                var sum3 = sum0;
                for (var j = 0; j < QkDim / 2; j++)
                {
                    var query = Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_si256((Unity.Burst.Intrinsics.v256*)(packed + j * Lanes));
                    sum0 = Unity.Burst.Intrinsics.X86.Avx2.mm256_add_epi32(sum0,
                        Unity.Burst.Intrinsics.X86.Avx2.mm256_madd_epi16(query, new Unity.Burst.Intrinsics.v256(((int*)converted)[j])));
                    sum1 = Unity.Burst.Intrinsics.X86.Avx2.mm256_add_epi32(sum1,
                        Unity.Burst.Intrinsics.X86.Avx2.mm256_madd_epi16(query, new Unity.Burst.Intrinsics.v256(key1[j])));
                    sum2 = Unity.Burst.Intrinsics.X86.Avx2.mm256_add_epi32(sum2,
                        Unity.Burst.Intrinsics.X86.Avx2.mm256_madd_epi16(query, new Unity.Burst.Intrinsics.v256(key2[j])));
                    sum3 = Unity.Burst.Intrinsics.X86.Avx2.mm256_add_epi32(sum3,
                        Unity.Burst.Intrinsics.X86.Avx2.mm256_madd_epi16(query, new Unity.Burst.Intrinsics.v256(key3[j])));
                }
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(scores + s * Lanes,
                    ScoreX86(sum0, keyScales[(long)s * scaleStride], scale));
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(scores + (s + 1) * Lanes,
                    ScoreX86(sum1, keyScales[(long)Math.Min(s + 1, last) * scaleStride], scale));
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(scores + (s + 2) * Lanes,
                    ScoreX86(sum2, keyScales[(long)Math.Min(s + 2, last) * scaleStride], scale));
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(scores + (s + 3) * Lanes,
                    ScoreX86(sum3, keyScales[(long)Math.Min(s + 3, last) * scaleStride], scale));
            }
        }

        static Unity.Burst.Intrinsics.v256 ScoreX86(Unity.Burst.Intrinsics.v256 dot, float keyScale, Unity.Burst.Intrinsics.v256 scale)
        {
            return Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(
                Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(
                    Unity.Burst.Intrinsics.X86.Avx.mm256_cvtepi32_ps(dot),
                    Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(keyScale)),
                scale);
        }

        // Mirrors ExponentiateAcrossPositions + Normalizers (peak/sum in the
        // same 4-accumulator order; the deferred inverse for the chunked path
        // is 1; the direct inverse is a per-lane 1/sum through a scalar spill,
        // bitwise identical to vdivps).
        static void ExponentiateX86(float* scores, int count, float* peaks, float* sums, float* inverse)
        {
            var negInf = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(float.NegativeInfinity);
            var peak0 = negInf;
            var peak1 = negInf;
            var peak2 = negInf;
            var peak3 = negInf;
            var s = 0;
            for (; s + 4 <= count; s += 4)
            {
                peak0 = Unity.Burst.Intrinsics.X86.Avx.mm256_max_ps(peak0, Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(scores + s * Lanes));
                peak1 = Unity.Burst.Intrinsics.X86.Avx.mm256_max_ps(peak1, Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(scores + (s + 1) * Lanes));
                peak2 = Unity.Burst.Intrinsics.X86.Avx.mm256_max_ps(peak2, Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(scores + (s + 2) * Lanes));
                peak3 = Unity.Burst.Intrinsics.X86.Avx.mm256_max_ps(peak3, Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(scores + (s + 3) * Lanes));
            }
            for (; s < count; s++) peak0 = Unity.Burst.Intrinsics.X86.Avx.mm256_max_ps(peak0, Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(scores + s * Lanes));
            var peak = Unity.Burst.Intrinsics.X86.Avx.mm256_max_ps(
                Unity.Burst.Intrinsics.X86.Avx.mm256_max_ps(peak0, peak1),
                Unity.Burst.Intrinsics.X86.Avx.mm256_max_ps(peak2, peak3));
            var zero = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(0f);
            var sum0 = zero;
            var sum1 = zero;
            var sum2 = zero;
            var sum3 = zero;
            for (s = 0; s + 4 <= count; s += 4)
            {
                var e0 = ExpX86(Unity.Burst.Intrinsics.X86.Avx.mm256_sub_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(scores + s * Lanes), peak));
                var e1 = ExpX86(Unity.Burst.Intrinsics.X86.Avx.mm256_sub_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(scores + (s + 1) * Lanes), peak));
                var e2 = ExpX86(Unity.Burst.Intrinsics.X86.Avx.mm256_sub_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(scores + (s + 2) * Lanes), peak));
                var e3 = ExpX86(Unity.Burst.Intrinsics.X86.Avx.mm256_sub_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(scores + (s + 3) * Lanes), peak));
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(scores + s * Lanes, e0);
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(scores + (s + 1) * Lanes, e1);
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(scores + (s + 2) * Lanes, e2);
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(scores + (s + 3) * Lanes, e3);
                sum0 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(sum0, e0);
                sum1 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(sum1, e1);
                sum2 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(sum2, e2);
                sum3 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(sum3, e3);
            }
            for (; s < count; s++)
            {
                var e = ExpX86(Unity.Burst.Intrinsics.X86.Avx.mm256_sub_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(scores + s * Lanes), peak));
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(scores + s * Lanes, e);
                sum0 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(sum0, e);
            }
            var sum = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(
                Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(sum0, sum1),
                Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(sum2, sum3));
            if (peaks == null)
            {
                float* spill = stackalloc float[Lanes];
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(spill, sum);
                for (var h = 0; h < Lanes; h++) spill[h] = 1f / spill[h];
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(inverse, Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(spill));
                return;
            }
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(peaks, peak);
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(sums, sum);
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(inverse, Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1f));
        }

        static void ValuesFloatX86(BurstAttentionTile* tile, float* inverse)
        {
            var scores = tile->Scores;
            var valueScales = tile->ValueScales + (long)tile->First * tile->ScaleStride;
            for (var s = 0; s < tile->Count; s++)
                Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(scores + s * Lanes,
                    Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(
                        Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(scores + s * Lanes),
                        Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(valueScales[(long)s * tile->ScaleStride])));
            if (tile->Heads == MaxTileHeads)
            {
                for (var firstDim = 0; firstDim < VDim; firstDim += 16)
                    AccumulateFloatSixX86(tile, inverse, firstDim);
                return;
            }
            for (var firstHead = 0; firstHead < tile->Heads; firstHead += 3)
                for (var firstDim = 0; firstDim < VDim; firstDim += 32)
                {
                    var rest = tile->Heads - firstHead;
                    if (rest == 1)
                        AccumulateFloatX86(tile, inverse, firstHead, firstDim, 1);
                    else if (rest == 2)
                        AccumulateFloatX86(tile, inverse, firstHead, firstDim, 2);
                    else
                        AccumulateFloatX86(tile, inverse, firstHead, firstDim, 3);
                }
        }

        // One V-row read feeds all six GQA heads (mirrors AccumulateFloatSix:
        // 12 accumulators over 16-wide dim slices, per-head sum order over s
        // unchanged).
        static void AccumulateFloatSixX86(BurstAttentionTile* tile, float* inverse, int firstDim)
        {
            var zero = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(0f);
            var a00 = zero;
            var a01 = zero;
            var a10 = zero;
            var a11 = zero;
            var a20 = zero;
            var a21 = zero;
            var a30 = zero;
            var a31 = zero;
            var a40 = zero;
            var a41 = zero;
            var a50 = zero;
            var a51 = zero;
            var count = tile->Count;
            var rows = tile->Values + (long)tile->First * tile->ValueStride + firstDim;
            var rowStride = tile->ValueStride;
            var weights = tile->Scores;
            for (var s = 0; s < count; s++)
            {
                var row = rows + (long)s * rowStride;
                var v0 = Unity.Burst.Intrinsics.X86.Avx.mm256_cvtepi32_ps(
                    Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi32(
                        Unity.Burst.Intrinsics.X86.Sse2.cvtsi64_si128(*(long*)(row))));
                var v1 = Unity.Burst.Intrinsics.X86.Avx.mm256_cvtepi32_ps(
                    Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi32(
                        Unity.Burst.Intrinsics.X86.Sse2.cvtsi64_si128(*(long*)(row + 8))));
                var w0 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(weights[s * Lanes]);
                a00 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w0, v0, a00);
                a01 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w0, v1, a01);
                var w1 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(weights[s * Lanes + 1]);
                a10 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w1, v0, a10);
                a11 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w1, v1, a11);
                var w2 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(weights[s * Lanes + 2]);
                a20 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w2, v0, a20);
                a21 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w2, v1, a21);
                var w3 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(weights[s * Lanes + 3]);
                a30 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w3, v0, a30);
                a31 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w3, v1, a31);
                var w4 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(weights[s * Lanes + 4]);
                a40 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w4, v0, a40);
                a41 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w4, v1, a41);
                var w5 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(weights[s * Lanes + 5]);
                a50 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w5, v0, a50);
                a51 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w5, v1, a51);
            }
            var output = tile->Output + firstDim;
            StoreScaledPairX86(output, a00, a01, inverse[0]);
            StoreScaledPairX86(output + VDim, a10, a11, inverse[1]);
            StoreScaledPairX86(output + 2 * VDim, a20, a21, inverse[2]);
            StoreScaledPairX86(output + 3 * VDim, a30, a31, inverse[3]);
            StoreScaledPairX86(output + 4 * VDim, a40, a41, inverse[4]);
            StoreScaledPairX86(output + 5 * VDim, a50, a51, inverse[5]);
        }

        static void StoreScaledPairX86(float* output, Unity.Burst.Intrinsics.v256 a0, Unity.Burst.Intrinsics.v256 a1, float factor)
        {
            var scale = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(factor);
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output, Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a0, scale));
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output + 8, Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a1, scale));
        }

        // Mirrors AccumulateFloat<THeads> (3-head chunked path): heads is the
        // live remainder (1..3); per-head FMA order over s is unchanged.
        static void AccumulateFloatX86(BurstAttentionTile* tile, float* inverse, int firstHead, int firstDim, int heads)
        {
            var zero = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(0f);
            var a00 = zero;
            var a01 = zero;
            var a02 = zero;
            var a03 = zero;
            var a10 = zero;
            var a11 = zero;
            var a12 = zero;
            var a13 = zero;
            var a20 = zero;
            var a21 = zero;
            var a22 = zero;
            var a23 = zero;
            var count = tile->Count;
            var rows = tile->Values + (long)tile->First * tile->ValueStride + firstDim;
            var rowStride = tile->ValueStride;
            var weights = tile->Scores + firstHead;
            for (var s = 0; s < count; s++)
            {
                var row = rows + (long)s * rowStride;
                var v0 = Unity.Burst.Intrinsics.X86.Avx.mm256_cvtepi32_ps(
                    Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi32(
                        Unity.Burst.Intrinsics.X86.Sse2.cvtsi64_si128(*(long*)(row))));
                var v1 = Unity.Burst.Intrinsics.X86.Avx.mm256_cvtepi32_ps(
                    Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi32(
                        Unity.Burst.Intrinsics.X86.Sse2.cvtsi64_si128(*(long*)(row + 8))));
                var v2 = Unity.Burst.Intrinsics.X86.Avx.mm256_cvtepi32_ps(
                    Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi32(
                        Unity.Burst.Intrinsics.X86.Sse2.cvtsi64_si128(*(long*)(row + 16))));
                var v3 = Unity.Burst.Intrinsics.X86.Avx.mm256_cvtepi32_ps(
                    Unity.Burst.Intrinsics.X86.Avx2.mm256_cvtepi8_epi32(
                        Unity.Burst.Intrinsics.X86.Sse2.cvtsi64_si128(*(long*)(row + 24))));
                var w0 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(weights[s * Lanes]);
                a00 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w0, v0, a00);
                a01 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w0, v1, a01);
                a02 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w0, v2, a02);
                a03 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w0, v3, a03);
                if (heads > 1)
                {
                    var w1 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(weights[s * Lanes + 1]);
                    a10 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w1, v0, a10);
                    a11 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w1, v1, a11);
                    a12 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w1, v2, a12);
                    a13 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w1, v3, a13);
                }
                if (heads > 2)
                {
                    var w2 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(weights[s * Lanes + 2]);
                    a20 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w2, v0, a20);
                    a21 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w2, v1, a21);
                    a22 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w2, v2, a22);
                    a23 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w2, v3, a23);
                }
            }
            var output = tile->Output + firstHead * VDim + firstDim;
            StoreScaledX86(output, a00, a01, a02, a03, inverse[firstHead]);
            if (heads > 1) StoreScaledX86(output + VDim, a10, a11, a12, a13, inverse[firstHead + 1]);
            if (heads > 2) StoreScaledX86(output + 2 * VDim, a20, a21, a22, a23, inverse[firstHead + 2]);
        }

        static void StoreScaledX86(float* output, Unity.Burst.Intrinsics.v256 a0, Unity.Burst.Intrinsics.v256 a1, Unity.Burst.Intrinsics.v256 a2, Unity.Burst.Intrinsics.v256 a3, float factor)
        {
            var scale = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(factor);
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output, Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a0, scale));
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output + 8, Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a1, scale));
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output + 16, Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a2, scale));
            Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output + 24, Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a3, scale));
        }

        // ---- MergePartials (mirrors Attention.MergePartials) ----
        // All three lanes are bitwise identical: same FMA order, fused
        // multiply-add everywhere (vfmaq / mm256_fmadd_ps / FusedMultiplyAdd).

        static void MergeX86(float* partials, int count, int heads, float* output)
        {
            float* weights = stackalloc float[Lanes];
            for (var h = 0; h < heads; h++)
            {
                if (count == 2)
                {
                    var peak0 = partials[PeaksOffset + h];
                    var peak1 = partials[PartialFloats + PeaksOffset + h];
                    var pk = MathF.Max(peak0, peak1);
                    var w0 = MathF.Exp(peak0 - pk);
                    var w1 = MathF.Exp(peak1 - pk);
                    var tot = 0f;
                    tot += w0 * partials[SumsOffset + h];
                    tot += w1 * partials[PartialFloats + SumsOffset + h];
                    var inv = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1f / tot);
                    var w0v = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(w0);
                    var w1v = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(w1);
                    for (var d = 0; d < VDim; d += Lanes)
                    {
                        var sum = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(0f);
                        sum = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w0v,
                            Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(partials + h * VDim + d), sum);
                        sum = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(w1v,
                            Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(partials + PartialFloats + h * VDim + d), sum);
                        Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output + h * VDim + d,
                            Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(sum, inv));
                    }
                    continue;
                }
                var peak = float.NegativeInfinity;
                for (var c = 0; c < count; c++) peak = MathF.Max(peak, partials[c * PartialFloats + PeaksOffset + h]);
                var total = 0f;
                for (var c = 0; c < count; c++)
                {
                    weights[c] = MathF.Exp(partials[c * PartialFloats + PeaksOffset + h] - peak);
                    total += weights[c] * partials[c * PartialFloats + SumsOffset + h];
                }
                var inverse = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1f / total);
                for (var d = 0; d < VDim; d += Lanes)
                {
                    var sum = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(0f);
                    for (var c = 0; c < count; c++)
                        sum = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(
                            Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(weights[c]),
                            Unity.Burst.Intrinsics.X86.Avx.mm256_loadu_ps(partials + c * PartialFloats + h * VDim + d), sum);
                    Unity.Burst.Intrinsics.X86.Avx.mm256_storeu_ps(output + h * VDim + d,
                        Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(sum, inverse));
                }
            }
        }

        static void MergeNeon(float* partials, int count, int heads, float* output)
        {
            float* weights = stackalloc float[Lanes];
            for (var h = 0; h < heads; h++)
            {
                if (count == 2)
                {
                    var peak0 = partials[PeaksOffset + h];
                    var peak1 = partials[PartialFloats + PeaksOffset + h];
                    var pk = MathF.Max(peak0, peak1);
                    var w0 = MathF.Exp(peak0 - pk);
                    var w1 = MathF.Exp(peak1 - pk);
                    var tot = 0f;
                    tot += w0 * partials[SumsOffset + h];
                    tot += w1 * partials[PartialFloats + SumsOffset + h];
                    var inv = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(1f / tot);
                    var w0v = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(w0);
                    var w1v = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(w1);
                    for (var d = 0; d < VDim; d += 4)
                    {
                        var sum = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(0f);
                        sum = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(sum, w0v,
                            Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32(partials + h * VDim + d));
                        sum = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(sum, w1v,
                            Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32(partials + PartialFloats + h * VDim + d));
                        Unity.Burst.Intrinsics.Arm.Neon.vst1q_f32(output + h * VDim + d,
                            Unity.Burst.Intrinsics.Arm.Neon.vmulq_f32(sum, inv));
                    }
                    continue;
                }
                var peak = float.NegativeInfinity;
                for (var c = 0; c < count; c++) peak = MathF.Max(peak, partials[c * PartialFloats + PeaksOffset + h]);
                var total = 0f;
                for (var c = 0; c < count; c++)
                {
                    weights[c] = MathF.Exp(partials[c * PartialFloats + PeaksOffset + h] - peak);
                    total += weights[c] * partials[c * PartialFloats + SumsOffset + h];
                }
                var inverse = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(1f / total);
                for (var d = 0; d < VDim; d += 4)
                {
                    var sum = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(0f);
                    for (var c = 0; c < count; c++)
                        sum = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(sum,
                            Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(weights[c]),
                            Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32(partials + c * PartialFloats + h * VDim + d));
                    Unity.Burst.Intrinsics.Arm.Neon.vst1q_f32(output + h * VDim + d,
                        Unity.Burst.Intrinsics.Arm.Neon.vmulq_f32(sum, inverse));
                }
            }
        }

        static void MergeScalar(float* partials, int count, int heads, float* output)
        {
            float* weights = stackalloc float[Lanes];
            for (var h = 0; h < heads; h++)
            {
                if (count == 2)
                {
                    var peak0 = partials[PeaksOffset + h];
                    var peak1 = partials[PartialFloats + PeaksOffset + h];
                    var pk = MathF.Max(peak0, peak1);
                    var w0 = MathF.Exp(peak0 - pk);
                    var w1 = MathF.Exp(peak1 - pk);
                    var tot = 0f;
                    tot += w0 * partials[SumsOffset + h];
                    tot += w1 * partials[PartialFloats + SumsOffset + h];
                    var inv = 1f / tot;
                    for (var d = 0; d < VDim; d++)
                        output[h * VDim + d] = (((w1) * (partials[PartialFloats + h * VDim + d]) + (((w0) * (partials[h * VDim + d]) + (0f)))) * inv);
                    continue;
                }
                var peak = float.NegativeInfinity;
                for (var c = 0; c < count; c++) peak = MathF.Max(peak, partials[c * PartialFloats + PeaksOffset + h]);
                var total = 0f;
                for (var c = 0; c < count; c++)
                {
                    weights[c] = MathF.Exp(partials[c * PartialFloats + PeaksOffset + h] - peak);
                    total += weights[c] * partials[c * PartialFloats + SumsOffset + h];
                }
                var inverse = 1f / total;
                for (var d = 0; d < VDim; d++)
                {
                    var sum = 0f;
                    for (var c = 0; c < count; c++)
                        sum = ((weights[c]) * (partials[c * PartialFloats + h * VDim + d]) + (sum));
                    output[h * VDim + d] = sum * inverse;
                }
            }
        }

        // ---- Neon lane (mirrors Neon.AttentionTile + Peaks support) ----
        // Flat per-head scores (scores[h * count + s]); dot via vdotq (or the
        // exact scalar Dot48 where dotprod is unavailable); values via the
        // pair/single 16-column accumulate with scalar widen (int8 -> float is
        // exact) and vfmaq in the same per-s order.

        static void TileNeon(BurstAttentionTile* tile)
        {
            var count = tile->Count;
            for (var h = 0; h < tile->Heads; h++)
            {
                var scores = tile->Scores + h * count;
                var query = tile->Queries + h * QkDim;
                var scale = tile->QueryScales[h] * InverseSqrtQkDim;
                for (var s = 0; s < count; s++)
                    scores[s] = DotNeon(query, tile->Keys + ((long)tile->First + s) * tile->KeyStride) *
                        tile->KeyScales[((long)tile->First + s) * tile->ScaleStride] * scale;
                SoftmaxFlatNeon(tile, h, scores, count);
            }
            for (var h = 0; h < tile->Heads; h += 2)
                for (var column = 0; column < VDim; column += 16)
                {
                    if (tile->Heads - h >= 2)
                        AccumulateValuePairNeon(tile, h, column);
                    else
                        AccumulateValueSingleNeon(tile, h, column);
                }
        }

        static int DotNeon(sbyte* a, sbyte* b)
        {
            if (Unity.Burst.Intrinsics.Arm.Neon.IsNeonDotProdSupported)
            {
                var acc = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_s32(0);
                acc = Unity.Burst.Intrinsics.Arm.Neon.vdotq_s32(acc,
                    Unity.Burst.Intrinsics.Arm.Neon.vld1q_s8(a), Unity.Burst.Intrinsics.Arm.Neon.vld1q_s8(b));
                acc = Unity.Burst.Intrinsics.Arm.Neon.vdotq_s32(acc,
                    Unity.Burst.Intrinsics.Arm.Neon.vld1q_s8(a + 16), Unity.Burst.Intrinsics.Arm.Neon.vld1q_s8(b + 16));
                acc = Unity.Burst.Intrinsics.Arm.Neon.vdotq_s32(acc,
                    Unity.Burst.Intrinsics.Arm.Neon.vld1q_s8(a + 32), Unity.Burst.Intrinsics.Arm.Neon.vld1q_s8(b + 32));
                return Unity.Burst.Intrinsics.Arm.Neon.vaddvq_s32(acc);
            }
            return Dot48(a, b);
        }

        // Per-head normalize with chunked-path support: direct tiles divide by
        // the sum (mirrors Neon.Softmax); chunked tiles (Peaks set) store
        // peak/sums and defer with an inverse of 1 (mirrors Normalizers).
        static void SoftmaxFlatNeon(BurstAttentionTile* tile, int head, float* scores, int count)
        {
            var peak = float.NegativeInfinity;
            for (var i = 0; i < count; i++) peak = MathF.Max(peak, scores[i]);
            var shift = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(peak);
            var sum = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(0f);
            int vectorEnd = count & ~3;
            int i2 = 0;
            for (; i2 < vectorEnd; i2 += 4)
            {
                var e = ExpNeon(Unity.Burst.Intrinsics.Arm.Neon.vsubq_f32(Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32(scores + i2), shift));
                Unity.Burst.Intrinsics.Arm.Neon.vst1q_f32(scores + i2, e);
                sum = Unity.Burst.Intrinsics.Arm.Neon.vaddq_f32(sum, e);
            }
            var total = HorizontalSumNeon(sum);
            for (; i2 < count; i2++)
            {
                scores[i2] = MathF.Exp(scores[i2] - peak);
                total += scores[i2];
            }
            if (tile->Peaks == null)
            {
                var vinv = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(1f / total);
                for (i2 = 0; i2 < vectorEnd; i2 += 4)
                    Unity.Burst.Intrinsics.Arm.Neon.vst1q_f32(scores + i2,
                        Unity.Burst.Intrinsics.Arm.Neon.vmulq_f32(Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32(scores + i2), vinv));
                for (; i2 < count; i2++) scores[i2] /= total;
            }
            else
            {
                tile->Peaks[head] = peak;
                tile->Sums[head] = total;
            }
            // Scores are already in their final weighting here (normalized in
            // place in direct mode; raw exps with deferred inverse 1 in
            // chunked mode, where capture never runs in practice). No second
            // divide: net10's Neon lane stores the softmax outputs directly.
            if (tile->Probabilities != null)
                for (var s = 0; s < count; s++)
                    tile->Probabilities[(long)head * tile->ProbabilityStride + tile->First + s] = scores[s];
        }

        static void AccumulateValuePairNeon(BurstAttentionTile* tile, int head, int column)
        {
            var count = tile->Count;
            var first = tile->Scores + head * count;
            var second = first + count;
            var a0 = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(0f);
            var a1 = a0;
            var a2 = a0;
            var a3 = a0;
            var b0 = a0;
            var b1 = a0;
            var b2 = a0;
            var b3 = a0;
            float* widened = stackalloc float[16];
            for (var s = 0; s < count; s++)
            {
                Widen16Neon(tile->Values + ((long)tile->First + s) * tile->ValueStride + column, widened);
                var valueScale = tile->ValueScales[((long)tile->First + s) * tile->ScaleStride];
                var wa = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(first[s] * valueScale);
                var wb = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(second[s] * valueScale);
                var f0 = Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32(widened);
                var f1 = Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32(widened + 4);
                var f2 = Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32(widened + 8);
                var f3 = Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32(widened + 12);
                a0 = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(a0, wa, f0);
                a1 = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(a1, wa, f1);
                a2 = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(a2, wa, f2);
                a3 = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(a3, wa, f3);
                b0 = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(b0, wb, f0);
                b1 = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(b1, wb, f1);
                b2 = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(b2, wb, f2);
                b3 = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(b3, wb, f3);
            }
            Store4Neon(tile->Output + head * VDim + column, a0, a1, a2, a3);
            Store4Neon(tile->Output + (head + 1) * VDim + column, b0, b1, b2, b3);
        }

        static void AccumulateValueSingleNeon(BurstAttentionTile* tile, int head, int column)
        {
            var count = tile->Count;
            var weights = tile->Scores + head * count;
            var a0 = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(0f);
            var a1 = a0;
            var a2 = a0;
            var a3 = a0;
            float* widened = stackalloc float[16];
            for (var s = 0; s < count; s++)
            {
                Widen16Neon(tile->Values + ((long)tile->First + s) * tile->ValueStride + column, widened);
                var w = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(
                    weights[s] * tile->ValueScales[((long)tile->First + s) * tile->ScaleStride]);
                a0 = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(a0, w, Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32(widened));
                a1 = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(a1, w, Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32(widened + 4));
                a2 = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(a2, w, Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32(widened + 8));
                a3 = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(a3, w, Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32(widened + 12));
            }
            Store4Neon(tile->Output + head * VDim + column, a0, a1, a2, a3);
        }

        static void Widen16Neon(sbyte* row, float* dst)
        {
            for (var i = 0; i < 16; i++) dst[i] = row[i];
        }

        static void Store4Neon(float* destination, Unity.Burst.Intrinsics.v128 a, Unity.Burst.Intrinsics.v128 b, Unity.Burst.Intrinsics.v128 c, Unity.Burst.Intrinsics.v128 d)
        {
            Unity.Burst.Intrinsics.Arm.Neon.vst1q_f32(destination, a);
            Unity.Burst.Intrinsics.Arm.Neon.vst1q_f32(destination + 4, b);
            Unity.Burst.Intrinsics.Arm.Neon.vst1q_f32(destination + 8, c);
            Unity.Burst.Intrinsics.Arm.Neon.vst1q_f32(destination + 12, d);
        }

        // Mirrors Neon.Exp: clamp, FRINTN round, two vfmaq reductions
        // (negation is exact, fma rounds once), ascending-Horner six-term
        // polynomial, 2^n scale via an s32->f32 spill (same bits as .AsSingle).
        // The polynomial MUST read vfmaq(c, p, r) = c + r * p; the transposed
        // order vfmaq(p, c, r) = p + c * r is only linear in r (harness-checked
        // against MathF.Exp: percents off everywhere except r == 0). The s32
        // convert/add/shift mirrors ConvertToInt32RoundToEven: the unsigned
        // convert saturates negatives to 0, which would pin the scale to 1.0
        // for every softmax input (s - peak <= 0 gives n <= 0).
        static Unity.Burst.Intrinsics.v128 ExpNeon(Unity.Burst.Intrinsics.v128 x)
        {
            x = Unity.Burst.Intrinsics.Arm.Neon.vminq_f32(
                Unity.Burst.Intrinsics.Arm.Neon.vmaxq_f32(x, Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(-87.3f)),
                Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(88.7f));
            var n = Unity.Burst.Intrinsics.Arm.Neon.vrndnq_f32(
                Unity.Burst.Intrinsics.Arm.Neon.vmulq_f32(x, Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(1.44269504f)));
            var r = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(x, n, Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(-0.693359375f));
            r = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(r, n, Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(2.12194440e-4f));
            var p = Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(1.9875691500e-4f);
            p = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(1.3981999507e-3f), p, r);
            p = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(8.3334519073e-3f), p, r);
            p = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(4.1665795894e-2f), p, r);
            p = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(1.6666665459e-1f), p, r);
            p = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(5.0000001201e-1f), p, r);
            // y = (r + 1) + r * r * p: addend first, same as net10 and x86.
            var y = Unity.Burst.Intrinsics.Arm.Neon.vfmaq_f32(
                Unity.Burst.Intrinsics.Arm.Neon.vaddq_f32(r, Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_f32(1f)),
                Unity.Burst.Intrinsics.Arm.Neon.vmulq_f32(r, r),
                p);
            var bits = Unity.Burst.Intrinsics.Arm.Neon.vshlq_n_s32(
                Unity.Burst.Intrinsics.Arm.Neon.vaddq_s32(
                    Unity.Burst.Intrinsics.Arm.Neon.vcvtnq_s32_f32(n),
                    Unity.Burst.Intrinsics.Arm.Neon.vdupq_n_s32(127)), 23);
            uint* spill = stackalloc uint[4];
            Unity.Burst.Intrinsics.Arm.Neon.vst1q_u32(spill, bits);
            var scale = Unity.Burst.Intrinsics.Arm.Neon.vld1q_f32((float*)spill);
            return Unity.Burst.Intrinsics.Arm.Neon.vmulq_f32(y, scale);
        }

        static float HorizontalSumNeon(Unity.Burst.Intrinsics.v128 v)
        {
            var pairs = Unity.Burst.Intrinsics.Arm.Neon.vpaddq_f32(v, v);
            var quad = Unity.Burst.Intrinsics.Arm.Neon.vpaddq_f32(pairs, pairs);
            return Unity.Burst.Intrinsics.Arm.Neon.vgetq_lane_f32(quad, 0);
        }

        // ---- scalar lane (mirrors Scalar.AttentionTile + MergePartials) ----
        // Reference shape every lane must agree with; also the fallback for
        // targets with neither AVX2+FMA nor Neon.

        static void TileScalar(BurstAttentionTile* tile)
        {
            for (var h = 0; h < tile->Heads; h++)
                HeadScalar(
                    tile->Queries + h * QkDim, tile->QueryScales[h], tile->Keys, tile->KeyScales,
                    tile->Values, tile->ValueScales, tile->KeyStride, tile->ValueStride, tile->ScaleStride,
                    tile->First, tile->First + tile->Count - 1, tile->Scores,
                    tile->Output + h * VDim,
                    tile->Probabilities == null ? null : tile->Probabilities + h * tile->ProbabilityStride);
        }

        static void HeadScalar(
            sbyte* query, float queryScale, sbyte* keys, float* keyScales, sbyte* values, float* valueScales,
            int keyStride, int valueStride, int scaleStride, int first, int last, float* scores, float* output, float* probabilities)
        {
            var count = last - first + 1;
            var scale = queryScale * InverseSqrtQkDim;
            for (var s = 0; s < count; s++)
            {
                var key = keys + ((long)first + s) * keyStride;
                scores[s] = Dot48(query, key) * keyScales[((long)first + s) * scaleStride] * scale;
            }
            SoftmaxScalar(scores, count);
            if (probabilities != null) for (var s = 0; s < count; s++) probabilities[first + s] = scores[s];
            for (var d = 0; d < VDim; d++) output[d] = 0f;
            for (var s = 0; s < count; s++)
            {
                var row = values + ((long)first + s) * valueStride;
                var weight = scores[s] * valueScales[((long)first + s) * scaleStride];
                for (var d = 0; d < VDim; d++) output[d] += weight * row[d];
            }
        }

        static void SoftmaxScalar(float* s, int n)
        {
            var peak = float.NegativeInfinity;
            for (var i = 0; i < n; i++) peak = MathF.Max(peak, s[i]);
            var total = 0f;
            for (var i = 0; i < n; i++)
            {
                s[i] = MathF.Exp(s[i] - peak);
                total += s[i];
            }
            for (var i = 0; i < n; i++) s[i] /= total;
        }
    }
}
