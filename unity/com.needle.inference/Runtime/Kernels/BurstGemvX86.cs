// N32: Burst MatMulRows T=1 x86 port (Gemv.Rows/Block/Single, Avx2Dot).
//
// Ports src/Needle/Gemv.cs Rows/Block/Single (TwoBitPlanes, FourBitPlanes,
// Avx2Dot) 1:1 to Unity.Burst.Intrinsics. The x86 lane keeps the net10
// instruction order (H7): same loads, same unpack/shuffle/dot sequence, same
// FMA order, same hadd reduction order, same scalar tail order. Anything the
// net10 kernel does with a value, this file does in the same order, so
// fixtures that compare against the net10 AVX2 kernel must match bitwise
// (log maxAbsDiff in the conformance run; see the report in the N32 notes).
//
// Burst rules (plan §2.1 cookbook) this file follows:
//   - FloatMode.Strict on the entry point, never Fast: no reassociation.
//   - Explicit FMA intrinsics (mm256_fmadd_ps / mm256_fnmadd_ps); no reliance
//     on contraction.
//   - No [NoAlias]: rows alias nothing, but the table passes shared buffers.
//   - Pointers in, scalars out: no allocation, no static mutable state, safe
//     from any thread (pool workers call via the table).
//   - No data-dependent branches in the hot loops: one bits dispatch per
//     token, the same convertible-row split net10 uses, plain counted loops.
//
// H7 note: the x86 lane reproduces the BCL Vector256.Sum order used by
// Single (add low/high halves, two hadd passes, lane 0 out) and the
// hadd/add/final-store order used by Block, lane for lane. The Dot fold
// starts from the first madd (net10 adds a +0 first; x+0 is exact, so the
// elision is value-identical). Prefetch (Sse.Prefetch0) is omitted: it has
// no value effect and Burst schedules its own prefetches.
//
// H8 note: Burst X86 exposes Avx..Sse4_2/Ssse3 only (verified against the
// Burst 1.8.25 intrinsics sources; N31 settled the same for the installed
// Burst 2.x): no AVX-VNNI, no AVX-512. The integer dots therefore use the
// Avx2Dot sequence (maddubs+madd, plan cookbook), and only the non-permute
// plane layouts (TwoBitPlanes/FourBitPlanes) are ported: the Permute*
// layouts need AVX512-VBMI+VL, which Burst cannot emit.
//
// Cq/Activations/Codebook note: those structs are internal to the
// precompiled Needle.Inference.dll, which this source asmdef cannot name.
// The *Mirror structs below are layout-identical ABI mirrors (sequential,
// same fields in the same order, same Codebook payload offsets); the table
// ABI only moves pointers, so the reinterpretation is sound. Offsets
// verified against src/Needle/Model.cs (Cq/Codebook) and
// src/Needle/Quant.cs (Activations).
//
// Neon/arm note: N34 owns the tuned Neon port (BurstGemvNeon.cs). The Neon
// lane here runs the scalar algorithm (bit-exact vs Scalar.MatMulRows by
// construction) so Arm targets are correct until N34 lands; the x86 lane is
// the optimized one. Both lanes plus scalar are selected with the same
// compile-time-folded feature branches BurstSoftmax uses.
//
// C# 9 safe: block namespace, no target-typed new, no static lambdas.
using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;

namespace Needle.BurstKernels
{
    // Layout-identical mirror of the internal Codebook struct (Model.cs):
    // 16 float levels, 32 codebook bytes, float byte scale.
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct CodebookMirror
    {
        public fixed float Levels[16];
        public fixed byte Bytes[32];
        public float ByteScale;
    }

    // Layout-identical mirror of the internal Cq struct (Model.cs).
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct CqMirror
    {
        public byte* Indices;
        public ushort* Norms;
        public CodebookMirror* Book;
        public int Rows;
        public int Cols;
        public int Bits;
    }

    // Layout-identical mirror of the internal Activations struct (Quant.cs).
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct ActMirror
    {
        public sbyte* Values;
        public float* Scales;
        public float* Offsets;
        public int Cols;
        public int Groups;
        public int Stride;
    }

    public static unsafe class BurstGemvX86
    {
        const int GroupSize = 128;
        const int RowBlock = 4;
        const int HalvesPerVector = 8;
        const float HalfExponentRebias = 5.192296858534828e33f;

        // Cdecl delegate for the table slot:
        // void(Cq*, Activations*, int tokens, float* y, int ldy,
        //      int rowStart, int rowEnd).
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void MatMulRowsFn(
            CqMirror* w, ActMirror* a, int tokens, float* y, int ldy,
            int rowStart, int rowEnd);

        public static bool IsInstalled => Kernels.Active.MatMulRows != 0;

        // Idempotent: never overwrites an explicit or foreign provider's
        // MatMulRows entry (plan §2 selection order). The integration owner
        // calls this after/before BurstKernelInstaller.EnsureInstalled (order
        // is irrelevant; both refuse to overwrite) and from the Editor
        // bootstrap for EditMode conformance.
        public static void Install()
        {
            if (Kernels.Active.MatMulRows != 0)
                return;
            FunctionPointer<MatMulRowsFn> compiled =
                BurstCompiler.CompileFunctionPointer<MatMulRowsFn>(InvokeMatMulRows);
            if (!compiled.IsCreated)
                throw new InvalidOperationException(
                    "N32: Burst refused to compile BurstGemvX86.InvokeMatMulRows. " +
                    "The kernel must live in package source (H6); " +
                    "a precompiled DLL entry cannot be compiled.");
            KernelTable table = Kernels.Active;
            table.MatMulRows = (nint)compiled.Value;
            Kernels.Install(table, Kernels.IsInstalled ? Kernels.Provider + "+gemvx86" : "burst-gemvx86");
        }

        // Table entry point. Tokens loop over the T=1 Rows path (N32 scope);
        // multi-token tiles stay on the net10 path until N33.
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeMatMulRows(
            CqMirror* w, ActMirror* a, int tokens, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            // Branches fold at Burst compile time: exactly one lane survives
            // per target (x64 with AVX2+FMA, arm64 Neon, scalar elsewhere).
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                MatMulRowsX86(w, a, tokens, y, ldy, rowStart, rowEnd);
                return;
            }
            if (Arm.Neon.IsNeonSupported)
            {
                MatMulRowsScalar(w, a, tokens, y, ldy, rowStart, rowEnd);
                return;
            }
            MatMulRowsScalar(w, a, tokens, y, ldy, rowStart, rowEnd);
        }

        // Mirrors Gemv.Rows: per-token setup, then 4-row blocks plus scalar
        // single rows. One bits dispatch per token (loop invariant).
        static void MatMulRowsX86(
            CqMirror* w, ActMirror* a, int tokens, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            for (int t = 0; t < tokens; t++)
            {
                float* yt = y + (long)t * ldy;
                if (w->Bits == 2)
                    Rows2X86(w, a, t, yt, rowStart, rowEnd);
                else
                    Rows4X86(w, a, t, yt, rowStart, rowEnd);
            }
        }

        // Mirrors Gemv.Rows for TwoBitPlanes/Avx2Dot.
        static void Rows2X86(CqMirror* w, ActMirror* a, int token, float* y, int rowStart, int rowEnd)
        {
            int groups = w->Cols / GroupSize;
            int stride = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
            int cols = a->Cols;
            v256 codebook = X86.Avx.mm256_loadu_si256(w->Book->Bytes);
            // Mirrors TwoBitPlanes.Tables: First gathers codebook[i % 4],
            // Second gathers codebook[(i >> 2) & 3]; the v256 32-byte ctor
            // takes bytes in natural order, matching Vector256.Create.
            v256 first = X86.Avx2.mm256_shuffle_epi8(codebook, new v256(
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3));
            v256 second = X86.Avx2.mm256_shuffle_epi8(codebook, new v256(
                0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3));
            v256 mask = new v256((byte)0x0F);
            float* weightedScales = stackalloc float[stride];
            float* scaled = stackalloc float[RowBlock * stride];
            float* activationScales = a->Scales + token * stride;
            v256 byteScale = X86.Avx.mm256_set1_ps(w->Book->ByteScale * HalfExponentRebias);
            for (int v = 0; v < stride; v += HalvesPerVector)
                X86.Avx.mm256_storeu_ps(
                    weightedScales + v,
                    X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(activationScales + v), byteScale));
            float* offsets = a->Offsets + token * stride;
            sbyte* values = a->Values + (long)token * cols;
            int rowBytes = w->Cols * w->Bits / 8;
            int paddedGroups = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
            long total = (long)w->Rows * groups;
            int convertibleRows = total < paddedGroups ? 0 : (int)((total - paddedGroups) / groups) + 1;
            byte* packedBase = w->Indices;
            ushort* normsBase = w->Norms;
            int n = rowStart;
            for (; n + RowBlock <= rowEnd && n + RowBlock <= convertibleRows; n += RowBlock)
                Block2X86(
                    packedBase + (long)n * rowBytes, rowBytes,
                    normsBase + (long)n * groups, groups, stride,
                    codebook, first, second, mask,
                    weightedScales, offsets, values, scaled, y + n);
            for (; n < rowEnd; n++)
                y[n] = Single2X86(
                    packedBase + (long)n * rowBytes, normsBase + (long)n * groups,
                    groups, stride, n < convertibleRows,
                    codebook, first, second, mask,
                    weightedScales, offsets, values, scaled);
        }

        // Mirrors Gemv.Rows for FourBitPlanes/Avx2Dot.
        static void Rows4X86(CqMirror* w, ActMirror* a, int token, float* y, int rowStart, int rowEnd)
        {
            int groups = w->Cols / GroupSize;
            int stride = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
            int cols = a->Cols;
            // Mirrors FourBitPlanes.Tables: First is the codebook itself.
            v256 codebook = X86.Avx.mm256_loadu_si256(w->Book->Bytes);
            v256 mask = new v256((byte)0x0F);
            float* weightedScales = stackalloc float[stride];
            float* scaled = stackalloc float[RowBlock * stride];
            float* activationScales = a->Scales + token * stride;
            v256 byteScale = X86.Avx.mm256_set1_ps(w->Book->ByteScale * HalfExponentRebias);
            for (int v = 0; v < stride; v += HalvesPerVector)
                X86.Avx.mm256_storeu_ps(
                    weightedScales + v,
                    X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(activationScales + v), byteScale));
            float* offsets = a->Offsets + token * stride;
            sbyte* values = a->Values + (long)token * cols;
            int rowBytes = w->Cols * w->Bits / 8;
            int paddedGroups = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
            long total = (long)w->Rows * groups;
            int convertibleRows = total < paddedGroups ? 0 : (int)((total - paddedGroups) / groups) + 1;
            byte* packedBase = w->Indices;
            ushort* normsBase = w->Norms;
            int n = rowStart;
            for (; n + RowBlock <= rowEnd && n + RowBlock <= convertibleRows; n += RowBlock)
                Block4X86(
                    packedBase + (long)n * rowBytes, rowBytes,
                    normsBase + (long)n * groups, groups, stride,
                    codebook, mask,
                    weightedScales, offsets, values, scaled, y + n);
            for (; n < rowEnd; n++)
                y[n] = Single4X86(
                    packedBase + (long)n * rowBytes, normsBase + (long)n * groups,
                    groups, stride, n < convertibleRows,
                    codebook, mask,
                    weightedScales, offsets, values, scaled);
        }

        // Mirrors Gemv.Corrections: unbiased-half norms times weighted
        // scales, stored for the dot loop; the fnmadd chain accumulates the
        // offset correction in the same order as net10.
        static v256 CorrectionsX86(
            ushort* norms, int stride, float* weightedScales, float* offsets, float* scaled)
        {
            v256 correction = X86.Avx.mm256_set1_ps(0f);
            for (int v = 0; v < stride; v += HalvesPerVector)
            {
                // ConvertToVector256Int32(ushort*) zero-extends (vpmovzxwd);
                // the Burst cvtepu16 is the same instruction.
                v256 half = X86.Avx2.mm256_cvtepu16_epi32(X86.Sse2.loadu_si128(norms + v));
                v256 factors = X86.Avx.mm256_mul_ps(
                    X86.Avx.mm256_castsi256_ps(X86.Avx2.mm256_slli_epi32(half, 13)),
                    X86.Avx.mm256_loadu_ps(weightedScales + v));
                X86.Avx.mm256_storeu_ps(scaled + v, factors);
                correction = X86.Fma.mm256_fnmadd_ps(
                    factors, X86.Avx.mm256_loadu_ps(offsets + v), correction);
            }
            return correction;
        }

        // Mirrors Gemv.CorrectionsAtEdge: scalar zero-pad past groups, then
        // the same vector correction loop. The two counted loops replace the
        // per-element ternary with identical values (no branch).
        static v256 CorrectionsAtEdgeX86(
            ushort* norms, int groups, int stride,
            float* weightedScales, float* offsets, float* scaled)
        {
            for (int i = 0; i < groups; i++)
            {
                int bits = ((int)(short)norms[i]) << 13;
                scaled[i] = *(float*)&bits;
            }
            for (int i = groups; i < stride; i++)
                scaled[i] = 0f;
            v256 correction = X86.Avx.mm256_set1_ps(0f);
            for (int v = 0; v < stride; v += HalvesPerVector)
            {
                v256 factors = X86.Avx.mm256_mul_ps(
                    X86.Avx.mm256_loadu_ps(scaled + v),
                    X86.Avx.mm256_loadu_ps(weightedScales + v));
                X86.Avx.mm256_storeu_ps(scaled + v, factors);
                correction = X86.Fma.mm256_fnmadd_ps(
                    factors, X86.Avx.mm256_loadu_ps(offsets + v), correction);
            }
            return correction;
        }

        // Mirrors Avx2Dot.Dot: maddubs (u8*i8) then madd (i16*i16 pairwise)
        // per plane, accumulated in order. Net10 folds a +0 into the first
        // lane; x+0 is exact, so the fold starts from the first madd.
        static v256 Dot8X86(
            v256 p0, v256 p1, v256 p2, v256 p3,
            v256 x0, v256 x1, v256 x2, v256 x3)
        {
            v256 ones = new v256(0x00010001);
            v256 sum = X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(p0, x0), ones);
            sum = X86.Avx2.mm256_add_epi32(
                sum, X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(p1, x1), ones));
            sum = X86.Avx2.mm256_add_epi32(
                sum, X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(p2, x2), ones));
            return X86.Avx2.mm256_add_epi32(
                sum, X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(p3, x3), ones));
        }

        // Mirrors TwoBitPlanes.Expand.
        static void Expand2X86(
            byte* packed, v256 first, v256 second, v256 mask,
            out v256 p0, out v256 p1, out v256 p2, out v256 p3)
        {
            v256 v = X86.Avx.mm256_loadu_si256(packed);
            v256 lower = X86.Avx2.mm256_and_si256(v, mask);
            v256 upper = X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(v, 4), mask);
            p0 = X86.Avx2.mm256_shuffle_epi8(first, lower);
            p1 = X86.Avx2.mm256_shuffle_epi8(second, lower);
            p2 = X86.Avx2.mm256_shuffle_epi8(first, upper);
            p3 = X86.Avx2.mm256_shuffle_epi8(second, upper);
        }

        // Mirrors FourBitPlanes.Expand.
        static void Expand4X86(
            byte* packed, v256 codebook, v256 mask,
            out v256 p0, out v256 p1, out v256 p2, out v256 p3)
        {
            v256 lo = X86.Avx.mm256_loadu_si256(packed);
            v256 hi = X86.Avx.mm256_loadu_si256(packed + 32);
            p0 = X86.Avx2.mm256_shuffle_epi8(codebook, X86.Avx2.mm256_and_si256(lo, mask));
            p1 = X86.Avx2.mm256_shuffle_epi8(
                codebook, X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(lo, 4), mask));
            p2 = X86.Avx2.mm256_shuffle_epi8(codebook, X86.Avx2.mm256_and_si256(hi, mask));
            p3 = X86.Avx2.mm256_shuffle_epi8(
                codebook, X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(hi, 4), mask));
        }

        // Mirrors Gemv.Group for TwoBitPlanes/Avx2Dot.
        static v256 Group2X86(
            byte* packed, v256 first, v256 second, v256 mask,
            v256 x0, v256 x1, v256 x2, v256 x3, float* scale, v256 accumulator)
        {
            Expand2X86(packed, first, second, mask, out v256 p0, out v256 p1, out v256 p2, out v256 p3);
            return X86.Fma.mm256_fmadd_ps(
                X86.Avx.mm256_cvtepi32_ps(Dot8X86(p0, p1, p2, p3, x0, x1, x2, x3)),
                X86.Avx.mm256_broadcast_ss(scale),
                accumulator);
        }

        // Mirrors Gemv.Group for FourBitPlanes/Avx2Dot.
        static v256 Group4X86(
            byte* packed, v256 codebook, v256 mask,
            v256 x0, v256 x1, v256 x2, v256 x3, float* scale, v256 accumulator)
        {
            Expand4X86(packed, codebook, mask, out v256 p0, out v256 p1, out v256 p2, out v256 p3);
            return X86.Fma.mm256_fmadd_ps(
                X86.Avx.mm256_cvtepi32_ps(Dot8X86(p0, p1, p2, p3, x0, x1, x2, x3)),
                X86.Avx.mm256_broadcast_ss(scale),
                accumulator);
        }

        // Mirrors Gemv.Block for TwoBitPlanes/Avx2Dot: four correction rows,
        // the group loop over four packed rows, then the hadd/add/store fold
        // in net10 order. Prefetch is omitted (no value effect).
        static void Block2X86(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            v256 codebook, v256 first, v256 second, v256 mask,
            float* weightedScales, float* offsets, sbyte* values, float* scaled, float* y)
        {
            v256 start0 = CorrectionsX86(norms, stride, weightedScales, offsets, scaled);
            v256 start1 = CorrectionsX86(norms + groups, stride, weightedScales, offsets, scaled + stride);
            v256 start2 = CorrectionsX86(norms + 2 * groups, stride, weightedScales, offsets, scaled + 2 * stride);
            v256 start3 = CorrectionsX86(norms + 3 * groups, stride, weightedScales, offsets, scaled + 3 * stride);
            byte* row1 = packed + rowBytes;
            byte* row2 = packed + 2 * rowBytes;
            byte* row3 = packed + 3 * rowBytes;
            v256 accumulator0 = start0;
            v256 accumulator1 = start1;
            v256 accumulator2 = start2;
            v256 accumulator3 = start3;
            const int groupBytes = 32;
            for (int g = 0; g < groups; g++)
            {
                sbyte* x = values + g * GroupSize;
                v256 x0 = X86.Avx.mm256_loadu_si256(x);
                v256 x1 = X86.Avx.mm256_loadu_si256(x + 32);
                v256 x2 = X86.Avx.mm256_loadu_si256(x + 64);
                v256 x3 = X86.Avx.mm256_loadu_si256(x + 96);
                int offset = g * groupBytes;
                accumulator0 = Group2X86(
                    packed + offset, first, second, mask, x0, x1, x2, x3, scaled + g, accumulator0);
                accumulator1 = Group2X86(
                    row1 + offset, first, second, mask, x0, x1, x2, x3, scaled + stride + g, accumulator1);
                accumulator2 = Group2X86(
                    row2 + offset, first, second, mask, x0, x1, x2, x3, scaled + 2 * stride + g, accumulator2);
                accumulator3 = Group2X86(
                    row3 + offset, first, second, mask, x0, x1, x2, x3, scaled + 3 * stride + g, accumulator3);
            }
            v256 pairs01 = X86.Avx.mm256_hadd_ps(accumulator0, accumulator1);
            v256 pairs23 = X86.Avx.mm256_hadd_ps(accumulator2, accumulator3);
            v256 quads = X86.Avx.mm256_hadd_ps(pairs01, pairs23);
            X86.Sse.storeu_ps(
                y,
                X86.Sse.add_ps(
                    X86.Avx.mm256_castps256_ps128(quads),
                    X86.Avx.mm256_extractf128_ps(quads, 1)));
        }

        // Mirrors Gemv.Block for FourBitPlanes/Avx2Dot.
        static void Block4X86(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            v256 codebook, v256 mask,
            float* weightedScales, float* offsets, sbyte* values, float* scaled, float* y)
        {
            v256 start0 = CorrectionsX86(norms, stride, weightedScales, offsets, scaled);
            v256 start1 = CorrectionsX86(norms + groups, stride, weightedScales, offsets, scaled + stride);
            v256 start2 = CorrectionsX86(norms + 2 * groups, stride, weightedScales, offsets, scaled + 2 * stride);
            v256 start3 = CorrectionsX86(norms + 3 * groups, stride, weightedScales, offsets, scaled + 3 * stride);
            byte* row1 = packed + rowBytes;
            byte* row2 = packed + 2 * rowBytes;
            byte* row3 = packed + 3 * rowBytes;
            v256 accumulator0 = start0;
            v256 accumulator1 = start1;
            v256 accumulator2 = start2;
            v256 accumulator3 = start3;
            const int groupBytes = 64;
            for (int g = 0; g < groups; g++)
            {
                sbyte* x = values + g * GroupSize;
                v256 x0 = X86.Avx.mm256_loadu_si256(x);
                v256 x1 = X86.Avx.mm256_loadu_si256(x + 32);
                v256 x2 = X86.Avx.mm256_loadu_si256(x + 64);
                v256 x3 = X86.Avx.mm256_loadu_si256(x + 96);
                int offset = g * groupBytes;
                accumulator0 = Group4X86(
                    packed + offset, codebook, mask, x0, x1, x2, x3, scaled + g, accumulator0);
                accumulator1 = Group4X86(
                    row1 + offset, codebook, mask, x0, x1, x2, x3, scaled + stride + g, accumulator1);
                accumulator2 = Group4X86(
                    row2 + offset, codebook, mask, x0, x1, x2, x3, scaled + 2 * stride + g, accumulator2);
                accumulator3 = Group4X86(
                    row3 + offset, codebook, mask, x0, x1, x2, x3, scaled + 3 * stride + g, accumulator3);
            }
            v256 pairs01 = X86.Avx.mm256_hadd_ps(accumulator0, accumulator1);
            v256 pairs23 = X86.Avx.mm256_hadd_ps(accumulator2, accumulator3);
            v256 quads = X86.Avx.mm256_hadd_ps(pairs01, pairs23);
            X86.Sse.storeu_ps(
                y,
                X86.Sse.add_ps(
                    X86.Avx.mm256_castps256_ps128(quads),
                    X86.Avx.mm256_extractf128_ps(quads, 1)));
        }

        // Mirrors the BCL Vector256.Sum order used by Gemv.Single: add the
        // halves, then two hadd passes, lane 0 out.
        static float Sum8X86(v256 v)
        {
            v128 s = X86.Sse.add_ps(
                X86.Avx.mm256_castps256_ps128(v),
                X86.Avx.mm256_extractf128_ps(v, 1));
            v128 t = X86.Sse3.hadd_ps(s, s);
            t = X86.Sse3.hadd_ps(t, t);
            return X86.Sse.cvtss_f32(t);
        }

        // Mirrors Gemv.Single for TwoBitPlanes/Avx2Dot.
        static float Single2X86(
            byte* packed, ushort* norms, int groups, int stride, bool convertible,
            v256 codebook, v256 first, v256 second, v256 mask,
            float* weightedScales, float* offsets, sbyte* values, float* scaled)
        {
            v256 accumulator = convertible
                ? CorrectionsX86(norms, stride, weightedScales, offsets, scaled)
                : CorrectionsAtEdgeX86(norms, groups, stride, weightedScales, offsets, scaled);
            const int groupBytes = 32;
            for (int g = 0; g < groups; g++)
            {
                sbyte* x = values + g * GroupSize;
                accumulator = Group2X86(
                    packed + g * groupBytes, first, second, mask,
                    X86.Avx.mm256_loadu_si256(x),
                    X86.Avx.mm256_loadu_si256(x + 32),
                    X86.Avx.mm256_loadu_si256(x + 64),
                    X86.Avx.mm256_loadu_si256(x + 96),
                    scaled + g, accumulator);
            }
            return Sum8X86(accumulator);
        }

        // Mirrors Gemv.Single for FourBitPlanes/Avx2Dot.
        static float Single4X86(
            byte* packed, ushort* norms, int groups, int stride, bool convertible,
            v256 codebook, v256 mask,
            float* weightedScales, float* offsets, sbyte* values, float* scaled)
        {
            v256 accumulator = convertible
                ? CorrectionsX86(norms, stride, weightedScales, offsets, scaled)
                : CorrectionsAtEdgeX86(norms, groups, stride, weightedScales, offsets, scaled);
            const int groupBytes = 64;
            for (int g = 0; g < groups; g++)
            {
                sbyte* x = values + g * GroupSize;
                accumulator = Group4X86(
                    packed + g * groupBytes, codebook, mask,
                    X86.Avx.mm256_loadu_si256(x),
                    X86.Avx.mm256_loadu_si256(x + 32),
                    X86.Avx.mm256_loadu_si256(x + 64),
                    X86.Avx.mm256_loadu_si256(x + 96),
                    scaled + g, accumulator);
            }
            return Sum8X86(accumulator);
        }

        // Scalar lane for targets with neither AVX2+FMA nor Neon, and the
        // Neon-lane body until N34 lands the tuned Neon dotprod port:
        // mirrors Scalar.MatMulRows exactly (same products, same order).
        static void MatMulRowsScalar(
            CqMirror* w, ActMirror* a, int tokens, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            int groups = w->Cols / GroupSize;
            byte* table = w->Book->Bytes;
            float byteScale = w->Book->ByteScale;
            for (int t = 0; t < tokens; t++)
            {
                sbyte* values = a->Values + (long)t * a->Cols;
                float* scales = a->Scales + t * a->Stride;
                float* offsets = a->Offsets + t * a->Stride;
                for (int n = rowStart; n < rowEnd; n++)
                {
                    byte* packed = w->Indices + (long)n * (w->Cols * w->Bits / 8);
                    ushort* norms = w->Norms + (long)n * groups;
                    float accumulator = 0f;
                    float bias = 0f;
                    for (int g = 0; g < groups; g++)
                    {
                        sbyte* x = values + g * GroupSize;
                        int dot = 0;
                        if (w->Bits == 2)
                        {
                            byte* p = packed + g * 32;
                            for (int j = 0; j < 32; j++)
                            {
                                int b = p[j];
                                dot += table[b & 3] * x[j]
                                    + table[b >> 2 & 3] * x[32 + j]
                                    + table[b >> 4 & 3] * x[64 + j]
                                    + table[b >> 6 & 3] * x[96 + j];
                            }
                        }
                        else
                        {
                            byte* p = packed + g * 64;
                            for (int half = 0; half < 2; half++)
                                for (int j = 0; j < 32; j++)
                                {
                                    int b = p[half * 32 + j];
                                    dot += table[b & 15] * x[2 * half * 32 + j]
                                        + table[b >> 4] * x[(2 * half + 1) * 32 + j];
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

        // Mirrors Scalar.HalfToSingle bit for bit.
        static float HalfToSingle(ushort half)
        {
            uint magnitude = half & 0x7FFFu;
            if (magnitude == 0)
                return 0f;
            if (magnitude < 0x0400u)
                return (half & 0x8000u) != 0 ? -(float)(int)magnitude * 5.9604645e-8f : (float)(int)magnitude * 5.9604645e-8f;
            uint bits = (magnitude << 13) + 0x38000000u | (uint)(half & 0x8000) << 16;
            return *(float*)&bits;
        }
    }
}
