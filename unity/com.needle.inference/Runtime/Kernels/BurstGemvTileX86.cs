// N33: Burst T>=2 GEMV tiles, x86 (Tile4, Tile2, Pair4, Row4, Pair2, Row2).
//
// Ports Gemv.Tile4/Tile2 (and their Pair4/Row4/Pair2/Row2 workers) 1:1 to
// Unity.Burst.Intrinsics for the AVX2+FMA lane: the same plane expansion,
// the same Avx2Dot integer sequence, the same FMA correction order and the
// same horizontal reductions as the net10 kernel. Anything else (Neon, old
// x86 without FMA) falls back to an exact scalar port of Scalar.MatMulRows,
// so every target runs correctly today and N34 can add the Neon lane later.
//
// Cookbook mapping (plan-netstandard.md section 2), net10 -> Burst:
//   Avx.LoadVector256(float*)              -> X86.Avx.mm256_loadu_ps
//   Avx.Store(float*)                      -> X86.Avx.mm256_storeu_ps
//   Avx.LoadVector256(byte*/sbyte*)        -> X86.Avx.mm256_loadu_si256
//   Avx2.ConvertToVector256Int32(ushort*)  -> mm256_cvtepu16_epi32(loadu_si128)
//   Avx2.ShiftLeftLogical(_, 13).AsSingle() -> mm256_castsi256_ps(mm256_slli_epi32)
//   Avx.Multiply (float)                    -> X86.Avx.mm256_mul_ps
//   Fma.MultiplyAddNegated                 -> X86.Fma.mm256_fnmadd_ps
//   Fma.MultiplyAdd                        -> X86.Fma.mm256_fmadd_ps
//   Avx.BroadcastScalarToVector256(float*) -> X86.Avx.mm256_broadcast_ss
//   Avx.ConvertToVector256Single(int)      -> X86.Avx.mm256_cvtepi32_ps
//   Avx2.MultiplyAddAdjacent(byte, sbyte)  -> X86.Avx2.mm256_maddubs_epi16
//   Avx2.MultiplyAddAdjacent(short, short) -> X86.Avx2.mm256_madd_epi16
//   Avx2.Add (int)                         -> X86.Avx2.mm256_add_epi32
//   Avx2.Shuffle (byte)                    -> X86.Avx2.mm256_shuffle_epi8
//   Avx2.And (byte)                        -> X86.Avx2.mm256_and_si256
//   Avx2.ShiftRightLogical(u16, 4)         -> X86.Avx2.mm256_srli_epi16
//   Avx.HorizontalAdd                      -> X86.Avx.mm256_hadd_ps
//   GetLower/GetUpper                      -> mm256_castps256_ps128/mm256_extractf128_ps
//   Sse.Add/Sse41.Extract lane reads       -> add_ps + shuffle_ps/cvtss_f32
//                                            (pure lane moves, same values)
//   Sse.Prefetch0                          -> omitted (cache hint, no numerics)
//   Vector256.Sum                          -> Sum128(lower) + Sum128(upper),
//                                            Sum128 = two hadd_ps + cvtss_f32
//                                            (the BCL order, probed on x64)
//
// Deliberately NOT ported ("No branches"):
//   - PermuteTwoBit/PermuteFourBitPlanes: need AVX512-VBMI+VL
//     (PermuteVar32x8). The non-permute TwoBit/FourBitPlanes path is the
//     portable AVX2 baseline; bitwise-vs-net10 holds whenever net10 runs it
//     (no-VBMI hardware, or NEEDLE_NOAVX512=1).
//   - Tile4Panel/Tile2Panel (NEEDLE_PANEL=1): an opt-in repack layout, not
//     the default path. Default path only.
//   - VnniDot/VnniInt8Dot/BwDot: H8 holds (Burst exposes no AVX-VNNI and no
//     AVX-512), so every dot product uses the Avx2Dot sequence per plan.
//   - The tier dispatch itself: one entry checks
//     IsAvx2Supported && IsFmaSupported (folds at Burst compile time) and
//     the bits dispatch (2 vs 4) happens once per call, never in a group
//     loop; each (tile x bits) worker below is straight-line vector code.
//
// H7 note: each worker keeps the net10 instruction order op for op (FMA
// correction chain, per-group fmadd accumulate, hadd/sum reduction order),
// so the tile-covered tokens compare bitwise with the net10 AVX2 kernel.
// The int32 dot sums are additionally bit-exact with Scalar.MatMulRows
// (same integer arithmetic); the float values meet the N13 parity bound
// (cosine >= 0.99995, same argmax), as net10's own tiles do.
// Odd token counts leave one T=1 tail row range: it runs the exact scalar
// reference here (the x86 T=1 Single port belongs to N32's BurstGemvX86.cs;
// integration routes tails there once it lands).
//
// ABI notes:
//   - GemvWeight/GemvActivations/GemvCodebook are layout mirrors of the
//     engine-internal Cq/Activations/Codebook (same field order and types,
//     so the same sizes: 40/40/104 bytes). The Burst asmdef cannot see the
//     engine internals (no InternalsVisibleTo), and pointers cross the table
//     opaquely, so mirrors keep the call ABI identical.
//   - Blittable arguments only; no allocation (stackalloc scratch mirrors
//     the net10 Tile4/Tile2 frames); no static mutable state; safe from any
//     thread. Install() never overwrites an already-installed MatMulRows
//     entry (plan section 2 selection order).
//
// C# 9 safe: block namespace, no target-typed new, no static lambdas.
using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;

namespace Needle.BurstKernels
{
    // Layout mirror of internal Needle.Codebook: 16 float levels, 32
    // byte codebook entries, one float scale. Must stay field-identical.
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct GemvCodebook
    {
        public fixed float Levels[16];
        public fixed byte Bytes[32];
        public float ByteScale;
    }

    // Layout mirror of internal Needle.Cq (Groups = Cols / 128,
    // RowBytes = Cols * Bits / 8, same as the engine).
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct GemvWeight
    {
        public byte* Indices;
        public ushort* Norms;
        public GemvCodebook* Book;
        public int Rows;
        public int Cols;
        public int Bits;
    }

    // Layout mirror of internal Needle.Activations
    // (Groups = Cols / 128, Stride = (Groups + 7) & ~7).
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct GemvActivations
    {
        public sbyte* Values;
        public float* Scales;
        public float* Offsets;
        public int Cols;
        public int Groups;
        public int Stride;
    }

    public static unsafe class BurstGemvTileX86
    {
        const float HalfExponentRebias = 5.192296858534828e33f;
        const int GroupSize = 128;
        const int HalvesPerVector = 8;

        // Packed-lane constants, expressed as int32 broadcasts so no shuffle
        // control tables are needed: 0x0F0F0F0F is 32 bytes of 0x0F (the
        // TwoBit/FourBit nibble mask), 0x03020100 is bytes (0,1,2,3) x8 (the
        // TwoBit first-plane control), 0x00010001 is 16 int16 ones (the
        // Avx2Dot madd_epi16 factor). Each matches a Vector256.Create value.
        const int NibbleMaskWord = 0x0F0F0F0F;
        const int FirstPlaneWord = 0x03020100;
        const int ShortOnesWord = 0x00010001;

        // Cdecl delegates for CompileFunctionPointer. TileFn mirrors one
        // Gemv.Tile4/Tile2 chunk (exactly 4 / 2 tokens at `token`);
        // TileMatMulRowsFn mirrors the KernelTable.MatMulRows slot. (Named
        // apart from BurstGemvX86.MatMulRowsFn: same namespace, one type
        // per name; the table slot takes either function pointer.)
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void TileFn(
            GemvWeight* w, GemvActivations* a, int token, float* y, int ldy,
            int rowStart, int rowEnd);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void TileMatMulRowsFn(
            GemvWeight* w, GemvActivations* a, int tokens, float* y, int ldy,
            int rowStart, int rowEnd);

        public static bool IsInstalled => Needle.Kernels.Active.MatMulRows != 0;

        // Installs the MatMulRows dispatcher into the kernel table unless a
        // provider already claimed the slot. Idempotent; integration calls
        // this from the Burst installer (N31) once the N32 entry exists.
        public static void Install()
        {
            if (Needle.Kernels.Active.MatMulRows != 0) return;
            FunctionPointer<TileMatMulRowsFn> compiled =
                BurstCompiler.CompileFunctionPointer<TileMatMulRowsFn>(MatMulRows);
            if (!compiled.IsCreated)
                throw new InvalidOperationException(
                    "N33: Burst refused to compile BurstGemvTileX86.MatMulRows. " +
                    "The kernel must live in package source (H6); " +
                    "a precompiled DLL entry cannot be compiled.");
            Needle.KernelTable table = Needle.Kernels.Active;
            table.MatMulRows = (nint)compiled.Value;
            Needle.Kernels.Install(
                table,
                Needle.Kernels.IsInstalled ? Needle.Kernels.Provider + "+gemvtile" : "burst-gemvtile");
        }

        // Full §2.1 MatMulRows provider: 4-token Tile4 chunks, one 2-token
        // Tile2 chunk when the remainder allows, exact-scalar T=1 tails.
        // Mirrors Quant.MatMulRows chunking (tiledTokens = tokens & ~3,
        // then >= 2, then singles) so every token runs the same worker the
        // net10 build would run, except T=1 tails (see the H7 note above).
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void MatMulRows(
            GemvWeight* w, GemvActivations* a, int tokens, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            if (tokens <= 0 || rowEnd <= rowStart) return;
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                bool twoBit = w->Bits == 2;
                int tiled = tokens & ~3;
                if (twoBit)
                {
                    int t = 0;
                    for (; t < tiled; t += 4)
                        Tile4_2(w, a, t, y + (long)t * ldy, ldy, rowStart, rowEnd);
                    if (tokens - t >= 2)
                    {
                        Tile2_2(w, a, t, y + (long)t * ldy, ldy, rowStart, rowEnd);
                        t += 2;
                    }
                    for (; t < tokens; t++)
                        ScalarSingle(w, a, t, y + (long)t * ldy, rowStart, rowEnd);
                }
                else
                {
                    int t = 0;
                    for (; t < tiled; t += 4)
                        Tile4_4(w, a, t, y + (long)t * ldy, ldy, rowStart, rowEnd);
                    if (tokens - t >= 2)
                    {
                        Tile2_4(w, a, t, y + (long)t * ldy, ldy, rowStart, rowEnd);
                        t += 2;
                    }
                    for (; t < tokens; t++)
                        ScalarSingle(w, a, t, y + (long)t * ldy, rowStart, rowEnd);
                }
                return;
            }
            ScalarTile(w, a, 0, tokens, y, ldy, rowStart, rowEnd);
        }

        // One Gemv.Tile4 chunk: exactly 4 tokens at `token`. Granular Cdecl
        // entry for tests and for integration-built dispatchers.
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void Tile4(
            GemvWeight* w, GemvActivations* a, int token, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            if (rowEnd <= rowStart) return;
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                if (w->Bits == 2) Tile4_2(w, a, token, y, ldy, rowStart, rowEnd);
                else Tile4_4(w, a, token, y, ldy, rowStart, rowEnd);
                return;
            }
            ScalarTile(w, a, token, 4, y, ldy, rowStart, rowEnd);
        }

        // One Gemv.Tile2 chunk: exactly 2 tokens at `token`.
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void Tile2(
            GemvWeight* w, GemvActivations* a, int token, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            if (rowEnd <= rowStart) return;
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                if (w->Bits == 2) Tile2_2(w, a, token, y, ldy, rowStart, rowEnd);
                else Tile2_4(w, a, token, y, ldy, rowStart, rowEnd);
                return;
            }
            ScalarTile(w, a, token, 2, y, ldy, rowStart, rowEnd);
        }

        // Gemv.Tile4<TwoBitPlanes, Avx2Dot>: mirrors the net10 worker,
        // including stackalloc frame sizes (4*stride weights, 8*stride dot
        // scales, 2*stride row factors) and the pair/row loop guards.
        static void Tile4_2(
            GemvWeight* w, GemvActivations* a, int token, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            int groups = w->Cols / GroupSize;
            int stride = a->Stride;
            int cols = a->Cols;
            v256 codebook = X86.Avx.mm256_loadu_si256(w->Book->Bytes);
            v256 first = X86.Avx2.mm256_shuffle_epi8(codebook, new v256(FirstPlaneWord));
            byte* control = stackalloc byte[32];
            for (int k = 0; k < 32; k++) control[k] = (byte)((k >> 2) & 3);
            v256 second = X86.Avx2.mm256_shuffle_epi8(
                codebook, X86.Avx.mm256_loadu_si256(control));
            v256 mask = new v256(NibbleMaskWord);
            float* weightedScales = stackalloc float[4 * stride];
            float* scaled = stackalloc float[8 * stride];
            float* rowFactors = stackalloc float[2 * stride];
            v256 byteScale = X86.Avx.mm256_set1_ps(w->Book->ByteScale * HalfExponentRebias);
            for (int t = 0; t < 4; t++)
                for (int v = 0; v < stride; v += HalvesPerVector)
                    X86.Avx.mm256_storeu_ps(
                        weightedScales + t * stride + v,
                        X86.Avx.mm256_mul_ps(
                            X86.Avx.mm256_loadu_ps(a->Scales + (token + t) * stride + v),
                            byteScale));
            float* offsets = a->Offsets + token * stride;
            sbyte* values = a->Values + (long)token * cols;
            int rowBytes = (w->Cols * w->Bits) / 8;
            int paddedGroups = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
            long rowsAndGroups = (long)w->Rows * groups;
            int convertibleRows = rowsAndGroups < paddedGroups
                ? 0 : (int)((rowsAndGroups - paddedGroups) / groups) + 1;
            int n = rowStart;
            for (; n + 2 <= rowEnd && n + 2 <= convertibleRows; n += 2)
                Pair4_2(
                    w->Indices + (long)n * rowBytes, rowBytes,
                    w->Norms + (long)n * groups, groups, stride, cols,
                    first, second, mask, weightedScales, offsets, values,
                    scaled, rowFactors, y + n, ldy);
            for (; n < rowEnd; n++)
                Row4_2(
                    w->Indices + (long)n * rowBytes, w->Norms + (long)n * groups,
                    n < convertibleRows, groups, stride, cols, first, second, mask,
                    weightedScales, offsets, values, scaled, rowFactors, y + n, ldy);
        }

        // Gemv.Tile4<FourBitPlanes, Avx2Dot>.
        static void Tile4_4(
            GemvWeight* w, GemvActivations* a, int token, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            int groups = w->Cols / GroupSize;
            int stride = a->Stride;
            int cols = a->Cols;
            v256 codebook = X86.Avx.mm256_loadu_si256(w->Book->Bytes);
            v256 mask = new v256(NibbleMaskWord);
            float* weightedScales = stackalloc float[4 * stride];
            float* scaled = stackalloc float[8 * stride];
            float* rowFactors = stackalloc float[2 * stride];
            v256 byteScale = X86.Avx.mm256_set1_ps(w->Book->ByteScale * HalfExponentRebias);
            for (int t = 0; t < 4; t++)
                for (int v = 0; v < stride; v += HalvesPerVector)
                    X86.Avx.mm256_storeu_ps(
                        weightedScales + t * stride + v,
                        X86.Avx.mm256_mul_ps(
                            X86.Avx.mm256_loadu_ps(a->Scales + (token + t) * stride + v),
                            byteScale));
            float* offsets = a->Offsets + token * stride;
            sbyte* values = a->Values + (long)token * cols;
            int rowBytes = (w->Cols * w->Bits) / 8;
            int paddedGroups = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
            long rowsAndGroups = (long)w->Rows * groups;
            int convertibleRows = rowsAndGroups < paddedGroups
                ? 0 : (int)((rowsAndGroups - paddedGroups) / groups) + 1;
            int n = rowStart;
            for (; n + 2 <= rowEnd && n + 2 <= convertibleRows; n += 2)
                Pair4_4(
                    w->Indices + (long)n * rowBytes, rowBytes,
                    w->Norms + (long)n * groups, groups, stride, cols,
                    codebook, mask, weightedScales, offsets, values,
                    scaled, rowFactors, y + n, ldy);
            for (; n < rowEnd; n++)
                Row4_4(
                    w->Indices + (long)n * rowBytes, w->Norms + (long)n * groups,
                    n < convertibleRows, groups, stride, cols, codebook, mask,
                    weightedScales, offsets, values, scaled, rowFactors, y + n, ldy);
        }

        // Gemv.Tile2<TwoBitPlanes, Avx2Dot>.
        static void Tile2_2(
            GemvWeight* w, GemvActivations* a, int token, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            int groups = w->Cols / GroupSize;
            int stride = a->Stride;
            int cols = a->Cols;
            v256 codebook = X86.Avx.mm256_loadu_si256(w->Book->Bytes);
            v256 first = X86.Avx2.mm256_shuffle_epi8(codebook, new v256(FirstPlaneWord));
            byte* control = stackalloc byte[32];
            for (int k = 0; k < 32; k++) control[k] = (byte)((k >> 2) & 3);
            v256 second = X86.Avx2.mm256_shuffle_epi8(
                codebook, X86.Avx.mm256_loadu_si256(control));
            v256 mask = new v256(NibbleMaskWord);
            float* weightedScales = stackalloc float[2 * stride];
            float* scaled = stackalloc float[4 * stride];
            float* rowFactors = stackalloc float[2 * stride];
            v256 byteScale = X86.Avx.mm256_set1_ps(w->Book->ByteScale * HalfExponentRebias);
            for (int t = 0; t < 2; t++)
                for (int v = 0; v < stride; v += HalvesPerVector)
                    X86.Avx.mm256_storeu_ps(
                        weightedScales + t * stride + v,
                        X86.Avx.mm256_mul_ps(
                            X86.Avx.mm256_loadu_ps(a->Scales + (token + t) * stride + v),
                            byteScale));
            float* offsets = a->Offsets + token * stride;
            sbyte* values = a->Values + (long)token * cols;
            int rowBytes = (w->Cols * w->Bits) / 8;
            int paddedGroups = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
            long rowsAndGroups = (long)w->Rows * groups;
            int convertibleRows = rowsAndGroups < paddedGroups
                ? 0 : (int)((rowsAndGroups - paddedGroups) / groups) + 1;
            int n = rowStart;
            for (; n + 2 <= rowEnd && n + 2 <= convertibleRows; n += 2)
                Pair2_2(
                    w->Indices + (long)n * rowBytes, rowBytes,
                    w->Norms + (long)n * groups, groups, stride, cols,
                    first, second, mask, weightedScales, offsets, values,
                    scaled, rowFactors, y + n, ldy);
            for (; n < rowEnd; n++)
                Row2_2(
                    w->Indices + (long)n * rowBytes, w->Norms + (long)n * groups,
                    n < convertibleRows, groups, stride, cols, first, second, mask,
                    weightedScales, offsets, values, scaled, rowFactors, y + n, ldy);
        }

        // Gemv.Tile2<FourBitPlanes, Avx2Dot>.
        static void Tile2_4(
            GemvWeight* w, GemvActivations* a, int token, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            int groups = w->Cols / GroupSize;
            int stride = a->Stride;
            int cols = a->Cols;
            v256 codebook = X86.Avx.mm256_loadu_si256(w->Book->Bytes);
            v256 mask = new v256(NibbleMaskWord);
            float* weightedScales = stackalloc float[2 * stride];
            float* scaled = stackalloc float[4 * stride];
            float* rowFactors = stackalloc float[2 * stride];
            v256 byteScale = X86.Avx.mm256_set1_ps(w->Book->ByteScale * HalfExponentRebias);
            for (int t = 0; t < 2; t++)
                for (int v = 0; v < stride; v += HalvesPerVector)
                    X86.Avx.mm256_storeu_ps(
                        weightedScales + t * stride + v,
                        X86.Avx.mm256_mul_ps(
                            X86.Avx.mm256_loadu_ps(a->Scales + (token + t) * stride + v),
                            byteScale));
            float* offsets = a->Offsets + token * stride;
            sbyte* values = a->Values + (long)token * cols;
            int rowBytes = (w->Cols * w->Bits) / 8;
            int paddedGroups = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
            long rowsAndGroups = (long)w->Rows * groups;
            int convertibleRows = rowsAndGroups < paddedGroups
                ? 0 : (int)((rowsAndGroups - paddedGroups) / groups) + 1;
            int n = rowStart;
            for (; n + 2 <= rowEnd && n + 2 <= convertibleRows; n += 2)
                Pair2_4(
                    w->Indices + (long)n * rowBytes, rowBytes,
                    w->Norms + (long)n * groups, groups, stride, cols,
                    codebook, mask, weightedScales, offsets, values,
                    scaled, rowFactors, y + n, ldy);
            for (; n < rowEnd; n++)
                Row2_4(
                    w->Indices + (long)n * rowBytes, w->Norms + (long)n * groups,
                    n < convertibleRows, groups, stride, cols, codebook, mask,
                    weightedScales, offsets, values, scaled, rowFactors, y + n, ldy);
        }

        // Gemv.Pair4<TwoBitPlanes, Avx2Dot>: two rows x four tokens. The
        // eight accumulators, the per-group expand/accumulate order and the
        // hadd-tree store match net10 exactly.
        static void Pair4_2(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            int cols, v256 first, v256 second, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            for (int v = 0; v < stride; v += HalvesPerVector)
            {
                X86.Avx.mm256_storeu_ps(
                    rowFactors + v,
                    UnbiasedHalf(X86.Sse2.loadu_si128(norms + v)));
                X86.Avx.mm256_storeu_ps(
                    rowFactors + stride + v,
                    UnbiasedHalf(X86.Sse2.loadu_si128(norms + groups + v)));
            }
            v256 a00 = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
            v256 a01 = TileCorrections(
                rowFactors, weightedScales + stride, offsets + stride, scaled + stride, stride);
            v256 a02 = TileCorrections(
                rowFactors, weightedScales + 2 * stride, offsets + 2 * stride,
                scaled + 2 * stride, stride);
            v256 a03 = TileCorrections(
                rowFactors, weightedScales + 3 * stride, offsets + 3 * stride,
                scaled + 3 * stride, stride);
            v256 a10 = TileCorrections(
                rowFactors + stride, weightedScales, offsets, scaled + 4 * stride, stride);
            v256 a11 = TileCorrections(
                rowFactors + stride, weightedScales + stride, offsets + stride,
                scaled + 5 * stride, stride);
            v256 a12 = TileCorrections(
                rowFactors + stride, weightedScales + 2 * stride, offsets + 2 * stride,
                scaled + 6 * stride, stride);
            v256 a13 = TileCorrections(
                rowFactors + stride, weightedScales + 3 * stride, offsets + 3 * stride,
                scaled + 7 * stride, stride);
            byte* row1 = packed + rowBytes;
            for (int g = 0; g < groups; g++)
            {
                v256 v0 = X86.Avx.mm256_loadu_si256(packed + g * 32);
                v256 lower = X86.Avx2.mm256_and_si256(v0, mask);
                v256 upper = X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(v0, 4), mask);
                v256 p0 = X86.Avx2.mm256_shuffle_epi8(first, lower);
                v256 p1 = X86.Avx2.mm256_shuffle_epi8(second, lower);
                v256 p2 = X86.Avx2.mm256_shuffle_epi8(first, upper);
                v256 p3 = X86.Avx2.mm256_shuffle_epi8(second, upper);
                v256 w0 = X86.Avx.mm256_loadu_si256(row1 + g * 32);
                v256 lower1 = X86.Avx2.mm256_and_si256(w0, mask);
                v256 upper1 = X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(w0, 4), mask);
                v256 q0 = X86.Avx2.mm256_shuffle_epi8(first, lower1);
                v256 q1 = X86.Avx2.mm256_shuffle_epi8(second, lower1);
                v256 q2 = X86.Avx2.mm256_shuffle_epi8(first, upper1);
                v256 q3 = X86.Avx2.mm256_shuffle_epi8(second, upper1);
                sbyte* x = values + g * GroupSize;
                sbyte* x1 = x + cols;
                sbyte* x2 = x + 2 * cols;
                sbyte* x3 = x + 3 * cols;
                a00 = AccumulateX86(p0, p1, p2, p3, x, scaled + g, a00);
                a10 = AccumulateX86(q0, q1, q2, q3, x, scaled + 4 * stride + g, a10);
                a01 = AccumulateX86(p0, p1, p2, p3, x1, scaled + stride + g, a01);
                a11 = AccumulateX86(q0, q1, q2, q3, x1, scaled + 5 * stride + g, a11);
                a02 = AccumulateX86(p0, p1, p2, p3, x2, scaled + 2 * stride + g, a02);
                a12 = AccumulateX86(q0, q1, q2, q3, x2, scaled + 6 * stride + g, a12);
                a03 = AccumulateX86(p0, p1, p2, p3, x3, scaled + 3 * stride + g, a03);
                a13 = AccumulateX86(q0, q1, q2, q3, x3, scaled + 7 * stride + g, a13);
            }
            StoreQuad4(a00, a01, a02, a03, y, ldy, 0);
            StoreQuad4(a10, a11, a12, a13, y, ldy, 1);
        }

        // Gemv.Pair4<FourBitPlanes, Avx2Dot>.
        static void Pair4_4(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            int cols, v256 codebook, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            for (int v = 0; v < stride; v += HalvesPerVector)
            {
                X86.Avx.mm256_storeu_ps(
                    rowFactors + v,
                    UnbiasedHalf(X86.Sse2.loadu_si128(norms + v)));
                X86.Avx.mm256_storeu_ps(
                    rowFactors + stride + v,
                    UnbiasedHalf(X86.Sse2.loadu_si128(norms + groups + v)));
            }
            v256 a00 = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
            v256 a01 = TileCorrections(
                rowFactors, weightedScales + stride, offsets + stride, scaled + stride, stride);
            v256 a02 = TileCorrections(
                rowFactors, weightedScales + 2 * stride, offsets + 2 * stride,
                scaled + 2 * stride, stride);
            v256 a03 = TileCorrections(
                rowFactors, weightedScales + 3 * stride, offsets + 3 * stride,
                scaled + 3 * stride, stride);
            v256 a10 = TileCorrections(
                rowFactors + stride, weightedScales, offsets, scaled + 4 * stride, stride);
            v256 a11 = TileCorrections(
                rowFactors + stride, weightedScales + stride, offsets + stride,
                scaled + 5 * stride, stride);
            v256 a12 = TileCorrections(
                rowFactors + stride, weightedScales + 2 * stride, offsets + 2 * stride,
                scaled + 6 * stride, stride);
            v256 a13 = TileCorrections(
                rowFactors + stride, weightedScales + 3 * stride, offsets + 3 * stride,
                scaled + 7 * stride, stride);
            byte* row1 = packed + rowBytes;
            for (int g = 0; g < groups; g++)
            {
                v256 lo = X86.Avx.mm256_loadu_si256(packed + g * 64);
                v256 hi = X86.Avx.mm256_loadu_si256(packed + g * 64 + 32);
                v256 p0 = X86.Avx2.mm256_shuffle_epi8(
                    codebook, X86.Avx2.mm256_and_si256(lo, mask));
                v256 p1 = X86.Avx2.mm256_shuffle_epi8(
                    codebook,
                    X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(lo, 4), mask));
                v256 p2 = X86.Avx2.mm256_shuffle_epi8(
                    codebook, X86.Avx2.mm256_and_si256(hi, mask));
                v256 p3 = X86.Avx2.mm256_shuffle_epi8(
                    codebook,
                    X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(hi, 4), mask));
                v256 lo1 = X86.Avx.mm256_loadu_si256(row1 + g * 64);
                v256 hi1 = X86.Avx.mm256_loadu_si256(row1 + g * 64 + 32);
                v256 q0 = X86.Avx2.mm256_shuffle_epi8(
                    codebook, X86.Avx2.mm256_and_si256(lo1, mask));
                v256 q1 = X86.Avx2.mm256_shuffle_epi8(
                    codebook,
                    X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(lo1, 4), mask));
                v256 q2 = X86.Avx2.mm256_shuffle_epi8(
                    codebook, X86.Avx2.mm256_and_si256(hi1, mask));
                v256 q3 = X86.Avx2.mm256_shuffle_epi8(
                    codebook,
                    X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(hi1, 4), mask));
                sbyte* x = values + g * GroupSize;
                sbyte* x1 = x + cols;
                sbyte* x2 = x + 2 * cols;
                sbyte* x3 = x + 3 * cols;
                a00 = AccumulateX86(p0, p1, p2, p3, x, scaled + g, a00);
                a10 = AccumulateX86(q0, q1, q2, q3, x, scaled + 4 * stride + g, a10);
                a01 = AccumulateX86(p0, p1, p2, p3, x1, scaled + stride + g, a01);
                a11 = AccumulateX86(q0, q1, q2, q3, x1, scaled + 5 * stride + g, a11);
                a02 = AccumulateX86(p0, p1, p2, p3, x2, scaled + 2 * stride + g, a02);
                a12 = AccumulateX86(q0, q1, q2, q3, x2, scaled + 6 * stride + g, a12);
                a03 = AccumulateX86(p0, p1, p2, p3, x3, scaled + 3 * stride + g, a03);
                a13 = AccumulateX86(q0, q1, q2, q3, x3, scaled + 7 * stride + g, a13);
            }
            StoreQuad4(a00, a01, a02, a03, y, ldy, 0);
            StoreQuad4(a10, a11, a12, a13, y, ldy, 1);
        }

        // Gemv.Row4<TwoBitPlanes, Avx2Dot>: one row x four tokens.
        static void Row4_2(
            byte* packed, ushort* norms, bool convertible, int groups, int stride,
            int cols, v256 first, v256 second, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            if (convertible)
                for (int v = 0; v < stride; v += HalvesPerVector)
                    X86.Avx.mm256_storeu_ps(
                        rowFactors + v,
                        UnbiasedHalf(X86.Sse2.loadu_si128(norms + v)));
            else
                EdgeRowFactors(norms, groups, rowFactors, stride);
            v256 a0 = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
            v256 a1 = TileCorrections(
                rowFactors, weightedScales + stride, offsets + stride, scaled + stride, stride);
            v256 a2 = TileCorrections(
                rowFactors, weightedScales + 2 * stride, offsets + 2 * stride,
                scaled + 2 * stride, stride);
            v256 a3 = TileCorrections(
                rowFactors, weightedScales + 3 * stride, offsets + 3 * stride,
                scaled + 3 * stride, stride);
            for (int g = 0; g < groups; g++)
            {
                v256 v0 = X86.Avx.mm256_loadu_si256(packed + g * 32);
                v256 lower = X86.Avx2.mm256_and_si256(v0, mask);
                v256 upper = X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(v0, 4), mask);
                v256 p0 = X86.Avx2.mm256_shuffle_epi8(first, lower);
                v256 p1 = X86.Avx2.mm256_shuffle_epi8(second, lower);
                v256 p2 = X86.Avx2.mm256_shuffle_epi8(first, upper);
                v256 p3 = X86.Avx2.mm256_shuffle_epi8(second, upper);
                sbyte* x = values + g * GroupSize;
                a0 = AccumulateX86(p0, p1, p2, p3, x, scaled + g, a0);
                a1 = AccumulateX86(p0, p1, p2, p3, x + cols, scaled + stride + g, a1);
                a2 = AccumulateX86(p0, p1, p2, p3, x + 2 * cols, scaled + 2 * stride + g, a2);
                a3 = AccumulateX86(p0, p1, p2, p3, x + 3 * cols, scaled + 3 * stride + g, a3);
            }
            StoreQuad4(a0, a1, a2, a3, y, ldy, 0);
        }

        // Gemv.Row4<FourBitPlanes, Avx2Dot>.
        static void Row4_4(
            byte* packed, ushort* norms, bool convertible, int groups, int stride,
            int cols, v256 codebook, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            if (convertible)
                for (int v = 0; v < stride; v += HalvesPerVector)
                    X86.Avx.mm256_storeu_ps(
                        rowFactors + v,
                        UnbiasedHalf(X86.Sse2.loadu_si128(norms + v)));
            else
                EdgeRowFactors(norms, groups, rowFactors, stride);
            v256 a0 = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
            v256 a1 = TileCorrections(
                rowFactors, weightedScales + stride, offsets + stride, scaled + stride, stride);
            v256 a2 = TileCorrections(
                rowFactors, weightedScales + 2 * stride, offsets + 2 * stride,
                scaled + 2 * stride, stride);
            v256 a3 = TileCorrections(
                rowFactors, weightedScales + 3 * stride, offsets + 3 * stride,
                scaled + 3 * stride, stride);
            for (int g = 0; g < groups; g++)
            {
                v256 lo = X86.Avx.mm256_loadu_si256(packed + g * 64);
                v256 hi = X86.Avx.mm256_loadu_si256(packed + g * 64 + 32);
                v256 p0 = X86.Avx2.mm256_shuffle_epi8(
                    codebook, X86.Avx2.mm256_and_si256(lo, mask));
                v256 p1 = X86.Avx2.mm256_shuffle_epi8(
                    codebook,
                    X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(lo, 4), mask));
                v256 p2 = X86.Avx2.mm256_shuffle_epi8(
                    codebook, X86.Avx2.mm256_and_si256(hi, mask));
                v256 p3 = X86.Avx2.mm256_shuffle_epi8(
                    codebook,
                    X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(hi, 4), mask));
                sbyte* x = values + g * GroupSize;
                a0 = AccumulateX86(p0, p1, p2, p3, x, scaled + g, a0);
                a1 = AccumulateX86(p0, p1, p2, p3, x + cols, scaled + stride + g, a1);
                a2 = AccumulateX86(p0, p1, p2, p3, x + 2 * cols, scaled + 2 * stride + g, a2);
                a3 = AccumulateX86(p0, p1, p2, p3, x + 3 * cols, scaled + 3 * stride + g, a3);
            }
            StoreQuad4(a0, a1, a2, a3, y, ldy, 0);
        }

        // Gemv.Pair2<TwoBitPlanes, Avx2Dot>: two rows x two tokens, with the
        // BCL Vector256.Sum reduction order per accumulator.
        static void Pair2_2(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            int cols, v256 first, v256 second, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            for (int v = 0; v < stride; v += HalvesPerVector)
            {
                X86.Avx.mm256_storeu_ps(
                    rowFactors + v,
                    UnbiasedHalf(X86.Sse2.loadu_si128(norms + v)));
                X86.Avx.mm256_storeu_ps(
                    rowFactors + stride + v,
                    UnbiasedHalf(X86.Sse2.loadu_si128(norms + groups + v)));
            }
            v256 a00 = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
            v256 a01 = TileCorrections(
                rowFactors, weightedScales + stride, offsets + stride, scaled + stride, stride);
            v256 a10 = TileCorrections(
                rowFactors + stride, weightedScales, offsets, scaled + 2 * stride, stride);
            v256 a11 = TileCorrections(
                rowFactors + stride, weightedScales + stride, offsets + stride,
                scaled + 3 * stride, stride);
            byte* row1 = packed + rowBytes;
            for (int g = 0; g < groups; g++)
            {
                v256 v0 = X86.Avx.mm256_loadu_si256(packed + g * 32);
                v256 lower = X86.Avx2.mm256_and_si256(v0, mask);
                v256 upper = X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(v0, 4), mask);
                v256 p0 = X86.Avx2.mm256_shuffle_epi8(first, lower);
                v256 p1 = X86.Avx2.mm256_shuffle_epi8(second, lower);
                v256 p2 = X86.Avx2.mm256_shuffle_epi8(first, upper);
                v256 p3 = X86.Avx2.mm256_shuffle_epi8(second, upper);
                v256 w0 = X86.Avx.mm256_loadu_si256(row1 + g * 32);
                v256 lower1 = X86.Avx2.mm256_and_si256(w0, mask);
                v256 upper1 = X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(w0, 4), mask);
                v256 q0 = X86.Avx2.mm256_shuffle_epi8(first, lower1);
                v256 q1 = X86.Avx2.mm256_shuffle_epi8(second, lower1);
                v256 q2 = X86.Avx2.mm256_shuffle_epi8(first, upper1);
                v256 q3 = X86.Avx2.mm256_shuffle_epi8(second, upper1);
                sbyte* x = values + g * GroupSize;
                sbyte* x1 = x + cols;
                a00 = AccumulateX86(p0, p1, p2, p3, x, scaled + g, a00);
                a10 = AccumulateX86(q0, q1, q2, q3, x, scaled + 2 * stride + g, a10);
                a01 = AccumulateX86(p0, p1, p2, p3, x1, scaled + stride + g, a01);
                a11 = AccumulateX86(q0, q1, q2, q3, x1, scaled + 3 * stride + g, a11);
            }
            y[0] = SumX86(a00);
            y[ldy] = SumX86(a01);
            y[1] = SumX86(a10);
            y[ldy + 1] = SumX86(a11);
        }

        // Gemv.Pair2<FourBitPlanes, Avx2Dot>.
        static void Pair2_4(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            int cols, v256 codebook, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            for (int v = 0; v < stride; v += HalvesPerVector)
            {
                X86.Avx.mm256_storeu_ps(
                    rowFactors + v,
                    UnbiasedHalf(X86.Sse2.loadu_si128(norms + v)));
                X86.Avx.mm256_storeu_ps(
                    rowFactors + stride + v,
                    UnbiasedHalf(X86.Sse2.loadu_si128(norms + groups + v)));
            }
            v256 a00 = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
            v256 a01 = TileCorrections(
                rowFactors, weightedScales + stride, offsets + stride, scaled + stride, stride);
            v256 a10 = TileCorrections(
                rowFactors + stride, weightedScales, offsets, scaled + 2 * stride, stride);
            v256 a11 = TileCorrections(
                rowFactors + stride, weightedScales + stride, offsets + stride,
                scaled + 3 * stride, stride);
            byte* row1 = packed + rowBytes;
            for (int g = 0; g < groups; g++)
            {
                v256 lo = X86.Avx.mm256_loadu_si256(packed + g * 64);
                v256 hi = X86.Avx.mm256_loadu_si256(packed + g * 64 + 32);
                v256 p0 = X86.Avx2.mm256_shuffle_epi8(
                    codebook, X86.Avx2.mm256_and_si256(lo, mask));
                v256 p1 = X86.Avx2.mm256_shuffle_epi8(
                    codebook,
                    X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(lo, 4), mask));
                v256 p2 = X86.Avx2.mm256_shuffle_epi8(
                    codebook, X86.Avx2.mm256_and_si256(hi, mask));
                v256 p3 = X86.Avx2.mm256_shuffle_epi8(
                    codebook,
                    X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(hi, 4), mask));
                v256 lo1 = X86.Avx.mm256_loadu_si256(row1 + g * 64);
                v256 hi1 = X86.Avx.mm256_loadu_si256(row1 + g * 64 + 32);
                v256 q0 = X86.Avx2.mm256_shuffle_epi8(
                    codebook, X86.Avx2.mm256_and_si256(lo1, mask));
                v256 q1 = X86.Avx2.mm256_shuffle_epi8(
                    codebook,
                    X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(lo1, 4), mask));
                v256 q2 = X86.Avx2.mm256_shuffle_epi8(
                    codebook, X86.Avx2.mm256_and_si256(hi1, mask));
                v256 q3 = X86.Avx2.mm256_shuffle_epi8(
                    codebook,
                    X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(hi1, 4), mask));
                sbyte* x = values + g * GroupSize;
                sbyte* x1 = x + cols;
                a00 = AccumulateX86(p0, p1, p2, p3, x, scaled + g, a00);
                a10 = AccumulateX86(q0, q1, q2, q3, x, scaled + 2 * stride + g, a10);
                a01 = AccumulateX86(p0, p1, p2, p3, x1, scaled + stride + g, a01);
                a11 = AccumulateX86(q0, q1, q2, q3, x1, scaled + 3 * stride + g, a11);
            }
            y[0] = SumX86(a00);
            y[ldy] = SumX86(a01);
            y[1] = SumX86(a10);
            y[ldy + 1] = SumX86(a11);
        }

        // Gemv.Row2<TwoBitPlanes, Avx2Dot>: one row x two tokens.
        static void Row2_2(
            byte* packed, ushort* norms, bool convertible, int groups, int stride,
            int cols, v256 first, v256 second, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            if (convertible)
                for (int v = 0; v < stride; v += HalvesPerVector)
                    X86.Avx.mm256_storeu_ps(
                        rowFactors + v,
                        UnbiasedHalf(X86.Sse2.loadu_si128(norms + v)));
            else
                EdgeRowFactors(norms, groups, rowFactors, stride);
            v256 a0 = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
            v256 a1 = TileCorrections(
                rowFactors, weightedScales + stride, offsets + stride, scaled + stride, stride);
            for (int g = 0; g < groups; g++)
            {
                v256 v0 = X86.Avx.mm256_loadu_si256(packed + g * 32);
                v256 lower = X86.Avx2.mm256_and_si256(v0, mask);
                v256 upper = X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(v0, 4), mask);
                v256 p0 = X86.Avx2.mm256_shuffle_epi8(first, lower);
                v256 p1 = X86.Avx2.mm256_shuffle_epi8(second, lower);
                v256 p2 = X86.Avx2.mm256_shuffle_epi8(first, upper);
                v256 p3 = X86.Avx2.mm256_shuffle_epi8(second, upper);
                sbyte* x = values + g * GroupSize;
                a0 = AccumulateX86(p0, p1, p2, p3, x, scaled + g, a0);
                a1 = AccumulateX86(p0, p1, p2, p3, x + cols, scaled + stride + g, a1);
            }
            y[0] = SumX86(a0);
            y[ldy] = SumX86(a1);
        }

        // Gemv.Row2<FourBitPlanes, Avx2Dot>.
        static void Row2_4(
            byte* packed, ushort* norms, bool convertible, int groups, int stride,
            int cols, v256 codebook, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            if (convertible)
                for (int v = 0; v < stride; v += HalvesPerVector)
                    X86.Avx.mm256_storeu_ps(
                        rowFactors + v,
                        UnbiasedHalf(X86.Sse2.loadu_si128(norms + v)));
            else
                EdgeRowFactors(norms, groups, rowFactors, stride);
            v256 a0 = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
            v256 a1 = TileCorrections(
                rowFactors, weightedScales + stride, offsets + stride, scaled + stride, stride);
            for (int g = 0; g < groups; g++)
            {
                v256 lo = X86.Avx.mm256_loadu_si256(packed + g * 64);
                v256 hi = X86.Avx.mm256_loadu_si256(packed + g * 64 + 32);
                v256 p0 = X86.Avx2.mm256_shuffle_epi8(
                    codebook, X86.Avx2.mm256_and_si256(lo, mask));
                v256 p1 = X86.Avx2.mm256_shuffle_epi8(
                    codebook,
                    X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(lo, 4), mask));
                v256 p2 = X86.Avx2.mm256_shuffle_epi8(
                    codebook, X86.Avx2.mm256_and_si256(hi, mask));
                v256 p3 = X86.Avx2.mm256_shuffle_epi8(
                    codebook,
                    X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(hi, 4), mask));
                sbyte* x = values + g * GroupSize;
                a0 = AccumulateX86(p0, p1, p2, p3, x, scaled + g, a0);
                a1 = AccumulateX86(p0, p1, p2, p3, x + cols, scaled + stride + g, a1);
            }
            y[0] = SumX86(a0);
            y[ldy] = SumX86(a1);
        }

        // Gemv.ApplyPlanes<Avx2Dot>: four int8 activation loads, the Avx2Dot
        // integer dot, int32->float convert, broadcast-scale FMA accumulate.
        // The leading Add(Zero, madd(...)) and the p0..p3 accumulation order
        // mirror Tiers.Avx2Dot.Dot exactly (vpmaddubsw, then vpmaddwd
        // against int16 ones, then vpaddd).
        static v256 AccumulateX86(
            v256 p0, v256 p1, v256 p2, v256 p3, sbyte* x, float* scale, v256 acc)
        {
            v256 x0 = X86.Avx.mm256_loadu_si256(x);
            v256 x1 = X86.Avx.mm256_loadu_si256(x + 32);
            v256 x2 = X86.Avx.mm256_loadu_si256(x + 64);
            v256 x3 = X86.Avx.mm256_loadu_si256(x + 96);
            v256 dot = DotAvx2(p0, p1, p2, p3, x0, x1, x2, x3);
            return X86.Fma.mm256_fmadd_ps(
                X86.Avx.mm256_cvtepi32_ps(dot),
                X86.Avx.mm256_broadcast_ss(scale),
                acc);
        }

        static v256 DotAvx2(
            v256 p0, v256 p1, v256 p2, v256 p3,
            v256 x0, v256 x1, v256 x2, v256 x3)
        {
            v256 ones = new v256(ShortOnesWord);
            v256 sum = X86.Avx2.mm256_add_epi32(
                new v256(0),
                X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(p0, x0), ones));
            sum = X86.Avx2.mm256_add_epi32(
                sum,
                X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(p1, x1), ones));
            sum = X86.Avx2.mm256_add_epi32(
                sum,
                X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(p2, x2), ones));
            return X86.Avx2.mm256_add_epi32(
                sum,
                X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(p3, x3), ones));
        }

        // Gemv.TileCorrections: factors = rowFactor * weightedScale (stored
        // back to `scaled`), correction = -(factors * offset) + correction.
        static v256 TileCorrections(
            float* rowFactors, float* weightedScales, float* offsets,
            float* scaled, int stride)
        {
            v256 correction = X86.Avx.mm256_set1_ps(0f);
            for (int v = 0; v < stride; v += HalvesPerVector)
            {
                v256 factors = X86.Avx.mm256_mul_ps(
                    X86.Avx.mm256_loadu_ps(rowFactors + v),
                    X86.Avx.mm256_loadu_ps(weightedScales + v));
                X86.Avx.mm256_storeu_ps(scaled + v, factors);
                correction = X86.Fma.mm256_fnmadd_ps(
                    factors, X86.Avx.mm256_loadu_ps(offsets + v), correction);
            }
            return correction;
        }

        // Gemv.UnbiasedHalfToSingle: (half << 13) reinterpreted as float
        // (VPMOVZXWD zero-extends the 8 half norms, then VPSLLD, no-op cast).
        static v256 UnbiasedHalf(v128 packed)
        {
            return X86.Avx.mm256_castsi256_ps(
                X86.Avx2.mm256_slli_epi32(
                    X86.Avx2.mm256_cvtepu16_epi32(packed), 13));
        }

        // Gemv.CorrectionsAtEdge row-factor path (scalar, as net10):
        // BitConverter.Int32BitsToSingle((short)norms[i] << 13), zero past
        // `groups`. The pointer cast is the same bit pattern, no rounding.
        static void EdgeRowFactors(ushort* norms, int groups, float* rowFactors, int stride)
        {
            for (int i = 0; i < stride; i++)
            {
                if (i < groups)
                {
                    int bits = ((short)norms[i]) << 13;
                    rowFactors[i] = *(float*)&bits;
                }
                else rowFactors[i] = 0f;
            }
        }

        // BCL Vector256.Sum order used by Pair2/Row2 (checked against the
        // .NET 10 reference source and probed on hardware): Sum128(lower) +
        // Sum128(upper), where Sum128 is two hadd passes with lane 0 out.
        // An add-then-hadd sequence differs by 1 ulp; do not "simplify".
        static float SumX86(v256 v)
        {
            v128 s0 = X86.Avx.mm256_castps256_ps128(v);
            s0 = X86.Sse3.hadd_ps(s0, s0);
            s0 = X86.Sse3.hadd_ps(s0, s0);
            v128 s1 = X86.Avx.mm256_extractf128_ps(v, 1);
            s1 = X86.Sse3.hadd_ps(s1, s1);
            s1 = X86.Sse3.hadd_ps(s1, s1);
            return X86.Sse.cvtss_f32(X86.Sse.add_ps(s0, s1));
        }

        // Pair4/Row4 tail: quads = hadd(hadd(a0, a1), hadd(a2, a3)), then
        // the low/high add. Each output lane is a pure lane read (shuffle
        // broadcast + cvtss), bitwise identical to Sse41.Extract(sums, k).
        static void StoreQuad4(
            v256 a0, v256 a1, v256 a2, v256 a3, float* y, int ldy, int row)
        {
            v256 quads = X86.Avx.mm256_hadd_ps(
                X86.Avx.mm256_hadd_ps(a0, a1),
                X86.Avx.mm256_hadd_ps(a2, a3));
            v128 sums = X86.Sse.add_ps(
                X86.Avx.mm256_castps256_ps128(quads),
                X86.Avx.mm256_extractf128_ps(quads, 1));
            y[row] = X86.Sse.cvtss_f32(sums);
            y[ldy + row] = X86.Sse.cvtss_f32(X86.Sse.shuffle_ps(sums, sums, 0x55));
            y[2 * ldy + row] = X86.Sse.cvtss_f32(X86.Sse.shuffle_ps(sums, sums, 0xAA));
            y[3 * ldy + row] = X86.Sse.cvtss_f32(X86.Sse.shuffle_ps(sums, sums, 0xFF));
        }

        // Exact scalar reference (Scalar.MatMulRows op for op): fallback for
        // targets without AVX2+FMA, and for the T=1 tail of odd token counts
        // until N32's x86 Single port takes those tails.
        static void ScalarTile(
            GemvWeight* w, GemvActivations* a, int tokenBase, int count,
            float* y, int ldy, int rowStart, int rowEnd)
        {
            for (int t = 0; t < count; t++)
                ScalarSingle(w, a, tokenBase + t, y + (long)t * ldy, rowStart, rowEnd);
        }

        static void ScalarSingle(
            GemvWeight* w, GemvActivations* a, int token, float* y,
            int rowStart, int rowEnd)
        {
            int groups = w->Cols / GroupSize;
            byte* table = w->Book->Bytes;
            float byteScale = w->Book->ByteScale;
            int rowBytes = (w->Cols * w->Bits) / 8;
            sbyte* values = a->Values + (long)token * a->Cols;
            float* scales = a->Scales + token * a->Stride;
            float* offsets = a->Offsets + token * a->Stride;
            bool twoBit = w->Bits == 2;
            for (int n = rowStart; n < rowEnd; n++)
            {
                byte* packed = w->Indices + (long)n * rowBytes;
                ushort* norms = w->Norms + (long)n * groups;
                float accumulator = 0f;
                float bias = 0f;
                for (int g = 0; g < groups; g++)
                {
                    sbyte* x = values + g * GroupSize;
                    int dot = 0;
                    if (twoBit)
                    {
                        byte* p = packed + g * 32;
                        for (int j = 0; j < 32; j++)
                        {
                            int b = p[j];
                            dot += table[b & 3] * x[j]
                                + table[(b >> 2) & 3] * x[32 + j]
                                + table[(b >> 4) & 3] * x[64 + j]
                                + table[(b >> 6) & 3] * x[96 + j];
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
                y[n] = accumulator - bias;
            }
        }

        // Scalar.HalfToSingle bit for bit (used by the scalar fallback only).
        static float HalfToSingle(ushort half)
        {
            int magnitude = half & 0x7FFF;
            if (magnitude == 0) return 0f;
            if (magnitude < 0x0400)
                return (half & 0x8000) != 0
                    ? -magnitude * 5.9604645e-8f : magnitude * 5.9604645e-8f;
            uint bits = ((uint)magnitude << 13) + 0x38000000u | (uint)(half & 0x8000) << 16;
            return *(float*)&bits;
        }
    }
}
