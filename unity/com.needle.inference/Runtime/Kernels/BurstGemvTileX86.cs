using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;

namespace Needle.BurstKernels
{
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct GemvCodebook
    {
        public fixed float Levels[16];
        public fixed byte Bytes[32];
        public float ByteScale;
    }

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

    [BurstCompile]
    public static unsafe class BurstGemvTileX86
    {
        const float HalfExponentRebias = 5.192296858534828e33f;
        const int GroupSize = 128;
        const int HalvesPerVector = 8;

        const int NibbleMaskWord = 0x0F0F0F0F;
        const int FirstPlaneWord = 0x03020100;
        const int ShortOnesWord = 0x00010001;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void TileFn(
            GemvWeight* w, GemvActivations* a, int token, float* y, int ldy,
            int rowStart, int rowEnd);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void TileMatMulRowsFn(
            GemvWeight* w, GemvActivations* a, int tokens, float* y, int ldy,
            int rowStart, int rowEnd);

        public static bool IsInstalled => Needle.Kernels.Active.MatMulRows != 0;

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
                        SingleToken2(w, a, t, y + (long)t * ldy, rowStart, rowEnd);
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
                        SingleToken4(w, a, t, y + (long)t * ldy, rowStart, rowEnd);
                }
                return;
            }
            ScalarTile(w, a, 0, tokens, y, ldy, rowStart, rowEnd);
        }

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

        static void Tile4_2(
            GemvWeight* w, GemvActivations* a, int token, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
        }

        static void Tile4_4(
            GemvWeight* w, GemvActivations* a, int token, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
        }

        static void Tile2_2(
            GemvWeight* w, GemvActivations* a, int token, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
        }

        static void Tile2_4(
            GemvWeight* w, GemvActivations* a, int token, float* y, int ldy,
            int rowStart, int rowEnd)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
        }

        static void Pair4_2(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            int cols, v256 first, v256 second, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
                    sbyte* x = values + g * GroupSize;
                    sbyte* x1 = x + cols;
                    sbyte* x2 = x + 2 * cols;
                    sbyte* x3 = x + 3 * cols;
                    a00 = AccumulateX86(p0, p1, p2, p3, x, scaled + g, a00);
                    a01 = AccumulateX86(p0, p1, p2, p3, x1, scaled + stride + g, a01);
                    a02 = AccumulateX86(p0, p1, p2, p3, x2, scaled + 2 * stride + g, a02);
                    a03 = AccumulateX86(p0, p1, p2, p3, x3, scaled + 3 * stride + g, a03);
                    v256 w0 = X86.Avx.mm256_loadu_si256(row1 + g * 32);
                    v256 lower1 = X86.Avx2.mm256_and_si256(w0, mask);
                    v256 upper1 = X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(w0, 4), mask);
                    p0 = X86.Avx2.mm256_shuffle_epi8(first, lower1);
                    p1 = X86.Avx2.mm256_shuffle_epi8(second, lower1);
                    p2 = X86.Avx2.mm256_shuffle_epi8(first, upper1);
                    p3 = X86.Avx2.mm256_shuffle_epi8(second, upper1);
                    a10 = AccumulateX86(p0, p1, p2, p3, x, scaled + 4 * stride + g, a10);
                    a11 = AccumulateX86(p0, p1, p2, p3, x1, scaled + 5 * stride + g, a11);
                    a12 = AccumulateX86(p0, p1, p2, p3, x2, scaled + 6 * stride + g, a12);
                    a13 = AccumulateX86(p0, p1, p2, p3, x3, scaled + 7 * stride + g, a13);
                }
                StoreQuad4(a00, a01, a02, a03, y, ldy, 0);
                StoreQuad4(a10, a11, a12, a13, y, ldy, 1);
            }
        }

        static void Pair4_4(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            int cols, v256 codebook, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
                    sbyte* x = values + g * GroupSize;
                    sbyte* x1 = x + cols;
                    sbyte* x2 = x + 2 * cols;
                    sbyte* x3 = x + 3 * cols;
                    a00 = AccumulateX86(p0, p1, p2, p3, x, scaled + g, a00);
                    a01 = AccumulateX86(p0, p1, p2, p3, x1, scaled + stride + g, a01);
                    a02 = AccumulateX86(p0, p1, p2, p3, x2, scaled + 2 * stride + g, a02);
                    a03 = AccumulateX86(p0, p1, p2, p3, x3, scaled + 3 * stride + g, a03);
                    v256 lo1 = X86.Avx.mm256_loadu_si256(row1 + g * 64);
                    v256 hi1 = X86.Avx.mm256_loadu_si256(row1 + g * 64 + 32);
                    p0 = X86.Avx2.mm256_shuffle_epi8(
                        codebook, X86.Avx2.mm256_and_si256(lo1, mask));
                    p1 = X86.Avx2.mm256_shuffle_epi8(
                        codebook,
                        X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(lo1, 4), mask));
                    p2 = X86.Avx2.mm256_shuffle_epi8(
                        codebook, X86.Avx2.mm256_and_si256(hi1, mask));
                    p3 = X86.Avx2.mm256_shuffle_epi8(
                        codebook,
                        X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(hi1, 4), mask));
                    a10 = AccumulateX86(p0, p1, p2, p3, x, scaled + 4 * stride + g, a10);
                    a11 = AccumulateX86(p0, p1, p2, p3, x1, scaled + 5 * stride + g, a11);
                    a12 = AccumulateX86(p0, p1, p2, p3, x2, scaled + 6 * stride + g, a12);
                    a13 = AccumulateX86(p0, p1, p2, p3, x3, scaled + 7 * stride + g, a13);
                }
                StoreQuad4(a00, a01, a02, a03, y, ldy, 0);
                StoreQuad4(a10, a11, a12, a13, y, ldy, 1);
            }
        }

        static void Row4_2(
            byte* packed, ushort* norms, bool convertible, int groups, int stride,
            int cols, v256 first, v256 second, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
        }

        static void Row4_4(
            byte* packed, ushort* norms, bool convertible, int groups, int stride,
            int cols, v256 codebook, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
        }

        static void Pair2_2(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            int cols, v256 first, v256 second, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
                    sbyte* x = values + g * GroupSize;
                    sbyte* x1 = x + cols;
                    a00 = AccumulateX86(p0, p1, p2, p3, x, scaled + g, a00);
                    a01 = AccumulateX86(p0, p1, p2, p3, x1, scaled + stride + g, a01);
                    v256 w0 = X86.Avx.mm256_loadu_si256(row1 + g * 32);
                    v256 lower1 = X86.Avx2.mm256_and_si256(w0, mask);
                    v256 upper1 = X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(w0, 4), mask);
                    p0 = X86.Avx2.mm256_shuffle_epi8(first, lower1);
                    p1 = X86.Avx2.mm256_shuffle_epi8(second, lower1);
                    p2 = X86.Avx2.mm256_shuffle_epi8(first, upper1);
                    p3 = X86.Avx2.mm256_shuffle_epi8(second, upper1);
                    a10 = AccumulateX86(p0, p1, p2, p3, x, scaled + 2 * stride + g, a10);
                    a11 = AccumulateX86(p0, p1, p2, p3, x1, scaled + 3 * stride + g, a11);
                }
                y[0] = SumX86(a00);
                y[ldy] = SumX86(a01);
                y[1] = SumX86(a10);
                y[ldy + 1] = SumX86(a11);
            }
        }

        static void Pair2_4(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            int cols, v256 codebook, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
                    sbyte* x = values + g * GroupSize;
                    sbyte* x1 = x + cols;
                    a00 = AccumulateX86(p0, p1, p2, p3, x, scaled + g, a00);
                    a01 = AccumulateX86(p0, p1, p2, p3, x1, scaled + stride + g, a01);
                    v256 lo1 = X86.Avx.mm256_loadu_si256(row1 + g * 64);
                    v256 hi1 = X86.Avx.mm256_loadu_si256(row1 + g * 64 + 32);
                    p0 = X86.Avx2.mm256_shuffle_epi8(
                        codebook, X86.Avx2.mm256_and_si256(lo1, mask));
                    p1 = X86.Avx2.mm256_shuffle_epi8(
                        codebook,
                        X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(lo1, 4), mask));
                    p2 = X86.Avx2.mm256_shuffle_epi8(
                        codebook, X86.Avx2.mm256_and_si256(hi1, mask));
                    p3 = X86.Avx2.mm256_shuffle_epi8(
                        codebook,
                        X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(hi1, 4), mask));
                    a10 = AccumulateX86(p0, p1, p2, p3, x, scaled + 2 * stride + g, a10);
                    a11 = AccumulateX86(p0, p1, p2, p3, x1, scaled + 3 * stride + g, a11);
                }
                y[0] = SumX86(a00);
                y[ldy] = SumX86(a01);
                y[1] = SumX86(a10);
                y[ldy + 1] = SumX86(a11);
            }
        }

        static void Row2_2(
            byte* packed, ushort* norms, bool convertible, int groups, int stride,
            int cols, v256 first, v256 second, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
        }

        static void Row2_4(
            byte* packed, ushort* norms, bool convertible, int groups, int stride,
            int cols, v256 codebook, v256 mask, float* weightedScales,
            float* offsets, sbyte* values, float* scaled, float* rowFactors,
            float* y, int ldy)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static v256 AccumulateX86(
            v256 p0, v256 p1, v256 p2, v256 p3, sbyte* x, float* scale, v256 acc)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 dot = DotAvx2Ptr(p0, p1, p2, p3, x);
                return X86.Fma.mm256_fmadd_ps(
                    X86.Avx.mm256_cvtepi32_ps(dot),
                    X86.Avx.mm256_broadcast_ss(scale),
                    acc);
            }
            return default;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static v256 DotAvx2Ptr(
            v256 p0, v256 p1, v256 p2, v256 p3, sbyte* x)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 ones = new v256(ShortOnesWord);
                v256 sum = X86.Avx2.mm256_madd_epi16(
                    X86.Avx2.mm256_maddubs_epi16(p0, X86.Avx.mm256_loadu_si256(x)), ones);
                sum = X86.Avx2.mm256_add_epi32(
                    sum,
                    X86.Avx2.mm256_madd_epi16(
                        X86.Avx2.mm256_maddubs_epi16(p1, X86.Avx.mm256_loadu_si256(x + 32)), ones));
                sum = X86.Avx2.mm256_add_epi32(
                    sum,
                    X86.Avx2.mm256_madd_epi16(
                        X86.Avx2.mm256_maddubs_epi16(p2, X86.Avx.mm256_loadu_si256(x + 64)), ones));
                return X86.Avx2.mm256_add_epi32(
                    sum,
                    X86.Avx2.mm256_madd_epi16(
                        X86.Avx2.mm256_maddubs_epi16(p3, X86.Avx.mm256_loadu_si256(x + 96)), ones));
            }
            return default;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static v256 DotAvx2(
            v256 p0, v256 p1, v256 p2, v256 p3,
            v256 x0, v256 x1, v256 x2, v256 x3)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 ones = new v256(ShortOnesWord);
                v256 sum = X86.Avx2.mm256_madd_epi16(X86.Avx2.mm256_maddubs_epi16(p0, x0), ones);
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
            return default;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void Expand2Local(
            byte* packed, v256 first, v256 second, v256 mask,
            out v256 p0, out v256 p1, out v256 p2, out v256 p3)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 v = X86.Avx.mm256_loadu_si256(packed);
                v256 lower = X86.Avx2.mm256_and_si256(v, mask);
                v256 upper = X86.Avx2.mm256_and_si256(X86.Avx2.mm256_srli_epi16(v, 4), mask);
                p0 = X86.Avx2.mm256_shuffle_epi8(first, lower);
                p1 = X86.Avx2.mm256_shuffle_epi8(second, lower);
                p2 = X86.Avx2.mm256_shuffle_epi8(first, upper);
                p3 = X86.Avx2.mm256_shuffle_epi8(second, upper);
            }
            else
            {
                p0 = p1 = p2 = p3 = default;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void Expand4Local(
            byte* packed, v256 codebook, v256 mask,
            out v256 p0, out v256 p1, out v256 p2, out v256 p3)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
            else
            {
                p0 = p1 = p2 = p3 = default;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static v256 Group2Local(
            byte* packed, v256 first, v256 second, v256 mask,
            v256 x0, v256 x1, v256 x2, v256 x3, float* scale, v256 accumulator)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                Expand2Local(packed, first, second, mask, out v256 p0, out v256 p1, out v256 p2, out v256 p3);
                return X86.Fma.mm256_fmadd_ps(
                    X86.Avx.mm256_cvtepi32_ps(DotAvx2(p0, p1, p2, p3, x0, x1, x2, x3)),
                    X86.Avx.mm256_broadcast_ss(scale),
                    accumulator);
            }
            return default;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static v256 Group4Local(
            byte* packed, v256 codebook, v256 mask,
            v256 x0, v256 x1, v256 x2, v256 x3, float* scale, v256 accumulator)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                Expand4Local(packed, codebook, mask, out v256 p0, out v256 p1, out v256 p2, out v256 p3);
                return X86.Fma.mm256_fmadd_ps(
                    X86.Avx.mm256_cvtepi32_ps(DotAvx2(p0, p1, p2, p3, x0, x1, x2, x3)),
                    X86.Avx.mm256_broadcast_ss(scale),
                    accumulator);
            }
            return default;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static v256 Corrections1Local(
            ushort* norms, int stride, float* weightedScales, float* offsets, float* scaled)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 correction = X86.Avx.mm256_set1_ps(0f);
                for (int v = 0; v < stride; v += HalvesPerVector)
                {
                    v256 factors = X86.Avx.mm256_mul_ps(
                        UnbiasedHalf(X86.Sse2.loadu_si128(norms + v)),
                        X86.Avx.mm256_loadu_ps(weightedScales + v));
                    X86.Avx.mm256_storeu_ps(scaled + v, factors);
                    correction = X86.Fma.mm256_fnmadd_ps(
                        factors, X86.Avx.mm256_loadu_ps(offsets + v), correction);
                }
                return correction;
            }
            return default;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void Block1_2(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            v256 first, v256 second, v256 mask,
            float* weightedScales, float* offsets, sbyte* values, float* scaled, float* y)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 accumulator0 = Corrections1Local(norms, stride, weightedScales, offsets, scaled);
                v256 accumulator1 = Corrections1Local(norms + groups, stride, weightedScales, offsets, scaled + stride);
                v256 accumulator2 = Corrections1Local(norms + 2 * groups, stride, weightedScales, offsets, scaled + 2 * stride);
                v256 accumulator3 = Corrections1Local(norms + 3 * groups, stride, weightedScales, offsets, scaled + 3 * stride);
                byte* row1 = packed + rowBytes;
                byte* row2 = packed + 2 * rowBytes;
                byte* row3 = packed + 3 * rowBytes;
                for (int g = 0; g < groups; g++)
                {
                    sbyte* x = values + g * GroupSize;
                    v256 x0 = X86.Avx.mm256_loadu_si256(x);
                    v256 x1 = X86.Avx.mm256_loadu_si256(x + 32);
                    v256 x2 = X86.Avx.mm256_loadu_si256(x + 64);
                    v256 x3 = X86.Avx.mm256_loadu_si256(x + 96);
                    int offset = g * 32;
                    accumulator0 = Group2Local(packed + offset, first, second, mask, x0, x1, x2, x3, scaled + g, accumulator0);
                    accumulator1 = Group2Local(row1 + offset, first, second, mask, x0, x1, x2, x3, scaled + stride + g, accumulator1);
                    accumulator2 = Group2Local(row2 + offset, first, second, mask, x0, x1, x2, x3, scaled + 2 * stride + g, accumulator2);
                    accumulator3 = Group2Local(row3 + offset, first, second, mask, x0, x1, x2, x3, scaled + 3 * stride + g, accumulator3);
                }
                v256 quads = X86.Avx.mm256_hadd_ps(
                    X86.Avx.mm256_hadd_ps(accumulator0, accumulator1),
                    X86.Avx.mm256_hadd_ps(accumulator2, accumulator3));
                X86.Sse.storeu_ps(
                    y,
                    X86.Sse.add_ps(
                        X86.Avx.mm256_castps256_ps128(quads),
                        X86.Avx.mm256_extractf128_ps(quads, 1)));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void Block1_4(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            v256 codebook, v256 mask,
            float* weightedScales, float* offsets, sbyte* values, float* scaled, float* y)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 accumulator0 = Corrections1Local(norms, stride, weightedScales, offsets, scaled);
                v256 accumulator1 = Corrections1Local(norms + groups, stride, weightedScales, offsets, scaled + stride);
                v256 accumulator2 = Corrections1Local(norms + 2 * groups, stride, weightedScales, offsets, scaled + 2 * stride);
                v256 accumulator3 = Corrections1Local(norms + 3 * groups, stride, weightedScales, offsets, scaled + 3 * stride);
                byte* row1 = packed + rowBytes;
                byte* row2 = packed + 2 * rowBytes;
                byte* row3 = packed + 3 * rowBytes;
                for (int g = 0; g < groups; g++)
                {
                    sbyte* x = values + g * GroupSize;
                    v256 x0 = X86.Avx.mm256_loadu_si256(x);
                    v256 x1 = X86.Avx.mm256_loadu_si256(x + 32);
                    v256 x2 = X86.Avx.mm256_loadu_si256(x + 64);
                    v256 x3 = X86.Avx.mm256_loadu_si256(x + 96);
                    int offset = g * 64;
                    accumulator0 = Group4Local(packed + offset, codebook, mask, x0, x1, x2, x3, scaled + g, accumulator0);
                    accumulator1 = Group4Local(row1 + offset, codebook, mask, x0, x1, x2, x3, scaled + stride + g, accumulator1);
                    accumulator2 = Group4Local(row2 + offset, codebook, mask, x0, x1, x2, x3, scaled + 2 * stride + g, accumulator2);
                    accumulator3 = Group4Local(row3 + offset, codebook, mask, x0, x1, x2, x3, scaled + 3 * stride + g, accumulator3);
                }
                v256 quads = X86.Avx.mm256_hadd_ps(
                    X86.Avx.mm256_hadd_ps(accumulator0, accumulator1),
                    X86.Avx.mm256_hadd_ps(accumulator2, accumulator3));
                X86.Sse.storeu_ps(
                    y,
                    X86.Sse.add_ps(
                        X86.Avx.mm256_castps256_ps128(quads),
                        X86.Avx.mm256_extractf128_ps(quads, 1)));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float Single1_2(
            byte* packed, ushort* norms, int groups, int stride, bool convertible,
            v256 first, v256 second, v256 mask,
            float* weightedScales, float* offsets, sbyte* values, float* scaled, float* rowFactors)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 accumulator;
                if (convertible)
                    accumulator = Corrections1Local(norms, stride, weightedScales, offsets, scaled);
                else
                {
                    EdgeRowFactors(norms, groups, rowFactors, stride);
                    accumulator = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
                }
                for (int g = 0; g < groups; g++)
                {
                    sbyte* x = values + g * GroupSize;
                    accumulator = Group2Local(
                        packed + g * 32, first, second, mask,
                        X86.Avx.mm256_loadu_si256(x),
                        X86.Avx.mm256_loadu_si256(x + 32),
                        X86.Avx.mm256_loadu_si256(x + 64),
                        X86.Avx.mm256_loadu_si256(x + 96),
                        scaled + g, accumulator);
                }
                return SumX86(accumulator);
            }
            return 0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float Single1_4(
            byte* packed, ushort* norms, int groups, int stride, bool convertible,
            v256 codebook, v256 mask,
            float* weightedScales, float* offsets, sbyte* values, float* scaled, float* rowFactors)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 accumulator;
                if (convertible)
                    accumulator = Corrections1Local(norms, stride, weightedScales, offsets, scaled);
                else
                {
                    EdgeRowFactors(norms, groups, rowFactors, stride);
                    accumulator = TileCorrections(rowFactors, weightedScales, offsets, scaled, stride);
                }
                for (int g = 0; g < groups; g++)
                {
                    sbyte* x = values + g * GroupSize;
                    accumulator = Group4Local(
                        packed + g * 64, codebook, mask,
                        X86.Avx.mm256_loadu_si256(x),
                        X86.Avx.mm256_loadu_si256(x + 32),
                        X86.Avx.mm256_loadu_si256(x + 64),
                        X86.Avx.mm256_loadu_si256(x + 96),
                        scaled + g, accumulator);
                }
                return SumX86(accumulator);
            }
            return 0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void SingleToken2(GemvWeight* w, GemvActivations* a, int token, float* y, int rowStart, int rowEnd)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                int groups = w->Cols / GroupSize;
                int stride = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
                int cols = a->Cols;
                v256 codebook = X86.Avx.mm256_loadu_si256(w->Book->Bytes);
                v256 first = X86.Avx2.mm256_shuffle_epi8(codebook, new v256(FirstPlaneWord));
                v256 second = X86.Avx2.mm256_shuffle_epi8(
                    codebook,
                    new v256(0x00000000, 0x01010101, 0x02020202, 0x03030303, 0x00000000, 0x01010101, 0x02020202, 0x03030303));
                v256 mask = new v256(NibbleMaskWord);
                float* weightedScales = stackalloc float[stride];
                float* scaled = stackalloc float[4 * stride];
                float* rowFactors = stackalloc float[stride];
                float* activationScales = a->Scales + token * stride;
                v256 byteScale = X86.Avx.mm256_set1_ps(w->Book->ByteScale * HalfExponentRebias);
                for (int v = 0; v < stride; v += HalvesPerVector)
                    X86.Avx.mm256_storeu_ps(
                        weightedScales + v,
                        X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(activationScales + v), byteScale));
                float* offsets = a->Offsets + token * stride;
                sbyte* values = a->Values + (long)token * cols;
                int rowBytes = (w->Cols * w->Bits) / 8;
                int paddedGroups = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
                long total = (long)w->Rows * groups;
                int convertibleRows = total < paddedGroups ? 0 : (int)((total - paddedGroups) / groups) + 1;
                byte* packedBase = w->Indices;
                ushort* normsBase = w->Norms;
                int n = rowStart;
                for (; n + 4 <= rowEnd && n + 4 <= convertibleRows; n += 4)
                    Block1_2(
                        packedBase + (long)n * rowBytes, rowBytes,
                        normsBase + (long)n * groups, groups, stride,
                        first, second, mask,
                        weightedScales, offsets, values, scaled, y + n);
                for (; n < rowEnd; n++)
                    y[n] = Single1_2(
                        packedBase + (long)n * rowBytes, normsBase + (long)n * groups,
                        groups, stride, n < convertibleRows,
                        first, second, mask,
                        weightedScales, offsets, values, scaled, rowFactors);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void SingleToken4(GemvWeight* w, GemvActivations* a, int token, float* y, int rowStart, int rowEnd)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                int groups = w->Cols / GroupSize;
                int stride = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
                int cols = a->Cols;
                v256 codebook = X86.Avx.mm256_loadu_si256(w->Book->Bytes);
                v256 mask = new v256(NibbleMaskWord);
                float* weightedScales = stackalloc float[stride];
                float* scaled = stackalloc float[4 * stride];
                float* rowFactors = stackalloc float[stride];
                float* activationScales = a->Scales + token * stride;
                v256 byteScale = X86.Avx.mm256_set1_ps(w->Book->ByteScale * HalfExponentRebias);
                for (int v = 0; v < stride; v += HalvesPerVector)
                    X86.Avx.mm256_storeu_ps(
                        weightedScales + v,
                        X86.Avx.mm256_mul_ps(X86.Avx.mm256_loadu_ps(activationScales + v), byteScale));
                float* offsets = a->Offsets + token * stride;
                sbyte* values = a->Values + (long)token * cols;
                int rowBytes = (w->Cols * w->Bits) / 8;
                int paddedGroups = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
                long total = (long)w->Rows * groups;
                int convertibleRows = total < paddedGroups ? 0 : (int)((total - paddedGroups) / groups) + 1;
                byte* packedBase = w->Indices;
                ushort* normsBase = w->Norms;
                int n = rowStart;
                for (; n + 4 <= rowEnd && n + 4 <= convertibleRows; n += 4)
                    Block1_4(
                        packedBase + (long)n * rowBytes, rowBytes,
                        normsBase + (long)n * groups, groups, stride,
                        codebook, mask,
                        weightedScales, offsets, values, scaled, y + n);
                for (; n < rowEnd; n++)
                    y[n] = Single1_4(
                        packedBase + (long)n * rowBytes, normsBase + (long)n * groups,
                        groups, stride, n < convertibleRows,
                        codebook, mask,
                        weightedScales, offsets, values, scaled, rowFactors);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static v256 TileCorrections(
            float* rowFactors, float* weightedScales, float* offsets,
            float* scaled, int stride)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
            return default;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static v256 UnbiasedHalf(v128 packed)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                return X86.Avx.mm256_castsi256_ps(
                    X86.Avx2.mm256_slli_epi32(
                        X86.Avx2.mm256_cvtepu16_epi32(packed), 13));
            }
            return default;
        }

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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float SumX86(v256 v)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported && X86.Sse3.IsSse3Supported)
            {
                v128 s0 = X86.Avx.mm256_castps256_ps128(v);
                s0 = X86.Sse3.hadd_ps(s0, s0);
                s0 = X86.Sse3.hadd_ps(s0, s0);
                v128 s1 = X86.Avx.mm256_extractf128_ps(v, 1);
                s1 = X86.Sse3.hadd_ps(s1, s1);
                s1 = X86.Sse3.hadd_ps(s1, s1);
                return X86.Sse.cvtss_f32(X86.Sse.add_ps(s0, s1));
            }
            return 0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void StoreQuad4(
            v256 a0, v256 a1, v256 a2, v256 a3, float* y, int ldy, int row)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
        }

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
