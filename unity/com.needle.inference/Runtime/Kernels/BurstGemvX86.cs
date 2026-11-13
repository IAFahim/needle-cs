using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;

namespace Needle.BurstKernels
{
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct CodebookMirror
    {
        public fixed float Levels[16];
        public fixed byte Bytes[32];
        public float ByteScale;
    }

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

    [BurstCompile]
    public static unsafe class BurstGemvX86
    {
        const int GroupSize = 128;
        const int RowBlock = 4;
        const int HalvesPerVector = 8;
        const float HalfExponentRebias = 5.192296858534828e33f;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void MatMulRowsFn(
            CqMirror* w, ActMirror* a, int tokens, float* y, int ldy,
            int rowStart, int rowEnd);

        public static bool IsInstalled => Kernels.Active.MatMulRows != 0;

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

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void InvokeMatMulRows(
            CqMirror* w, ActMirror* a, int tokens, float* y, int ldy,
            int rowStart, int rowEnd)
        {
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

        internal static void Rows2X86(CqMirror* w, ActMirror* a, int token, float* y, int rowStart, int rowEnd)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                int groups = w->Cols / GroupSize;
                int stride = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
                int cols = a->Cols;
                v256 codebook = X86.Avx.mm256_loadu_si256(w->Book->Bytes);
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
        }

        internal static void Rows4X86(CqMirror* w, ActMirror* a, int token, float* y, int rowStart, int rowEnd)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                int groups = w->Cols / GroupSize;
                int stride = (groups + HalvesPerVector - 1) & ~(HalvesPerVector - 1);
                int cols = a->Cols;
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
        }

        static v256 CorrectionsX86(
            ushort* norms, int stride, float* weightedScales, float* offsets, float* scaled)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                v256 correction = X86.Avx.mm256_set1_ps(0f);
                for (int v = 0; v < stride; v += HalvesPerVector)
                {
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
            return default;
        }

        static v256 CorrectionsAtEdgeX86(
            ushort* norms, int groups, int stride,
            float* weightedScales, float* offsets, float* scaled)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
            return default;
        }

        static v256 Dot8X86(
            v256 p0, v256 p1, v256 p2, v256 p3,
            v256 x0, v256 x1, v256 x2, v256 x3)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
            return default;
        }

        static void Expand2X86(
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

        static void Expand4X86(
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

        static v256 Group2X86(
            byte* packed, v256 first, v256 second, v256 mask,
            v256 x0, v256 x1, v256 x2, v256 x3, float* scale, v256 accumulator)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                Expand2X86(packed, first, second, mask, out v256 p0, out v256 p1, out v256 p2, out v256 p3);
                return X86.Fma.mm256_fmadd_ps(
                    X86.Avx.mm256_cvtepi32_ps(Dot8X86(p0, p1, p2, p3, x0, x1, x2, x3)),
                    X86.Avx.mm256_broadcast_ss(scale),
                    accumulator);
            }
            return default;
        }

        static v256 Group4X86(
            byte* packed, v256 codebook, v256 mask,
            v256 x0, v256 x1, v256 x2, v256 x3, float* scale, v256 accumulator)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
            {
                Expand4X86(packed, codebook, mask, out v256 p0, out v256 p1, out v256 p2, out v256 p3);
                return X86.Fma.mm256_fmadd_ps(
                    X86.Avx.mm256_cvtepi32_ps(Dot8X86(p0, p1, p2, p3, x0, x1, x2, x3)),
                    X86.Avx.mm256_broadcast_ss(scale),
                    accumulator);
            }
            return default;
        }

        static void Block2X86(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            v256 codebook, v256 first, v256 second, v256 mask,
            float* weightedScales, float* offsets, sbyte* values, float* scaled, float* y)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
        }

        static void Block4X86(
            byte* packed, int rowBytes, ushort* norms, int groups, int stride,
            v256 codebook, v256 mask,
            float* weightedScales, float* offsets, sbyte* values, float* scaled, float* y)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
        }

        static float Sum8X86(v256 v)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported && X86.Sse3.IsSse3Supported)
            {
                v128 s = X86.Sse.add_ps(
                    X86.Avx.mm256_castps256_ps128(v),
                    X86.Avx.mm256_extractf128_ps(v, 1));
                v128 t = X86.Sse3.hadd_ps(s, s);
                t = X86.Sse3.hadd_ps(t, t);
                return X86.Sse.cvtss_f32(t);
            }
            return 0f;
        }

        static float Single2X86(
            byte* packed, ushort* norms, int groups, int stride, bool convertible,
            v256 codebook, v256 first, v256 second, v256 mask,
            float* weightedScales, float* offsets, sbyte* values, float* scaled)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
            return 0f;
        }

        static float Single4X86(
            byte* packed, ushort* norms, int groups, int stride, bool convertible,
            v256 codebook, v256 mask,
            float* weightedScales, float* offsets, sbyte* values, float* scaled)
        {
            if (X86.Avx2.IsAvx2Supported && X86.Fma.IsFmaSupported)
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
            return 0f;
        }

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
