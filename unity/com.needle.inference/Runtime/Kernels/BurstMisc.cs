using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Needle;

namespace Needle.BurstKernels
{
    [BurstCompile]
    public static unsafe class BurstMisc
    {
        [StructLayout(LayoutKind.Explicit, Size = 40)]
        public struct BurstCq
        {
            [FieldOffset(0)] public byte* Indices;
            [FieldOffset(8)] public ushort* Norms;
            [FieldOffset(16)] public byte* Book;
            [FieldOffset(24)] public int Rows;
            [FieldOffset(28)] public int Cols;
            [FieldOffset(32)] public int Bits;
        }

        const int GroupSize = 128;
        const float HadamardScale = 0.08838834764831845f;

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void GatherRow(BurstCq* w, int row, float* destination)
        {
            DecodeRowCore(w, row, destination);
            int groups = w->Cols / GroupSize;
            for (int g = 0; g < groups; g++)
                HadamardInPlace(destination + (long)g * GroupSize);
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static int ArgMaxMasked(float* x, uint* mask, int words, int length)
        {
            int best = -1;
            float peak = float.NegativeInfinity;
            for (int w = 0; w < words; w++)
            {
                uint bits = mask[w];
                if (bits == 0)
                    continue;
                int bottom = w * 32;
                if (bits == 0xFFFFFFFFu)
                {
                    int top = bottom + 32 <= length ? bottom + 32 : length;
                    for (int id = bottom; id < top; id++)
                    {
                        float value = x[id];
                        if (value > peak)
                        {
                            peak = value;
                            best = id;
                        }
                    }
                    continue;
                }
                while (bits != 0)
                {
                    int id = bottom + TrailingZeros(bits);
                    bits &= bits - 1;
                    if (id >= length)
                        continue;
                    float value = x[id];
                    if (value > peak)
                    {
                        peak = value;
                        best = id;
                    }
                }
            }
            return best;
        }

        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static float BestProbability(float* values, int count, float best)
        {
            float total = 0f;
            for (int i = 0; i < count; i++)
                total += MathF.Exp(values[i] - best);
            return 1f / total;
        }

        static void DecodeRowCore(BurstCq* w, int row, float* destination)
        {
            int groups = w->Cols / GroupSize;
            int rowBytes = w->Cols * w->Bits / 8;
            byte* indices = w->Indices + (long)row * rowBytes;
            ushort* norms = w->Norms + (long)row * groups;
            float* levels = (float*)w->Book;
            for (int g = 0; g < groups; g++)
            {
                float norm = HalfToSingle(norms[g]);
                float* d = destination + (long)g * GroupSize;
                if (w->Bits == 2)
                {
                    byte* packed = indices + (long)g * 32;
                    for (int i = 0; i < GroupSize; i++)
                        d[i] = levels[(packed[i >> 2] >> ((i & 3) * 2)) & 3] * norm;
                }
                else
                {
                    byte* packed = indices + (long)g * 64;
                    for (int i = 0; i < GroupSize; i++)
                        d[i] = levels[(packed[i >> 1] >> ((i & 1) * 4)) & 15] * norm;
                }
            }
        }

        static void HadamardInPlace(float* p)
        {
            for (int stride = 1; stride < GroupSize; stride <<= 1)
                for (int block = 0; block < GroupSize; block += stride * 2)
                    for (int i = block; i < block + stride; i++)
                    {
                        float a = p[i];
                        float b = p[i + stride];
                        p[i] = a + b;
                        p[i + stride] = a - b;
                    }
            for (int i = 0; i < GroupSize; i++)
                p[i] *= HadamardScale;
        }

        static float HalfToSingle(ushort half)
        {
            int magnitude = half & 0x7FFF;
            if (magnitude == 0)
                return 0f;
            if (magnitude < 0x0400)
                return (half & 0x8000) != 0 ? -magnitude * 5.9604645e-8f : magnitude * 5.9604645e-8f;
            uint bits = ((uint)magnitude << 13) + 0x38000000u | (uint)(half & 0x8000) << 16;
            return *(float*)&bits;
        }

        static int TrailingZeros(uint value)
        {
            int n = 0;
            while ((value & 1u) == 0)
            {
                value >>= 1;
                n++;
            }
            return n;
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void GatherRowFn(BurstCq* w, int row, float* destination);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int ArgMaxMaskedFn(float* x, uint* mask, int words, int length);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate float BestProbabilityFn(float* values, int count, float best);

        public static bool IsInstalled =>
            Kernels.Active.GatherRow != 0 &&
            Kernels.Active.ArgMaxMasked != 0 &&
            Kernels.Active.BestProbability != 0;

        public static void Install()
        {
            KernelTable table = Kernels.Active;
            bool changed = false;
            if (table.GatherRow == 0)
            {
                FunctionPointer<GatherRowFn> compiled =
                    BurstCompiler.CompileFunctionPointer<GatherRowFn>(GatherRow);
                if (!compiled.IsCreated)
                    throw new InvalidOperationException(
                        "N39: Burst refused to compile BurstMisc.GatherRow. " +
                        "The kernel must live in package source (H6); " +
                        "a precompiled DLL entry cannot be compiled.");
                table.GatherRow = (nint)compiled.Value;
                changed = true;
            }
            if (table.ArgMaxMasked == 0)
            {
                FunctionPointer<ArgMaxMaskedFn> compiled =
                    BurstCompiler.CompileFunctionPointer<ArgMaxMaskedFn>(ArgMaxMasked);
                if (!compiled.IsCreated)
                    throw new InvalidOperationException(
                        "N39: Burst refused to compile BurstMisc.ArgMaxMasked. " +
                        "The kernel must live in package source (H6); " +
                        "a precompiled DLL entry cannot be compiled.");
                table.ArgMaxMasked = (nint)compiled.Value;
                changed = true;
            }
            if (table.BestProbability == 0)
            {
                FunctionPointer<BestProbabilityFn> compiled =
                    BurstCompiler.CompileFunctionPointer<BestProbabilityFn>(BestProbability);
                if (!compiled.IsCreated)
                    throw new InvalidOperationException(
                        "N39: Burst refused to compile BurstMisc.BestProbability. " +
                        "The kernel must live in package source (H6); " +
                        "a precompiled DLL entry cannot be compiled.");
                table.BestProbability = (nint)compiled.Value;
                changed = true;
            }
            if (changed)
                Kernels.Install(table, Kernels.IsInstalled ? Kernels.Provider + "+burst-misc" : "burst-misc");
        }
    }
}
