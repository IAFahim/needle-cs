// N39: Burst DecodeRow/GatherRow, ArgMaxMasked, BestProbability.
//
// Ports the exact scalar reference loops 1:1 to Unity.Burst.Intrinsics-free
// Burst C# (plain scalar C#, Strict float mode), so every output bit matches
// the ns2.1 reference (Scalar.DecodeRow, Scalar.Hadamard, ArgMaxMaskedScalar,
// ScalarBestProbability) that the N13 conformance suite checks against:
//
//   GatherRow    = Scalar.DecodeRow then Scalar.Hadamard per 128-group.
//                  DecodeRow has no KernelTable slot (Quant.DecodeRow runs it
//                  inline), so only the fused GatherRow entry is installed;
//                  DecodeRowCore is the shared helper underneath it.
//   ArgMaxMasked = ArgMaxMaskedScalar verbatim: skip zero words, scan
//                  full-mask words linearly, walk sparse words low-bit first
//                  via TrailingZeros. First (lowest-id) max wins ties; -1 on
//                  empty masks; length bounds every id (odd tails, over-long
//                  final words). The net10 Avx lane picks the same winner by
//                  construction (strict `>` per 8-block, first lane of max),
//                  so this lane is bitwise with both.
//   BestProbability = ScalarBestProbability verbatim: sequential
//                  sum of Exp(v - best), then 1/total. Summation order is the
//                  whole result, so no vector fold is used here (the net10
//                  Vector256.Sum hadd order would round differently).
//
// Why scalar, not intrinsics: these entries are tiny and branchy per element
// (bit walks, table lookups, one Exp each). Like N31's scalar fallback lane,
// the work is a few hundred ns; a vector lane would buy nothing while risking
// the bitwise acceptance. Larger kernels (N32+) own the GB/s lanes.
//
// Burst rules (plan section 2.1 cookbook):
//   - FloatMode.Strict on every entry point, never Fast: no reassociation.
//   - No [NoAlias]: GatherRow reads codebook/indices/norms while writing
//     destination; aliasing promises would be false.
//   - Blittable arguments only; no allocation, no static mutable state; safe
//     from any thread (pool workers call via the table).
//   - No managed objects: TrailingZeros is a bit loop, not the DeBruijn
//     table (a static byte[] cannot cross into Burst).
//
// ABI note: Cq/Codebook stay internal to Needle.Inference, so this assembly
// cannot name them. BurstCq mirrors Cq field for field (explicit offsets;
// Book is carried as byte* to the codebook base: Levels[16] floats at +0,
// Bytes[32] at +64, ByteScale at +96 per Model.cs). A single pointer is a
// single pointer under Cdecl, so installing a BurstCq* entry into the
// Cq* GatherRow slot is address-identical; only field offsets matter, and
// they are pinned by the explicit layout below.
//
// H7 note: the x86/Neon lanes need no bitwise probe here because there are
// no intrinsic lanes: every lane is the reference loop. The one shared
// lowering risk is MathF.Exp in BestProbability (Burst lowers System.MathF.Exp;
// an FTZ difference on denormal exp(-87.3)-scale outputs cannot be ruled out
// from docs alone, same caveat as N31). If the bitwise probe in Tests/ ever
// diverges on a denormal tail, this kernel still meets the N13 parity bound
// (relative tolerance 1e-5, 5e-5 past 100) by construction.
//
// C# 9 safe: block namespace, no target-typed new, no static lambdas.
using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Needle;

namespace Needle.BurstKernels
{
    public static unsafe class BurstMisc
    {
        // Mirrors Needle.Cq (Model.cs) field for field. Offsets: Indices +0,
        // Norms +8, Book +16, Rows +24, Cols +28, Bits +32, size 40.
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

        // Entry point installed into KernelTable.GatherRow:
        // void(Cq*, int row, float* destination), Cdecl.
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static void GatherRow(BurstCq* w, int row, float* destination)
        {
            DecodeRowCore(w, row, destination);
            int groups = w->Cols / GroupSize;
            for (int g = 0; g < groups; g++)
                HadamardInPlace(destination + (long)g * GroupSize);
        }

        // Entry point installed into KernelTable.ArgMaxMasked:
        // int(float* x, uint* mask, int words, int length), Cdecl.
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

        // Entry point installed into KernelTable.BestProbability:
        // float(float* values, int count, float best), Cdecl.
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]
        public static float BestProbability(float* values, int count, float best)
        {
            float total = 0f;
            for (int i = 0; i < count; i++)
                total += MathF.Exp(values[i] - best);
            return 1f / total;
        }

        // Mirrors Scalar.DecodeRow bit for bit: per group, HalfToSingle norm,
        // then levels[packed field] * norm per element (2-bit: 4 per byte,
        // 4-bit: 2 per byte). Levels live at the codebook base (+0).
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

        // Mirrors Scalar.Hadamard in place (the source == p path GatherRow
        // takes): stride 1..64 pairwise add/sub, then the scale multiply.
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

        // Mirrors Scalar.HalfToSingle bit for bit, including the subnormal
        // branch and the positive-zero collapse (magnitude == 0 returns +0).
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

        // Bits.TrailingZeros without the static DeBruijn table (managed
        // arrays cannot enter Burst). Only called with nonzero bits.
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

        // Cdecl delegates for the table slots (match the KernelTable ABI
        // comments: void(Cq*, int, float*), int(float*, uint*, int, int),
        // float(float*, int, float)).
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

        // Idempotent: never overwrites an explicit or foreign provider's
        // entries (plan section 2 selection order); installing nothing when
        // all three slots are already filled.
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
