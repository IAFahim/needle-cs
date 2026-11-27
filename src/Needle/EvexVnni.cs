#if NEEDLE_SIMD
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace Needle;

// AVX512-VNNI (EVEX VPDPBUSD) GEMV row blocks for CPUs that have it without VEX AVX-VNNI (AMD Zen 4,
// Intel Ice/Tiger/Rocket Lake). .NET 10 exposes VPDPBUSD only through AvxVnni (VEX, NoEvexSemantics)
// and Avx10v1 (needs AVX512-FP16), so neither is reachable there from managed code. The kernels in
// Asm/EvexVnni.s are mapped executable once and called through an unmanaged function pointer.
// They reproduce Gemv.Block's float operation order exactly, so output is bit-identical to the
// managed tiers (the int8 dot is exact: weight bytes <= 126, so no tier saturates).
internal static unsafe partial class EvexVnni
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct BlockArgs
    {
        public byte* Packed;
        public long RowBytes;
        public ushort* Norms;
        public long Groups;
        public long Stride;
        public float* WeightedScales;
        public float* Offsets;
        public sbyte* Values;
        public float* Scaled;
        public byte* Tables;
        public float* Y;
        public long Count;
        public long PrefetchDistance;
        public long Cols;
        public long Ldy;
    }

    static readonly nint Base = Load();

    // CPUID.(EAX=7,ECX=0):ECX[11] is AVX512_VNNI; Avx512F.IsSupported already implies OS zmm state.
    internal static bool CpuHasAvx512Vnni =>
        Avx512F.IsSupported && Avx512F.VL.IsSupported && Avx512BW.IsSupported && Avx512Vbmi.VL.IsSupported
        && (X86Base.CpuId(7, 0).Ecx & (1 << 11)) != 0;

    internal static bool Ready => Base != 0;

    // Count four-row blocks of one token (Gemv.Block).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Blocks(int bits, BlockArgs* args) =>
        ((delegate* unmanaged[SuppressGCTransition]<BlockArgs*, void>)(Base + (bits == 2 ? Block2Offset : Block4Offset)))(args);

    // Count two-row pairs of four tokens (Gemv.Pair4).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Pairs(int bits, BlockArgs* args) =>
        ((delegate* unmanaged[SuppressGCTransition]<BlockArgs*, void>)(Base + (bits == 2 ? Pair2Offset : Pair4Offset)))(args);

    static nint Load()
    {
        if (Environment.GetEnvironmentVariable("NEEDLE_NOEVEX") != null || !CpuHasAvx512Vnni) return 0;
        try
        {
            var code = Code;
            var size = (nuint)((code.Length + 4095) & ~4095);
            var memory = Allocate(size);
            if (memory == 0) return 0;
            code.CopyTo(new Span<byte>((void*)memory, code.Length));
            if (OperatingSystem.IsWindows())
            {
                // The kernels take their one argument in rax; the first instruction moves it there
                // from the platform's first argument register (SysV rdi, Win64 rcx).
                foreach (var offset in new[] { Block2Offset, Block4Offset, Pair2Offset, Pair4Offset })
                {
                    var p = (byte*)memory + offset;
                    if (p[0] != 0x48 || p[1] != 0x89 || p[2] != 0xF8) return 0;
                    p[2] = 0xC8;
                }
            }
            if (!Protect(memory, size)) return 0;
            return SelfTest(memory) ? memory : 0;
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or MarshalDirectiveException)
        {
            return 0;
        }
    }

    // Runs both kernels on exact-integer inputs (unit factors, zero offsets) and checks every output
    // against a scalar reference of the documented plane layout, so a wrong encoding or a hostile
    // environment disables the tier instead of producing wrong logits.
    static bool SelfTest(nint memory) =>
        SelfTest(memory, 2, pairs: false) && SelfTest(memory, 4, pairs: false) &&
        SelfTest(memory, 2, pairs: true) && SelfTest(memory, 4, pairs: true);

    static bool SelfTest(nint memory, int bits, bool pairs)
    {
        const int rows = 8, groups = 3, cols = groups * Cq.GroupSize, stride = 8;
        var tokens = pairs ? 4 : 1;
        var rowBytes = cols * bits / 8;
        var packed = stackalloc byte[rows * rowBytes + 64];
        var norms = stackalloc ushort[rows * groups + stride];
        var weightedScales = stackalloc float[4 * stride];
        var offsets = stackalloc float[4 * stride];
        var values = stackalloc sbyte[4 * cols];
        var scaled = stackalloc float[8 * stride];
        var tables = stackalloc byte[64];
        var y = stackalloc float[4 * rows];
        var book = stackalloc byte[16];
        var seed = 0x2545F491u * (uint)(bits + (pairs ? 7 : 0));
        uint Next() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; }
        for (var i = 0; i < 16; i++) book[i] = (byte)(Next() % 127);
        for (var i = 0; i < 32; i++)
        {
            tables[i] = bits == 2 ? book[i & 3] : book[i & 15];
            tables[32 + i] = bits == 2 ? book[(i >> 2) & 3] : book[i & 15];
        }
        for (var i = 0; i < rows * rowBytes + 64; i++) packed[i] = (byte)Next();
        for (var i = 0; i < rows * groups + stride; i++) norms[i] = 0x3C00;
        for (var i = 0; i < 4 * stride; i++) { weightedScales[i] = 5.192296858534828e33f; offsets[i] = 0f; }
        for (var i = 0; i < 4 * cols; i++) values[i] = (sbyte)((int)(Next() % 255) - 127);
        var args = new BlockArgs
        {
            Packed = packed, RowBytes = rowBytes, Norms = norms, Groups = groups, Stride = stride,
            WeightedScales = weightedScales, Offsets = offsets, Values = values, Scaled = scaled,
            Tables = tables, Y = y, Count = pairs ? rows / 2 : rows / 4, PrefetchDistance = 0,
            Cols = cols, Ldy = rows,
        };
        var entry = memory + (pairs ? (bits == 2 ? Pair2Offset : Pair4Offset) : (bits == 2 ? Block2Offset : Block4Offset));
        ((delegate* unmanaged[SuppressGCTransition]<BlockArgs*, void>)entry)(&args);
        for (var t = 0; t < tokens; t++)
            for (var r = 0; r < rows; r++)
            {
                long expected = 0;
                for (var k = 0; k < cols; k++)
                {
                    int g = k / Cq.GroupSize, j = k % Cq.GroupSize, plane = j / 32, lane = j % 32;
                    int code;
                    if (bits == 2) code = (packed[r * rowBytes + g * 32 + lane] >> (2 * plane)) & 3;
                    else
                    {
                        var b = packed[r * rowBytes + g * 64 + (plane >> 1) * 32 + lane];
                        code = (plane & 1) == 0 ? b & 15 : b >> 4;
                    }
                    expected += book[code] * values[t * cols + k];
                }
                if (y[t * rows + r] != expected) return false;
            }
        return true;
    }

    const int ProtRead = 1, ProtWrite = 2, ProtExec = 4, MapPrivate = 2;
    const uint MemCommit = 0x1000, MemReserve = 0x2000, PageReadWrite = 4, PageExecuteRead = 0x20;

    static nint Allocate(nuint size)
    {
        if (OperatingSystem.IsWindows()) return VirtualAlloc(0, size, MemCommit | MemReserve, PageReadWrite);
        var anonymous = OperatingSystem.IsLinux() || OperatingSystem.IsAndroid() ? 0x20 : 0x1000;
        var p = Mmap(0, size, ProtRead | ProtWrite, MapPrivate | anonymous, -1, 0);
        return p == -1 ? 0 : p;
    }

    static bool Protect(nint memory, nuint size)
    {
        if (OperatingSystem.IsWindows()) return VirtualProtect(memory, size, PageExecuteRead, out _) != 0;
        return Mprotect(memory, size, ProtRead | ProtExec) == 0;
    }

    [LibraryImport("libc", EntryPoint = "mmap")]
    private static partial nint Mmap(nint address, nuint length, int protection, int flags, int fd, nint offset);

    [LibraryImport("libc", EntryPoint = "mprotect")]
    private static partial int Mprotect(nint address, nuint length, int protection);

    [LibraryImport("kernel32", EntryPoint = "VirtualAlloc")]
    private static partial nint VirtualAlloc(nint address, nuint size, uint type, uint protection);

    [LibraryImport("kernel32", EntryPoint = "VirtualProtect")]
    private static partial int VirtualProtect(nint address, nuint size, uint protection, out uint old);
}
#endif
