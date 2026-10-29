using System;
using System.IO;
using System.Runtime.InteropServices;

namespace Needle;

internal static unsafe class Mem
{
#if NETSTANDARD2_1
    public static void* Alloc(nuint bytes) => (void*)Marshal.AllocHGlobal((nint)Math.Max(1, (long)bytes));

    public static void* AllocZeroed(nuint bytes)
    {
        var p = Alloc(bytes);
        Clear(p, bytes);
        return p;
    }

    public static void Free(void* p) => Marshal.FreeHGlobal((nint)p);

    public static void* Realloc(void* p, nuint usedBytes, nuint newBytes)
    {
        var grown = Alloc(newBytes);
        Buffer.MemoryCopy(p, grown, newBytes, Math.Min(usedBytes, newBytes));
        Free(p);
        return grown;
    }

    public static void Clear(void* p, nuint bytes)
    {
        var b = (byte*)p;
        for (nuint i = 0; i < bytes; i++) b[i] = 0;
    }

    public static void* AlignedAlloc(nuint bytes, nuint alignment)
    {
        var raw = (byte*)Alloc(bytes + alignment + (nuint)sizeof(void*));
        var aligned = (byte*)(((nuint)raw + (nuint)sizeof(void*) + alignment - 1) & ~(alignment - 1));
        ((byte**)aligned)[-1] = raw;
        return aligned;
    }

    public static void AlignedFree(void* p) => Free(((byte**)p)[-1]);
#else
    public static void* Alloc(nuint bytes) => NativeMemory.Alloc(bytes);

    public static void* AllocZeroed(nuint bytes) => NativeMemory.AllocZeroed(bytes);

    public static void Free(void* p) => NativeMemory.Free(p);

    public static void* Realloc(void* p, nuint usedBytes, nuint newBytes) => NativeMemory.Realloc(p, newBytes);

    public static void Clear(void* p, nuint bytes) => NativeMemory.Clear(p, bytes);

    public static void* AlignedAlloc(nuint bytes, nuint alignment) => NativeMemory.AlignedAlloc(bytes, alignment);

    public static void AlignedFree(void* p) => NativeMemory.AlignedFree(p);
#endif
}

internal static unsafe partial class Paging
{
#if NETSTANDARD2_1
    public static bool WillNeed(byte* address, nuint length) => false;
#else
    const int AdviceWillNeed = 3;

    [LibraryImport("libc", EntryPoint = "madvise")]
    private static partial int Madvise(nint address, nuint length, int advice);

    static bool HasMadvise => OperatingSystem.IsLinux() || OperatingSystem.IsMacOS() || OperatingSystem.IsAndroid() || OperatingSystem.IsIOS() || OperatingSystem.IsFreeBSD();

    public static bool WillNeed(byte* address, nuint length)
    {
        if (!HasMadvise) return false;
        try
        {
            return Madvise((nint)address, length, AdviceWillNeed) == 0;
        }
        catch (Exception e) when (e is DllNotFoundException || e is EntryPointNotFoundException)
        {
            return false;
        }
    }
#endif
}

internal unsafe struct Arena
{
    byte* cursor;
    byte* end;
    public readonly byte* Base;

    public Arena(nuint capacity)
    {
        Base = (byte*)Mem.AlignedAlloc(capacity, 64);
        cursor = Base;
        end = Base + capacity;
    }

    public T* Take<T>(nuint count) where T : unmanaged
    {
        var aligned = (byte*)(((nuint)cursor + 63) & ~(nuint)63);
        var next = aligned + count * (nuint)sizeof(T);
        if (next > end) throw new OutOfMemoryException("arena exhausted");
        cursor = next;
        Mem.Clear(aligned, count * (nuint)sizeof(T));
        return (T*)aligned;
    }

    public T* TakeUncleared<T>(nuint count) where T : unmanaged
    {
        var aligned = (byte*)(((nuint)cursor + 63) & ~(nuint)63);
        var next = aligned + count * (nuint)sizeof(T);
        if (next > end) throw new OutOfMemoryException("arena exhausted");
        cursor = next;
        return (T*)aligned;
    }

    public void Free() => Mem.AlignedFree(Base);
}
