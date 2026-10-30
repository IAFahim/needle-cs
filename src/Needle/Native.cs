using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

namespace Needle;

internal static unsafe class Mem
{
#if NEEDLE_ALLOC_COUNTERS
    public static bool CountersEnabled => true;
#else
    public static bool CountersEnabled => false;
#endif

    static long allocations;
    static long allocatedBytes;

    public static long Allocations => Volatile.Read(ref allocations);

    public static long AllocatedBytes => Volatile.Read(ref allocatedBytes);

    [Conditional("NEEDLE_ALLOC_COUNTERS")]
    static void Count(nuint bytes)
    {
        Interlocked.Increment(ref allocations);
        Interlocked.Add(ref allocatedBytes, (long)bytes);
    }

#if NETSTANDARD2_1
    public static void* Alloc(nuint bytes)
    {
        Count(bytes);
        return (void*)Marshal.AllocHGlobal((nint)Math.Max(1, (long)bytes));
    }

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
    public static void* Alloc(nuint bytes)
    {
        Count(bytes);
        return NativeMemory.Alloc(bytes);
    }

    public static void* AllocZeroed(nuint bytes)
    {
        Count(bytes);
        return NativeMemory.AllocZeroed(bytes);
    }

    public static void Free(void* p) => NativeMemory.Free(p);

    public static void* Realloc(void* p, nuint usedBytes, nuint newBytes) => NativeMemory.Realloc(p, newBytes);

    public static void Clear(void* p, nuint bytes) => NativeMemory.Clear(p, bytes);

    public static void* AlignedAlloc(nuint bytes, nuint alignment)
    {
        Count(bytes);
        return NativeMemory.AlignedAlloc(bytes, alignment);
    }

    public static void AlignedFree(void* p) => NativeMemory.AlignedFree(p);
#endif
}

internal static unsafe partial class Paging
{
    public const nuint HugePageBytes = 2u << 20;

    public static readonly bool HugePagesEnabled = ParseHugePagesSetting(Environment.GetEnvironmentVariable("NEEDLE_HUGEPAGES"));

    internal static bool ParseHugePagesSetting(string? value) => value is { Length: > 0 } && value != "0";

#if NETSTANDARD2_1
    public static bool WillNeed(byte* address, nuint length) => false;

    public static bool AdviseHugePages(byte* address, nuint length) => false;
#else
    const int AdviceWillNeed = 3;
    const int AdviceHugePage = 14;

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

    public static bool AdviseHugePages(byte* address, nuint length)
    {
        if (!HugePagesEnabled || length == 0 || !OperatingSystem.IsLinux()) return false;
        const nuint PageBytes = 4096;
        var start = (nuint)address & ~(PageBytes - 1);
        try
        {
            return Madvise((nint)start, length + ((nuint)address - start), AdviceHugePage) == 0;
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
        if (Paging.HugePagesEnabled && capacity >= Paging.HugePageBytes)
        {
            Base = (byte*)Mem.AlignedAlloc(capacity, Paging.HugePageBytes);
            Paging.AdviseHugePages(Base, capacity);
        }
        else
        {
            Base = (byte*)Mem.AlignedAlloc(capacity, 64);
        }
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
