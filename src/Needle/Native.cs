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

    public static void Clear(void* p, nuint bytes) => NativeMemory.Clear(p, bytes);

    public static void* AlignedAlloc(nuint bytes, nuint alignment) => NativeMemory.AlignedAlloc(bytes, alignment);

    public static void AlignedFree(void* p) => NativeMemory.AlignedFree(p);
#endif
}

internal static unsafe partial class Libc
{
#if NETSTANDARD2_1
    public static byte* MapReadOnly(string path, out nuint length)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.SequentialScan);
        length = (nuint)stream.Length;
        var memory = (byte*)Mem.Alloc(length);
        for (nuint done = 0; done < length;)
        {
            var read = stream.Read(new Span<byte>(memory + done, (int)Math.Min(1 << 24, (long)(length - done))));
            if (read <= 0) throw new IOException($"short read on {path}");
            done += (nuint)read;
        }
        return memory;
    }

    public static void Unmap(byte* address, nuint length) => Mem.Free(address);
#else
    const int ProtRead = 1;
    const int MapPrivate = 2;
    const int MapPopulate = 0x8000;

    [LibraryImport("libc", EntryPoint = "mmap")]
    private static partial nint Mmap(nint address, nuint length, int protection, int flags, int fd, nint offset);

    [LibraryImport("libc", EntryPoint = "munmap")]
    private static partial int Munmap(nint address, nuint length);

    public static byte* MapReadOnly(string path, out nuint length)
    {
        using var handle = File.OpenHandle(path);
        length = (nuint)RandomAccess.GetLength(handle);
        var address = Mmap(0, length, ProtRead, MapPrivate | MapPopulate, (int)handle.DangerousGetHandle(), 0);
        if (address == -1) throw new IOException($"mmap failed for {path}");
        return (byte*)address;
    }

    public static void Unmap(byte* address, nuint length) => Munmap((nint)address, length);
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
