using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
#if !NETSTANDARD2_1
using System.IO.MemoryMappedFiles;
#endif

namespace Needle;

internal enum ModelStorage
{
    Borrowed,
    Owned,
    Mapped,
}

internal sealed unsafe class ModelMemory : IDisposable
{
    const int PageBytes = 4096;

    public readonly byte* Address;
    public readonly nuint Length;
    public readonly ModelStorage Storage;

#if !NETSTANDARD2_1
    readonly MemoryMappedFile? file;
    readonly MemoryMappedViewAccessor? view;
#endif
    int released;

    ModelMemory(byte* address, nuint length, ModelStorage storage)
    {
        Address = address;
        Length = length;
        Storage = storage;
    }

#if !NETSTANDARD2_1
    ModelMemory(byte* address, nuint length, MemoryMappedFile file, MemoryMappedViewAccessor view)
        : this(address, length, ModelStorage.Mapped)
    {
        this.file = file;
        this.view = view;
    }
#endif

    public static ModelMemory Borrow(byte* address, nuint length) => new(address, length, ModelStorage.Borrowed);

    public static ModelMemory Copy(byte[] bytes)
    {
        var copy = (byte*)Mem.Alloc((nuint)bytes.Length);
        if (bytes.Length > 0) Marshal.Copy(bytes, 0, (nint)copy, bytes.Length);
        return new ModelMemory(copy, (nuint)bytes.Length, ModelStorage.Owned);
    }

    public static ModelMemory Read(Stream stream)
    {
        if (!stream.CanRead) throw new ArgumentException("stream is not readable", nameof(stream));
        var known = stream.CanSeek;
        var capacity = known ? (nuint)Math.Max(1, stream.Length - stream.Position) : 1u << 20;
        var buffer = (byte*)Mem.Alloc(capacity);
        nuint filled = 0;
        try
        {
            while (true)
            {
                if (filled == capacity)
                {
                    if (known) break;
                    buffer = (byte*)Mem.Realloc(buffer, filled, capacity * 2);
                    capacity *= 2;
                }
                var read = stream.Read(new Span<byte>(buffer + filled, (int)Math.Min(1 << 24, (long)(capacity - filled))));
                if (read <= 0) break;
                filled += (nuint)read;
            }
        }
        catch
        {
            Mem.Free(buffer);
            throw;
        }
        return new ModelMemory(buffer, filled, ModelStorage.Owned);
    }

#if NETSTANDARD2_1
    public static ModelMemory Map(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.SequentialScan);
        return Read(stream);
    }

    public void Prefault()
    {
    }
#else
    public static ModelMemory Map(string path)
    {
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.RandomAccess);
        var length = stream.Length;
        if (length == 0)
        {
            stream.Dispose();
            throw new InvalidDataException($"{path} is empty");
        }
        MemoryMappedFile file;
        try
        {
            file = MemoryMappedFile.CreateFromFile(stream, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, false);
        }
        catch
        {
            stream.Dispose();
            throw;
        }
        try
        {
            var view = file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* pointer = null;
            view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            var address = pointer + view.PointerOffset;
            Paging.AdviseHugePages(address, (nuint)length);
            return new ModelMemory(address, (nuint)length, file, view);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    public void Prefault()
    {
        if (Storage == ModelStorage.Mapped) Paging.WillNeed(Address, Length);
        for (nuint offset = 0; offset < Length; offset += PageBytes) Volatile.Read(ref Address[offset]);
    }
#endif

    public void Dispose()
    {
        if (Interlocked.Exchange(ref released, 1) != 0) return;
        switch (Storage)
        {
            case ModelStorage.Owned:
                Mem.Free(Address);
                break;
#if !NETSTANDARD2_1
            case ModelStorage.Mapped:
                view!.SafeMemoryMappedViewHandle.ReleasePointer();
                view.Dispose();
                file!.Dispose();
                break;
#endif
        }
    }
}
