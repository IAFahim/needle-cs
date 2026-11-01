using System;
using System.Runtime.CompilerServices;
using System.Text;

namespace Needle.Tools;

#if NET8_0_OR_GREATER
[InlineArray(Capacity)]
#endif
unsafe struct Utf8Text8Buffer
{
#if NET8_0_OR_GREATER
    public const int Capacity = 8;
    public byte E0;
#else
    public const int Capacity = 8;
    public fixed byte Data[8];
#endif
}

#if NET8_0_OR_GREATER
[InlineArray(Capacity)]
#endif
unsafe struct Utf8Text16Buffer
{
    public const int Capacity = 16;
#if NET8_0_OR_GREATER
    public byte E0;
#else
    public fixed byte Data[16];
#endif
}

#if NET8_0_OR_GREATER
[InlineArray(Capacity)]
#endif
unsafe struct Utf8Text32Buffer
{
    public const int Capacity = 32;
#if NET8_0_OR_GREATER
    public byte E0;
#else
    public fixed byte Data[32];
#endif
}

#if NET8_0_OR_GREATER
[InlineArray(Capacity)]
#endif
unsafe struct Utf8Text64Buffer
{
    public const int Capacity = 64;
#if NET8_0_OR_GREATER
    public byte E0;
#else
    public fixed byte Data[64];
#endif
}

#if NET8_0_OR_GREATER
[InlineArray(Capacity)]
#endif
unsafe struct Utf8Text128Buffer
{
    public const int Capacity = 128;
#if NET8_0_OR_GREATER
    public byte E0;
#else
    public fixed byte Data[128];
#endif
}

#if NET8_0_OR_GREATER
[InlineArray(Capacity)]
#endif
unsafe struct Utf8Text256Buffer
{
    public const int Capacity = 256;
#if NET8_0_OR_GREATER
    public byte E0;
#else
    public fixed byte Data[256];
#endif
}

/// <summary>Fixed UTF-8 text slot. Blittable, truncating, allocation-free to fill.</summary>
public struct Utf8Text8
{
    Utf8Text8Buffer data;
    public int Length;

    public const int Capacity = Utf8Text8Buffer.Capacity;

    public unsafe Span<byte> Buffer
    {
        get
        {
#if NET8_0_OR_GREATER
            return System.Runtime.InteropServices.MemoryMarshal.CreateSpan(ref Unsafe.As<Utf8Text8Buffer, byte>(ref data), Capacity);
#else
            ref var local = ref data;
            fixed (byte* p = local.Data) return new Span<byte>(p, Capacity);
#endif
        }
    }

    public ReadOnlySpan<byte> Span => Buffer[..Length];

    public void Set(ReadOnlySpan<byte> utf8)
    {
        var count = Math.Min(utf8.Length, Capacity);
        utf8[..count].CopyTo(Buffer);
        Length = count;
    }

    public void Set(string value)
    {
        var length = 0;
        for (var i = 0; i < value.Length;)
        {
            var take = i + 1 < value.Length && char.IsHighSurrogate(value[i]) && char.IsLowSurrogate(value[i + 1]) ? 2 : 1;
            var need = Encoding.UTF8.GetByteCount(value.AsSpan(i, take));
            if (length + need > Capacity) break;
            Encoding.UTF8.GetBytes(value.AsSpan(i, take), Buffer[length..]);
            length += need;
            i += take;
        }
        Length = length;
    }

    public override string ToString() => Encoding.UTF8.GetString(Span);
}

/// <summary>Fixed UTF-8 text slot. Blittable, truncating, allocation-free to fill.</summary>
public struct Utf8Text16
{
    Utf8Text16Buffer data;
    public int Length;

    public const int Capacity = Utf8Text16Buffer.Capacity;

    public unsafe Span<byte> Buffer
    {
        get
        {
#if NET8_0_OR_GREATER
            return System.Runtime.InteropServices.MemoryMarshal.CreateSpan(ref Unsafe.As<Utf8Text16Buffer, byte>(ref data), Capacity);
#else
            ref var local = ref data;
            fixed (byte* p = local.Data) return new Span<byte>(p, Capacity);
#endif
        }
    }

    public ReadOnlySpan<byte> Span => Buffer[..Length];

    public void Set(ReadOnlySpan<byte> utf8)
    {
        var count = Math.Min(utf8.Length, Capacity);
        utf8[..count].CopyTo(Buffer);
        Length = count;
    }

    public void Set(string value)
    {
        var length = 0;
        for (var i = 0; i < value.Length;)
        {
            var take = i + 1 < value.Length && char.IsHighSurrogate(value[i]) && char.IsLowSurrogate(value[i + 1]) ? 2 : 1;
            var need = Encoding.UTF8.GetByteCount(value.AsSpan(i, take));
            if (length + need > Capacity) break;
            Encoding.UTF8.GetBytes(value.AsSpan(i, take), Buffer[length..]);
            length += need;
            i += take;
        }
        Length = length;
    }

    public override string ToString() => Encoding.UTF8.GetString(Span);
}

/// <summary>Fixed UTF-8 text slot. Blittable, truncating, allocation-free to fill.</summary>
public struct Utf8Text32
{
    Utf8Text32Buffer data;
    public int Length;

    public const int Capacity = Utf8Text32Buffer.Capacity;

    public unsafe Span<byte> Buffer
    {
        get
        {
#if NET8_0_OR_GREATER
            return System.Runtime.InteropServices.MemoryMarshal.CreateSpan(ref Unsafe.As<Utf8Text32Buffer, byte>(ref data), Capacity);
#else
            ref var local = ref data;
            fixed (byte* p = local.Data) return new Span<byte>(p, Capacity);
#endif
        }
    }

    public ReadOnlySpan<byte> Span => Buffer[..Length];

    public void Set(ReadOnlySpan<byte> utf8)
    {
        var count = Math.Min(utf8.Length, Capacity);
        utf8[..count].CopyTo(Buffer);
        Length = count;
    }

    public void Set(string value)
    {
        var length = 0;
        for (var i = 0; i < value.Length;)
        {
            var take = i + 1 < value.Length && char.IsHighSurrogate(value[i]) && char.IsLowSurrogate(value[i + 1]) ? 2 : 1;
            var need = Encoding.UTF8.GetByteCount(value.AsSpan(i, take));
            if (length + need > Capacity) break;
            Encoding.UTF8.GetBytes(value.AsSpan(i, take), Buffer[length..]);
            length += need;
            i += take;
        }
        Length = length;
    }

    public override string ToString() => Encoding.UTF8.GetString(Span);
}

/// <summary>Fixed UTF-8 text slot. Blittable, truncating, allocation-free to fill.</summary>
public struct Utf8Text64
{
    Utf8Text64Buffer data;
    public int Length;

    public const int Capacity = Utf8Text64Buffer.Capacity;

    public unsafe Span<byte> Buffer
    {
        get
        {
#if NET8_0_OR_GREATER
            return System.Runtime.InteropServices.MemoryMarshal.CreateSpan(ref Unsafe.As<Utf8Text64Buffer, byte>(ref data), Capacity);
#else
            ref var local = ref data;
            fixed (byte* p = local.Data) return new Span<byte>(p, Capacity);
#endif
        }
    }

    public ReadOnlySpan<byte> Span => Buffer[..Length];

    public void Set(ReadOnlySpan<byte> utf8)
    {
        var count = Math.Min(utf8.Length, Capacity);
        utf8[..count].CopyTo(Buffer);
        Length = count;
    }

    public void Set(string value)
    {
        var length = 0;
        for (var i = 0; i < value.Length;)
        {
            var take = i + 1 < value.Length && char.IsHighSurrogate(value[i]) && char.IsLowSurrogate(value[i + 1]) ? 2 : 1;
            var need = Encoding.UTF8.GetByteCount(value.AsSpan(i, take));
            if (length + need > Capacity) break;
            Encoding.UTF8.GetBytes(value.AsSpan(i, take), Buffer[length..]);
            length += need;
            i += take;
        }
        Length = length;
    }

    public override string ToString() => Encoding.UTF8.GetString(Span);
}

/// <summary>Fixed UTF-8 text slot. Blittable, truncating, allocation-free to fill.</summary>
public struct Utf8Text128
{
    Utf8Text128Buffer data;
    public int Length;

    public const int Capacity = Utf8Text128Buffer.Capacity;

    public unsafe Span<byte> Buffer
    {
        get
        {
#if NET8_0_OR_GREATER
            return System.Runtime.InteropServices.MemoryMarshal.CreateSpan(ref Unsafe.As<Utf8Text128Buffer, byte>(ref data), Capacity);
#else
            ref var local = ref data;
            fixed (byte* p = local.Data) return new Span<byte>(p, Capacity);
#endif
        }
    }

    public ReadOnlySpan<byte> Span => Buffer[..Length];

    public void Set(ReadOnlySpan<byte> utf8)
    {
        var count = Math.Min(utf8.Length, Capacity);
        utf8[..count].CopyTo(Buffer);
        Length = count;
    }

    public void Set(string value)
    {
        var length = 0;
        for (var i = 0; i < value.Length;)
        {
            var take = i + 1 < value.Length && char.IsHighSurrogate(value[i]) && char.IsLowSurrogate(value[i + 1]) ? 2 : 1;
            var need = Encoding.UTF8.GetByteCount(value.AsSpan(i, take));
            if (length + need > Capacity) break;
            Encoding.UTF8.GetBytes(value.AsSpan(i, take), Buffer[length..]);
            length += need;
            i += take;
        }
        Length = length;
    }

    public override string ToString() => Encoding.UTF8.GetString(Span);
}

/// <summary>Fixed UTF-8 text slot. Blittable, truncating, allocation-free to fill.</summary>
public struct Utf8Text256
{
    Utf8Text256Buffer data;
    public int Length;

    public const int Capacity = Utf8Text256Buffer.Capacity;

    public unsafe Span<byte> Buffer
    {
        get
        {
#if NET8_0_OR_GREATER
            return System.Runtime.InteropServices.MemoryMarshal.CreateSpan(ref Unsafe.As<Utf8Text256Buffer, byte>(ref data), Capacity);
#else
            ref var local = ref data;
            fixed (byte* p = local.Data) return new Span<byte>(p, Capacity);
#endif
        }
    }

    public ReadOnlySpan<byte> Span => Buffer[..Length];

    public void Set(ReadOnlySpan<byte> utf8)
    {
        var count = Math.Min(utf8.Length, Capacity);
        utf8[..count].CopyTo(Buffer);
        Length = count;
    }

    public void Set(string value)
    {
        var length = 0;
        for (var i = 0; i < value.Length;)
        {
            var take = i + 1 < value.Length && char.IsHighSurrogate(value[i]) && char.IsLowSurrogate(value[i + 1]) ? 2 : 1;
            var need = Encoding.UTF8.GetByteCount(value.AsSpan(i, take));
            if (length + need > Capacity) break;
            Encoding.UTF8.GetBytes(value.AsSpan(i, take), Buffer[length..]);
            length += need;
            i += take;
        }
        Length = length;
    }

    public override string ToString() => Encoding.UTF8.GetString(Span);
}
