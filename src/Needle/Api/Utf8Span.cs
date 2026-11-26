using System;
using System.Text;

namespace Needle;

/// <summary>
/// UTF-8 text that lives inside the agent. Compare it, copy it or format it without allocating;
/// <see cref="ToString"/> is the one member that makes a <see cref="string"/>.
/// </summary>
public readonly ref struct Utf8Span
{
    public Utf8Span(ReadOnlySpan<byte> bytes) => Bytes = bytes;

    public ReadOnlySpan<byte> Bytes { get; }

    public int Length => Bytes.Length;

    public bool IsEmpty => Bytes.IsEmpty;

    public static implicit operator ReadOnlySpan<byte>(Utf8Span text) => text.Bytes;

    public bool Equals(ReadOnlySpan<byte> utf8) => Bytes.SequenceEqual(utf8);

    /// <summary>Ordinal comparison with a .NET string, without allocating.</summary>
    public bool Equals(string? text) => text != null && Utf8Compare.EqualsOrdinal(Bytes, text.AsSpan());

    public bool Equals(ReadOnlySpan<char> text) => Utf8Compare.EqualsOrdinal(Bytes, text);

    /// <summary>The number of UTF-16 chars <see cref="TryCopyTo"/> needs.</summary>
    public int CharCount => Bytes.IsEmpty ? 0 : Encoding.UTF8.GetCharCount(Bytes);

    public bool TryCopyTo(Span<char> destination, out int charsWritten)
    {
        if (CharCount > destination.Length)
        {
            charsWritten = 0;
            return false;
        }
        charsWritten = Bytes.IsEmpty ? 0 : Encoding.UTF8.GetChars(Bytes, destination);
        return true;
    }

    public override string ToString() => Bytes.IsEmpty ? "" : Encoding.UTF8.GetString(Bytes);
}

internal static class Utf8Compare
{
    public static bool EqualsOrdinal(ReadOnlySpan<byte> utf8, ReadOnlySpan<char> text)
    {
        var i = 0;
        for (; i < utf8.Length && i < text.Length; i++)
        {
            var b = utf8[i];
            if (b >= 0x80 || text[i] >= 0x80) return SlowEquals(utf8, text);
            if (b != text[i]) return false;
        }
        return i == utf8.Length && i == text.Length;
    }

    static bool SlowEquals(ReadOnlySpan<byte> utf8, ReadOnlySpan<char> text)
    {
        var count = Encoding.UTF8.GetByteCount(text);
        if (count != utf8.Length) return false;
        Span<byte> stack = stackalloc byte[256];
        var buffer = count <= stack.Length ? stack[..count] : new byte[count];
        Encoding.UTF8.GetBytes(text, buffer);
        return buffer.SequenceEqual(utf8);
    }
}
