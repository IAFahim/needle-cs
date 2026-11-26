using System;
using System.Buffers.Text;

namespace Needle;

/// <summary>Reads members of one call's argument object straight from its UTF-8 text, without a parse tree.</summary>
internal static class JsonArgs
{
    public static bool TryFind(ReadOnlySpan<byte> json, ReadOnlySpan<byte> key, out ReadOnlySpan<byte> value, out bool isString)
        => TryFind(json, key, default, false, out value, out isString);

    public static bool TryFind(ReadOnlySpan<byte> json, ReadOnlySpan<char> key, out ReadOnlySpan<byte> value, out bool isString)
        => TryFind(json, default, key, true, out value, out isString);

    static bool TryFind(ReadOnlySpan<byte> json, ReadOnlySpan<byte> keyUtf8, ReadOnlySpan<char> keyChars, bool chars, out ReadOnlySpan<byte> value, out bool isString)
    {
        value = default;
        isString = false;
        var at = 0;
        SkipSpace(json, ref at);
        if (at >= json.Length || json[at] != (byte)'{') return false;
        at++;
        while (true)
        {
            SkipSpace(json, ref at);
            if (at >= json.Length || json[at] == (byte)'}') return false;
            if (json[at] == (byte)',') { at++; continue; }
            if (json[at] != (byte)'"') return false;
            var keyStart = at + 1;
            if (!SkipString(json, ref at)) return false;
            var name = json.Slice(keyStart, at - 1 - keyStart);
            SkipSpace(json, ref at);
            if (at >= json.Length || json[at] != (byte)':') return false;
            at++;
            SkipSpace(json, ref at);
            var valueStart = at;
            var str = at < json.Length && json[at] == (byte)'"';
            if (!SkipValue(json, ref at)) return false;
            var matched = chars ? Utf8Compare.EqualsOrdinal(name, keyChars) : name.SequenceEqual(keyUtf8);
            if (matched)
            {
                isString = str;
                value = str ? json.Slice(valueStart + 1, at - valueStart - 2) : json.Slice(valueStart, at - valueStart);
                return true;
            }
        }
    }

    static void SkipSpace(ReadOnlySpan<byte> json, ref int at)
    {
        while (at < json.Length && json[at] is (byte)' ' or (byte)'\n' or (byte)'\r' or (byte)'\t') at++;
    }

    static bool SkipString(ReadOnlySpan<byte> json, ref int at)
    {
        at++;
        while (at < json.Length)
        {
            var c = json[at++];
            if (c == (byte)'\\') at++;
            else if (c == (byte)'"') return true;
        }
        return false;
    }

    static bool SkipValue(ReadOnlySpan<byte> json, ref int at)
    {
        if (at >= json.Length) return false;
        var c = json[at];
        if (c == (byte)'"') return SkipString(json, ref at);
        if (c == (byte)'{' || c == (byte)'[')
        {
            var depth = 0;
            while (at < json.Length)
            {
                c = json[at];
                if (c == (byte)'"') { if (!SkipString(json, ref at)) return false; continue; }
                if (c == (byte)'{' || c == (byte)'[') depth++;
                else if (c == (byte)'}' || c == (byte)']')
                {
                    depth--;
                    if (depth == 0) { at++; return true; }
                }
                at++;
            }
            return false;
        }
        var start = at;
        while (at < json.Length && json[at] is not ((byte)',' or (byte)'}' or (byte)']' or (byte)' ' or (byte)'\n' or (byte)'\r' or (byte)'\t')) at++;
        return at > start;
    }

    /// <summary>Copies a JSON string's contents with its escapes decoded.</summary>
    public static bool TryUnescape(ReadOnlySpan<byte> escaped, Span<byte> destination, out int written)
    {
        written = 0;
        for (var i = 0; i < escaped.Length; i++)
        {
            var c = escaped[i];
            if (c != (byte)'\\')
            {
                if (written >= destination.Length) return false;
                destination[written++] = c;
                continue;
            }
            if (++i >= escaped.Length) return false;
            c = escaped[i];
            if (c == (byte)'u')
            {
                if (i + 4 >= escaped.Length || !Utf8Parser.TryParse(escaped.Slice(i + 1, 4), out ushort unit, out var used, 'x') || used != 4) return false;
                i += 4;
                int code = unit;
                if (unit >= 0xD800 && unit <= 0xDBFF && i + 6 < escaped.Length && escaped[i + 1] == (byte)'\\' && escaped[i + 2] == (byte)'u'
                    && Utf8Parser.TryParse(escaped.Slice(i + 3, 4), out ushort low, out used, 'x') && used == 4 && low >= 0xDC00 && low <= 0xDFFF)
                {
                    code = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00);
                    i += 6;
                }
                if (!WriteCodePoint(code, destination, ref written)) return false;
                continue;
            }
            byte decoded = c switch
            {
                (byte)'n' => (byte)'\n',
                (byte)'t' => (byte)'\t',
                (byte)'r' => (byte)'\r',
                (byte)'b' => 8,
                (byte)'f' => 12,
                _ => c,
            };
            if (written >= destination.Length) return false;
            destination[written++] = decoded;
        }
        return true;
    }

    static bool WriteCodePoint(int code, Span<byte> destination, ref int written)
    {
        var needed = code < 0x80 ? 1 : code < 0x800 ? 2 : code < 0x10000 ? 3 : 4;
        if (written + needed > destination.Length) return false;
        switch (needed)
        {
            case 1: destination[written++] = (byte)code; break;
            case 2:
                destination[written++] = (byte)(0xC0 | (code >> 6));
                destination[written++] = (byte)(0x80 | (code & 0x3F));
                break;
            case 3:
                destination[written++] = (byte)(0xE0 | (code >> 12));
                destination[written++] = (byte)(0x80 | ((code >> 6) & 0x3F));
                destination[written++] = (byte)(0x80 | (code & 0x3F));
                break;
            default:
                destination[written++] = (byte)(0xF0 | (code >> 18));
                destination[written++] = (byte)(0x80 | ((code >> 12) & 0x3F));
                destination[written++] = (byte)(0x80 | ((code >> 6) & 0x3F));
                destination[written++] = (byte)(0x80 | (code & 0x3F));
                break;
        }
        return true;
    }
}
