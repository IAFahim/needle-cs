using System;

namespace Needle;

internal readonly struct CallSpan
{
    public readonly int Tool;
    public readonly int ArgumentsStart;
    public readonly int ArgumentsLength;

    public CallSpan(int tool, int argumentsStart, int argumentsLength)
    {
        Tool = tool;
        ArgumentsStart = argumentsStart;
        ArgumentsLength = argumentsLength;
    }
}

internal ref struct ByteSink
{
    readonly Span<byte> destination;
    int length;

    public ByteSink(Span<byte> destination)
    {
        this.destination = destination;
        length = 0;
    }

    public int Length => length;

    public void Put(byte value)
    {
        if (length < destination.Length) destination[length] = value;
        length++;
    }

    public void Put(ReadOnlySpan<byte> values)
    {
        var room = destination.Length - length;
        if (room > 0) values[..Math.Min(room, values.Length)].CopyTo(destination[length..]);
        length += values.Length;
    }
}

internal static class CallRestorer
{
    const int MaxNesting = 64;
    const int Replacement = 0xFFFD;

    static ReadOnlySpan<byte> NameHead => "{\"name\":\""u8;
    static ReadOnlySpan<byte> ArgumentsHead => "\",\"arguments\":"u8;
    static ReadOnlySpan<byte> HexDigits => "0123456789abcdef"u8;

    public static bool TryRestore(AliasTable aliases, ReadOnlySpan<byte> generated, Span<byte> destination, Span<byte> keyScratch, Span<CallSpan> spans, out int required, out int callCount)
    {
        var sink = new ByteSink(destination);
        var at = 0;
        callCount = 0;
        required = 0;
        if (!Take(generated, ref at, (byte)'[')) return false;
        sink.Put((byte)'[');
        if (at < generated.Length && generated[at] == (byte)']') { sink.Put((byte)']'); at++; }
        else
            for (;;)
            {
                if (!Take(generated, ref at, NameHead)) return false;
                var aliasStart = at;
                while (at < generated.Length && generated[at] != (byte)'"') at++;
                if (at >= generated.Length || !aliases.TryFindTool(generated[aliasStart..at], out var tool) || callCount >= spans.Length) return false;
                if (!Take(generated, ref at, ArgumentsHead)) return false;
                sink.Put(NameHead);
                sink.Put(aliases.OriginalToolEscaped(tool));
                sink.Put(ArgumentsHead);
                var argumentsStart = sink.Length;
                if (!CopyValue(aliases, tool, generated, ref at, ref sink, keyScratch)) return false;
                spans[callCount++] = new CallSpan(tool, argumentsStart, sink.Length - argumentsStart);
                if (!Take(generated, ref at, (byte)'}')) return false;
                sink.Put((byte)'}');
                if (at < generated.Length && generated[at] == (byte)',') { at++; sink.Put((byte)','); continue; }
                if (!Take(generated, ref at, (byte)']')) return false;
                sink.Put((byte)']');
                break;
            }
        required = sink.Length;
        return at == generated.Length;
    }

    static bool Take(ReadOnlySpan<byte> source, ref int at, byte expected)
    {
        if (at >= source.Length || source[at] != expected) return false;
        at++;
        return true;
    }

    static bool Take(ReadOnlySpan<byte> source, ref int at, ReadOnlySpan<byte> expected)
    {
        if (!source[at..].StartsWith(expected)) return false;
        at += expected.Length;
        return true;
    }

    static bool CopyValue(AliasTable aliases, int tool, ReadOnlySpan<byte> source, ref int at, ref ByteSink sink, Span<byte> keyScratch)
    {
        var depth = 0;
        var objects = 0UL;
        var expectKey = false;
        do
        {
            if (at >= source.Length) return false;
            var c = source[at];
            switch (c)
            {
                case (byte)'{':
                    if (depth >= MaxNesting) return false;
                    objects = (objects << 1) | 1;
                    depth++;
                    expectKey = true;
                    sink.Put(c);
                    at++;
                    break;
                case (byte)'[':
                    if (depth >= MaxNesting) return false;
                    objects <<= 1;
                    depth++;
                    expectKey = false;
                    sink.Put(c);
                    at++;
                    break;
                case (byte)'}':
                case (byte)']':
                    if (depth == 0) return false;
                    objects >>= 1;
                    depth--;
                    expectKey = false;
                    sink.Put(c);
                    at++;
                    break;
                case (byte)',':
                    expectKey = (objects & 1) != 0;
                    sink.Put(c);
                    at++;
                    break;
                case (byte)':':
                    expectKey = false;
                    sink.Put(c);
                    at++;
                    break;
                case (byte)'"':
                    if (expectKey && (objects & 1) != 0) { if (!CopyKey(aliases, tool, source, ref at, ref sink, keyScratch)) return false; }
                    else if (!CopyString(source, ref at, ref sink)) return false;
                    expectKey = false;
                    break;
                case (byte)' ':
                case (byte)'\t':
                case (byte)'\r':
                case (byte)'\n':
                    at++;
                    break;
                default:
                    while (at < source.Length && source[at] is not ((byte)',' or (byte)'}' or (byte)']' or (byte)' ' or (byte)'\t' or (byte)'\r' or (byte)'\n'))
                        sink.Put(source[at++]);
                    expectKey = false;
                    break;
            }
        }
        while (depth > 0);
        return true;
    }

    static bool CopyKey(AliasTable aliases, int tool, ReadOnlySpan<byte> source, ref int at, ref ByteSink sink, Span<byte> keyScratch)
    {
        var start = at;
        var scratch = new ByteSink(keyScratch);
        if (!CopyString(source, ref at, ref scratch)) return false;
        if (scratch.Length > keyScratch.Length)
        {
            at = start;
            return CopyString(source, ref at, ref sink);
        }
        var canonical = keyScratch[..scratch.Length];
        if (aliases.TryFindProperty(tool, canonical[1..^1], out var original))
        {
            sink.Put((byte)'"');
            sink.Put(original);
            sink.Put((byte)'"');
        }
        else sink.Put(canonical);
        return true;
    }

    static bool CopyString(ReadOnlySpan<byte> source, ref int at, ref ByteSink sink)
    {
        sink.Put((byte)'"');
        at++;
        while (at < source.Length)
        {
            var c = source[at];
            if (c == (byte)'"')
            {
                at++;
                sink.Put((byte)'"');
                return true;
            }
            if (c == (byte)'\\')
            {
                if (!ReadEscape(source, ref at, out var escaped)) return false;
                PutCodePoint(ref sink, escaped);
            }
            else if (c < 0x80)
            {
                PutCodePoint(ref sink, c);
                at++;
            }
            else
            {
                var consumed = DecodeSequence(source[at..], out var codePoint);
                PutCodePoint(ref sink, codePoint);
                at += consumed;
            }
        }
        return false;
    }

    static int DecodeSequence(ReadOnlySpan<byte> source, out int codePoint)
    {
        codePoint = Replacement;
        var lead = source[0];
        int needed;
        var lowest = 0x80;
        var highest = 0xBF;
        if (lead >= 0xC2 && lead <= 0xDF) { needed = 1; codePoint = lead & 0x1F; }
        else if (lead >= 0xE0 && lead <= 0xEF)
        {
            needed = 2;
            codePoint = lead & 0x0F;
            if (lead == 0xE0) lowest = 0xA0;
            else if (lead == 0xED) highest = 0x9F;
        }
        else if (lead >= 0xF0 && lead <= 0xF4)
        {
            needed = 3;
            codePoint = lead & 0x07;
            if (lead == 0xF0) lowest = 0x90;
            else if (lead == 0xF4) highest = 0x8F;
        }
        else return 1;
        for (var i = 1; i <= needed; i++)
        {
            if (i >= source.Length || source[i] < lowest || source[i] > highest)
            {
                codePoint = Replacement;
                return i;
            }
            codePoint = (codePoint << 6) | (source[i] & 0x3F);
            lowest = 0x80;
            highest = 0xBF;
        }
        return needed + 1;
    }

    static bool ReadEscape(ReadOnlySpan<byte> source, ref int at, out int codePoint)
    {
        codePoint = Replacement;
        if (at + 1 >= source.Length) return false;
        var escape = source[at + 1];
        at += 2;
        switch (escape)
        {
            case (byte)'n': codePoint = '\n'; return true;
            case (byte)'r': codePoint = '\r'; return true;
            case (byte)'t': codePoint = '\t'; return true;
            case (byte)'b': codePoint = '\b'; return true;
            case (byte)'f': codePoint = '\f'; return true;
            case (byte)'u':
                if (!ReadUnit(source, ref at, out var unit)) return false;
                if (unit >= 0xD800 && unit <= 0xDBFF && at + 1 < source.Length && source[at] == (byte)'\\' && source[at + 1] == (byte)'u')
                {
                    var next = at + 2;
                    if (ReadUnit(source, ref next, out var low) && low >= 0xDC00 && low <= 0xDFFF)
                    {
                        at = next;
                        codePoint = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00);
                        return true;
                    }
                }
                codePoint = unit >= 0xD800 && unit <= 0xDFFF ? Replacement : unit;
                return true;
            default:
                codePoint = escape;
                return true;
        }
    }

    static bool ReadUnit(ReadOnlySpan<byte> source, ref int at, out int unit)
    {
        unit = 0;
        if (at + 4 > source.Length) return false;
        for (var i = 0; i < 4; i++)
        {
            var c = source[at + i];
            var nibble = c >= (byte)'0' && c <= (byte)'9' ? c - '0' : c >= (byte)'a' && c <= (byte)'f' ? c - 'a' + 10 : c >= (byte)'A' && c <= (byte)'F' ? c - 'A' + 10 : -1;
            if (nibble < 0) return false;
            unit = unit * 16 + nibble;
        }
        at += 4;
        return true;
    }

    static void PutCodePoint(ref ByteSink sink, int codePoint)
    {
        switch (codePoint)
        {
            case '"': sink.Put((byte)'\\'); sink.Put((byte)'"'); return;
            case '\\': sink.Put((byte)'\\'); sink.Put((byte)'\\'); return;
            case '\n': sink.Put((byte)'\\'); sink.Put((byte)'n'); return;
            case '\r': sink.Put((byte)'\\'); sink.Put((byte)'r'); return;
            case '\t': sink.Put((byte)'\\'); sink.Put((byte)'t'); return;
            case '\b': sink.Put((byte)'\\'); sink.Put((byte)'b'); return;
            case '\f': sink.Put((byte)'\\'); sink.Put((byte)'f'); return;
        }
        if (codePoint < 0x20)
        {
            sink.Put("\\u00"u8);
            sink.Put(HexDigits[codePoint >> 4]);
            sink.Put(HexDigits[codePoint & 15]);
        }
        else if (codePoint < 0x80) sink.Put((byte)codePoint);
        else if (codePoint < 0x800)
        {
            sink.Put((byte)(0xC0 | (codePoint >> 6)));
            sink.Put((byte)(0x80 | (codePoint & 0x3F)));
        }
        else if (codePoint < 0x10000)
        {
            sink.Put((byte)(0xE0 | (codePoint >> 12)));
            sink.Put((byte)(0x80 | ((codePoint >> 6) & 0x3F)));
            sink.Put((byte)(0x80 | (codePoint & 0x3F)));
        }
        else
        {
            sink.Put((byte)(0xF0 | (codePoint >> 18)));
            sink.Put((byte)(0x80 | ((codePoint >> 12) & 0x3F)));
            sink.Put((byte)(0x80 | ((codePoint >> 6) & 0x3F)));
            sink.Put((byte)(0x80 | (codePoint & 0x3F)));
        }
    }
}
