using System;

namespace Needle;

/// <summary>
/// Follows the constrained call text as it grows, a byte at a time, and says where each call's name and arguments
/// are and when the call closes. The text is the grammar's exact form,
/// <c>[{"name":"tool","arguments":{...}},...]</c>, so the name always starts nine bytes after the call's brace.
/// </summary>
internal struct CallScanner
{
    const int NameOffset = 9;

    int position;
    int depth;
    bool inString;
    bool escape;
    int stringStart;

    /// <summary>The index of the call being read, or -1 before the first.</summary>
    public int Call;
    public int CallStart;
    public int NameStart;
    public int NameEnd;
    public int ArgumentsStart;
    public int ArgumentsEnd;

    public static CallScanner Create() => new() { Call = -1, CallStart = -1, NameStart = -1, NameEnd = -1, ArgumentsStart = -1, ArgumentsEnd = -1, stringStart = -1 };

    public int Position => position;

    /// <summary>True once the current call's name has been read in full.</summary>
    public bool HasName => Call >= 0 && NameEnd >= 0;

    /// <summary>
    /// Reads on from where it stopped until something happens or the text runs out. Returns
    /// <see cref="CallScan.Named"/> when a call's name is complete, <see cref="CallScan.Closed"/> when a call ends
    /// (<paramref name="at"/> is just past its closing brace), and <see cref="CallScan.None"/> at the end of the text.
    /// </summary>
    public CallScan Next(ReadOnlySpan<byte> text, out int at)
    {
        while (position < text.Length)
        {
            var i = position++;
            var c = text[i];
            if (inString)
            {
                if (escape) escape = false;
                else if (c == (byte)'\\') escape = true;
                else if (c == (byte)'"')
                {
                    inString = false;
                    if (depth == 2 && Call >= 0 && NameEnd < 0 && stringStart == NameStart - 1)
                    {
                        NameEnd = i;
                        at = i + 1;
                        return CallScan.Named;
                    }
                }
                continue;
            }
            switch (c)
            {
                case (byte)'"':
                    inString = true;
                    stringStart = i;
                    break;
                case (byte)'{':
                case (byte)'[':
                    depth++;
                    if (c == (byte)'{' && depth == 2)
                    {
                        Call++;
                        CallStart = i;
                        NameStart = i + NameOffset;
                        NameEnd = -1;
                        ArgumentsStart = -1;
                        ArgumentsEnd = -1;
                    }
                    else if (depth == 3 && Call >= 0 && NameEnd >= 0 && ArgumentsStart < 0) ArgumentsStart = i;
                    break;
                case (byte)'}':
                case (byte)']':
                    if (depth == 3 && ArgumentsStart >= 0 && ArgumentsEnd < 0) ArgumentsEnd = i + 1;
                    depth--;
                    if (c == (byte)'}' && depth == 1 && Call >= 0)
                    {
                        at = i + 1;
                        return CallScan.Closed;
                    }
                    break;
            }
        }
        at = position;
        return CallScan.None;
    }

    /// <summary>How far the current call's arguments reach in <paramref name="text"/>; 0 before they start.</summary>
    public int ArgumentsLength(ReadOnlySpan<byte> text)
    {
        if (ArgumentsStart < 0) return 0;
        var end = ArgumentsEnd >= 0 ? ArgumentsEnd : Math.Min(position, text.Length);
        return Math.Max(0, end - ArgumentsStart);
    }
}

internal enum CallScan : byte { None, Named, Closed }
