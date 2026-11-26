using System;
using System.Text;
using System.Text.RegularExpressions;

namespace Needle;

/// <summary>
/// 4.x per-tool triggers: case-insensitive regexes beside a tool's description. A request that matches one
/// routes its first call to that tool, requires a call, and ships the call even below the confidence floor.
/// </summary>
internal sealed class Triggers
{
    static readonly TimeSpan Timeout = TimeSpan.FromMilliseconds(50);

    readonly Regex?[] patterns;
    readonly int[] tool;
    readonly char[] chars;

    Triggers(Regex?[] patterns, int[] tool, int maxInputBytes)
    {
        this.patterns = patterns;
        this.tool = tool;
        chars = new char[maxInputBytes + 1];
    }

    /// <summary>Null when no tool has triggers, so sessions without them pay nothing.</summary>
    public static Triggers? Compile(ToolSet set, string rawToolsJson, int maxInputBytes)
    {
        var raw = RepairPlan.RawTriggers(rawToolsJson);
        if (raw.Count == 0) return null;
        var patterns = new System.Collections.Generic.List<Regex?>();
        var tools = new System.Collections.Generic.List<int>();
        for (var t = 0; t < set.Tools.Count && t < 64; t++)
        {
            var original = set.OriginalName(set.Tools[t].GetString("name") ?? "");
            if (!raw.TryGetValue(original, out var list)) continue;
            foreach (var text in list)
            {
                Regex? regex;
                try { regex = new Regex(text, RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, Timeout); }
                catch (ArgumentException) { regex = null; }   // a pattern .NET cannot read never fires
                patterns.Add(regex);
                tools.Add(t);
            }
        }
        return patterns.Count == 0 ? null : new Triggers(patterns.ToArray(), tools.ToArray(), maxInputBytes);
    }

    /// <summary>Bit t set when tool t's trigger matches the request.</summary>
    public ulong Match(ReadOnlySpan<byte> request)
    {
        if (request.Length >= chars.Length) request = request[..(chars.Length - 1)];
        var length = Encoding.UTF8.GetChars(request, chars);
        var mask = 0UL;
        for (var p = 0; p < patterns.Length; p++)
        {
            if (patterns[p] is not { } regex || (mask >> tool[p] & 1) != 0) continue;
            try
            {
#if NET7_0_OR_GREATER
                if (regex.IsMatch(chars.AsSpan(0, length))) mask |= 1UL << tool[p];
#else
                if (regex.IsMatch(new string(chars, 0, length))) mask |= 1UL << tool[p];
#endif
            }
            catch (RegexMatchTimeoutException) { }
        }
        return mask;
    }
}
