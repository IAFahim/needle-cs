using System;
using System.Collections.Generic;
using System.Text;

namespace Needle;

internal sealed class AliasTable
{
    readonly byte[][] toolAliases;
    readonly byte[][] toolOriginalsEscaped;
    readonly string[] toolOriginals;
    readonly byte[][][] propertyAliases;
    readonly byte[][][] propertyOriginalsEscaped;

    public int LongestPropertyAlias { get; }

    public AliasTable(IReadOnlyDictionary<string, string> toolNames, IReadOnlyDictionary<string, Dictionary<string, string>> properties)
    {
        var count = toolNames.Count;
        toolAliases = new byte[count][];
        toolOriginalsEscaped = new byte[count][];
        toolOriginals = new string[count];
        propertyAliases = new byte[count][][];
        propertyOriginalsEscaped = new byte[count][][];
        var tool = 0;
        foreach (var pair in toolNames)
        {
            toolAliases[tool] = Encoding.UTF8.GetBytes(pair.Key);
            toolOriginalsEscaped[tool] = EscapedUtf8(pair.Value);
            toolOriginals[tool] = pair.Value;
            var map = properties[pair.Key];
            propertyAliases[tool] = new byte[map.Count][];
            propertyOriginalsEscaped[tool] = new byte[map.Count][];
            var property = 0;
            foreach (var entry in map)
            {
                propertyAliases[tool][property] = EscapedUtf8(entry.Key);
                propertyOriginalsEscaped[tool][property] = EscapedUtf8(entry.Value);
                LongestPropertyAlias = Math.Max(LongestPropertyAlias, propertyAliases[tool][property].Length);
                property++;
            }
            tool++;
        }
    }

    public string OriginalTool(int tool) => toolOriginals[tool];

    public ReadOnlySpan<byte> OriginalToolEscaped(int tool) => toolOriginalsEscaped[tool];

    public bool TryFindTool(ReadOnlySpan<byte> alias, out int tool)
    {
        for (tool = 0; tool < toolAliases.Length; tool++)
            if (alias.SequenceEqual(toolAliases[tool])) return true;
        return false;
    }

    public bool TryFindProperty(int tool, ReadOnlySpan<byte> aliasEscaped, out ReadOnlySpan<byte> originalEscaped)
    {
        var aliases = propertyAliases[tool];
        for (var property = 0; property < aliases.Length; property++)
            if (aliasEscaped.SequenceEqual(aliases[property]))
            {
                originalEscaped = propertyOriginalsEscaped[tool][property];
                return true;
            }
        originalEscaped = default;
        return false;
    }

    static byte[] EscapedUtf8(string text)
    {
        var builder = new StringBuilder();
        JsonValue.WriteString(builder, text);
        return Encoding.UTF8.GetBytes(builder.ToString(1, builder.Length - 2));
    }
}
