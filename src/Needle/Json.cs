using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Needle;

public enum JsonKind { Null, Bool, Number, String, Array, Object }

public sealed class JsonValue
{
    public JsonKind Kind { get; init; }
    public bool Boolean { get; init; }
    public string Text { get; init; } = "";
    public List<JsonValue> Items { get; init; } = new();
    public List<KeyValuePair<string, JsonValue>> Members { get; init; } = new();

    public static JsonValue Null { get; } = new() { Kind = JsonKind.Null };
    public static JsonValue Str(string text) => new() { Kind = JsonKind.String, Text = text };
    public static JsonValue Num(string text) => new() { Kind = JsonKind.Number, Text = text };
    public static JsonValue Bool(bool value) => new() { Kind = JsonKind.Bool, Boolean = value };
    public static JsonValue Array(IEnumerable<JsonValue> items) => new() { Kind = JsonKind.Array, Items = new List<JsonValue>(items) };
    public static JsonValue Object(IEnumerable<KeyValuePair<string, JsonValue>> members) => new() { Kind = JsonKind.Object, Members = new List<KeyValuePair<string, JsonValue>>(members) };

    public JsonValue? Get(string key)
    {
        foreach (var member in Members) if (member.Key == key) return member.Value;
        return null;
    }

    public string? GetString(string key) => Get(key) is { Kind: JsonKind.String } v ? v.Text : null;

    public static JsonValue Parse(ReadOnlySpan<char> text)
    {
        var at = 0;
        var value = ParseValue(text, ref at);
        SkipSpace(text, ref at);
        if (at != text.Length) throw new FormatException($"unexpected trailing JSON at {at}");
        return value;
    }

    public override string ToString()
    {
        var builder = new StringBuilder();
        Write(builder);
        return builder.ToString();
    }

    public void Write(StringBuilder b)
    {
        switch (Kind)
        {
            case JsonKind.Null: b.Append("null"); break;
            case JsonKind.Bool: b.Append(Boolean ? "true" : "false"); break;
            case JsonKind.Number: b.Append(Text); break;
            case JsonKind.String: WriteString(b, Text); break;
            case JsonKind.Array:
                b.Append('[');
                for (var i = 0; i < Items.Count; i++) { if (i > 0) b.Append(','); Items[i].Write(b); }
                b.Append(']');
                break;
            default:
                b.Append('{');
                for (var i = 0; i < Members.Count; i++)
                {
                    if (i > 0) b.Append(',');
                    WriteString(b, Members[i].Key);
                    b.Append(':');
                    Members[i].Value.Write(b);
                }
                b.Append('}');
                break;
        }
    }

    public static void WriteString(StringBuilder b, string s)
    {
        b.Append('"');
        foreach (var c in s)
        {
            switch (c)
            {
                case '"': b.Append("\\\""); break;
                case '\\': b.Append("\\\\"); break;
                case '\n': b.Append("\\n"); break;
                case '\r': b.Append("\\r"); break;
                case '\t': b.Append("\\t"); break;
                case '\b': b.Append("\\b"); break;
                case '\f': b.Append("\\f"); break;
                default:
                    if (c < 0x20) b.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    else b.Append(c);
                    break;
            }
        }
        b.Append('"');
    }

    static void SkipSpace(ReadOnlySpan<char> t, ref int at)
    {
        while (at < t.Length && char.IsWhiteSpace(t[at])) at++;
    }

    static JsonValue ParseValue(ReadOnlySpan<char> t, ref int at)
    {
        SkipSpace(t, ref at);
        if (at >= t.Length) throw new FormatException("unexpected end of JSON");
        switch (t[at])
        {
            case '{':
            {
                at++;
                var members = new List<KeyValuePair<string, JsonValue>>();
                SkipSpace(t, ref at);
                if (t[at] == '}') { at++; return Object(members); }
                for (;;)
                {
                    SkipSpace(t, ref at);
                    var key = ParseString(t, ref at);
                    SkipSpace(t, ref at);
                    Expect(t, ref at, ':');
                    members.Add(new KeyValuePair<string, JsonValue>(key, ParseValue(t, ref at)));
                    SkipSpace(t, ref at);
                    if (t[at] == ',') { at++; continue; }
                    Expect(t, ref at, '}');
                    return Object(members);
                }
            }
            case '[':
            {
                at++;
                var items = new List<JsonValue>();
                SkipSpace(t, ref at);
                if (t[at] == ']') { at++; return Array(items); }
                for (;;)
                {
                    items.Add(ParseValue(t, ref at));
                    SkipSpace(t, ref at);
                    if (t[at] == ',') { at++; continue; }
                    Expect(t, ref at, ']');
                    return Array(items);
                }
            }
            case '"': return Str(ParseString(t, ref at));
            case 't': at += 4; return Bool(true);
            case 'f': at += 5; return Bool(false);
            case 'n': at += 4; return Null;
            default:
            {
                var start = at;
                while (at < t.Length && (char.IsDigit(t[at]) || t[at] is '-' or '+' or '.' or 'e' or 'E')) at++;
                if (start == at) throw new FormatException($"unexpected character '{t[at]}' at {at}");
                return Num(t[start..at].ToString());
            }
        }
    }

    static void Expect(ReadOnlySpan<char> t, ref int at, char c)
    {
        if (at >= t.Length || t[at] != c) throw new FormatException($"expected '{c}' at {at}");
        at++;
    }

    static string ParseString(ReadOnlySpan<char> t, ref int at)
    {
        Expect(t, ref at, '"');
        var b = new StringBuilder();
        while (t[at] != '"')
        {
            var c = t[at++];
            if (c != '\\') { b.Append(c); continue; }
            var e = t[at++];
            switch (e)
            {
                case 'n': b.Append('\n'); break;
                case 'r': b.Append('\r'); break;
                case 't': b.Append('\t'); break;
                case 'b': b.Append('\b'); break;
                case 'f': b.Append('\f'); break;
                case 'u': b.Append((char)int.Parse(t.Slice(at, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture)); at += 4; break;
                default: b.Append(e); break;
            }
        }
        at++;
        return b.ToString();
    }
}
