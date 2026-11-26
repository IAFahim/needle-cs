using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Needle;

internal enum ParamKind : byte { Other, String, Integer, Number, Boolean }

internal enum Polar : byte { None, Power, Lock, Open, Mute, Level, Run }

internal enum ParamRole : byte { None, FirstName, LastName, Origin, Destination }

internal sealed class RepairParam
{
    public byte[] Key = Array.Empty<byte>();
    public byte[] KeyWords = Array.Empty<byte>();
    public ParamKind Kind;
    public bool Required;
    public byte[]? DefaultJson;
    public byte[][] Options = Array.Empty<byte[]>();
    public byte[][] OptionWords = Array.Empty<byte[]>();
    public Polar[] OptionFamily = Array.Empty<Polar>();
    public sbyte[] OptionDirection = Array.Empty<sbyte>();
    public Polar OptionsPolar;
    public Polar BoolPolar;
    public double Minimum = double.NegativeInfinity;
    public double Maximum = double.PositiveInfinity;
    public bool MinimumExclusive;
    public bool MaximumExclusive;
    public bool Date;
    public bool Phone;
    public ParamRole Role;
}

internal sealed class RepairTool
{
    public byte[] Name = Array.Empty<byte>();
    public byte[] EscapedName = Array.Empty<byte>();
    public byte[][] NameWords = Array.Empty<byte[]>();
    public RepairParam[] Params = Array.Empty<RepairParam>();
    public Polar Family;
    public sbyte Direction;
    public byte[] Stem = Array.Empty<byte>();
    public int Paired = -1;
    public int SingleNumber = -1;
    public int FirstName = -1;
    public int LastName = -1;
    public int Origin = -1;
    public int Destination = -1;
    public int DateParams;
    public string[] Triggers = Array.Empty<string>();

    public int FindParam(ReadOnlySpan<byte> key)
    {
        for (var i = 0; i < Params.Length; i++)
            if (key.SequenceEqual(Params[i].Key)) return i;
        return -1;
    }
}

internal sealed class RepairPlan
{
    public readonly RepairTool[] Tools;
    public readonly int LongestValue;
    public readonly int TotalParams;
    public readonly bool HasTriggers;

    static readonly (string Word, Polar Family, sbyte Direction)[] Vocabulary =
    {
        ("on", Polar.Power, 1), ("enable", Polar.Power, 1), ("enabled", Polar.Power, 1), ("activate", Polar.Power, 1), ("power on", Polar.Power, 1),
        ("off", Polar.Power, -1), ("disable", Polar.Power, -1), ("disabled", Polar.Power, -1), ("deactivate", Polar.Power, -1),
        ("lock", Polar.Lock, 1), ("locked", Polar.Lock, 1), ("unlock", Polar.Lock, -1), ("unlocked", Polar.Lock, -1),
        ("open", Polar.Open, 1), ("opened", Polar.Open, 1), ("close", Polar.Open, -1), ("closed", Polar.Open, -1), ("shut", Polar.Open, -1),
        ("mute", Polar.Mute, 1), ("muted", Polar.Mute, 1), ("silence", Polar.Mute, 1), ("unmute", Polar.Mute, -1), ("unmuted", Polar.Mute, -1),
        ("up", Polar.Level, 1), ("louder", Polar.Level, 1), ("raise", Polar.Level, 1), ("increase", Polar.Level, 1), ("higher", Polar.Level, 1), ("brighten", Polar.Level, 1), ("boost", Polar.Level, 1),
        ("down", Polar.Level, -1), ("quieter", Polar.Level, -1), ("lower", Polar.Level, -1), ("decrease", Polar.Level, -1), ("reduce", Polar.Level, -1), ("softer", Polar.Level, -1),
        ("start", Polar.Run, 1), ("resume", Polar.Run, 1), ("play", Polar.Run, 1), ("begin", Polar.Run, 1),
        ("stop", Polar.Run, -1), ("pause", Polar.Run, -1), ("halt", Polar.Run, -1),
    };

    public static readonly byte[][] CueWords;
    public static readonly Polar[] CueFamily;
    public static readonly sbyte[] CueDirection;

    static RepairPlan()
    {
        CueWords = new byte[Vocabulary.Length][];
        CueFamily = new Polar[Vocabulary.Length];
        CueDirection = new sbyte[Vocabulary.Length];
        for (var i = 0; i < Vocabulary.Length; i++)
        {
            CueWords[i] = Encoding.UTF8.GetBytes(Vocabulary[i].Word);
            CueFamily[i] = Vocabulary[i].Family;
            CueDirection[i] = Vocabulary[i].Direction;
        }
    }

    public RepairPlan(ToolSet tools, string rawToolsJson)
    {
        var raw = RawTriggers(rawToolsJson);
        Tools = new RepairTool[tools.Tools.Count];
        var longest = 16;
        for (var t = 0; t < Tools.Length; t++)
        {
            var schema = tools.Tools[t];
            var alias = schema.GetString("name") ?? "";
            var original = tools.OriginalName(alias);
            var tool = new RepairTool { Name = Encoding.UTF8.GetBytes(original), EscapedName = EscapedUtf8(original), NameWords = SplitWords(original) };
            if (raw.TryGetValue(original, out var triggers)) tool.Triggers = triggers;
            var parameters = schema.Get("parameters");
            var properties = parameters?.Get("properties");
            var required = new HashSet<string>();
            if (parameters?.Get("required") is { Kind: JsonKind.Array } list)
                foreach (var item in list.Items) required.Add(item.Text);
            var built = new List<RepairParam>();
            if (properties is { Kind: JsonKind.Object })
                foreach (var member in properties.Members)
                {
                    var param = BuildParam(tools.OriginalProperty(alias, member.Key), member.Value, required.Contains(member.Key));
                    foreach (var option in param.Options) longest = Math.Max(longest, option.Length);
                    if (param.DefaultJson != null) longest = Math.Max(longest, param.DefaultJson.Length);
                    built.Add(param);
                }
            tool.Params = built.ToArray();
            Classify(tool, original);
            TotalParams += tool.Params.Length;
            Tools[t] = tool;
        }
        LinkPairs();
        LongestValue = longest;
        foreach (var tool in Tools) HasTriggers |= tool.Triggers.Length > 0;
    }

    static Dictionary<string, string[]> RawTriggers(string toolsJson)
    {
        var map = new Dictionary<string, string[]>();
        JsonValue parsed;
        try { parsed = JsonValue.Parse(toolsJson); }
        catch (FormatException) { return map; }
        foreach (var entry in parsed.Kind == JsonKind.Array ? parsed.Items : new List<JsonValue> { parsed })
        {
            var tool = entry.Get("function") is { Kind: JsonKind.Object } wrapped ? wrapped : entry;
            var name = tool.GetString("name");
            var triggers = tool.Get("triggers") ?? entry.Get("triggers");
            if (name == null || triggers is not { Kind: JsonKind.Array }) continue;
            var patterns = new List<string>();
            foreach (var item in triggers.Items)
                if (item.Kind == JsonKind.String && item.Text.Length > 0) patterns.Add(item.Text);
            if (patterns.Count > 0) map[name] = patterns.ToArray();
        }
        return map;
    }

    static RepairParam BuildParam(string name, JsonValue schema, bool required)
    {
        var resolved = Concrete(schema);
        var param = new RepairParam
        {
            Key = EscapedUtf8(name),
            KeyWords = NormalizedBytes(name),
            Required = required,
            Kind = KindOf(resolved),
        };
        if ((schema.Get("default") ?? resolved.Get("default")) is { } fallback && fallback.Kind != JsonKind.Null)
            param.DefaultJson = Encoding.UTF8.GetBytes(fallback.ToString());
        if (resolved.Get("enum") is { Kind: JsonKind.Array } options)
        {
            var values = new List<byte[]>();
            var words = new List<byte[]>();
            foreach (var option in options.Items)
            {
                if (option.Kind == JsonKind.Null) continue;
                values.Add(Encoding.UTF8.GetBytes(option.ToString()));
                words.Add(option.Kind == JsonKind.String ? NormalizedBytes(option.Text) : NormalizedBytes(option.ToString()));
            }
            param.Options = values.ToArray();
            param.OptionWords = words.ToArray();
            if (param.Kind == ParamKind.Other) param.Kind = ParamKind.String;
            ClassifyOptions(param);
        }
        if (Number(resolved, "minimum") is { } min) param.Minimum = min;
        if (Number(resolved, "maximum") is { } max) param.Maximum = max;
        if (Number(resolved, "exclusiveMinimum") is { } xmin) { param.Minimum = xmin; param.MinimumExclusive = true; }
        if (Number(resolved, "exclusiveMaximum") is { } xmax) { param.Maximum = xmax; param.MaximumExclusive = true; }
        var lower = name.ToLowerInvariant();
        var format = resolved.GetString("format");
        param.Date = param.Kind == ParamKind.String && param.Options.Length == 0 &&
            (format is "date" or "date-time" || lower == "date" || lower == "day" || lower.EndsWith("_date") || lower.StartsWith("date_") || lower.EndsWith("date"));
        param.Phone = param.Kind == ParamKind.String && param.Options.Length == 0 &&
            (lower.Contains("phone") || lower == "tel" || lower.Contains("mobile") || lower == "number" || lower.EndsWith("_number") && lower.Contains("contact"));
        if (param.Kind == ParamKind.Boolean) param.BoolPolar = BoolFamily(lower);
        param.Role = param.Kind != ParamKind.String || param.Options.Length > 0 ? ParamRole.None : lower switch
        {
            "first_name" or "firstname" or "first" or "given_name" or "givenname" => ParamRole.FirstName,
            "last_name" or "lastname" or "last" or "family_name" or "surname" => ParamRole.LastName,
            "pickup" or "pickup_location" or "origin" or "from" or "source" or "start_location" or "departure" or "from_location" or "start" => ParamRole.Origin,
            "destination" or "dropoff" or "dropoff_location" or "to" or "to_location" or "end_location" or "arrival" or "target_location" or "end" => ParamRole.Destination,
            _ => ParamRole.None,
        };
        return param;
    }

    static JsonValue Concrete(JsonValue schema)
    {
        foreach (var key in new[] { "anyOf", "oneOf" })
            if (schema.Get(key) is { Kind: JsonKind.Array } variants)
            {
                JsonValue? only = null;
                var count = 0;
                foreach (var variant in variants.Items)
                {
                    if (variant.GetString("type") == "null") continue;
                    only = variant;
                    count++;
                }
                if (count == 1 && only != null) return only;
            }
        return schema;
    }

    static ParamKind KindOf(JsonValue schema)
    {
        var type = schema.Get("type");
        var name = type?.Kind == JsonKind.String ? type.Text : null;
        if (type is { Kind: JsonKind.Array })
            foreach (var item in type.Items)
                if (item.Text != "null") { name = item.Text; break; }
        return name switch
        {
            "string" => ParamKind.String,
            "integer" => ParamKind.Integer,
            "number" => ParamKind.Number,
            "boolean" => ParamKind.Boolean,
            _ => ParamKind.Other,
        };
    }

    static double? Number(JsonValue schema, string key) =>
        schema.Get(key) is { Kind: JsonKind.Number } value && double.TryParse(value.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) ? parsed : null;

    static Polar BoolFamily(string name)
    {
        if (name.Contains("lock")) return Polar.Lock;
        if (name.Contains("mute")) return Polar.Mute;
        if (name.Contains("open")) return Polar.Open;
        if (name is "on" or "power" or "powered" or "enabled" or "enable" or "active" or "state" or "is_on" or "turn_on" or "switch" || name.EndsWith("_on") || name.EndsWith("enabled")) return Polar.Power;
        return Polar.None;
    }

    static void ClassifyOptions(RepairParam param)
    {
        var count = param.OptionWords.Length;
        param.OptionFamily = new Polar[count];
        param.OptionDirection = new sbyte[count];
        Span<int> positive = stackalloc int[8];
        Span<int> negative = stackalloc int[8];
        for (var i = 0; i < count; i++)
            for (var v = 0; v < CueWords.Length; v++)
                if (param.OptionWords[i].AsSpan().SequenceEqual(CueWords[v]))
                {
                    param.OptionFamily[i] = CueFamily[v];
                    param.OptionDirection[i] = CueDirection[v];
                    if (CueDirection[v] > 0) positive[(int)CueFamily[v]]++;
                    else negative[(int)CueFamily[v]]++;
                    break;
                }
        for (var f = 1; f < 7; f++)
            if (positive[f] == 1 && negative[f] == 1) { param.OptionsPolar = (Polar)f; break; }
    }

    static void Classify(RepairTool tool, string name)
    {
        var numbers = 0;
        for (var i = 0; i < tool.Params.Length; i++)
        {
            var p = tool.Params[i];
            if (p.Kind is ParamKind.Integer or ParamKind.Number && p.Options.Length == 0) { numbers++; tool.SingleNumber = i; }
            if (p.Date) tool.DateParams++;
            switch (p.Role)
            {
                case ParamRole.FirstName: tool.FirstName = i; break;
                case ParamRole.LastName: tool.LastName = i; break;
                case ParamRole.Origin: tool.Origin = i; break;
                case ParamRole.Destination: tool.Destination = i; break;
            }
        }
        if (numbers != 1) tool.SingleNumber = -1;
        if (tool.FirstName < 0 || tool.LastName < 0) tool.FirstName = tool.LastName = -1;
        if (tool.Origin < 0 || tool.Destination < 0) tool.Origin = tool.Destination = -1;
        var lower = ToolSet.SnakeCase(name);
        foreach (var (prefix, family, direction) in new[]
        {
            ("turn_on_", Polar.Power, (sbyte)1), ("turn_off_", Polar.Power, (sbyte)-1), ("switch_on_", Polar.Power, (sbyte)1), ("switch_off_", Polar.Power, (sbyte)-1),
            ("enable_", Polar.Power, (sbyte)1), ("disable_", Polar.Power, (sbyte)-1), ("activate_", Polar.Power, (sbyte)1), ("deactivate_", Polar.Power, (sbyte)-1),
            ("unlock_", Polar.Lock, (sbyte)-1), ("lock_", Polar.Lock, (sbyte)1), ("open_", Polar.Open, (sbyte)1), ("close_", Polar.Open, (sbyte)-1),
            ("unmute_", Polar.Mute, (sbyte)-1), ("mute_", Polar.Mute, (sbyte)1), ("start_", Polar.Run, (sbyte)1), ("stop_", Polar.Run, (sbyte)-1),
            ("resume_", Polar.Run, (sbyte)1), ("pause_", Polar.Run, (sbyte)-1), ("increase_", Polar.Level, (sbyte)1), ("decrease_", Polar.Level, (sbyte)-1),
            ("raise_", Polar.Level, (sbyte)1), ("lower_", Polar.Level, (sbyte)-1),
        })
        {
            if (lower.StartsWith(prefix, StringComparison.Ordinal))
            {
                tool.Family = family;
                tool.Direction = direction;
                tool.Stem = Encoding.UTF8.GetBytes(lower.Substring(prefix.Length));
                return;
            }
            var suffix = "_" + prefix.TrimEnd('_');
            if (lower.EndsWith(suffix, StringComparison.Ordinal))
            {
                tool.Family = family;
                tool.Direction = direction;
                tool.Stem = Encoding.UTF8.GetBytes(lower.Substring(0, lower.Length - suffix.Length));
                return;
            }
        }
        foreach (var (whole, family, direction) in new[] { ("lock", Polar.Lock, (sbyte)1), ("unlock", Polar.Lock, (sbyte)-1), ("mute", Polar.Mute, (sbyte)1), ("unmute", Polar.Mute, (sbyte)-1) })
            if (lower == whole)
            {
                tool.Family = family;
                tool.Direction = direction;
                tool.Stem = Array.Empty<byte>();
            }
    }

    void LinkPairs()
    {
        for (var a = 0; a < Tools.Length; a++)
        {
            if (Tools[a].Family == Polar.None) continue;
            for (var b = 0; b < Tools.Length; b++)
                if (b != a && Tools[b].Family == Tools[a].Family && Tools[b].Direction == -Tools[a].Direction &&
                    Tools[b].Stem.AsSpan().SequenceEqual(Tools[a].Stem) && SameKeys(Tools[a], Tools[b]))
                {
                    Tools[a].Paired = b;
                    break;
                }
        }
    }

    static bool SameKeys(RepairTool a, RepairTool b)
    {
        foreach (var p in b.Params)
            if (p.Required && a.FindParam(p.Key) < 0) return false;
        return true;
    }

    static byte[][] SplitWords(string name)
    {
        var words = new List<byte[]>();
        foreach (var part in ToolSet.SnakeCase(name).Split('_'))
            if (part.Length >= 3 && !IsGeneric(part)) words.Add(Encoding.UTF8.GetBytes(part));
        return words.ToArray();
    }

    static bool IsGeneric(string word) => word is "set" or "get" or "the" or "and" or "for" or "add" or "new" or "make" or "send" or "control" or "tool" or "run" or "use" or "update" or "create" or "change" or "start" or "stop";

    public static byte[] NormalizedBytes(string text)
    {
        var source = Encoding.UTF8.GetBytes(text);
        var buffer = new byte[source.Length + 1];
        var length = RepairText.Normalize(source, buffer);
        return buffer.AsSpan(0, length).ToArray();
    }

    static byte[] EscapedUtf8(string text)
    {
        var builder = new StringBuilder();
        JsonValue.WriteString(builder, text);
        return Encoding.UTF8.GetBytes(builder.ToString(1, builder.Length - 2));
    }
}
