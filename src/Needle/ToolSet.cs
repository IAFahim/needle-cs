using System;
using System.Collections.Generic;
using System.Text;

namespace Needle;

public sealed class ToolSet
{
    static readonly string[] KeptKeys = { "name", "description", "parameters" };

    public IReadOnlyList<JsonValue> Tools { get; }
    public IReadOnlyDictionary<string, string> NameAliases { get; }
    readonly Dictionary<string, Dictionary<string, string>> propertyAliases;

    public ToolSet(string toolsJson)
    {
        var parsed = JsonValue.Parse(toolsJson);
        var list = new List<JsonValue>();
        var names = new Dictionary<string, string>();
        var properties = new Dictionary<string, Dictionary<string, string>>();
        var taken = new HashSet<string>();
        foreach (var entry in parsed.Kind == JsonKind.Array ? parsed.Items : new List<JsonValue> { parsed })
        {
            var raw = entry.Get("function") is { Kind: JsonKind.Object } wrapped ? wrapped : entry;
            var original = raw.GetString("name") ?? throw new FormatException("a tool needs a name");
            var alias = Unique(SnakeCase(original), taken);
            names[alias] = original;
            var aliases = new Dictionary<string, string>();
            properties[alias] = aliases;
            var members = new List<KeyValuePair<string, JsonValue>>();
            foreach (var key in KeptKeys)
            {
                var value = raw.Get(key);
                if (value == null) continue;
                members.Add(new KeyValuePair<string, JsonValue>(key, key == "name" ? JsonValue.Str(alias) : key == "parameters" ? NormalizeSchema(value, aliases) : value));
            }
            list.Add(JsonValue.Object(members));
        }
        Tools = list;
        NameAliases = names;
        propertyAliases = properties;
    }

    ToolSet(List<JsonValue> tools, IReadOnlyDictionary<string, string> names, Dictionary<string, Dictionary<string, string>> properties)
    {
        Tools = tools;
        NameAliases = names;
        propertyAliases = properties;
    }

    public ToolSet Subset(IReadOnlyList<int> indices)
    {
        var list = new List<JsonValue>();
        foreach (var index in indices) list.Add(Tools[index]);
        return new ToolSet(list, NameAliases, propertyAliases);
    }

    public string Render() => JsonValue.Array(Tools).ToString();

    public string OriginalName(string alias) => NameAliases.TryGetValue(alias, out var original) ? original : alias;

    public string OriginalProperty(string toolAlias, string alias) =>
        propertyAliases.TryGetValue(toolAlias, out var map) && map.TryGetValue(alias, out var original) ? original : alias;

    static JsonValue NormalizeSchema(JsonValue schema, Dictionary<string, string> aliases)
    {
        if (schema.Kind == JsonKind.Array)
        {
            var items = new List<JsonValue>();
            foreach (var item in schema.Items) items.Add(NormalizeSchema(item, aliases));
            return JsonValue.Array(items);
        }
        if (schema.Kind != JsonKind.Object) return schema;
        var members = new List<KeyValuePair<string, JsonValue>>();
        var taken = new HashSet<string>(aliases.Keys);
        foreach (var member in schema.Members)
        {
            if (member.Key == "properties" && member.Value.Kind == JsonKind.Object)
            {
                var renamed = new List<KeyValuePair<string, JsonValue>>();
                foreach (var property in member.Value.Members)
                {
                    var alias = Unique(SnakeCase(property.Key), taken);
                    aliases[alias] = property.Key;
                    renamed.Add(new KeyValuePair<string, JsonValue>(alias, NormalizeSchema(property.Value, aliases)));
                }
                members.Add(new KeyValuePair<string, JsonValue>("properties", JsonValue.Object(renamed)));
            }
            else if (member.Key == "required" && member.Value.Kind == JsonKind.Array)
            {
                var required = new List<JsonValue>();
                foreach (var name in member.Value.Items) required.Add(JsonValue.Str(AliasFor(aliases, name.Text)));
                members.Add(new KeyValuePair<string, JsonValue>("required", JsonValue.Array(required)));
            }
            else members.Add(new KeyValuePair<string, JsonValue>(member.Key, member.Value.Kind is JsonKind.Object or JsonKind.Array ? NormalizeSchema(member.Value, aliases) : member.Value));
        }
        return JsonValue.Object(members);
    }

    static string AliasFor(Dictionary<string, string> aliases, string original)
    {
        foreach (var pair in aliases) if (pair.Value == original) return pair.Key;
        return SnakeCase(original);
    }

    static string Unique(string name, HashSet<string> taken)
    {
        var candidate = name;
        for (var n = 2; !taken.Add(candidate); n++) candidate = name + "_" + n;
        return candidate;
    }

    public static string SnakeCase(string name)
    {
        var b = new StringBuilder();
        for (var i = 0; i < name.Length; i++)
        {
            var c = name[i];
            if (char.IsUpper(c))
            {
                var boundary = i > 0 && (char.IsLower(name[i - 1]) || char.IsDigit(name[i - 1]) || (char.IsUpper(name[i - 1]) && i + 1 < name.Length && char.IsLower(name[i + 1])));
                if (boundary && b.Length > 0 && b[^1] != '_') b.Append('_');
                b.Append(char.ToLowerInvariant(c));
            }
            else if (char.IsLetterOrDigit(c) || c == '_') b.Append(c);
            else if (b.Length > 0 && b[^1] != '_') b.Append('_');
        }
        return b.ToString().Trim('_') is { Length: > 0 } result ? result : "tool";
    }
}
