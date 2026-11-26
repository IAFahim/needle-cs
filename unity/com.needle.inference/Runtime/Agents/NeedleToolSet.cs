using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;

namespace Needle
{
    /// <summary>The kinds of value a tool parameter can take.</summary>
    public enum NeedleParameterType
    {
        Text,
        Integer,
        Number,
        Flag,
        Choice,
    }

    /// <summary>One parameter of a tool, as a designer fills it in.</summary>
    [Serializable]
    public sealed class NeedleToolParameter
    {
        [Tooltip("Name the model fills in, in snake_case, for example minutes or target_name.")]
        public string Name = "value";

        [Tooltip("What kind of value it is. Choice limits it to the options below.")]
        public NeedleParameterType Type = NeedleParameterType.Text;

        [Tooltip("What the value means, in a few words. The model reads this, so a clear description raises its confidence.")]
        public string Description = string.Empty;

        [Tooltip("The model must always fill it in.")]
        public bool Required = true;

        [Tooltip("For Choice: the allowed values, one per entry.")]
        public string[] Options = new string[0];
    }

    /// <summary>One thing the model can do, as a designer fills it in.</summary>
    [Serializable]
    public sealed class NeedleToolDefinition
    {
        [Tooltip("Name of the action, in snake_case, for example move_to or attack.")]
        public string Name = "new_tool";

        [Tooltip("What the action does, in one sentence. The model reads this to decide when to use it.")]
        [TextArea(1, 4)]
        public string Description = string.Empty;

        public List<NeedleToolParameter> Parameters = new List<NeedleToolParameter>();
    }

    /// <summary>
    /// The actions an agent can take. Designers list tools and parameters in the Inspector, or point
    /// at a tools JSON file; either way the agent reads <see cref="ToJson"/>.
    /// </summary>
    [CreateAssetMenu(menuName = "Needle/Tool Set", fileName = "Tools", order = 101)]
    public sealed class NeedleToolSet : ScriptableObject
    {
        [Tooltip("Optional: a tools JSON file in the OpenAI function format. When set, it is used instead of the list below.")]
        public TextAsset Json;

        public List<NeedleToolDefinition> Tools = new List<NeedleToolDefinition>();

        /// <summary>The tool names, in order, from the JSON file or the list.</summary>
        public string[] ToolNames()
        {
            if (Json != null) return NamesFromJson(Json.text);
            var names = new string[Tools.Count];
            for (int i = 0; i < Tools.Count; i++) names[i] = Tools[i].Name;
            return names;
        }

        /// <summary>The parameter names of one tool from the list; empty for a JSON file or an unknown tool.</summary>
        public string[] ParameterNames(string tool)
        {
            if (Json == null)
            {
                for (int i = 0; i < Tools.Count; i++)
                {
                    if (Tools[i].Name != tool) continue;
                    var names = new string[Tools[i].Parameters.Count];
                    for (int p = 0; p < names.Length; p++) names[p] = Tools[i].Parameters[p].Name;
                    return names;
                }
            }
            return new string[0];
        }

        /// <summary>The tools as JSON, the way the model expects them.</summary>
        public string ToJson()
        {
            if (Json != null) return Json.text;
            var b = new StringBuilder(256);
            b.Append('[');
            for (int i = 0; i < Tools.Count; i++)
            {
                var tool = Tools[i];
                if (i > 0) b.Append(',');
                b.Append("{\"name\":");
                Quote(b, tool.Name);
                b.Append(",\"description\":");
                Quote(b, tool.Description);
                b.Append(",\"parameters\":{\"type\":\"object\",\"properties\":{");
                var required = new List<string>();
                for (int p = 0; p < tool.Parameters.Count; p++)
                {
                    var parameter = tool.Parameters[p];
                    if (p > 0) b.Append(',');
                    Quote(b, parameter.Name);
                    b.Append(":{\"type\":");
                    Quote(b, JsonType(parameter.Type));
                    if (!string.IsNullOrEmpty(parameter.Description))
                    {
                        b.Append(",\"description\":");
                        Quote(b, parameter.Description);
                    }
                    if (parameter.Type == NeedleParameterType.Choice && parameter.Options != null && parameter.Options.Length > 0)
                    {
                        b.Append(",\"enum\":[");
                        for (int o = 0; o < parameter.Options.Length; o++)
                        {
                            if (o > 0) b.Append(',');
                            Quote(b, parameter.Options[o]);
                        }
                        b.Append(']');
                    }
                    b.Append('}');
                    if (parameter.Required) required.Add(parameter.Name);
                }
                b.Append("},\"required\":[");
                for (int r = 0; r < required.Count; r++)
                {
                    if (r > 0) b.Append(',');
                    Quote(b, required[r]);
                }
                b.Append("]}}");
            }
            b.Append(']');
            return b.ToString();
        }

        /// <summary>Problems a designer should fix, one line each; empty when the set is usable.</summary>
        public List<string> Validate()
        {
            var problems = new List<string>();
            if (Json != null)
            {
                if (NamesFromJson(Json.text).Length == 0) problems.Add("The JSON file has no tools with a \"name\".");
                return problems;
            }
            if (Tools.Count == 0) problems.Add("Add at least one tool.");
            var seen = new HashSet<string>();
            for (int i = 0; i < Tools.Count; i++)
            {
                var tool = Tools[i];
                if (!IsIdentifier(tool.Name)) problems.Add("Tool " + (i + 1) + ": the name '" + tool.Name + "' should be snake_case letters, digits and underscores.");
                else if (!seen.Add(tool.Name)) problems.Add("Two tools are called '" + tool.Name + "'.");
                if (string.IsNullOrWhiteSpace(tool.Description)) problems.Add("Tool '" + tool.Name + "' has no description; the model needs one to choose it.");
                var parameters = new HashSet<string>();
                foreach (var parameter in tool.Parameters)
                {
                    if (!IsIdentifier(parameter.Name)) problems.Add("Tool '" + tool.Name + "': the parameter name '" + parameter.Name + "' should be snake_case.");
                    else if (!parameters.Add(parameter.Name)) problems.Add("Tool '" + tool.Name + "' has two parameters called '" + parameter.Name + "'.");
                    if (parameter.Type == NeedleParameterType.Choice && (parameter.Options == null || parameter.Options.Length == 0))
                        problems.Add("Tool '" + tool.Name + "': the choice '" + parameter.Name + "' has no options.");
                }
            }
            return problems;
        }

        static string JsonType(NeedleParameterType type)
        {
            switch (type)
            {
                case NeedleParameterType.Integer: return "integer";
                case NeedleParameterType.Number: return "number";
                case NeedleParameterType.Flag: return "boolean";
                default: return "string";
            }
        }

        static bool IsIdentifier(string name)
        {
            if (string.IsNullOrEmpty(name) || char.IsDigit(name[0])) return false;
            foreach (char c in name)
                if (!(c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) return false;
            return true;
        }

        static void Quote(StringBuilder b, string text)
        {
            b.Append('"');
            foreach (char c in text ?? string.Empty)
            {
                switch (c)
                {
                    case '"': b.Append("\\\""); break;
                    case '\\': b.Append("\\\\"); break;
                    case '\n': b.Append("\\n"); break;
                    case '\r': b.Append("\\r"); break;
                    case '\t': b.Append("\\t"); break;
                    default:
                        if (c < 0x20) b.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else b.Append(c);
                        break;
                }
            }
            b.Append('"');
        }

        // Reads the top-level "name" of each tool, enough to list them in the Inspector without a JSON library.
        static string[] NamesFromJson(string json)
        {
            var names = new List<string>();
            if (string.IsNullOrEmpty(json)) return names.ToArray();
            int depth = 0;
            for (int i = 0; i < json.Length; i++)
            {
                char c = json[i];
                if (c == '"')
                {
                    int end = SkipString(json, i);
                    if (depth == 2 && end > i && string.CompareOrdinal(json, i, "\"name\"", 0, 6) == 0 && end == i + 5)
                    {
                        int colon = json.IndexOf(':', end + 1);
                        int open = colon < 0 ? -1 : json.IndexOf('"', colon + 1);
                        if (open > 0)
                        {
                            int close = SkipString(json, open);
                            if (close > open) names.Add(json.Substring(open + 1, close - open - 1));
                            i = close;
                            continue;
                        }
                    }
                    i = end;
                }
                else if (c == '{' || c == '[') depth++;
                else if (c == '}' || c == ']') depth--;
            }
            return names.ToArray();
        }

        static int SkipString(string json, int start)
        {
            for (int i = start + 1; i < json.Length; i++)
            {
                if (json[i] == '\\') i++;
                else if (json[i] == '"') return i;
            }
            return json.Length - 1;
        }
    }
}
