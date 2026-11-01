using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Needle.Cli;

internal sealed record EvalCase(string Id, string ToolSet, string Query, ExpectedCall[] Expected, string[] Tags)
{
    public bool InDomain => Expected.Length > 0;

    public string DescribeExpected() => InDomain ? string.Join(" ; ", Expected.Select(call => call.Describe())) : "(no call)";
}

internal sealed record EvalDataset(IReadOnlyDictionary<string, string> ToolSets, IReadOnlyList<EvalCase> Cases)
{
    public static EvalDataset Load(string path, string? tag = null, int limit = 0)
    {
        var files = Directory.Exists(path)
            ? Directory.GetFiles(path, "*.jsonl").OrderBy(file => file, StringComparer.Ordinal).ToArray()
            : new[] { path };
        if (files.Length == 0) throw new FileNotFoundException($"no .jsonl dataset files in {path}");
        var toolsDirectory = Path.Combine(Directory.Exists(path) ? path : Path.GetDirectoryName(Path.GetFullPath(path))!, "tools");
        var cases = files.SelectMany(ReadCases).Where(item => tag == null || item.Tags.Contains(tag)).ToList();
        var toolSets = new Dictionary<string, string>();
        foreach (var id in cases.Select(item => item.ToolSet).Distinct())
        {
            var toolsPath = Path.Combine(toolsDirectory, id + ".json");
            if (!File.Exists(toolsPath)) throw new FileNotFoundException($"tool set '{id}' needs {toolsPath}");
            toolSets[id] = File.ReadAllText(toolsPath);
        }
        return new EvalDataset(toolSets, Limit(cases, limit));
    }

    static IReadOnlyList<EvalCase> Limit(List<EvalCase> cases, int limit)
    {
        if (limit <= 0 || limit >= cases.Count) return cases;
        var order = cases.Select(item => item.ToolSet).Distinct().ToList();
        return cases
            .GroupBy(item => item.ToolSet)
            .SelectMany(group => group.Select((item, index) => (item, index)))
            .OrderBy(entry => entry.index)
            .ThenBy(entry => order.IndexOf(entry.item.ToolSet))
            .Take(limit)
            .Select(entry => entry.item)
            .ToList();
    }

    static IEnumerable<EvalCase> ReadCases(string file)
    {
        var stem = Path.GetFileNameWithoutExtension(file);
        var lineNumber = 0;
        foreach (var line in File.ReadLines(file))
        {
            lineNumber++;
            if (line.Trim().Length == 0 || line.TrimStart().StartsWith('#')) continue;
            EvalCase parsed;
            try { parsed = ParseCase($"{stem}:{lineNumber}", line); }
            catch (Exception e) when (e is JsonException or FormatException or KeyNotFoundException or InvalidOperationException)
            {
                throw new FormatException($"{file} line {lineNumber}: {e.Message}", e);
            }
            yield return parsed;
        }
    }

    static EvalCase ParseCase(string id, string line)
    {
        using var document = JsonDocument.Parse(line);
        var root = document.RootElement;
        var expected = root.GetProperty("expected").EnumerateArray().Select(ParseCall).ToArray();
        var tags = root.TryGetProperty("tags", out var tagList) ? tagList.EnumerateArray().Select(tag => tag.GetString() ?? "").ToArray() : Array.Empty<string>();
        return new EvalCase(id, root.GetProperty("tools").GetString()!, root.GetProperty("query").GetString()!, expected, tags);
    }

    static ExpectedCall ParseCall(JsonElement call)
    {
        var arguments = call.TryGetProperty("args", out var args)
            ? args.EnumerateObject().Select(property => KeyValuePair.Create(property.Name, ArgumentMatcher.Parse(property.Value))).ToArray()
            : Array.Empty<KeyValuePair<string, ArgumentMatcher>>();
        return new ExpectedCall(call.GetProperty("name").GetString()!, arguments);
    }
}
