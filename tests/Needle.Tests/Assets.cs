using System;
using System.IO;
using System.Text.Json;

namespace Needle.Tests;

internal static class Assets
{
    public static readonly string Root = FindRoot();
    public static string Model => Path.Combine(Root, "ref/hf/needle3.cact");
    public static string Golden => Path.Combine(Root, "ref/golden");
    public static string Tools => Path.Combine(Root, "tools/tools.json");

    static string FindRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
            if (File.Exists(Path.Combine(dir.FullName, "Directory.Build.props"))) return dir.FullName;
        throw new InvalidOperationException("repository root not found");
    }

    public static void Require()
    {
        foreach (var path in new[] { Model, Path.Combine(Golden, "golden.json"), Path.Combine(Golden, "case0/logits.bin"), Path.Combine(Golden, "case0/confidence.txt"), Path.Combine(Golden, "case0/pooled.bin") })
            if (!File.Exists(path)) throw new InvalidOperationException($"missing {path}; generate it with: tools/fetch.sh && pip install numpy sentencepiece && python3 tools/golden.py && python3 tools/golden_confidence.py (golden.py takes about 15 CPU minutes)");
    }

    public static int GoldenBudget(JsonElement golden) =>
        golden.TryGetProperty("budget", out var budget)
            ? budget.GetInt32()
            : throw new InvalidOperationException("golden.json has no decode budget; regenerate it with: python3 tools/golden.py");

    public static JsonElement GoldenJson()
    {
        Require();
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(Golden, "golden.json"))).RootElement;
    }
}
