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
        foreach (var path in new[] { Model, Path.Combine(Golden, "golden.json"), Path.Combine(Golden, "case0/logits.bin") })
            if (!File.Exists(path)) throw new InvalidOperationException($"missing {path}; run tools/fetch.sh and tools/golden.py");
    }

    public static JsonElement GoldenJson()
    {
        Require();
        return JsonDocument.Parse(File.ReadAllText(Path.Combine(Golden, "golden.json"))).RootElement;
    }
}
