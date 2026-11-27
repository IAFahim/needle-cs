using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using Xunit;

namespace Needle.Tests.Bughunt;

public sealed class CiBughuntTests
{
    static string Workflow => File.ReadAllText(Path.Combine(Assets.Root, ".github/workflows/ci.yml"));

    static Dictionary<string, string[]> ShardPatterns()
    {
        var shards = new Dictionary<string, string[]>();
        foreach (Match match in Regex.Matches(Workflow, @"(\w+)\) echo 'FILTER=([^']*)'"))
            shards[match.Groups[1].Value] = match.Groups[2].Value.Split('|').Select(p => p.Substring(p.IndexOf('~') + 1)).ToArray();
        return shards;
    }

    static IEnumerable<string> TestNames() =>
        typeof(CiBughuntTests).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract)
            .SelectMany(t => t.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                .Where(m => m.GetCustomAttributes<FactAttribute>(true).Any())
                .Select(m => t.FullName + "." + m.Name));

    [Fact]
    public void EveryTestIsSelectedByACiShard()
    {
        var shards = ShardPatterns();
        Assert.Equal(new[] { "core", "heavy", "unit" }, shards.Keys.OrderBy(k => k));
        var orphans = TestNames().Where(name => !shards.Values.Any(patterns => patterns.Any(p => name.Contains(p, StringComparison.OrdinalIgnoreCase))))
            .Select(name => name.Substring(0, name.LastIndexOf('.'))).Distinct().OrderBy(n => n).ToArray();
        Assert.True(orphans.Length == 0, "test classes no CI shard runs: " + string.Join(", ", orphans));
    }

    [Fact]
    public void FixtureCacheKeyHashesTheRefLock()
    {
        var key = Regex.Match(Workflow, @"key: ref-[^\n]*").Value;
        Assert.Contains("hashFiles(", key);
        Assert.Contains("'tools/ref.lock'", key);
    }

    [Fact]
    public void SimdGatedTestsCompileIntoTheTestAssembly()
    {
        var assembly = typeof(CiBughuntTests).Assembly;
        Assert.NotNull(assembly.GetType("Needle.Tests.Avx512OpsTests"));
        Assert.NotNull(assembly.GetType("Needle.Tests.HadamardMlpTests")!.GetMethod("WideKroneckerProductMatchesAvx2BitForBit"));
    }
}
