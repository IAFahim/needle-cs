using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace Needle.Cli;

internal sealed record EvalPolicy(int Threads, float RefuseBelow, bool AutoDate, int MaxReasoningTokens, int RetrievalTopK)
{
    public static EvalPolicy Defaults(int threads)
    {
        var library = new NeedleOptions();
        return new EvalPolicy(threads, library.RefuseBelow, library.AutoDate, library.MaxReasoningTokens, library.RetrievalTopK);
    }

    public NeedleOptions ToOptions() => new()
    {
        Threads = Threads,
        RefuseBelow = RefuseBelow,
        AutoDate = AutoDate,
        MaxReasoningTokens = MaxReasoningTokens,
        RetrievalTopK = RetrievalTopK,
    };
}

internal sealed record PolicySetting(string Name, Func<EvalPolicy, string> Read, Func<EvalPolicy, string, EvalPolicy> Write);

internal static class EvalPolicies
{
    public static readonly IReadOnlyList<PolicySetting> Settings = new[]
    {
        new PolicySetting("RefuseBelow", p => Format(p.RefuseBelow), (p, v) => p with { RefuseBelow = float.Parse(v, CultureInfo.InvariantCulture) }),
        new PolicySetting("AutoDate", p => p.AutoDate ? "true" : "false", (p, v) => p with { AutoDate = bool.Parse(v) }),
        new PolicySetting("MaxReasoningTokens", p => p.MaxReasoningTokens.ToString(CultureInfo.InvariantCulture), (p, v) => p with { MaxReasoningTokens = int.Parse(v, CultureInfo.InvariantCulture) }),
        new PolicySetting("RetrievalTopK", p => p.RetrievalTopK.ToString(CultureInfo.InvariantCulture), (p, v) => p with { RetrievalTopK = int.Parse(v, CultureInfo.InvariantCulture) }),
    };

    public static EvalPolicy Apply(EvalPolicy start, IEnumerable<string> assignments)
    {
        var policy = start;
        foreach (var assignment in assignments)
        {
            var separator = assignment.IndexOf('=');
            if (separator <= 0) throw new ArgumentException($"policy '{assignment}' must look like name=value");
            var name = assignment[..separator];
            var setting = Settings.FirstOrDefault(s => string.Equals(s.Name, name, StringComparison.OrdinalIgnoreCase))
                ?? throw new ArgumentException($"unknown policy '{name}'; available: {string.Join(", ", Settings.Select(s => s.Name))}");
            try { policy = setting.Write(policy, assignment[(separator + 1)..]); }
            catch (FormatException) { throw new ArgumentException($"policy {setting.Name} cannot parse '{assignment[(separator + 1)..]}'"); }
        }
        return policy;
    }

    public static string Describe(EvalPolicy policy, EvalPolicy defaults)
    {
        var changed = Settings.Where(s => s.Read(policy) != s.Read(defaults)).Select(s => $"{s.Name}={s.Read(policy)}").ToArray();
        return changed.Length == 0 ? "defaults" : string.Join(" ", changed);
    }

    static string Format(float value) => value.ToString("0.###", CultureInfo.InvariantCulture);
}
