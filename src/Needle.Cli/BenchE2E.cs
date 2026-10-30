using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Needle;

namespace Needle.Cli;

internal sealed record BenchE2EOptions
{
    public int[] Threads { get; init; } = { 1, 2, 4 };
    public int Reps { get; init; } = 5;
    public int Warmups { get; init; } = 1;
    public string Sets { get; init; } = "all";
    public bool Json { get; init; }
    public bool Profile { get; init; }
    public int DecodeBudget { get; init; } = 40;
    public long? MaxManagedBytes { get; init; }
    public long? MaxNativeAllocs { get; init; }

    public static BenchE2EOptions Parse(string[] args)
    {
        var options = new BenchE2EOptions();
        for (var i = 0; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value");
            options = args[i] switch
            {
                "--threads" => options with { Threads = Next().Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray() },
                "--reps" => options with { Reps = int.Parse(Next()) },
                "--warmup" => options with { Warmups = int.Parse(Next()) },
                "--sets" => options with { Sets = Next() },
                "--decode-budget" => options with { DecodeBudget = int.Parse(Next()) },
                "--max-managed-bytes" => options with { MaxManagedBytes = long.Parse(Next()) },
                "--max-native-allocs" => options with { MaxNativeAllocs = long.Parse(Next()) },
                "--json" => options with { Json = true },
                "--profile" => options with { Profile = true },
                var unknown => throw new ArgumentException($"unknown option {unknown}"),
            };
        }
        if (options.Threads.Length == 0 || options.Threads.Any(t => t < 1)) throw new ArgumentException("--threads needs positive integers");
        if (options.Reps < 1) throw new ArgumentException("--reps must be at least 1");
        return options;
    }
}

internal sealed class PromptMeasurement
{
    public string Query = "";
    public string Calls = "";
    public int Failures;
    public int PrefillTokens;
    public int DecodeTokens;
    public readonly List<double> PrefillMs = new();
    public readonly List<double> DecodeUsPerToken = new();
    public readonly List<double> TtftMs = new();
    public readonly List<double> TotalMs = new();
    public readonly List<double> PrefillTps = new();
    public readonly List<double> DecodeTps = new();
    public readonly List<long> ManagedBytes = new();
    public readonly List<long> NativeAllocs = new();
}

internal sealed class SetMeasurement
{
    public string Name = "";
    public int ToolCount;
    public int PrefixTokens;
    public double SessionCtorMs;
    public long SessionCtorPrivateBytes;
    public readonly List<PromptMeasurement> Prompts = new();

    public Spread Mean(Func<PromptMeasurement, Spread> select)
    {
        var best = 0.0;
        var median = 0.0;
        foreach (var prompt in Prompts)
        {
            var spread = select(prompt);
            best += spread.Best;
            median += spread.Median;
        }
        return new Spread(best / Prompts.Count, median / Prompts.Count);
    }
}

internal sealed class ThreadMeasurement
{
    public int Threads;
    public readonly List<SetMeasurement> Sets = new();
    public ProfileResult? Profile;
}

internal sealed class E2EReport
{
    public string Model = "";
    public BenchE2EOptions Options = new();
    public readonly List<ThreadMeasurement> Runs = new();
}

internal static class BenchE2E
{
    public static int Run(string modelPath, string toolsPath, string[] args)
    {
        BenchE2EOptions options;
        try { options = BenchE2EOptions.Parse(args); }
        catch (Exception e) when (e is ArgumentException or FormatException or OverflowException)
        {
            Console.Error.WriteLine(e.Message);
            return 2;
        }
        using var model = Model.Open(modelPath);
        var report = Measure(model, File.ReadAllText(toolsPath), options, modelPath, Console.Error);
        Console.Out.WriteLine(options.Json ? ToJson(report) : ToText(report));
        var violations = Violations(report);
        foreach (var violation in violations) Console.Error.WriteLine("gate failed: " + violation);
        return violations.Count == 0 ? 0 : 1;
    }

    public static E2EReport Measure(Model model, string goldenToolsJson, BenchE2EOptions options, string modelLabel, TextWriter? progress)
    {
        var report = new E2EReport { Model = modelLabel, Options = options };
        var sets = BenchPrompts.Select(options.Sets, goldenToolsJson);
        foreach (var threads in options.Threads)
        {
            var run = new ThreadMeasurement { Threads = threads };
            foreach (var set in sets)
            {
                progress?.WriteLine($"threads {threads}: {set.Name} ({set.Queries.Length} prompts x {options.Reps} reps)");
                run.Sets.Add(MeasureSet(model, set, threads, options));
            }
            if (options.Profile)
            {
                progress?.WriteLine($"threads {threads}: stage profile");
                run.Profile = BenchProfile.Run(model, sets[0].ToolsJson, sets[0].Queries[0], threads, options.Reps, options.DecodeBudget);
            }
            report.Runs.Add(run);
        }
        return report;
    }

    static SetMeasurement MeasureSet(Model model, PromptSet set, int threads, BenchE2EOptions options)
    {
        var sessionOptions = new NeedleOptions { Threads = threads, AutoDate = false, System = "", RetrievalTopK = 64 };
        var privateBefore = PrivateBytes();
        var started = Stopwatch.GetTimestamp();
        using var session = new NeedleSession(model, set.ToolsJson, sessionOptions);
        var measurement = new SetMeasurement { Name = set.Name, ToolCount = new ToolSet(set.ToolsJson).Tools.Count, PrefixTokens = session.ContextUsed,
            SessionCtorMs = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency,
            SessionCtorPrivateBytes = PrivateBytes() - privateBefore };
        foreach (var query in set.Queries) measurement.Prompts.Add(new PromptMeasurement { Query = query });
        for (var warm = 0; warm < options.Warmups; warm++)
            foreach (var query in set.Queries)
            {
                session.Reset();
                session.Complete(query);
            }
        for (var rep = 0; rep < options.Reps; rep++)
            for (var i = 0; i < set.Queries.Length; i++)
                Record(measurement.Prompts[i], session, set.Queries[i]);
        return measurement;
    }

    static long PrivateBytes()
    {
        using var process = Process.GetCurrentProcess();
        return process.PrivateMemorySize64;
    }

    static void Record(PromptMeasurement prompt, NeedleSession session, string query)
    {
        session.Reset();
        var managedBefore = GC.GetTotalAllocatedBytes(true);
        var nativeBefore = Mem.Allocations;
        var result = session.Complete(query);
        var nativeAllocs = Mem.Allocations - nativeBefore;
        var managedBytes = GC.GetTotalAllocatedBytes(true) - managedBefore;
        if (!result.Success || result.PrefillTps <= 0)
        {
            prompt.Failures++;
            return;
        }
        prompt.Calls = result.FunctionCalls.Count > 0 ? string.Join(",", result.FunctionCalls.Select(c => c.Name)) : "respond";
        prompt.PrefillTokens = result.PrefillTokens;
        prompt.DecodeTokens = result.DecodeTokens;
        prompt.PrefillMs.Add(result.PrefillTokens / result.PrefillTps * 1000.0);
        prompt.DecodeUsPerToken.Add(result.DecodeTokens > 0 && result.DecodeTps > 0 ? 1e6 / result.DecodeTps : 0);
        prompt.TtftMs.Add(result.TimeToFirstTokenMs);
        prompt.TotalMs.Add(result.TotalMs);
        prompt.PrefillTps.Add(result.PrefillTps);
        prompt.DecodeTps.Add(result.DecodeTps);
        prompt.ManagedBytes.Add(managedBytes);
        prompt.NativeAllocs.Add(nativeAllocs);
    }

    public static List<string> Violations(E2EReport report)
    {
        var violations = new List<string>();
        foreach (var run in report.Runs)
            foreach (var set in run.Sets)
                foreach (var prompt in set.Prompts)
                {
                    if (prompt.Failures > 0) violations.Add($"threads {run.Threads} {set.Name} \"{prompt.Query}\": {prompt.Failures} failed calls");
                    if (prompt.ManagedBytes.Count == 0) continue;
                    var managed = Spread.Of(prompt.ManagedBytes).Median;
                    if (report.Options.MaxManagedBytes is { } maxManaged && managed > maxManaged)
                        violations.Add($"threads {run.Threads} {set.Name} \"{prompt.Query}\": {managed:F0} managed bytes per call exceeds {maxManaged}");
                    var native = Spread.Of(prompt.NativeAllocs).Median;
                    if (report.Options.MaxNativeAllocs is { } maxNative && (!Mem.CountersEnabled || native > maxNative))
                        violations.Add(Mem.CountersEnabled
                            ? $"threads {run.Threads} {set.Name} \"{prompt.Query}\": {native:F0} native allocations per call exceeds {maxNative}"
                            : "--max-native-allocs needs a build with -p:NeedleAllocCounters=true");
                }
        return violations;
    }

    public static string ToText(E2EReport report)
    {
        var writer = new StringWriter();
        writer.WriteLine($"bench-e2e {report.Model} | {BenchMachine.Describe()} | reps {report.Options.Reps}");
        foreach (var run in report.Runs)
        {
            foreach (var set in run.Sets)
            {
                writer.WriteLine();
                writer.WriteLine($"threads {run.Threads} | {set.Name} set | {set.ToolCount} tools | prefix {set.PrefixTokens} tokens | ctor {set.SessionCtorMs:F1} ms (+{set.SessionCtorPrivateBytes / 1048576.0:F1} MB private)");
                writer.WriteLine($"{"query",-48}{"ptok",5}{"dtok",5}{"prefill ms",16}{"decode us/tok",18}{"ttft ms",14}{"total ms",16}{"managed B",16}{"native alloc",14}");
                writer.WriteLine($"{"",-58}{"best",8}{"med",8}{"best",9}{"med",9}{"best",7}{"med",7}{"best",8}{"med",8}{"best",8}{"med",8}{"med",14}");
                foreach (var prompt in set.Prompts) WriteRow(writer, Row.Of(prompt));
                WriteRow(writer, Row.Mean(set));
            }
            if (run.Profile != null)
            {
                writer.WriteLine();
                BenchProfile.WriteText(writer, run.Profile);
            }
        }
        return writer.ToString();
    }

    sealed record Row(string Label, int PrefillTokens, int DecodeTokens, Spread Prefill, Spread Decode, Spread Ttft, Spread Total, Spread Managed, Spread Native)
    {
        public static Row Of(PromptMeasurement p) =>
            new(p.Query, p.PrefillTokens, p.DecodeTokens, Spread.Of(p.PrefillMs), Spread.Of(p.DecodeUsPerToken), Spread.Of(p.TtftMs), Spread.Of(p.TotalMs), Spread.Of(p.ManagedBytes), Spread.Of(p.NativeAllocs));

        public static Row Mean(SetMeasurement set) =>
            new("mean of prompts", (int)set.Prompts.Average(p => p.PrefillTokens), (int)set.Prompts.Average(p => p.DecodeTokens),
                set.Mean(p => Spread.Of(p.PrefillMs)), set.Mean(p => Spread.Of(p.DecodeUsPerToken)), set.Mean(p => Spread.Of(p.TtftMs)),
                set.Mean(p => Spread.Of(p.TotalMs)), set.Mean(p => Spread.Of(p.ManagedBytes)), set.Mean(p => Spread.Of(p.NativeAllocs)));
    }

    static void WriteRow(StringWriter writer, Row r)
    {
        var native = Mem.CountersEnabled ? r.Native.Median.ToString("F0") : "n/a";
        var label = r.Label.Length > 47 ? r.Label[..47] : r.Label;
        writer.WriteLine($"{label,-48}{r.PrefillTokens,5}{r.DecodeTokens,5}{r.Prefill.Best,8:F2}{r.Prefill.Median,8:F2}{r.Decode.Best,9:F0}{r.Decode.Median,9:F0}{r.Ttft.Best,7:F1}{r.Ttft.Median,7:F1}{r.Total.Best,8:F1}{r.Total.Median,8:F1}{r.Managed.Best,8:F0}{r.Managed.Median,8:F0}{native,14}");
    }

    public static string ToJson(E2EReport report)
    {
        var json = new BenchJson().BeginObject()
            .Field("schema", "needle.bench-e2e/1")
            .Field("engine", "csharp")
            .Field("model", report.Model)
            .Field("reps", report.Options.Reps);
        BenchMachine.Write(json);
        json.Key("runs").BeginArray();
        foreach (var run in report.Runs)
        {
            json.BeginObject().Field("threads", run.Threads).Key("sets").BeginArray();
            foreach (var set in run.Sets) WriteSet(json, set);
            json.EndArray();
            if (run.Profile != null)
            {
                json.Key("profile");
                BenchProfile.WriteJson(json, run.Profile);
            }
            json.EndObject();
        }
        return json.EndArray().EndObject().ToString();
    }

    static void WriteSet(BenchJson json, SetMeasurement set)
    {
        json.BeginObject()
            .Field("name", set.Name)
            .Field("tools", set.ToolCount)
            .Field("prefix_tokens", set.PrefixTokens)
            .Field("session_ctor_ms", set.SessionCtorMs)
            .Field("session_ctor_private_bytes", set.SessionCtorPrivateBytes)
            .Key("prompts").BeginArray();
        foreach (var p in set.Prompts)
        {
            json.BeginObject()
                .Field("query", p.Query)
                .Field("calls", p.Calls)
                .Field("failures", p.Failures)
                .Field("prefill_tokens", p.PrefillTokens)
                .Field("decode_tokens", p.DecodeTokens)
                .Field("prefill_ms", Spread.Of(p.PrefillMs))
                .Field("decode_us_per_token", Spread.Of(p.DecodeUsPerToken))
                .Field("ttft_ms", Spread.Of(p.TtftMs))
                .Field("total_ms", Spread.Of(p.TotalMs))
                .Field("prefill_tps", Spread.OfThroughput(p.PrefillTps))
                .Field("decode_tps", Spread.OfThroughput(p.DecodeTps))
                .Field("managed_bytes", Spread.Of(p.ManagedBytes));
            if (Mem.CountersEnabled) json.Field("native_allocs", Spread.Of(p.NativeAllocs)); else json.Key("native_allocs").Null();
            json.EndObject();
        }
        json.EndArray().Key("summary").BeginObject()
            .Field("prefill_ms", set.Mean(p => Spread.Of(p.PrefillMs)))
            .Field("decode_us_per_token", set.Mean(p => Spread.Of(p.DecodeUsPerToken)))
            .Field("ttft_ms", set.Mean(p => Spread.Of(p.TtftMs)))
            .Field("total_ms", set.Mean(p => Spread.Of(p.TotalMs)))
            .Field("prefill_tps", set.Mean(p => Spread.OfThroughput(p.PrefillTps)))
            .Field("decode_tps", set.Mean(p => Spread.OfThroughput(p.DecodeTps)))
            .Field("managed_bytes", set.Mean(p => Spread.Of(p.ManagedBytes)))
            .EndObject().EndObject();
    }
}
