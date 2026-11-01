using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace Needle.Cli;

internal sealed record EvalOptions
{
    public int Threads { get; init; } = Math.Clamp(Environment.ProcessorCount / 2, 1, 8);
    public string[] Policy { get; init; } = Array.Empty<string>();
    public string[]? Baseline { get; init; }
    public bool Json { get; init; }
    public bool Verbose { get; init; }
    public int Limit { get; init; }
    public string? Tag { get; init; }
    public double? MinAccuracy { get; init; }
    public double? MaxDrop { get; init; }

    public bool Compare => Baseline != null;

    public static EvalOptions Parse(string[] args)
    {
        var options = new EvalOptions();
        for (var i = 0; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value");
            string[] Assignments()
            {
                var found = new List<string>();
                while (i + 1 < args.Length && !args[i + 1].StartsWith("--")) found.Add(args[++i]);
                return found.ToArray();
            }
            options = args[i] switch
            {
                "--policy" => options with { Policy = options.Policy.Concat(Assignments()).ToArray() },
                "--baseline" => options with { Baseline = (options.Baseline ?? Array.Empty<string>()).Concat(Assignments()).ToArray() },
                "--compare" => options with { Baseline = options.Baseline ?? Array.Empty<string>() },
                "--threads" => options with { Threads = int.Parse(Next(), CultureInfo.InvariantCulture) },
                "--limit" => options with { Limit = int.Parse(Next(), CultureInfo.InvariantCulture) },
                "--tag" => options with { Tag = Next() },
                "--min-accuracy" => options with { MinAccuracy = double.Parse(Next(), CultureInfo.InvariantCulture) },
                "--max-drop" => options with { MaxDrop = double.Parse(Next(), CultureInfo.InvariantCulture) },
                "--json" => options with { Json = true },
                "--verbose" => options with { Verbose = true },
                var unknown => throw new ArgumentException($"unknown option {unknown}"),
            };
        }
        if (options.Threads < 1) throw new ArgumentException("--threads must be at least 1");
        return options;
    }
}

internal static class Eval
{
    public const string Usage = "needle eval <needle3.cact> <dataset.jsonl|dir> [--policy name=value ...] [--compare] [--baseline name=value ...] [--threads N] [--limit N] [--tag name] [--min-accuracy f] [--max-drop f] [--json] [--verbose]";

    public static int Run(string modelPath, string datasetPath, string[] args)
    {
        EvalOptions options;
        EvalDataset dataset;
        EvalPolicy defaults;
        EvalPolicy candidate;
        EvalPolicy baseline;
        try
        {
            options = EvalOptions.Parse(args);
            defaults = EvalPolicy.Defaults(options.Threads);
            candidate = EvalPolicies.Apply(defaults, options.Policy);
            baseline = EvalPolicies.Apply(defaults, options.Baseline ?? Array.Empty<string>());
            dataset = EvalDataset.Load(datasetPath, options.Tag, options.Limit);
        }
        catch (Exception e) when (e is ArgumentException or FormatException or System.IO.IOException or OverflowException)
        {
            Console.Error.WriteLine(e.Message);
            Console.Error.WriteLine(Usage);
            return 2;
        }
        if (dataset.Cases.Count == 0)
        {
            Console.Error.WriteLine("dataset selects no prompts");
            return 2;
        }

        using var model = Model.Open(modelPath);
        var runs = new List<EvalRun>();
        if (options.Compare) runs.Add(EvalRunner.Run(model, dataset, baseline, "baseline", Console.Error));
        runs.Add(EvalRunner.Run(model, dataset, candidate, options.Compare ? "candidate" : "run", Console.Error));

        if (options.Json) Console.Out.WriteLine(EvalJson.Render(modelPath, datasetPath, defaults, runs));
        else
        {
            foreach (var run in runs) Console.Out.WriteLine(EvalText.Render(run, defaults, options.Verbose));
            if (runs.Count == 2) Console.Out.WriteLine(EvalText.RenderComparison(runs[0], runs[1], defaults));
        }
        return Gate(options, runs);
    }

    static int Gate(EvalOptions options, IReadOnlyList<EvalRun> runs)
    {
        var final = EvalMetrics.Summarize("overall", runs[^1].Results).OverallAccuracy;
        var failed = false;
        if (options.MinAccuracy is { } floor && final < floor)
        {
            Console.Error.WriteLine($"gate failed: overall accuracy {EvalText.Percent(final)} is below {EvalText.Percent(floor)}");
            failed = true;
        }
        if (options.MaxDrop is { } drop && runs.Count == 2)
        {
            var before = EvalMetrics.Summarize("overall", runs[0].Results).OverallAccuracy;
            if (before - final > drop)
            {
                Console.Error.WriteLine($"gate failed: accuracy fell from {EvalText.Percent(before)} to {EvalText.Percent(final)}, more than {EvalText.Percent(drop)}");
                failed = true;
            }
        }
        return failed ? 1 : 0;
    }
}
