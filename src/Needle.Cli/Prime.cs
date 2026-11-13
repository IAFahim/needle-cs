using System;
using System.Diagnostics;
using System.IO;
using Needle;

namespace Needle.Cli;

internal static class Prime
{
    public static int Run(string modelPath, string toolsPath, string outPath, string[] options)
    {
        var context = 2048;
        var threads = Math.Clamp(Environment.ProcessorCount / 2, 1, 8);
        var maxReasoning = 96;
        var maxCalls = 384;
        string? system = Environment.GetEnvironmentVariable("NEEDLE_SYSTEM");

        for (var i = 0; i < options.Length; i++)
        {
            switch (options[i])
            {
                case "--context":
                    if (!TakeInt(options, ref i, out context) || context < 1) return Usage();
                    break;
                case "--threads":
                    if (!TakeInt(options, ref i, out threads) || threads < 1) return Usage();
                    break;
                case "--max-reasoning":
                    if (!TakeInt(options, ref i, out maxReasoning) || maxReasoning < 0) return Usage();
                    break;
                case "--max-calls":
                    if (!TakeInt(options, ref i, out maxCalls) || maxCalls < 1) return Usage();
                    break;
                case "--system":
                    if (i + 1 >= options.Length) return Usage();
                    system = options[++i];
                    break;
                case "--help":
                case "-h":
                    return Usage();
                default:
                    Console.Error.WriteLine($"unknown option: {options[i]}");
                    return Usage();
            }
        }

        string toolsJson;
        try
        {
            toolsJson = File.ReadAllText(toolsPath);
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"cannot read tools {toolsPath}: {e.Message}");
            return 1;
        }

        var sessionOptions = new NeedleOptions
        {
            Threads = threads,
            Context = context,
            MaxReasoningTokens = maxReasoning,
            MaxCallTokens = maxCalls,
            System = system,
            AutoDate = false,
            RetrievalTopK = int.MaxValue,
        };

        try
        {
            using var model = Model.Open(modelPath);
            var build = Stopwatch.StartNew();
            using var session = new NeedleSession(model, toolsJson, sessionOptions);
            build.Stop();
            var tokens = session.ContextUsed;
            var bytes = session.ExportPrimedPrefix();
            File.WriteAllBytes(outPath, bytes);

            var restore = Stopwatch.StartNew();
            using var revived = new NeedleSession(model, toolsJson, bytes, sessionOptions);
            restore.Stop();
            if (revived.ContextUsed != tokens)
            {
                Console.Error.WriteLine($"primed prefix mismatch: baked {tokens} tokens, restored {revived.ContextUsed}");
                return 1;
            }

            Console.Error.WriteLine($"prime: {tokens} tokens in {build.Elapsed.TotalMilliseconds:F1} ms, wrote {bytes.Length} bytes -> {outPath}");
            Console.Error.WriteLine($"prime: restored {tokens} tokens in {restore.Elapsed.TotalMilliseconds:F1} ms (context {context}, threads {threads})");
            Console.WriteLine(outPath);
            return 0;
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"cannot bake primed prefix: {e.Message}");
            return 1;
        }
    }

    static bool TakeInt(string[] options, ref int i, out int value)
    {
        value = 0;
        if (i + 1 >= options.Length) return false;
        return int.TryParse(options[++i], out value);
    }

    static int Usage()
    {
        Console.Error.WriteLine("needle prime <model.cact> <tools.json> <out.prefix> [--system TEXT] [--context N] [--threads N] [--max-reasoning N] [--max-calls N]");
        Console.Error.WriteLine("  Bake the post-prefill KV snapshot (AutoDate=false, full tool set, no retrieval).");
        Console.Error.WriteLine("  Load with the NeedleSession(model, tools, prefixBytes, options) constructor,");
        Console.Error.WriteLine("  'needle serve/call --prefix <out.prefix> --no-date', or HostOptions.PrimedPrefixPath.");
        return 2;
    }
}
