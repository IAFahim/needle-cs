// N11a entry point: CoreCLR + ns2.1. Prints needle.bench-e2e/1 JSON to stdout,
// progress to stderr, exit code = failure count (golden calls not reproduced).
using System;

namespace Needle.Bench
{
    internal static class CoreClrMain
    {
        internal static int Main(string[] args)
        {
            BenchArgs parsed;
            try
            {
                parsed = BenchArgs.Parse(args);
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(e.Message);
                return 2;
            }
            string json;
            int failures = BenchDriver.Run(
                Console.Error.WriteLine,
                parsed.Model,
                parsed.ToolsJson(),
                parsed.GameToolsJson(),
                parsed.ThreadCounts(),
                parsed.Reps,
                parsed.Warmups,
                parsed.Sets,
                out json);
            Console.Out.WriteLine(json);
            return failures < 0 ? 2 : Math.Min(failures, 126);
        }
    }
}
