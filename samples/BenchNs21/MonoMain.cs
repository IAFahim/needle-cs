using System;

namespace Needle.Bench
{
    internal static class MonoMain
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
                delegate (string line) { Console.Error.WriteLine(line); },
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
