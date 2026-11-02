// N11 arg parsing shared by the CoreCLR host and the Mono harness.
// Kept at C#-7 level so Unity's bundled Mono 6.13 csc can compile it.
using System;
using System.IO;

namespace Needle.Bench
{
    sealed class BenchArgs
    {
        public string Model = "";
        public string Tools = "";
        public string GameTools = "";
        public string Threads = "4";
        public int Reps = 2;
        public int Warmups = 1;
        public string Sets = "golden";

        public static BenchArgs Parse(string[] args)
        {
            var parsed = new BenchArgs();
            var rest = new System.Collections.Generic.List<string>();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--threads" && i + 1 < args.Length) parsed.Threads = args[++i];
                else if (args[i] == "--reps" && i + 1 < args.Length) parsed.Reps = int.Parse(args[++i]);
                else if (args[i] == "--warmups" && i + 1 < args.Length) parsed.Warmups = int.Parse(args[++i]);
                else if (args[i] == "--sets" && i + 1 < args.Length) parsed.Sets = args[++i];
                else if (args[i] == "--game-tools" && i + 1 < args.Length) parsed.GameTools = args[++i];
                else rest.Add(args[i]);
            }
            if (rest.Count < 2)
                throw new ArgumentException("usage: bench-ns21 <model.cact> <tools.json> [--threads 4] [--reps N] [--warmups N] [--sets golden|game|all] [--game-tools game_tools.json]");
            parsed.Model = rest[0];
            parsed.Tools = rest[1];
            return parsed;
        }

        public string ToolsJson() { return File.ReadAllText(Tools); }

        public string GameToolsJson()
        {
            if (string.IsNullOrEmpty(GameTools)) return null;
            return File.ReadAllText(GameTools);
        }

        public int[] ThreadCounts()
        {
            string[] parts = Threads.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            var counts = new int[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                counts[i] = int.Parse(parts[i].Trim());
            return counts;
        }
    }
}
