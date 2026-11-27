using System;
using System.Linq;
using System.Text;
using Needle;

namespace Needle.Cli;

internal static class DecodeHash
{
    public static int Run(string modelPath, string toolsPath, string[] options)
    {
        var threads = new[] { 1, 2, 3, 4, 6, 8 };
        var decode = 48;
        for (var i = 0; i < options.Length; i++)
        {
            switch (options[i])
            {
                case "--threads": threads = options[++i].Split(',').Select(int.Parse).ToArray(); break;
                case "--decode": decode = int.Parse(options[++i]); break;
                default: throw new ArgumentException($"unknown option {options[i]}");
            }
        }
        var toolsJson = System.IO.File.ReadAllText(toolsPath);
        using var model = Model.Open(modelPath);
        using var tokenizer = new Tokenizer(model, toolsJson.Length + 4096);
        var prefix = Encode(tokenizer, "<|im_start|>user\n<tools>" + toolsJson + "</tools>", true);
        string[] queries = { "what's it like in Lagos right now?", "set a timer for 12 minutes and play some Miles Davis" };
        foreach (var t in threads)
        {
            using var engine = new Engine(model, 2048, 64, t);
            foreach (var query in queries)
            {
                engine.Reset();
                engine.Feed(prefix, computeLogits: false);
                engine.Feed(Encode(tokenizer, "\n" + query + "<|im_end|>\n<|im_start|>assistant\n", false));
                var hash = 14695981039346656037UL;
                var tokens = new StringBuilder();
                var next = new int[1];
                for (var step = 0; step < decode; step++)
                {
                    foreach (var value in engine.Logits)
                    {
                        hash ^= (uint)BitConverter.SingleToInt32Bits(value);
                        hash *= 1099511628211UL;
                    }
                    next[0] = engine.Greedy();
                    tokens.Append(next[0]).Append(' ');
                    engine.Feed(next);
                }
                Console.WriteLine($"threads {t} \"{query}\" logits {hash:x16} tokens {tokens}");
            }
        }
        return 0;
    }

    static int[] Encode(Tokenizer tokenizer, string text, bool leadingSpace)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        var ids = new int[bytes.Length + 8];
        var begin = leadingSpace ? 1 : 0;
        if (leadingSpace) ids[0] = tokenizer.BeginId;
        var count = begin + tokenizer.Encode(bytes, ids.AsSpan(begin), leadingSpace);
        return ids.AsSpan(0, count).ToArray();
    }
}
