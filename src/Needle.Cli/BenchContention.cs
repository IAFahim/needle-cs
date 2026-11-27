using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using Needle;

namespace Needle.Cli;

internal static class BenchContention
{
    const string PrefixHead = "<|im_start|>user\n<tools>";
    const string PrefixTail = "</tools>";
    const string QueryHead = "\n";
    const string QueryTail = "<|im_end|>\n<|im_start|>assistant\n";

    public static int Run(string modelPath, string toolsPath, string[] options)
    {
        var threads = 4;
        var hogs = new[] { 0, Environment.ProcessorCount };
        var policies = new[] { "default", "game" };
        var gaps = new[] { 0, 300 };
        var asks = 6;
        var decode = 40;
        var paths = new[] { "session-default", "session-game", "host-default", "host-game", "agent-post-default", "agent-post-game" };
        var engineTable = true;
        for (var i = 0; i < options.Length; i++)
        {
            switch (options[i])
            {
                case "--threads": threads = int.Parse(options[++i]); break;
                case "--hogs": hogs = options[++i].Split(',').Select(int.Parse).ToArray(); break;
                case "--policies": policies = options[++i].Split(','); break;
                case "--gap-us": gaps = options[++i].Split(',').Select(int.Parse).ToArray(); break;
                case "--asks": asks = int.Parse(options[++i]); break;
                case "--decode": decode = int.Parse(options[++i]); break;
                case "--paths": paths = options[++i].Split(',', StringSplitOptions.RemoveEmptyEntries); break;
                case "--no-engine": engineTable = false; break;
                default: throw new ArgumentException($"unknown option {options[i]}");
            }
        }
        var toolsJson = System.IO.File.ReadAllText(toolsPath);
        using var model = Model.Open(modelPath);
        using var tokenizer = new Tokenizer(model, toolsJson.Length + 4096);
        var prefix = Encode(tokenizer, PrefixHead + toolsJson + PrefixTail, true);
        var prompt = Encode(tokenizer, QueryHead + "set a timer for 12 minutes and play some Miles Davis" + QueryTail, false);
        if (!engineTable) goto Paths;
        Console.WriteLine($"contention bench: {Environment.ProcessorCount} logical cpus, {threads} engine threads, {asks} asks x up to {decode} decode tokens, hogs are busy normal-priority threads");
        Console.WriteLine($"{"policy",-8}{"hogs",6}{"gap us",8}{"token p50 us",14}{"token p90 us",14}{"token p99 us",14}{"token max us",14}{"ask p50 ms",12}{"ask max ms",12}");
        foreach (var hogCount in hogs)
            foreach (var gap in gaps)
                foreach (var name in policies)
                {
                    var policy = (name == "game" ? ThreadPolicy.Game : ThreadPolicy.Default).WithThreads(threads);
                    using var engine = new Engine(model, policy);
                    engine.Reset();
                    engine.Feed(prefix, computeLogits: false);
                    engine.Mark();
                    var stop = 0;
                    var workers = new Thread[hogCount];
                    for (var h = 0; h < hogCount; h++)
                    {
                        workers[h] = new Thread(() => Hog(ref stop)) { IsBackground = true, Name = $"hog-{h}" };
                        workers[h].Start();
                    }
                    var tokens = new List<double>();
                    var askMs = new List<double>();
                    var next = new int[1];
                    for (var ask = -1; ask < asks; ask++)
                    {
                        engine.Rewind();
                        var askStart = Stopwatch.GetTimestamp();
                        engine.Feed(prompt);
                        for (var step = 0; step < decode; step++)
                        {
                            next[0] = engine.Greedy();
                            if (tokenizer.IsStop(next[0])) break;
                            if (gap > 0) Busy(gap);
                            var started = Stopwatch.GetTimestamp();
                            engine.Feed(next);
                            if (ask >= 0) tokens.Add((Stopwatch.GetTimestamp() - started) * 1e6 / Stopwatch.Frequency);
                        }
                        if (ask >= 0) askMs.Add((Stopwatch.GetTimestamp() - askStart) * 1e3 / Stopwatch.Frequency);
                    }
                    Volatile.Write(ref stop, 1);
                    foreach (var worker in workers) worker.Join();
                    tokens.Sort();
                    askMs.Sort();
                    Console.WriteLine($"{name,-8}{hogCount,6}{gap,8}{Pct(tokens, 0.5),14:F0}{Pct(tokens, 0.9),14:F0}{Pct(tokens, 0.99),14:F0}{tokens[^1],14:F0}{Pct(askMs, 0.5),12:F1}{askMs[^1],12:F1}");
                }
        Paths:
        if (paths.Length == 0) return 0;
        Console.WriteLine();
        Console.WriteLine($"ask latency by path: {threads} threads, {asks} asks each after one warm-up, host and agent polled every 1 ms from a normal-priority thread");
        Console.WriteLine($"{"path",-20}{"hogs",6}{"ask p50 ms",12}{"ask p90 ms",12}{"ask max ms",12}{"tokens",8}");
        foreach (var hogCount in hogs)
            foreach (var path in paths)
            {
                var times = new List<double>();
                var decoded = 0;
                using (var asker = Asker.Open(path, modelPath, toolsJson, threads))
                {
                    var stop = 0;
                    var workers = new Thread[hogCount];
                    for (var h = 0; h < hogCount; h++)
                    {
                        workers[h] = new Thread(() => Hog(ref stop)) { IsBackground = true, Name = $"hog-{h}" };
                        workers[h].Start();
                    }
                    try
                    {
                        for (var ask = -1; ask < asks; ask++)
                        {
                            var started = Stopwatch.GetTimestamp();
                            var tokensOut = asker.Ask(Query);
                            if (ask < 0) continue;
                            times.Add((Stopwatch.GetTimestamp() - started) * 1e3 / Stopwatch.Frequency);
                            decoded = tokensOut;
                        }
                    }
                    finally
                    {
                        Volatile.Write(ref stop, 1);
                        foreach (var worker in workers) worker.Join();
                    }
                }
                times.Sort();
                Console.WriteLine($"{path,-20}{hogCount,6}{Pct(times, 0.5),12:F1}{Pct(times, 0.9),12:F1}{times[^1],12:F1}{decoded,8}");
            }
        return 0;
    }

    const string Query = "set a timer for 12 minutes and play some Miles Davis";

    sealed class Asker : IDisposable
    {
        NeedleSession? session;
        NeedleHost? host;
        NeedleChannel channel;
        NeedleModel publicModel;
        NeedleAgent agent;
        bool hasAgent;
        Model? model;
        readonly byte[] calls = new byte[8192];
        readonly byte[] reasoning = new byte[8192];
        readonly byte[] response = new byte[8192];

        public static Asker Open(string path, string modelPath, string toolsJson, int threads)
        {
            var policy = (path.EndsWith("game", StringComparison.Ordinal) ? ThreadPolicy.Game : ThreadPolicy.Default).WithThreads(threads);
            var options = new NeedleOptions { Threading = policy, AutoDate = false, Stateless = true };
            var asker = new Asker();
            if (path.StartsWith("session", StringComparison.Ordinal))
            {
                asker.model = Model.Open(modelPath);
                asker.session = new NeedleSession(asker.model, toolsJson, options);
            }
            else if (path.StartsWith("host", StringComparison.Ordinal))
            {
                asker.host = new NeedleHost();
                asker.channel = asker.host.Start(new HostOptions { ModelPath = modelPath, ToolsJson = toolsJson, Session = options, StatelessQueries = true });
            }
            else if (path.StartsWith("agent-post", StringComparison.Ordinal))
            {
                asker.publicModel = NeedleModel.Open(modelPath);
                asker.agent = asker.publicModel.Agent(toolsJson, new AgentOptions { Threading = policy, AutoDate = Toggle.Off, Stateless = true });
                asker.hasAgent = true;
            }
            else throw new ArgumentException($"unknown path {path}");
            return asker;
        }

        public int Ask(string query)
        {
            var utf8 = Encoding.UTF8.GetBytes(query);
            if (session != null)
            {
                session.Reset();
                return session.Complete(utf8, calls, reasoning, response).DecodeTokens;
            }
            if (host != null)
            {
                if (!channel.TrySubmitQuery(utf8, out var ticket)) throw new InvalidOperationException("host queue full");
                ChannelReply reply;
                while (!channel.TryReceive(out reply) || reply.Ticket != ticket) Thread.Sleep(1);
                return reply.Reply.DecodeTokens;
            }
            agent.Post(utf8);
            Answer answer;
            while (!agent.TryTake(out answer)) Thread.Sleep(1);
            return answer.Calls.Count;
        }

        public void Dispose()
        {
            session?.Dispose();
            model?.Dispose();
            host?.Dispose();
            if (hasAgent) agent.Dispose();
            if (hasAgent) publicModel.Dispose();
        }
    }

    static void Hog(ref int stop)
    {
        var x = 1.0;
        while (Volatile.Read(ref stop) == 0)
            for (var i = 0; i < 10000; i++) x = x * 1.0000001 + 1e-9;
        GC.KeepAlive(x);
    }

    static void Busy(int microseconds)
    {
        var end = Stopwatch.GetTimestamp() + microseconds * Stopwatch.Frequency / 1_000_000;
        while (Stopwatch.GetTimestamp() < end) Thread.SpinWait(8);
    }

    static double Pct(List<double> sorted, double q) => sorted.Count == 0 ? double.NaN : sorted[Math.Min(sorted.Count - 1, (int)Math.Ceiling(q * sorted.Count) - 1)];

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
