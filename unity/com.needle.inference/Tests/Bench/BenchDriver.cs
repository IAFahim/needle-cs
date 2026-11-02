// N11 bench driver: one runner core for every Unity runtime, all emitting
// `needle.bench-e2e/1` JSON with the same shape as `needle bench-e2e`
// (src/Needle.Cli/BenchE2E.cs). Single source of truth: the CoreCLR host
// (samples/BenchNs21), the Mono harness (MonoMain), the EditMode test and
// the IL2CPP player scene all call BenchDriver.Run.
//
// Constraints: C# 7-level syntax (Unity 7000 compiles at C# 9, the bundled
// Mono 6.13 csc is older), netstandard2.1 BCL only, no UnityEngine
// references (progress flows out through Action<string>), public Needle API
// only (the test/player assemblies are not in InternalsVisibleTo).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Needle;

namespace Needle.Bench
{
    /// <summary>
    /// Emits <c>needle.bench-e2e/1</c> JSON for the golden + game prompt sets and
    /// checks the golden-set call outputs. Returns the failure count (0 = every
    /// call reproduced, every number finite). The JSON string is byte-compatible
    /// in shape with BenchE2E.ToJson: same object, same fields, same order.
    /// </summary>
    public static class BenchDriver
    {
        public const string Schema = "needle.bench-e2e/1";

        public static readonly string[] GoldenQueries = new string[]
        {
            "what's it like in Lagos right now?",
            "set a timer for 12 minutes and play some Miles Davis",
        };

        // Expected FunctionCalls for GoldenQueries under tools/tools.json, taken
        // from SessionTests (get_weather for Lagos; set_timer + play_music chain).
        // A golden query is only checked when every expected name is present in
        // the tools JSON, so custom tools files degrade to record-only.
        public static readonly string[][] GoldenCalls = new string[][]
        {
            new string[] { "get_weather" },
            new string[] { "set_timer", "play_music" },
        };

        public static readonly string[] GameQueries = new string[]
        {
            "move to 120, 45",
            "attack the orc chieftain",
            "give Lyra 3 potions",
            "switch to wedge formation and follow the captain",
            "say hello traveler, the bridge is out",
            "stop everything",
            "give 50 gold to Borin then move to 10, 20",
            "follow Mira",
            "attack the watchtower near the river",
            "form a circle",
        };

        /// <summary>
        /// Runs the bench. <paramref name="log"/> takes progress lines (stderr /
        /// Debug.Log at the call site). <paramref name="gameToolsJson"/> may be
        /// null or empty to skip the game set. <paramref name="sets"/> is one of
        /// "golden", "game", "all".
        /// </summary>
        public static int Run(
            Action<string> log,
            string modelPath,
            string goldenToolsJson,
            string gameToolsJson,
            int[] threads,
            int reps,
            int warmups,
            string sets,
            out string json)
        {
            json = "";
            bool wantGolden = sets == "all" || sets == "golden";
            bool wantGame = sets == "all" || sets == "game";
            if (!wantGolden && !wantGame)
            {
                log("bench-ns21: --sets expects golden, game or all, got '" + sets + "'");
                return -1;
            }
            if (wantGame && string.IsNullOrEmpty(gameToolsJson))
            {
                log("bench-ns21: game set requested but no game tools JSON was given");
                return -1;
            }
            if (threads == null || threads.Length == 0)
            {
                log("bench-ns21: no thread counts given");
                return -1;
            }
            if (reps < 1) reps = 1;
            if (warmups < 0) warmups = 0;

            var report = new BenchReport();
            report.Model = modelPath;
            report.Reps = reps;
            int failures = 0;
            using (var model = Model.Open(modelPath))
            {
                foreach (int threadCount in threads)
                {
                    var run = new BenchThreadRun();
                    run.Threads = threadCount;
                    if (wantGolden)
                    {
                        log("threads " + threadCount + ": golden (" + GoldenQueries.Length + " prompts x " + reps + " reps)");
                        run.Sets.Add(MeasureSet(model, "golden", goldenToolsJson, GoldenQueries, GoldenCalls, threadCount, reps, warmups, log, ref failures));
                    }
                    if (wantGame)
                    {
                        log("threads " + threadCount + ": game (" + GameQueries.Length + " prompts x " + reps + " reps)");
                        run.Sets.Add(MeasureSet(model, "game", gameToolsJson, GameQueries, null, threadCount, reps, warmups, log, ref failures));
                    }
                    report.Runs.Add(run);
                }
            }
            json = report.ToJson();
            return failures;
        }

        static BenchSet MeasureSet(
            Model model,
            string name,
            string toolsJson,
            string[] queries,
            string[][] expectedCalls,
            int threads,
            int reps,
            int warmups,
            Action<string> log,
            ref int failures)
        {
            var options = new NeedleOptions { Threads = threads, AutoDate = false, System = "", RetrievalTopK = 64 };
            long privateBefore = PrivateBytes();
            long started = Stopwatch.GetTimestamp();
            var session = new NeedleSession(model, toolsJson, options);
            var set = new BenchSet();
            set.Name = name;
            set.ToolCount = new ToolSet(toolsJson).Tools.Count;
            set.PrefixTokens = session.ContextUsed;
            set.SessionCtorMs = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
            set.SessionCtorPrivateBytes = PrivateBytes() - privateBefore;
            foreach (string query in queries)
                set.Prompts.Add(new BenchPrompt(query));
            for (int w = 0; w < warmups; w++)
            {
                foreach (string query in queries)
                {
                    session.Reset();
                    session.Complete(query);
                }
            }
            for (int rep = 0; rep < reps; rep++)
            {
                for (int i = 0; i < queries.Length; i++)
                    Record(set.Prompts[i], session, queries[i], expectedCalls == null ? null : expectedCalls[i], toolsJson, ref failures);
            }
            session.Dispose();
            log("threads " + threads + ": " + name + " done, ctor " + set.SessionCtorMs.ToString("F1", CultureInfo.InvariantCulture) + " ms");
            return set;
        }

        static void Record(BenchPrompt prompt, NeedleSession session, string query, string[] expected, string toolsJson, ref int failures)
        {
            session.Reset();
            long managedBefore = ManagedHeap();
            long nativeBefore = NativeAllocations();
            CallResult result = session.Complete(query);
            long nativeAllocs = NativeAllocations() - nativeBefore;
            long managedBytes = ManagedHeap() - managedBefore;
            if (!result.Success || result.PrefillTps <= 0)
            {
                prompt.Failures++;
                failures++;
                return;
            }
            string calls = CallNames(result);
            prompt.Calls = calls;
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
            if (expected != null && CallsExpected(toolsJson, expected) && !CallsMatch(result, expected))
            {
                prompt.Failures++;
                failures++;
            }
        }
        // GC.GetTotalAllocatedBytes is netstandard2.1+; Unity's Mono facade set
        // (and the NET_Standard_2_0 script profile) only guarantees the 2.0
        // surface, so gauge the managed heap with GetTotalMemory instead.
        // Deltas are noisier than per-call counters but stay finite and honest.
        static long ManagedHeap() { return GC.GetTotalMemory(false); }

        // double.IsFinite is netstandard2.1+; spell it out for the 2.0 surface.
        static bool IsFinite(double value) { return !double.IsNaN(value) && !double.IsInfinity(value); }


        static string CallNames(CallResult result)
        {
            if (result.FunctionCalls.Count == 0) return "respond";
            var names = new List<string>(result.FunctionCalls.Count);
            foreach (FunctionCall call in result.FunctionCalls)
                names.Add(call.Name);
            return string.Join(",", names.ToArray());
        }

        static bool CallsMatch(CallResult result, string[] expected)
        {
            if (result.FunctionCalls.Count != expected.Length) return false;
            for (int i = 0; i < expected.Length; i++)
            {
                if (result.FunctionCalls[i].Name != expected[i]) return false;
            }
            return true;
        }

        static bool CallsExpected(string toolsJson, string[] expected)
        {
            foreach (string name in expected)
            {
                if (toolsJson.IndexOf(name, StringComparison.Ordinal) < 0) return false;
            }
            return true;
        }

        static long PrivateBytes()
        {
            try
            {
                using (var process = Process.GetCurrentProcess())
                    return process.PrivateMemorySize64;
            }
            catch (Exception)
            {
                return 0;
            }
        }

        static bool NativeCountersEnabled()
        {
            try
            {
                Type mem = Type.GetType("Needle.Mem, Needle.Inference");
                if (mem == null) return false;
                PropertyInfo enabled = mem.GetProperty("CountersEnabled", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
                if (enabled == null) return false;
                return (bool)enabled.GetValue(null, null);
            }
            catch (Exception)
            {
                return false;
            }
        }

        static long NativeAllocations()
        {
            try
            {
                Type mem = Type.GetType("Needle.Mem, Needle.Inference");
                if (mem == null) return 0;
                PropertyInfo allocations = mem.GetProperty("Allocations", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
                if (allocations == null) return 0;
                return (long)allocations.GetValue(null, null);
            }
            catch (Exception)
            {
                return 0;
            }
        }

        internal static string SimdTier()
        {
            try
            {
                Type tiers = Type.GetType("Needle.Tiers, Needle.Inference");
                if (tiers == null) return "unknown";
                FieldInfo active = tiers.GetField("Active", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
                if (active == null) return "unknown";
                string tier = active.GetValue(null).ToString().ToLowerInvariant();
                if (tier == "scalar") return "scalar";
                if (tier.IndexOf("neon", StringComparison.Ordinal) >= 0) return "neon";
                if (tier.IndexOf("avx", StringComparison.Ordinal) >= 0) return "avx2";
                return tier;
            }
            catch (Exception)
            {
                return "unknown";
            }
        }

        sealed class BenchPrompt
        {
            public string Query;
            public string Calls = "";
            public int Failures;
            public int PrefillTokens;
            public int DecodeTokens;
            public readonly List<double> PrefillMs = new List<double>();
            public readonly List<double> DecodeUsPerToken = new List<double>();
            public readonly List<double> TtftMs = new List<double>();
            public readonly List<double> TotalMs = new List<double>();
            public readonly List<double> PrefillTps = new List<double>();
            public readonly List<double> DecodeTps = new List<double>();
            public readonly List<long> ManagedBytes = new List<long>();
            public readonly List<long> NativeAllocs = new List<long>();

            public BenchPrompt(string query) { Query = query; }
        }

        sealed class BenchSet
        {
            public string Name = "";
            public int ToolCount;
            public int PrefixTokens;
            public double SessionCtorMs;
            public long SessionCtorPrivateBytes;
            public readonly List<BenchPrompt> Prompts = new List<BenchPrompt>();
        }

        sealed class BenchThreadRun
        {
            public int Threads;
            public readonly List<BenchSet> Sets = new List<BenchSet>();
        }

        sealed class BenchReport
        {
            public string Model = "";
            public int Reps;
            public readonly List<BenchThreadRun> Runs = new List<BenchThreadRun>();

            public string ToJson()
            {
                var json = new JsonWriter();
                json.BeginObject();
                json.Field("schema", Schema);
                json.Field("engine", "csharp");
                json.Field("model", Model);
                json.Field("api", "text");
                json.Field("reps", (long)Reps);
                json.Key("machine");
                json.BeginObject();
                json.Field("os", RuntimeInformation.OSDescription);
                json.Field("arch", RuntimeInformation.ProcessArchitecture.ToString());
                json.Field("logical_cores", (long)Environment.ProcessorCount);
                json.Field("runtime", RuntimeInformation.FrameworkDescription);
                json.Field("simd", SimdTier());
                json.Field("native_alloc_counters", NativeCountersEnabled());
                json.EndObject();
                json.Key("runs");
                json.BeginArray();
                foreach (BenchThreadRun run in Runs)
                {
                    json.BeginObject();
                    json.Field("threads", (long)run.Threads);
                    json.Key("sets");
                    json.BeginArray();
                    foreach (BenchSet set in run.Sets)
                        WriteSet(json, set);
                    json.EndArray();
                    json.EndObject();
                }
                json.EndArray();
                json.EndObject();
                return json.ToString();
            }

            static void WriteSet(JsonWriter json, BenchSet set)
            {
                json.BeginObject();
                json.Field("name", set.Name);
                json.Field("tools", (long)set.ToolCount);
                json.Field("prefix_tokens", (long)set.PrefixTokens);
                json.Field("session_ctor_ms", set.SessionCtorMs);
                json.Field("session_ctor_private_bytes", set.SessionCtorPrivateBytes);
                json.Key("prompts");
                json.BeginArray();
                foreach (BenchPrompt prompt in set.Prompts)
                {
                    json.BeginObject();
                    json.Field("query", prompt.Query);
                    json.Field("calls", prompt.Calls);
                    json.Field("failures", (long)prompt.Failures);
                    json.Field("prefill_tokens", (long)prompt.PrefillTokens);
                    json.Field("decode_tokens", (long)prompt.DecodeTokens);
                    json.Field("prefill_ms", SpreadOf(prompt.PrefillMs));
                    json.Field("decode_us_per_token", SpreadOf(prompt.DecodeUsPerToken));
                    json.Field("ttft_ms", SpreadOf(prompt.TtftMs));
                    json.Field("total_ms", SpreadOf(prompt.TotalMs));
                    json.Field("prefill_tps", ThroughputOf(prompt.PrefillTps));
                    json.Field("decode_tps", ThroughputOf(prompt.DecodeTps));
                    json.Field("managed_bytes", SpreadOfLong(prompt.ManagedBytes));
                    if (NativeCountersEnabled()) json.Field("native_allocs", SpreadOfLong(prompt.NativeAllocs));
                    else { json.Key("native_allocs"); json.Null(); }
                    json.EndObject();
                }
                json.EndArray();
                json.Key("summary");
                json.BeginObject();
                json.Field("prefill_ms", MeanOf(delegate (BenchPrompt p) { return SpreadOf(p.PrefillMs); }, set));
                json.Field("decode_us_per_token", MeanOf(delegate (BenchPrompt p) { return SpreadOf(p.DecodeUsPerToken); }, set));
                json.Field("ttft_ms", MeanOf(delegate (BenchPrompt p) { return SpreadOf(p.TtftMs); }, set));
                json.Field("total_ms", MeanOf(delegate (BenchPrompt p) { return SpreadOf(p.TotalMs); }, set));
                json.Field("prefill_tps", MeanOf(delegate (BenchPrompt p) { return ThroughputOf(p.PrefillTps); }, set));
                json.Field("decode_tps", MeanOf(delegate (BenchPrompt p) { return ThroughputOf(p.DecodeTps); }, set));
                json.Field("managed_bytes", MeanOf(delegate (BenchPrompt p) { return SpreadOfLong(p.ManagedBytes); }, set));
                json.EndObject();
                json.EndObject();
            }

            static Spread MeanOf(Func<BenchPrompt, Spread> select, BenchSet set)
            {
                double best = 0;
                double median = 0;
                foreach (BenchPrompt prompt in set.Prompts)
                {
                    Spread spread = select(prompt);
                    best += spread.Best;
                    median += spread.Median;
                }
                return new Spread(best / set.Prompts.Count, median / set.Prompts.Count);
            }
        }

        struct Spread
        {
            public double Best;
            public double Median;

            public Spread(double best, double median)
            {
                Best = best;
                Median = median;
            }
        }

        static Spread SpreadOf(List<double> values)
        {
            if (values.Count == 0) return new Spread(double.NaN, double.NaN);
            double[] sorted = values.ToArray();
            Array.Sort(sorted);
            return new Spread(sorted[0], (sorted[sorted.Length / 2] + sorted[(sorted.Length - 1) / 2]) / 2);
        }

        static Spread SpreadOfLong(List<long> values)
        {
            double[] converted = new double[values.Count];
            for (int i = 0; i < converted.Length; i++) converted[i] = values[i];
            var asList = new List<double>(converted);
            return SpreadOf(asList);
        }

        static Spread ThroughputOf(List<double> values)
        {
            Spread spread = SpreadOf(values);
            if (values.Count == 0) return spread;
            double peak = double.MinValue;
            foreach (double value in values)
            {
                if (value > peak) peak = value;
            }
            spread.Best = peak;
            return spread;
        }

        sealed class JsonWriter
        {
            readonly StringBuilder text = new StringBuilder();
            readonly List<bool> hasItems = new List<bool>();
            bool afterKey;

            public override string ToString() { return text.ToString(); }

            void Separate()
            {
                if (afterKey) { afterKey = false; return; }
                if (hasItems.Count > 0 && hasItems[hasItems.Count - 1]) text.Append(',');
                if (hasItems.Count > 0) hasItems[hasItems.Count - 1] = true;
            }

            public void BeginObject()
            {
                Separate();
                text.Append('{');
                hasItems.Add(false);
            }

            public void BeginArray()
            {
                Separate();
                text.Append('[');
                hasItems.Add(false);
            }

            public void EndObject()
            {
                text.Append('}');
                hasItems.RemoveAt(hasItems.Count - 1);
            }

            public void EndArray()
            {
                text.Append(']');
                hasItems.RemoveAt(hasItems.Count - 1);
            }

            public void Key(string name)
            {
                Separate();
                WriteString(name);
                text.Append(':');
                afterKey = true;
            }

            public void Value(double value)
            {
                Separate();
                text.Append(IsFinite(value) ? value.ToString("0.####", CultureInfo.InvariantCulture) : "null");
            }

            public void Value(long value)
            {
                Separate();
                text.Append(value.ToString(CultureInfo.InvariantCulture));
            }

            public void Value(string value)
            {
                Separate();
                WriteString(value);
            }

            public void Value(bool value)
            {
                Separate();
                text.Append(value ? "true" : "false");
            }

            public void Null()
            {
                Separate();
                text.Append("null");
            }

            public void Field(string name, double value) { Key(name); Value(value); }

            public void Field(string name, long value) { Key(name); Value(value); }

            public void Field(string name, string value) { Key(name); Value(value); }

            public void Field(string name, bool value) { Key(name); Value(value); }

            public void Field(string name, Spread value)
            {
                Key(name);
                BeginObject();
                Field("best", value.Best);
                Field("median", value.Median);
                EndObject();
            }

            void WriteString(string s)
            {
                text.Append('"');
                foreach (char c in s)
                {
                    switch (c)
                    {
                        case '"': text.Append("\\\""); break;
                        case '\\': text.Append("\\\\"); break;
                        case '\n': text.Append("\\n"); break;
                        case '\r': text.Append("\\r"); break;
                        case '\t': text.Append("\\t"); break;
                        case '\b': text.Append("\\b"); break;
                        case '\f': text.Append("\\f"); break;
                        default:
                            if (c < 0x20) text.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                            else text.Append(c);
                            break;
                    }
                }
                text.Append('"');
            }
        }
    }
}
