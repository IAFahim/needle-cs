using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Needle;

namespace Needle.Cli;

internal sealed class StageProfile
{
    public static readonly int StageCount = Enum.GetValues<Stage>().Length;

    public readonly long[] Ticks = new long[StageCount];
    public long PoolRuns;
    public long Mark;

    public void Clear()
    {
        Array.Clear(Ticks);
        PoolRuns = 0;
    }

    public static string Name(int stage)
    {
        var pascal = ((Stage)stage).ToString();
        var builder = new StringBuilder();
        for (var i = 0; i < pascal.Length; i++)
        {
            if (i > 0 && char.IsUpper(pascal[i])) builder.Append('_');
            builder.Append(char.ToLowerInvariant(pascal[i]));
        }
        return builder.ToString();
    }
}

internal readonly struct ProfileClock : IStageClock
{
    readonly StageProfile profile;

    public ProfileClock(StageProfile profile) => this.profile = profile;

    public void Begin() => profile.Mark = Stopwatch.GetTimestamp();

    public void Lap(Stage stage)
    {
        var now = Stopwatch.GetTimestamp();
        profile.Ticks[(int)stage] += now - profile.Mark;
        profile.Mark = now;
    }

    public void PoolRun() => profile.PoolRuns++;
}

internal sealed class ProfileRep
{
    public int PrefillTokens;
    public int DecodeTokens;
    public double PrefillMs;
    public double[] PrefillStageMs = Array.Empty<double>();
    public long PrefillPoolRuns;
    public double DecodeUsPerToken;
    public double[] DecodeStageUsPerToken = Array.Empty<double>();
    public double DecodePoolRunsPerToken;
}

internal sealed class ProfileResult
{
    public int Threads;
    public string Query = "";
    public List<ProfileRep> Reps = new();

    public Spread Prefill => Of(r => r.PrefillMs);

    public Spread DecodePerToken => Of(r => r.DecodeUsPerToken);

    public Spread DecodeStage(int stage) => Of(r => r.DecodeStageUsPerToken[stage]);

    public Spread PrefillStage(int stage) => Of(r => r.PrefillStageMs[stage]);

    public Spread DecodeUnattributed => Of(r => r.DecodeUsPerToken - Sum(r.DecodeStageUsPerToken));

    public Spread PrefillUnattributed => Of(r => r.PrefillMs - Sum(r.PrefillStageMs));

    public double PoolRunsPerToken => Reps.Count == 0 ? double.NaN : Reps[^1].DecodePoolRunsPerToken;

    public int PrefillTokens => Reps.Count == 0 ? 0 : Reps[^1].PrefillTokens;

    public int DecodeTokens => Reps.Count == 0 ? 0 : Reps[^1].DecodeTokens;

    Spread Of(Func<ProfileRep, double> select)
    {
        var values = new double[Reps.Count];
        for (var i = 0; i < values.Length; i++) values[i] = select(Reps[i]);
        return Spread.Of(values);
    }

    static double Sum(double[] values)
    {
        var total = 0.0;
        foreach (var value in values) total += value;
        return total;
    }
}

internal static unsafe class BenchProfile
{
    const string PrefixHead = "<|im_start|>user\n<tools>";
    const string PrefixTail = "</tools>";
    const string QueryHead = "\n";
    const string QueryTail = "<|im_end|>\n<|im_start|>assistant\n";
    const int QueryHeadroomBytes = 4096;

    public static ProfileResult Run(Model model, string toolsJson, string query, int threads, int reps, int decodeBudget)
    {
        var profile = new StageProfile();
        using var engine = new EngineCore<ProfileClock>(model, 1024, 64, threads, null, new ProfileClock(profile));
        using var tokenizer = new Tokenizer(model, toolsJson.Length + QueryHeadroomBytes);
        Prime(engine, tokenizer, toolsJson);
        var prompt = Encode(tokenizer, QueryHead + query + QueryTail, leadingSpace: false);
        var result = new ProfileResult { Threads = threads, Query = query };
        var next = new int[1];
        for (var rep = -1; rep < reps; rep++)
        {
            engine.Rewind();
            profile.Clear();
            var started = Stopwatch.GetTimestamp();
            engine.Feed(prompt);
            var prefillTicks = Stopwatch.GetTimestamp() - started;
            var prefillStages = StageMilliseconds(profile);
            var prefillRuns = profile.PoolRuns;

            profile.Clear();
            var generated = 0;
            started = Stopwatch.GetTimestamp();
            for (var step = 0; step < decodeBudget; step++)
            {
                next[0] = engine.Greedy();
                if (tokenizer.IsStop(next[0])) break;
                engine.Feed(next);
                generated++;
            }
            var decodeTicks = Stopwatch.GetTimestamp() - started;
            if (rep < 0 || generated == 0) continue;
            var decodeStages = StageMicrosecondsPerToken(profile, generated);
            result.Reps.Add(new ProfileRep
            {
                PrefillTokens = prompt.Length,
                DecodeTokens = generated,
                PrefillMs = BenchTiming.Milliseconds(prefillTicks),
                PrefillStageMs = prefillStages,
                PrefillPoolRuns = prefillRuns,
                DecodeUsPerToken = BenchTiming.Milliseconds(decodeTicks) * 1000.0 / generated,
                DecodeStageUsPerToken = decodeStages,
                DecodePoolRunsPerToken = profile.PoolRuns / (double)generated,
            });
        }
        return result;
    }

    static void Prime(EngineCore<ProfileClock> engine, Tokenizer tokenizer, string toolsJson)
    {
        var prefix = Encode(tokenizer, PrefixHead + toolsJson + PrefixTail, leadingSpace: true);
        engine.Reset();
        engine.Feed(prefix, computeLogits: false);
        engine.Mark();
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

    static double[] StageMilliseconds(StageProfile profile)
    {
        var values = new double[StageProfile.StageCount];
        for (var i = 0; i < values.Length; i++) values[i] = BenchTiming.Milliseconds(profile.Ticks[i]);
        return values;
    }

    static double[] StageMicrosecondsPerToken(StageProfile profile, int tokens)
    {
        var values = new double[StageProfile.StageCount];
        for (var i = 0; i < values.Length; i++) values[i] = BenchTiming.Milliseconds(profile.Ticks[i]) * 1000.0 / tokens;
        return values;
    }

    public static void WriteJson(BenchJson json, ProfileResult result)
    {
        json.BeginObject()
            .Field("threads", result.Threads)
            .Field("query", result.Query)
            .Field("prefill_tokens", result.PrefillTokens)
            .Field("decode_tokens", result.DecodeTokens)
            .Field("pool_runs_per_token", result.PoolRunsPerToken)
            .Field("prefill_ms", result.Prefill)
            .Field("decode_us_per_token", result.DecodePerToken);
        json.Key("prefill_stage_ms").BeginObject();
        for (var s = 0; s < StageProfile.StageCount; s++) json.Field(StageProfile.Name(s), result.PrefillStage(s));
        json.Field("unattributed", result.PrefillUnattributed).EndObject();
        json.Key("decode_stage_us_per_token").BeginObject();
        for (var s = 0; s < StageProfile.StageCount; s++) json.Field(StageProfile.Name(s), result.DecodeStage(s));
        json.Field("unattributed", result.DecodeUnattributed).EndObject();
        json.EndObject();
    }

    public static void WriteText(System.IO.TextWriter writer, ProfileResult result)
    {
        var decode = result.DecodePerToken;
        var prefill = result.Prefill;
        writer.WriteLine($"profile threads={result.Threads} prompt=\"{result.Query}\" prefill {result.PrefillTokens} tok, decode {result.DecodeTokens} tok, {result.PoolRunsPerToken:F1} Pool.Run per token, {result.Reps.Count} reps");
        writer.WriteLine($"{"stage",-26}{"decode us/tok best",20}{"median",10}{"share",8}{"prefill ms best",18}{"median",10}");
        for (var s = 0; s < StageProfile.StageCount; s++)
        {
            var d = result.DecodeStage(s);
            var p = result.PrefillStage(s);
            writer.WriteLine($"{StageProfile.Name(s),-26}{d.Best,20:F1}{d.Median,10:F1}{d.Median / Math.Max(1e-9, decode.Median),8:P0}{p.Best,18:F3}{p.Median,10:F3}");
        }
        var other = result.DecodeUnattributed;
        var otherPrefill = result.PrefillUnattributed;
        writer.WriteLine($"{"unattributed",-26}{other.Best,20:F1}{other.Median,10:F1}{other.Median / Math.Max(1e-9, decode.Median),8:P0}{otherPrefill.Best,18:F3}{otherPrefill.Median,10:F3}");
        writer.WriteLine($"{"total",-26}{decode.Best,20:F1}{decode.Median,10:F1}{"",8}{prefill.Best,18:F3}{prefill.Median,10:F3}");
    }
}
