using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Needle;

namespace Needle.Cli;

// The six 4.x acceptance suites (tests/eval/environments) scored with upstream's rule.
// Suites and rule are derived from the Apache-2.0 cactus-compute 4.2.0 package
// (cactus/environments/*.py, _harness.py, cactus/agent/grounding.py); tools/eval_rule.py
// is the Python twin used to score upstream libcactus 4.x on the same bytes.
internal sealed record SuiteCase(string Query, string Category, bool Critical, string WantJson, string[] WantKeys);

internal sealed record Suite(string Name, string System, string ToolsJson, IReadOnlyList<SuiteCase> Cases);

internal sealed record SuiteCaseResult(SuiteCase Case, CallResult Result, string[] GotKeys, string? Gated, bool Ok, double WallMs);

internal static class EvalSuites
{
    public static readonly string[] Names = { "smart_home", "media_player", "productivity", "wearable", "kitchen_appliance", "data_capture" };

    public const string Usage = "needle eval <needle3.cact> --suite <name|all> [--threads N] [--json out.json] [--date \"YYYY-MM-DD Ddd HH:MM\"] [--min-confidence F] [--dir tests/eval/environments] [--reasoning N] [--quiet]";

    public static int Run(string modelPath, string[] args)
    {
        var suites = new List<string>();
        var threads = Math.Clamp(Environment.ProcessorCount / 2, 1, 8);
        string? jsonPath = null;
        string? directory = null;
        var date = DateTime.Now.ToString("yyyy-MM-dd ddd HH:mm", CultureInfo.InvariantCulture);
        var minConfidence = 0.0;
        int? reasoning = null;
        var quiet = false;
        try
        {
            for (var i = 0; i < args.Length; i++)
            {
                string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value");
                switch (args[i])
                {
                    case "--suite":
                        suites.AddRange(Next().Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
                        while (i + 1 < args.Length && !args[i + 1].StartsWith("--")) suites.Add(args[++i]);
                        break;
                    case "--threads": threads = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--json": jsonPath = Next(); break;
                    case "--dir": directory = Next(); break;
                    case "--date": date = Next(); break;
                    case "--min-confidence": minConfidence = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--reasoning": reasoning = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--quiet": quiet = true; break;
                    default: throw new ArgumentException($"unknown option {args[i]}");
                }
            }
            if (threads < 1) throw new ArgumentException("needle-cs has no automatic thread count yet (threads 0); pass --threads 1 or more");
            if (suites.Count == 0 || suites.Contains("all")) suites = Names.ToList();
            foreach (var name in suites)
                if (!Names.Contains(name)) throw new ArgumentException($"unknown suite {name}; expected one of {string.Join(", ", Names)} or all");
        }
        catch (Exception e) when (e is ArgumentException or FormatException or OverflowException)
        {
            Console.Error.WriteLine(e.Message);
            Console.Error.WriteLine(Usage);
            return 2;
        }

        var loaded = suites.Select(name => Load(name, directory)).ToList();
        foreach (var suite in loaded)
            if (suite.ToolsJson.Contains("\"format\": \"date") )
                Console.Error.WriteLine($"warning: {suite.Name} has date-format arguments; the package's client-side year grounding is not ported to C# (tools/eval_rule.py has it)");

        using var model = Model.Open(modelPath);
        var json = new MemoryStream();
        using var writer = new Utf8JsonWriter(json, new JsonWriterOptions { Indented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
        writer.WriteStartObject();
        writer.WriteString("schema", "needle.eval-4x/1");
        writer.WriteString("engine", "needle-cs");
        writer.WriteString("model", Path.GetFileName(modelPath));
        writer.WriteNumber("threads", threads);
        writer.WriteString("date", date);
        writer.WriteNumber("min_confidence", minConfidence);
        if (reasoning is { } r) writer.WriteNumber("reasoning_budget", r);
        writer.WriteStartObject("machine");
        writer.WriteString("os", RuntimeInformationText());
        writer.WriteNumber("logical_cores", Environment.ProcessorCount);
        writer.WriteEndObject();
        writer.WriteStartArray("suites");
        var summaries = new List<(string Name, int Passed, int N, int Critical, bool Pass)>();
        foreach (var suite in loaded)
        {
            var system = WithDateFact(suite.System, date);
            var options = new NeedleOptions { Threads = threads, System = system, AutoDate = false };
            if (reasoning is { } budget) options = new NeedleOptions { Threads = threads, System = system, AutoDate = false, ReasoningBudget = budget };
            using var session = new NeedleSession(model, suite.ToolsJson, options);
            var results = new List<SuiteCaseResult>();
            foreach (var item in suite.Cases)
            {
                session.Reset();
                var started = Stopwatch.GetTimestamp();
                var result = session.Complete(item.Query);
                var wall = (Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency;
                results.Add(Score(item, result, minConfidence, wall));
            }
            var passed = results.Count(x => x.Ok);
            var critical = results.Count(x => x.Case.Critical && !x.Ok);
            var need = (int)Math.Round(0.9 * results.Count, MidpointRounding.ToEven);
            var pass = passed >= need && critical == 0;
            summaries.Add((suite.Name, passed, results.Count, critical, pass));
            if (!quiet) Console.Error.WriteLine(RenderSuite(suite.Name, results, passed, critical));
            WriteSuite(writer, suite.Name, results, passed, need, critical, pass);
        }
        writer.WriteEndArray();
        writer.WriteStartObject("overall");
        writer.WriteNumber("passed", summaries.Sum(s => s.Passed));
        writer.WriteNumber("n", summaries.Sum(s => s.N));
        writer.WriteNumber("suites_passed", summaries.Count(s => s.Pass));
        writer.WriteNumber("suites", summaries.Count);
        writer.WriteBoolean("all_passed", summaries.All(s => s.Pass));
        writer.WriteEndObject();
        writer.WriteEndObject();
        writer.Flush();
        if (jsonPath != null) File.WriteAllBytes(jsonPath, json.ToArray());

        Console.Out.WriteLine("| suite | passed | critical fails | verdict |");
        Console.Out.WriteLine("|---|---|---|---|");
        foreach (var s in summaries) Console.Out.WriteLine($"| {s.Name} | {s.Passed}/{s.N} | {s.Critical} | {(s.Pass ? "PASS" : "FAIL")} |");
        Console.Out.WriteLine($"| **all** | {summaries.Sum(s => s.Passed)}/{summaries.Sum(s => s.N)} | {summaries.Sum(s => s.Critical)} | {summaries.Count(s => s.Pass)}/{summaries.Count} suites |");
        return summaries.All(s => s.Pass) ? 0 : 1;
    }

    static string RuntimeInformationText() => System.Runtime.InteropServices.RuntimeInformation.OSDescription + " " + System.Runtime.InteropServices.RuntimeInformation.OSArchitecture;

    static readonly Regex IsoStamp = new(@"\d{4}-\d{2}-\d{2}", RegexOptions.CultureInvariant);
    static readonly Regex DateKey = new(@"(?<![A-Za-z])date:", RegexOptions.CultureInvariant);

    // cactus.agent.grounding._with_date_fact with the clock pinned to `date`.
    public static string WithDateFact(string system, string date)
    {
        if (DateKey.IsMatch(system) || IsoStamp.IsMatch(system)) return system;
        var fact = "date: " + date;
        if (system.Trim().Length == 0) return fact;
        if (system.TrimStart().StartsWith('{')) return system;
        return fact + "; " + system;
    }

    static SuiteCaseResult Score(SuiteCase item, CallResult result, double minConfidence, double wallMs)
    {
        var got = result.FunctionCalls;
        string? gated = null;
        if (got.Count > 0 && (result.Validation.Ungrounded.Count > 0 || result.Validation.Negated)) { got = Array.Empty<FunctionCall>(); gated = "validation"; }
        if (got.Count > 0 && (result.Confidence ?? 0f) < minConfidence) { got = Array.Empty<FunctionCall>(); gated = "confidence"; }
        var keys = got.Select(call => CallKey(call.Name, call.ArgumentsJson)).OrderBy(k => k, StringComparer.Ordinal).ToArray();
        var ok = result.Success && keys.SequenceEqual(item.WantKeys);
        return new SuiteCaseResult(item, result, keys, gated, ok, wallMs);
    }

    // cactus.environments._harness._key: casefolded strings, integral floats as ints, sorted keys.
    public static string CallKey(string name, string argumentsJson)
    {
        var b = new StringBuilder("{\"arguments\":");
        using (var document = JsonDocument.Parse(string.IsNullOrWhiteSpace(argumentsJson) ? "{}" : argumentsJson)) Canonical(b, document.RootElement);
        b.Append(",\"name\":");
        AppendString(b, name.ToLowerInvariant());
        return b.Append('}').ToString();
    }

    static void Canonical(StringBuilder b, JsonElement value)
    {
        switch (value.ValueKind)
        {
            case JsonValueKind.Object:
                b.Append('{');
                var first = true;
                foreach (var property in value.EnumerateObject().OrderBy(p => p.Name, StringComparer.Ordinal))
                {
                    if (!first) b.Append(',');
                    first = false;
                    AppendString(b, property.Name);
                    b.Append(':');
                    Canonical(b, property.Value);
                }
                b.Append('}');
                break;
            case JsonValueKind.Array:
                b.Append('[');
                var index = 0;
                foreach (var item in value.EnumerateArray())
                {
                    if (index++ > 0) b.Append(',');
                    Canonical(b, item);
                }
                b.Append(']');
                break;
            case JsonValueKind.String: AppendString(b, value.GetString()!.ToLowerInvariant()); break;
            case JsonValueKind.Number:
                var number = value.GetDouble();
                b.Append(number == Math.Floor(number) && Math.Abs(number) < 1e18 ? ((long)number).ToString(CultureInfo.InvariantCulture) : number.ToString("R", CultureInfo.InvariantCulture));
                break;
            case JsonValueKind.True: b.Append("true"); break;
            case JsonValueKind.False: b.Append("false"); break;
            default: b.Append("null"); break;
        }
    }

    static void AppendString(StringBuilder b, string s) => b.Append(JsonSerializerString(s));

    static string JsonSerializerString(string s)
    {
        var buffer = new MemoryStream();
        using (var w = new Utf8JsonWriter(buffer, new JsonWriterOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping })) w.WriteStringValue(s);
        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    static Suite Load(string name, string? directory)
    {
        var (caseText, toolsText) = directory != null
            ? (File.ReadAllText(Path.Combine(directory, name + ".json")), File.ReadAllText(Path.Combine(directory, name + ".tools.json")))
            : (Resource($"environments/{name}.json"), Resource($"environments/{name}.tools.json"));
        using var document = JsonDocument.Parse(caseText);
        var root = document.RootElement;
        var cases = new List<SuiteCase>();
        foreach (var item in root.GetProperty("cases").EnumerateArray())
        {
            var calls = item.GetProperty("calls");
            var keys = calls.EnumerateArray().Select(call => CallKey(call.GetProperty("name").GetString()!, call.TryGetProperty("arguments", out var a) ? a.GetRawText() : "{}"))
                .OrderBy(k => k, StringComparer.Ordinal).ToArray();
            cases.Add(new SuiteCase(item.GetProperty("query").GetString()!, item.GetProperty("category").GetString()!,
                item.TryGetProperty("critical", out var critical) && critical.GetBoolean(), calls.GetRawText(), keys));
        }
        return new Suite(name, root.GetProperty("system").GetString()!, toolsText, cases);
    }

    static string Resource(string name)
    {
        using var stream = typeof(EvalSuites).Assembly.GetManifestResourceStream(name)
            ?? throw new InvalidOperationException($"embedded resource {name} is missing");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    static string CallsJson(IReadOnlyList<FunctionCall> calls) =>
        "[" + string.Join(",", calls.Select(c => "{\"name\":" + JsonSerializerString(c.Name) + ",\"arguments\":" + (string.IsNullOrWhiteSpace(c.ArgumentsJson) ? "{}" : c.ArgumentsJson) + "}")) + "]";

    static string RenderSuite(string name, List<SuiteCaseResult> results, int passed, int critical)
    {
        var b = new StringBuilder(name).Append('\n');
        foreach (var r in results.Where(r => !r.Ok))
        {
            b.Append($"FAIL [{r.Case.Category}{(r.Case.Critical ? ", critical" : "")}] {r.Case.Query}\n");
            b.Append($"  want {r.Case.WantJson.Replace("\n", "").Replace(" ", "")}\n");
            b.Append($"  got  {(r.Gated != null ? "[]" : CallsJson(r.Result.FunctionCalls))}");
            if (r.Gated != null) b.Append($" (raw {CallsJson(r.Result.FunctionCalls)}, gated by {r.Gated})");
            if (!r.Result.Success) b.Append($" (error {r.Result.Error})");
            b.Append('\n');
        }
        return b.Append($"{passed}/{results.Count} passed, {critical} critical failures").ToString();
    }

    static void WriteSuite(Utf8JsonWriter w, string name, List<SuiteCaseResult> results, int passed, int need, int critical, bool pass)
    {
        w.WriteStartObject();
        w.WriteString("suite", name);
        w.WriteStartArray("cases");
        foreach (var r in results)
        {
            var c = r.Result;
            w.WriteStartObject();
            w.WriteString("query", r.Case.Query);
            w.WriteString("category", r.Case.Category);
            w.WriteBoolean("critical", r.Case.Critical);
            w.WritePropertyName("want"); w.WriteRawValue(r.Case.WantJson);
            w.WritePropertyName("raw_calls"); w.WriteRawValue(CallsJson(c.FunctionCalls));
            w.WritePropertyName("suppressed_calls"); w.WriteRawValue(CallsJson(c.SuppressedCalls));
            w.WritePropertyName("got"); w.WriteRawValue(r.Gated != null ? "[]" : CallsJson(c.FunctionCalls));
            if (r.Gated != null) w.WriteString("gated", r.Gated); else w.WriteNull("gated");
            w.WriteStartArray("ungrounded");
            foreach (var u in c.Validation.Ungrounded) w.WriteStringValue(u);
            w.WriteEndArray();
            w.WriteBoolean("negation", c.Validation.Negated);
            w.WriteBoolean("repaired", c.Validation.Repaired);
            if (c.Confidence is { } confidence) w.WriteNumber("confidence", Math.Round(confidence, 4)); else w.WriteNull("confidence");
            w.WriteBoolean("success", c.Success);
            if (c.Error != null) w.WriteString("error", c.Error); else w.WriteNull("error");
            w.WriteNumber("prefill_tokens", c.PrefillTokens);
            w.WriteNumber("decode_tokens", c.DecodeTokens);
            w.WriteNumber("engine_ms", Math.Round(c.TotalMs, 2));
            w.WriteNumber("wall_ms", Math.Round(r.WallMs, 2));
            w.WriteBoolean("ok", r.Ok);
            w.WriteEndObject();
        }
        w.WriteEndArray();
        w.WriteNumber("passed", passed);
        w.WriteNumber("n", results.Count);
        w.WriteNumber("need", need);
        w.WriteNumber("critical_failures", critical);
        w.WriteStartArray("critical_failed");
        foreach (var r in results.Where(r => r.Case.Critical && !r.Ok)) w.WriteStringValue(r.Case.Query);
        w.WriteEndArray();
        w.WriteBoolean("pass", pass);
        w.WriteEndObject();
    }
}
