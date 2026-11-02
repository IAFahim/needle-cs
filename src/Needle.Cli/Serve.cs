using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using Needle;

namespace Needle.Cli;

/// <summary>
/// N70: `needle serve &lt;model.cact&gt; &lt;tools.json&gt; [options]`.
///
/// NativeAOT inference child for runtimes with nothing fast (Mono-only Unity
/// LTS users until the Burst provider lands). One JSON request per stdin line,
/// one JSON reply per stdout line; diagnostics go to stderr so stdout stays a
/// clean reply stream. The parent (Unity) kills the child on exit; the child
/// also exits on its own when stdin closes (EOF).
///
/// Request line: {"id":1,"type":"query","text":"..."}
///   type: "query" (or "q"), "tool_result" (or "tool"), "reset", "ping".
///   "text" aliases "query"/"result" are accepted. Missing type means query.
/// Reply line: {"id":1,&lt;CallResult.ToJson() fields...&gt;} for query/tool_result,
///   {"id":N,"success":true,"error":null,"type":"pong"} for ping,
///   {"id":N,"success":true,"error":null,"type":"reset"} for reset,
///   {"id":&lt;id|null&gt;,"success":false,"error":"...","type":"error",
///     "function_calls":[]} for malformed requests.
/// Unknown "id" shape echoes as null; numeric and string ids round-trip.
///
/// Overhead budget (acceptance): steady-state round trip (client wall clock
/// minus the server "total_time_ms" field) must stay within 1 ms of in-process
/// net10 latency. Measure with: ping for pure pipe cost, then a warm query
/// and subtract its total_time_ms. Keep per-request work to one parse, one
/// session turn on the already-primed session, and one emit: no per-request
/// allocation beyond the reply string itself.
///
/// Wiring (Program.cs, one line): ["serve", var model, var tools, .. var rest]
///   =&gt; Serve.Run(model, tools, rest), plus a usage line.
/// </summary>
internal static class Serve
{
    public static int Run(string modelPath, string toolsPath, string[] options)
    {
        var threads = Math.Clamp(Environment.ProcessorCount / 2, 1, 8);
        var context = 2048;
        var maxReasoning = 96;
        var maxCalls = 384;
        string? system = Environment.GetEnvironmentVariable("NEEDLE_SYSTEM");
        var systemSet = false;
        var autoDate = true;
        var topK = 5;
        var stateless = false;
        var warmup = false;
        string? prefixPath = null;
        for (var i = 0; i < options.Length; i++)
        {
            switch (options[i])
            {
                case "--threads":
                    if (!TakeInt(options, ref i, out threads) || threads < 1) return Usage();
                    break;
                case "--context":
                    if (!TakeInt(options, ref i, out context) || context < 1) return Usage();
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
                    systemSet = true;
                    break;
                case "--no-date":
                    autoDate = false;
                    break;
                case "--no-retrieval":
                    topK = int.MaxValue;
                    break;
                case "--stateless":
                    stateless = true;
                    break;
                case "--warmup":
                    warmup = true;
                    break;
                case "--prefix":
                    if (i + 1 >= options.Length) return Usage();
                    prefixPath = options[++i];
                    break;
                case "-h":
                    return Usage();
                default:
                    Console.Error.WriteLine($"unknown option: {options[i]}");
                    return Usage();
            }
        }

        if (!systemSet && string.IsNullOrEmpty(system)) system = null;

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
            AutoDate = autoDate,
            RetrievalTopK = topK,
        };

        Model model;
        NeedleSession session;
        var load = Stopwatch.StartNew();
        try
        {
            model = Model.Open(modelPath);
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"cannot open model {modelPath}: {e.Message}");
            return 1;
        }
        byte[]? prefix = null;
        if (prefixPath != null)
        {
            // A shipped prefix bakes a date-free system prompt (AutoDate=false,
            // plan.md §10.5); fail fast instead of tripping prefix validation.
            if (autoDate) { model.Dispose(); Console.Error.WriteLine("cannot use --prefix without --no-date (primed prefixes bake AutoDate=false)"); return 1; }
            try { prefix = File.ReadAllBytes(prefixPath); }
            catch (Exception e) { model.Dispose(); Console.Error.WriteLine($"cannot read prefix {prefixPath}: {e.Message}"); return 1; }
        }
        try
        {
            session = prefix != null
                ? new NeedleSession(model, toolsJson, prefix, sessionOptions)
                : new NeedleSession(model, toolsJson, sessionOptions);
        }
        catch (Exception e)
        {
            model.Dispose();
            Console.Error.WriteLine($"cannot prime session: {e.Message}");
            return 1;
        }
        using (model)
        using (session)
        {
            Console.Error.WriteLine($"ready in {load.Elapsed.TotalMilliseconds:F1} ms, prefix {session.ContextUsed} tokens (threads {threads}, context {context}, stateless {stateless})");
            if (warmup)
            {
                var started = Stopwatch.GetTimestamp();
                session.WarmUp();
                Console.Error.WriteLine($"warmup in {(Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency:F1} ms");
            }

            var debug = Environment.GetEnvironmentVariable("NEEDLE_DEBUG") != null;
            string? line;
            while ((line = Console.In.ReadLine()) != null)
            {
                if (line.Length == 0) continue;
                var reply = HandleLine(session, line, stateless, debug);
                Console.Out.WriteLine(reply);
                Console.Out.Flush();
            }
        }
        return 0;
    }

    static string HandleLine(NeedleSession session, string line, bool stateless, bool debug)
    {
        JsonValue request;
        try
        {
            request = JsonValue.Parse(line.AsSpan());
        }
        catch (Exception e)
        {
            return ErrorReply("null", "malformed request: " + e.Message);
        }
        if (request.Kind != JsonKind.Object) return ErrorReply("null", "request must be a JSON object");

        var idJson = RenderId(request.Get("id"));
        var type = request.GetString("type") ?? request.GetString("op") ?? "query";
        type = type.Trim().ToLowerInvariant();
        var text = request.GetString("text") ?? request.GetString("query") ?? request.GetString("result") ?? "";

        try
        {
            switch (type)
            {
                case "ping":
                    return "{\"id\":" + idJson + ",\"success\":true,\"error\":null,\"type\":\"pong\"}";
                case "reset":
                    session.Reset();
                    return "{\"id\":" + idJson + ",\"success\":true,\"error\":null,\"type\":\"reset\"}";
                case "query":
                case "q":
                case "":
                    if (stateless) session.Reset();
                    return WithId(idJson, Complete(session, text, toolResult: false, debug, idJson));
                case "tool_result":
                case "toolresult":
                case "tool":
                case "t":
                    return WithId(idJson, Complete(session, text, toolResult: true, debug, idJson));
                default:
                    return ErrorReply(idJson, "unknown type \"" + type + "\"");
            }
        }
        catch (Exception e)
        {
            return ErrorReply(idJson, "inference failed: " + e.Message);
        }
    }

    static string Complete(NeedleSession session, string text, bool toolResult, bool debug, string idJson)
    {
        var started = Stopwatch.GetTimestamp();
        var result = toolResult ? session.CompleteWithToolResult(text) : session.Complete(text);
        if (debug)
            Console.Error.WriteLine($"req {idJson} {(toolResult ? "tool" : "query")} total {result.TotalMs:F1} ms wall {(Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency:F1} ms");
        return result.ToJson();
    }

    static string WithId(string idJson, string callJson)
    {
        // CallResult.ToJson emits {"success":...}; splice the id in front.
        if (callJson.Length > 0 && callJson[0] == '{') return "{\"id\":" + idJson + "," + callJson.Substring(1);
        return "{\"id\":" + idJson + ",\"success\":false,\"error\":\"internal reply\",\"type\":\"error\",\"function_calls\":[]}";
    }

    static string ErrorReply(string idJson, string message)
    {
        var b = new StringBuilder("{\"id\":");
        b.Append(idJson).Append(",\"success\":false,\"error\":");
        JsonValue.WriteString(b, message);
        b.Append(",\"type\":\"error\",\"function_calls\":[]}");
        return b.ToString();
    }

    static string RenderId(JsonValue? id)
    {
        switch (id)
        {
            case { Kind: JsonKind.Number } n: return n.Text.Length > 0 ? n.Text : "null";
            case { Kind: JsonKind.String } s:
                var b = new StringBuilder();
                JsonValue.WriteString(b, s.Text);
                return b.ToString();
            case { Kind: JsonKind.Bool } v: return v.Boolean ? "true" : "false";
            default: return "null";
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
        Console.Error.WriteLine("needle serve <model.cact> <tools.json> [--threads N] [--context N] [--max-reasoning N] [--max-calls N] [--system TEXT] [--no-date] [--no-retrieval] [--stateless] [--warmup] [--prefix <out.prefix>] (--prefix requires --no-date)");
        return 2;
    }
}
