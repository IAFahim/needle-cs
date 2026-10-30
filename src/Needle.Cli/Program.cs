using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using Needle;
using Needle.Cli;

return args switch
{
    ["verify", var model, var golden] => Verify(model, golden),
    ["bench", var model, .. var options] => BenchKernels.Run(model, options),
    ["bench-e2e", var model, var tools, .. var options] => BenchE2E.Run(model, tools, options),
    ["generate", var model, var text, .. var count] => Generate(model, text, count.Length > 0 ? int.Parse(count[0]) : 80),
    ["transcribe", var model, var wav, .. var options] => Transcribe(model, wav, options),
    ["stream", var model, var wav] => StreamClip(model, wav),
    ["voice", var needle, var whistle, var tools, var wav] => Voice(needle, whistle, tools, wav),
    ["embed", var model, var text] => EmbedText(model, text),
    ["eval", var model, var dataset, .. var options] => Eval.Run(model, dataset, options),
    ["call", var model, var tools, .. var queries] => Call(model, tools, queries),
    ["run", var model, var tools, var query, .. var rest] => Run(model, tools, query, rest),
    _ => Usage(),
};

static int Usage()
{
    Console.Error.WriteLine("needle verify <model.cact> <golden-dir>");
    Console.Error.WriteLine("needle run <model.cact> <tools.json> <query> [repeat]");
    Console.Error.WriteLine("needle bench <needle3.cact> [--threads 4] [--json]");
    Console.Error.WriteLine("needle bench-e2e <needle3.cact> <tools.json> [--threads 1,2,4] [--reps N] [--sets golden|game|all] [--profile] [--json] [--max-managed-bytes N] [--max-native-allocs N]");
    Console.Error.WriteLine(Eval.Usage);
    Console.Error.WriteLine("needle call <needle3.cact> <tools.json> [--no-date] [--no-retrieval] <query>... ('tool:<json>' feeds a tool result)");
    Console.Error.WriteLine("needle transcribe <whistle.cact> <clip.wav> [--language xx] [--words] [--keywords a,b] [--repeat n]");
    Console.Error.WriteLine("needle stream <whistle.cact> <clip.wav>");
    Console.Error.WriteLine("needle voice <needle3.cact> <whistle.cact> <tools.json> <clip.wav>");
    Console.Error.WriteLine("needle embed <needle3.cact> <text>");
    Console.Error.WriteLine("needle generate <needle3.cact> <raw prompt> [tokens]");
    return 2;
}

static int Run(string modelPath, string toolsPath, string query, string[] rest)
{
    var repeat = rest.Length > 0 ? int.Parse(rest[0]) : 1;
    var tools = File.ReadAllBytes(toolsPath);
    var queryBytes = System.Text.Encoding.UTF8.GetBytes(query);
    var output = new byte[16384];
    var load = Stopwatch.StartNew();
    using var model = Model.Open(modelPath);
    var openMs = load.Elapsed.TotalMilliseconds;
    var threads = int.Parse(Environment.GetEnvironmentVariable("NEEDLE_THREADS") ?? Math.Clamp(Environment.ProcessorCount / 2, 1, 8).ToString());
    using var caller = new ToolCaller(model, tools, threads: threads);
    var primeMs = load.Elapsed.TotalMilliseconds - openMs;
    Console.Error.WriteLine($"open={openMs:F1}ms prefix={caller.PrefixLength} tokens in {primeMs:F1}ms");
    Completion result = default;
    var totals = new List<double>();
    var perToken = new List<double>();
    for (var i = 0; i < repeat; i++)
    {
        result = caller.Complete(queryBytes, output, 256);
        var prefill = result.PrefillTicks * 1000.0 / Stopwatch.Frequency;
        var decode = result.DecodeTicks * 1000.0 / Stopwatch.Frequency;
        totals.Add(prefill + decode);
        perToken.Add(decode / Math.Max(1, result.GeneratedTokens));
        Console.Error.WriteLine($"query {i}: prefill {result.PromptTokens} tok {prefill:F2}ms | decode {result.GeneratedTokens} tok {decode:F2}ms = {decode / Math.Max(1, result.GeneratedTokens):F3}ms/tok ({result.GeneratedTokens * 1000.0 / decode:F0} tok/s) | total {prefill + decode:F2}ms");
    }
    totals.Sort();
    perToken.Sort();
    Console.Error.WriteLine($"summary: best {totals[0]:F2}ms median {totals[totals.Count / 2]:F2}ms | decode best {perToken[0]:F3}ms/tok ({1000.0 / perToken[0]:F0} tok/s) median {perToken[perToken.Count / 2]:F3}ms/tok");
    Console.WriteLine(System.Text.Encoding.UTF8.GetString(output, 0, result.Bytes));
    return 0;
}

static int Verify(string modelPath, string goldenDir)
{
    var ids = MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(goldenDir, "ids.bin"))).ToArray();
    var expected = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(goldenDir, "logits.bin"))).ToArray();
    using var model = Model.Open(modelPath);
    using var engine = new Engine(model, 512, 256);
    engine.Observer = (layer, stream) => CompareLayer(layer, stream, goldenDir, ids.Length);
    var watch = Stopwatch.StartNew();
    engine.Feed(ids);
    watch.Stop();
    var actual = engine.Logits;
    double dot = 0, na = 0, nb = 0, worst = 0;
    var argA = 0;
    var argB = 0;
    for (var i = 0; i < expected.Length; i++)
    {
        dot += actual[i] * expected[i];
        na += actual[i] * actual[i];
        nb += expected[i] * expected[i];
        worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]));
        if (actual[i] > actual[argA]) argA = i;
        if (expected[i] > expected[argB]) argB = i;
    }
    Console.WriteLine($"tokens={ids.Length} ms={watch.Elapsed.TotalMilliseconds:F1} cosine={dot / Math.Sqrt(na * nb):F6} maxAbs={worst:F4} argmax={argA} expected={argB}");
    return argA == argB ? 0 : 1;
}

static void CompareLayer(int layer, ReadOnlySpan<float> stream, string goldenDir, int tokens)
{
    var name = layer < 0 ? "x0.bin" : $"stream{layer:D2}.bin";
    var expected = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(goldenDir, name)));
    var lanes = layer < 0 ? 1 : 4;
    double dot = 0, na = 0, nb = 0, worst = 0;
    for (var t = 0; t < tokens; t++)
        for (var i = 0; i < lanes * 768; i++)
        {
            double a = stream[t * 3072 + i];
            double b = expected[t * lanes * 768 + i];
            dot += a * b; na += a * a; nb += b * b;
            worst = Math.Max(worst, Math.Abs(a - b));
        }
    Console.WriteLine($"layer {layer,2}: cosine={dot / Math.Sqrt(na * nb):F6} maxAbs={worst:F4} rms={Math.Sqrt(nb / (tokens * lanes * 768)):F3}");
}

static int Call(string modelPath, string toolsPath, string[] rest)
{
    var threads = int.Parse(Environment.GetEnvironmentVariable("NEEDLE_THREADS") ?? Math.Clamp(Environment.ProcessorCount / 2, 1, 8).ToString());
    var system = Environment.GetEnvironmentVariable("NEEDLE_SYSTEM");
    var autoDate = !rest.Contains("--no-date");
    var topK = rest.Contains("--no-retrieval") ? int.MaxValue : 5;
    var queries = rest.Where(q => q != "--no-date" && q != "--no-retrieval").ToArray();
    using var model = Model.Open(modelPath);
    var started = Stopwatch.GetTimestamp();
    using var session = new NeedleSession(model, File.ReadAllText(toolsPath), new NeedleOptions { Threads = threads, System = system, AutoDate = autoDate, RetrievalTopK = topK });
    Console.Error.WriteLine($"ready in {(Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency:F1} ms, prefix {session.ContextUsed} tokens");
    foreach (var query in queries)
    {
        var result = query.StartsWith("tool:") ? session.CompleteWithToolResult(query[5..]) : session.Complete(query);
        Console.WriteLine(result.ToJson());
        if (Environment.GetEnvironmentVariable("NEEDLE_DEBUG") != null) Console.Error.WriteLine($"  head {result.HeadConfidence:F3} decode {result.DecodeConfidence:F3}");
    }
    return 0;
}

static int Transcribe(string modelPath, string wavPath, string[] options)
{
    var threads = int.Parse(Environment.GetEnvironmentVariable("NEEDLE_THREADS") ?? Math.Clamp(Environment.ProcessorCount / 2, 1, 8).ToString());
    string? language = null;
    var words = false;
    var repeat = 1;
    var keywords = new List<string>();
    for (var i = 0; i < options.Length; i++)
        switch (options[i])
        {
            case "--language": language = options[++i]; break;
            case "--words": words = true; break;
            case "--repeat": repeat = int.Parse(options[++i]); break;
            case "--keywords": keywords.AddRange(options[++i].Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)); break;
        }
    var pcm = Wav.Read(wavPath);
    using var model = Model.Open(modelPath);
    var load = Stopwatch.GetTimestamp();
    using var whistle = new Whistle(model, threads);
    Console.Error.WriteLine($"ready in {(Stopwatch.GetTimestamp() - load) * 1000.0 / Stopwatch.Frequency:F1} ms, audio {pcm.Length / 16000.0:F1} s");
    Transcription result = null!;
    for (var i = 0; i < repeat; i++)
    {
        var started = Stopwatch.GetTimestamp();
        result = whistle.Transcribe(pcm, language, keywords, words);
        Console.Error.WriteLine($"pass {i}: {(Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency:F1} ms (encode {result.EncodeMs:F1}, ttft {result.TimeToFirstTokenMs:F1}, {result.DecodeTps:F0} tok/s)");
    }
    Console.WriteLine($"[{result.Language ?? "none"}] {result.Text}");
    if (result.Words != null)
        foreach (var w in result.Words) Console.WriteLine($"  {w.Start,6:F2} - {w.End,5:F2}  {w.Text,-18} {w.Probability:F2}");
    return 0;
}

static int StreamClip(string modelPath, string wavPath)
{
    var pcm = Wav.Read(wavPath);
    using var model = Model.Open(modelPath);
    using var whistle = new Whistle(model, Math.Clamp(Environment.ProcessorCount / 2, 1, 8));
    var stream = whistle.Stream();
    var committed = new List<string>();
    for (var at = 0; at < pcm.Length; at += 16000)
    {
        var step = stream.Process(pcm.AsSpan(at, Math.Min(16000, pcm.Length - at)));
        if (step.Text.Length > 0) committed.Add(step.Text);
        Console.WriteLine($"{step.Received,5:F1}s  pass {step.PassMs,5:F0} ms  | {string.Join(" ", committed)} \u001b[90m{step.Pending}\u001b[0m");
    }
    var tail = stream.Stop();
    if (tail.Text.Length > 0) committed.Add(tail.Text);
    Console.WriteLine($"final [{tail.Language}]: {string.Join(" ", committed)}");
    return 0;
}

static int Voice(string needlePath, string whistlePath, string toolsPath, string wavPath)
{
    var threads = int.Parse(Environment.GetEnvironmentVariable("NEEDLE_THREADS") ?? Math.Clamp(Environment.ProcessorCount / 2, 1, 8).ToString());
    using var needle = Model.Open(needlePath);
    using var whistle = Model.Open(whistlePath);
    using var voice = new VoiceSession(needle, whistle, File.ReadAllText(toolsPath), new NeedleOptions { Threads = threads });
    var pcm = Wav.Read(wavPath);
    var started = Stopwatch.GetTimestamp();
    var result = voice.Complete(pcm);
    Console.Error.WriteLine($"audio {pcm.Length / 16000.0:F1} s -> tool call in {(Stopwatch.GetTimestamp() - started) * 1000.0 / Stopwatch.Frequency:F1} ms");
    Console.WriteLine(result.ToJson());
    return 0;
}

static int EmbedText(string modelPath, string text)
{
    using var model = Model.Open(modelPath);
    using var session = new NeedleSession(model, "[]", new NeedleOptions { Threads = 4 });
    var vector = session.Embed(text);
    Console.WriteLine($"{vector.Length} floats, first: {string.Join(" ", vector.Take(6).Select(v => v.ToString("F4")))}");
    return 0;
}

static int Generate(string modelPath, string text, int count)
{
    using var model = Model.Open(modelPath);
    using var engine = new Engine(model, 2048, 64, 4);
    using var tokenizer = new Tokenizer(model, 1 << 16);
    var bytes = System.Text.Encoding.UTF8.GetBytes(text.Replace("\\n", "\n"));
    var ids = new int[bytes.Length + 8];
    ids[0] = tokenizer.BeginId;
    var n = 1 + tokenizer.Encode(bytes, ids.AsSpan(1));
    engine.Feed(ids.AsSpan(0, n));
    var piece = new byte[64];
    var output = new System.Text.StringBuilder();
    for (var i = 0; i < count; i++)
    {
        var next = engine.Greedy();
        if (tokenizer.IsStop(next)) break;
        output.Append(System.Text.Encoding.UTF8.GetString(piece, 0, tokenizer.Append(next, piece, i == 0)));
        engine.Feed(new[] { next });
    }
    Console.WriteLine(output.ToString());
    return 0;
}
