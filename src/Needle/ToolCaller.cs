using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Needle;

public readonly record struct Completion(int Bytes, int PromptTokens, int GeneratedTokens, long PrefillTicks, long DecodeTicks);

public sealed unsafe class ToolCaller : IDisposable
{
    static ReadOnlySpan<byte> PrefixHead => "<|im_start|>user\n<tools>"u8;
    static ReadOnlySpan<byte> PrefixTail => "</tools>"u8;
    static ReadOnlySpan<byte> QueryHead => "\n"u8;
    static ReadOnlySpan<byte> QueryTail => "<|im_end|>\n<|im_start|>assistant\n"u8;

    readonly Engine engine;
    readonly Tokenizer tokenizer;
    readonly int maxQueryBytes;
    readonly byte* text;
    readonly int* ids;

    public ToolCaller(Model model, ReadOnlySpan<byte> toolsJson, int contextLength = 1024, int maxQueryBytes = 4096, int threads = 1)
    {
        engine = new Engine(model, contextLength, 64, threads);
        tokenizer = new Tokenizer(model, Math.Max(toolsJson.Length + 64, maxQueryBytes + 64));
        this.maxQueryBytes = maxQueryBytes;
        var capacity = Math.Max(toolsJson.Length + 64, maxQueryBytes + 64);
        text = (byte*)Mem.AlignedAlloc((nuint)capacity, 64);
        ids = (int*)Mem.AlignedAlloc((nuint)capacity * sizeof(int), 64);
        Prime(toolsJson, capacity);
    }

    public int PrefixLength => engine.Length;

    void Prime(ReadOnlySpan<byte> toolsJson, int capacity)
    {
        var prefix = new Span<byte>(text, capacity);
        PrefixHead.CopyTo(prefix);
        toolsJson.CopyTo(prefix[PrefixHead.Length..]);
        PrefixTail.CopyTo(prefix[(PrefixHead.Length + toolsJson.Length)..]);
        var length = PrefixHead.Length + toolsJson.Length + PrefixTail.Length;
        var span = new Span<int>(ids, capacity);
        span[0] = tokenizer.BeginId;
        var count = 1 + tokenizer.Encode(prefix[..length], span[1..]);
        engine.Reset();
        engine.Feed(span[..count], computeLogits: false);
        engine.Mark();
    }

    public Completion Complete(ReadOnlySpan<byte> query, Span<byte> output, int maxNewTokens = 256)
    {
        if (query.Length > maxQueryBytes) throw new ArgumentException("query exceeds maxQueryBytes");
        var started = Stopwatch.GetTimestamp();
        engine.Rewind();
        var buffer = new Span<byte>(text, maxQueryBytes + 64);
        QueryHead.CopyTo(buffer);
        query.CopyTo(buffer[QueryHead.Length..]);
        QueryTail.CopyTo(buffer[(QueryHead.Length + query.Length)..]);
        var length = QueryHead.Length + query.Length + QueryTail.Length;
        var count = tokenizer.Encode(buffer[..length], new Span<int>(ids, maxQueryBytes + 64), leadingSpace: false);
        engine.Feed(new ReadOnlySpan<int>(ids, count));
        var prefilled = Stopwatch.GetTimestamp();

        var written = 0;
        var first = true;
        var generated = 0;
        for (var step = 0; step < maxNewTokens; step++)
        {
            var next = engine.Greedy();
            if (tokenizer.IsStop(next)) break;
            if (written + 16 > output.Length) break;
            written += tokenizer.Append(next, output[written..], first);
            first = false;
            ids[0] = next;
            generated++;
            engine.Feed(new ReadOnlySpan<int>(ids, 1));
        }
        return new Completion(written, count, generated, prefilled - started, Stopwatch.GetTimestamp() - prefilled);
    }

    public void Dispose()
    {
        engine.Dispose();
        tokenizer.Dispose();
        Mem.AlignedFree(text);
        Mem.AlignedFree(ids);
    }
}
