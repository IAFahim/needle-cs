using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Needle;

namespace Needle.Cli;

internal readonly record struct KernelRow(string Name, string Shape, int Tokens, int Threads, Spread Nanoseconds, double Bytes, double Macs)
{
    public double GigabytesPerSecond => Bytes > 0 ? Bytes / Nanoseconds.Best : double.NaN;

    public double GigaMacsPerSecond => Macs > 0 ? Macs / Nanoseconds.Best : double.NaN;
}

internal sealed record BenchKernelOptions(int[] PoolThreads, bool Json)
{
    public static BenchKernelOptions Parse(string[] args)
    {
        var options = new BenchKernelOptions(new[] { 4 }, false);
        for (var i = 0; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value");
            options = args[i] switch
            {
                "--threads" => options with { PoolThreads = Next().Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).Where(t => t > 1).ToArray() },
                "--json" => options with { Json = true },
                var unknown => throw new ArgumentException($"unknown option {unknown}"),
            };
        }
        return options;
    }
}

internal static unsafe class BenchKernels
{
    static readonly int[] TokenCounts = { 1, 2, 4, 8, 16, 64 };
    static readonly int[] PrepareTokenCounts = { 1, 4, 64 };
    static readonly (string Name, string Prefix)[] TrieProbes =
    {
        ("start", ""),
        ("tool_name", "[{\"name\":\""),
        ("property_key", "[{\"name\":\"give_item\",\"arguments\":{"),
        ("string_start", "[{\"name\":\"say\",\"arguments\":{\"text\":\""),
        ("free_string", "[{\"name\":\"say\",\"arguments\":{\"text\":\"hello"),
        ("after_string", "[{\"name\":\"say\",\"arguments\":{\"text\":\"hello\""),
        ("number", "[{\"name\":\"move_to\",\"arguments\":{\"x\":"),
        ("integer", "[{\"name\":\"give_item\",\"arguments\":{\"item\":\"gold\",\"count\":"),
        ("enum_string", "[{\"name\":\"set_formation\",\"arguments\":{\"formation\":\""),
    };

    const string RepresentativeCall = "[{\"name\":\"give_item\",\"arguments\":{\"item\":\"potion\",\"count\":3,\"to\":\"Lyra\"}},{\"name\":\"move_to\",\"arguments\":{\"x\":10,\"y\":20}}]";

    public static int Run(string modelPath, string[] args)
    {
        BenchKernelOptions options;
        try { options = BenchKernelOptions.Parse(args); }
        catch (Exception e) when (e is ArgumentException or FormatException or OverflowException)
        {
            Console.Error.WriteLine(e.Message);
            return 2;
        }
        using var model = Model.Open(modelPath);
        var rows = Measure(model, options, options.Json ? null : Console.Out);
        if (options.Json) Console.Out.WriteLine(ToJson(modelPath, rows));
        return 0;
    }

    public static List<KernelRow> Measure(Model model, BenchKernelOptions options, TextWriter? live)
    {
        var rows = new List<KernelRow>();
        void Add(KernelRow row)
        {
            rows.Add(row);
            live?.WriteLine(Format(row));
        }
        live?.WriteLine($"bench kernels | {BenchMachine.Describe()}");
        live?.WriteLine(Header);
        foreach (var shape in MatMulShapes(model))
            foreach (var tokens in TokenCounts)
                Add(MatMulSerial(shape.Name, shape.Weights, tokens));
        foreach (var (cols, bits) in new[] { (3072, 4), (768, 2), (768, 4) })
            foreach (var tokens in PrepareTokenCounts)
                Add(Prepare(cols, bits, tokens));
        Add(Mlp(model));
        Add(SinkhornNormalize());
        foreach (var row in Vocabulary(model.VocabSize)) Add(row);
        foreach (var row in TrieAllow(model)) Add(row);
        foreach (var threads in options.PoolThreads)
        {
            using var pool = new Pool(threads);
            foreach (var tasks in new[] { 4, 12 }) Add(Barrier(pool, tasks));
            foreach (var shape in MatMulShapes(model))
                foreach (var tokens in TokenCounts)
                    Add(MatMulPooled(pool, shape.Name, shape.Weights, tokens));
        }
        return rows;
    }

    static (string Name, Cq[] Weights)[] MatMulShapes(Model model)
    {
        var g = model.Geometry;
        Cq[] PerLayer(Func<int, Cq> select) => Enumerable.Range(0, g.Layers).Select(select).ToArray();
        var shapes = new List<(string, Cq[])>
        {
            ("q_proj", PerLayer(l => model.Blocks[l].Q)),
            ("k_proj", PerLayer(l => model.Blocks[l].K)),
            ("v_proj", PerLayer(l => model.Blocks[l].V)),
            ("gate_proj", PerLayer(l => model.Blocks[l].Gate)),
            ("out_proj", PerLayer(l => model.Blocks[l].Out)),
            ("hyper_phi_pre", PerLayer(l => model.Hyper.PhiPre.Slice(l * 4, 4))),
            ("hyper_phi_post", PerLayer(l => model.Hyper.PhiPost.Slice(l * 4, 4))),
            ("hyper_phi_res", PerLayer(l => model.Hyper.PhiRes.Slice(l * 16, 16))),
            ("engram_key", Enumerable.Range(0, g.SiteCount).Select(s => model.Sites[s].Key).ToArray()),
            ("engram_value", Enumerable.Range(0, g.SiteCount).Select(s => model.Sites[s].Value).ToArray()),
            ("head", new[] { model.Embedding }),
        };
        return shapes.ToArray();
    }

    static string ShapeLabel(in Cq w) => $"{w.Rows}x{w.Cols} {w.Bits}b";

    static void FillSine(float* x, int count)
    {
        for (var i = 0; i < count; i++) x[i] = MathF.Sin(i * 0.37f);
    }

    static KernelRow MatMulSerial(string name, Cq[] weights, int tokens)
    {
        var w = weights[0];
        var arena = new Arena(32 << 20);
        var act = new Activations(ref arena, tokens, w.Cols);
        var x = arena.Take<float>((nuint)(tokens * w.Cols));
        var y = arena.Take<float>((nuint)(tokens * w.Rows));
        FillSine(x, tokens * w.Cols);
        Quant.Prepare(x, tokens, w.Cols, w.Bits, act);
        var next = 0;
        var spread = BenchTiming.NanosecondsPerCall(() =>
        {
            Quant.MatMul(weights[next], act, tokens, y, w.Rows);
            next = next + 1 == weights.Length ? 0 : next + 1;
        });
        arena.Free();
        return new KernelRow("matmul " + name, ShapeLabel(w), tokens, 1, spread, (double)w.Rows * w.RowBytes, (double)w.Rows * w.Cols * tokens);
    }

    static KernelRow MatMulPooled(Pool pool, string name, Cq[] weights, int tokens)
    {
        var w = weights[0];
        var arena = new Arena(32 << 20);
        var act = new Activations(ref arena, tokens, w.Cols);
        var x = arena.Take<float>((nuint)(tokens * w.Cols));
        var y = arena.Take<float>((nuint)(tokens * w.Rows));
        FillSine(x, tokens * w.Cols);
        Quant.Prepare(x, tokens, w.Cols, w.Bits, act);
        var batches = new MatMulBatch[weights.Length];
        for (var i = 0; i < batches.Length; i++)
        {
            batches[i] = new MatMulBatch(act, tokens);
            batches[i].Add(weights[i], y, w.Rows);
        }
        var next = 0;
        var spread = BenchTiming.NanosecondsPerCall(() =>
        {
            fixed (MatMulBatch* batch = &batches[next]) MatMulBatch.Run(pool, batch);
            next = next + 1 == batches.Length ? 0 : next + 1;
        });
        arena.Free();
        return new KernelRow("matmul " + name, ShapeLabel(w), tokens, pool.Threads, spread, (double)w.Rows * w.RowBytes, (double)w.Rows * w.Cols * tokens);
    }

    static KernelRow Prepare(int cols, int bits, int tokens)
    {
        var arena = new Arena(8 << 20);
        var act = new Activations(ref arena, tokens, cols);
        var x = arena.Take<float>((nuint)(tokens * cols));
        FillSine(x, tokens * cols);
        var spread = BenchTiming.NanosecondsPerCall(() => Quant.Prepare(x, tokens, cols, bits, act));
        arena.Free();
        return new KernelRow("prepare", $"{cols} cols {bits}b", tokens, 1, spread, (double)tokens * cols * sizeof(float), 0);
    }

    static KernelRow Mlp(Model model)
    {
        var g = model.Geometry;
        var dim = g.Dim;
        var width = g.HadaRows * g.HadaCols;
        var residual = (float*)Mem.AllocZeroed((nuint)dim * sizeof(float));
        var scratch = (float*)Mem.AllocZeroed((nuint)HadamardMlp.ScratchFloats(dim, width) * sizeof(float));
        FillSine(residual, dim);
        var rows = g.HadaRows;
        var cols = g.HadaCols;
        var layer = 0;
        var spread = BenchTiming.NanosecondsPerCall(() =>
        {
            var batch = new MlpBatch
            {
                Weights = &model.Blocks[layer].Mlp, PreNorm = model.Blocks[layer].PreHada, PermA = model.Hyper.PermA, PermB = model.Hyper.PermB,
                Residual = residual, Dim = dim, Rows = rows, Cols = cols, Scale = 0f,
            };
            HadamardMlp.Apply(&batch, residual, scratch);
            layer = layer + 1 == g.Layers ? 0 : layer + 1;
        });
        Mem.Free(residual);
        Mem.Free(scratch);
        return new KernelRow("hadamard_mlp_apply", $"{rows}x{cols} one token", 1, 1, spread, 0, 3.0 * rows * cols * (rows + cols));
    }

    static KernelRow SinkhornNormalize()
    {
        var logits = stackalloc float[16];
        var mix = stackalloc float[16];
        for (var i = 0; i < 16; i++) logits[i] = MathF.Cos(i);
        var spread = BenchTiming.NanosecondsPerCall(() => Needle.Sinkhorn.Normalize(logits, mix));
        return new KernelRow("sinkhorn_normalize", "4x4 20 iterations", 1, 1, spread, 0, 0);
    }

    static KernelRow[] Vocabulary(int vocab)
    {
        var source = (float*)Mem.Alloc((nuint)vocab * sizeof(float));
        var scratch = (float*)Mem.Alloc((nuint)vocab * sizeof(float));
        for (var i = 0; i < vocab; i++) source[i] = MathF.Sin(i * 0.11f) * 5;
        var bytes = (double)vocab * sizeof(float);
        var softmax = BenchTiming.NanosecondsPerCall(() =>
        {
            Buffer.MemoryCopy(source, scratch, (long)bytes, (long)bytes);
            Ops.Softmax(scratch, vocab);
        });
        var argmax = BenchTiming.NanosecondsPerCall(() => Ops.ArgMax(source, vocab));
        Mem.Free(source);
        Mem.Free(scratch);
        return new[]
        {
            new KernelRow("softmax_with_copy", $"{vocab} logits", 1, 1, softmax, bytes, 0),
            new KernelRow("argmax", $"{vocab} logits", 1, 1, argmax, bytes, 0),
        };
    }

    static KernelRow Barrier(Pool pool, int tasks)
    {
        var spread = BenchTiming.NanosecondsPerCall(() => pool.Run(&EmptyTask, null, tasks), rounds: 9, roundMilliseconds: 6);
        return new KernelRow("pool_run_barrier", $"{tasks} empty tasks", 1, pool.Threads, spread, 0, 0);
    }

    static void EmptyTask(void* context, int index)
    {
    }

    static List<KernelRow> TrieAllow(Model model)
    {
        var rows = new List<KernelRow>();
        using var tokenizer = new Tokenizer(model, 1 << 16);
        using var trie = new TokenTrie(tokenizer, model.VocabSize);
        using var grammar = new Grammar(new ToolSet(BenchPrompts.GameToolsJson).Tools);
        var program = grammar.Program;
        var mask = (uint*)Mem.AllocZeroed((nuint)trie.Words * sizeof(uint));
        foreach (var (name, prefix) in TrieProbes)
        {
            var state = StateAfter(grammar, prefix, name);
            var spread = BenchTiming.NanosecondsPerCall(() => trie.Allow(program, state, mask), roundMilliseconds: 20);
            rows.Add(new KernelRow("token_trie_allow", $"{name} ({AllowedCount(mask, trie.Words)} allowed)", 1, 1, spread, 0, 0));
        }
        var walk = StatesAlong(grammar, RepresentativeCall);
        var whole = BenchTiming.NanosecondsPerCall(() =>
        {
            foreach (var state in walk) trie.Allow(program, state, mask);
        }, roundMilliseconds: 40);
        rows.Add(new KernelRow("token_trie_allow", $"mean of {walk.Length} byte positions", 1, 1, new Spread(whole.Best / walk.Length, whole.Median / walk.Length), 0, 0));
        Mem.Free(mask);
        return rows;
    }

    static int AllowedCount(uint* mask, int words)
    {
        var allowed = 0;
        for (var i = 0; i < words; i++) allowed += System.Numerics.BitOperations.PopCount(mask[i]);
        return allowed;
    }

    static GrammarState[] StatesAlong(Grammar grammar, string text)
    {
        var states = new List<GrammarState>();
        var state = Grammar.Start(grammar.Program);
        foreach (var b in System.Text.Encoding.UTF8.GetBytes(text))
        {
            states.Add(state);
            if (!Grammar.Advance(grammar.Program, &state, b)) throw new InvalidOperationException("grammar rejected the representative call");
        }
        return states.ToArray();
    }

    static GrammarState StateAfter(Grammar grammar, string prefix, string label)
    {
        var state = Grammar.Start(grammar.Program);
        foreach (var b in System.Text.Encoding.UTF8.GetBytes(prefix))
            if (!Grammar.Advance(grammar.Program, &state, b))
                throw new InvalidOperationException($"grammar rejected benchmark prefix for {label}");
        return state;
    }

    const string Header = "kernel                      shape                              T thr   best us  median us     GB/s   GMAC/s";

    static string Format(in KernelRow r)
    {
        var gb = double.IsFinite(r.GigabytesPerSecond) ? r.GigabytesPerSecond.ToString("F1") : "-";
        var gmac = double.IsFinite(r.GigaMacsPerSecond) ? r.GigaMacsPerSecond.ToString("F1") : "-";
        return $"{r.Name,-27} {r.Shape,-34} {r.Tokens,3} {r.Threads,3} {r.Nanoseconds.Best / 1000,9:F2} {r.Nanoseconds.Median / 1000,10:F2} {gb,8} {gmac,8}";
    }

    public static string ToJson(string modelPath, IReadOnlyList<KernelRow> rows)
    {
        var json = new BenchJson().BeginObject().Field("schema", "needle.bench-kernels/1").Field("model", modelPath);
        BenchMachine.Write(json);
        json.Key("rows").BeginArray();
        foreach (var r in rows)
        {
            json.BeginObject()
                .Field("kernel", r.Name)
                .Field("shape", r.Shape)
                .Field("tokens", r.Tokens)
                .Field("threads", r.Threads)
                .Field("ns", r.Nanoseconds)
                .Field("gb_per_s", r.GigabytesPerSecond)
                .Field("gmac_per_s", r.GigaMacsPerSecond)
                .EndObject();
        }
        return json.EndArray().EndObject().ToString();
    }
}
