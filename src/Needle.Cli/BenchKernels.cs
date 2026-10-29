using System;
using System.Diagnostics;
using System.Linq;
using Needle;

namespace Needle.Cli;

internal static unsafe class BenchKernels
{
    const int WarmupSweeps = 5;

    public static int Run(string modelPath, string[] options)
    {
        var threadCounts = options.Length > 0 ? options[0].Split(',').Select(int.Parse).ToArray() : new[] { 1, 4 };
        var sweeps = options.Length > 1 ? int.Parse(options[1]) : 300;
        using var model = Model.Open(modelPath);
        var arena = new Arena(96 << 20);
        var act2 = new Activations(ref arena, 1, model.Geometry.Dim);
        var act4 = new Activations(ref arena, 1, model.Geometry.Dim);
        var x = arena.Take<float>((nuint)model.Geometry.Dim);
        for (var i = 0; i < model.Geometry.Dim; i++) x[i] = MathF.Sin(i * 0.37f) * 1.5f;
        Quant.Prepare(x, 1, model.Geometry.Dim, 2, act2);
        Quant.Prepare(x, 1, model.Geometry.Dim, 4, act4);
        var layers = model.Geometry.Layers;

        Console.WriteLine($"simd={(Simd.Enabled ? "on" : "off")} sweeps={sweeps}  (time = one sweep over all layers of the stage, best / median)");
        Console.WriteLine($"{"stage",-26} {"threads",7} {"MB",7} {"best us",9} {"med us",9} {"best GB/s",10} {"med GB/s",9}");

        var stages = new (string Name, int Count, Func<int, MatMulBatch> Build, Activations Act)[]
        {
            ("qkvg 20 layers", layers, l => Batch(act2, model.Blocks[l].Q, model.Blocks[l].K, model.Blocks[l].V, model.Blocks[l].Gate), act2),
            ("out 20 layers", layers, l => Batch(act2, model.Blocks[l].Out), act2),
            ("engram key+value 5 sites", model.Geometry.SiteCount, s => Batch(act2, model.Sites[s].Key, model.Sites[s].Value), act2),
            ("head 8192x768 4b", 1, _ => Batch(act4, model.Embedding), act4),
        };

        foreach (var threads in threadCounts)
        {
            using var pool = new Pool(threads);
            foreach (var stage in stages)
            {
                var batches = (MatMulBatch*)Mem.Alloc((nuint)(stage.Count * sizeof(MatMulBatch)));
                var bytes = 0L;
                for (var i = 0; i < stage.Count; i++)
                {
                    batches[i] = stage.Build(i);
                    for (var j = 0; j < batches[i].Count; j++)
                    {
                        var w = batches[i].Jobs.At(j)->W;
                        bytes += (long)w.Rows * w.RowBytes;
                    }
                }
                var y = arena.Take<float>(16384);
                for (var i = 0; i < stage.Count; i++)
                    for (var j = 0; j < batches[i].Count; j++)
                        batches[i].Jobs.At(j)->Y = y + (j * 4096 % 12288);
                for (var s = 0; s < WarmupSweeps; s++)
                    for (var i = 0; i < stage.Count; i++) MatMulBatch.Run(pool, batches + i);
                var times = new double[sweeps];
                for (var s = 0; s < sweeps; s++)
                {
                    var start = Stopwatch.GetTimestamp();
                    for (var i = 0; i < stage.Count; i++) MatMulBatch.Run(pool, batches + i);
                    times[s] = (Stopwatch.GetTimestamp() - start) * 1e6 / Stopwatch.Frequency;
                }
                Array.Sort(times);
                Report(stage.Name, threads, bytes, times);
                Mem.Free(batches);
            }
        }

        Console.WriteLine();
        Console.WriteLine($"{"single matrix, cache-hot",-26} {"threads",7} {"MB",7} {"best us",9} {"med us",9} {"best GB/s",10} {"med GB/s",9}");
        var shapes = new (string Name, Cq W, Activations Act)[]
        {
            ("q_proj 576x768 2b", model.Blocks[5].Q, act2),
            ("out_proj 768x768 2b", model.Blocks[5].Out, act2),
            ("head 8192x768 4b", model.Embedding, act4),
        };
        foreach (var (name, w, act) in shapes)
        {
            var y = arena.Take<float>((nuint)w.Rows);
            for (var i = 0; i < 200; i++) Quant.MatMul(w, act, 1, y, w.Rows);
            var probe = Stopwatch.GetTimestamp();
            for (var i = 0; i < 20; i++) Quant.MatMul(w, act, 1, y, w.Rows);
            var passMicroseconds = (Stopwatch.GetTimestamp() - probe) * 1e6 / Stopwatch.Frequency / 20;
            var inner = Math.Max(1, (int)(150 / passMicroseconds));
            var times = new double[sweeps * 10];
            for (var s = 0; s < times.Length; s++)
            {
                var start = Stopwatch.GetTimestamp();
                for (var i = 0; i < inner; i++) Quant.MatMul(w, act, 1, y, w.Rows);
                times[s] = (Stopwatch.GetTimestamp() - start) * 1e6 / Stopwatch.Frequency / inner;
            }
            Array.Sort(times);
            Report(name, 1, (long)w.Rows * w.RowBytes, times);
        }
        return 0;
    }

    static MatMulBatch Batch(in Activations act, params Cq[] matrices)
    {
        var batch = new MatMulBatch(act, 1);
        foreach (var w in matrices) batch.Add(w, null, w.Rows);
        return batch;
    }

    static void Report(string name, int threads, long bytes, double[] sorted)
    {
        var best = sorted[0];
        var median = sorted[sorted.Length / 2];
        Console.WriteLine($"{name,-26} {threads,7} {bytes / 1e6,7:F2} {best,9:F1} {median,9:F1} {bytes / best / 1e3,10:F1} {bytes / median / 1e3,9:F1}");
    }
}
