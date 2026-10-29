using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using Needle;

var threads = int.Parse(Environment.GetEnvironmentVariable("THREADS") ?? "4");
using var model = Model.Open("/home/user/needle-cs/ref/hf/whistle/whistle.cact");
using var whistle = new Whistle(model, threads);
var mode = args[0];
if (mode == "dump")
{
    var wav = args[1];
    var dir = args[2];
    Directory.CreateDirectory(dir);
    var pcm = Wav.Read(wav);
    foreach (var old in Directory.GetFiles(dir)) File.Delete(old);
    whistle.Observer = (stage, values) =>
    {
        using var stream = new FileStream(Path.Combine(dir, stage + ".f32"), FileMode.Append);
        stream.Write(MemoryMarshal.AsBytes(values));
    };
    whistle.Observe(pcm, "en");
    Console.WriteLine($"{wav}: samples {pcm.Length}");
}
else if (mode == "text")
{
    for (var i = 1; i < args.Length; i++)
    {
        var pcm = Wav.Read(args[i]);
        var r = whistle.Transcribe(pcm);
        Console.WriteLine($"{Path.GetFileName(args[i])}: {r.Text}");
    }
}
else if (mode == "time")
{
    var pcm = Wav.Read(args[1]);
    var best = double.MaxValue;
    for (var i = 0; i < 6; i++)
    {
        var sw = Stopwatch.StartNew();
        whistle.Embed(pcm);
        if (i > 0) best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
    }
    Console.WriteLine($"encode best {best:F1} ms");
}
