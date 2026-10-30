using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using Xunit;

namespace Needle.Tests;

public sealed class WhistleStageTests : IDisposable
{
    static readonly string WhistleModel = Path.Combine(Assets.Root, "ref/hf/whistle/whistle.cact");
    static string GoldenDirectory(string clip) => Path.Combine(Assets.Root, "ref/golden/whistle", clip);

    static readonly (string Stage, int Width, double MeanCosine)[] Stages =
    {
        ("mel", 80, 0.999999),
        ("stem", 512, 0.9995),
        ("layer0", 2048, 0.9997),
        ("layer3", 2048, 0.9997),
        ("layer7", 2048, 0.9997),
        ("encoder", 512, 0.9995),
        ("memory", 512, 0.9995),
        ("cross_key0", 48, 0.9995),
        ("cross_value_last", 64, 0.9990),
        ("logits_bos", 0, 0.9999),
        ("logits_tag", 0, 0.9999),
    };

    readonly Model model;
    readonly Whistle whistle;

    public WhistleStageTests()
    {
        if (!File.Exists(WhistleModel)) throw new InvalidOperationException($"missing {WhistleModel}; run tools/fetch.sh");
        model = Model.Open(WhistleModel);
        whistle = new Whistle(model, 4);
    }

    public void Dispose()
    {
        whistle.Dispose();
        model.Dispose();
    }

    static float[] ReadFloats(string path)
    {
        if (!File.Exists(path)) throw new InvalidOperationException($"missing {path}; run tools/golden_whistle.py");
        return MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray();
    }

    static double MeanRowCosine(float[] actual, float[] expected, int width)
    {
        Assert.Equal(expected.Length, actual.Length);
        var rows = expected.Length / width;
        var sum = 0.0;
        for (var r = 0; r < rows; r++)
        {
            double dot = 0, a = 0, b = 0;
            for (var i = r * width; i < (r + 1) * width; i++)
            {
                dot += (double)actual[i] * expected[i];
                a += (double)actual[i] * actual[i];
                b += (double)expected[i] * expected[i];
            }
            sum += dot / Math.Sqrt(a * b);
        }
        return sum / rows;
    }

    [Theory]
    [InlineData("record")]
    [InlineData("hotword")]
    public void StagesMatchTheNumpyReference(string clip)
    {
        var directory = GoldenDirectory(clip);
        var pcm = ReadFloats(Path.Combine(directory, "pcm.f32"));
        var tag = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "meta.json"))).RootElement.GetProperty("tag").GetInt32();
        var seen = new Dictionary<string, float[]>();
        whistle.Observer = (stage, values) => seen[stage] = values.ToArray();
        try { whistle.Observe(pcm, whistle.Languages[tag - 8192]); }
        finally { whistle.Observer = null; }
        foreach (var (stage, width, floor) in Stages)
        {
            var expected = ReadFloats(Path.Combine(directory, stage + ".f32"));
            var cosine = MeanRowCosine(seen[stage], expected, width == 0 ? expected.Length : width);
            Assert.True(cosine >= floor, $"{clip} {stage}: mean cosine {cosine:F6} is below {floor}");
        }
    }
}
