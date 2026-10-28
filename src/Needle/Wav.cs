using System;
using System.IO;

namespace Needle;

public static class Wav
{
    public const int SampleRate = 16000;

    public static float[] Read(string path) => Decode(File.ReadAllBytes(path));

    public static float[] Decode(ReadOnlySpan<byte> data)
    {
        if (data.Length < 12 || !data[..4].SequenceEqual("RIFF"u8) || !data.Slice(8, 4).SequenceEqual("WAVE"u8)) throw new InvalidDataException("not a WAV file");
        int tag = 0, channels = 1, rate = SampleRate, width = 2;
        ReadOnlySpan<byte> samples = default;
        for (var at = 12; at + 8 <= data.Length;)
        {
            var size = (int)Math.Min(BitConverter.ToUInt32(data.Slice(at + 4, 4)), (uint)(data.Length - at - 8));
            var body = data.Slice(at + 8, size);
            if (data.Slice(at, 4).SequenceEqual("fmt "u8))
            {
                tag = BitConverter.ToUInt16(body[..2]);
                channels = BitConverter.ToUInt16(body.Slice(2, 2));
                rate = (int)BitConverter.ToUInt32(body.Slice(4, 4));
                width = BitConverter.ToUInt16(body.Slice(12, 2)) / channels;
                if (tag == 0xFFFE) tag = BitConverter.ToUInt16(body.Slice(24, 2));
            }
            else if (data.Slice(at, 4).SequenceEqual("data"u8)) samples = body;
            at += 8 + size + (size & 1);
        }
        if (samples.IsEmpty) throw new InvalidDataException("WAV has no data chunk");
        var frames = samples.Length / (width * channels);
        var mono = new float[frames];
        for (var f = 0; f < frames; f++)
        {
            var sum = 0.0;
            for (var c = 0; c < channels; c++) sum += Sample(samples, (f * channels + c) * width, tag, width);
            mono[f] = (float)(sum / channels);
        }
        return rate == SampleRate ? mono : Resample(mono, rate, SampleRate);
    }

    static double Sample(ReadOnlySpan<byte> s, int at, int tag, int width) => (tag, width) switch
    {
        (1, 1) => (s[at] - 128) / 128.0,
        (1, 2) => BitConverter.ToInt16(s.Slice(at, 2)) / 32768.0,
        (1, 3) => (((s[at] | s[at + 1] << 8 | s[at + 2] << 16) << 8) >> 8) / 8388608.0,
        (1, 4) => BitConverter.ToInt32(s.Slice(at, 4)) / 2147483648.0,
        (3, 4) => BitConverter.ToSingle(s.Slice(at, 4)),
        (3, 8) => BitConverter.ToDouble(s.Slice(at, 8)),
        _ => throw new InvalidDataException($"unsupported WAV format {tag}/{width}"),
    };

    public static float[] Resample(float[] input, int from, int to)
    {
        if (from == to || input.Length == 0) return input;
        const int HalfWidth = 24;
        var ratio = (double)to / from;
        var scale = Math.Min(1.0, ratio);
        var radius = HalfWidth / scale;
        var output = new float[(int)(input.Length * ratio)];
        for (var n = 0; n < output.Length; n++)
        {
            var center = n / ratio;
            double sum = 0, weight = 0;
            for (var i = Math.Max(0, (int)Math.Ceiling(center - radius)); i <= Math.Min(input.Length - 1, (int)Math.Floor(center + radius)); i++)
            {
                var t = (i - center) * scale;
                var sinc = t == 0 ? 1.0 : Math.Sin(Math.PI * t) / (Math.PI * t);
                var kernel = sinc * (0.5 + 0.5 * Math.Cos(Math.PI * (i - center) / radius));
                sum += input[i] * kernel;
                weight += kernel;
            }
            output[n] = (float)(weight == 0 ? 0 : sum / weight);
        }
        return output;
    }
}
