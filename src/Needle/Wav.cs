using System;
using System.IO;
using System.Threading;

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
                if (body.Length < 16) throw new InvalidDataException("WAV fmt chunk is shorter than 16 bytes");
                tag = BitConverter.ToUInt16(body[..2]);
                channels = BitConverter.ToUInt16(body.Slice(2, 2));
                rate = (int)BitConverter.ToUInt32(body.Slice(4, 4));
                if (channels == 0) throw new InvalidDataException("WAV has zero channels");
                width = BitConverter.ToUInt16(body.Slice(12, 2)) / channels;
                if (tag == 0xFFFE)
                {
                    if (body.Length < 26) throw new InvalidDataException("WAV extensible fmt chunk is shorter than 26 bytes");
                    tag = BitConverter.ToUInt16(body.Slice(24, 2));
                }
            }
            else if (data.Slice(at, 4).SequenceEqual("data"u8)) samples = body;
            at += 8 + size + (size & 1);
        }
        if (samples.IsEmpty) throw new InvalidDataException("WAV has no data chunk");
        if (width < 1 || width > 8) throw new InvalidDataException($"unsupported WAV sample width {width} bytes");
        if (rate < 1000 || rate > 768000) throw new InvalidDataException($"unsupported WAV sample rate {rate}");
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
        var output = new float[ResampledLength(input.Length, from, to)];
        Resample(input, from, to, output);
        return output;
    }

    public static int ResampledLength(int samples, int from, int to)
    {
        if (from < 1) throw new ArgumentOutOfRangeException(nameof(from));
        if (to < 1) throw new ArgumentOutOfRangeException(nameof(to));
        if (samples < 0) throw new ArgumentOutOfRangeException(nameof(samples));
        return from == to ? samples : (int)(samples * ((double)to / from));
    }

    public static int Resample(ReadOnlySpan<float> input, int from, int to, Span<float> output)
    {
        var length = ResampledLength(input.Length, from, to);
        if (output.Length < length) throw new ArgumentException($"output holds {output.Length} samples, {length} needed", nameof(output));
        if (from == to) { input.CopyTo(output); return length; }
        var kernel = Polyphase.For(from, to);
        if (kernel == null) { Direct(input, from, to, output.Slice(0, length)); return length; }
        kernel.Run(input, output.Slice(0, length));
        return length;
    }

    const int HalfWidth = 24;

    static double Tap(double distance, double scale, double radius)
    {
        var t = distance * scale;
        var sinc = t == 0 ? 1.0 : Math.Sin(Math.PI * t) / (Math.PI * t);
        return sinc * (0.5 + 0.5 * Math.Cos(Math.PI * distance / radius));
    }

    static void Direct(ReadOnlySpan<float> input, int from, int to, Span<float> output)
    {
        var ratio = (double)to / from;
        var scale = Math.Min(1.0, ratio);
        var radius = HalfWidth / scale;
        for (var n = 0; n < output.Length; n++)
        {
            var center = n / ratio;
            double sum = 0, weight = 0;
            for (var i = Math.Max(0, (int)Math.Ceiling(center - radius)); i <= Math.Min(input.Length - 1, (int)Math.Floor(center + radius)); i++)
            {
                var kernel = Tap(i - center, scale, radius);
                sum += input[i] * kernel;
                weight += kernel;
            }
            output[n] = (float)(weight == 0 ? 0 : sum / weight);
        }
    }

    sealed class Polyphase
    {
        const int MaxCoefficients = 1 << 20;
        static Polyphase? last;

        readonly int from;
        readonly int to;
        readonly int up;
        readonly int down;
        readonly int taps;
        readonly int[] starts;
        readonly double[] weights;
        readonly double[] totals;

        Polyphase(int from, int to, int up, int down, int taps, int[] starts, double[] weights, double[] totals)
        {
            this.from = from;
            this.to = to;
            this.up = up;
            this.down = down;
            this.taps = taps;
            this.starts = starts;
            this.weights = weights;
            this.totals = totals;
        }

        public static Polyphase? For(int from, int to)
        {
            var cached = Volatile.Read(ref last);
            if (cached != null && cached.from == from && cached.to == to) return cached;
            var common = Gcd(from, to);
            var up = to / common;
            var down = from / common;
            var scale = Math.Min(1.0, (double)to / from);
            var radius = HalfWidth / scale;
            var taps = (int)Math.Ceiling(2 * radius) + 2;
            if ((long)up * taps > MaxCoefficients) return null;
            var starts = new int[up];
            var weights = new double[up * taps];
            var totals = new double[up];
            for (var phase = 0; phase < up; phase++)
            {
                var fraction = (double)phase / up;
                var first = (int)Math.Ceiling(fraction - radius);
                var lastTap = (int)Math.Floor(fraction + radius);
                starts[phase] = first;
                var total = 0.0;
                for (var k = 0; k < taps; k++)
                {
                    var offset = first + k;
                    var w = offset <= lastTap ? Tap(offset - fraction, scale, radius) : 0.0;
                    weights[phase * taps + k] = w;
                    total += w;
                }
                totals[phase] = total;
            }
            var built = new Polyphase(from, to, up, down, taps, starts, weights, totals);
            Volatile.Write(ref last, built);
            return built;
        }

        public void Run(ReadOnlySpan<float> input, Span<float> output)
        {
            var count = input.Length;
            for (var n = 0; n < output.Length; n++)
            {
                var position = (long)n * down;
                var whole = (int)(position / up);
                var phase = (int)(position - (long)whole * up);
                var first = whole + starts[phase];
                var row = weights.AsSpan(phase * taps, taps);
                double sum = 0;
                if (first >= 0 && first + taps <= count)
                {
                    var source = input.Slice(first, taps);
                    for (var k = 0; k < row.Length; k++) sum += source[k] * row[k];
                    var total = totals[phase];
                    output[n] = (float)(total == 0 ? 0 : sum / total);
                    continue;
                }
                double weight = 0;
                var lo = Math.Max(0, -first);
                var hi = Math.Min(taps, count - first);
                for (var k = lo; k < hi; k++)
                {
                    sum += input[first + k] * row[k];
                    weight += row[k];
                }
                output[n] = (float)(weight == 0 ? 0 : sum / weight);
            }
        }

        static int Gcd(int a, int b)
        {
            while (b != 0) (a, b) = (b, a % b);
            return a;
        }
    }
}
