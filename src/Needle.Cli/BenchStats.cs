using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Needle;

namespace Needle.Cli;

internal readonly record struct Spread(double Best, double Median)
{
    public static Spread Of(IReadOnlyList<double> values)
    {
        if (values.Count == 0) return new Spread(double.NaN, double.NaN);
        var sorted = new double[values.Count];
        for (var i = 0; i < sorted.Length; i++) sorted[i] = values[i];
        Array.Sort(sorted);
        return new Spread(sorted[0], (sorted[sorted.Length / 2] + sorted[(sorted.Length - 1) / 2]) / 2);
    }

    public static Spread Of(IReadOnlyList<long> values)
    {
        var converted = new double[values.Count];
        for (var i = 0; i < converted.Length; i++) converted[i] = values[i];
        return Of(converted);
    }

    public static Spread OfThroughput(IReadOnlyList<double> values)
    {
        var spread = Of(values);
        if (values.Count == 0) return spread;
        var peak = double.MinValue;
        foreach (var value in values) peak = Math.Max(peak, value);
        return spread with { Best = peak };
    }

    public bool IsFinite => double.IsFinite(Best) && double.IsFinite(Median);
}

internal static class BenchTiming
{
    public static double Milliseconds(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    public static double Nanoseconds(long ticks) => ticks * 1e9 / Stopwatch.Frequency;

    public static Spread NanosecondsPerCall(Action body, int rounds = 9, double roundMilliseconds = 12)
    {
        var iterations = 1;
        for (;;)
        {
            var started = Stopwatch.GetTimestamp();
            for (var i = 0; i < iterations; i++) body();
            var elapsed = Milliseconds(Stopwatch.GetTimestamp() - started);
            if (elapsed >= roundMilliseconds / 4 || iterations >= 1 << 24) break;
            iterations *= 4;
        }
        var samples = new double[rounds];
        for (var round = 0; round < rounds; round++)
        {
            var started = Stopwatch.GetTimestamp();
            for (var i = 0; i < iterations; i++) body();
            samples[round] = Nanoseconds(Stopwatch.GetTimestamp() - started) / iterations;
        }
        return Spread.Of(samples);
    }
}

internal sealed class BenchJson
{
    readonly StringBuilder text = new();
    readonly List<bool> hasItems = new() { false };
    bool afterKey;

    public override string ToString() => text.ToString();

    void Separate()
    {
        if (afterKey) { afterKey = false; return; }
        if (hasItems[^1]) text.Append(',');
        hasItems[^1] = true;
    }

    public BenchJson BeginObject()
    {
        Separate();
        text.Append('{');
        hasItems.Add(false);
        return this;
    }

    public BenchJson BeginArray()
    {
        Separate();
        text.Append('[');
        hasItems.Add(false);
        return this;
    }

    public BenchJson EndObject()
    {
        text.Append('}');
        hasItems.RemoveAt(hasItems.Count - 1);
        return this;
    }

    public BenchJson EndArray()
    {
        text.Append(']');
        hasItems.RemoveAt(hasItems.Count - 1);
        return this;
    }

    public BenchJson Key(string name)
    {
        Separate();
        JsonValue.WriteString(text, name);
        text.Append(':');
        afterKey = true;
        return this;
    }

    public BenchJson Value(double value)
    {
        Separate();
        text.Append(double.IsFinite(value) ? value.ToString("0.####", CultureInfo.InvariantCulture) : "null");
        return this;
    }

    public BenchJson Value(long value)
    {
        Separate();
        text.Append(value.ToString(CultureInfo.InvariantCulture));
        return this;
    }

    public BenchJson Value(string value)
    {
        Separate();
        JsonValue.WriteString(text, value);
        return this;
    }

    public BenchJson Value(bool value)
    {
        Separate();
        text.Append(value ? "true" : "false");
        return this;
    }

    public BenchJson Null()
    {
        Separate();
        text.Append("null");
        return this;
    }

    public BenchJson Field(string name, double value) => Key(name).Value(value);

    public BenchJson Field(string name, long value) => Key(name).Value(value);

    public BenchJson Field(string name, string value) => Key(name).Value(value);

    public BenchJson Field(string name, bool value) => Key(name).Value(value);

    public BenchJson Field(string name, Spread value) =>
        Key(name).BeginObject().Field("best", value.Best).Field("median", value.Median).EndObject();
}

internal static class BenchMachine
{
    public static string SimdTier => Simd.UseNeon ? "neon" : Simd.UseAvx ? "avx2" : "scalar";

    public static void Write(BenchJson json) =>
        json.Key("machine").BeginObject()
            .Field("os", RuntimeInformation.OSDescription)
            .Field("arch", RuntimeInformation.ProcessArchitecture.ToString())
            .Field("logical_cores", Environment.ProcessorCount)
            .Field("runtime", RuntimeInformation.FrameworkDescription)
            .Field("simd", SimdTier)
            .Field("native_alloc_counters", Mem.CountersEnabled)
            .EndObject();

    public static string Describe() =>
        $"{RuntimeInformation.ProcessArchitecture} {RuntimeInformation.FrameworkDescription} simd={SimdTier} cores={Environment.ProcessorCount} alloc-counters={(Mem.CountersEnabled ? "on" : "off")}";
}
