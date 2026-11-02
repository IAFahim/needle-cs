using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Xunit;

namespace Needle.Tests.Conformance
{
    struct XorShift64
    {
        ulong state;

        public XorShift64(ulong seed)
        {
            state = seed != 0 ? seed : 0x9E3779B97F4A7C15uL;
        }

        public ulong Next()
        {
            var x = state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            state = x;
            return x;
        }

        public int NextInt(int lo, int hi)
        {
            return lo + (int)(Next() % (ulong)(hi - lo));
        }

        public float NextFloat(float scale)
        {
            return ((Next() >> 11) * (1f / 9007199254740992f) - 0.5f) * 2f * scale;
        }
    }

    static unsafe class ConformanceInputs
    {
        public static void FillUniform(float* x, int n, ref XorShift64 rng, float scale)
        {
            for (var i = 0; i < n; i++)
            {
                x[i] = rng.NextFloat(scale);
            }
        }

        public static void FillEdges(float* x, int n, ref XorShift64 rng, float scale)
        {
            for (var i = 0; i < n; i++)
            {
                x[i] = rng.NextFloat(scale);
            }
            for (var i = 0; i < n; i += 7)
            {
                x[i] = 0f;
            }
            for (var i = 3; i < n; i += 11)
            {
                x[i] = 1e-38f * (i % 2 == 0 ? 1f : -1f);
            }
            for (var i = 5; i < n; i += 13)
            {
                x[i] = (i % 26 < 13 ? 0.5f : -0.5f) * scale;
            }
        }

        public static void FillSByte(sbyte* x, int n, ref XorShift64 rng, bool extremes)
        {
            for (var i = 0; i < n; i++)
            {
                x[i] = (sbyte)rng.NextInt(-128, 128);
            }
            if (extremes && n > 0)
            {
                x[0] = -128;
                if (n > 1)
                {
                    x[1] = 127;
                }
                for (var i = 2; i < n; i += 5)
                {
                    x[i] = (i % 2 == 0) ? (sbyte)-128 : (sbyte)127;
                }
            }
        }

        public static void FillBytes(byte* x, int n, ref XorShift64 rng, bool extremes)
        {
            for (var i = 0; i < n; i++)
            {
                x[i] = (byte)rng.NextInt(0, 256);
            }
            if (extremes && n >= 2)
            {
                x[0] = 0x00;
                x[1] = 0xFF;
            }
        }

        public static ushort HalfBits(float value)
        {
            var bits = (uint)BitConverter.SingleToInt32Bits(value);
            var sign = (bits >> 16) & 0x8000u;
            var magnitude = bits & 0x7FFFFFFFu;
            if (magnitude >= 0x47800000u)
            {
                return (ushort)(sign | 0x7BFFu);
            }
            if (magnitude <= 0x38000000u)
            {
                return (ushort)sign;
            }
            var rounded = (magnitude >> 13) + ((magnitude >> 12) & 1u);
            return (ushort)(sign | (rounded - 0x1C000u));
        }
    }

    static unsafe class ConformanceAssert
    {
        public static void Finite(float* x, int n, string what)
        {
            for (var i = 0; i < n; i++)
            {
                Assert.False(float.IsNaN(x[i]) || float.IsInfinity(x[i]), what + " [" + i + "]");
            }
        }

        public static void BitExact(float* expected, float* actual, int n, string what)
        {
            for (var i = 0; i < n; i++)
            {
                var e = expected[i] == 0f ? 0f : expected[i];
                var a = actual[i] == 0f ? 0f : actual[i];
                Assert.True(e == a, what + " [" + i + "]: " + e + " != " + a);
            }
        }

        public static void BitExact(sbyte* expected, sbyte* actual, int n, string what)
        {
            for (var i = 0; i < n; i++)
            {
                Assert.True(expected[i] == actual[i], what + " [" + i + "]: " + expected[i] + " != " + actual[i]);
            }
        }

        public static void Close(float* expected, float* actual, int n, float relative, string what)
        {
            var magnitude = 1f;
            for (var i = 0; i < n; i++)
            {
                magnitude = MathF.Max(magnitude, MathF.Abs(expected[i]));
            }
            for (var i = 0; i < n; i++)
            {
                var delta = actual[i] - expected[i];
                var bound = relative * magnitude;
                Assert.True(delta >= -bound && delta <= bound, what + " [" + i + "]: " + delta + " bound " + bound);
            }
        }

        public static double Cosine(float* a, float* b, int n)
        {
            double dot = 0, left = 0, right = 0;
            for (var i = 0; i < n; i++)
            {
                dot += (double)a[i] * b[i];
                left += (double)a[i] * a[i];
                right += (double)b[i] * b[i];
            }
            if (left == 0 && right == 0)
            {
                return 1;
            }
            if (left == 0 || right == 0)
            {
                return 0;
            }
            return dot / Math.Sqrt(left * right);
        }

        public static int ArgMax(float* x, int n)
        {
            var best = 0;
            for (var i = 1; i < n; i++)
            {
                if (x[i] > x[best])
                {
                    best = i;
                }
            }
            return best;
        }

        public static void Parity(float* expected, float* actual, int n, string what)
        {
            Finite(actual, n, what);
            var cosine = Cosine(expected, actual, n);
            Assert.True(cosine >= 0.99995, what + " cosine " + cosine + " < 0.99995");
        }

        public static void ParityWithArgMax(float* expected, float* actual, int n, string what)
        {
            Parity(expected, actual, n, what);
            var top = ArgMax(expected, n);
            var runner = top == 0 ? 1 : 0;
            for (var i = 0; i < n; i++)
            {
                if (i != top && expected[i] > expected[runner])
                {
                    runner = i;
                }
            }
            if (n > 1 && expected[top] - expected[runner] > 1e-5f)
            {
                Assert.Equal(top, ArgMax(actual, n));
            }
        }
    }

    static unsafe class ConformanceDump
    {
        static readonly object gate = new object();

        public static void Dump(string name, void* data, int bytes)
        {
            var dir = Environment.GetEnvironmentVariable("CONFORMANCE_DUMP");
            if (dir == null || bytes <= 0)
            {
                return;
            }
            var safe = new StringBuilder(name.Length);
            foreach (var c in name)
            {
                safe.Append(char.IsLetterOrDigit(c) || c == '_' || c == '-' ? c : '_');
            }
            var file = safe.ToString();
            var manifest = Path.Combine(dir, "manifest.json");
            lock (gate)
            {
                Directory.CreateDirectory(dir);
                using (var stream = new FileStream(Path.Combine(dir, file + ".bin"), FileMode.Create, FileAccess.Write))
                {
                    stream.Write(new ReadOnlySpan<byte>(data, bytes));
                }
                var hash = 0xCBF29CE484222325uL;
                var raw = new ReadOnlySpan<byte>(data, bytes);
                for (var i = 0; i < raw.Length; i++)
                {
                    hash ^= raw[i];
                    hash *= 0x100000001B3uL;
                }
                var entry = "{\"name\":\"" + file + "\",\"bytes\":" + bytes + ",\"fnv1a\":\"" + hash.ToString("x16") + "\"}";
                string json;
                if (File.Exists(manifest))
                {
                    var old = File.ReadAllText(manifest);
                    var at = old.LastIndexOf(']');
                    if (at < 0)
                    {
                        json = "{\"entries\":[" + entry + "]}";
                    }
                    else if (old.Contains("\"" + file + "\""))
                    {
                        json = old;
                    }
                    else
                    {
                        var comma = old.Contains('{') && old.IndexOf('{') != old.LastIndexOf('{') ? "," : string.Empty;
                        json = old.Substring(0, at) + comma + entry + old.Substring(at);
                    }
                }
                else
                {
                    json = "{\"entries\":[" + entry + "]}";
                }
                File.WriteAllText(manifest, json);
            }
        }
    }

    sealed unsafe class HGlobal : IDisposable
    {
        readonly List<nint> owned = new List<nint>();

        public void* Alloc(nuint bytes)
        {
            var size = (nint)Math.Max(bytes, (nuint)1);
            var p = Marshal.AllocHGlobal(size);
            owned.Add(p);
            new Span<byte>((void*)p, (int)size).Clear();
            return (void*)p;
        }

        public void Dispose()
        {
            foreach (var p in owned)
            {
                Marshal.FreeHGlobal(p);
            }
            owned.Clear();
        }
    }
}
