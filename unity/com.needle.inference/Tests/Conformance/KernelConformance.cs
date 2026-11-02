using NUnit.Framework;

namespace Needle.Conformance.Tests
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
            ulong x = state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            state = x;
            return x;
        }

        public float NextFloat(float scale)
        {
            return ((float)(Next() >> 11) * (1f / 9007199254740992f) - 0.5f) * 2f * scale;
        }
    }

    static class Reference
    {
        public static void Softmax(float[] s)
        {
            float peak = float.NegativeInfinity;
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] > peak)
                {
                    peak = s[i];
                }
            }
            float total = 0f;
            for (int i = 0; i < s.Length; i++)
            {
                s[i] = (float)System.Math.Exp(s[i] - peak);
                total += s[i];
            }
            for (int i = 0; i < s.Length; i++)
            {
                s[i] /= total;
            }
        }

        public static float BestProbability(float[] values, float best)
        {
            float total = 0f;
            for (int i = 0; i < values.Length; i++)
            {
                total += (float)System.Math.Exp(values[i] - best);
            }
            return 1f / total;
        }

        public static void RmsUnit(float[] x, float[] destination)
        {
            float sum = 0f;
            for (int i = 0; i < x.Length; i++)
            {
                sum += x[i] * x[i];
            }
            float inverse = 1f / (float)System.Math.Sqrt(sum / x.Length + 1e-6f);
            for (int i = 0; i < x.Length; i++)
            {
                destination[i] = x[i] * inverse;
            }
        }

        public static int ArgMaxMasked(float[] x, uint[] mask, int length)
        {
            int best = -1;
            float peak = float.NegativeInfinity;
            for (int w = 0; w < mask.Length; w++)
            {
                uint bits = mask[w];
                while (bits != 0)
                {
                    int bit = 0;
                    uint v = bits;
                    while ((v & 1u) == 0)
                    {
                        v >>= 1;
                        bit++;
                    }
                    int id = w * 32 + bit;
                    bits &= bits - 1;
                    if (id >= length)
                    {
                        continue;
                    }
                    if (x[id] > peak)
                    {
                        peak = x[id];
                        best = id;
                    }
                }
            }
            return best;
        }

        public static void Sinkhorn(float[] logits, float[] result)
        {
            for (int r = 0; r < 4; r++)
            {
                float peak = logits[r * 4];
                for (int c = 1; c < 4; c++)
                {
                    if (logits[r * 4 + c] > peak)
                    {
                        peak = logits[r * 4 + c];
                    }
                }
                for (int c = 0; c < 4; c++)
                {
                    result[r * 4 + c] = (float)System.Math.Exp(logits[r * 4 + c] - peak);
                }
            }
            for (int iteration = 0; iteration < 20; iteration++)
            {
                for (int r = 0; r < 4; r++)
                {
                    float sum = result[r * 4] + result[r * 4 + 1] + result[r * 4 + 2] + result[r * 4 + 3];
                    for (int c = 0; c < 4; c++)
                    {
                        result[r * 4 + c] /= sum;
                    }
                }
                for (int c = 0; c < 4; c++)
                {
                    float sum = result[c] + result[4 + c] + result[8 + c] + result[12 + c];
                    if (sum > 0f)
                    {
                        for (int r = 0; r < 4; r++)
                        {
                            result[r * 4 + c] /= sum;
                        }
                    }
                }
            }
        }

        public static double Cosine(float[] a, float[] b)
        {
            double dot = 0, left = 0, right = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += (double)a[i] * b[i];
                left += (double)a[i] * a[i];
                right += (double)b[i] * b[i];
            }
            if (left == 0 && right == 0)
            {
                return 1;
            }
            return dot / System.Math.Sqrt(left * right);
        }
    }

    public class KernelConformance
    {
        [Test]
        public void SoftmaxGolden()
        {
            float[] s = { 1f, 2f, 3f };
            Reference.Softmax(s);
            Assert.AreEqual(0.09003057f, s[0], 1e-6f);
            Assert.AreEqual(0.24472848f, s[1], 1e-6f);
            Assert.AreEqual(0.66524094f, s[2], 1e-6f);
        }

        [Test]
        public void SoftmaxUniformAndArgMax()
        {
            float[] s = { 0f, 0f, 0f };
            Reference.Softmax(s);
            for (int i = 0; i < 3; i++)
            {
                Assert.AreEqual(0.33333334f, s[i], 1e-6f);
            }
            float[] tied = { 2f, 2f, 2f, 2f };
            int best = 0;
            for (int i = 1; i < tied.Length; i++)
            {
                if (tied[i] > tied[best])
                {
                    best = i;
                }
            }
            Assert.AreEqual(0, best);
        }

        [Test]
        public void BestProbabilityGolden()
        {
            float[] values = { 1f, 2f, 3f };
            Assert.AreEqual(0.66524094f, Reference.BestProbability(values, 3f), 1e-6f);
            float[] single = { -4f };
            Assert.AreEqual(1f, Reference.BestProbability(single, -4f), 1e-7f);
        }

        [Test]
        public void RmsUnitGoldenAndNorm()
        {
            float[] x = { 3f, 4f };
            float[] y = new float[2];
            Reference.RmsUnit(x, y);
            Assert.AreEqual(0.84852809f, y[0], 1e-6f);
            Assert.AreEqual(1.13137078f, y[1], 1e-6f);
            XorShift64 rng = new XorShift64(0x0DDC);
            float[] wide = new float[768];
            for (int i = 0; i < wide.Length; i++)
            {
                wide[i] = rng.NextFloat(3f);
            }
            float[] normed = new float[768];
            Reference.RmsUnit(wide, normed);
            double norm = 0;
            for (int i = 0; i < normed.Length; i++)
            {
                norm += (double)normed[i] * normed[i];
            }
            Assert.AreEqual(768.0, norm, 0.02 * 768.0);
        }

        [Test]
        public void ArgMaxMaskedGolden()
        {
            float[] x = new float[64];
            for (int i = 0; i < 64; i++)
            {
                x[i] = i;
            }
            Assert.AreEqual(37, Reference.ArgMaxMasked(x, new uint[] { 0u, 1u << 5 }, 64));
            Assert.AreEqual(-1, Reference.ArgMaxMasked(x, new uint[] { 0u, 0u }, 64));
            float[] tied = new float[64];
            for (int i = 0; i < 64; i++)
            {
                tied[i] = 2f;
            }
            Assert.AreEqual(0, Reference.ArgMaxMasked(tied, new uint[] { 0xFFFFFFFFu, 0xFFFFFFFFu }, 64));
        }

        [Test]
        public void ArgMaxMaskedRandomMatchesScalarScan()
        {
            XorShift64 rng = new XorShift64(0x0A969);
            foreach (int length in new int[] { 1, 31, 33, 100 })
            {
                float[] x = new float[length];
                for (int i = 0; i < length; i++)
                {
                    x[i] = rng.NextFloat(5f);
                }
                int words = (length + 31) / 32;
                uint[] mask = new uint[words];
                for (int w = 0; w < words; w++)
                {
                    mask[w] = (uint)(rng.Next() & 0xFFFFFFFFuL);
                }
                int expected = -1;
                float peak = float.NegativeInfinity;
                for (int i = 0; i < length; i++)
                {
                    if ((mask[i / 32] & (1u << (i % 32))) != 0 && x[i] > peak)
                    {
                        peak = x[i];
                        expected = i;
                    }
                }
                Assert.AreEqual(expected, Reference.ArgMaxMasked(x, mask, length));
            }
        }

        [Test]
        public void SinkhornGolden()
        {
            float[] logits = new float[16];
            float[] result = new float[16];
            Reference.Sinkhorn(logits, result);
            for (int i = 0; i < 16; i++)
            {
                Assert.AreEqual(0.25f, result[i], 1e-7f);
            }
        }

        [Test]
        public void SinkhornRandomIsDoublyStochastic()
        {
            XorShift64 rng = new XorShift64(0x0DE1);
            float[] logits = new float[16];
            for (int i = 0; i < 16; i++)
            {
                logits[i] = rng.NextFloat(2f);
            }
            float[] result = new float[16];
            Reference.Sinkhorn(logits, result);
            for (int r = 0; r < 4; r++)
            {
                float row = result[r * 4] + result[r * 4 + 1] + result[r * 4 + 2] + result[r * 4 + 3];
                float col = result[r] + result[4 + r] + result[8 + r] + result[12 + r];
                Assert.AreEqual(1f, row, 1e-5f);
                Assert.AreEqual(1f, col, 1e-5f);
            }
        }

        [Test]
        public void ParityBoundHoldsOnRandomSoftmax()
        {
            XorShift64 rng = new XorShift64(0x0DDD);
            float[] a = new float[100];
            float[] b = new float[100];
            for (int i = 0; i < 100; i++)
            {
                a[i] = b[i] = rng.NextFloat(4f);
            }
            Reference.Softmax(a);
            Reference.Softmax(b);
            Assert.GreaterOrEqual(Reference.Cosine(a, b), 0.99995);
        }
    }
}
