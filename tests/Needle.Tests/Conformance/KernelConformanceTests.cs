using System;
using Xunit;

namespace Needle.Tests.Conformance
{
    sealed unsafe class ConformanceWeights : IDisposable
    {
        readonly HGlobal owned = new HGlobal();
        public Cq Weight;

        public static ConformanceWeights Build(ulong seed, int rows, int cols, int bits, bool extremes)
        {
            var self = new ConformanceWeights();
            var rng = new XorShift64(seed);
            var count = bits == 2 ? 4 : 16;
            var levels = stackalloc float[16];
            for (var i = 0; i < count; i++)
            {
                levels[i] = (i - (count - 1) / 2f) * 0.4f + ((float)(rng.Next() >> 11) * (1f / 9007199254740992f) - 0.5f) * 0.1f;
            }
            var book = (Codebook*)self.owned.Alloc((nuint)sizeof(Codebook));
            Codebook.Fill(book, levels, count);
            var rowBytes = cols * bits / 8;
            var indices = (byte*)self.owned.Alloc((nuint)Math.Max(rows * rowBytes, 1));
            var fill = new XorShift64(seed ^ 0x5BD1E995uL);
            ConformanceInputs.FillBytes(indices, Math.Max(rows * rowBytes, 1), ref fill, extremes);
            var groups = cols / Cq.GroupSize;
            var norms = (ushort*)self.owned.Alloc((nuint)Math.Max(rows * groups * 2, 2));
            for (var r = 0; r < rows; r++)
            {
                for (var g = 0; g < groups; g++)
                {
                    norms[r * groups + g] = ConformanceInputs.HalfBits(0.2f + (float)(fill.Next() % 1000) * 0.0018f);
                }
            }
            if (rows > 0 && groups > 0)
            {
                norms[0] = 0;
                norms[rows * groups - 1] = 0x7BFF;
            }
            self.Weight = new Cq(indices, norms, book, rows, cols, bits);
            return self;
        }

        public void Dispose()
        {
            owned.Dispose();
        }
    }

    public sealed unsafe class MatMulConformance : IDisposable
    {
        readonly HGlobal owned = new HGlobal();

        void RunCase(ulong seed, int bits, int rows, int cols, int tokens, int rowStart, int rowEnd, string name)
        {
            using (var weights = ConformanceWeights.Build(seed, rows, cols, bits, true))
            {
                var w = weights.Weight;
                var groups = cols / Cq.GroupSize;
                var stride = (groups + 7) & ~7;
                var arena = new Arena((nuint)(tokens * cols + 8 * tokens * stride + 4096));
                try
                {
                    var a = new Activations(ref arena, tokens, cols);
                    var fill = new XorShift64(seed ^ 0x243F6A88uL);
                    var x = (float*)owned.Alloc((nuint)Math.Max(tokens * cols * 4, 4));
                    ConformanceInputs.FillEdges(x, Math.Max(tokens * cols, 1), ref fill, 3f);
                    if (tokens > 0)
                    {
                        Quant.Prepare(x, tokens, cols, bits, a);
                    }
                    var ldy = rows == 0 ? 1 : rows;
                    var count = Math.Max(tokens * ldy, 1);
                    var expected = (float*)owned.Alloc((nuint)count * 4);
                    var actual = (float*)owned.Alloc((nuint)count * 4);
                    for (var i = 0; i < count; i++)
                    {
                        expected[i] = 1.25f;
                        actual[i] = 1.25f;
                    }
                    Scalar.MatMulRows(w, a, tokens, expected, ldy, rowStart, rowEnd);
                    Quant.MatMulRows(w, a, tokens, actual, ldy, rowStart, rowEnd);
                    if (rowEnd == rowStart || tokens == 0)
                    {
                        for (var i = 0; i < count; i++)
                        {
                            Assert.Equal(1.25f, actual[i]);
                        }
                    }
                    if (rowEnd > rowStart && tokens > 0)
                    {
                        for (var t = 0; t < tokens; t++)
                        {
                            for (var n = rowStart; n < rowEnd; n++)
                            {
                                var delta = actual[t * ldy + n] - expected[t * ldy + n];
                                var bound = 1e-3f * MathF.Max(1f, MathF.Abs(expected[t * ldy + n]));
                                Assert.True(delta >= -bound && delta <= bound, name + " [" + t + "," + n + "]: " + delta);
                            }
                        }
                        var slice = (float*)owned.Alloc((nuint)(tokens * (rowEnd - rowStart) * 4));
                        for (var t = 0; t < tokens; t++)
                        {
                            for (var n = rowStart; n < rowEnd; n++)
                            {
                                slice[t * (rowEnd - rowStart) + n - rowStart] = actual[t * ldy + n];
                            }
                        }
                        if (tokens * (rowEnd - rowStart) >= 4)
                        {
                            var refSlice = (float*)owned.Alloc((nuint)(tokens * (rowEnd - rowStart) * 4));
                            for (var t = 0; t < tokens; t++)
                            {
                                for (var n = rowStart; n < rowEnd; n++)
                                {
                                    refSlice[t * (rowEnd - rowStart) + n - rowStart] = expected[t * ldy + n];
                                }
                            }
                            ConformanceAssert.Parity(refSlice, slice, tokens * (rowEnd - rowStart), name);
                        }
                        ConformanceDump.Dump("matmul_" + name, slice, tokens * (rowEnd - rowStart) * 4);
                    }
                }
                finally
                {
                    arena.Free();
                }
            }
        }

        [Fact]
        public void GoldenTwoBit()
        {
            RunCase(0x1234, 2, 4, 128, 2, 0, 4, "golden_2b");
        }

        [Fact]
        public void GoldenFourBit()
        {
            RunCase(0x1234, 4, 4, 128, 2, 0, 4, "golden_4b");
        }

        [Fact]
        public void ZeroRowsLeaveOutputUntouched()
        {
            RunCase(0x1234, 2, 4, 128, 2, 2, 2, "zero_rows");
            using (var weights = ConformanceWeights.Build(0x5678, 0, 128, 2, false))
            {
                var w = weights.Weight;
                var arena = new Arena(4096);
                try
                {
                    var a = new Activations(ref arena, 1, 128);
                    var y = (float*)owned.Alloc(4);
                    y[0] = 7.5f;
                    Scalar.MatMulRows(w, a, 1, y, 1, 0, 0);
                    Quant.MatMulRows(w, a, 1, y, 1, 0, 0);
                    Assert.Equal(7.5f, y[0]);
                }
                finally
                {
                    arena.Free();
                }
            }
        }

        [Theory]
        [InlineData(2, 5, 768, 1)]
        [InlineData(2, 5, 768, 3)]
        [InlineData(4, 7, 384, 5)]
        [InlineData(2, 3, 128, 8)]
        [InlineData(4, 97, 768, 3)]
        [InlineData(2, 5, 768, 0)]
        public void ShapesAndTokenCounts(int bits, int rows, int cols, int tokens)
        {
            RunCase(0x9ABC, bits, rows, cols, tokens, 0, rows, "shape_" + bits + "_" + rows + "x" + cols + "_t" + tokens);
        }

        [Fact]
        public void RowRangeEdges()
        {
            RunCase(0x9ABC, 2, 8, 128, 3, 3, 7, "range_odd");
        }

        public void Dispose()
        {
            owned.Dispose();
        }
    }

    public sealed unsafe class PrepareConformance : IDisposable
    {
        readonly HGlobal owned = new HGlobal();

        void Reference(float* x, int tokens, int cols, int bits, float* scales, float* offsets, sbyte* values)
        {
            var groups = cols / Cq.GroupSize;
            var rotated = stackalloc float[Cq.GroupSize];
            var quantized = stackalloc sbyte[Cq.GroupSize];
            var stride = (groups + 7) & ~7;
            for (var t = 0; t < tokens; t++)
            {
                for (var g = 0; g < groups; g++)
                {
                    Scalar.Hadamard(x + (long)t * cols + g * Cq.GroupSize, rotated);
                    scales[t * stride + g] = Scalar.QuantizeInt8(rotated, Cq.GroupSize, quantized);
                    Scalar.Planes(quantized, bits, values + ((long)t * groups + g) * Cq.GroupSize, out var sum);
                    offsets[t * stride + g] = Codebook.Bias * sum;
                }
            }
        }

        void RunCase(ulong seed, int cols, int tokens, int bits, bool edges, string name)
        {
            var groups = cols / Cq.GroupSize;
            var stride = (groups + 7) & ~7;
            var arena = new Arena((nuint)(tokens * cols + 8 * tokens * stride + 4096));
            try
            {
                var a = new Activations(ref arena, tokens, cols);
                var rng = new XorShift64(seed);
                var x = (float*)owned.Alloc((nuint)(tokens * cols * 4));
                if (edges)
                {
                    ConformanceInputs.FillEdges(x, tokens * cols, ref rng, 2f);
                }
                else
                {
                    ConformanceInputs.FillUniform(x, tokens * cols, ref rng, 2f);
                }
                Quant.Prepare(x, tokens, cols, bits, a);
                var refScales = (float*)owned.Alloc((nuint)(tokens * stride * 4));
                var refOffsets = (float*)owned.Alloc((nuint)(tokens * stride * 4));
                var refValues = (sbyte*)owned.Alloc((nuint)(tokens * cols));
                Reference(x, tokens, cols, bits, refScales, refOffsets, refValues);
                for (var t = 0; t < tokens; t++)
                {
                    for (var g = 0; g < groups; g++)
                    {
                        var slot = t * stride + g;
                        var scaleDrift = a.Scales[slot] - refScales[slot];
                        var scaleBound = 1e-6f * MathF.Max(1f, MathF.Abs(refScales[slot]));
                        Assert.True(scaleDrift >= -scaleBound && scaleDrift <= scaleBound, name + " scale [" + slot + "]: " + scaleDrift);
                        var drift = a.Offsets[slot] - refOffsets[slot];
                        Assert.True(drift >= -8064f && drift <= 8064f, name + " offset [" + slot + "]: " + drift);
                    }
                }
                for (var i = 0; i < tokens * cols; i++)
                {
                    var drift = a.Values[i] - refValues[i];
                    Assert.True(drift >= -1 && drift <= 1, name + " value [" + i + "]: " + drift);
                }
                ConformanceDump.Dump("prepare_" + name + "_scales", a.Scales, tokens * stride * 4);
                ConformanceDump.Dump("prepare_" + name + "_values", a.Values, tokens * cols);
            }
            finally
            {
                arena.Free();
            }
        }

        [Theory]
        [InlineData(128, 1, 2)]
        [InlineData(128, 3, 2)]
        [InlineData(768, 1, 4)]
        [InlineData(768, 3, 4)]
        public void RandomInputs(int cols, int tokens, int bits)
        {
            RunCase(0x2458, cols, tokens, bits, false, "random_" + cols + "_t" + tokens + "_" + bits + "b");
        }

        [Fact]
        public void EdgeInputs()
        {
            RunCase(0x2458, 128, 3, 2, true, "edges");
        }

        [Fact]
        public void ZeroInputIsBitExact()
        {
            var cols = 128;
            var tokens = 2;
            var groups = cols / Cq.GroupSize;
            var stride = (groups + 7) & ~7;
            var arena = new Arena((nuint)(tokens * cols + 8 * tokens * stride + 4096));
            try
            {
                var a = new Activations(ref arena, tokens, cols);
                var x = (float*)owned.Alloc((nuint)(tokens * cols * 4));
                Quant.Prepare(x, tokens, cols, 2, a);
                for (var t = 0; t < tokens; t++)
                {
                    for (var g = 0; g < groups; g++)
                    {
                        Assert.Equal(1f, a.Scales[t * stride + g]);
                        Assert.Equal(0f, a.Offsets[t * stride + g]);
                    }
                }
                for (var i = 0; i < tokens * cols; i++)
                {
                    Assert.Equal((sbyte)0, a.Values[i]);
                }
            }
            finally
            {
                arena.Free();
            }
        }

        [Fact]
        public void RoundingTiesAreBitExact()
        {
            var ties = new float[] { 0.5f, 1.5f, 2.5f, 3.5f, -0.5f, -1.5f, -2.5f, 127.5f };
            var n = ties.Length;
            var expected = (float*)owned.Alloc((nuint)n * 4);
            var actual = (float*)owned.Alloc((nuint)n * 4);
            for (var i = 0; i < n; i++)
            {
                expected[i] = actual[i] = ties[i];
            }
            Scalar.RoundTrip(expected, n, 1f, 1f);
            Quant.RoundTrip(actual, n, 1f, 1f);
            ConformanceAssert.BitExact(expected, actual, n, "ties");
            Assert.Equal(0f, actual[0]);
            Assert.Equal(2f, actual[1]);
            Assert.Equal(2f, actual[2]);
            Assert.Equal(127f, actual[7]);
        }

        public void Dispose()
        {
            owned.Dispose();
        }
    }

    public sealed unsafe class GatherRowConformance : IDisposable
    {
        readonly HGlobal owned = new HGlobal();

        void RunCase(ulong seed, int bits, int cols, int rows, int row, string name)
        {
            using (var weights = ConformanceWeights.Build(seed, rows, cols, bits, true))
            {
                var w = weights.Weight;
                var expected = (float*)owned.Alloc((nuint)cols * 4);
                var actual = (float*)owned.Alloc((nuint)cols * 4);
                Scalar.DecodeRow(w, row, expected);
                Quant.DecodeRow(w, row, actual);
                ConformanceAssert.Parity(expected, actual, cols, name + " decode");
                for (var g = 0; g < w.Groups; g++)
                {
                    Scalar.Hadamard(expected + g * Cq.GroupSize, expected + g * Cq.GroupSize);
                }
                Quant.GatherRow(w, row, actual);
                ConformanceAssert.Parity(expected, actual, cols, name);
                ConformanceAssert.Close(expected, actual, cols, 1e-5f, name);
                ConformanceDump.Dump("gather_" + name, actual, cols * 4);
            }
        }

        [Theory]
        [InlineData(2, 128)]
        [InlineData(4, 128)]
        [InlineData(2, 768)]
        [InlineData(4, 768)]
        public void FirstAndLastRows(int bits, int cols)
        {
            RunCase(0xACE1, bits, cols, 6, 0, "first_" + bits + "_" + cols);
            RunCase(0xACE1, bits, cols, 6, 5, "last_" + bits + "_" + cols);
        }

        [Fact]
        public void ZeroNormYieldsZeros()
        {
            using (var weights = ConformanceWeights.Build(0xACE1, 2, 128, 2, false))
            {
                var w = weights.Weight;
                var groups = w.Groups;
                for (var g = 0; g < groups; g++)
                {
                    w.Norms[g] = 0;
                }
                var actual = (float*)owned.Alloc((nuint)128 * 4);
                Quant.GatherRow(w, 0, actual);
                for (var i = 0; i < 128; i++)
                {
                    Assert.Equal(0f, actual[i]);
                }
            }
        }

        public void Dispose()
        {
            owned.Dispose();
        }
    }

    public sealed unsafe class AttentionTileConformance : IDisposable
    {
        readonly HGlobal owned = new HGlobal();
        const int Positions = 45;

        static AttentionTile Tile(
            sbyte* queries, float* queryScales, int heads, sbyte* keys, float* keyScales,
            sbyte* values, float* valueScales, int kvHeads, int kv, int first, int count,
            float* scores, float* output, float* probabilities) => new AttentionTile
        {
            Queries = queries, QueryScales = queryScales, Heads = heads,
            Keys = keys + kv * 48, KeyScales = keyScales + kv, Values = values + kv * 64, ValueScales = valueScales + kv,
            KeyStride = kvHeads * 48, ValueStride = kvHeads * 64, ScaleStride = kvHeads, First = first, Count = count,
            Scores = scores, Output = output, Probabilities = probabilities, ProbabilityStride = Positions,
        };

        void RunCase(int kvHeads, int heads, int first, int count, bool capture, bool extremes, string name)
        {
            var rng = new XorShift64((ulong)(kvHeads * 1000 + heads * 100 + first * 7 + count * 13 + 0xBEEF));
            var queries = (sbyte*)owned.Alloc((nuint)(heads * 48));
            var queryScales = (float*)owned.Alloc((nuint)(heads * 4));
            var keys = (sbyte*)owned.Alloc((nuint)(Positions * kvHeads * 48));
            var values = (sbyte*)owned.Alloc((nuint)(Positions * kvHeads * 64));
            var keyScales = (float*)owned.Alloc((nuint)(Positions * kvHeads * 4));
            var valueScales = (float*)owned.Alloc((nuint)(Positions * kvHeads * 4));
            ConformanceInputs.FillSByte(queries, heads * 48, ref rng, extremes);
            ConformanceInputs.FillSByte(keys, Positions * kvHeads * 48, ref rng, extremes);
            ConformanceInputs.FillSByte(values, Positions * kvHeads * 64, ref rng, extremes);
            for (var i = 0; i < heads; i++)
            {
                queryScales[i] = 0.4f + 0.1f * i;
            }
            for (var i = 0; i < Positions * kvHeads; i++)
            {
                keyScales[i] = 0.0004f * (1 + rng.NextInt(0, 4));
                valueScales[i] = 0.003f * (1 + rng.NextInt(0, 4));
            }
            var kv = kvHeads - 1;
            var expectedOutput = (float*)owned.Alloc((nuint)(heads * 64 * 4));
            var expectedProbabilities = (float*)owned.Alloc((nuint)(heads * Positions * 4));
            var expectedScores = (float*)owned.Alloc((nuint)Attention.ScoreFloats(count) * 4);
            Scalar.AttentionTile(Tile(queries, queryScales, heads, keys, keyScales, values, valueScales, kvHeads, kv, first, count, expectedScores, expectedOutput, capture ? expectedProbabilities : null));
            var actualOutput = (float*)owned.Alloc((nuint)(heads * 64 * 4));
            var actualProbabilities = (float*)owned.Alloc((nuint)(heads * Positions * 4));
            var actualScores = (float*)owned.Alloc((nuint)Attention.ScoreFloats(count) * 4);
            var tile = Tile(queries, queryScales, heads, keys, keyScales, values, valueScales, kvHeads, kv, first, count, actualScores, actualOutput, capture ? actualProbabilities : null);
            Attention.Run(tile);
            var peak = 0f;
            for (var i = 0; i < heads * 64; i++)
            {
                peak = MathF.Max(peak, MathF.Abs(expectedOutput[i]));
            }
            for (var i = 0; i < heads * 64; i++)
            {
                var delta = actualOutput[i] - expectedOutput[i];
                var bound = 5e-5f * peak;
                Assert.True(delta >= -bound && delta <= bound, name + " [" + i + "]: " + delta);
            }
            ConformanceAssert.Parity(expectedOutput, actualOutput, heads * 64, name);
            if (capture)
            {
                for (var i = 0; i < heads * Positions; i++)
                {
                    var delta = actualProbabilities[i] - expectedProbabilities[i];
                    Assert.True(delta >= -1e-5f && delta <= 1e-5f, name + " prob [" + i + "]: " + delta);
                }
            }
            ConformanceDump.Dump("attention_" + name, actualOutput, heads * 64 * 4);
        }

        [Theory]
        [InlineData(2, 6, 0, 45, true)]
        [InlineData(2, 6, 13, 32, true)]
        [InlineData(2, 6, 41, 4, true)]
        [InlineData(8, 1, 0, 45, true)]
        [InlineData(1, 6, 0, 1, false)]
        [InlineData(2, 6, 7, 2, false)]
        public void TileMatchesScalar(int kvHeads, int heads, int first, int count, bool capture)
        {
            RunCase(kvHeads, heads, first, count, capture, false, "tile_" + kvHeads + "_" + heads + "_" + first + "_" + count);
        }

        [Fact]
        public void Int8ExtremesStayFinite()
        {
            RunCase(2, 6, 0, 45, true, true, "extremes");
        }

        public void Dispose()
        {
            owned.Dispose();
        }
    }

#if NET10_0_OR_GREATER
    public sealed unsafe class MergePartialsConformance : IDisposable
    {
        readonly HGlobal owned = new HGlobal();

        static void Reference(float* partials, int count, int heads, float* output)
        {
            for (var h = 0; h < heads; h++)
            {
                var peak = float.NegativeInfinity;
                for (var c = 0; c < count; c++)
                {
                    peak = MathF.Max(peak, partials[c * Attention.PartialFloats + Attention.MaxTileHeads * 64 + h]);
                }
                double total = 0;
                for (var c = 0; c < count; c++)
                {
                    total += Math.Exp(partials[c * Attention.PartialFloats + Attention.MaxTileHeads * 64 + h] - peak)
                        * partials[c * Attention.PartialFloats + Attention.MaxTileHeads * 64 + 8 + h];
                }
                var inverse = 1.0 / total;
                for (var d = 0; d < 64; d++)
                {
                    double sum = 0;
                    for (var c = 0; c < count; c++)
                    {
                        var w = Math.Exp(partials[c * Attention.PartialFloats + Attention.MaxTileHeads * 64 + h] - peak);
                        sum += w * partials[c * Attention.PartialFloats + h * 64 + d];
                    }
                    output[h * 64 + d] = (float)(sum * inverse);
                }
            }
        }

        void RunCase(int count, int heads, string name)
        {
            var rng = new XorShift64((ulong)(count * 31 + heads * 7 + 0xF001));
            var partials = (float*)owned.Alloc((nuint)(count * Attention.PartialFloats * 4));
            for (var c = 0; c < count; c++)
            {
                var chunk = partials + c * Attention.PartialFloats;
                ConformanceInputs.FillEdges(chunk, heads * 64, ref rng, 2f);
                for (var h = 0; h < heads; h++)
                {
                    chunk[Attention.MaxTileHeads * 64 + h] = rng.NextFloat(2f);
                    chunk[Attention.MaxTileHeads * 64 + 8 + h] = 0.5f + MathF.Abs(rng.NextFloat(2f));
                }
            }
            var expected = (float*)owned.Alloc((nuint)(heads * 64 * 4));
            var actual = (float*)owned.Alloc((nuint)(heads * 64 * 4));
            Reference(partials, count, heads, expected);
            Attention.MergePartials(partials, count, heads, actual);
            ConformanceAssert.Parity(expected, actual, heads * 64, name);
            ConformanceAssert.Close(expected, actual, heads * 64, 1e-6f, name);
            ConformanceDump.Dump("merge_" + name, actual, heads * 64 * 4);
        }

        [Theory]
        [InlineData(1, 1)]
        [InlineData(1, 6)]
        [InlineData(2, 1)]
        [InlineData(2, 6)]
        [InlineData(3, 6)]
        public void CountsAndHeads(int count, int heads)
        {
            RunCase(count, heads, "count" + count + "_heads" + heads);
        }

        public void Dispose()
        {
            owned.Dispose();
        }
    }
#endif

    public sealed unsafe class HadamardMlpConformance : IDisposable
    {
        readonly HGlobal owned = new HGlobal();

        struct ShadowWeights
        {
            public float* D1, D2, B2, D3, D4;
            public float* W1A, W1B, W2A, W2B, W3A, W3B;
            public float* CondV, CondU;
        }

        void BuildBatch(MlpBatch* batch, ShadowWeights* shadow, ulong seed, int dim, int rows, int cols, int* permA, int* permB, bool rotated)
        {
            var width = rows * cols;
            var rng = new XorShift64(seed);
            shadow->D1 = (float*)owned.Alloc((nuint)(dim * 4));
            shadow->D2 = (float*)owned.Alloc((nuint)(width * 4));
            shadow->B2 = (float*)owned.Alloc((nuint)(width * 4));
            shadow->D3 = (float*)owned.Alloc((nuint)(width * 4));
            shadow->D4 = (float*)owned.Alloc((nuint)(dim * 4));
            shadow->W1A = (float*)owned.Alloc((nuint)(4 * rows * 4));
            shadow->W1B = (float*)owned.Alloc((nuint)(cols * cols * 4));
            shadow->W2A = (float*)owned.Alloc((nuint)(rows * rows * 4));
            shadow->W2B = (float*)owned.Alloc((nuint)(cols * cols * 4));
            shadow->W3A = (float*)owned.Alloc((nuint)(rows * rows * 4));
            shadow->W3B = (float*)owned.Alloc((nuint)(cols * cols * 4));
            shadow->CondV = (float*)owned.Alloc((nuint)(dim * 8 * 4));
            shadow->CondU = (float*)owned.Alloc((nuint)(8 * width * 4));
            ConformanceInputs.FillUniform(shadow->D1, dim, ref rng, 0.1f);
            ConformanceInputs.FillUniform(shadow->D2, width, ref rng, 0.1f);
            ConformanceInputs.FillUniform(shadow->B2, width, ref rng, 0.1f);
            ConformanceInputs.FillUniform(shadow->D3, width, ref rng, 0.1f);
            ConformanceInputs.FillUniform(shadow->D4, dim, ref rng, 0.1f);
            ConformanceInputs.FillUniform(shadow->W1A, 4 * rows, ref rng, 0.2f);
            ConformanceInputs.FillUniform(shadow->W1B, cols * cols, ref rng, 0.2f);
            ConformanceInputs.FillUniform(shadow->W2A, rows * rows, ref rng, 0.2f);
            ConformanceInputs.FillUniform(shadow->W2B, cols * cols, ref rng, 0.2f);
            ConformanceInputs.FillUniform(shadow->W3A, rows * rows, ref rng, 0.2f);
            ConformanceInputs.FillUniform(shadow->W3B, cols * cols, ref rng, 0.2f);
            ConformanceInputs.FillUniform(shadow->CondV, dim * 8, ref rng, 0.2f);
            ConformanceInputs.FillUniform(shadow->CondU, 8 * width, ref rng, 0.2f);
            for (var i = 0; i < width; i++)
            {
                permA[i] = rotated ? (i + 1) % width : i;
                permB[i] = rotated ? (i + width - 1) % width : i;
            }
            var preNorm = (float*)owned.Alloc((nuint)(dim * 4));
            var residual = (float*)owned.Alloc((nuint)(dim * 4));
            ConformanceInputs.FillUniform(preNorm, dim, ref rng, 0.1f);
            ConformanceInputs.FillUniform(residual, dim, ref rng, 1f);
            batch->Weights = (MlpWeights*)shadow;
            batch->PreNorm = preNorm;
            batch->PermA = permA;
            batch->PermB = permB;
            batch->Residual = residual;
            batch->Dim = dim;
            batch->Rows = rows;
            batch->Cols = cols;
            batch->Scale = 0.7f;
        }

        static void ZeroCentered(float* x, float* scale, int n, float* destination)
        {
            var sum = 0f;
            for (var i = 0; i < n; i++)
            {
                sum += x[i] * x[i];
            }
            var inverse = 1f / MathF.Sqrt(sum / n + 1e-6f);
            for (var i = 0; i < n; i++)
            {
                destination[i] = (1f + scale[i]) * x[i] * inverse;
            }
        }

        void RunCase(ulong seed, bool rotated, string name)
        {
            const int dim = 128;
            const int rows = 4;
            const int cols = 32;
            var width = rows * cols;
            var shadow = stackalloc ShadowWeights[1];
            var batch = stackalloc MlpBatch[1];
            var permA = (int*)owned.Alloc((nuint)(width * 4));
            var permB = (int*)owned.Alloc((nuint)(width * 4));
            BuildBatch(batch, shadow, seed, dim, rows, cols, permA, permB, rotated);
            var inputRows = 4;
            var outputRows = 4;
            var scratch = (float*)owned.Alloc((nuint)HadamardMlp.ScratchFloats(dim, width) * 4);
            var expected = (float*)owned.Alloc((nuint)(dim * 4));
            for (var i = 0; i < dim; i++)
            {
                expected[i] = batch->Residual[i];
            }
            var normalized = (float*)owned.Alloc((nuint)(dim * 4));
            var weights = (float*)owned.Alloc(32);
            var front = (float*)owned.Alloc((nuint)(width * 4));
            var back = (float*)owned.Alloc((nuint)(width * 4));
            ZeroCentered(batch->Residual, batch->PreNorm, dim, normalized);
            Scalar.ConditionWeights(shadow->CondV, normalized, dim, weights);
            Scalar.ScaleInput(normalized, shadow->D1, dim, inputRows * cols, front);
            Scalar.Kron(front, inputRows, shadow->W1A, shadow->W1B, rows, cols, rows, back);
            Scalar.PermuteModulateActivate(permA, back, shadow->D2, weights, shadow->CondU, shadow->B2, width, front);
            Scalar.Kron(front, rows, shadow->W2A, shadow->W2B, rows, cols, rows, back);
            Scalar.PermuteScale(permB, back, shadow->D3, width, front);
            Scalar.Kron(front, rows, shadow->W3A, shadow->W3B, rows, cols, outputRows, back);
            Scalar.AccumulateOutput(expected, back, shadow->D4, batch->Scale, dim);
            var before = (float*)owned.Alloc((nuint)(dim * 4));
            for (var i = 0; i < dim; i++)
            {
                before[i] = batch->Residual[i];
            }
            HadamardMlp.Apply(batch, batch->Residual, scratch);
            ConformanceAssert.Parity(expected, batch->Residual, dim, name);
            ConformanceAssert.Close(expected, batch->Residual, dim, 1e-5f, name);
            HadamardMlp.Apply(batch, before, scratch);
            ConformanceAssert.BitExact(batch->Residual, before, dim, name + " determinism");
            ConformanceDump.Dump("mlp_" + name, batch->Residual, dim * 4);
        }

        [Fact]
        public void IdentityPermutation()
        {
            RunCase(0x1A5E, false, "identity");
        }

        [Fact]
        public void RotatedPermutation()
        {
            RunCase(0x1A5E, true, "rotated");
        }

        [Fact]
        public void KronOddTails()
        {
            var rng = new XorShift64(0x1A5E);
            foreach (var cols in new int[] { 15, 16, 17, 31, 32, 33 })
            {
                const int activeRows = 3;
                const int rows = 4;
                const int resultRows = 4;
                var z = (float*)owned.Alloc((nuint)(activeRows * cols * 4));
                var left = (float*)owned.Alloc((nuint)(activeRows * rows * 4));
                var right = (float*)owned.Alloc((nuint)(cols * cols * 4));
                var expected = (float*)owned.Alloc((nuint)(resultRows * cols * 4));
                var actual = (float*)owned.Alloc((nuint)(resultRows * cols * 4));
                ConformanceInputs.FillUniform(z, activeRows * cols, ref rng, 1f);
                ConformanceInputs.FillUniform(left, activeRows * rows, ref rng, 1f);
                ConformanceInputs.FillUniform(right, cols * cols, ref rng, 0.2f);
                for (var k = 0; k < resultRows; k++)
                {
                    for (var c = 0; c < cols; c++)
                    {
                        double sum = 0;
                        for (var j = 0; j < cols; j++)
                        {
                            double inner = 0;
                            for (var i = 0; i < activeRows; i++)
                            {
                                inner += left[i * rows + k] * z[i * cols + j];
                            }
                            sum += inner * right[j * cols + c];
                        }
                        expected[k * cols + c] = (float)sum;
                    }
                }
                HadamardMlp.Kron(z, activeRows, left, right, rows, cols, resultRows, actual);
                ConformanceAssert.Parity(expected, actual, resultRows * cols, "kron_" + cols);
                ConformanceDump.Dump("kron_" + cols, actual, resultRows * cols * 4);
            }
        }

        public void Dispose()
        {
            owned.Dispose();
        }
    }

    public sealed unsafe class RowOpsConformance : IDisposable
    {
        readonly HGlobal owned = new HGlobal();

        static void RmsUnitReference(float* x, int n, float* destination)
        {
            var sum = 0f;
            for (var i = 0; i < n; i++)
            {
                sum += x[i] * x[i];
            }
            var inverse = 1f / MathF.Sqrt(sum / n + 1e-6f);
            for (var i = 0; i < n; i++)
            {
                destination[i] = x[i] * inverse;
            }
        }

        static void ZeroCenteredReference(float* x, float* scale, int n, float* destination)
        {
            var sum = 0f;
            for (var i = 0; i < n; i++)
            {
                sum += x[i] * x[i];
            }
            var inverse = 1f / MathF.Sqrt(sum / n + 1e-6f);
            for (var i = 0; i < n; i++)
            {
                destination[i] = (1f + scale[i]) * x[i] * inverse;
            }
        }

        [Theory]
        [InlineData(8)]
        [InlineData(24)]
        [InlineData(768)]
        public void RmsFakeA8(int n)
        {
            var rng = new XorShift64(0x0DD5);
            var x = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var expected = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var actual = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            ConformanceInputs.FillEdges(x, Math.Max(n, 1), ref rng, 3f);
            for (var i = 0; i < n; i++)
            {
                expected[i] = actual[i] = x[i];
            }
            Scalar.RmsFakeA8(expected, n, expected);
            Ops.RmsFakeA8(actual, n, actual);
            if (n > 0)
            {
                ConformanceAssert.Parity(expected, actual, n, "rmsfakea8");
                ConformanceDump.Dump("rmsfakea8_" + n, actual, n * 4);
            }
        }

        [Theory]
        [InlineData(8)]
        [InlineData(24)]
        [InlineData(768)]
        public void CenteredRmsFakeA8(int n)
        {
            var rng = new XorShift64(0x0DD6);
            var x = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var s = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var expected = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var actual = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            ConformanceInputs.FillEdges(x, Math.Max(n, 1), ref rng, 3f);
            ConformanceInputs.FillUniform(s, Math.Max(n, 1), ref rng, 0.1f);
            for (var i = 0; i < n; i++)
            {
                expected[i] = actual[i] = x[i];
            }
            Scalar.CenteredRmsFakeA8(expected, s, n, expected);
            Ops.CenteredRmsFakeA8(actual, s, n, actual);
            if (n > 0)
            {
                ConformanceAssert.Parity(expected, actual, n, "centered");
                ConformanceDump.Dump("centered_" + n, actual, n * 4);
            }
        }

        [Theory]
        [InlineData(8)]
        [InlineData(24)]
        [InlineData(768)]
        public void GateFakeA8(int n)
        {
            var rng = new XorShift64(0x0DD7);
            var x = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var g = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var expected = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var actual = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            ConformanceInputs.FillEdges(x, Math.Max(n, 1), ref rng, 2f);
            ConformanceInputs.FillEdges(g, Math.Max(n, 1), ref rng, 2f);
            for (var i = 0; i < n; i++)
            {
                expected[i] = actual[i] = x[i];
            }
            Scalar.GateFakeA8(expected, g, n);
            Ops.GateFakeA8(actual, g, n);
            if (n > 0)
            {
                ConformanceAssert.Parity(expected, actual, n, "gate");
                ConformanceDump.Dump("gate_" + n, actual, n * 4);
            }
        }

        [Theory]
        [InlineData(8)]
        [InlineData(24)]
        [InlineData(768)]
        public void PostNormAdd(int n)
        {
            var rng = new XorShift64(0x0DD8);
            var residual = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var x = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var s = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var expected = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var actual = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            ConformanceInputs.FillUniform(residual, Math.Max(n, 1), ref rng, 1f);
            ConformanceInputs.FillEdges(x, Math.Max(n, 1), ref rng, 3f);
            ConformanceInputs.FillUniform(s, Math.Max(n, 1), ref rng, 0.1f);
            for (var i = 0; i < n; i++)
            {
                expected[i] = actual[i] = residual[i];
            }
            Scalar.PostNormAdd(expected, x, s, 0.7f, n);
            Ops.PostNormAdd(actual, x, s, 0.7f, n);
            if (n > 0)
            {
                ConformanceAssert.Parity(expected, actual, n, "postnorm");
            }
        }

        [Theory]
        [InlineData(8)]
        [InlineData(64)]
        [InlineData(768)]
        public void Mix4Combine4CombineDiff4(int dim)
        {
            var rng = new XorShift64(0x0DD9);
            var s = (float*)owned.Alloc((nuint)(4 * dim * 4));
            var y = (float*)owned.Alloc((nuint)(dim * 4));
            var u = (float*)owned.Alloc((nuint)(dim * 4));
            var expected = (float*)owned.Alloc((nuint)(dim * 4));
            var actual = (float*)owned.Alloc((nuint)(dim * 4));
            ConformanceInputs.FillEdges(s, 4 * dim, ref rng, 2f);
            ConformanceInputs.FillUniform(y, dim, ref rng, 1f);
            ConformanceInputs.FillUniform(u, dim, ref rng, 1f);
            Scalar.Mix4(s, dim, 0.3f, -0.2f, 0.1f, 0.4f, expected);
            Ops.Mix4(s, dim, 0.3f, -0.2f, 0.1f, 0.4f, actual);
            ConformanceAssert.Parity(expected, actual, dim, "mix4");
            Scalar.Combine4(s, dim, 0.3f, -0.2f, 0.1f, 0.4f, y, 0.5f, expected);
            Ops.Combine4(s, dim, 0.3f, -0.2f, 0.1f, 0.4f, y, 0.5f, actual);
            ConformanceAssert.Parity(expected, actual, dim, "combine4");
            Scalar.CombineDiff4(s, dim, 0.3f, -0.2f, 0.1f, 0.4f, y, u, 0.5f, expected);
            Ops.CombineDiff4(s, dim, 0.3f, -0.2f, 0.1f, 0.4f, y, u, 0.5f, actual);
            ConformanceAssert.Parity(expected, actual, dim, "combinediff4");
            ConformanceDump.Dump("mix4_" + dim, actual, dim * 4);
        }

        [Theory]
        [InlineData(8)]
        [InlineData(24)]
        [InlineData(64)]
        public void Taps3(int width)
        {
            var rng = new XorShift64(0x0DDA);
            var taps = (float*)owned.Alloc((nuint)(3 * width * 4));
            var backing = (float*)owned.Alloc((nuint)(3 * width * 4));
            var expected = (float*)owned.Alloc((nuint)(width * 4));
            var actual = (float*)owned.Alloc((nuint)(width * 4));
            ConformanceInputs.FillUniform(taps, 3 * width, ref rng, 1f);
            ConformanceInputs.FillEdges(backing, 3 * width, ref rng, 1f);
            var current = backing + 2 * width;
            Scalar.Taps3(taps, width, current, expected);
            Ops.Taps3(taps, width, current, actual);
            ConformanceAssert.Parity(expected, actual, width, "taps3");
        }

        [Theory]
        [InlineData(1, 16)]
        [InlineData(3, 48)]
        [InlineData(6, 64)]
        public void Rope(int heads, int headDim)
        {
            var rng = new XorShift64(0x0DDB);
            var n = heads * headDim;
            var half = headDim / 2;
            var x = (float*)owned.Alloc((nuint)(n * 4));
            var cos = (float*)owned.Alloc((nuint)(half * 4));
            var sin = (float*)owned.Alloc((nuint)(half * 4));
            var expected = (float*)owned.Alloc((nuint)(n * 4));
            var actual = (float*)owned.Alloc((nuint)(n * 4));
            ConformanceInputs.FillEdges(x, n, ref rng, 2f);
            ConformanceInputs.FillUniform(cos, half, ref rng, 1f);
            ConformanceInputs.FillUniform(sin, half, ref rng, 1f);
            for (var i = 0; i < n; i++)
            {
                expected[i] = actual[i] = x[i];
            }
            Scalar.Rope(expected, heads, headDim, cos, sin);
            Ops.Rope(actual, heads, headDim, cos, sin);
            ConformanceAssert.Parity(expected, actual, n, "rope");
        }

        [Theory]
        [InlineData(8)]
        [InlineData(24)]
        [InlineData(768)]
        public void ZeroCenteredRmsAndRmsUnit(int n)
        {
            var rng = new XorShift64(0x0DDC);
            var x = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var s = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var expected = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var actual = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            ConformanceInputs.FillEdges(x, Math.Max(n, 1), ref rng, 3f);
            ConformanceInputs.FillUniform(s, Math.Max(n, 1), ref rng, 0.1f);
            ZeroCenteredReference(x, s, n, expected);
            Ops.ZeroCenteredRms(x, s, n, actual);
            if (n > 0)
            {
                ConformanceAssert.Parity(expected, actual, n, "zerocentered");
            }
            RmsUnitReference(x, n, expected);
            Ops.RmsUnit(x, n, actual);
            if (n > 0)
            {
                ConformanceAssert.Parity(expected, actual, n, "rmsunit");
                var norm = 0.0;
                for (var i = 0; i < n; i++)
                {
                    norm += (double)actual[i] * actual[i];
                }
                Assert.InRange(Math.Sqrt(norm), 0.99 * Math.Sqrt(n), 1.01 * Math.Sqrt(n));
            }
        }

        [Theory]
        [InlineData(1)]
        [InlineData(7)]
        [InlineData(8)]
        [InlineData(9)]
        [InlineData(33)]
        [InlineData(100)]
        [InlineData(8192)]
        public void Softmax(int n)
        {
            var rng = new XorShift64(0x0DDD);
            var expected = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var actual = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            ConformanceInputs.FillEdges(expected, Math.Max(n, 1), ref rng, 4f);
            for (var i = 0; i < n; i++)
            {
                actual[i] = expected[i];
            }
            Scalar.Softmax(expected, n);
            Ops.Softmax(actual, n);
            if (n > 0)
            {
                ConformanceAssert.ParityWithArgMax(expected, actual, n, "softmax");
                var total = 0.0;
                for (var i = 0; i < n; i++)
                {
                    total += actual[i];
                }
                Assert.InRange(total, 1 - 1e-4 * Math.Max(n / 64, 1), 1 + 1e-4 * Math.Max(n / 64, 1));
                if (n <= 100)
                {
                    ConformanceDump.Dump("softmax_" + n, actual, n * 4);
                }
            }
        }

        [Fact]
        public void SoftmaxGolden()
        {
            var s = stackalloc float[3];
            s[0] = 0; s[1] = 0; s[2] = 0;
            Ops.Softmax(s, 3);
            for (var i = 0; i < 3; i++)
            {
                Assert.InRange(s[i] - 0.33333334f, -1e-6f, 1e-6f);
            }
            s[0] = 1; s[1] = 2; s[2] = 3;
            Ops.Softmax(s, 3);
            Assert.InRange(s[0] - 0.09003057f, -1e-6f, 1e-6f);
            Assert.InRange(s[1] - 0.24472848f, -1e-6f, 1e-6f);
            Assert.InRange(s[2] - 0.66524094f, -1e-6f, 1e-6f);
        }

        [Theory]
        [InlineData(1)]
        [InlineData(7)]
        [InlineData(9)]
        [InlineData(31)]
        [InlineData(100)]
        [InlineData(8192)]
        public void Dot(int n)
        {
            var rng = new XorShift64(0x0DDE);
            var a = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var b = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            ConformanceInputs.FillEdges(a, Math.Max(n, 1), ref rng, 2f);
            ConformanceInputs.FillEdges(b, Math.Max(n, 1), ref rng, 2f);
            var expected = Scalar.Dot(a, b, n);
            var actual = Ops.Dot(a, b, n);
            if (n > 0)
            {
                Assert.InRange(actual - expected, -1e-4f * MathF.Max(1f, MathF.Abs(expected)), 1e-4f * MathF.Max(1f, MathF.Abs(expected)));
            }
        }

        [Fact]
        public void DotGolden()
        {
            var a = stackalloc float[8];
            var b = stackalloc float[8];
            for (var i = 0; i < 8; i++)
            {
                a[i] = i + 1;
                b[i] = 8 - i;
            }
            Assert.Equal(120f, Ops.Dot(a, b, 8));
        }

        [Theory]
        [InlineData(8)]
        [InlineData(64)]
        [InlineData(768)]
        public void AddScaledScaleAddMultiplyAccumulate(int n)
        {
            var rng = new XorShift64(0x0DDF);
            var d = (float*)owned.Alloc((nuint)(n * 4));
            var x = (float*)owned.Alloc((nuint)(n * 4));
            var y = (float*)owned.Alloc((nuint)(n * 4));
            var expected = (float*)owned.Alloc((nuint)(n * 4));
            var actual = (float*)owned.Alloc((nuint)(n * 4));
            ConformanceInputs.FillEdges(d, n, ref rng, 2f);
            ConformanceInputs.FillEdges(x, n, ref rng, 2f);
            ConformanceInputs.FillEdges(y, n, ref rng, 2f);
            for (var i = 0; i < n; i++)
            {
                expected[i] = actual[i] = d[i];
            }
            Scalar.AddScaled(expected, x, 0.37f, n);
            Ops.AddScaled(actual, x, 0.37f, n);
            ConformanceAssert.Parity(expected, actual, n, "addscaled");
            for (var i = 0; i < n; i++)
            {
                expected[i] = actual[i] = d[i];
            }
            Scalar.ScaleAdd(expected, 0.5f, x, 1.5f, n);
            Ops.ScaleAdd(actual, 0.5f, x, 1.5f, n);
            ConformanceAssert.Parity(expected, actual, n, "scaleadd");
            for (var i = 0; i < n; i++)
            {
                expected[i] = actual[i] = d[i];
            }
            Scalar.MultiplyAccumulate(expected, x, y, n);
            Ops.MultiplyAccumulate(actual, x, y, n);
            ConformanceAssert.Parity(expected, actual, n, "mad");
        }

        [Theory]
        [InlineData(8)]
        [InlineData(32)]
        [InlineData(128)]
        public void QuantizeInt8Bounds(int n)
        {
            var rng = new XorShift64(0x0DE0);
            var x = (float*)owned.Alloc((nuint)(Math.Max(n, 1) * 4));
            var expected = (sbyte*)owned.Alloc((nuint)Math.Max(n, 1));
            var actual = (sbyte*)owned.Alloc((nuint)Math.Max(n, 1));
            ConformanceInputs.FillEdges(x, Math.Max(n, 1), ref rng, 3f);
            var expectedScale = Scalar.QuantizeInt8(x, n, expected);
            var actualScale = Quant.QuantizeInt8(x, n, actual);
            Assert.Equal(expectedScale, actualScale);
            for (var i = 0; i < n; i++)
            {
                var drift = actual[i] - expected[i];
                Assert.True(drift >= -1 && drift <= 1, "q8 [" + i + "]: " + drift);
            }
            if (n > 0)
            {
                ConformanceDump.Dump("q8_" + n, actual, n);
            }
        }

        [Fact]
        public void QuantizeInt8Golden()
        {
            var x = stackalloc float[8];
            var dst = stackalloc sbyte[8];
            x[0] = -4; x[1] = 0; x[2] = 4;
            for (var i = 3; i < 8; i++)
            {
                x[i] = 0;
            }
            var scale = Quant.QuantizeInt8(x, 8, dst);
            Assert.InRange(scale - 0.03149606f, -1e-6f, 1e-6f);
            Assert.Equal((sbyte)-127, dst[0]);
            Assert.Equal((sbyte)0, dst[1]);
            Assert.Equal((sbyte)127, dst[2]);
            for (var i = 3; i < 8; i++)
            {
                Assert.Equal((sbyte)0, dst[i]);
            }
        }

        [Fact]
        public void FakeA8SnapsToGrid()
        {
            var x = stackalloc float[8];
            for (var i = 0; i < 8; i++)
            {
                x[i] = (i - 4) * 1.3f;
            }
            var scale = Quant.FakeA8(x, 8);
            var inverse = 1f / scale;
            for (var i = 0; i < 8; i++)
            {
                Assert.Equal(Math.Round(x[i] * inverse), x[i] * inverse, 4);
            }
        }

        [Fact]
        public void SinkhornGolden()
        {
            var logits = stackalloc float[16];
            var result = stackalloc float[16];
            for (var i = 0; i < 16; i++)
            {
                logits[i] = 0f;
            }
            Ops.Sinkhorn4x4(logits, result);
            for (var i = 0; i < 16; i++)
            {
                Assert.Equal(0.25f, result[i]);
            }
        }

        [Fact]
        public void SinkhornRandomIsDoublyStochastic()
        {
            var rng = new XorShift64(0x0DE1);
            var logits = (float*)owned.Alloc(64);
            var expected = (float*)owned.Alloc(64);
            var actual = (float*)owned.Alloc(64);
            ConformanceInputs.FillUniform(logits, 16, ref rng, 2f);
            Scalar.Sinkhorn4x4(logits, expected);
            Ops.Sinkhorn4x4(logits, actual);
            ConformanceAssert.Parity(expected, actual, 16, "sinkhorn");
            for (var r = 0; r < 4; r++)
            {
                var row = actual[r * 4] + actual[r * 4 + 1] + actual[r * 4 + 2] + actual[r * 4 + 3];
                var col = actual[r] + actual[4 + r] + actual[8 + r] + actual[12 + r];
                Assert.InRange(row - 1f, -1e-5f, 1e-5f);
                Assert.InRange(col - 1f, -1e-5f, 1e-5f);
            }
        }

        public void Dispose()
        {
            owned.Dispose();
        }
    }

    public sealed unsafe class MaskedConformance : IDisposable
    {
        readonly HGlobal owned = new HGlobal();

        static int Reference(float* x, uint* mask, int words, int length)
        {
            var best = -1;
            var peak = float.NegativeInfinity;
            for (var w = 0; w < words; w++)
            {
                var bits = mask[w];
                while (bits != 0)
                {
                    var id = w * 32 + BitTrailingZeros(bits);
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

        static int BitTrailingZeros(uint v)
        {
            var n = 0;
            while ((v & 1) == 0)
            {
                v >>= 1;
                n++;
            }
            return n;
        }

        void RunCase(ulong seed, int length, int pattern, string name)
        {
            var words = (length + 31) / 32;
            var rng = new XorShift64(seed);
            var x = (float*)owned.Alloc((nuint)(Math.Max(length, 1) * 4));
            var mask = (uint*)owned.Alloc((nuint)(Math.Max(words, 1) * 4));
            ConformanceInputs.FillEdges(x, Math.Max(length, 1), ref rng, 5f);
            for (var w = 0; w < words; w++)
            {
                mask[w] = pattern == 0 ? 0u : pattern == 1 ? 0xFFFFFFFFu : (uint)rng.Next();
            }
            if (pattern == 2 && words > 0)
            {
                mask[0] = 0;
            }
            if (length % 32 != 0 && words > 0 && pattern == 1)
            {
                mask[words - 1] &= 0xFFFFFFFFu >> (32 - length % 32);
            }
            if (pattern == 3)
            {
                for (var i = 0; i < length; i++)
                {
                    x[i] = 1f;
                }
            }
            var expected = Reference(x, mask, words, length);
            var actual = Ops.ArgMaxMasked(x, mask, words, length);
            Assert.Equal(expected, actual);
            if (length > 0 && pattern != 0)
            {
                ConformanceDump.Dump("argmax_" + name, &actual, 4);
            }
        }

        [Theory]
        [InlineData(0, 0)]
        [InlineData(1, 1)]
        [InlineData(31, 1)]
        [InlineData(32, 1)]
        [InlineData(33, 2)]
        [InlineData(64, 2)]
        [InlineData(100, 1)]
        [InlineData(100, 3)]
        [InlineData(8192, 2)]
        [InlineData(8192, 0)]
        public void ArgMaxMasked(int length, int pattern)
        {
            RunCase(0x0A969, length, pattern, length + "_" + pattern);
        }

        [Fact]
        public void ArgMaxMaskedGolden()
        {
            var x = stackalloc float[64];
            var mask = stackalloc uint[2];
            for (var i = 0; i < 64; i++)
            {
                x[i] = i;
            }
            mask[0] = 0;
            mask[1] = 1u << 5;
            Assert.Equal(37, Ops.ArgMaxMasked(x, mask, 2, 64));
            mask[1] = 0;
            Assert.Equal(-1, Ops.ArgMaxMasked(x, mask, 2, 64));
            for (var i = 0; i < 64; i++)
            {
                x[i] = 2f;
            }
            mask[0] = 0xFFFFFFFFu;
            mask[1] = 0xFFFFFFFFu;
            Assert.Equal(0, Ops.ArgMaxMasked(x, mask, 2, 64));
        }

        [Theory]
        [InlineData(1)]
        [InlineData(7)]
        [InlineData(8)]
        [InlineData(9)]
        [InlineData(8192)]
        public void BestProbability(int count)
        {
            var rng = new XorShift64(0x0BE57);
            var values = (float*)owned.Alloc((nuint)(count * 4));
            ConformanceInputs.FillEdges(values, count, ref rng, 6f);
            var best = float.NegativeInfinity;
            for (var i = 0; i < count; i++)
            {
                best = MathF.Max(best, values[i]);
            }
            float total = 0f;
            for (var i = 0; i < count; i++)
            {
                total += MathF.Exp(values[i] - best);
            }
            var expected = 1f / total;
            var actual = Ops.BestProbability(values, count, best);
            var tolerance = count > 100 ? 5e-5f * expected : 1e-5f * expected;
            Assert.InRange(actual - expected, -tolerance, tolerance);
            if (count == 1)
            {
                Assert.Equal(1f, actual);
            }
        }

        [Fact]
        public void BestProbabilityGolden()
        {
            var values = stackalloc float[3];
            values[0] = 1; values[1] = 2; values[2] = 3;
            Assert.InRange(Ops.BestProbability(values, 3, 3f) - 0.66524094f, -1e-6f, 1e-6f);
        }

        public void Dispose()
        {
            owned.Dispose();
        }
    }
}
