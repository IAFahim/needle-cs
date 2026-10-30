using System;
using System.Runtime.CompilerServices;
using System.Threading;
#if NEEDLE_SIMD
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace Needle;

internal unsafe struct AttentionTile
{
    public sbyte* Queries;
    public float* QueryScales;
    public int Heads;
    public sbyte* Keys;
    public float* KeyScales;
    public sbyte* Values;
    public float* ValueScales;
    public int KeyStride, ValueStride, ScaleStride, First, Count;
    public float* Scores;
    public float* Output;
    public float* Probabilities;
    public int ProbabilityStride;
    public float* Peaks;
    public float* Sums;
}

#if NEEDLE_SIMD
internal interface IHeadCount
{
    static abstract int Value { get; }
}

internal readonly struct OneHead : IHeadCount { public static int Value => 1; }
internal readonly struct TwoHeads : IHeadCount { public static int Value => 2; }
internal readonly struct ThreeHeads : IHeadCount { public static int Value => 3; }
#endif

internal static unsafe class Attention
{
    public const int MaxTileHeads = 6;
    public const int PartialFloats = MaxTileHeads * 64 + 16;

    const int QkDim = 48;
    const int VDim = 64;
    const int Lanes = 8;
    const int PositionBlock = 4;
    const float InverseSqrtQkDim = 0.14433756729740643f;
    const float WeightLevels = 65535f;

#if NEEDLE_SIMD
    static readonly bool UseVnni = AvxVnni.IsSupported && Environment.GetEnvironmentVariable("NEEDLE_NOVNNI") == null;
#endif

    public static bool CanSplitPositions =>
#if NEEDLE_SIMD
        Simd.UseAvx;
#else
        false;
#endif

    public static int ScoreFloats(int count) => (count + PositionBlock) * Lanes;

    public static void Run(in AttentionTile tile)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.AttentionTile(tile); return; }
#endif
        if (!Simd.Enabled) { Scalar.AttentionTile(tile); return; }
#if NEEDLE_SIMD
        if (tile.Heads == 1)
            Head(tile.Queries, tile.QueryScales[0], tile.Keys, tile.KeyScales, tile.Values, tile.ValueScales,
                tile.KeyStride, tile.ValueStride, tile.ScaleStride, tile.First, tile.First + tile.Count - 1, tile.Scores, tile.Output, tile.Probabilities);
        else if (UseVnni) GroupVnni(tile);
        else GroupAvx2(tile);
#endif
    }

#if NEEDLE_SIMD
    public static int DotInt8x48(sbyte* a, sbyte* b)
    {
        var sum = Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(a), Avx2.ConvertToVector256Int16(b));
        sum = Avx2.Add(sum, Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(a + 16), Avx2.ConvertToVector256Int16(b + 16)));
        sum = Avx2.Add(sum, Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(a + 32), Avx2.ConvertToVector256Int16(b + 32)));
        return Vector256.Sum(sum);
    }

    public static void Head(
        sbyte* query, float queryScale, sbyte* keys, float* keyScales, sbyte* values, float* valueScales,
        int keyStride, int valueStride, int scaleStride, int first, int last, float* scores, float* output, float* probabilities = null)
    {
        var count = last - first + 1;
        var scale = queryScale * InverseSqrtQkDim;
        for (var s = 0; s < count; s++)
            scores[s] = DotInt8x48(query, keys + (long)(first + s) * keyStride) * keyScales[(long)(first + s) * scaleStride] * scale;
        Ops.Softmax(scores, count);
        if (probabilities != null) for (var s = 0; s < count; s++) probabilities[first + s] = scores[s];

        var a0 = Vector256<float>.Zero;
        var a1 = Vector256<float>.Zero;
        var a2 = Vector256<float>.Zero;
        var a3 = Vector256<float>.Zero;
        var a4 = Vector256<float>.Zero;
        var a5 = Vector256<float>.Zero;
        var a6 = Vector256<float>.Zero;
        var a7 = Vector256<float>.Zero;
        for (var s = 0; s < count; s++)
        {
            var row = values + (long)(first + s) * valueStride;
            var weight = Vector256.Create(scores[s] * valueScales[(long)(first + s) * scaleStride]);
            a0 = Fma.MultiplyAdd(weight, Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row)), a0);
            a1 = Fma.MultiplyAdd(weight, Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row + 8)), a1);
            a2 = Fma.MultiplyAdd(weight, Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row + 16)), a2);
            a3 = Fma.MultiplyAdd(weight, Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row + 24)), a3);
            a4 = Fma.MultiplyAdd(weight, Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row + 32)), a4);
            a5 = Fma.MultiplyAdd(weight, Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row + 40)), a5);
            a6 = Fma.MultiplyAdd(weight, Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row + 48)), a6);
            a7 = Fma.MultiplyAdd(weight, Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row + 56)), a7);
        }
        Avx.Store(output, a0);
        Avx.Store(output + 8, a1);
        Avx.Store(output + 16, a2);
        Avx.Store(output + 24, a3);
        Avx.Store(output + 32, a4);
        Avx.Store(output + 40, a5);
        Avx.Store(output + 48, a6);
        Avx.Store(output + 56, a7);
    }

    [SkipLocalsInit]
    public static void GroupVnni(in AttentionTile tile)
    {
        var inverse = stackalloc float[Lanes];
        ScoresVnni(tile);
        Normalizers(tile, inverse);
        CaptureProbabilities(tile, inverse);
        ValuesInt8(tile, inverse);
    }

    [SkipLocalsInit]
    public static void GroupAvx2(in AttentionTile tile)
    {
        var inverse = stackalloc float[Lanes];
        ScoresAvx2(tile);
        Normalizers(tile, inverse);
        CaptureProbabilities(tile, inverse);
        ValuesFloat(tile, inverse);
    }

    static void Normalizers(in AttentionTile tile, float* inverse)
    {
        var sum = ExponentiateAcrossPositions(tile.Scores, tile.Count, out var peak);
        if (tile.Peaks == null)
        {
            Avx.Store(inverse, Avx.Divide(Vector256.Create(1f), sum));
            return;
        }
        Avx.Store(tile.Peaks, peak);
        Avx.Store(tile.Sums, sum);
        Avx.Store(inverse, Vector256.Create(1f));
    }

    [SkipLocalsInit]
    public static void MergePartials(float* partials, int count, int heads, float* output)
    {
        // Decode always lands on count == 2: same accumulation order as the
        // generic loop (identical bits), with the weight broadcasts hoisted
        // out of the dim loop.
        var weights = stackalloc float[Lanes];
        for (var h = 0; h < heads; h++)
        {
            if (count == 2)
            {
                var peak0 = partials[MaxTileHeads * VDim + h];
                var peak1 = partials[PartialFloats + MaxTileHeads * VDim + h];
                var pk = MathF.Max(peak0, peak1);
                var w0 = MathF.Exp(peak0 - pk);
                var w1 = MathF.Exp(peak1 - pk);
                var tot = 0f;
                tot += w0 * partials[MaxTileHeads * VDim + Lanes + h];
                tot += w1 * partials[PartialFloats + MaxTileHeads * VDim + Lanes + h];
                var inv = Vector256.Create(1f / tot);
                var w0v = Vector256.Create(w0);
                var w1v = Vector256.Create(w1);
                for (var d = 0; d < VDim; d += Lanes)
                {
                    var sum = Vector256<float>.Zero;
                    sum = Fma.MultiplyAdd(w0v, Avx.LoadVector256(partials + h * VDim + d), sum);
                    sum = Fma.MultiplyAdd(w1v, Avx.LoadVector256(partials + PartialFloats + h * VDim + d), sum);
                    Avx.Store(output + h * VDim + d, Avx.Multiply(sum, inv));
                }
                continue;
            }
            var peak = float.NegativeInfinity;
            for (var c = 0; c < count; c++) peak = MathF.Max(peak, partials[c * PartialFloats + MaxTileHeads * VDim + h]);
            var total = 0f;
            for (var c = 0; c < count; c++)
            {
                weights[c] = MathF.Exp(partials[c * PartialFloats + MaxTileHeads * VDim + h] - peak);
                total += weights[c] * partials[c * PartialFloats + MaxTileHeads * VDim + Lanes + h];
            }
            var inverse = Vector256.Create(1f / total);
            for (var d = 0; d < VDim; d += Lanes)
            {
                var sum = Vector256<float>.Zero;
                for (var c = 0; c < count; c++)
                    sum = Fma.MultiplyAdd(Vector256.Create(weights[c]), Avx.LoadVector256(partials + c * PartialFloats + h * VDim + d), sum);
                Avx.Store(output + h * VDim + d, Avx.Multiply(sum, inverse));
            }
        }
    }

    [SkipLocalsInit]
    public static void ScoresVnni(in AttentionTile tile)
    {
        var queries = stackalloc byte[QkDim / 4 * 32];
        PackVnniQueries(tile.Queries, tile.Heads, (uint*)queries);
        var scale = ScaleVector(tile);
        var biasLane = Vector256.Create(tile.Heads);
        var count = tile.Count;
        var last = count - 1;
        var keys = tile.Keys + (long)tile.First * tile.KeyStride;
        var keyStride = tile.KeyStride;
        var keyScales = tile.KeyScales + (long)tile.First * tile.ScaleStride;
        var scaleStride = tile.ScaleStride;
        var scores = tile.Scores;
        for (var s = 0; s < count; s += PositionBlock)
        {
            var row0 = (int*)(keys + (long)s * keyStride);
            var row1 = (int*)(keys + (long)Math.Min(s + 1, last) * keyStride);
            var row2 = (int*)(keys + (long)Math.Min(s + 2, last) * keyStride);
            var row3 = (int*)(keys + (long)Math.Min(s + 3, last) * keyStride);
            var even0 = Vector256<int>.Zero;
            var even1 = even0;
            var even2 = even0;
            var even3 = even0;
            var odd0 = even0;
            var odd1 = even0;
            var odd2 = even0;
            var odd3 = even0;
            for (var j = 0; j < QkDim / 4; j += 2)
            {
                var queryEven = Avx.LoadVector256(queries + j * 32);
                var queryOdd = Avx.LoadVector256(queries + (j + 1) * 32);
                even0 = AvxVnni.MultiplyWideningAndAdd(even0, queryEven, Vector256.Create(row0[j]).AsSByte());
                even1 = AvxVnni.MultiplyWideningAndAdd(even1, queryEven, Vector256.Create(row1[j]).AsSByte());
                even2 = AvxVnni.MultiplyWideningAndAdd(even2, queryEven, Vector256.Create(row2[j]).AsSByte());
                even3 = AvxVnni.MultiplyWideningAndAdd(even3, queryEven, Vector256.Create(row3[j]).AsSByte());
                odd0 = AvxVnni.MultiplyWideningAndAdd(odd0, queryOdd, Vector256.Create(row0[j + 1]).AsSByte());
                odd1 = AvxVnni.MultiplyWideningAndAdd(odd1, queryOdd, Vector256.Create(row1[j + 1]).AsSByte());
                odd2 = AvxVnni.MultiplyWideningAndAdd(odd2, queryOdd, Vector256.Create(row2[j + 1]).AsSByte());
                odd3 = AvxVnni.MultiplyWideningAndAdd(odd3, queryOdd, Vector256.Create(row3[j + 1]).AsSByte());
            }
            Avx.Store(scores + s * Lanes, Score(Debias(Avx2.Add(even0, odd0), biasLane), keyScales[(long)s * scaleStride], scale));
            Avx.Store(scores + (s + 1) * Lanes, Score(Debias(Avx2.Add(even1, odd1), biasLane), keyScales[(long)Math.Min(s + 1, last) * scaleStride], scale));
            Avx.Store(scores + (s + 2) * Lanes, Score(Debias(Avx2.Add(even2, odd2), biasLane), keyScales[(long)Math.Min(s + 2, last) * scaleStride], scale));
            Avx.Store(scores + (s + 3) * Lanes, Score(Debias(Avx2.Add(even3, odd3), biasLane), keyScales[(long)Math.Min(s + 3, last) * scaleStride], scale));
        }
    }

    [SkipLocalsInit]
    public static void ScoresAvx2(in AttentionTile tile)
    {
        var queries = stackalloc int[QkDim / 2 * Lanes];
        for (var j = 0; j < QkDim / 2; j++)
            for (var h = 0; h < Lanes; h++)
            {
                if (h >= tile.Heads) { queries[j * Lanes + h] = 0; continue; }
                var q = tile.Queries + h * QkDim + 2 * j;
                queries[j * Lanes + h] = (int)(ushort)(short)q[0] | q[1] << 16;
            }
        var converted = stackalloc short[PositionBlock * QkDim];
        var scale = ScaleVector(tile);
        var count = tile.Count;
        var last = count - 1;
        var keys = tile.Keys + (long)tile.First * tile.KeyStride;
        var keyStride = tile.KeyStride;
        var keyScales = tile.KeyScales + (long)tile.First * tile.ScaleStride;
        var scaleStride = tile.ScaleStride;
        var scores = tile.Scores;
        var key1 = (int*)(converted + QkDim);
        var key2 = (int*)(converted + 2 * QkDim);
        var key3 = (int*)(converted + 3 * QkDim);
        for (var s = 0; s < count; s += PositionBlock)
        {
            for (var p = 0; p < PositionBlock; p++)
            {
                var row = keys + (long)Math.Min(s + p, last) * keyStride;
                Avx.Store(converted + p * QkDim, Avx2.ConvertToVector256Int16(row));
                Avx.Store(converted + p * QkDim + 16, Avx2.ConvertToVector256Int16(row + 16));
                Avx.Store(converted + p * QkDim + 32, Avx2.ConvertToVector256Int16(row + 32));
            }
            var sum0 = Vector256<int>.Zero;
            var sum1 = sum0;
            var sum2 = sum0;
            var sum3 = sum0;
            for (var j = 0; j < QkDim / 2; j++)
            {
                var query = Avx.LoadVector256(queries + j * Lanes).AsInt16();
                sum0 = Avx2.Add(sum0, Avx2.MultiplyAddAdjacent(query, Vector256.Create(((int*)converted)[j]).AsInt16()));
                sum1 = Avx2.Add(sum1, Avx2.MultiplyAddAdjacent(query, Vector256.Create(key1[j]).AsInt16()));
                sum2 = Avx2.Add(sum2, Avx2.MultiplyAddAdjacent(query, Vector256.Create(key2[j]).AsInt16()));
                sum3 = Avx2.Add(sum3, Avx2.MultiplyAddAdjacent(query, Vector256.Create(key3[j]).AsInt16()));
            }
            Avx.Store(scores + s * Lanes, Score(sum0, keyScales[(long)s * scaleStride], scale));
            Avx.Store(scores + (s + 1) * Lanes, Score(sum1, keyScales[(long)Math.Min(s + 1, last) * scaleStride], scale));
            Avx.Store(scores + (s + 2) * Lanes, Score(sum2, keyScales[(long)Math.Min(s + 2, last) * scaleStride], scale));
            Avx.Store(scores + (s + 3) * Lanes, Score(sum3, keyScales[(long)Math.Min(s + 3, last) * scaleStride], scale));
        }
    }

    static void PackVnniQueries(sbyte* queries, int heads, uint* packed)
    {
        // Branchless: clear once with vector stores, pack only live heads,
        // then stamp the ones-lane used by Debias. Zeroed lanes contribute
        // nothing downstream (their scale lane is 0 in ScaleVector).
        for (var j = 0; j < QkDim / 4; j++) Avx.Store(packed + j * Lanes, Vector256<uint>.Zero);
        for (var j = 0; j < QkDim / 4; j++)
        {
            for (var h = 0; h < heads; h++)
            {
                var q = queries + h * QkDim + 4 * j;
                packed[j * Lanes + h] = (uint)(byte)(q[0] + 128) | (uint)(byte)(q[1] + 128) << 8 | (uint)(byte)(q[2] + 128) << 16 | (uint)(byte)(q[3] + 128) << 24;
            }
            packed[j * Lanes + heads] = 0x01010101u;
        }
    }

    static Vector256<float> ScaleVector(in AttentionTile tile)
    {
        var lanes = stackalloc float[Lanes];
        var scales = tile.QueryScales;
        var heads = tile.Heads;
        lanes[0] = scales[0] * InverseSqrtQkDim;
        lanes[1] = heads > 1 ? scales[1] * InverseSqrtQkDim : 0f;
        lanes[2] = heads > 2 ? scales[2] * InverseSqrtQkDim : 0f;
        lanes[3] = heads > 3 ? scales[3] * InverseSqrtQkDim : 0f;
        lanes[4] = heads > 4 ? scales[4] * InverseSqrtQkDim : 0f;
        lanes[5] = heads > 5 ? scales[5] * InverseSqrtQkDim : 0f;
        lanes[6] = 0f;
        lanes[7] = 0f;
        return Avx.LoadVector256(lanes);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<int> Debias(Vector256<int> sum, Vector256<int> biasLane) =>
        Avx2.Subtract(sum, Avx2.ShiftLeftLogical(Avx2.PermuteVar8x32(sum, biasLane), 7));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> Score(Vector256<int> dot, float keyScale, Vector256<float> scale) =>
        Avx.Multiply(Avx.Multiply(Avx.ConvertToVector256Single(dot), Vector256.Create(keyScale)), scale);

    public static Vector256<float> ExponentiateAcrossPositions(float* scores, int count, out Vector256<float> peak)
    {
        var peak0 = Vector256.Create(float.NegativeInfinity);
        var peak1 = peak0;
        var peak2 = peak0;
        var peak3 = peak0;
        var s = 0;
        for (; s + 4 <= count; s += 4)
        {
            peak0 = Avx.Max(peak0, Avx.LoadVector256(scores + s * Lanes));
            peak1 = Avx.Max(peak1, Avx.LoadVector256(scores + (s + 1) * Lanes));
            peak2 = Avx.Max(peak2, Avx.LoadVector256(scores + (s + 2) * Lanes));
            peak3 = Avx.Max(peak3, Avx.LoadVector256(scores + (s + 3) * Lanes));
        }
        for (; s < count; s++) peak0 = Avx.Max(peak0, Avx.LoadVector256(scores + s * Lanes));
        peak = Avx.Max(Avx.Max(peak0, peak1), Avx.Max(peak2, peak3));

        var sum0 = Vector256<float>.Zero;
        var sum1 = sum0;
        var sum2 = sum0;
        var sum3 = sum0;
        for (s = 0; s + 4 <= count; s += 4)
        {
            var e0 = Ops.Exp(Avx.Subtract(Avx.LoadVector256(scores + s * Lanes), peak));
            var e1 = Ops.Exp(Avx.Subtract(Avx.LoadVector256(scores + (s + 1) * Lanes), peak));
            var e2 = Ops.Exp(Avx.Subtract(Avx.LoadVector256(scores + (s + 2) * Lanes), peak));
            var e3 = Ops.Exp(Avx.Subtract(Avx.LoadVector256(scores + (s + 3) * Lanes), peak));
            Avx.Store(scores + s * Lanes, e0);
            Avx.Store(scores + (s + 1) * Lanes, e1);
            Avx.Store(scores + (s + 2) * Lanes, e2);
            Avx.Store(scores + (s + 3) * Lanes, e3);
            sum0 = Avx.Add(sum0, e0);
            sum1 = Avx.Add(sum1, e1);
            sum2 = Avx.Add(sum2, e2);
            sum3 = Avx.Add(sum3, e3);
        }
        for (; s < count; s++)
        {
            var e = Ops.Exp(Avx.Subtract(Avx.LoadVector256(scores + s * Lanes), peak));
            Avx.Store(scores + s * Lanes, e);
            sum0 = Avx.Add(sum0, e);
        }
        return Avx.Add(Avx.Add(sum0, sum1), Avx.Add(sum2, sum3));
    }

    static void CaptureProbabilities(in AttentionTile tile, float* inverse)
    {
        if (tile.Probabilities == null) return;
        for (var h = 0; h < tile.Heads; h++)
            for (var s = 0; s < tile.Count; s++)
                tile.Probabilities[h * tile.ProbabilityStride + tile.First + s] = tile.Scores[s * Lanes + h] * inverse[h];
    }

    public static void ValuesFloat(in AttentionTile tile, float* inverse)
    {
        var scores = tile.Scores;
        var valueScales = tile.ValueScales + (long)tile.First * tile.ScaleStride;
        for (var s = 0; s < tile.Count; s++)
            Avx.Store(scores + s * Lanes, Avx.Multiply(Avx.LoadVector256(scores + s * Lanes), Vector256.Create(valueScales[(long)s * tile.ScaleStride])));
        for (var firstHead = 0; firstHead < tile.Heads; firstHead += 3)
            for (var firstDim = 0; firstDim < VDim; firstDim += 32)
                switch (tile.Heads - firstHead)
                {
                    case 1: AccumulateFloat<OneHead>(tile, inverse, firstHead, firstDim); break;
                    case 2: AccumulateFloat<TwoHeads>(tile, inverse, firstHead, firstDim); break;
                    default: AccumulateFloat<ThreeHeads>(tile, inverse, firstHead, firstDim); break;
                }
    }

    static void AccumulateFloat<THeads>(in AttentionTile tile, float* inverse, int firstHead, int firstDim) where THeads : struct, IHeadCount
    {
        var heads = THeads.Value;
        var a00 = Vector256<float>.Zero;
        var a01 = a00;
        var a02 = a00;
        var a03 = a00;
        var a10 = a00;
        var a11 = a00;
        var a12 = a00;
        var a13 = a00;
        var a20 = a00;
        var a21 = a00;
        var a22 = a00;
        var a23 = a00;
        var count = tile.Count;
        var rows = tile.Values + (long)tile.First * tile.ValueStride + firstDim;
        var rowStride = tile.ValueStride;
        var weights = tile.Scores + firstHead;
        for (var s = 0; s < count; s++)
        {
            var row = rows + (long)s * rowStride;
            var v0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row));
            var v1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row + 8));
            var v2 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row + 16));
            var v3 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(row + 24));
            var w0 = Avx.BroadcastScalarToVector256(weights + s * Lanes);
            a00 = Fma.MultiplyAdd(w0, v0, a00);
            a01 = Fma.MultiplyAdd(w0, v1, a01);
            a02 = Fma.MultiplyAdd(w0, v2, a02);
            a03 = Fma.MultiplyAdd(w0, v3, a03);
            if (heads > 1)
            {
                var w1 = Avx.BroadcastScalarToVector256(weights + s * Lanes + 1);
                a10 = Fma.MultiplyAdd(w1, v0, a10);
                a11 = Fma.MultiplyAdd(w1, v1, a11);
                a12 = Fma.MultiplyAdd(w1, v2, a12);
                a13 = Fma.MultiplyAdd(w1, v3, a13);
            }
            if (heads > 2)
            {
                var w2 = Avx.BroadcastScalarToVector256(weights + s * Lanes + 2);
                a20 = Fma.MultiplyAdd(w2, v0, a20);
                a21 = Fma.MultiplyAdd(w2, v1, a21);
                a22 = Fma.MultiplyAdd(w2, v2, a22);
                a23 = Fma.MultiplyAdd(w2, v3, a23);
            }
        }
        var output = tile.Output + firstHead * VDim + firstDim;
        StoreScaled(output, a00, a01, a02, a03, inverse[firstHead]);
        if (heads > 1) StoreScaled(output + VDim, a10, a11, a12, a13, inverse[firstHead + 1]);
        if (heads > 2) StoreScaled(output + 2 * VDim, a20, a21, a22, a23, inverse[firstHead + 2]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void StoreScaled(float* output, Vector256<float> a0, Vector256<float> a1, Vector256<float> a2, Vector256<float> a3, float factor)
    {
        var scale = Vector256.Create(factor);
        Avx.Store(output, Avx.Multiply(a0, scale));
        Avx.Store(output + 8, Avx.Multiply(a1, scale));
        Avx.Store(output + 16, Avx.Multiply(a2, scale));
        Avx.Store(output + 24, Avx.Multiply(a3, scale));
    }

    [SkipLocalsInit]
    public static void ValuesInt8(in AttentionTile tile, float* inverse)
    {
        var count = tile.Count;
        var last = count - 1;
        var blocks = (count + PositionBlock - 1) / PositionBlock;
        var scores = tile.Scores;
        var valueScales = tile.ValueScales + (long)tile.First * tile.ScaleStride;
        var scaleStride = tile.ScaleStride;
        for (var s = count; s < blocks * PositionBlock; s++) Avx.Store(scores + s * Lanes, Vector256<float>.Zero);
        var peak0 = Vector256<float>.Zero;
        var peak1 = peak0;
        var peak2 = peak0;
        var peak3 = peak0;
        for (var s = 0; s < blocks * PositionBlock; s += PositionBlock)
        {
            var at = scores + s * Lanes;
            var w0 = Avx.Multiply(Avx.LoadVector256(at), Vector256.Create(valueScales[(long)s * scaleStride]));
            var w1 = Avx.Multiply(Avx.LoadVector256(at + Lanes), Vector256.Create(valueScales[(long)Math.Min(s + 1, last) * scaleStride]));
            var w2 = Avx.Multiply(Avx.LoadVector256(at + 2 * Lanes), Vector256.Create(valueScales[(long)Math.Min(s + 2, last) * scaleStride]));
            var w3 = Avx.Multiply(Avx.LoadVector256(at + 3 * Lanes), Vector256.Create(valueScales[(long)Math.Min(s + 3, last) * scaleStride]));
            Avx.Store(at, w0);
            Avx.Store(at + Lanes, w1);
            Avx.Store(at + 2 * Lanes, w2);
            Avx.Store(at + 3 * Lanes, w3);
            peak0 = Avx.Max(peak0, w0);
            peak1 = Avx.Max(peak1, w1);
            peak2 = Avx.Max(peak2, w2);
            peak3 = Avx.Max(peak3, w3);
        }
        var peak = Avx.Max(Avx.Max(peak0, peak1), Avx.Max(peak2, peak3));
        var positive = Avx.Compare(peak, Vector256<float>.Zero, FloatComparisonMode.OrderedGreaterThanNonSignaling);
        var quantize = Avx.And(Avx.Divide(Vector256.Create(WeightLevels), peak), positive);
        var factors = stackalloc float[Lanes];
        Avx.Store(factors, Avx.Multiply(Avx.Divide(peak, Vector256.Create(WeightLevels)), Avx.LoadVector256(inverse)));
        var transpose = Vector256.Create(0x0C080400u, 0x0D090501u, 0x0E0A0602u, 0x0F0B0703u, 0x0C080400u, 0x0D090501u, 0x0E0A0602u, 0x0F0B0703u).AsByte();
        var levels = Vector256.Create((int)WeightLevels);
        var lowMask = Vector256.Create(255);
        var weights = (byte*)scores;
        for (var b = 0; b < blocks; b++)
        {
            var at = scores + b * PositionBlock * Lanes;
            var x0 = Avx2.Min(Avx.ConvertToVector256Int32(Avx.Multiply(Avx.LoadVector256(at), quantize)), levels);
            var x1 = Avx2.Min(Avx.ConvertToVector256Int32(Avx.Multiply(Avx.LoadVector256(at + Lanes), quantize)), levels);
            var x2 = Avx2.Min(Avx.ConvertToVector256Int32(Avx.Multiply(Avx.LoadVector256(at + 2 * Lanes), quantize)), levels);
            var x3 = Avx2.Min(Avx.ConvertToVector256Int32(Avx.Multiply(Avx.LoadVector256(at + 3 * Lanes), quantize)), levels);
            var upper01 = Avx2.PackUnsignedSaturate(Avx2.ShiftRightLogical(x0, 8), Avx2.ShiftRightLogical(x1, 8)).AsInt16();
            var upper23 = Avx2.PackUnsignedSaturate(Avx2.ShiftRightLogical(x2, 8), Avx2.ShiftRightLogical(x3, 8)).AsInt16();
            var lower01 = Avx2.PackUnsignedSaturate(Avx2.And(x0, lowMask), Avx2.And(x1, lowMask)).AsInt16();
            var lower23 = Avx2.PackUnsignedSaturate(Avx2.And(x2, lowMask), Avx2.And(x3, lowMask)).AsInt16();
            Avx.Store(weights + b * 64, Avx2.Shuffle(Avx2.PackUnsignedSaturate(upper01, upper23), transpose));
            Avx.Store(weights + b * 64 + 32, Avx2.Shuffle(Avx2.PackUnsignedSaturate(lower01, lower23), transpose));
        }
        for (var firstHead = 0; firstHead < tile.Heads; firstHead += 3)
            for (var firstDim = 0; firstDim < VDim; firstDim += 32)
                switch (tile.Heads - firstHead)
                {
                    case 1: AccumulateInt8<OneHead>(tile, weights, factors, blocks, firstHead, firstDim); break;
                    case 2: AccumulateInt8<TwoHeads>(tile, weights, factors, blocks, firstHead, firstDim); break;
                    default: AccumulateInt8<ThreeHeads>(tile, weights, factors, blocks, firstHead, firstDim); break;
                }
    }

    static void AccumulateInt8<THeads>(in AttentionTile tile, byte* weights, float* factors, int blocks, int firstHead, int firstDim) where THeads : struct, IHeadCount
    {
        var heads = THeads.Value;
        var h00 = Vector256<int>.Zero;
        var h01 = h00;
        var h02 = h00;
        var h03 = h00;
        var h10 = h00;
        var h11 = h00;
        var h12 = h00;
        var h13 = h00;
        var h20 = h00;
        var h21 = h00;
        var h22 = h00;
        var h23 = h00;
        var l00 = h00;
        var l01 = h00;
        var l02 = h00;
        var l03 = h00;
        var l10 = h00;
        var l11 = h00;
        var l12 = h00;
        var l13 = h00;
        var l20 = h00;
        var l21 = h00;
        var l22 = h00;
        var l23 = h00;
        var last = tile.Count - 1;
        var rows = tile.Values + (long)tile.First * tile.ValueStride + firstDim;
        var rowStride = tile.ValueStride;
        for (var b = 0; b < blocks; b++)
        {
            var s = b * PositionBlock;
            var r0 = Avx.LoadVector256(rows + (long)s * rowStride);
            var r1 = Avx.LoadVector256(rows + (long)Math.Min(s + 1, last) * rowStride);
            var r2 = Avx.LoadVector256(rows + (long)Math.Min(s + 2, last) * rowStride);
            var r3 = Avx.LoadVector256(rows + (long)Math.Min(s + 3, last) * rowStride);
            var low01 = Avx2.UnpackLow(r0, r1).AsInt16();
            var high01 = Avx2.UnpackHigh(r0, r1).AsInt16();
            var low23 = Avx2.UnpackLow(r2, r3).AsInt16();
            var high23 = Avx2.UnpackHigh(r2, r3).AsInt16();
            var v0 = Avx2.UnpackLow(low01, low23).AsSByte();
            var v1 = Avx2.UnpackHigh(low01, low23).AsSByte();
            var v2 = Avx2.UnpackLow(high01, high23).AsSByte();
            var v3 = Avx2.UnpackHigh(high01, high23).AsSByte();
            var packed = (uint*)weights + b * 2 * Lanes + firstHead;
            var upper = Vector256.Create(packed[0]).AsByte();
            var lower = Vector256.Create(packed[Lanes]).AsByte();
            h00 = AvxVnni.MultiplyWideningAndAdd(h00, upper, v0);
            h01 = AvxVnni.MultiplyWideningAndAdd(h01, upper, v1);
            h02 = AvxVnni.MultiplyWideningAndAdd(h02, upper, v2);
            h03 = AvxVnni.MultiplyWideningAndAdd(h03, upper, v3);
            l00 = AvxVnni.MultiplyWideningAndAdd(l00, lower, v0);
            l01 = AvxVnni.MultiplyWideningAndAdd(l01, lower, v1);
            l02 = AvxVnni.MultiplyWideningAndAdd(l02, lower, v2);
            l03 = AvxVnni.MultiplyWideningAndAdd(l03, lower, v3);
            if (heads > 1)
            {
                upper = Vector256.Create(packed[1]).AsByte();
                lower = Vector256.Create(packed[Lanes + 1]).AsByte();
                h10 = AvxVnni.MultiplyWideningAndAdd(h10, upper, v0);
                h11 = AvxVnni.MultiplyWideningAndAdd(h11, upper, v1);
                h12 = AvxVnni.MultiplyWideningAndAdd(h12, upper, v2);
                h13 = AvxVnni.MultiplyWideningAndAdd(h13, upper, v3);
                l10 = AvxVnni.MultiplyWideningAndAdd(l10, lower, v0);
                l11 = AvxVnni.MultiplyWideningAndAdd(l11, lower, v1);
                l12 = AvxVnni.MultiplyWideningAndAdd(l12, lower, v2);
                l13 = AvxVnni.MultiplyWideningAndAdd(l13, lower, v3);
            }
            if (heads > 2)
            {
                upper = Vector256.Create(packed[2]).AsByte();
                lower = Vector256.Create(packed[Lanes + 2]).AsByte();
                h20 = AvxVnni.MultiplyWideningAndAdd(h20, upper, v0);
                h21 = AvxVnni.MultiplyWideningAndAdd(h21, upper, v1);
                h22 = AvxVnni.MultiplyWideningAndAdd(h22, upper, v2);
                h23 = AvxVnni.MultiplyWideningAndAdd(h23, upper, v3);
                l20 = AvxVnni.MultiplyWideningAndAdd(l20, lower, v0);
                l21 = AvxVnni.MultiplyWideningAndAdd(l21, lower, v1);
                l22 = AvxVnni.MultiplyWideningAndAdd(l22, lower, v2);
                l23 = AvxVnni.MultiplyWideningAndAdd(l23, lower, v3);
            }
        }
        var output = tile.Output + firstHead * VDim + firstDim;
        StoreDequantized(output, h00, h01, h02, h03, l00, l01, l02, l03, factors[firstHead]);
        if (heads > 1) StoreDequantized(output + VDim, h10, h11, h12, h13, l10, l11, l12, l13, factors[firstHead + 1]);
        if (heads > 2) StoreDequantized(output + 2 * VDim, h20, h21, h22, h23, l20, l21, l22, l23, factors[firstHead + 2]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void StoreDequantized(
        float* output, Vector256<int> h0, Vector256<int> h1, Vector256<int> h2, Vector256<int> h3,
        Vector256<int> l0, Vector256<int> l1, Vector256<int> l2, Vector256<int> l3, float factor)
    {
        var scale = Vector256.Create(factor);
        var plane = Vector256.Create(256f);
        var f0 = Avx.Multiply(Fma.MultiplyAdd(Avx.ConvertToVector256Single(h0), plane, Avx.ConvertToVector256Single(l0)), scale);
        var f1 = Avx.Multiply(Fma.MultiplyAdd(Avx.ConvertToVector256Single(h1), plane, Avx.ConvertToVector256Single(l1)), scale);
        var f2 = Avx.Multiply(Fma.MultiplyAdd(Avx.ConvertToVector256Single(h2), plane, Avx.ConvertToVector256Single(l2)), scale);
        var f3 = Avx.Multiply(Fma.MultiplyAdd(Avx.ConvertToVector256Single(h3), plane, Avx.ConvertToVector256Single(l3)), scale);
        Avx.Store(output, Avx.Permute2x128(f0, f1, 0x20));
        Avx.Store(output + 8, Avx.Permute2x128(f2, f3, 0x20));
        Avx.Store(output + 16, Avx.Permute2x128(f0, f1, 0x31));
        Avx.Store(output + 24, Avx.Permute2x128(f2, f3, 0x31));
    }
#endif
}

internal unsafe struct AttentionBatch
{
    const int MaxChunks = 8;
    const int MinChunkPositions = 48;

    public sbyte* Queries;
    public float* QueryScales;
    public sbyte* Keys;
    public float* KeyScales;
    public sbyte* Values;
    public float* ValueScales;
    public float* Output;
    public float* Probabilities;
    public int Heads, KvHeads, QkDim, VDim, Position, Window, Capacity, Frames;
    float* partials;
    int* arrivals;
    int tileHeads, tilesPerGroup, chunks, planGroups;

    public static void Run(Pool pool, AttentionBatch* batch, int tokens)
    {
        batch->Plan(pool.Threads, tokens, null, null);
        if (batch->chunks == 1)
        {
            pool.Run(&Execute, batch, batch->TeamTasks);
            return;
        }
        RunChunked(pool, batch);
    }

    public void Plan(int threads, int tokens, float* partials, int* arrivals)
    {
        var group = Heads / KvHeads;
        tilesPerGroup = (group + Attention.MaxTileHeads - 1) / Attention.MaxTileHeads;
        tileHeads = (group + tilesPerGroup - 1) / tilesPerGroup;
        planGroups = tokens * KvHeads;
        var splittable = Attention.CanSplitPositions && tilesPerGroup == 1 && group > 1 && Probabilities == null;
        chunks = splittable ? Math.Clamp((threads + planGroups - 1) / planGroups, 1, MaxChunks) : 1;
        if (chunks == 1 || partials == null || arrivals == null) return;
        this.partials = partials;
        this.arrivals = arrivals;
    }

    public int TeamTasks => planGroups * (chunks == 1 ? tilesPerGroup : chunks);

    public static void RunTask(AttentionBatch* batch, int task)
    {
        if (batch->chunks == 1) Execute(batch, task);
        else ExecuteChunk(batch, task);
    }

    [SkipLocalsInit]
    static void RunChunked(Pool pool, AttentionBatch* batch)
    {
        var partials = stackalloc float[batch->planGroups * batch->chunks * Attention.PartialFloats];
        var arrivals = stackalloc int[batch->planGroups];
        batch->Plan(pool.Threads, batch->planGroups / batch->KvHeads, partials, arrivals);
        for (var i = 0; i < batch->planGroups; i++) arrivals[i] = 0;
        pool.Run(&ExecuteChunk, batch, batch->TeamTasks);
    }

    static void Execute(void* context, int task) => Execute((AttentionBatch*)context, task);

    static void ExecuteChunk(void* context, int task) => ExecuteChunk((AttentionBatch*)context, task);

    [SkipLocalsInit]
    static void Execute(AttentionBatch* b, int task)
    {
        var group = b->Heads / b->KvHeads;
        var token = task / (b->KvHeads * b->tilesPerGroup);
        var kv = task / b->tilesPerGroup % b->KvHeads;
        var firstHead = kv * group + task % b->tilesPerGroup * b->tileHeads;
        b->Span(token, out var first, out var count);
        var scores = stackalloc float[Attention.ScoreFloats(count)];
        var work = b->Tile(token, kv, firstHead, Math.Min(b->tileHeads, (kv + 1) * group - firstHead), first, count, scores);
        work.Output = b->Output + (long)token * b->Heads * b->VDim + firstHead * b->VDim;
        work.Probabilities = b->Probabilities == null ? null : b->Probabilities + ((long)token * b->Heads + firstHead) * b->Frames;
        work.ProbabilityStride = b->Frames;
        Attention.Run(work);
    }

    [SkipLocalsInit]
    static void ExecuteChunk(AttentionBatch* b, int task)
    {
        var group = b->Heads / b->KvHeads;
        var chunk = task % b->chunks;
        var groupIndex = task / b->chunks;
        var token = groupIndex / b->KvHeads;
        var kv = groupIndex % b->KvHeads;
        b->Span(token, out var first, out var count);
        var needed = Math.Clamp(count / MinChunkPositions, 1, b->chunks);
        var length = ((count + needed - 1) / needed + 3) & ~3;
        needed = (count + length - 1) / length;
        if (chunk >= needed) return;
        var begin = chunk * length;
        var slot = b->partials + (long)(groupIndex * b->chunks + chunk) * Attention.PartialFloats;
        var scores = stackalloc float[Attention.ScoreFloats(Math.Min(length, count - begin))];
        var work = b->Tile(token, kv, kv * group, group, first + begin, Math.Min(length, count - begin), scores);
        work.Output = slot;
        work.Peaks = slot + Attention.MaxTileHeads * 64;
        work.Sums = work.Peaks + 8;
        Attention.Run(work);
#if NEEDLE_SIMD
        if (Interlocked.Increment(ref b->arrivals[groupIndex]) == needed)
            Attention.MergePartials(b->partials + (long)groupIndex * b->chunks * Attention.PartialFloats, needed, group, b->Output + (long)token * b->Heads * b->VDim + kv * group * b->VDim);
#endif
    }

    void Span(int token, out int first, out int count)
    {
        var position = Position + token;
        first = Window == 0 ? 0 : Math.Max(0, position - Window + 1);
        count = (Frames > 0 ? Frames - 1 : position) - first + 1;
    }

    AttentionTile Tile(int token, int kv, int firstHead, int heads, int first, int count, float* scores) => new()
    {
        Queries = Queries + (long)token * Heads * QkDim + firstHead * QkDim,
        QueryScales = QueryScales + token * Heads + firstHead,
        Heads = heads,
        Keys = Keys + kv * QkDim, KeyScales = KeyScales + kv,
        Values = Values + kv * VDim, ValueScales = ValueScales + kv,
        KeyStride = KvHeads * QkDim, ValueStride = KvHeads * VDim, ScaleStride = KvHeads,
        First = first, Count = count, Scores = scores,
    };
}

internal unsafe struct MlpBatch
{
    public MlpWeights* Weights;
    public float* PreNorm;
    public int* PermA;
    public int* PermB;
    public float* Residual;
    public int Dim;
    public int Rows;
    public int Cols;
    public float Scale;

    public int Width => Rows * Cols;

    public static void Run(Pool pool, MlpBatch* batch, int tokens) => pool.Run(&Execute, batch, tokens);

    public static void RunToken(MlpBatch* batch, int token)
    {
        var scratch = stackalloc float[HadamardMlp.ScratchFloats(batch->Dim, batch->Width)];
        HadamardMlp.Apply(batch, batch->Residual + (long)token * batch->Dim, scratch);
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void Execute(void* context, int token) => RunToken((MlpBatch*)context, token);
}

[System.Runtime.CompilerServices.SkipLocalsInit]
internal static unsafe class HadamardMlp
{
    const int Conditions = 8;
    const int TileRows = 4;
    const int TileColumns = 16;

    // +8 floats of headroom so Apply can align the base to 32 bytes: every
    // partition offset (Conditions, dim, width) is a multiple of 8 floats on
    // supported geometries, so one alignment keeps weights/normalized/front/back
    // 32-byte aligned and avoids split AVX loads on 16-byte stackalloc scratch.
    public static int ScratchFloats(int dim, int width) => Conditions + dim + 2 * width + 8;

    static bool Blocked(int rows, int cols) => Simd.Enabled && rows % TileRows == 0 && cols % TileColumns == 0;

    public static void Apply(MlpBatch* b, float* residual, float* scratch)
    {
        var w = b->Weights;
        var dim = b->Dim;
        var rows = b->Rows;
        var cols = b->Cols;
        var width = b->Width;
        var inputRows = (dim + cols - 1) / cols;
        var outputRows = Math.Min(rows, (inputRows + TileRows - 1) / TileRows * TileRows);
        var aligned = (float*)(((nuint)scratch + 31) & ~(nuint)31);
        var weights = aligned;
        var normalized = weights + Conditions;
        var front = normalized + dim;
        var back = front + width;

        Ops.ZeroCenteredRms(residual, b->PreNorm, dim, normalized);
        ConditionWeights(w->CondV, normalized, dim, weights);
        ScaleInput(normalized, w->D1, dim, inputRows * cols, front);
        Kron(front, inputRows, w->W1A, w->W1B, rows, cols, rows, back);
        PermuteModulateActivate(b->PermA, back, w->D2, weights, w->CondU, w->B2, width, front);
        Kron(front, rows, w->W2A, w->W2B, rows, cols, rows, back);
        PermuteScale(b->PermB, back, w->D3, width, front);
        Kron(front, rows, w->W3A, w->W3B, rows, cols, outputRows, back);
        AccumulateOutput(residual, back, w->D4, b->Scale, dim);
    }

#if NEEDLE_SIMD
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    static Vector256<float> Gather(float* source, int* permutation) =>
        Vector256.Create(
            source[permutation[0]], source[permutation[1]], source[permutation[2]], source[permutation[3]],
            source[permutation[4]], source[permutation[5]], source[permutation[6]], source[permutation[7]]);
#endif

    static void ConditionWeights(float* condV, float* input, int dim, float* weights)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.ConditionWeights(condV, input, dim, weights); return; }
#endif
        if (!Simd.Enabled) { Scalar.ConditionWeights(condV, input, dim, weights); return; }
#if NEEDLE_SIMD
        var l0 = Vector256<float>.Zero;
        var l1 = Vector256<float>.Zero;
        var l2 = Vector256<float>.Zero;
        var l3 = Vector256<float>.Zero;
        var i = 0;
        for (; i + 4 <= dim; i += 4)
        {
            l0 = Fma.MultiplyAdd(Vector256.Create(input[i]), Avx.LoadVector256(condV + i * Conditions), l0);
            l1 = Fma.MultiplyAdd(Vector256.Create(input[i + 1]), Avx.LoadVector256(condV + (i + 1) * Conditions), l1);
            l2 = Fma.MultiplyAdd(Vector256.Create(input[i + 2]), Avx.LoadVector256(condV + (i + 2) * Conditions), l2);
            l3 = Fma.MultiplyAdd(Vector256.Create(input[i + 3]), Avx.LoadVector256(condV + (i + 3) * Conditions), l3);
        }
        for (; i < dim; i++) l0 = Fma.MultiplyAdd(Vector256.Create(input[i]), Avx.LoadVector256(condV + i * Conditions), l0);
        Avx.Store(weights, Avx.Add(Avx.Add(l0, l1), Avx.Add(l2, l3)));
        Ops.Softmax(weights, Conditions);
#endif
    }

    static void ScaleInput(float* input, float* d1, int dim, int padded, float* destination)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.ScaleInput(input, d1, dim, padded, destination); return; }
#endif
        if (!Simd.Enabled) { Scalar.ScaleInput(input, d1, dim, padded, destination); return; }
#if NEEDLE_SIMD
        for (var i = 0; i < dim; i += 8) Avx.Store(destination + i, Avx.Multiply(Avx.LoadVector256(input + i), Avx.LoadVector256(d1 + i)));
        for (var i = dim; i < padded; i += 8) Avx.Store(destination + i, Vector256<float>.Zero);
#endif
    }

    static void PermuteModulateActivate(int* permutation, float* source, float* d2, float* weights, float* condU, float* b2, int width, float* destination)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.PermuteModulateActivate(permutation, source, d2, weights, condU, b2, width, destination); return; }
#endif
        if (!Simd.Enabled) { Scalar.PermuteModulateActivate(permutation, source, d2, weights, condU, b2, width, destination); return; }
#if NEEDLE_SIMD
        var w0 = Vector256.Create(weights[0]);
        var w1 = Vector256.Create(weights[1]);
        var w2 = Vector256.Create(weights[2]);
        var w3 = Vector256.Create(weights[3]);
        var w4 = Vector256.Create(weights[4]);
        var w5 = Vector256.Create(weights[5]);
        var w6 = Vector256.Create(weights[6]);
        var w7 = Vector256.Create(weights[7]);
        var one = Vector256.Create(1f);
        var u1 = condU + width;
        var u2 = u1 + width;
        var u3 = u2 + width;
        var u4 = u3 + width;
        var u5 = u4 + width;
        var u6 = u5 + width;
        var u7 = u6 + width;
        for (var i = 0; i < width; i += 8)
        {
            var condition = Fma.MultiplyAdd(w0, Avx.LoadVector256(condU + i), one);
            condition = Fma.MultiplyAdd(w1, Avx.LoadVector256(u1 + i), condition);
            condition = Fma.MultiplyAdd(w2, Avx.LoadVector256(u2 + i), condition);
            condition = Fma.MultiplyAdd(w3, Avx.LoadVector256(u3 + i), condition);
            condition = Fma.MultiplyAdd(w4, Avx.LoadVector256(u4 + i), condition);
            condition = Fma.MultiplyAdd(w5, Avx.LoadVector256(u5 + i), condition);
            condition = Fma.MultiplyAdd(w6, Avx.LoadVector256(u6 + i), condition);
            condition = Fma.MultiplyAdd(w7, Avx.LoadVector256(u7 + i), condition);
            var v = Fma.MultiplyAdd(
                Avx.Multiply(Avx.LoadVector256(d2 + i), condition),
                Gather(source, permutation + i),
                Avx.LoadVector256(b2 + i));
            Avx.Store(destination + i, Avx.Multiply(v, Ops.Sigmoid(v)));
        }
#endif
    }

    static void PermuteScale(int* permutation, float* source, float* scale, int width, float* destination)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.PermuteScale(permutation, source, scale, width, destination); return; }
#endif
        if (!Simd.Enabled) { Scalar.PermuteScale(permutation, source, scale, width, destination); return; }
#if NEEDLE_SIMD
        for (var i = 0; i < width; i += 8)
            Avx.Store(destination + i, Avx.Multiply(Gather(source, permutation + i), Avx.LoadVector256(scale + i)));
#endif
    }

    static void AccumulateOutput(float* residual, float* product, float* d4, float scale, int count)
    {
#if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.AccumulateOutput(residual, product, d4, scale, count); return; }
#endif
        if (!Simd.Enabled) { Scalar.AccumulateOutput(residual, product, d4, scale, count); return; }
#if NEEDLE_SIMD
        var s = Vector256.Create(scale);
        for (var i = 0; i < count; i += 8)
            Avx.Store(residual + i, Fma.MultiplyAdd(s, Avx.Multiply(Avx.LoadVector256(product + i), Avx.LoadVector256(d4 + i)), Avx.LoadVector256(residual + i)));
#endif
    }

#if NEEDLE_SIMD
    public static readonly bool Wide = Simd.UseAvx && Avx512F.IsSupported && Vector512.IsHardwareAccelerated && Environment.GetEnvironmentVariable("NEEDLE_NOAVX512") == null;

    [System.Runtime.CompilerServices.SkipLocalsInit]
    public static void KronAvx512(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
    {
        // Over-allocate so the tile stays 64-byte aligned when the stack slot is
        // only 16-byte aligned; every phase fully writes the tile before reading
        // it, so skipping zero-init is safe.
        var raw = stackalloc float[TileRows * cols + 16];
        var tile = (float*)(((nuint)raw + 63) & ~(nuint)63);
        nint stride1 = cols;
        nint stride2 = 2 * cols;
        nint stride3 = 3 * cols;
        for (var k = 0; k < resultRows; k += TileRows)
        {
            for (var c = 0; c < cols; c += 32)
            {
                var a00 = Vector512<float>.Zero;
                var a01 = Vector512<float>.Zero;
                var a10 = Vector512<float>.Zero;
                var a11 = Vector512<float>.Zero;
                var a20 = Vector512<float>.Zero;
                var a21 = Vector512<float>.Zero;
                var a30 = Vector512<float>.Zero;
                var a31 = Vector512<float>.Zero;
                var column = z + c;
                var factor = left + k;
                for (var i = 0; i < activeRows; i++)
                {
                    var z0 = Avx512F.LoadVector512(column);
                    var z1 = Avx512F.LoadVector512(column + 16);
                    var l0 = Vector512.Create(factor[0]);
                    a00 = Avx512F.FusedMultiplyAdd(l0, z0, a00);
                    a01 = Avx512F.FusedMultiplyAdd(l0, z1, a01);
                    var l1 = Vector512.Create(factor[1]);
                    a10 = Avx512F.FusedMultiplyAdd(l1, z0, a10);
                    a11 = Avx512F.FusedMultiplyAdd(l1, z1, a11);
                    var l2 = Vector512.Create(factor[2]);
                    a20 = Avx512F.FusedMultiplyAdd(l2, z0, a20);
                    a21 = Avx512F.FusedMultiplyAdd(l2, z1, a21);
                    var l3 = Vector512.Create(factor[3]);
                    a30 = Avx512F.FusedMultiplyAdd(l3, z0, a30);
                    a31 = Avx512F.FusedMultiplyAdd(l3, z1, a31);
                    column += cols;
                    factor += rows;
                }
                Avx512F.Store(tile + c, a00);
                Avx512F.Store(tile + c + 16, a01);
                Avx512F.Store(tile + stride1 + c, a10);
                Avx512F.Store(tile + stride1 + c + 16, a11);
                Avx512F.Store(tile + stride2 + c, a20);
                Avx512F.Store(tile + stride2 + c + 16, a21);
                Avx512F.Store(tile + stride3 + c, a30);
                Avx512F.Store(tile + stride3 + c + 16, a31);
            }
            for (var c = 0; c < cols; c += 32)
            {
                var a00 = Vector512<float>.Zero;
                var a01 = Vector512<float>.Zero;
                var a10 = Vector512<float>.Zero;
                var a11 = Vector512<float>.Zero;
                var a20 = Vector512<float>.Zero;
                var a21 = Vector512<float>.Zero;
                var a30 = Vector512<float>.Zero;
                var a31 = Vector512<float>.Zero;
                var row = right + c;
                var scalars = tile;
                for (var j = 0; j < cols; j++)
                {
                    var r0 = Avx512F.LoadVector512(row);
                    var r1 = Avx512F.LoadVector512(row + 16);
                    var s0 = Vector512.Create(*scalars);
                    a00 = Avx512F.FusedMultiplyAdd(s0, r0, a00);
                    a01 = Avx512F.FusedMultiplyAdd(s0, r1, a01);
                    var s1 = Vector512.Create(scalars[stride1]);
                    a10 = Avx512F.FusedMultiplyAdd(s1, r0, a10);
                    a11 = Avx512F.FusedMultiplyAdd(s1, r1, a11);
                    var s2 = Vector512.Create(scalars[stride2]);
                    a20 = Avx512F.FusedMultiplyAdd(s2, r0, a20);
                    a21 = Avx512F.FusedMultiplyAdd(s2, r1, a21);
                    var s3 = Vector512.Create(scalars[stride3]);
                    a30 = Avx512F.FusedMultiplyAdd(s3, r0, a30);
                    a31 = Avx512F.FusedMultiplyAdd(s3, r1, a31);
                    row += cols;
                    scalars++;
                }
                var output = result + k * cols + c;
                Avx512F.Store(output, a00);
                Avx512F.Store(output + 16, a01);
                Avx512F.Store(output + stride1, a10);
                Avx512F.Store(output + stride1 + 16, a11);
                Avx512F.Store(output + stride2, a20);
                Avx512F.Store(output + stride2 + 16, a21);
                Avx512F.Store(output + stride3, a30);
                Avx512F.Store(output + stride3 + 16, a31);
            }
        }
    }
#endif

    public static void Kron(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
    {
        if (!Blocked(rows, cols)) { Scalar.Kron(z, activeRows, left, right, rows, cols, resultRows, result); return; }
#if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Kron(z, activeRows, left, right, rows, cols, resultRows, result); return; }
        if (Wide && cols % 32 == 0) { KronAvx512(z, activeRows, left, right, rows, cols, resultRows, result); return; }
        KronAvx2(z, activeRows, left, right, rows, cols, resultRows, result);
#endif
    }

#if NEEDLE_SIMD
    [System.Runtime.CompilerServices.SkipLocalsInit]
    public static void KronAvx2(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
    {
        // Over-allocate so the tile stays 32-byte aligned when the stack slot is
        // only 16-byte aligned; every phase fully writes the tile before reading
        // it, so skipping zero-init is safe.
        var raw = stackalloc float[TileRows * cols + 8];
        var tile = (float*)(((nuint)raw + 31) & ~(nuint)31);
        nint stride1 = cols;
        nint stride2 = 2 * cols;
        nint stride3 = 3 * cols;
        for (var k = 0; k < resultRows; k += TileRows)
        {
            for (var c = 0; c < cols; c += TileColumns)
            {
                var a00 = Vector256<float>.Zero;
                var a01 = Vector256<float>.Zero;
                var a10 = Vector256<float>.Zero;
                var a11 = Vector256<float>.Zero;
                var a20 = Vector256<float>.Zero;
                var a21 = Vector256<float>.Zero;
                var a30 = Vector256<float>.Zero;
                var a31 = Vector256<float>.Zero;
                var column = z + c;
                var factor = left + k;
                for (var i = 0; i < activeRows; i++)
                {
                    var z0 = Avx.LoadVector256(column);
                    var z1 = Avx.LoadVector256(column + 8);
                    var l0 = Avx.BroadcastScalarToVector256(factor);
                    a00 = Fma.MultiplyAdd(l0, z0, a00);
                    a01 = Fma.MultiplyAdd(l0, z1, a01);
                    var l1 = Avx.BroadcastScalarToVector256(factor + 1);
                    a10 = Fma.MultiplyAdd(l1, z0, a10);
                    a11 = Fma.MultiplyAdd(l1, z1, a11);
                    var l2 = Avx.BroadcastScalarToVector256(factor + 2);
                    a20 = Fma.MultiplyAdd(l2, z0, a20);
                    a21 = Fma.MultiplyAdd(l2, z1, a21);
                    var l3 = Avx.BroadcastScalarToVector256(factor + 3);
                    a30 = Fma.MultiplyAdd(l3, z0, a30);
                    a31 = Fma.MultiplyAdd(l3, z1, a31);
                    column += cols;
                    factor += rows;
                }
                Avx.Store(tile + c, a00);
                Avx.Store(tile + c + 8, a01);
                Avx.Store(tile + stride1 + c, a10);
                Avx.Store(tile + stride1 + c + 8, a11);
                Avx.Store(tile + stride2 + c, a20);
                Avx.Store(tile + stride2 + c + 8, a21);
                Avx.Store(tile + stride3 + c, a30);
                Avx.Store(tile + stride3 + c + 8, a31);
            }
            for (var c = 0; c < cols; c += TileColumns)
            {
                var a00 = Vector256<float>.Zero;
                var a01 = Vector256<float>.Zero;
                var a10 = Vector256<float>.Zero;
                var a11 = Vector256<float>.Zero;
                var a20 = Vector256<float>.Zero;
                var a21 = Vector256<float>.Zero;
                var a30 = Vector256<float>.Zero;
                var a31 = Vector256<float>.Zero;
                var row = right + c;
                var scalars = tile;
                for (var j = 0; j < cols; j++)
                {
                    var r0 = Avx.LoadVector256(row);
                    var r1 = Avx.LoadVector256(row + 8);
                    var s0 = Avx.BroadcastScalarToVector256(scalars);
                    a00 = Fma.MultiplyAdd(s0, r0, a00);
                    a01 = Fma.MultiplyAdd(s0, r1, a01);
                    var s1 = Avx.BroadcastScalarToVector256(scalars + stride1);
                    a10 = Fma.MultiplyAdd(s1, r0, a10);
                    a11 = Fma.MultiplyAdd(s1, r1, a11);
                    var s2 = Avx.BroadcastScalarToVector256(scalars + stride2);
                    a20 = Fma.MultiplyAdd(s2, r0, a20);
                    a21 = Fma.MultiplyAdd(s2, r1, a21);
                    var s3 = Avx.BroadcastScalarToVector256(scalars + stride3);
                    a30 = Fma.MultiplyAdd(s3, r0, a30);
                    a31 = Fma.MultiplyAdd(s3, r1, a31);
                    row += cols;
                    scalars++;
                }
                var output = result + k * cols + c;
                Avx.Store(output, a00);
                Avx.Store(output + 8, a01);
                Avx.Store(output + stride1, a10);
                Avx.Store(output + stride1 + 8, a11);
                Avx.Store(output + stride2, a20);
                Avx.Store(output + stride2 + 8, a21);
                Avx.Store(output + stride3, a30);
                Avx.Store(output + stride3 + 8, a31);
            }
        }
    }
#endif
}

internal static unsafe class Sinkhorn
{
    const int Lanes = 4;
    const int Iterations = 20;

    public static void Normalize(float* logits, float* result)
    {
        for (var r = 0; r < Lanes; r++)
        {
            var peak = MathF.Max(MathF.Max(logits[r * 4], logits[r * 4 + 1]), MathF.Max(logits[r * 4 + 2], logits[r * 4 + 3]));
            for (var c = 0; c < Lanes; c++) result[r * 4 + c] = MathF.Exp(logits[r * 4 + c] - peak);
        }
        for (var iteration = 0; iteration < Iterations; iteration++)
        {
            for (var r = 0; r < Lanes; r++)
            {
                var sum = result[r * 4] + result[r * 4 + 1] + result[r * 4 + 2] + result[r * 4 + 3];
                for (var c = 0; c < Lanes; c++) result[r * 4 + c] /= sum;
            }
            for (var c = 0; c < Lanes; c++)
            {
                var sum = result[c] + result[4 + c] + result[8 + c] + result[12 + c];
                if (sum > 0f) for (var r = 0; r < Lanes; r++) result[r * 4 + c] /= sum;
            }
        }
    }
}

internal static unsafe class EngramHash
{
    const uint Seed = 0x9E3779B9;
    const uint Prime = 0x01000193;

    public static void Rows(in Geometry g, int* tokens, int count, int position, int* previous, int* rows)
    {
        var heads = g.EngramHeads;
        for (var t = 0; t < count; t++)
            for (var o = 0; o < g.OrderCount; o++)
                for (var h = 0; h < heads; h++)
                {
                    var table = o * heads + h;
                    var order = g.Orders[o];
                    if (position + t < order - 1)
                    {
                        rows[t * g.EngramTables + table] = -1;
                        continue;
                    }
                    var acc = Seed * (uint)(table + 1);
                    for (var j = 0; j < order; j++)
                    {
                        var index = t - j;
                        var token = index >= 0 ? tokens[index] : previous[index + 2];
                        acc = (acc ^ (uint)token) * Prime;
                    }
                    acc ^= acc >> 15;
                    rows[t * g.EngramTables + table] = table * g.EngramSlots + (int)(acc % (uint)g.EngramSlots);
                }
    }
}
