#if NEEDLE_SIMD
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Needle;

// Head count of a NEON attention value pass, as a type parameter so the unused accumulators compile away.
internal interface INeonHeads
{
    static abstract int Count { get; }
}

internal readonly struct NeonHeads1 : INeonHeads { public static int Count => 1; }
internal readonly struct NeonHeads2 : INeonHeads { public static int Count => 2; }
internal readonly struct NeonHeads3 : INeonHeads { public static int Count => 3; }
internal readonly struct NeonHeads4 : INeonHeads { public static int Count => 4; }
internal readonly struct NeonHeads5 : INeonHeads { public static int Count => 5; }
internal readonly struct NeonHeads6 : INeonHeads { public static int Count => 6; }

// The NEON attention tile. Scores take four key positions at a time: the int8 dot products are exact,
// so reducing them with pairwise adds instead of one add-across per position changes no bit. The value
// pass widens each value row once for every head of the tile and accumulates with by-element FMAs, in
// the same position order and with the same fused operations as the per-head kernel it replaces.
[SkipLocalsInit]
internal static unsafe class NeonAttention
{
    const float InverseSqrtQkDim = 0.14433756729740643f;
    const int MaxHeads = 6;

    static readonly bool UseDotProduct = Dp.IsSupported && Environment.GetEnvironmentVariable("NEEDLE_NODP") == null;

    public static void Tile(in AttentionTile tile)
    {
        if (UseDotProduct) Scores<SdotNeonDot>(tile);
        else Scores<WideningNeonDot>(tile);
        var count = tile.Count;
        for (var h = 0; h < tile.Heads; h++)
        {
            var scores = tile.Scores + h * count;
            Neon.Softmax(scores, count);
            if (tile.Probabilities != null)
                for (var s = 0; s < count; s++) tile.Probabilities[h * tile.ProbabilityStride + tile.First + s] = scores[s];
        }
        ScaleByValueScales(tile);
        for (var first = 0; first < tile.Heads; first += MaxHeads)
        {
            switch (Math.Min(MaxHeads, tile.Heads - first))
            {
                case 1: Values<NeonHeads1>(tile, first); break;
                case 2: Values<NeonHeads2>(tile, first); break;
                case 3: Values<NeonHeads3>(tile, first); break;
                case 4: Values<NeonHeads4>(tile, first); break;
                case 5: Values<NeonHeads5>(tile, first); break;
                case 6: Values<NeonHeads6>(tile, first); break;
            }
        }
    }

    static void Scores<TDot>(in AttentionTile tile) where TDot : struct, INeonDot
    {
        var count = tile.Count;
        var heads = tile.Heads;
        nint keyStride = tile.KeyStride;
        nint scaleStride = tile.ScaleStride;
        var keys = tile.Keys + tile.First * keyStride;
        var keyScales = tile.KeyScales + tile.First * scaleStride;
        var s = 0;
        for (; s + 4 <= count; s += 4)
        {
            var k0 = keys + s * keyStride;
            var k1 = k0 + keyStride;
            var k2 = k1 + keyStride;
            var k3 = k2 + keyStride;
            var k00 = AdvSimd.LoadVector128(k0);
            var k01 = AdvSimd.LoadVector128(k0 + 16);
            var k02 = AdvSimd.LoadVector128(k0 + 32);
            var k10 = AdvSimd.LoadVector128(k1);
            var k11 = AdvSimd.LoadVector128(k1 + 16);
            var k12 = AdvSimd.LoadVector128(k1 + 32);
            var k20 = AdvSimd.LoadVector128(k2);
            var k21 = AdvSimd.LoadVector128(k2 + 16);
            var k22 = AdvSimd.LoadVector128(k2 + 32);
            var k30 = AdvSimd.LoadVector128(k3);
            var k31 = AdvSimd.LoadVector128(k3 + 16);
            var k32 = AdvSimd.LoadVector128(k3 + 32);
            var scale = keyScales + s * scaleStride;
            var keyScale = Vector128.Create(scale[0], scale[scaleStride], scale[2 * scaleStride], scale[3 * scaleStride]);
            for (var h = 0; h < heads; h++)
            {
                var query = tile.Queries + h * 48;
                var q0 = AdvSimd.LoadVector128(query);
                var q1 = AdvSimd.LoadVector128(query + 16);
                var q2 = AdvSimd.LoadVector128(query + 32);
                var d0 = TDot.Dot(TDot.Dot(TDot.Dot(Vector128<int>.Zero, q0, k00), q1, k01), q2, k02);
                var d1 = TDot.Dot(TDot.Dot(TDot.Dot(Vector128<int>.Zero, q0, k10), q1, k11), q2, k12);
                var d2 = TDot.Dot(TDot.Dot(TDot.Dot(Vector128<int>.Zero, q0, k20), q1, k21), q2, k22);
                var d3 = TDot.Dot(TDot.Dot(TDot.Dot(Vector128<int>.Zero, q0, k30), q1, k31), q2, k32);
                var dots = AdvSimd.ConvertToSingle(AdvSimd.Arm64.AddPairwise(AdvSimd.Arm64.AddPairwise(d0, d1), AdvSimd.Arm64.AddPairwise(d2, d3)));
                var queryScale = Vector128.Create(tile.QueryScales[h] * InverseSqrtQkDim);
                AdvSimd.Store(tile.Scores + h * count + s, AdvSimd.Multiply(AdvSimd.Multiply(dots, keyScale), queryScale));
            }
        }
        for (; s < count; s++)
        {
            var key = keys + s * keyStride;
            var k0 = AdvSimd.LoadVector128(key);
            var k1 = AdvSimd.LoadVector128(key + 16);
            var k2 = AdvSimd.LoadVector128(key + 32);
            var keyScale = keyScales[s * scaleStride];
            for (var h = 0; h < heads; h++)
            {
                var query = tile.Queries + h * 48;
                var d = TDot.Dot(TDot.Dot(TDot.Dot(Vector128<int>.Zero, AdvSimd.LoadVector128(query), k0), AdvSimd.LoadVector128(query + 16), k1), AdvSimd.LoadVector128(query + 32), k2);
                tile.Scores[h * count + s] = AdvSimd.Arm64.AddAcross(d).ToScalar() * keyScale * (tile.QueryScales[h] * InverseSqrtQkDim);
            }
        }
    }

    // Turns each head's probabilities into value weights (probability times the position's value scale).
    static void ScaleByValueScales(in AttentionTile tile)
    {
        var count = tile.Count;
        nint scaleStride = tile.ScaleStride;
        var valueScales = tile.ValueScales + tile.First * scaleStride;
        var s = 0;
        for (; s + 4 <= count; s += 4)
        {
            var scale = valueScales + s * scaleStride;
            var valueScale = Vector128.Create(scale[0], scale[scaleStride], scale[2 * scaleStride], scale[3 * scaleStride]);
            for (var h = 0; h < tile.Heads; h++)
            {
                var weights = tile.Scores + h * count + s;
                AdvSimd.Store(weights, AdvSimd.Multiply(AdvSimd.LoadVector128(weights), valueScale));
            }
        }
        for (; s < count; s++)
            for (var h = 0; h < tile.Heads; h++) tile.Scores[h * count + s] *= valueScales[s * scaleStride];
    }

    static void Values<THeads>(in AttentionTile tile, int firstHead) where THeads : struct, INeonHeads
    {
        var count = tile.Count;
        nint valueStride = tile.ValueStride;
        var values = tile.Values + tile.First * valueStride;
        var weights = tile.Scores + firstHead * count;
        var p0 = weights;
        var p1 = weights + 1 * count;
        var p2 = weights + 2 * count;
        var p3 = weights + 3 * count;
        var p4 = weights + 4 * count;
        var p5 = weights + 5 * count;
        for (var column = 0; column < 64; column += 8)
        {
            var a0 = Vector128<float>.Zero;
            var b0 = Vector128<float>.Zero;
            var a1 = Vector128<float>.Zero;
            var b1 = Vector128<float>.Zero;
            var a2 = Vector128<float>.Zero;
            var b2 = Vector128<float>.Zero;
            var a3 = Vector128<float>.Zero;
            var b3 = Vector128<float>.Zero;
            var a4 = Vector128<float>.Zero;
            var b4 = Vector128<float>.Zero;
            var a5 = Vector128<float>.Zero;
            var b5 = Vector128<float>.Zero;
            var row = values + column;
            var s = 0;
            for (; s + 4 <= count; s += 4)
            {
                var w0 = AdvSimd.LoadVector128(p0 + s);
                var w1 = THeads.Count > 1 ? AdvSimd.LoadVector128(p1 + s) : default;
                var w2 = THeads.Count > 2 ? AdvSimd.LoadVector128(p2 + s) : default;
                var w3 = THeads.Count > 3 ? AdvSimd.LoadVector128(p3 + s) : default;
                var w4 = THeads.Count > 4 ? AdvSimd.LoadVector128(p4 + s) : default;
                var w5 = THeads.Count > 5 ? AdvSimd.LoadVector128(p5 + s) : default;
                {
                    var wide = AdvSimd.SignExtendWideningLower(AdvSimd.LoadVector64(row));
                    var f0 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(wide.GetLower()));
                    var f1 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(wide));
                    a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0, f0, w0, 0);
                    b0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b0, f1, w0, 0);
                    if (THeads.Count > 1)
                    {
                        a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1, f0, w1, 0);
                        b1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b1, f1, w1, 0);
                    }
                    if (THeads.Count > 2)
                    {
                        a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2, f0, w2, 0);
                        b2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b2, f1, w2, 0);
                    }
                    if (THeads.Count > 3)
                    {
                        a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3, f0, w3, 0);
                        b3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b3, f1, w3, 0);
                    }
                    if (THeads.Count > 4)
                    {
                        a4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a4, f0, w4, 0);
                        b4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b4, f1, w4, 0);
                    }
                    if (THeads.Count > 5)
                    {
                        a5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a5, f0, w5, 0);
                        b5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b5, f1, w5, 0);
                    }
                    row += valueStride;
                }
                {
                    var wide = AdvSimd.SignExtendWideningLower(AdvSimd.LoadVector64(row));
                    var f0 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(wide.GetLower()));
                    var f1 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(wide));
                    a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0, f0, w0, 1);
                    b0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b0, f1, w0, 1);
                    if (THeads.Count > 1)
                    {
                        a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1, f0, w1, 1);
                        b1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b1, f1, w1, 1);
                    }
                    if (THeads.Count > 2)
                    {
                        a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2, f0, w2, 1);
                        b2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b2, f1, w2, 1);
                    }
                    if (THeads.Count > 3)
                    {
                        a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3, f0, w3, 1);
                        b3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b3, f1, w3, 1);
                    }
                    if (THeads.Count > 4)
                    {
                        a4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a4, f0, w4, 1);
                        b4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b4, f1, w4, 1);
                    }
                    if (THeads.Count > 5)
                    {
                        a5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a5, f0, w5, 1);
                        b5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b5, f1, w5, 1);
                    }
                    row += valueStride;
                }
                {
                    var wide = AdvSimd.SignExtendWideningLower(AdvSimd.LoadVector64(row));
                    var f0 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(wide.GetLower()));
                    var f1 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(wide));
                    a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0, f0, w0, 2);
                    b0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b0, f1, w0, 2);
                    if (THeads.Count > 1)
                    {
                        a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1, f0, w1, 2);
                        b1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b1, f1, w1, 2);
                    }
                    if (THeads.Count > 2)
                    {
                        a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2, f0, w2, 2);
                        b2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b2, f1, w2, 2);
                    }
                    if (THeads.Count > 3)
                    {
                        a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3, f0, w3, 2);
                        b3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b3, f1, w3, 2);
                    }
                    if (THeads.Count > 4)
                    {
                        a4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a4, f0, w4, 2);
                        b4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b4, f1, w4, 2);
                    }
                    if (THeads.Count > 5)
                    {
                        a5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a5, f0, w5, 2);
                        b5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b5, f1, w5, 2);
                    }
                    row += valueStride;
                }
                {
                    var wide = AdvSimd.SignExtendWideningLower(AdvSimd.LoadVector64(row));
                    var f0 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(wide.GetLower()));
                    var f1 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(wide));
                    a0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0, f0, w0, 3);
                    b0 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b0, f1, w0, 3);
                    if (THeads.Count > 1)
                    {
                        a1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1, f0, w1, 3);
                        b1 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b1, f1, w1, 3);
                    }
                    if (THeads.Count > 2)
                    {
                        a2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2, f0, w2, 3);
                        b2 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b2, f1, w2, 3);
                    }
                    if (THeads.Count > 3)
                    {
                        a3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3, f0, w3, 3);
                        b3 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b3, f1, w3, 3);
                    }
                    if (THeads.Count > 4)
                    {
                        a4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a4, f0, w4, 3);
                        b4 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b4, f1, w4, 3);
                    }
                    if (THeads.Count > 5)
                    {
                        a5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a5, f0, w5, 3);
                        b5 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(b5, f1, w5, 3);
                    }
                    row += valueStride;
                }
            }
            for (; s < count; s++)
            {
                var wide = AdvSimd.SignExtendWideningLower(AdvSimd.LoadVector64(row));
                var f0 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(wide.GetLower()));
                var f1 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(wide));
                var v0 = Vector128.Create(p0[s]);
                a0 = AdvSimd.FusedMultiplyAdd(a0, v0, f0);
                b0 = AdvSimd.FusedMultiplyAdd(b0, v0, f1);
                if (THeads.Count > 1)
                {
                    var v1 = Vector128.Create(p1[s]);
                    a1 = AdvSimd.FusedMultiplyAdd(a1, v1, f0);
                    b1 = AdvSimd.FusedMultiplyAdd(b1, v1, f1);
                }
                if (THeads.Count > 2)
                {
                    var v2 = Vector128.Create(p2[s]);
                    a2 = AdvSimd.FusedMultiplyAdd(a2, v2, f0);
                    b2 = AdvSimd.FusedMultiplyAdd(b2, v2, f1);
                }
                if (THeads.Count > 3)
                {
                    var v3 = Vector128.Create(p3[s]);
                    a3 = AdvSimd.FusedMultiplyAdd(a3, v3, f0);
                    b3 = AdvSimd.FusedMultiplyAdd(b3, v3, f1);
                }
                if (THeads.Count > 4)
                {
                    var v4 = Vector128.Create(p4[s]);
                    a4 = AdvSimd.FusedMultiplyAdd(a4, v4, f0);
                    b4 = AdvSimd.FusedMultiplyAdd(b4, v4, f1);
                }
                if (THeads.Count > 5)
                {
                    var v5 = Vector128.Create(p5[s]);
                    a5 = AdvSimd.FusedMultiplyAdd(a5, v5, f0);
                    b5 = AdvSimd.FusedMultiplyAdd(b5, v5, f1);
                }
                row += valueStride;
            }
            var output = tile.Output + firstHead * 64 + column;
            AdvSimd.Store(output, a0);
            AdvSimd.Store(output + 4, b0);
            if (THeads.Count > 1)
            {
                AdvSimd.Store(output + 64, a1);
                AdvSimd.Store(output + 68, b1);
            }
            if (THeads.Count > 2)
            {
                AdvSimd.Store(output + 128, a2);
                AdvSimd.Store(output + 132, b2);
            }
            if (THeads.Count > 3)
            {
                AdvSimd.Store(output + 192, a3);
                AdvSimd.Store(output + 196, b3);
            }
            if (THeads.Count > 4)
            {
                AdvSimd.Store(output + 256, a4);
                AdvSimd.Store(output + 260, b4);
            }
            if (THeads.Count > 5)
            {
                AdvSimd.Store(output + 320, a5);
                AdvSimd.Store(output + 324, b5);
            }
        }
    }
}
#endif
