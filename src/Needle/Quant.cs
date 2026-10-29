using System;
using System.Runtime.CompilerServices;
#if NEEDLE_SIMD
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace Needle;

internal readonly unsafe struct Activations
{
    public readonly sbyte* Values;
    public readonly float* Scales;
    public readonly float* Offsets;
    public readonly int Cols;
    public readonly int Groups;
    public readonly int Stride;

    public Activations(sbyte* values, float* scales, float* offsets, int cols)
    {
        Values = values;
        Scales = scales;
        Offsets = offsets;
        Cols = cols;
        Groups = cols / Cq.GroupSize;
        Stride = (Groups + 7) & ~7;
    }

    public Activations Slice(int token) =>
        new(Values + (long)token * Cols, Scales + token * Stride, Offsets + token * Stride, Cols);

    public Activations(ref Arena arena, int tokens, int cols)
    {
        Cols = cols;
        Groups = cols / Cq.GroupSize;
        Stride = (Groups + 7) & ~7;
        Values = arena.Take<sbyte>((nuint)(tokens * cols));
        Scales = arena.Take<float>((nuint)(tokens * Stride));
        Offsets = arena.Take<float>((nuint)(tokens * Stride));
    }
}

internal static unsafe class Quant
{
    const float HadamardScale = 0.08838834764831845f;
    const int TokenTile = 4;

    public static float AbsMax(float* x, int n)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { return Neon.AbsMax(x, n); }
#endif
        if (!Simd.Enabled) return Scalar.AbsMax(x, n);
#if NEEDLE_SIMD
        var peak = Vector256<float>.Zero;
        for (var i = 0; i < n; i += 8) peak = Avx.Max(peak, Vector256.Abs(Avx.LoadVector256(x + i)));
        var folded = Sse.Max(peak.GetLower(), peak.GetUpper());
        folded = Sse.Max(folded, Sse.Shuffle(folded, folded, 0b01_00_11_10));
        return Sse.Max(folded, Sse.Shuffle(folded, folded, 0b10_11_00_01)).ToScalar();
#endif
    }

    public static float FakeA8(float* x, int n)
    {
        var peak = AbsMax(x, n);
        var scale = peak > 0f ? peak / 127f : 1f;
        RoundTrip(x, n, 1f / scale, scale);
        return scale;
    }

    public static void RoundTrip(float* x, int n, float inverse, float scale)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.RoundTrip(x, n, inverse, scale); return; }
#endif
        if (!Simd.Enabled) { Scalar.RoundTrip(x, n, inverse, scale); return; }
#if NEEDLE_SIMD
        var inv = Vector256.Create(inverse);
        var sc = Vector256.Create(scale);
        var low = Vector256.Create(-128f);
        var high = Vector256.Create(127f);
        for (var i = 0; i < n; i += 8)
        {
            var q = Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(x + i), inv));
            Avx.Store(x + i, Avx.Multiply(Avx.Min(Avx.Max(q, low), high), sc));
        }
#endif
    }

    public static float QuantizeInt8(float* x, int n, sbyte* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { return Neon.QuantizeInt8(x, n, destination); }
#endif
        if (!Simd.Enabled) return Scalar.QuantizeInt8(x, n, destination);
#if NEEDLE_SIMD
        var peak = AbsMax(x, n);
        var scale = peak > 0f ? peak / 127f : 1f;
        var inv = Vector256.Create(1f / scale);
        for (var i = 0; i < n; i += 8)
        {
            var q = Avx.ConvertToVector256Int32(Avx.Multiply(Avx.LoadVector256(x + i), inv));
            var packed = Sse2.PackSignedSaturate(q.GetLower(), q.GetUpper());
            *(long*)(destination + i) = Sse2.PackSignedSaturate(packed, packed).AsInt64().ToScalar();
        }
        return scale;
#endif
    }

    public static void Hadamard(float* p) => Hadamard(p, p);

    public static void Hadamard(float* source, float* p)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Hadamard(source, p); return; }
#endif
        if (!Simd.Enabled) { Scalar.Hadamard(source, p); return; }
#if NEEDLE_SIMD
        for (var i = 0; i < 64; i += 8)
        {
            var a = Avx.LoadVector256(source + i);
            var b = Avx.LoadVector256(source + i + 64);
            Avx.Store(p + i, Avx.Add(a, b));
            Avx.Store(p + i + 64, Avx.Subtract(a, b));
        }
        for (var stride = 32; stride >= 8; stride >>= 1)
            for (var block = 0; block < Cq.GroupSize; block += stride * 2)
                for (var i = block; i < block + stride; i += 8)
                {
                    var a = Avx.LoadVector256(p + i);
                    var b = Avx.LoadVector256(p + i + stride);
                    Avx.Store(p + i, Avx.Add(a, b));
                    Avx.Store(p + i + stride, Avx.Subtract(a, b));
                }
        var pairSign = Vector256.Create(1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f);
        var quadSign = Vector256.Create(1f, 1f, -1f, -1f, 1f, 1f, -1f, -1f);
        var halfSign = Vector256.Create(1f, 1f, 1f, 1f, -1f, -1f, -1f, -1f);
        var scale = Vector256.Create(HadamardScale);
        for (var i = 0; i < Cq.GroupSize; i += 8)
        {
            var v = Avx.LoadVector256(p + i);
            v = Fma.MultiplyAdd(v, pairSign, Avx.Permute(v, 0b10_11_00_01));
            v = Fma.MultiplyAdd(v, quadSign, Avx.Permute(v, 0b01_00_11_10));
            v = Fma.MultiplyAdd(v, halfSign, Avx.Permute2x128(v, v, 0x01));
            Avx.Store(p + i, Avx.Multiply(v, scale));
        }
#endif
    }

    public static void Prepare(float* x, int tokens, int cols, int bits, in Activations a)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Prepare(x, tokens, cols, bits, a); return; }
#endif
        var rotated = stackalloc float[Cq.GroupSize];
        var quantized = stackalloc sbyte[Cq.GroupSize];
#if NEEDLE_SIMD
        var ones = Vector256.Create((byte)1);
#endif
        for (var t = 0; t < tokens; t++)
            for (var g = 0; g < a.Groups; g++)
            {
                var slot = t * a.Stride + g;
                Hadamard(x + (long)t * cols + g * Cq.GroupSize, rotated);
                a.Scales[slot] = QuantizeInt8(rotated, Cq.GroupSize, quantized);
                if (!Simd.Enabled)
                {
                    Scalar.Planes(quantized, bits, a.Values + ((long)t * a.Groups + g) * Cq.GroupSize, out var scalarSum);
                    a.Offsets[slot] = Codebook.Bias * (float)scalarSum;
                    continue;
                }
#if NEEDLE_SIMD
                var q0 = Avx.LoadVector256(quantized);
                var q1 = Avx.LoadVector256(quantized + 32);
                var q2 = Avx.LoadVector256(quantized + 64);
                var q3 = Avx.LoadVector256(quantized + 96);
                var sum = Accumulate(Accumulate(Accumulate(Accumulate(Vector256<int>.Zero, ones, q0), ones, q1), ones, q2), ones, q3);
                a.Offsets[slot] = Codebook.Bias * (float)Vector256.Sum(sum);
                var destination = a.Values + ((long)t * a.Groups + g) * Cq.GroupSize;
                if (bits == 2) Planes2(q0, q1, q2, q3, destination);
                else Planes4(q0, q1, q2, q3, destination);
#endif
            }
    }

#if NEEDLE_SIMD
    static void Planes2(Vector256<sbyte> q0, Vector256<sbyte> q1, Vector256<sbyte> q2, Vector256<sbyte> q3, sbyte* destination)
    {
        var gather = Vector256.Create((byte)0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        var s0 = Avx2.Shuffle(q0.AsByte(), gather).AsInt32();
        var s1 = Avx2.Shuffle(q1.AsByte(), gather).AsInt32();
        var s2 = Avx2.Shuffle(q2.AsByte(), gather).AsInt32();
        var s3 = Avx2.Shuffle(q3.AsByte(), gather).AsInt32();
        var low01 = Avx2.UnpackLow(s0, s1);
        var high01 = Avx2.UnpackHigh(s0, s1);
        var low23 = Avx2.UnpackLow(s2, s3);
        var high23 = Avx2.UnpackHigh(s2, s3);
        var interleave = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
        Avx.Store(destination, Avx2.PermuteVar8x32(Avx2.UnpackLow(low01.AsInt64(), low23.AsInt64()).AsInt32(), interleave).AsSByte());
        Avx.Store(destination + 32, Avx2.PermuteVar8x32(Avx2.UnpackHigh(low01.AsInt64(), low23.AsInt64()).AsInt32(), interleave).AsSByte());
        Avx.Store(destination + 64, Avx2.PermuteVar8x32(Avx2.UnpackLow(high01.AsInt64(), high23.AsInt64()).AsInt32(), interleave).AsSByte());
        Avx.Store(destination + 96, Avx2.PermuteVar8x32(Avx2.UnpackHigh(high01.AsInt64(), high23.AsInt64()).AsInt32(), interleave).AsSByte());
    }
#endif

#if NEEDLE_SIMD
    static void Planes4(Vector256<sbyte> q0, Vector256<sbyte> q1, Vector256<sbyte> q2, Vector256<sbyte> q3, sbyte* destination)
    {
        var gather = Vector256.Create((byte)0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
        var s0 = Avx2.Shuffle(q0.AsByte(), gather).AsInt64();
        var s1 = Avx2.Shuffle(q1.AsByte(), gather).AsInt64();
        var s2 = Avx2.Shuffle(q2.AsByte(), gather).AsInt64();
        var s3 = Avx2.Shuffle(q3.AsByte(), gather).AsInt64();
        Avx.Store(destination, Avx2.Permute4x64(Avx2.UnpackLow(s0, s1), 0b11_01_10_00).AsSByte());
        Avx.Store(destination + 32, Avx2.Permute4x64(Avx2.UnpackHigh(s0, s1), 0b11_01_10_00).AsSByte());
        Avx.Store(destination + 64, Avx2.Permute4x64(Avx2.UnpackLow(s2, s3), 0b11_01_10_00).AsSByte());
        Avx.Store(destination + 96, Avx2.Permute4x64(Avx2.UnpackHigh(s2, s3), 0b11_01_10_00).AsSByte());
    }
#endif

    public static void DecodeRow(in Cq w, int row, float* destination)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.DecodeRow(w, row, destination); return; }
#endif
        if (!Simd.Enabled) { Scalar.DecodeRow(w, row, destination); return; }
#if NEEDLE_SIMD
        var indices = w.Indices + (long)row * w.RowBytes;
        var norms = w.Norms + (long)row * w.Groups;
        var lut = Avx.LoadVector256(w.Book->Levels);
        if (w.Bits == 2)
            for (var g = 0; g < w.Groups; g++)
                DecodeGroup2(indices + g * 32, Avx.Multiply(lut, Vector256.Create(Scalar.HalfToSingle(norms[g]))), destination + g * Cq.GroupSize);
        else
        {
            var upper = Avx.LoadVector256(w.Book->Levels + 8);
            for (var g = 0; g < w.Groups; g++)
            {
                var norm = Vector256.Create(Scalar.HalfToSingle(norms[g]));
                DecodeGroup4(indices + g * 64, Avx.Multiply(lut, norm), Avx.Multiply(upper, norm), destination + g * Cq.GroupSize);
            }
        }
#endif
    }

    public static void GatherRow(in Cq w, int row, float* destination)
    {
        DecodeRow(w, row, destination);
        for (var g = 0; g < w.Groups; g++) Hadamard(destination + g * Cq.GroupSize);
    }

#if NEEDLE_SIMD
    static void DecodeGroup2(byte* packed, Vector256<float> lut, float* destination)
    {
        var lowShift = Vector256.Create(0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u);
        var highShift = Vector256.Create(16u, 18u, 20u, 22u, 24u, 26u, 28u, 30u);
        for (var k = 0; k < 8; k++)
        {
            var word = Vector256.Create(((uint*)packed)[k]);
            Avx.Store(destination + 16 * k, Avx2.PermuteVar8x32(lut, Avx2.ShiftRightLogicalVariable(word, lowShift).AsInt32()));
            Avx.Store(destination + 16 * k + 8, Avx2.PermuteVar8x32(lut, Avx2.ShiftRightLogicalVariable(word, highShift).AsInt32()));
        }
    }
#endif

#if NEEDLE_SIMD
    static void DecodeGroup4(byte* packed, Vector256<float> lower, Vector256<float> upper, float* destination)
    {
        var shift = Vector256.Create(0u, 4u, 8u, 12u, 16u, 20u, 24u, 28u);
        for (var k = 0; k < 16; k++)
        {
            var index = Avx2.ShiftRightLogicalVariable(Vector256.Create(((uint*)packed)[k]), shift);
            var selector = Avx2.ShiftLeftLogical(index, 28).AsSingle();
            var index3 = index.AsInt32();
            Avx.Store(destination + 8 * k, Avx.BlendVariable(Avx2.PermuteVar8x32(lower, index3), Avx2.PermuteVar8x32(upper, index3), selector));
        }
    }
#endif

#if NEEDLE_SIMD
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<int> Accumulate(Vector256<int> sum, Vector256<byte> weights, Vector256<sbyte> activations) =>
        AvxVnni.IsSupported
            ? AvxVnni.MultiplyWideningAndAdd(sum, weights, activations)
            : Avx2.Add(sum, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), Vector256.Create((short)1)));
#endif

#if NEEDLE_SIMD
    static Vector128<float> HalfToSingle(Vector128<uint> half)
    {
        var magnitude = Sse2.And(half, Vector128.Create(0x7FFFu));
        var sign = Sse2.ShiftLeftLogical(Sse2.And(half, Vector128.Create(0x8000u)), 16);
        var bits = Sse2.Add(Sse2.ShiftLeftLogical(magnitude, 13), Vector128.Create(0x38000000u));
        var nonZero = Sse2.CompareGreaterThan(magnitude.AsInt32(), Vector128<int>.Zero).AsUInt32();
        return Sse2.Or(Sse2.And(bits, nonZero), sign).AsSingle();
    }
#endif

#if NEEDLE_SIMD
    static void LoadFactors(ushort* norms, int groups, float scale, float* factors)
    {
        var multiplier = Vector128.Create(scale);
        var g = 0;
        for (; g + 4 <= groups; g += 4)
            Sse.Store(factors + g, Sse.Multiply(HalfToSingle(Sse41.ConvertToVector128Int32((ushort*)(norms + g)).AsUInt32()), multiplier));
        for (; g < groups; g++)
            factors[g] = HalfToSingle(Vector128.Create((uint)*(ushort*)(norms + g), 0u, 0u, 0u)).ToScalar() * scale;
    }
#endif

    public static void MatMul(in Cq w, in Activations a, int tokens, float* y, int ldy) =>
        MatMulRows(w, a, tokens, y, ldy, 0, w.Rows);

    public static void MatMulRows(in Cq w, in Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.MatMulRows(w, a, tokens, y, ldy, rowStart, rowEnd); return; }
#endif
        if (!Simd.Enabled) { Scalar.MatMulRows(w, a, tokens, y, ldy, rowStart, rowEnd); return; }
#if NEEDLE_SIMD
        var tiledTokens = tokens & ~(TokenTile - 1);
        for (var t = tiledTokens; t < tokens; t++) Gemv.Rows(w, a, t, y + (long)t * ldy, rowStart, rowEnd);
        if (tiledTokens == 0) return;
        var factors = stackalloc float[a.Stride];
        var scaled = stackalloc float[a.Stride * TokenTile];
        var groups = w.Groups;
        var rowBytes = w.RowBytes;
        var table = Avx.LoadVector256(w.Book->Bytes);
        var byteScale = w.Book->ByteScale;
        var two = w.Bits == 2;
        if ((rowEnd - rowStart) * TokenTile <= tokens)
        {
            for (var t = 0; t < tiledTokens; t += TokenTile)
                for (var n = rowStart; n < rowEnd; n++)
                {
                    LoadFactors(w.Norms + (long)n * groups, groups, byteScale, factors);
                    var packed = w.Indices + (long)n * rowBytes;
                    if (two) DotTile2(packed, table, factors, scaled, a, t, y + (long)t * ldy + n, ldy);
                    else DotTile4(packed, table, factors, scaled, a, t, y + (long)t * ldy + n, ldy);
                }
            return;
        }
        for (var n = rowStart; n < rowEnd; n++)
        {
            LoadFactors(w.Norms + (long)n * groups, groups, byteScale, factors);
            var packed = w.Indices + (long)n * rowBytes;
            for (var t = 0; t < tiledTokens; t += TokenTile)
                if (two) DotTile2(packed, table, factors, scaled, a, t, y + (long)t * ldy + n, ldy);
                else DotTile4(packed, table, factors, scaled, a, t, y + (long)t * ldy + n, ldy);
        }
#endif
    }

#if NEEDLE_SIMD
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> ScaleFactors(float* factors, float* scaled, float* scales, float* offsets, int stride)
    {
        var bias = Vector256<float>.Zero;
        for (var v = 0; v < stride; v += 8)
        {
            var f = Avx.Multiply(Avx.LoadVector256(factors + v), Avx.LoadVector256(scales + v));
            Avx.Store(scaled + v, f);
            bias = Fma.MultiplyAdd(f, Avx.LoadVector256(offsets + v), bias);
        }
        return bias;
    }
#endif

#if NEEDLE_SIMD
    static void DotTile2(byte* packed, Vector256<byte> table, float* factors, float* scaled, in Activations a, int token, float* y, int ldy)
    {
        var groups = a.Groups;
        var stride = a.Stride;
        var cols = a.Cols;
        var bias0 = ScaleFactors(factors, scaled, a.Scales + token * stride, a.Offsets + token * stride, stride);
        var bias1 = ScaleFactors(factors, scaled + stride, a.Scales + (token + 1) * stride, a.Offsets + (token + 1) * stride, stride);
        var bias2 = ScaleFactors(factors, scaled + 2 * stride, a.Scales + (token + 2) * stride, a.Offsets + (token + 2) * stride, stride);
        var bias3 = ScaleFactors(factors, scaled + 3 * stride, a.Scales + (token + 3) * stride, a.Offsets + (token + 3) * stride, stride);
        var v0 = a.Values + (long)token * cols;
        var mask = Vector256.Create((byte)0x03);
        var a0 = Vector256<float>.Zero;
        var a1 = Vector256<float>.Zero;
        var a2 = Vector256<float>.Zero;
        var a3 = Vector256<float>.Zero;
        for (var g = 0; g < groups; g++)
        {
            var offset = g * Cq.GroupSize;
            var v = Avx.LoadVector256(packed + g * 32).AsUInt16();
            var p0 = Avx2.Shuffle(table, Avx2.And(v.AsByte(), mask));
            var p1 = Avx2.Shuffle(table, Avx2.And(Avx2.ShiftRightLogical(v, 2).AsByte(), mask));
            var p2 = Avx2.Shuffle(table, Avx2.And(Avx2.ShiftRightLogical(v, 4).AsByte(), mask));
            var p3 = Avx2.Shuffle(table, Avx2.And(Avx2.ShiftRightLogical(v, 6).AsByte(), mask));
            a0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Planes4Dot(p0, p1, p2, p3, v0 + offset)), Vector256.Create(scaled[g]), a0);
            a1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Planes4Dot(p0, p1, p2, p3, v0 + cols + offset)), Vector256.Create(scaled[stride + g]), a1);
            a2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Planes4Dot(p0, p1, p2, p3, v0 + 2 * cols + offset)), Vector256.Create(scaled[2 * stride + g]), a2);
            a3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Planes4Dot(p0, p1, p2, p3, v0 + 3 * cols + offset)), Vector256.Create(scaled[3 * stride + g]), a3);
        }
        y[0] = Vector256.Sum(Avx.Subtract(a0, bias0));
        y[ldy] = Vector256.Sum(Avx.Subtract(a1, bias1));
        y[2 * ldy] = Vector256.Sum(Avx.Subtract(a2, bias2));
        y[3 * ldy] = Vector256.Sum(Avx.Subtract(a3, bias3));
    }
#endif

#if NEEDLE_SIMD
    static void DotTile4(byte* packed, Vector256<byte> table, float* factors, float* scaled, in Activations a, int token, float* y, int ldy)
    {
        var groups = a.Groups;
        var stride = a.Stride;
        var cols = a.Cols;
        var bias0 = ScaleFactors(factors, scaled, a.Scales + token * stride, a.Offsets + token * stride, stride);
        var bias1 = ScaleFactors(factors, scaled + stride, a.Scales + (token + 1) * stride, a.Offsets + (token + 1) * stride, stride);
        var bias2 = ScaleFactors(factors, scaled + 2 * stride, a.Scales + (token + 2) * stride, a.Offsets + (token + 2) * stride, stride);
        var bias3 = ScaleFactors(factors, scaled + 3 * stride, a.Scales + (token + 3) * stride, a.Offsets + (token + 3) * stride, stride);
        var v0 = a.Values + (long)token * cols;
        var mask = Vector256.Create((byte)0x0F);
        var a0 = Vector256<float>.Zero;
        var a1 = Vector256<float>.Zero;
        var a2 = Vector256<float>.Zero;
        var a3 = Vector256<float>.Zero;
        for (var g = 0; g < groups; g++)
        {
            var offset = g * Cq.GroupSize;
            var lo = Avx.LoadVector256(packed + g * 64).AsUInt16();
            var hi = Avx.LoadVector256(packed + g * 64 + 32).AsUInt16();
            var p0 = Avx2.Shuffle(table, Avx2.And(lo.AsByte(), mask));
            var p1 = Avx2.Shuffle(table, Avx2.And(Avx2.ShiftRightLogical(lo, 4).AsByte(), mask));
            var p2 = Avx2.Shuffle(table, Avx2.And(hi.AsByte(), mask));
            var p3 = Avx2.Shuffle(table, Avx2.And(Avx2.ShiftRightLogical(hi, 4).AsByte(), mask));
            a0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Planes4Dot(p0, p1, p2, p3, v0 + offset)), Vector256.Create(scaled[g]), a0);
            a1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Planes4Dot(p0, p1, p2, p3, v0 + cols + offset)), Vector256.Create(scaled[stride + g]), a1);
            a2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Planes4Dot(p0, p1, p2, p3, v0 + 2 * cols + offset)), Vector256.Create(scaled[2 * stride + g]), a2);
            a3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Planes4Dot(p0, p1, p2, p3, v0 + 3 * cols + offset)), Vector256.Create(scaled[3 * stride + g]), a3);
        }
        y[0] = Vector256.Sum(Avx.Subtract(a0, bias0));
        y[ldy] = Vector256.Sum(Avx.Subtract(a1, bias1));
        y[2 * ldy] = Vector256.Sum(Avx.Subtract(a2, bias2));
        y[3 * ldy] = Vector256.Sum(Avx.Subtract(a3, bias3));
    }
#endif

#if NEEDLE_SIMD
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<int> Planes4Dot(Vector256<byte> p0, Vector256<byte> p1, Vector256<byte> p2, Vector256<byte> p3, sbyte* values)
    {
        var sum = Accumulate(Vector256<int>.Zero, p0, Avx.LoadVector256(values));
        sum = Accumulate(sum, p1, Avx.LoadVector256(values + 32));
        sum = Accumulate(sum, p2, Avx.LoadVector256(values + 64));
        return Accumulate(sum, p3, Avx.LoadVector256(values + 96));
    }
#endif
}

internal unsafe struct MatMulJob
{
    public Cq W;
    public float* Y;
    public int Ldy;
}

internal unsafe struct MatMulJobs
{
    MatMulJob first, second, third, fourth;

    public MatMulJob* At(int index)
    {
        fixed (MatMulJob* p = &first) return p + index;
    }
}

internal unsafe struct MatMulBatch
{
    public MatMulJobs Jobs;
    public Activations Activations;
    public int Tokens;
    public int Count;
    public int RowsPerTask;

    public MatMulBatch(in Activations activations, int tokens)
    {
        Activations = activations;
        Tokens = tokens;
        Count = 0;
        RowsPerTask = 0;
    }

    public void Add(in Cq w, float* y, int ldy) => *Jobs.At(Count++) = new MatMulJob { W = w, Y = y, Ldy = ldy };

    public static void Run(Pool pool, MatMulBatch* batch)
    {
        var rows = 0;
        for (var j = 0; j < batch->Count; j++) rows += batch->Jobs.At(j)->W.Rows;
        batch->RowsPerTask = Math.Max(8, (rows + pool.Threads * 3 - 1) / (pool.Threads * 3));
        var tasks = 0;
        for (var j = 0; j < batch->Count; j++) tasks += TaskCount(batch, j);
        pool.Run(&Execute, batch, tasks);
    }

    static int TaskCount(MatMulBatch* batch, int job) => (batch->Jobs.At(job)->W.Rows + batch->RowsPerTask - 1) / batch->RowsPerTask;

    static void Execute(void* context, int index)
    {
        var batch = (MatMulBatch*)context;
        for (var j = 0; j < batch->Count; j++)
        {
            var tasks = TaskCount(batch, j);
            if (index >= tasks) { index -= tasks; continue; }
            var job = batch->Jobs.At(j);
            var start = index * batch->RowsPerTask;
            Quant.MatMulRows(job->W, batch->Activations, batch->Tokens, job->Y, job->Ldy, start, Math.Min(job->W.Rows, start + batch->RowsPerTask));
            return;
        }
    }
}
