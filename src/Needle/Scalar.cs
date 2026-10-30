using System;
#if NEEDLE_SIMD
using System.Runtime.Intrinsics.X86;
#endif

namespace Needle;

internal static class Simd
{
#if NEEDLE_SIMD
    static bool forceScalar;

    public static bool Avx => Avx2.IsSupported && Fma.IsSupported && Sse41.IsSupported;

    public static bool Neon => System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported;

    public static bool Enabled => !forceScalar && (Avx || Neon);

    public static bool UseAvx => !forceScalar && Avx;

    public static bool UseNeon => !forceScalar && Neon;

#pragma warning disable CA2255
    [System.Runtime.CompilerServices.ModuleInitializer]
    internal static void ReadEnvironment() => forceScalar = Environment.GetEnvironmentVariable("NEEDLE_SCALAR") != null;
#pragma warning restore CA2255
#else
    public const bool Enabled = false;
#endif
}

internal static unsafe class Scalar
{
    const float HadamardScale = 0.08838834764831845f;
    const float InverseSqrtQkDim = 0.14433756729740643f;

    public static float AbsMax(float* x, int n)
    {
        var peak = 0f;
        for (var i = 0; i < n; i++) peak = MathF.Max(peak, MathF.Abs(x[i]));
        return peak;
    }

    public static void RoundTrip(float* x, int n, float inverse, float scale)
    {
        for (var i = 0; i < n; i++) x[i] = MathF.Min(MathF.Max(MathF.Round(x[i] * inverse), -128f), 127f) * scale;
    }

    public static float QuantizeInt8(float* x, int n, sbyte* destination)
    {
        var peak = AbsMax(x, n);
        var scale = peak > 0f ? peak / 127f : 1f;
        var inverse = 1f / scale;
        for (var i = 0; i < n; i++) destination[i] = (sbyte)MathF.Min(MathF.Max(MathF.Round(x[i] * inverse), -128f), 127f);
        return scale;
    }

    public static void Hadamard(float* source, float* p)
    {
        if (source != p) for (var i = 0; i < Cq.GroupSize; i++) p[i] = source[i];
        for (var stride = 1; stride < Cq.GroupSize; stride <<= 1)
            for (var block = 0; block < Cq.GroupSize; block += stride * 2)
                for (var i = block; i < block + stride; i++)
                {
                    var a = p[i];
                    var b = p[i + stride];
                    p[i] = a + b;
                    p[i + stride] = a - b;
                }
        for (var i = 0; i < Cq.GroupSize; i++) p[i] *= HadamardScale;
    }

    public static void Planes(sbyte* quantized, int bits, sbyte* destination, out int sum)
    {
        sum = 0;
        if (bits == 2)
            for (var j = 0; j < 32; j++)
                for (var p = 0; p < 4; p++)
                {
                    var q = quantized[4 * j + p];
                    destination[p * 32 + j] = q;
                    sum += q;
                }
        else
            for (var half = 0; half < 2; half++)
                for (var j = 0; j < 32; j++)
                    for (var p = 0; p < 2; p++)
                    {
                        var q = quantized[half * 64 + 2 * j + p];
                        destination[half * 64 + p * 32 + j] = q;
                        sum += q;
                    }
    }

    public static float HalfToSingleUpstream(ushort half)
    {
        var magnitude = half & 0x7FFF;
        if (magnitude == 0) return 0f;
        var bits = ((uint)magnitude << 13) + 0x38000000u | (uint)(half & 0x8000) << 16;
        return *(float*)&bits;
    }

    public static float HalfToSingle(ushort half)
    {
        var magnitude = half & 0x7FFF;
        if (magnitude == 0) return 0f;
        if (magnitude < 0x0400) return (half & 0x8000) != 0 ? -magnitude * 5.9604645e-8f : magnitude * 5.9604645e-8f;
        var bits = ((uint)magnitude << 13) + 0x38000000u | (uint)(half & 0x8000) << 16;
        return *(float*)&bits;
    }

    public static void DecodeRow(in Cq w, int row, float* destination)
    {
        var indices = w.Indices + (long)row * w.RowBytes;
        var norms = (ushort*)w.Norms + (long)row * w.Groups;
        var levels = w.Book->Levels;
        for (var g = 0; g < w.Groups; g++)
        {
            var norm = HalfToSingle(norms[g]);
            var d = destination + g * Cq.GroupSize;
            if (w.Bits == 2)
                for (var i = 0; i < Cq.GroupSize; i++) d[i] = levels[indices[g * 32 + i / 4] >> (i % 4 * 2) & 3] * norm;
            else
                for (var i = 0; i < Cq.GroupSize; i++) d[i] = levels[indices[g * 64 + i / 2] >> (i % 2 * 4) & 15] * norm;
        }
    }

    public static void MatMulRows(in Cq w, in Activations a, int tokens, float* y, int ldy, int rowStart, int rowEnd)
    {
        var groups = w.Groups;
        var table = w.Book->Bytes;
        var byteScale = w.Book->ByteScale;
        for (var t = 0; t < tokens; t++)
        {
            var values = a.Values + (long)t * a.Cols;
            var scales = a.Scales + t * a.Stride;
            var offsets = a.Offsets + t * a.Stride;
            for (var n = rowStart; n < rowEnd; n++)
            {
                var packed = w.Indices + (long)n * w.RowBytes;
                var norms = (ushort*)w.Norms + (long)n * groups;
                var accumulator = 0f;
                var bias = 0f;
                for (var g = 0; g < groups; g++)
                {
                    var x = values + g * Cq.GroupSize;
                    var dot = 0;
                    if (w.Bits == 2)
                    {
                        var p = packed + g * 32;
                        for (var j = 0; j < 32; j++)
                        {
                            int b = p[j];
                            dot += table[b & 3] * x[j] + table[b >> 2 & 3] * x[32 + j] + table[b >> 4 & 3] * x[64 + j] + table[b >> 6 & 3] * x[96 + j];
                        }
                    }
                    else
                    {
                        var p = packed + g * 64;
                        for (var half = 0; half < 2; half++)
                            for (var j = 0; j < 32; j++)
                            {
                                int b = p[half * 32 + j];
                                dot += table[b & 15] * x[2 * half * 32 + j] + table[b >> 4] * x[(2 * half + 1) * 32 + j];
                            }
                    }
                    var scaled = HalfToSingle(norms[g]) * byteScale * scales[g];
                    accumulator += scaled * dot;
                    bias += scaled * offsets[g];
                }
                y[(long)t * ldy + n] = accumulator - bias;
            }
        }
    }

    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    public static void Silu(float* x, int n)
    {
        for (var i = 0; i < n; i++) x[i] *= Sigmoid(x[i]);
    }

    public static void MultiplyBySigmoid(float* x, float* gate, int n)
    {
        for (var i = 0; i < n; i++) x[i] *= Sigmoid(gate[i]);
    }

    public static float SumOfSquares(float* x, int n)
    {
        var sum = 0f;
        for (var i = 0; i < n; i++) sum += x[i] * x[i];
        return sum;
    }

    public static float Dot(float* a, float* b, int n)
    {
        var sum = 0f;
        for (var i = 0; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    public static void Softmax(float* s, int n)
    {
        var peak = float.NegativeInfinity;
        for (var i = 0; i < n; i++) peak = MathF.Max(peak, s[i]);
        var total = 0f;
        for (var i = 0; i < n; i++)
        {
            s[i] = MathF.Exp(s[i] - peak);
            total += s[i];
        }
        for (var i = 0; i < n; i++) s[i] /= total;
    }

    public static void Rope(float* x, int heads, int headDim, float* cos, float* sin)
    {
        var half = headDim / 2;
        for (var h = 0; h < heads; h++)
        {
            var lo = x + h * headDim;
            var hi = lo + half;
            for (var i = 0; i < half; i++)
            {
                var a = lo[i];
                var b = hi[i];
                lo[i] = a * cos[i] - b * sin[i];
                hi[i] = b * cos[i] + a * sin[i];
            }
        }
    }

    public static void Combine4(float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination)
    {
        for (var c = 0; c < dim; c++) destination[c] = post * y[c] + w0 * s[c] + w1 * s[dim + c] + w2 * s[2 * dim + c] + w3 * s[3 * dim + c];
    }

    public static void Mix4(float* s, int dim, float w0, float w1, float w2, float w3, float* destination)
    {
        for (var c = 0; c < dim; c++) destination[c] = w0 * s[c] + w1 * s[dim + c] + w2 * s[2 * dim + c] + w3 * s[3 * dim + c];
    }

    public static void Subtract(float* a, float* b, int n, float* destination)
    {
        for (var i = 0; i < n; i++) destination[i] = a[i] - b[i];
    }

    public static void ScaleAdd(float* destination, float scale, float* x, float xScale, int n)
    {
        for (var i = 0; i < n; i++) destination[i] = scale * destination[i] + xScale * x[i];
    }

    public static void MultiplyAccumulate(float* destination, float* a, float* b, int n)
    {
        for (var i = 0; i < n; i++) destination[i] += a[i] * b[i];
    }

    public static void AddScaled(float* destination, float* x, float scale, int n)
    {
        for (var i = 0; i < n; i++) destination[i] += scale * x[i];
    }

    public static void Glu(float* value, float* gate, float* destination, int n)
    {
        for (var i = 0; i < n; i++) destination[i] = value[i] * Sigmoid(gate[i]);
    }

    public static void Taps3(float* taps, int width, float* current, float* destination)
    {
        for (var c = 0; c < width; c++) destination[c] = taps[c] * current[c] + taps[width + c] * current[c - width] + taps[2 * width + c] * current[c - 2 * width];
    }

    public static void AttentionHead(
        sbyte* query, float queryScale, sbyte* keys, float* keyScales, sbyte* values, float* valueScales,
        int keyStride, int valueStride, int scaleStride, int first, int last, float* scores, float* output, float* probabilities)
    {
        var count = last - first + 1;
        var scale = queryScale * InverseSqrtQkDim;
        for (var s = 0; s < count; s++)
        {
            var key = keys + (long)(first + s) * keyStride;
            var dot = 0;
            for (var d = 0; d < 48; d++) dot += query[d] * key[d];
            scores[s] = dot * keyScales[(long)(first + s) * scaleStride] * scale;
        }
        Softmax(scores, count);
        if (probabilities != null) for (var s = 0; s < count; s++) probabilities[first + s] = scores[s];
        for (var d = 0; d < 64; d++) output[d] = 0f;
        for (var s = 0; s < count; s++)
        {
            var row = values + (long)(first + s) * valueStride;
            var weight = scores[s] * valueScales[(long)(first + s) * scaleStride];
            for (var d = 0; d < 64; d++) output[d] += weight * row[d];
        }
    }

    public static void AttentionTile(in AttentionTile tile)
    {
        for (var h = 0; h < tile.Heads; h++)
            AttentionHead(
                tile.Queries + h * 48, tile.QueryScales[h], tile.Keys, tile.KeyScales, tile.Values, tile.ValueScales,
                tile.KeyStride, tile.ValueStride, tile.ScaleStride, tile.First, tile.First + tile.Count - 1, tile.Scores,
                tile.Output + h * 64, tile.Probabilities == null ? null : tile.Probabilities + h * tile.ProbabilityStride);
    }

    public static void ConditionWeights(float* condV, float* input, int dim, float* weights)
    {
        for (var j = 0; j < 8; j++)
        {
            var sum = 0f;
            for (var i = 0; i < dim; i++) sum += input[i] * condV[i * 8 + j];
            weights[j] = sum;
        }
        Softmax(weights, 8);
    }

    public static void ScaleInput(float* input, float* d1, int dim, int padded, float* destination)
    {
        for (var i = 0; i < dim; i++) destination[i] = input[i] * d1[i];
        for (var i = dim; i < padded; i++) destination[i] = 0f;
    }

    public static void PermuteModulateActivate(int* permutation, float* source, float* d2, float* weights, float* condU, float* b2, int width, float* destination)
    {
        for (var n = 0; n < width; n++)
        {
            var condition = 1f;
            for (var j = 0; j < 8; j++) condition += weights[j] * condU[j * width + n];
            var v = d2[n] * condition * source[permutation[n]] + b2[n];
            destination[n] = v * Sigmoid(v);
        }
    }

    public static void PermuteScale(int* permutation, float* source, float* scale, int width, float* destination)
    {
        for (var n = 0; n < width; n++) destination[n] = source[permutation[n]] * scale[n];
    }

    public static void AccumulateOutput(float* residual, float* product, float* d4, float scale, int count)
    {
        for (var i = 0; i < count; i++) residual[i] += scale * (product[i] * d4[i]);
    }

    [System.Runtime.CompilerServices.SkipLocalsInit]
    public static void Kron(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
    {
        var scratch = stackalloc float[cols];
        for (var k = 0; k < resultRows; k++)
        {
            for (var c = 0; c < cols; c++)
            {
                var sum = 0f;
                for (var i = 0; i < activeRows; i++) sum += left[i * rows + k] * z[i * cols + c];
                scratch[c] = sum;
            }
            for (var c = 0; c < cols; c++)
            {
                var sum = 0f;
                for (var j = 0; j < cols; j++) sum += scratch[j] * right[j * cols + c];
                result[k * cols + c] = sum;
            }
        }
    }
}
