using System;
#if NEEDLE_SIMD
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace Needle;

internal static unsafe class Attention
{
    const float InverseSqrtQkDim = 0.14433756729740643f;

#if NEEDLE_SIMD
    public static int DotInt8x48(sbyte* a, sbyte* b)
    {
        var sum = Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(a), Avx2.ConvertToVector256Int16(b));
        sum = Avx2.Add(sum, Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(a + 16), Avx2.ConvertToVector256Int16(b + 16)));
        sum = Avx2.Add(sum, Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(a + 32), Avx2.ConvertToVector256Int16(b + 32)));
        return Vector256.Sum(sum);
    }
#endif

    public static void Head(
        sbyte* query, float queryScale, sbyte* keys, float* keyScales, sbyte* values, float* valueScales,
        int keyStride, int valueStride, int scaleStride, int first, int last, float* scores, float* output, float* probabilities = null)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.AttentionHead(query, queryScale, keys, keyScales, values, valueScales, keyStride, valueStride, scaleStride, first, last, scores, output, probabilities); return; }
#endif
        if (!Simd.Enabled) { Scalar.AttentionHead(query, queryScale, keys, keyScales, values, valueScales, keyStride, valueStride, scaleStride, first, last, scores, output, probabilities); return; }
#if NEEDLE_SIMD
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
#endif
    }
}

internal unsafe struct AttentionBatch
{
    public sbyte* Queries;
    public float* QueryScales;
    public sbyte* Keys;
    public float* KeyScales;
    public sbyte* Values;
    public float* ValueScales;
    public float* Output;
    public float* Probabilities;
    public int Heads, KvHeads, QkDim, VDim, Position, Window, Capacity, Frames;

    public static void Run(Pool pool, AttentionBatch* batch, int tokens) => pool.Run(&Execute, batch, tokens * batch->Heads);

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void Execute(void* context, int task)
    {
        var b = (AttentionBatch*)context;
        var token = task / b->Heads;
        var head = task % b->Heads;
        var kv = head / (b->Heads / b->KvHeads);
        var position = b->Position + token;
        var first = b->Window == 0 ? 0 : Math.Max(0, position - b->Window + 1);
        var last = b->Frames > 0 ? b->Frames - 1 : position;
        var scores = stackalloc float[b->Capacity + 16];
        Attention.Head(
            b->Queries + (long)token * b->Heads * b->QkDim + head * b->QkDim, b->QueryScales[token * b->Heads + head],
            b->Keys + kv * b->QkDim, b->KeyScales + kv, b->Values + kv * b->VDim, b->ValueScales + kv,
            b->KvHeads * b->QkDim, b->KvHeads * b->VDim, b->KvHeads, first, last, scores,
            b->Output + (long)token * b->Heads * b->VDim + head * b->VDim,
            b->Probabilities == null ? null : b->Probabilities + ((long)token * b->Heads + head) * b->Frames);
    }
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

    public static void Run(Pool pool, MlpBatch* batch, int tokens) => pool.Run(&Execute, batch, tokens);

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void Execute(void* context, int token)
    {
        var b = (MlpBatch*)context;
        var width = b->Rows * b->Cols;
        var normalized = stackalloc float[b->Dim];
        var output = stackalloc float[b->Dim];
        var condition = stackalloc float[width];
        var first = stackalloc float[width];
        var second = stackalloc float[width];
        var residual = b->Residual + (long)token * b->Dim;
        Ops.ZeroCenteredRms(residual, b->PreNorm, b->Dim, normalized);
        HadamardMlp.Apply(*b->Weights, b->PermA, b->PermB, normalized, b->Dim, b->Rows, b->Cols, condition, first, second, output);
        Ops.AddScaled(residual, output, b->Scale, b->Dim);
    }
}

internal static unsafe class HadamardMlp
{
    const int Conditions = 8;

    public static void Apply(in MlpWeights w, int* permA, int* permB, float* input, int dim, int rows, int cols, float* condition, float* a, float* b, float* output)
    {
        var width = rows * cols;
        Condition(w.CondV, w.CondU, input, dim, width, condition);
        for (var i = 0; i < dim; i++) a[i] = input[i] * w.D1[i];
        for (var i = dim; i < width; i++) a[i] = 0f;
        Kron(a, w.W1A, w.W1B, rows, cols, b);
        for (var i = 0; i < width; i++) a[i] = b[permA[i]];
        Modulate(w, condition, a, width);
        Ops.Silu(a, width);
        Kron(a, w.W2A, w.W2B, rows, cols, b);
        for (var i = 0; i < width; i++) a[i] = b[permB[i]] * w.D3[i];
        Kron(a, w.W3A, w.W3B, rows, cols, b);
        for (var i = 0; i < dim; i++) output[i] = b[i] * w.D4[i];
    }

    static void Modulate(in MlpWeights w, float* condition, float* a, int width)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.ModulateActivate(w.D2, condition, a, w.B2, width); return; }
#endif
        if (!Simd.Enabled) { Scalar.ModulateActivate(w.D2, condition, a, w.B2, width); return; }
#if NEEDLE_SIMD
        for (var i = 0; i < width; i += 8)
        {
            var v = Fma.MultiplyAdd(Avx.Multiply(Avx.LoadVector256(w.D2 + i), Avx.LoadVector256(condition + i)), Avx.LoadVector256(a + i), Avx.LoadVector256(w.B2 + i));
            Avx.Store(a + i, v);
        }
#endif
    }

    static void Condition(float* condV, float* condU, float* input, int dim, int width, float* condition)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Condition(condV, condU, input, dim, width, condition); return; }
#endif
        if (!Simd.Enabled) { Scalar.Condition(condV, condU, input, dim, width, condition); return; }
#if NEEDLE_SIMD
        var logits = Vector256<float>.Zero;
        for (var i = 0; i < dim; i++) logits = Fma.MultiplyAdd(Vector256.Create(input[i]), Avx.LoadVector256(condV + i * Conditions), logits);
        var weights = stackalloc float[Conditions];
        Avx.Store(weights, logits);
        Ops.Softmax(weights, Conditions);
        for (var n = 0; n < width; n += 8)
        {
            var acc = Vector256.Create(1f);
            for (var j = 0; j < Conditions; j++) acc = Fma.MultiplyAdd(Vector256.Create(weights[j]), Avx.LoadVector256(condU + j * width + n), acc);
            Avx.Store(condition + n, acc);
        }
#endif
    }

    static void Kron(float* z, float* left, float* right, int rows, int cols, float* result)
    {
        #if NEEDLE_SIMD
        if (Simd.UseNeon) { Neon.Kron(z, left, right, rows, cols, result); return; }
#endif
        if (!Simd.Enabled) { Scalar.Kron(z, left, right, rows, cols, result); return; }
#if NEEDLE_SIMD
        var scratch = stackalloc float[rows * cols];
        for (var k = 0; k < rows; k++)
        {
            for (var c = 0; c < cols; c += 8)
            {
                var acc = Vector256<float>.Zero;
                for (var i = 0; i < rows; i++) acc = Fma.MultiplyAdd(Vector256.Create(left[i * rows + k]), Avx.LoadVector256(z + i * cols + c), acc);
                Avx.Store(scratch + k * cols + c, acc);
            }
        }
        for (var k = 0; k < rows; k++)
        {
            for (var c = 0; c < cols; c += 8)
            {
                var acc = Vector256<float>.Zero;
                for (var j = 0; j < cols; j++) acc = Fma.MultiplyAdd(Vector256.Create(scratch[k * cols + j]), Avx.LoadVector256(right + j * cols + c), acc);
                Avx.Store(result + k * cols + c, acc);
            }
        }
#endif
    }
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
