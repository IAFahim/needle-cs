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

    public int Width => Rows * Cols;

    public static void Run(Pool pool, MlpBatch* batch, int tokens) => pool.Run(&Execute, batch, tokens);

    [System.Runtime.CompilerServices.SkipLocalsInit]
    static void Execute(void* context, int token)
    {
        var b = (MlpBatch*)context;
        var scratch = stackalloc float[HadamardMlp.ScratchFloats(b->Dim, b->Width)];
        HadamardMlp.Apply(b, b->Residual + (long)token * b->Dim, scratch);
    }
}

[System.Runtime.CompilerServices.SkipLocalsInit]
internal static unsafe class HadamardMlp
{
    const int Conditions = 8;
    const int TileRows = 4;
    const int TileColumns = 16;

    public static int ScratchFloats(int dim, int width) => Conditions + dim + 2 * width;

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
        var weights = scratch;
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

    public static void KronAvx512(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
    {
        var tile = stackalloc float[TileRows * cols];
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
    public static void KronAvx2(float* z, int activeRows, float* left, float* right, int rows, int cols, int resultRows, float* result)
    {
        var tile = stackalloc float[TileRows * cols];
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
