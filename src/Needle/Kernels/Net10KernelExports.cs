using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#if NEEDLE_SIMD
namespace Needle;

public static unsafe class Net10KernelExports
{
    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void MatMulRows(Cq* w, Activations* a, int tokens, float* y, int ldy, int rowStart, int rowEnd) =>
        Quant.MatMulRows(*w, *a, tokens, y, ldy, rowStart, rowEnd);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void Prepare(float* x, int tokens, int cols, int bits, Activations* a) =>
        Quant.Prepare(x, tokens, cols, bits, *a);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void GatherRow(Cq* w, int row, float* destination) =>
        Quant.GatherRow(*w, row, destination);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void AttentionTile(AttentionTile* tile) =>
        Attention.Run(*tile);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void MergePartials(float* partials, int count, int heads, float* output) =>
        Attention.MergePartials(partials, count, heads, output);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void HadamardMlpApply(MlpBatch* b, float* residual, float* scratch) =>
        HadamardMlp.Apply(b, residual, scratch);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void RmsFakeA8(float* x, int n, float* destination) =>
        Ops.RmsFakeA8(x, n, destination);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void CenteredRmsFakeA8(float* x, float* scale, int n, float* destination) =>
        Ops.CenteredRmsFakeA8(x, scale, n, destination);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void GateFakeA8(float* x, float* gate, int n) =>
        Ops.GateFakeA8(x, gate, n);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void PostNormAdd(float* residual, float* x, float* scale, float gate, int n) =>
        Ops.PostNormAdd(residual, x, scale, gate, n);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void CombineDiff4(float* s, int dim, float w0, float w1, float w2, float w3, float* x, float* u, float post, float* destination) =>
        Ops.CombineDiff4(s, dim, w0, w1, w2, w3, x, u, post, destination);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void Mix4(float* s, int dim, float w0, float w1, float w2, float w3, float* destination) =>
        Ops.Mix4(s, dim, w0, w1, w2, w3, destination);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void Combine4(float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination) =>
        Ops.Combine4(s, dim, w0, w1, w2, w3, y, post, destination);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void Taps3(float* taps, int width, float* current, float* destination) =>
        Ops.Taps3(taps, width, current, destination);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void Rope(float* x, int heads, int headDim, float* cos, float* sin) =>
        Ops.Rope(x, heads, headDim, cos, sin);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void ZeroCenteredRms(float* x, float* scale, int n, float* destination) =>
        Ops.ZeroCenteredRms(x, scale, n, destination);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void RmsUnit(float* x, int n, float* destination) =>
        Ops.RmsUnit(x, n, destination);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void Softmax(float* s, int n) =>
        Ops.Softmax(s, n);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static float Dot(float* a, float* b, int n) =>
        Ops.Dot(a, b, n);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void AddScaled(float* destination, float* x, float scale, int n) =>
        Ops.AddScaled(destination, x, scale, n);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void ScaleAdd(float* destination, float scale, float* x, float xScale, int n) =>
        Ops.ScaleAdd(destination, scale, x, xScale, n);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void MultiplyAccumulate(float* destination, float* a, float* b, int n) =>
        Ops.MultiplyAccumulate(destination, a, b, n);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static float QuantizeInt8(float* x, int n, sbyte* destination) =>
        Quant.QuantizeInt8(x, n, destination);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static float FakeA8(float* x, int n) =>
        Quant.FakeA8(x, n);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void Sinkhorn4x4(float* logits, float* result) =>
        Ops.Sinkhorn4x4(logits, result);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static int ArgMaxMasked(float* x, uint* mask, int words, int length) =>
        Ops.ArgMaxMasked(x, mask, words, length);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static float BestProbability(float* values, int count, float best) =>
        Ops.BestProbability(values, count, best);

    public static void Fill(nint tablePtr, int count)
    {
        if (count < KernelTable.FieldCount || tablePtr == 0)
            throw new ArgumentException("Invalid KernelTable target.", nameof(tablePtr));
        var p = (nint*)tablePtr;
        p[0] = (nint)(delegate* unmanaged[Cdecl]<Cq*, Activations*, int, float*, int, int, int, void>)&MatMulRows;
        p[1] = (nint)(delegate* unmanaged[Cdecl]<float*, int, int, int, Activations*, void>)&Prepare;
        p[2] = (nint)(delegate* unmanaged[Cdecl]<Cq*, int, float*, void>)&GatherRow;
        p[3] = (nint)(delegate* unmanaged[Cdecl]<AttentionTile*, void>)&AttentionTile;
        p[4] = (nint)(delegate* unmanaged[Cdecl]<float*, int, int, float*, void>)&MergePartials;
        p[5] = (nint)(delegate* unmanaged[Cdecl]<MlpBatch*, float*, float*, void>)&HadamardMlpApply;
        p[6] = (nint)(delegate* unmanaged[Cdecl]<float*, int, float*, void>)&RmsFakeA8;
        p[7] = (nint)(delegate* unmanaged[Cdecl]<float*, float*, int, float*, void>)&CenteredRmsFakeA8;
        p[8] = (nint)(delegate* unmanaged[Cdecl]<float*, float*, int, void>)&GateFakeA8;
        p[9] = (nint)(delegate* unmanaged[Cdecl]<float*, float*, float*, float, int, void>)&PostNormAdd;
        p[10] = (nint)(delegate* unmanaged[Cdecl]<float*, int, float, float, float, float, float*, float*, float, float*, void>)&CombineDiff4;
        p[11] = (nint)(delegate* unmanaged[Cdecl]<float*, int, float, float, float, float, float*, void>)&Mix4;
        p[12] = (nint)(delegate* unmanaged[Cdecl]<float*, int, float, float, float, float, float*, float, float*, void>)&Combine4;
        p[13] = (nint)(delegate* unmanaged[Cdecl]<float*, int, float*, float*, void>)&Taps3;
        p[14] = (nint)(delegate* unmanaged[Cdecl]<float*, int, int, float*, float*, void>)&Rope;
        p[15] = (nint)(delegate* unmanaged[Cdecl]<float*, float*, int, float*, void>)&ZeroCenteredRms;
        p[16] = (nint)(delegate* unmanaged[Cdecl]<float*, int, float*, void>)&RmsUnit;
        p[17] = (nint)(delegate* unmanaged[Cdecl]<float*, int, void>)&Softmax;
        p[18] = (nint)(delegate* unmanaged[Cdecl]<float*, float*, int, float>)&Dot;
        p[19] = (nint)(delegate* unmanaged[Cdecl]<float*, float*, float, int, void>)&AddScaled;
        p[20] = (nint)(delegate* unmanaged[Cdecl]<float*, float, float*, float, int, void>)&ScaleAdd;
        p[21] = (nint)(delegate* unmanaged[Cdecl]<float*, float*, float*, int, void>)&MultiplyAccumulate;
        p[22] = (nint)(delegate* unmanaged[Cdecl]<float*, int, sbyte*, float>)&QuantizeInt8;
        p[23] = (nint)(delegate* unmanaged[Cdecl]<float*, int, float>)&FakeA8;
        p[24] = (nint)(delegate* unmanaged[Cdecl]<float*, float*, void>)&Sinkhorn4x4;
        p[25] = (nint)(delegate* unmanaged[Cdecl]<float*, uint*, int, int, int>)&ArgMaxMasked;
        p[26] = (nint)(delegate* unmanaged[Cdecl]<float*, int, float, float>)&BestProbability;
    }

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void ExecuteDecodeTeam(void* context, int member) =>
        EngineCore<NoClock>.ExecuteDecodeTeam(context, member);

    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    static void ExecutePrefillTeam(void* context, int member) =>
        EngineCore<NoClock>.ExecutePrefillTeam(context, member);

    public static void FillTeam(nint tablePtr, int decodeJobBytes)
    {
        if (tablePtr == 0 || decodeJobBytes != sizeof(EngineCore<NoClock>.DecodeJob)) return;
        var p = (nint*)(void*)tablePtr;
        p[0] = (nint)(delegate* unmanaged[Cdecl]<void*, int, void>)&ExecuteDecodeTeam;
        p[1] = (nint)(delegate* unmanaged[Cdecl]<void*, int, void>)&ExecutePrefillTeam;
    }
}
#endif
