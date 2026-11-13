using System;
using System.Runtime.InteropServices;

#if !NEEDLE_SIMD
namespace Needle;

internal static unsafe class PortableKernels
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void MatMulRowsFn(Cq* w, Activations* a, int tokens, float* y, int ldy, int rowStart, int rowEnd);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void HadamardMlpFn(MlpBatch* b, float* residual, float* scratch);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void RmsFakeA8Fn(float* x, int n, float* destination);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void CenteredRmsFakeA8Fn(float* x, float* scale, int n, float* destination);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void GateFakeA8Fn(float* x, float* gate, int n);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void PostNormAddFn(float* residual, float* x, float* scale, float gate, int n);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void CombineDiff4Fn(float* s, int dim, float w0, float w1, float w2, float w3, float* x, float* u, float post, float* destination);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void Mix4Fn(float* s, int dim, float w0, float w1, float w2, float w3, float* destination);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void Combine4Fn(float* s, int dim, float w0, float w1, float w2, float w3, float* y, float post, float* destination);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void Taps3Fn(float* taps, int width, float* current, float* destination);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void RopeFn(float* x, int heads, int headDim, float* cos, float* sin);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void ZeroCenteredRmsFn(float* x, float* scale, int n, float* destination);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void RmsUnitFn(float* x, int n, float* destination);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void SoftmaxFn(float* s, int n);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate float DotFn(float* a, float* b, int n);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void AddScaledFn(float* destination, float* x, float scale, int n);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void ScaleAddFn(float* destination, float scale, float* x, float xScale, int n);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate void MultiplyAccumulateFn(float* destination, float* a, float* b, int n);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate float QuantizeInt8Fn(float* x, int n, sbyte* destination);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate float FakeA8Fn(float* x, int n);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate float BestProbabilityFn(float* values, int count, float best);

    static readonly MatMulRowsFn MatMulRowsHolder = MatMulRowsEntry;
    static readonly HadamardMlpFn HadamardMlpHolder = HadamardMlpEntry;
    static readonly RmsFakeA8Fn RmsFakeA8Holder = PortableVector.RmsFakeA8;
    static readonly CenteredRmsFakeA8Fn CenteredRmsFakeA8Holder = PortableVector.CenteredRmsFakeA8;
    static readonly GateFakeA8Fn GateFakeA8Holder = PortableVector.GateFakeA8;
    static readonly PostNormAddFn PostNormAddHolder = PortableVector.PostNormAdd;
    static readonly CombineDiff4Fn CombineDiff4Holder = PortableVector.CombineDiff4;
    static readonly Mix4Fn Mix4Holder = PortableVector.Mix4;
    static readonly Combine4Fn Combine4Holder = PortableVector.Combine4;
    static readonly Taps3Fn Taps3Holder = PortableVector.Taps3;
    static readonly RopeFn RopeHolder = PortableVector.Rope;
    static readonly ZeroCenteredRmsFn ZeroCenteredRmsHolder = PortableVector.ZeroCenteredRms;
    static readonly RmsUnitFn RmsUnitHolder = PortableVector.RmsUnit;
    static readonly SoftmaxFn SoftmaxHolder = PortableVector.Softmax;
    static readonly DotFn DotHolder = PortableVector.Dot;
    static readonly AddScaledFn AddScaledHolder = PortableVector.AddScaled;
    static readonly ScaleAddFn ScaleAddHolder = PortableVector.ScaleAdd;
    static readonly MultiplyAccumulateFn MultiplyAccumulateHolder = PortableVector.MultiplyAccumulate;
    static readonly QuantizeInt8Fn QuantizeInt8Holder = PortableVector.QuantizeInt8;
    static readonly FakeA8Fn FakeA8Holder = PortableVector.FakeA8;
    static readonly BestProbabilityFn BestProbabilityHolder = PortableVector.BestProbability;

    static void MatMulRowsEntry(Cq* w, Activations* a, int tokens, float* y, int ldy, int rowStart, int rowEnd) =>
        PortableLut.MatMulRows(*w, *a, tokens, y, ldy, rowStart, rowEnd);

    static void HadamardMlpEntry(MlpBatch* b, float* residual, float* scratch)
    {
        var w = b->Weights;
        var dim = b->Dim;
        var rows = b->Rows;
        var cols = b->Cols;
        var width = b->Width;
        var inputRows = (dim + cols - 1) / cols;
        var outputRows = Math.Min(rows, (inputRows + 3) / 4 * 4);
        HadamardMlp.Layout(scratch, dim, width, out var weights, out var normalized, out var front, out var back);
        PortableVector.ZeroCenteredRms(residual, b->PreNorm, dim, normalized);
        Scalar.ConditionWeights(w->CondV, normalized, dim, weights);
        Scalar.ScaleInput(normalized, w->D1, dim, inputRows * cols, front);
        PortableMlp.Kron(front, inputRows, w->W1A, w->W1B, rows, cols, rows, back);
        Scalar.PermuteModulateActivate(b->PermA, back, w->D2, weights, w->CondU, w->B2, width, front);
        PortableMlp.Kron(front, rows, w->W2A, w->W2B, rows, cols, rows, back);
        Scalar.PermuteScale(b->PermB, back, w->D3, width, front);
        PortableMlp.Kron(front, rows, w->W3A, w->W3B, rows, cols, outputRows, back);
        Scalar.AccumulateOutput(residual, back, w->D4, b->Scale, dim);
    }

    public static bool TryInstall()
    {
        try
        {
            var table = Kernels.Active;
            if (table.MatMulRows == 0) table.MatMulRows = Marshal.GetFunctionPointerForDelegate(MatMulRowsHolder);
            if (table.AttentionTile == 0) table.AttentionTile = PortableAttention.Address;
            if (table.HadamardMlp == 0) table.HadamardMlp = Marshal.GetFunctionPointerForDelegate(HadamardMlpHolder);
            if (table.RmsFakeA8 == 0) table.RmsFakeA8 = Marshal.GetFunctionPointerForDelegate(RmsFakeA8Holder);
            if (table.CenteredRmsFakeA8 == 0) table.CenteredRmsFakeA8 = Marshal.GetFunctionPointerForDelegate(CenteredRmsFakeA8Holder);
            if (table.GateFakeA8 == 0) table.GateFakeA8 = Marshal.GetFunctionPointerForDelegate(GateFakeA8Holder);
            if (table.PostNormAdd == 0) table.PostNormAdd = Marshal.GetFunctionPointerForDelegate(PostNormAddHolder);
            if (table.CombineDiff4 == 0) table.CombineDiff4 = Marshal.GetFunctionPointerForDelegate(CombineDiff4Holder);
            if (table.Mix4 == 0) table.Mix4 = Marshal.GetFunctionPointerForDelegate(Mix4Holder);
            if (table.Combine4 == 0) table.Combine4 = Marshal.GetFunctionPointerForDelegate(Combine4Holder);
            if (table.Taps3 == 0) table.Taps3 = Marshal.GetFunctionPointerForDelegate(Taps3Holder);
            if (table.Rope == 0) table.Rope = Marshal.GetFunctionPointerForDelegate(RopeHolder);
            if (table.ZeroCenteredRms == 0) table.ZeroCenteredRms = Marshal.GetFunctionPointerForDelegate(ZeroCenteredRmsHolder);
            if (table.RmsUnit == 0) table.RmsUnit = Marshal.GetFunctionPointerForDelegate(RmsUnitHolder);
            if (table.Softmax == 0) table.Softmax = Marshal.GetFunctionPointerForDelegate(SoftmaxHolder);
            if (table.Dot == 0) table.Dot = Marshal.GetFunctionPointerForDelegate(DotHolder);
            if (table.AddScaled == 0) table.AddScaled = Marshal.GetFunctionPointerForDelegate(AddScaledHolder);
            if (table.ScaleAdd == 0) table.ScaleAdd = Marshal.GetFunctionPointerForDelegate(ScaleAddHolder);
            if (table.MultiplyAccumulate == 0) table.MultiplyAccumulate = Marshal.GetFunctionPointerForDelegate(MultiplyAccumulateHolder);
            if (table.QuantizeInt8 == 0) table.QuantizeInt8 = Marshal.GetFunctionPointerForDelegate(QuantizeInt8Holder);
            if (table.FakeA8 == 0) table.FakeA8 = Marshal.GetFunctionPointerForDelegate(FakeA8Holder);
            if (table.BestProbability == 0) table.BestProbability = Marshal.GetFunctionPointerForDelegate(BestProbabilityHolder);
            Kernels.Install(table, Kernels.IsInstalled ? Kernels.Provider + "+portable" : "portable");
            return true;
        }
        catch
        {
            return false;
        }
    }
}
#endif
