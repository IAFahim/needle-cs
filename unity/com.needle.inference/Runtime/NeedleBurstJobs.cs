#if NEEDLE_BURST
using System;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;

namespace Needle.Unity
{
    [BurstCompile]
    internal unsafe struct MatMulRowsJob : IJobParallelFor
    {
        public Cq Weights;
        public Activations Input;
        public int Tokens;
        public int RowsPerBatch;
        public int OutputStride;
        [NativeDisableUnsafePtrRestriction] public float* Output;

        public void Execute(int batch)
        {
            var first = batch * RowsPerBatch;
            Quant.MatMulRows(Weights, Input, Tokens, Output, OutputStride, first, Math.Min(Weights.Rows, first + RowsPerBatch));
        }
    }

    [BurstCompile]
    internal unsafe struct PrepareActivationsJob : IJobParallelFor
    {
        public Activations Output;
        public int Columns;
        public int Bits;
        [NativeDisableUnsafePtrRestriction] public float* Input;

        public void Execute(int token) => Quant.Prepare(Input + (long)token * Columns, 1, Columns, Bits, Output.Slice(token));
    }
}
#endif
