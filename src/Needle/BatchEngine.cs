using System;

namespace Needle;

// W26 multi-sequence batched decode: N independent sequences share one
// engine pass per decode step. Weights stream once for all rows (the decode
// step is weight-bound: 17.5 MB per token), while attention, convolution
// history, engram state, pooling and the KV cache stay per sequence in
// slots. Prefill stays per sequence through FeedSlot; only the decode step
// batches. Grammar masks still differ per sequence: after each shared
// forward the caller applies its own mask and argmax (see BatchedSession),
// which is where the sharing stops — the per-sequence argmax, grammar walk
// and token bookkeeping remain serial CPU work.
//
// Bit-identity contract: one batched step over rows produces the identical
// logits and KV/history updates as feeding each slot's rows alone in the
// same order. Attention runs one batch per slot run (it reads only that
// sequence's KV, so sharing the pass would save nothing) to keep the exact
// tiling the single-sequence path uses.
public sealed unsafe class BatchedEngine : IDisposable
{
    readonly EngineCore<NoClock> engine;

    public BatchedEngine(Model model, ThreadPolicy policy, int contextLength = 1024, int chunkSize = 64, int slots = 8)
    {
        if (slots < 1) throw new ArgumentOutOfRangeException(nameof(slots));
        if (slots > 16) throw new ArgumentOutOfRangeException(nameof(slots), "at most 16 slots fit one decode step");
        engine = new EngineCore<NoClock>(model, contextLength, chunkSize, policy, null, default, slots) { Pooling = true };
    }

    public int Slots => engine.Slots;

    public int Capacity => engine.SlotCapacity;

    public int VocabSize => engine.Vocab;

    public int SlotLength(int slot) => engine.SlotLength(slot);

    public void ResetSlot(int slot) => engine.ResetSlot(slot);

    public void MarkSlot(int slot) => engine.MarkSlot(slot);

    public void RewindSlot(int slot) => engine.RewindSlot(slot);

    public void FeedSlot(int slot, ReadOnlySpan<int> tokens, bool computeLogits = true) =>
        engine.FeedSlot(slot, tokens, computeLogits);

    internal void ForwardBatch(int* tokens, int* rowSlots, int count, int* headRows, int headCount) =>
        engine.ForwardBatch(tokens, rowSlots, count, headRows, headCount);

    internal float* BatchLogitsRow(int compactRow) => engine.BatchLogitsRow(compactRow);

    internal void LogitsForBatchRow(int row, int* rows, int count, float* destination) =>
        engine.LogitsForBatchRow(row, rows, count, destination);

    internal float ConfidenceLogitForSlot(int slot, float* pooledOut, float* scratchRows) =>
        engine.ConfidenceLogitForSlot(slot, pooledOut, scratchRows);

    public void Pause() => engine.Pause();

    public void Resume() => engine.Resume();

    public void Dispose() => engine.Dispose();
}
