using System;
using System.Runtime.InteropServices;

namespace Needle;

internal unsafe struct CrossMemory
{
    public sbyte* Keys;
    public float* KeyScales;
    public sbyte* Values;
    public float* ValueScales;
    public int Frames;
}

public delegate void LayerObserver(int layer, ReadOnlySpan<float> stream);

public unsafe class EngineCore<TClock> : IDisposable where TClock : struct, IStageClock
{
    const int Lanes = 4;
    const int ConvTaps = 3;
    const int EngramHistory = 9;
    const int EngramTaps = 4;

    readonly Model model;
    readonly TClock clock;
    readonly Pool pool;
    readonly bool ownsPool;
    internal float* Alignment;
    internal int AlignmentFromLayer;
    float* captureBuffer;
    readonly Geometry g;
    readonly int chunk;
    readonly int capacity;
    readonly int dim;
    readonly int streamWidth;
    readonly int qWidth, kWidth, vWidth, historyPerLayer;
    readonly float embeddingScale;
    Arena arena;

    float* stream, nextStream, normalizedStream, preOut, postOut, resOut, postGate, residualMix;
    Activations streamAct, dimAct, headAct;
    float* input, residual, quantizedInput, qRaw, kRaw, vRaw, qConv, kConv, vConv, gateOut, attnOut, attnProj, scratch;
    float* engramInput, engramKey, engramConv, engramValueBuffer, logits;
    float* ropeCos, ropeSin, keyScale, valueScale;
    sbyte* queryInt8, keyCache, valueCache;
    float* queryScale;
    int* engramRows;

    readonly float* state;
    readonly float* snapshot;
    readonly nuint stateBytes, historyBytes, poolBytes;
    readonly int poolRows;
    int* snapshotTokens;
    float* snapshotPoolMax, snapshotPoolSum, snapshotPoolAcc;
    bool historyPending, poolPending;
    float* qkvHistory, engramHistory;
    int* previousTokens;
    int markedLength;
    // Token count of the last Forward chunk; the sparse head reads that row.
    int lastForwardCount;
    float* poolMax, poolSum, poolAcc, cell;
    bool pooling;
    CrossMemory cross;
    // W26 multi-sequence slots. The single-sequence fields above stay the
    // only state the prefetch/feed path touches; slots mirror their layout
    // (state block, snapshot block, KV cache, lengths, pending flags) so one
    // batched forward can step N independent sequences through the shared
    // weight reads. Null unless the engine was constructed with slots.
    readonly int slotCount;
    readonly SlotView[]? slotViews;
    // Batched-decode scratch, allocated only when slots > 0.
    float* batchLogits;
    float* batchQ, batchK, batchV;
    float* batchEngramStage;

    public int Length { get; private set; }

    internal long StateBytes => (long)stateBytes;

    public LayerObserver? Observer { get; set; }

    public int Greedy() => Ops.ArgMax(logits, model.Embedding.Rows);

    public ReadOnlySpan<float> Logits => new(logits, model.Embedding.Rows);

    internal EngineCore(Model model, int contextLength, int chunkSize, ThreadPolicy policy, Pool? shared, TClock clock, int slots = 0)
    {
        this.model = model;
        this.clock = clock;
        ownsPool = shared == null;
        pool = shared ?? new Pool(policy);
        g = model.Geometry;
        chunk = chunkSize;
        capacity = Math.Min(contextLength, g.MaxSeq);
        dim = g.Dim;
        streamWidth = Lanes * dim;
        qWidth = g.QueryWidth;
        kWidth = g.KeyWidth;
        vWidth = g.ValueWidth;
        historyPerLayer = (ConvTaps - 1) * (qWidth + kWidth + vWidth);
        embeddingScale = MathF.Sqrt(dim);

        var batchPoolRows = model.Confidence != null ? (g.Layers + 1) * model.Confidence->Probes : 0;
        var batchStateFloats = (nuint)(g.Layers * historyPerLayer + g.SiteCount * EngramHistory * dim + 2 + batchPoolRows * (dim + 2));
        var kvKeys = (nuint)capacity * (nuint)g.Layers * (nuint)kWidth;
        var kvValues = (nuint)capacity * (nuint)g.Layers * (nuint)vWidth;
        var kvScales = (nuint)capacity * (nuint)g.Layers * (nuint)g.KvHeads;
        var batchScratchFloats = (nuint)model.Embedding.Rows + (nuint)(ConvTaps * (qWidth + kWidth + vWidth)) + (nuint)((EngramHistory + 1) * dim);
        var extra = slots > 0
            ? (nuint)slots * (2 * batchStateFloats * sizeof(float) + kvKeys + kvValues + 2 * kvScales * sizeof(float))
                + (nuint)chunk * batchScratchFloats * sizeof(float)
            : 0;
        var kvBytes = (nuint)capacity * (nuint)g.Layers * (nuint)(kWidth + vWidth + 2 * g.KvHeads * sizeof(float));
        arena = new Arena(kvBytes + (nuint)chunk * 160_000 + (nuint)capacity * 512 + (16u << 20) + extra);

        stream = arena.Take<float>((nuint)(chunk * streamWidth));
        nextStream = arena.Take<float>((nuint)(chunk * streamWidth));
        normalizedStream = arena.Take<float>((nuint)(chunk * streamWidth));
        preOut = arena.Take<float>((nuint)(chunk * Lanes));
        postOut = arena.Take<float>((nuint)(chunk * Lanes));
        resOut = arena.Take<float>((nuint)(chunk * Lanes * Lanes));
        postGate = arena.Take<float>((nuint)(chunk * Lanes));
        residualMix = arena.Take<float>((nuint)(chunk * Lanes * Lanes));
        input = arena.Take<float>((nuint)(chunk * dim));
        residual = arena.Take<float>((nuint)(chunk * dim));
        quantizedInput = arena.Take<float>((nuint)(chunk * dim));
        streamAct = new Activations(ref arena, chunk, streamWidth);
        dimAct = new Activations(ref arena, chunk, dim);
        headAct = new Activations(ref arena, 1, dim);
        qRaw = arena.Take<float>((nuint)((chunk + ConvTaps - 1) * qWidth));
        kRaw = arena.Take<float>((nuint)((chunk + ConvTaps - 1) * kWidth));
        vRaw = arena.Take<float>((nuint)((chunk + ConvTaps - 1) * vWidth));
        qConv = arena.Take<float>((nuint)(chunk * qWidth));
        kConv = arena.Take<float>((nuint)(chunk * kWidth));
        vConv = arena.Take<float>((nuint)(chunk * vWidth));
        gateOut = arena.Take<float>((nuint)(chunk * g.OutputWidth));
        attnOut = arena.Take<float>((nuint)(chunk * g.OutputWidth));
        attnProj = arena.Take<float>((nuint)(chunk * dim));
        scratch = arena.Take<float>((nuint)(4 * dim));
        engramInput = arena.Take<float>((nuint)(chunk * dim));
        engramKey = arena.Take<float>((nuint)(chunk * dim));
        engramConv = arena.Take<float>((nuint)(chunk * dim));
        engramValueBuffer = arena.Take<float>((nuint)((chunk + EngramHistory) * dim));
        logits = arena.Take<float>((nuint)model.Embedding.Rows);
        queryInt8 = arena.Take<sbyte>((nuint)(chunk * qWidth));
        queryScale = arena.Take<float>((nuint)(chunk * g.Heads));
        engramRows = arena.Take<int>((nuint)(chunk * g.EngramTables));
        keyCache = arena.TakeUncleared<sbyte>((nuint)capacity * (nuint)g.Layers * (nuint)kWidth);
        valueCache = arena.TakeUncleared<sbyte>((nuint)capacity * (nuint)g.Layers * (nuint)vWidth);
        keyScale = arena.TakeUncleared<float>((nuint)capacity * (nuint)g.Layers * (nuint)g.KvHeads);
        valueScale = arena.TakeUncleared<float>((nuint)capacity * (nuint)g.Layers * (nuint)g.KvHeads);
        BuildRopeTables();

        poolRows = model.Confidence != null ? (g.Layers + 1) * model.Confidence->Probes : 0;
        var poolFloats = poolRows * (dim + 2);
        cell = arena.Take<float>((nuint)dim);
        var stateFloats = (nuint)(g.Layers * historyPerLayer + g.SiteCount * EngramHistory * dim + 2 + poolFloats);
        stateBytes = stateFloats * sizeof(float);
        state = arena.Take<float>(stateFloats);
        snapshot = arena.Take<float>(stateFloats);
        qkvHistory = state;
        engramHistory = state + g.Layers * historyPerLayer;
        previousTokens = (int*)(engramHistory + g.SiteCount * EngramHistory * dim);
        poolMax = (float*)(previousTokens + 2);
        poolSum = poolMax + poolRows;
        poolAcc = poolSum + poolRows;
        historyBytes = (nuint)((byte*)previousTokens - (byte*)state);
        poolBytes = (nuint)poolFloats * sizeof(float);
        snapshotTokens = (int*)(snapshot + ((float*)previousTokens - state));
        snapshotPoolMax = snapshot + (poolMax - state);
        snapshotPoolSum = snapshot + (poolSum - state);
        snapshotPoolAcc = snapshot + (poolAcc - state);
        ResetPool();
        slotCount = Math.Max(0, slots);
        if (slotCount > 0)
        {
            slotViews = new SlotView[slotCount];
            for (var s = 0; s < slotCount; s++)
            {
                ref var view = ref slotViews[s];
                view.State = arena.Take<float>(stateFloats);
                view.Snapshot = arena.Take<float>(stateFloats);
                view.Keys = arena.TakeUncleared<sbyte>((nuint)capacity * (nuint)g.Layers * (nuint)kWidth);
                view.Values = arena.TakeUncleared<sbyte>((nuint)capacity * (nuint)g.Layers * (nuint)vWidth);
                view.KeyScales = arena.TakeUncleared<float>((nuint)capacity * (nuint)g.Layers * (nuint)g.KvHeads);
                view.ValueScales = arena.TakeUncleared<float>((nuint)capacity * (nuint)g.Layers * (nuint)g.KvHeads);
                ResetView(ref view);
            }
            batchLogits = arena.Take<float>((nuint)chunk * (nuint)model.Embedding.Rows);
            batchQ = arena.Take<float>((nuint)(chunk * ConvTaps) * (nuint)qWidth);
            batchK = arena.Take<float>((nuint)(chunk * ConvTaps) * (nuint)kWidth);
            batchV = arena.Take<float>((nuint)(chunk * ConvTaps) * (nuint)vWidth);
            batchEngramStage = arena.Take<float>((nuint)chunk * (nuint)(EngramHistory + 1) * (nuint)dim);
        }
    }

    void ResetPool()
    {
        if (model.Confidence == null) return;
        for (var i = 0; i < poolRows; i++) poolMax[i] = -1e30f;
    }

    float* HistoryRead(float* live) => historyPending ? snapshot + (live - state) : live;

    void RestorePoolRow(int index)
    {
        poolMax[index] = snapshotPoolMax[index];
        poolSum[index] = snapshotPoolSum[index];
        Buffer.MemoryCopy(snapshotPoolAcc + (long)index * dim, poolAcc + (long)index * dim, dim * 4, dim * 4);
    }

    void RestorePool()
    {
        if (!poolPending) return;
        Buffer.MemoryCopy(snapshotPoolMax, poolMax, poolBytes, poolBytes);
        poolPending = false;
    }

    internal bool Pooling
    {
        get => pooling;
        set => pooling = value && model.Confidence != null;
    }

    internal void Attach(in CrossMemory memory) => cross = memory;

    void PoolTokens(int row, int count, bool initial)
    {
        var head = model.Confidence;
        var probes = head->Probes;
        var inverseRoot = 1f / MathF.Sqrt(dim);
        for (var t = 0; t < count; t++)
        {
            var s = stream + (long)t * streamWidth;
            if (initial) Buffer.MemoryCopy(s, cell, dim * 4, dim * 4);
            else Ops.Mix4(s, dim, 0.25f, 0.25f, 0.25f, 0.25f, cell);
            for (var p = 0; p < probes; p++)
            {
                var index = row * probes + p;
                if (poolPending && t == 0) RestorePoolRow(index);
                var score = Ops.Dot(cell, head->ProbeVectors + (long)index * dim, dim) * inverseRoot;
                var acc = poolAcc + (long)index * dim;
                if (score > poolMax[index])
                {
                    var rescale = MathF.Exp(poolMax[index] - score);
                    poolSum[index] = poolSum[index] * rescale + 1f;
                    Ops.ScaleAdd(acc, rescale, cell, 1f, dim);
                    poolMax[index] = score;
                }
                else
                {
                    var weight = MathF.Exp(score - poolMax[index]);
                    poolSum[index] += weight;
                    Ops.AddScaled(acc, cell, weight, dim);
                }
            }
        }
    }

    internal float ConfidenceLogit(float* pooledOut)
    {
        var scratch = (float*)Mem.AllocZeroed((nuint)((g.Layers + 1) * model.Confidence->Probes * (dim + 1)) * 4);
        var logit = ConfidenceLogit(pooledOut, scratch);
        Mem.Free(scratch);
        return logit;
    }

    internal float ConfidenceLogit(float* pooledOut, float* scratchRows)
    {
        RestorePool();
        var head = model.Confidence;
        var probes = head->Probes;
        var rows = (g.Layers + 1) * probes;
        var vectors = scratchRows;
        var weights = scratchRows + (long)rows * dim;
        var inverseRoot = 1f / MathF.Sqrt(dim);
        for (var i = 0; i < rows; i++)
        {
            var v = vectors + (long)i * dim;
            Ops.ScaleAdd(v, 0f, poolAcc + (long)i * dim, 1f / poolSum[i], dim);
            Ops.RmsUnit(v, dim, v);
            Ops.ScaleAdd(v, head->Gain[i], v, 0f, dim);
        }
        var logit = head->Bias[0];
        for (var q = 0; q < head->Queries; q++)
        {
            for (var i = 0; i < rows; i++)
                weights[i] = Ops.Dot(vectors + (long)i * dim, head->QueryVectors + (long)q * dim, dim) * inverseRoot + head->RowBias[q * rows + i];
            Ops.Softmax(weights, rows);
            var pooled = pooledOut + q * dim;
            Mem.Clear(pooled, (nuint)dim * 4);
            for (var i = 0; i < rows; i++) Ops.AddScaled(pooled, vectors + (long)i * dim, weights[i], dim);
        }
        logit += Ops.Dot(pooledOut, head->Projection, head->Queries * dim);
        return logit;
    }

    void BuildRopeTables()
    {
        var half = g.QkDim / 2;
        ropeCos = arena.Take<float>((nuint)(capacity * half));
        ropeSin = arena.Take<float>((nuint)(capacity * half));
        for (var p = 0; p < capacity; p++)
            for (var i = 0; i < half; i++)
            {
                var frequency = 1f / MathF.Pow(g.RopeTheta, 2f * i / g.QkDim);
                var angle = p * frequency;
                ropeCos[p * half + i] = MathF.Cos(angle);
                ropeSin[p * half + i] = MathF.Sin(angle);
            }
    }

    public void Reset()
    {
        Mem.Clear(state, stateBytes);
        ResetPool();
        historyPending = false;
        poolPending = false;
        Length = 0;
    }

    internal void CopyFrom(EngineCore<TClock> source)
    {
        Buffer.MemoryCopy(source.state, state, stateBytes, stateBytes);
        Length = source.Length;
        CopyPrefix(source.logits, logits, 0, model.Embedding.Rows);
        for (var l = 0; l < g.Layers; l++)
        {
            CopyPrefix(source.keyCache, keyCache, (long)l * capacity * kWidth, (long)Length * kWidth);
            CopyPrefix(source.valueCache, valueCache, (long)l * capacity * vWidth, (long)Length * vWidth);
            CopyPrefix(source.keyScale, keyScale, (long)l * capacity * g.KvHeads, (long)Length * g.KvHeads);
            CopyPrefix(source.valueScale, valueScale, (long)l * capacity * g.KvHeads, (long)Length * g.KvHeads);
        }
    }

    internal int MarkedLength => markedLength;

    internal long PrimedPrefixBytes =>
        sizeof(int) + (long)stateBytes +
        (long)g.Layers * markedLength * (kWidth + vWidth + 2L * g.KvHeads * sizeof(float));

    internal void ExportPrimedPrefix(Span<byte> destination)
    {
        if (destination.Length < PrimedPrefixBytes) throw new ArgumentException("primed prefix buffer too small");
        fixed (byte* dst = destination)
        {
            *(int*)dst = markedLength;
            var at = dst + sizeof(int);
            Buffer.MemoryCopy(snapshot, at, stateBytes, stateBytes);
            at += stateBytes;
            for (var l = 0; l < g.Layers; l++)
            {
                var keys = (long)markedLength * kWidth;
                var values = (long)markedLength * vWidth;
                var scales = (long)markedLength * g.KvHeads * sizeof(float);
                Buffer.MemoryCopy(keyCache + (long)l * capacity * kWidth, at, keys, keys);
                at += keys;
                Buffer.MemoryCopy(valueCache + (long)l * capacity * vWidth, at, values, values);
                at += values;
                Buffer.MemoryCopy(keyScale + (long)l * capacity * g.KvHeads, at, scales, scales);
                at += scales;
                Buffer.MemoryCopy(valueScale + (long)l * capacity * g.KvHeads, at, scales, scales);
                at += scales;
            }
        }
    }

    internal void ImportPrimedPrefix(ReadOnlySpan<byte> source)
    {
        fixed (byte* src = source)
        {
            var length = *(int*)src;
            if (length < 0 || length > capacity) throw new ArgumentException("primed prefix length exceeds context capacity");
            var expected = sizeof(int) + (long)stateBytes +
                (long)g.Layers * length * (kWidth + vWidth + 2L * g.KvHeads * sizeof(float));
            if (source.Length < expected) throw new ArgumentException("primed prefix is truncated");
            var at = src + sizeof(int);
            Buffer.MemoryCopy(at, state, stateBytes, stateBytes);
            Buffer.MemoryCopy(at, snapshot, stateBytes, stateBytes);
            at += stateBytes;
            for (var l = 0; l < g.Layers; l++)
            {
                var keys = (long)length * kWidth;
                var values = (long)length * vWidth;
                var scales = (long)length * g.KvHeads * sizeof(float);
                Buffer.MemoryCopy(at, keyCache + (long)l * capacity * kWidth, keys, keys);
                at += keys;
                Buffer.MemoryCopy(at, valueCache + (long)l * capacity * vWidth, values, values);
                at += values;
                Buffer.MemoryCopy(at, keyScale + (long)l * capacity * g.KvHeads, scales, scales);
                at += scales;
                Buffer.MemoryCopy(at, valueScale + (long)l * capacity * g.KvHeads, scales, scales);
                at += scales;
            }
            Length = length;
            markedLength = length;
            historyPending = false;
            poolPending = false;
        }
    }

    static void CopyPrefix<T>(T* from, T* to, long offset, long count) where T : unmanaged =>
        Buffer.MemoryCopy(from + offset, to + offset, count * sizeof(T), count * sizeof(T));

    public void Mark()
    {
        if (!historyPending) Buffer.MemoryCopy(state, snapshot, historyBytes, historyBytes);
        snapshotTokens[0] = previousTokens[0];
        snapshotTokens[1] = previousTokens[1];
        if (!poolPending) Buffer.MemoryCopy(poolMax, snapshotPoolMax, poolBytes, poolBytes);
        markedLength = Length;
    }

    public void Rewind()
    {
        previousTokens[0] = snapshotTokens[0];
        previousTokens[1] = snapshotTokens[1];
        historyPending = true;
        poolPending = poolRows > 0;
        Length = markedLength;
    }

    public void Feed(ReadOnlySpan<int> tokens, bool computeLogits = true)
    {
        if (Length + tokens.Length > capacity) throw new InvalidOperationException("context capacity exceeded");
        fixed (int* all = tokens)
            for (var start = 0; start < tokens.Length; start += chunk)
            {
                var count = Math.Min(chunk, tokens.Length - start);
                var last = start + count == tokens.Length;
                Forward(all + start, count, Length, last && computeLogits);
                RememberTokens(all + start, count);
                Length += count;
            }
    }

    void RememberTokens(int* tokens, int count)
    {
        if (count >= 2) { previousTokens[0] = tokens[count - 2]; previousTokens[1] = tokens[count - 1]; }
        else { previousTokens[0] = previousTokens[1]; previousTokens[1] = tokens[0]; }
    }

    void Forward(int* tokens, int count, int position, bool wantLogits)
    {
        lastForwardCount = count;
        clock.Begin();
        for (var t = 0; t < count; t++)
        {
            Quant.GatherRow(model.Embedding, tokens[t], scratch);
            var row = stream + (long)t * streamWidth;
            Ops.AddScaled(scratch, scratch, embeddingScale - 1f, dim);
            for (var lane = 0; lane < Lanes; lane++) Buffer.MemoryCopy(scratch, row + lane * dim, dim * 4, dim * 4);
        }
        EngramHash.Rows(g, tokens, count, position, previousTokens, engramRows);
        clock.Lap(Stage.Embed);
        Observer?.Invoke(-1, new ReadOnlySpan<float>(stream, count * streamWidth));
        if (pooling) PoolTokens(0, count, true);
        for (var layer = 0; layer < g.Layers; layer++)
        {
            RunLayer(layer, count, position);
            if (pooling) PoolTokens(layer + 1, count, false);
            Observer?.Invoke(layer, new ReadOnlySpan<float>(stream, count * streamWidth));
        }
        historyPending = false;
        if (pooling) poolPending = false;
        if (wantLogits) ComputeLogits(count - 1);
    }

    void RunLayer(int l, int count, int position)
    {
        ref readonly var w = ref model.Blocks[l];
        ref readonly var hyper = ref model.Hyper;
        var lane = l % Lanes;

        for (var t = 0; t < count; t++)
        {
            var normalized = normalizedStream + (long)t * streamWidth;
            Ops.RmsUnit(stream + (long)t * streamWidth, streamWidth, normalized);
            Quant.FakeA8(normalized, streamWidth);
        }
        Quant.Prepare(normalizedStream, count, streamWidth, 4, streamAct);
        clock.Lap(Stage.HyperPrepare);
        Quant.MatMul(hyper.PhiPre.Slice(l * Lanes, Lanes), streamAct, count, preOut, Lanes);
        Quant.MatMul(hyper.PhiPost.Slice(l * Lanes, Lanes), streamAct, count, postOut, Lanes);
        Quant.MatMul(hyper.PhiRes.Slice(l * Lanes * Lanes, Lanes * Lanes), streamAct, count, resOut, Lanes * Lanes);
        clock.Lap(Stage.HyperPhi);

        var mixLogits = stackalloc float[Lanes * Lanes];
        var hpre = stackalloc float[Lanes];
        for (var t = 0; t < count; t++)
        {
            var s = stream + (long)t * streamWidth;
            var u = input + (long)t * dim;
            for (var i = 0; i < Lanes; i++)
            {
                hpre[i] = Ops.Sigmoid(hyper.APre[l] * preOut[t * Lanes + i] + hyper.BPre[l * Lanes + i] + (i == lane ? 4f : -4f));
                postGate[t * Lanes + i] = 2f * Ops.Sigmoid(hyper.APost[l] * postOut[t * Lanes + i] + hyper.BPost[l * Lanes + i] + (i == lane ? 0f : -4f));
            }
            Ops.Mix4(s, dim, hpre[0], hpre[1], hpre[2], hpre[3], u);
            for (var i = 0; i < Lanes * Lanes; i++) mixLogits[i] = hyper.ARes[l] * resOut[t * Lanes * Lanes + i] + hyper.BRes[l * Lanes * Lanes + i];
            Sinkhorn.Normalize(mixLogits, residualMix + t * Lanes * Lanes);
            Buffer.MemoryCopy(u, residual + (long)t * dim, dim * 4, dim * 4);
        }
        clock.Lap(Stage.HyperMix);

        var site = g.SiteOf(l);
        if (site >= 0) ApplyEngram(site, count, position);
        clock.Lap(Stage.Engram);
        RunAttention(l, w, count, position);
        if (cross.Frames > 0) RunCross(l, count, position);
        RunMlp(l, count);
        clock.Lap(Stage.Mlp);

        for (var t = 0; t < count; t++)
        {
            var s = stream + (long)t * streamWidth;
            var n = nextStream + (long)t * streamWidth;
            var y = scratch;
            var x = residual + (long)t * dim;
            var u = input + (long)t * dim;
            Ops.Subtract(x, u, dim, y);
            var mix = residualMix + t * Lanes * Lanes;
            for (var i = 0; i < Lanes; i++)
                Ops.Combine4(s, dim, mix[i * 4], mix[i * 4 + 1], mix[i * 4 + 2], mix[i * 4 + 3], y, postGate[t * Lanes + i], n + i * dim);
        }
        clock.Lap(Stage.Combine);
        var swap = stream;
        stream = nextStream;
        nextStream = swap;
    }

    void ApplyEngram(int site, int count, int position)
    {
        var s = model.Sites[site];
        var sub = g.EngramSub;
        for (var t = 0; t < count; t++)
        {
            var e = engramInput + (long)t * dim;
            for (var table = 0; table < g.EngramTables; table++)
            {
                var row = engramRows[t * g.EngramTables + table];
                if (row < 0) Mem.Clear(e + table * sub, (nuint)sub * 4);
                else Quant.GatherRow(s.Tables, row, e + table * sub);
            }
            Quant.FakeA8(e, dim);
        }
        Quant.Prepare(engramInput, count, dim, 2, dimAct);
        var history = engramHistory + (long)site * EngramHistory * dim;
        Buffer.MemoryCopy(HistoryRead(history), engramValueBuffer, EngramHistory * dim * 4, EngramHistory * dim * 4);
        var batch = new MatMulBatch(dimAct, count);
        batch.Add(s.Key, engramKey, dim);
        batch.Add(s.Value, engramValueBuffer + EngramHistory * dim, dim);
        clock.PoolRun();
        MatMulBatch.Run(pool, &batch);
        for (var t = 0; t < count; t++)
        {
            var o = engramConv + (long)t * dim;
            Ops.Zero(o, dim);
            for (var j = 0; j < EngramTaps; j++)
                Ops.MultiplyAccumulate(o, s.Taps + j * dim, engramValueBuffer + (long)(EngramHistory + t - j * g.EngramDilation) * dim, dim);
        }
        Buffer.MemoryCopy(engramValueBuffer + (long)count * dim, history, EngramHistory * dim * 4, EngramHistory * dim * 4);

        var unitA = scratch;
        var unitB = scratch + dim;
        var inverseRootDim = 1f / MathF.Sqrt(dim);
        for (var t = 0; t < count; t++)
        {
            var x = input + (long)t * dim;
            Ops.RmsUnit(x, dim, unitA);
            Ops.RmsUnit(engramKey + (long)t * dim, dim, unitB);
            var alpha = Ops.Sigmoid(Ops.Dot(unitA, unitB, dim) * inverseRootDim);
            var v = engramConv + (long)t * dim;
            Ops.AddScaled(residual + (long)t * dim, v, alpha, dim);
        }
    }

    void RunAttention(int l, in LayerWeights w, int count, int position)
    {
        var normalized = scratch;
        for (var t = 0; t < count; t++)
        {
            var r = quantizedInput + (long)t * dim;
            Ops.ZeroCenteredRms(residual + (long)t * dim, w.NormIn, dim, r);
            Quant.FakeA8(r, dim);
        }
        Quant.Prepare(quantizedInput, count, dim, 2, dimAct);
        clock.Lap(Stage.AttentionPrep);

        var history = qkvHistory + (long)l * historyPerLayer;
        var qHistory = history;
        var kHistory = qHistory + (ConvTaps - 1) * qWidth;
        var vHistory = kHistory + (ConvTaps - 1) * kWidth;
        LoadHistory(HistoryRead(qHistory), qRaw, qWidth);
        LoadHistory(HistoryRead(kHistory), kRaw, kWidth);
        LoadHistory(HistoryRead(vHistory), vRaw, vWidth);
        var projections = new MatMulBatch(dimAct, count);
        projections.Add(w.Q, qRaw + (ConvTaps - 1) * qWidth, qWidth);
        projections.Add(w.K, kRaw + (ConvTaps - 1) * kWidth, kWidth);
        projections.Add(w.V, vRaw + (ConvTaps - 1) * vWidth, vWidth);
        projections.Add(w.Gate, gateOut, g.OutputWidth);
        clock.PoolRun();
        MatMulBatch.Run(pool, &projections);
        clock.Lap(Stage.AttentionQkvg);
        TapConvolve(qRaw, w.QTaps, qConv, count, qWidth);
        TapConvolve(kRaw, w.KTaps, kConv, count, kWidth);
        TapConvolve(vRaw, w.VTaps, vConv, count, vWidth);
        SaveHistory(qHistory, qRaw, count, qWidth);
        SaveHistory(kHistory, kRaw, count, kWidth);
        SaveHistory(vHistory, vRaw, count, vWidth);

        var half = g.QkDim / 2;
        var layerKeys = keyCache + (long)l * capacity * kWidth;
        var layerValues = valueCache + (long)l * capacity * vWidth;
        var layerKeyScales = keyScale + (long)l * capacity * g.KvHeads;
        var layerValueScales = valueScale + (long)l * capacity * g.KvHeads;
        for (var t = 0; t < count; t++)
        {
            var p = position + t;
            var q = qConv + (long)t * qWidth;
            var k = kConv + (long)t * kWidth;
            var v = vConv + (long)t * vWidth;
            for (var h = 0; h < g.Heads; h++) Ops.ZeroCenteredRms(q + h * g.QkDim, w.QNorm, g.QkDim, q + h * g.QkDim);
            for (var h = 0; h < g.KvHeads; h++) Ops.ZeroCenteredRms(k + h * g.QkDim, w.KNorm, g.QkDim, k + h * g.QkDim);
            Ops.Rope(q, g.Heads, g.QkDim, ropeCos + p * half, ropeSin + p * half);
            Ops.Rope(k, g.KvHeads, g.QkDim, ropeCos + p * half, ropeSin + p * half);
            for (var h = 0; h < g.Heads; h++)
                queryScale[t * g.Heads + h] = Quant.QuantizeInt8(q + h * g.QkDim, g.QkDim, queryInt8 + (long)t * qWidth + h * g.QkDim);
            for (var h = 0; h < g.KvHeads; h++)
            {
                layerKeyScales[(long)p * g.KvHeads + h] = Quant.QuantizeInt8(k + h * g.QkDim, g.QkDim, layerKeys + (long)p * kWidth + h * g.QkDim);
                layerValueScales[(long)p * g.KvHeads + h] = Quant.QuantizeInt8(v + h * g.VDim, g.VDim, layerValues + (long)p * vWidth + h * g.VDim);
            }
        }

        var attention = new AttentionBatch
        {
            Queries = queryInt8, QueryScales = queryScale, Keys = layerKeys, KeyScales = layerKeyScales,
            Values = layerValues, ValueScales = layerValueScales, Output = attnOut,
            Heads = g.Heads, KvHeads = g.KvHeads, QkDim = g.QkDim, VDim = g.VDim, Position = position,
            Window = g.IsGlobal(l) ? 0 : g.SlidingWindow, Capacity = capacity,
        };
        clock.Lap(Stage.AttentionPostProjection);
        clock.PoolRun();
        AttentionBatch.Run(pool, &attention, count);
        clock.Lap(Stage.AttentionCore);

        for (var t = 0; t < count; t++)
        {
            var o = attnOut + (long)t * g.OutputWidth;
            Ops.MultiplyBySigmoid(o, gateOut + (long)t * g.OutputWidth, g.OutputWidth);
            Quant.FakeA8(o, g.OutputWidth);
        }
        Quant.Prepare(attnOut, count, g.OutputWidth, 2, dimAct);
        clock.Lap(Stage.AttentionGate);
        var output = new MatMulBatch(dimAct, count);
        output.Add(w.Out, attnProj, dim);
        clock.PoolRun();
        MatMulBatch.Run(pool, &output);
        clock.Lap(Stage.AttentionOut);
        for (var t = 0; t < count; t++)
        {
            Ops.ZeroCenteredRms(attnProj + (long)t * dim, w.PostNorm, dim, normalized);
            Ops.AddScaled(residual + (long)t * dim, normalized, w.AttnGate, dim);
        }
        clock.Lap(Stage.AttentionPostNorm);
    }

    void RunCross(int l, int count, int position)
    {
        var c = model.Audio->Cross + l;
        var heads = g.Heads;
        var frames = cross.Frames;
        for (var t = 0; t < count; t++)
        {
            var r = quantizedInput + (long)t * dim;
            Ops.ZeroCenteredRms(residual + (long)t * dim, c->Norm, dim, r);
            Quant.FakeA8(r, dim);
        }
        Quant.Prepare(quantizedInput, count, dim, 2, dimAct);
        var projections = new MatMulBatch(dimAct, count);
        projections.Add(c->Q, qConv, heads * g.QkDim);
        projections.Add(c->Gate, gateOut, g.OutputWidth);
        clock.PoolRun();
        MatMulBatch.Run(pool, &projections);
        for (var t = 0; t < count; t++)
            for (var h = 0; h < heads; h++)
            {
                var q = qConv + (long)t * heads * g.QkDim + h * g.QkDim;
                Ops.ZeroCenteredRms(q, c->QNorm, g.QkDim, q);
                queryScale[t * heads + h] = Quant.QuantizeInt8(q, g.QkDim, queryInt8 + (long)t * heads * g.QkDim + h * g.QkDim);
            }
        var attention = new AttentionBatch
        {
            Queries = queryInt8, QueryScales = queryScale,
            Keys = cross.Keys + (long)l * frames * heads * g.QkDim, KeyScales = cross.KeyScales + (long)l * frames * heads,
            Values = cross.Values + (long)l * frames * heads * g.VDim, ValueScales = cross.ValueScales + (long)l * frames * heads,
            Output = attnOut, Heads = heads, KvHeads = heads, QkDim = g.QkDim, VDim = g.VDim, Position = 0, Window = 0,
            Capacity = Math.Max(frames, capacity), Frames = frames,
        };
        var capturing = Alignment != null && l >= AlignmentFromLayer;
        if (capturing)
        {
            if (captureBuffer == null) captureBuffer = (float*)Mem.Alloc((nuint)(chunk * heads * frames) * 4);
            attention.Probabilities = captureBuffer;
        }
        clock.PoolRun();
        AttentionBatch.Run(pool, &attention, count);
        if (capturing)
        {
            var share = 1f / (heads * (g.Layers - AlignmentFromLayer));
            for (var t = 0; t < count; t++)
                for (var h = 0; h < heads; h++)
                    Ops.AddScaled(Alignment + (long)(position + t) * frames, captureBuffer + ((long)t * heads + h) * frames, share, frames);
        }
        for (var t = 0; t < count; t++)
        {
            var o = attnOut + (long)t * g.OutputWidth;
            Ops.MultiplyBySigmoid(o, gateOut + (long)t * g.OutputWidth, g.OutputWidth);
            Quant.FakeA8(o, g.OutputWidth);
        }
        Quant.Prepare(attnOut, count, g.OutputWidth, 2, dimAct);
        var output = new MatMulBatch(dimAct, count);
        output.Add(c->Out, attnProj, dim);
        clock.PoolRun();
        MatMulBatch.Run(pool, &output);
        for (var t = 0; t < count; t++)
        {
            Ops.ZeroCenteredRms(attnProj + (long)t * dim, c->PostNorm, dim, scratch);
            Ops.AddScaled(residual + (long)t * dim, scratch, c->Mix, dim);
        }
        clock.Lap(Stage.Cross);
    }

    void RunMlp(int layer, int count)
    {
        var batch = new MlpBatch
        {
            Weights = &(model.Blocks + layer)->Mlp,
            PreNorm = model.Blocks[layer].PreHada, PermA = model.Hyper.PermA, PermB = model.Hyper.PermB,
            Residual = residual, Dim = dim, Rows = g.HadaRows, Cols = g.HadaCols, Scale = 1f,
        };
        clock.PoolRun();
        MlpBatch.Run(pool, &batch, count);
    }

    static void LoadHistory(float* history, float* raw, int width) =>
        Buffer.MemoryCopy(history, raw, (ConvTaps - 1) * width * 4, (ConvTaps - 1) * width * 4);

    static void SaveHistory(float* history, float* raw, int count, int width) =>
        Buffer.MemoryCopy(raw + (long)count * width, history, (ConvTaps - 1) * width * 4, (ConvTaps - 1) * width * 4);

    static void TapConvolve(float* raw, float* taps, float* output, int count, int width)
    {
        for (var t = 0; t < count; t++)
        {
            var current = raw + (long)(t + ConvTaps - 1) * width;
            var o = output + (long)t * width;
            Ops.Taps3(taps, width, current, o);
        }
    }

    void ComputeLogits(int last)
    {
        PrepareHead(last);
        var head = new MatMulBatch(headAct, 1);
        head.Add(model.Embedding, logits, model.Embedding.Rows);
        clock.PoolRun();
        MatMulBatch.Run(pool, &head);
        clock.Lap(Stage.Logits);
    }

    // Builds the quantized head input for one forwarded token. Shared by the
    // full head and LogitsForRows so both see the exact same normalization.
    void PrepareHead(int last)
    {
        var s = stream + (long)last * streamWidth;
        var mean = scratch;
        for (var c = 0; c < dim; c++) mean[c] = 0.25f * (s[c] + s[dim + c] + s[2 * dim + c] + s[3 * dim + c]);
        var normalized = scratch + dim;
        Ops.ZeroCenteredRms(mean, model.FinalNorm, dim, normalized);
        Quant.FakeA8(normalized, dim);
        Quant.Prepare(normalized, 1, dim, 4, headAct);
    }

    // Logits for an explicit row subset of the embedding head, read from the
    // last forwarded token. Ascending `rows` (mask-extraction order) lets
    // contiguous ids merge into runs; each run is extended to a multiple of 4
    // to match the full head's AVX block alignment, so gathered values equal
    // the full head's bit-for-bit. `destination[0..count)` aligns with `rows`.
    // No pool work happens here, hence no clock.PoolRun; the logits stage is
    // still attributed through Lap.
    internal void LogitsForRows(int* rows, int count, float* destination) =>
        LogitsForBatchRow(lastForwardCount - 1, rows, count, destination);

    // W26 multi-sequence batched decode. One forward over rows from N slots
    // shares the weight reads (projections, hyper phi, MLP, head) while
    // attention, convolution history, engram state, pooling and the KV cache
    // stay per sequence. Rows from one slot are contiguous; row positions
    // derive from slot lengths, so callers pass only tokens and slot ids.
    unsafe struct SlotView
    {
        public float* State;
        public float* Snapshot;
        public sbyte* Keys;
        public sbyte* Values;
        public float* KeyScales;
        public float* ValueScales;
        public int Length;
        public int Marked;
        public bool HistPending;
        public bool PoolPending;
    }

    float* EngramOf(float* @base) => @base + g.Layers * historyPerLayer;

    int* PrevOf(float* @base) => (int*)(EngramOf(@base) + g.SiteCount * EngramHistory * dim);

    float* PoolMaxOf(float* @base) => (float*)(PrevOf(@base) + 2);

    float* PoolSumOf(float* @base) => PoolMaxOf(@base) + poolRows;

    float* PoolAccOf(float* @base) => PoolSumOf(@base) + poolRows;

    float* QkvOf(float* @base, int layer) => @base + (long)layer * historyPerLayer;

    void ResetView(ref SlotView s)
    {
        Mem.Clear(s.State, stateBytes);
        var max = PoolMaxOf(s.State);
        for (var i = 0; i < poolRows; i++) max[i] = -1e30f;
        s.Length = 0;
        s.Marked = 0;
        s.HistPending = false;
        s.PoolPending = false;
    }

    void MarkView(ref SlotView s)
    {
        if (!s.HistPending) Buffer.MemoryCopy(s.State, s.Snapshot, historyBytes, historyBytes);
        var live = PrevOf(s.State);
        var snap = PrevOf(s.Snapshot);
        snap[0] = live[0];
        snap[1] = live[1];
        if (!s.PoolPending) Buffer.MemoryCopy(PoolMaxOf(s.State), PoolMaxOf(s.Snapshot), poolBytes, poolBytes);
        s.Marked = s.Length;
    }

    void RewindView(ref SlotView s)
    {
        var live = PrevOf(s.State);
        var snap = PrevOf(s.Snapshot);
        live[0] = snap[0];
        live[1] = snap[1];
        s.HistPending = true;
        s.PoolPending = poolRows > 0;
        s.Length = s.Marked;
    }

    internal int Slots => slotCount;

    internal int SlotCapacity => capacity;

    internal int SlotLength(int slot) => slotViews![slot].Length;

    internal int Vocab => model.Embedding.Rows;

    internal float* BatchLogitsRow(int compactRow) => batchLogits + (long)compactRow * model.Embedding.Rows;

    internal void ResetSlot(int slot)
    {
        ref var view = ref slotViews![slot];
        ResetView(ref view);
    }

    internal void MarkSlot(int slot)
    {
        ref var view = ref slotViews![slot];
        MarkView(ref view);
    }

    internal void RewindSlot(int slot)
    {
        ref var view = ref slotViews![slot];
        RewindView(ref view);
    }

    // Single-sequence-equivalent chunked prefill into one slot. Decode rows
    // fed through ForwardBatch chain histories the same way, so prefill and
    // batched decode compose bit-identically with Feed.
    internal void FeedSlot(int slot, ReadOnlySpan<int> tokens, bool computeLogits = true)
    {
        ref var view = ref slotViews![slot];
        if (view.Length + tokens.Length > capacity) throw new InvalidOperationException("context capacity exceeded");
        var rowSlots = stackalloc int[chunk];
        var head = stackalloc int[1];
        fixed (int* all = tokens)
            for (var start = 0; start < tokens.Length; start += chunk)
            {
                var count = Math.Min(chunk, tokens.Length - start);
                var last = start + count == tokens.Length;
                for (var i = 0; i < count; i++) rowSlots[i] = slot;
                var heads = 0;
                if (last && computeLogits) { head[0] = count - 1; heads = 1; }
                ForwardBatch(all + start, rowSlots, count, head, heads);
            }
    }

    // One decode step over rows from several slots. Rows from one slot are
    // contiguous; each run starts at its slot's length. `headRows` holds row
    // indices whose full-vocabulary head is needed; their logits land
    // compactly in batchLogits. Sparse and logit-free rows are handled by
    // the caller after the pass.
    internal void ForwardBatch(int* tokens, int* rowSlots, int count, int* headRows, int headCount)
    {
        if (slotViews == null) throw new InvalidOperationException("batched decode needs an engine constructed with slots");
        if (count < 1 || count > chunk) throw new ArgumentOutOfRangeException(nameof(count));
        if (headCount < 0 || headCount > count) throw new ArgumentOutOfRangeException(nameof(headCount));
        if (cross.Frames > 0) throw new NotSupportedException("batched decode does not support cross attention");
        clock.Begin();
        for (var t = 0; t < count; t++)
        {
            Quant.GatherRow(model.Embedding, tokens[t], scratch);
            var row = stream + (long)t * streamWidth;
            Ops.AddScaled(scratch, scratch, embeddingScale - 1f, dim);
            for (var lane = 0; lane < Lanes; lane++) Buffer.MemoryCopy(scratch, row + lane * dim, dim * 4, dim * 4);
        }
        EngramHashRowsBatch(tokens, rowSlots, count, engramRows);
        clock.Lap(Stage.Embed);
        Observer?.Invoke(-1, new ReadOnlySpan<float>(stream, count * streamWidth));
        if (pooling) PoolTokensBatch(0, rowSlots, count, true);
        for (var layer = 0; layer < g.Layers; layer++)
        {
            RunLayerBatch(layer, rowSlots, count);
            Observer?.Invoke(layer, new ReadOnlySpan<float>(stream, count * streamWidth));
            if (pooling) PoolTokensBatch(layer + 1, rowSlots, count, false);
        }
        for (var rs = 0; rs < count;)
        {
            var slot = rowSlots[rs];
            var rk = rs;
            while (rk + 1 < count && rowSlots[rk + 1] == slot) rk++;
            ref var view = ref slotViews[slot];
            view.HistPending = false;
            view.PoolPending = false;
            var prev = PrevOf(view.State);
            if (rk > rs) { prev[0] = tokens[rk - 1]; prev[1] = tokens[rk]; }
            else { prev[0] = prev[1]; prev[1] = tokens[rk]; }
            view.Length += rk - rs + 1;
            rs = rk + 1;
        }
        for (var i = 0; i < headCount; i++)
        {
            PrepareHead(headRows[i]);
            var head = new MatMulBatch(headAct, 1);
            head.Add(model.Embedding, batchLogits + (long)i * model.Embedding.Rows, model.Embedding.Rows);
            clock.PoolRun();
            MatMulBatch.Run(pool, &head);
        }
        clock.Lap(Stage.Logits);
    }

    void RunLayerBatch(int l, int* rowSlots, int count)
    {
        ref readonly var w = ref model.Blocks[l];
        ref readonly var hyper = ref model.Hyper;
        var lane = l % Lanes;

        for (var t = 0; t < count; t++)
        {
            var normalized = normalizedStream + (long)t * streamWidth;
            Ops.RmsUnit(stream + (long)t * streamWidth, streamWidth, normalized);
            Quant.FakeA8(normalized, streamWidth);
        }
        Quant.Prepare(normalizedStream, count, streamWidth, 4, streamAct);
        clock.Lap(Stage.HyperPrepare);
        Quant.MatMul(hyper.PhiPre.Slice(l * Lanes, Lanes), streamAct, count, preOut, Lanes);
        Quant.MatMul(hyper.PhiPost.Slice(l * Lanes, Lanes), streamAct, count, postOut, Lanes);
        Quant.MatMul(hyper.PhiRes.Slice(l * Lanes * Lanes, Lanes * Lanes), streamAct, count, resOut, Lanes * Lanes);
        clock.Lap(Stage.HyperPhi);

        var mixLogits = stackalloc float[Lanes * Lanes];
        var hpre = stackalloc float[Lanes];
        for (var t = 0; t < count; t++)
        {
            var s = stream + (long)t * streamWidth;
            var u = input + (long)t * dim;
            for (var i = 0; i < Lanes; i++)
            {
                hpre[i] = Ops.Sigmoid(hyper.APre[l] * preOut[t * Lanes + i] + hyper.BPre[l * Lanes + i] + (i == lane ? 4f : -4f));
                postGate[t * Lanes + i] = 2f * Ops.Sigmoid(hyper.APost[l] * postOut[t * Lanes + i] + hyper.BPost[l * Lanes + i] + (i == lane ? 0f : -4f));
            }
            Ops.Mix4(s, dim, hpre[0], hpre[1], hpre[2], hpre[3], u);
            for (var i = 0; i < Lanes * Lanes; i++) mixLogits[i] = hyper.ARes[l] * resOut[t * Lanes * Lanes + i] + hyper.BRes[l * Lanes * Lanes + i];
            Sinkhorn.Normalize(mixLogits, residualMix + t * Lanes * Lanes);
            Buffer.MemoryCopy(u, residual + (long)t * dim, dim * 4, dim * 4);
        }
        clock.Lap(Stage.HyperMix);

        var site = g.SiteOf(l);
        if (site >= 0) ApplyEngramBatch(site, rowSlots, count);
        clock.Lap(Stage.Engram);
        RunAttentionBatch(l, w, rowSlots, count);
        RunMlp(l, count);
        clock.Lap(Stage.Mlp);

        for (var t = 0; t < count; t++)
        {
            var s = stream + (long)t * streamWidth;
            var n = nextStream + (long)t * streamWidth;
            var y = scratch;
            var x = residual + (long)t * dim;
            var u = input + (long)t * dim;
            Ops.Subtract(x, u, dim, y);
            var mix = residualMix + t * Lanes * Lanes;
            for (var i = 0; i < Lanes; i++)
                Ops.Combine4(s, dim, mix[i * 4], mix[i * 4 + 1], mix[i * 4 + 2], mix[i * 4 + 3], y, postGate[t * Lanes + i], n + i * dim);
        }
        clock.Lap(Stage.Combine);
        var swap = stream;
        stream = nextStream;
        nextStream = swap;
    }
    void ApplyEngramBatch(int site, int* rowSlots, int count)
    {
        var s = model.Sites[site];
        var sub = g.EngramSub;
        for (var t = 0; t < count; t++)
        {
            var e = engramInput + (long)t * dim;
            for (var table = 0; table < g.EngramTables; table++)
            {
                var row = engramRows[t * g.EngramTables + table];
                if (row < 0) Mem.Clear(e + table * sub, (nuint)sub * 4);
                else Quant.GatherRow(s.Tables, row, e + table * sub);
            }
            Quant.FakeA8(e, dim);
        }
        Quant.Prepare(engramInput, count, dim, 2, dimAct);
        var stage = batchEngramStage;
        var stride = EngramHistory + 1;
        for (var rs = 0; rs < count;)
        {
            var slot = rowSlots[rs];
            var rk = rs;
            while (rk + 1 < count && rowSlots[rk + 1] == slot) rk++;
            ref var view = ref slotViews![slot];
            var history = view.HistPending ? EngramOf(view.Snapshot) : EngramOf(view.State);
            Buffer.MemoryCopy(history, stage + (long)rs * stride * dim, EngramHistory * dim * 4, EngramHistory * dim * 4);
            rs = rk + 1;
        }
        var batch = new MatMulBatch(dimAct, count);
        batch.Add(s.Key, engramKey, dim);
        batch.Add(s.Value, stage + EngramHistory * dim, stride * dim);
        clock.PoolRun();
        MatMulBatch.Run(pool, &batch);
        for (var rs = 0; rs < count;)
        {
            var slot = rowSlots[rs];
            var rk = rs;
            while (rk + 1 < count && rowSlots[rk + 1] == slot) rk++;
            for (var t = rs; t <= rk; t++)
            {
                var cur = stage + (long)t * stride * dim;
                if (t > rs) Buffer.MemoryCopy(cur - (long)stride * dim + dim, cur, EngramHistory * dim * 4, EngramHistory * dim * 4);
                var o = engramConv + (long)t * dim;
                Ops.Zero(o, dim);
                for (var j = 0; j < EngramTaps; j++)
                    Ops.MultiplyAccumulate(o, s.Taps + j * dim, cur + (long)(EngramHistory - j * g.EngramDilation) * dim, dim);
            }
            ref var view = ref slotViews![slot];
            Buffer.MemoryCopy(stage + ((long)rk * stride + 1) * dim, EngramOf(view.State), EngramHistory * dim * 4, EngramHistory * dim * 4);
            rs = rk + 1;
        }

        var unitA = scratch;
        var unitB = scratch + dim;
        var inverseRootDim = 1f / MathF.Sqrt(dim);
        for (var t = 0; t < count; t++)
        {
            var x = input + (long)t * dim;
            Ops.RmsUnit(x, dim, unitA);
            Ops.RmsUnit(engramKey + (long)t * dim, dim, unitB);
            var alpha = Ops.Sigmoid(Ops.Dot(unitA, unitB, dim) * inverseRootDim);
            var v = engramConv + (long)t * dim;
            Ops.AddScaled(residual + (long)t * dim, v, alpha, dim);
        }
    }

    static void ChainTriple(float* triple, int t, int k, float* slotHist, int width)
    {
        // triple[t] is [history, history, current]; chained rows reuse the
        // previous rows' projected values, the run head the slot tail.
        var cur = triple + (long)t * 3 * width;
        var second = k >= 2 ? triple + (long)(t - 2) * 3 * width + 2 * width : slotHist + width;
        var first = triple + (long)(t - 1) * 3 * width + 2 * width;
        Buffer.MemoryCopy(second, cur, width * 4, width * 4);
        Buffer.MemoryCopy(first, cur + width, width * 4, width * 4);
    }

    void RunAttentionBatch(int l, in LayerWeights w, int* rowSlots, int count)
    {
        var normalized = scratch;
        for (var t = 0; t < count; t++)
        {
            var r = quantizedInput + (long)t * dim;
            Ops.ZeroCenteredRms(residual + (long)t * dim, w.NormIn, dim, r);
            Quant.FakeA8(r, dim);
        }
        Quant.Prepare(quantizedInput, count, dim, 2, dimAct);
        clock.Lap(Stage.AttentionPrep);

        for (var rs = 0; rs < count;)
        {
            var slot = rowSlots[rs];
            var rk = rs;
            while (rk + 1 < count && rowSlots[rk + 1] == slot) rk++;
            ref var view = ref slotViews![slot];
            var read = view.HistPending ? QkvOf(view.Snapshot, l) : QkvOf(view.State, l);
            Buffer.MemoryCopy(read, batchQ + (long)rs * 3 * qWidth, (ConvTaps - 1) * qWidth * 4, (ConvTaps - 1) * qWidth * 4);
            Buffer.MemoryCopy(read + (ConvTaps - 1) * qWidth, batchK + (long)rs * 3 * kWidth, (ConvTaps - 1) * kWidth * 4, (ConvTaps - 1) * kWidth * 4);
            Buffer.MemoryCopy(read + (ConvTaps - 1) * (qWidth + kWidth), batchV + (long)rs * 3 * vWidth, (ConvTaps - 1) * vWidth * 4, (ConvTaps - 1) * vWidth * 4);
            rs = rk + 1;
        }
        var projections = new MatMulBatch(dimAct, count);
        projections.Add(w.Q, batchQ + 2 * qWidth, 3 * qWidth);
        projections.Add(w.K, batchK + 2 * kWidth, 3 * kWidth);
        projections.Add(w.V, batchV + 2 * vWidth, 3 * vWidth);
        projections.Add(w.Gate, gateOut, g.OutputWidth);
        clock.PoolRun();
        MatMulBatch.Run(pool, &projections);
        clock.Lap(Stage.AttentionQkvg);

        var half = g.QkDim / 2;
        for (var rs = 0; rs < count;)
        {
            var slot = rowSlots[rs];
            var rk = rs;
            while (rk + 1 < count && rowSlots[rk + 1] == slot) rk++;
            ref var view = ref slotViews![slot];
            var live = QkvOf(view.State, l);
            var liveK = live + (ConvTaps - 1) * qWidth;
            var liveV = liveK + (ConvTaps - 1) * kWidth;
            for (var t = rs; t <= rk; t++)
            {
                if (t > rs)
                {
                    ChainTriple(batchQ, t, t - rs, live, qWidth);
                    ChainTriple(batchK, t, t - rs, liveK, kWidth);
                    ChainTriple(batchV, t, t - rs, liveV, vWidth);
                }
                Ops.Taps3(w.QTaps, qWidth, batchQ + (long)t * 3 * qWidth, qConv + (long)t * qWidth);
                Ops.Taps3(w.KTaps, kWidth, batchK + (long)t * 3 * kWidth, kConv + (long)t * kWidth);
                Ops.Taps3(w.VTaps, vWidth, batchV + (long)t * 3 * vWidth, vConv + (long)t * vWidth);
            }
            if (rk > rs)
            {
                Buffer.MemoryCopy(batchQ + (long)(rk - 1) * 3 * qWidth + 2 * qWidth, live, qWidth * 4, qWidth * 4);
                Buffer.MemoryCopy(batchQ + (long)rk * 3 * qWidth + 2 * qWidth, live + qWidth, qWidth * 4, qWidth * 4);
                Buffer.MemoryCopy(batchK + (long)(rk - 1) * 3 * kWidth + 2 * kWidth, liveK, kWidth * 4, kWidth * 4);
                Buffer.MemoryCopy(batchK + (long)rk * 3 * kWidth + 2 * kWidth, liveK + kWidth, kWidth * 4, kWidth * 4);
                Buffer.MemoryCopy(batchV + (long)(rk - 1) * 3 * vWidth + 2 * vWidth, liveV, vWidth * 4, vWidth * 4);
                Buffer.MemoryCopy(batchV + (long)rk * 3 * vWidth + 2 * vWidth, liveV + vWidth, vWidth * 4, vWidth * 4);
            }
            else
            {
                Buffer.MemoryCopy(batchQ + (long)rk * 3 * qWidth + qWidth, live, 2 * qWidth * 4, 2 * qWidth * 4);
                Buffer.MemoryCopy(batchK + (long)rk * 3 * kWidth + kWidth, liveK, 2 * kWidth * 4, 2 * kWidth * 4);
                Buffer.MemoryCopy(batchV + (long)rk * 3 * vWidth + vWidth, liveV, 2 * vWidth * 4, 2 * vWidth * 4);
            }
            var enter = view.Length;
            var layerKeys = view.Keys + (long)l * capacity * kWidth;
            var layerValues = view.Values + (long)l * capacity * vWidth;
            var layerKeyScales = view.KeyScales + (long)l * capacity * g.KvHeads;
            var layerValueScales = view.ValueScales + (long)l * capacity * g.KvHeads;
            for (var t = rs; t <= rk; t++)
            {
                var p = enter + t - rs;
                var q = qConv + (long)t * qWidth;
                var k = kConv + (long)t * kWidth;
                var v = vConv + (long)t * vWidth;
                for (var h = 0; h < g.Heads; h++) Ops.ZeroCenteredRms(q + h * g.QkDim, w.QNorm, g.QkDim, q + h * g.QkDim);
                for (var h = 0; h < g.KvHeads; h++) Ops.ZeroCenteredRms(k + h * g.QkDim, w.KNorm, g.QkDim, k + h * g.QkDim);
                Ops.Rope(q, g.Heads, g.QkDim, ropeCos + p * half, ropeSin + p * half);
                Ops.Rope(k, g.KvHeads, g.QkDim, ropeCos + p * half, ropeSin + p * half);
                for (var h = 0; h < g.Heads; h++)
                    queryScale[t * g.Heads + h] = Quant.QuantizeInt8(q + h * g.QkDim, g.QkDim, queryInt8 + (long)t * qWidth + h * g.QkDim);
                for (var h = 0; h < g.KvHeads; h++)
                {
                    layerKeyScales[(long)p * g.KvHeads + h] = Quant.QuantizeInt8(k + h * g.QkDim, g.QkDim, layerKeys + (long)p * kWidth + h * g.QkDim);
                    layerValueScales[(long)p * g.KvHeads + h] = Quant.QuantizeInt8(v + h * g.VDim, g.VDim, layerValues + (long)p * vWidth + h * g.VDim);
                }
            }
            // One attention batch per slot run: the core reads only that
            // sequence's KV, so sharing the pass would save nothing, while a
            // shared span would change tiling and break bit-identity with the
            // single-sequence path.
            var attention = new AttentionBatch
            {
                Queries = queryInt8 + (long)rs * qWidth, QueryScales = queryScale + rs * g.Heads,
                Keys = layerKeys, KeyScales = layerKeyScales,
                Values = layerValues, ValueScales = layerValueScales, Output = attnOut + (long)rs * g.OutputWidth,
                Heads = g.Heads, KvHeads = g.KvHeads, QkDim = g.QkDim, VDim = g.VDim, Position = enter,
                Window = g.IsGlobal(l) ? 0 : g.SlidingWindow, Capacity = capacity,
            };
            clock.Lap(Stage.AttentionPostProjection);
            clock.PoolRun();
            AttentionBatch.Run(pool, &attention, rk - rs + 1);
            clock.Lap(Stage.AttentionCore);
            rs = rk + 1;
        }

        for (var t = 0; t < count; t++)
        {
            var o = attnOut + (long)t * g.OutputWidth;
            Ops.MultiplyBySigmoid(o, gateOut + (long)t * g.OutputWidth, g.OutputWidth);
            Quant.FakeA8(o, g.OutputWidth);
        }
        Quant.Prepare(attnOut, count, g.OutputWidth, 2, dimAct);
        clock.Lap(Stage.AttentionGate);
        var output = new MatMulBatch(dimAct, count);
        output.Add(w.Out, attnProj, dim);
        clock.PoolRun();
        MatMulBatch.Run(pool, &output);
        clock.Lap(Stage.AttentionOut);
        for (var t = 0; t < count; t++)
        {
            Ops.ZeroCenteredRms(attnProj + (long)t * dim, w.PostNorm, dim, normalized);
            Ops.AddScaled(residual + (long)t * dim, normalized, w.AttnGate, dim);
        }
        clock.Lap(Stage.AttentionPostNorm);
    }

    void PoolTokensBatch(int row, int* rowSlots, int count, bool initial)
    {
        var head = model.Confidence;
        var probes = head->Probes;
        var inverseRoot = 1f / MathF.Sqrt(dim);
        for (var t = 0; t < count; t++)
        {
            ref var view = ref slotViews![rowSlots[t]];
            if ((t == 0 || rowSlots[t] != rowSlots[t - 1]) && view.PoolPending)
                Buffer.MemoryCopy(PoolMaxOf(view.Snapshot), PoolMaxOf(view.State), poolBytes, poolBytes);
            var s = stream + (long)t * streamWidth;
            if (initial) Buffer.MemoryCopy(s, cell, dim * 4, dim * 4);
            else Ops.Mix4(s, dim, 0.25f, 0.25f, 0.25f, 0.25f, cell);
            var liveMax = PoolMaxOf(view.State);
            var liveSum = PoolSumOf(view.State);
            var liveAcc = PoolAccOf(view.State);
            for (var p = 0; p < probes; p++)
            {
                var index = row * probes + p;
                var score = Ops.Dot(cell, head->ProbeVectors + (long)index * dim, dim) * inverseRoot;
                var acc = liveAcc + (long)index * dim;
                if (score > liveMax[index])
                {
                    var rescale = MathF.Exp(liveMax[index] - score);
                    liveSum[index] = liveSum[index] * rescale + 1f;
                    Ops.ScaleAdd(acc, rescale, cell, 1f, dim);
                    liveMax[index] = score;
                }
                else
                {
                    var weight = MathF.Exp(score - liveMax[index]);
                    liveSum[index] += weight;
                    Ops.AddScaled(acc, cell, weight, dim);
                }
            }
        }
    }

    void EngramHashRowsBatch(int* tokens, int* rowSlots, int count, int* rows)
    {
        var heads = g.EngramHeads;
        var runStart = 0;
        for (var t = 0; t < count; t++)
        {
            if (t == 0 || rowSlots[t] != rowSlots[t - 1]) runStart = t;
            ref var view = ref slotViews![rowSlots[t]];
            var prev = PrevOf(view.State);
            var position = view.Length + t - runStart;
            for (var o = 0; o < g.OrderCount; o++)
                for (var h = 0; h < heads; h++)
                {
                    var table = o * heads + h;
                    var order = g.Orders[o];
                    if (position < order - 1) { rows[t * g.EngramTables + table] = -1; continue; }
                    var acc = 0x9E3779B9u * (uint)(table + 1);
                    for (var j = 0; j < order; j++)
                    {
                        int token;
                        var back = t - runStart - j;
                        if (back >= 0) token = tokens[runStart + back];
                        else token = prev[back + 2];
                        acc = (acc ^ (uint)token) * 0x01000193u;
                    }
                    acc ^= acc >> 15;
                    rows[t * g.EngramTables + table] = table * g.EngramSlots + (int)(acc % (uint)g.EngramSlots);
                }
        }
    }

    // Full-head logits are computed per row into the single head activation,
    // so the sparse gather sees the exact normalization the dense head does.
    internal void LogitsForBatchRow(int row, int* rows, int count, float* destination)
    {
        PrepareHead(row);
        var vocab = model.Embedding.Rows;
        var i = 0;
        while (i < count)
        {
            var start = rows[i];
            var end = start + 1;
            i++;
            while (i < count && rows[i] == end) { end++; i++; }
            var alignedStart = start & ~3;
            var alignedEnd = (end + 3) & ~3;
            if (alignedEnd > vocab) alignedEnd = vocab;
            Quant.MatMulRows(model.Embedding, headAct, 1, logits, vocab, alignedStart, alignedEnd);
        }
        for (var k = 0; k < count; k++) destination[k] = logits[rows[k]];
    }

    internal float ConfidenceLogitForSlot(int slot, float* pooledOut, float* scratchRows)
    {
        ref var view = ref slotViews![slot];
        if (view.PoolPending)
        {
            Buffer.MemoryCopy(PoolMaxOf(view.Snapshot), PoolMaxOf(view.State), poolBytes, poolBytes);
            view.PoolPending = false;
        }
        var head = model.Confidence;
        var probes = head->Probes;
        var rows = (g.Layers + 1) * probes;
        var vectors = scratchRows;
        var weights = scratchRows + (long)rows * dim;
        var inverseRoot = 1f / MathF.Sqrt(dim);
        var liveMax = PoolMaxOf(view.State);
        var liveSum = PoolSumOf(view.State);
        var liveAcc = PoolAccOf(view.State);
        for (var i = 0; i < rows; i++)
        {
            var v = vectors + (long)i * dim;
            Ops.ScaleAdd(v, 0f, liveAcc + (long)i * dim, 1f / liveSum[i], dim);
            Ops.RmsUnit(v, dim, v);
            Ops.ScaleAdd(v, head->Gain[i], v, 0f, dim);
        }
        var logit = head->Bias[0];
        for (var q = 0; q < head->Queries; q++)
        {
            for (var i = 0; i < rows; i++)
                weights[i] = Ops.Dot(vectors + (long)i * dim, head->QueryVectors + (long)q * dim, dim) * inverseRoot + head->RowBias[q * rows + i];
            Ops.Softmax(weights, rows);
            var pooled = pooledOut + q * dim;
            Mem.Clear(pooled, (nuint)dim * 4);
            for (var i = 0; i < rows; i++) Ops.AddScaled(pooled, vectors + (long)i * dim, weights[i], dim);
        }
        logit += Ops.Dot(pooledOut, head->Projection, head->Queries * dim);
        return logit;
    }

    internal Pool SharedPool => pool;

    public void Pause() => pool.Pause();

    public void Resume() => pool.Resume();

    public void Dispose()
    {
        if (ownsPool) pool.Dispose();
        arena.Free();
    }
}

public sealed unsafe class Engine : EngineCore<NoClock>
{
    public Engine(Model model, int contextLength = 1024, int chunkSize = 64, int threads = 1)
        : base(model, contextLength, chunkSize, ThreadPolicy.Default.WithThreads(threads), null, default)
    {
    }

    public Engine(Model model, ThreadPolicy policy, int contextLength = 1024, int chunkSize = 64)
        : base(model, contextLength, chunkSize, policy, null, default)
    {
    }

    internal Engine(Model model, int contextLength, int chunkSize, ThreadPolicy policy, Pool? shared)
        : base(model, contextLength, chunkSize, policy, shared, default)
    {
    }
}
