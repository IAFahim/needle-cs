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
    TeamBarrier* barrier;
    float* attnPartials;
    int* attnArrivals;

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

    public int Length { get; private set; }

    internal long StateBytes => (long)stateBytes;

    public LayerObserver? Observer { get; set; }

    public int Greedy() => Ops.ArgMax(logits, model.Embedding.Rows);

    public ReadOnlySpan<float> Logits => new(logits, model.Embedding.Rows);

    internal EngineCore(Model model, int contextLength, int chunkSize, ThreadPolicy policy, Pool? shared, TClock clock)
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

        var kvBytes = (nuint)capacity * (nuint)g.Layers * (nuint)(kWidth + vWidth + 2 * g.KvHeads * sizeof(float));
        arena = new Arena(kvBytes + (nuint)chunk * 160_000 + (nuint)capacity * 512 + (16u << 20));

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
        barrier = arena.Take<TeamBarrier>(1);
        attnPartials = arena.Take<float>((nuint)chunk * (nuint)g.KvHeads * 8 * Attention.PartialFloats);
        attnArrivals = arena.Take<int>((nuint)chunk * (nuint)g.KvHeads);
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
        if (cross.Frames != 0) { RunLayerLegacy(l, count, position); return; }
        RunLayerTeam(l, count, position);
    }

    unsafe struct LayerJob
    {
        public TeamBarrier* Barrier;
        public TClock Clock;
        public int Members, Layer, Lane, Count, Position, Capacity, HasEngram;
        public Geometry Geometry;
        public LayerWeights Weights;
        public HyperConnections Hyper;
        public EngramSite Engram;
        public Activations StreamAct, DimAct;
        public float* Stream;
        public float* Next;
        public float* Normed;
        public float* Input;
        public float* Residual;
        public float* Quantized;
        public float* QRaw;
        public float* KRaw;
        public float* VRaw;
        public float* QConv;
        public float* KConv;
        public float* VConv;
        public float* GateOut;
        public float* AttnOut;
        public float* AttnProj;
        public float* EngramInput;
        public float* EngramKey;
        public float* EngramConv;
        public float* EngramValueBuffer;
        public float* QueryScale;
        public float* KeyScale;
        public float* ValueScale;
        public float* RopeCos;
        public float* RopeSin;
        public float* ResidualMix;
        public float* PostGate;
        public float* PreOut;
        public float* PostOut;
        public float* ResOut;
        public float* AttnPartials;
        public float* QkvBase;
        public float* EngramBase;
        public sbyte* QueryInt8;
        public sbyte* KeyCache;
        public sbyte* ValueCache;
        public int* EngramRows;
        public int* AttnArrivals;
        public MlpWeights* Mlp;
    }

    static void Shard(int tasks, int member, int members, out int first, out int count)
    {
        var per = (tasks + members - 1) / members;
        first = Math.Min(member * per, tasks);
        count = Math.Min(first + per, tasks) - first;
    }

    void RunLayerTeam(int l, int count, int position)
    {
        var site = g.SiteOf(l);
        var job = new LayerJob
        {
            Barrier = barrier,
            Clock = clock,
            Members = pool.Members,
            Layer = l,
            Lane = l % Lanes,
            Count = count,
            Position = position,
            Capacity = capacity,
            HasEngram = site >= 0 ? 1 : 0,
            Geometry = g,
            Weights = model.Blocks[l],
            Hyper = model.Hyper,
            StreamAct = streamAct,
            DimAct = dimAct,
            Stream = stream,
            Next = nextStream,
            Normed = normalizedStream,
            Input = input,
            Residual = residual,
            Quantized = quantizedInput,
            QRaw = qRaw,
            KRaw = kRaw,
            VRaw = vRaw,
            QConv = qConv,
            KConv = kConv,
            VConv = vConv,
            GateOut = gateOut,
            AttnOut = attnOut,
            AttnProj = attnProj,
            EngramInput = engramInput,
            EngramKey = engramKey,
            EngramConv = engramConv,
            EngramValueBuffer = engramValueBuffer,
            QueryScale = queryScale,
            KeyScale = keyScale,
            ValueScale = valueScale,
            RopeCos = ropeCos,
            RopeSin = ropeSin,
            ResidualMix = residualMix,
            PostGate = postGate,
            PreOut = preOut,
            PostOut = postOut,
            ResOut = resOut,
            AttnPartials = attnPartials,
            AttnArrivals = attnArrivals,
            QkvBase = historyPending ? snapshot + (qkvHistory - state) : qkvHistory,
            EngramBase = historyPending ? snapshot + (engramHistory - state) : engramHistory,
            QueryInt8 = queryInt8,
            KeyCache = keyCache,
            ValueCache = valueCache,
            EngramRows = engramRows,
            Mlp = &model.Blocks[l].Mlp,
        };
        if (site >= 0) job.Engram = model.Sites[site];
        clock.PoolRun();
        pool.RunTeam(&ExecuteLayer, System.Runtime.CompilerServices.Unsafe.AsPointer(ref job), barrier);
        var swap = stream;
        stream = nextStream;
        nextStream = swap;
    }
    static void ExecuteLayer(void* context, int member)
    {
        ref var job = ref System.Runtime.CompilerServices.Unsafe.AsRef<LayerJob>(context);
        var members = job.Members;
        HyperNorm(in job, member, members);
        Pool.Barrier(job.Barrier);
        HyperPhi(in job, member, members);
        Pool.Barrier(job.Barrier);
        HyperMix(in job, member);
        if (job.HasEngram != 0)
        {
            EngramGather(in job, member, members);
            Pool.Barrier(job.Barrier);
            EngramMatMul(in job, member, members);
            Pool.Barrier(job.Barrier);
            EngramBlend(in job, member, members);
            Pool.Barrier(job.Barrier);
        }
        if (member == 0) job.Clock.Lap(Stage.Engram);
        AttentionPrepare(in job, member, members);
        Pool.Barrier(job.Barrier);
        AttentionQkvg(in job, member, members);
        Pool.Barrier(job.Barrier);
        TapsHistory(in job, member, members);
        AttentionHeads(in job, member, members);
        Pool.Barrier(job.Barrier);
        AttentionCore(in job, member, members);
        Pool.Barrier(job.Barrier);
        AttentionGate(in job, member, members);
        Pool.Barrier(job.Barrier);
        AttentionOut(in job, member, members);
        Pool.Barrier(job.Barrier);
        PostNorm(in job, member, members);
        Pool.Barrier(job.Barrier);
        Mlp(in job, member, members);
        Pool.Barrier(job.Barrier);
        Combine(in job, member, members);
    }

    static void HyperNorm(in LayerJob job, int member, int members)
    {
        var streamWidth = Lanes * job.Geometry.Dim;
        Shard(job.Count, member, members, out var first, out var count);
        for (var t = first; t < first + count; t++)
        {
            Ops.RmsFakeA8(job.Stream + (long)t * streamWidth, streamWidth, job.Normed + (long)t * streamWidth);
            var slice = job.StreamAct.Slice(t);
            Quant.Prepare(job.Normed + (long)t * streamWidth, 1, streamWidth, 4, slice);
        }
        if (member == 0) job.Clock.Lap(Stage.HyperPrepare);
    }

    static void HyperPhi(in LayerJob job, int member, int members)
    {
        var batch = new MatMulBatch(job.StreamAct, job.Count);
        var l = job.Layer;
        batch.Add(job.Hyper.PhiPre.Slice(l * Lanes, Lanes), job.PreOut, Lanes);
        batch.Add(job.Hyper.PhiPost.Slice(l * Lanes, Lanes), job.PostOut, Lanes);
        batch.Add(job.Hyper.PhiRes.Slice(l * Lanes * Lanes, Lanes * Lanes), job.ResOut, Lanes * Lanes);
        batch.Plan(members);
        Shard(batch.TaskCount, member, members, out var first, out var count);
        for (var i = first; i < first + count; i++) MatMulBatch.RunTask(&batch, i);
        if (member == 0) job.Clock.Lap(Stage.HyperPhi);
    }

    static void HyperMix(in LayerJob job, int member)
    {
        var dim = job.Geometry.Dim;
        var lane = job.Lane;
        var l = job.Layer;
        var mixLogits = stackalloc float[Lanes * Lanes];
        var hpre = stackalloc float[Lanes];
        for (var t = 0; t < job.Count; t++)
        {
            var s = job.Stream + (long)t * Lanes * dim;
            var u = job.Input + (long)t * dim;
            for (var i = 0; i < Lanes; i++)
            {
                hpre[i] = Ops.Sigmoid(job.Hyper.APre[l] * job.PreOut[t * Lanes + i] + job.Hyper.BPre[l * Lanes + i] + (i == lane ? 4f : -4f));
                job.PostGate[t * Lanes + i] = 2f * Ops.Sigmoid(job.Hyper.APost[l] * job.PostOut[t * Lanes + i] + job.Hyper.BPost[l * Lanes + i] + (i == lane ? 0f : -4f));
            }
            Ops.Mix4(s, dim, hpre[0], hpre[1], hpre[2], hpre[3], u);
            for (var i = 0; i < Lanes * Lanes; i++) mixLogits[i] = job.Hyper.ARes[l] * job.ResOut[t * Lanes * Lanes + i] + job.Hyper.BRes[l * Lanes * Lanes + i];
            Ops.Sinkhorn4x4(mixLogits, job.ResidualMix + (long)t * Lanes * Lanes);
            Buffer.MemoryCopy(u, job.Residual + (long)t * dim, dim * 4, dim * 4);
        }
        if (member == 0) job.Clock.Lap(Stage.HyperMix);
    }

    static void EngramGather(in LayerJob job, int member, int members)
    {
        var dim = job.Geometry.Dim;
        var tables = job.Geometry.EngramTables;
        var sub = job.Geometry.EngramSub;
        Shard(job.Count, member, members, out var first, out var count);
        for (var t = first; t < first + count; t++)
        {
            var e = job.EngramInput + (long)t * dim;
            for (var table = 0; table < tables; table++)
            {
                var row = job.EngramRows[t * tables + table];
                if (row < 0) Ops.Zero(e + table * sub, sub);
                else Quant.GatherRow(job.Engram.Tables, row, e + table * sub);
            }
            Quant.FakeA8(e, dim);
            var slice = job.DimAct.Slice(t);
            Quant.Prepare(e, 1, dim, 2, slice);
        }
        if (member == 0) Buffer.MemoryCopy(job.EngramBase, job.EngramValueBuffer, EngramHistory * dim * 4, EngramHistory * dim * 4);
    }

    static void EngramMatMul(in LayerJob job, int member, int members)
    {
        var batch = new MatMulBatch(job.DimAct, job.Count);
        batch.Add(job.Engram.Key, job.EngramKey, job.Geometry.Dim);
        batch.Add(job.Engram.Value, job.EngramValueBuffer + EngramHistory * job.Geometry.Dim, job.Geometry.Dim);
        batch.Plan(members);
        Shard(batch.TaskCount, member, members, out var first, out var count);
        for (var i = first; i < first + count; i++) MatMulBatch.RunTask(&batch, i);
    }

    static void EngramBlend(in LayerJob job, int member, int members)
    {
        var dim = job.Geometry.Dim;
        var unitA = stackalloc float[dim];
        var unitB = stackalloc float[dim];
        var inverseRootDim = 1f / MathF.Sqrt(dim);
        Shard(job.Count, member, members, out var first, out var count);
        for (var t = first; t < first + count; t++)
        {
            var o = job.EngramConv + (long)t * dim;
            Ops.Zero(o, dim);
            for (var j = 0; j < EngramTaps; j++)
                Ops.MultiplyAccumulate(o, job.Engram.Taps + j * dim, job.EngramValueBuffer + (long)(EngramHistory + t - j * job.Geometry.EngramDilation) * dim, dim);
        }
        if (member == 0) Buffer.MemoryCopy(job.EngramValueBuffer + (long)job.Count * dim, job.EngramBase, EngramHistory * dim * 4, EngramHistory * dim * 4);
        for (var t = first; t < first + count; t++)
        {
            var x = job.Input + (long)t * dim;
            Ops.RmsUnit(x, dim, unitA);
            Ops.RmsUnit(job.EngramKey + (long)t * dim, dim, unitB);
            var alpha = Ops.Sigmoid(Ops.Dot(unitA, unitB, dim) * inverseRootDim);
            Ops.AddScaled(job.Residual + (long)t * dim, job.EngramConv + (long)t * dim, alpha, dim);
        }
    }
    static void AttentionPrepare(in LayerJob job, int member, int members)
    {
        var dim = job.Geometry.Dim;
        Shard(job.Count, member, members, out var first, out var count);
        for (var t = first; t < first + count; t++)
        {
            Ops.CenteredRmsFakeA8(job.Residual + (long)t * dim, job.Weights.NormIn, dim, job.Quantized + (long)t * dim);
            var slice = job.DimAct.Slice(t);
            Quant.Prepare(job.Quantized + (long)t * dim, 1, dim, 2, slice);
        }
        if (member == 0) job.Clock.Lap(Stage.AttentionPrep);
    }

    static void AttentionQkvg(in LayerJob job, int member, int members)
    {
        var qWidth = job.Geometry.QueryWidth;
        var kWidth = job.Geometry.KeyWidth;
        var vWidth = job.Geometry.ValueWidth;
        var batch = new MatMulBatch(job.DimAct, job.Count);
        batch.Add(job.Weights.Q, job.QRaw + (ConvTaps - 1) * qWidth, qWidth);
        batch.Add(job.Weights.K, job.KRaw + (ConvTaps - 1) * kWidth, kWidth);
        batch.Add(job.Weights.V, job.VRaw + (ConvTaps - 1) * vWidth, vWidth);
        batch.Add(job.Weights.Gate, job.GateOut, job.Geometry.OutputWidth);
        batch.Plan(members);
        Shard(batch.TaskCount, member, members, out var first, out var count);
        for (var i = first; i < first + count; i++) MatMulBatch.RunTask(&batch, i);
        if (member == 0) job.Clock.Lap(Stage.AttentionQkvg);
    }

    static void TapsHistory(in LayerJob job, int member, int members)
    {
        var qWidth = job.Geometry.QueryWidth;
        var kWidth = job.Geometry.KeyWidth;
        var vWidth = job.Geometry.ValueWidth;
        var qHistory = job.QkvBase;
        var kHistory = qHistory + (ConvTaps - 1) * qWidth;
        var vHistory = kHistory + (ConvTaps - 1) * kWidth;
        if (member == 0)
        {
            LoadHistory(qHistory, job.QRaw, qWidth);
            LoadHistory(kHistory, job.KRaw, kWidth);
            LoadHistory(vHistory, job.VRaw, vWidth);
        }
        Pool.Barrier(job.Barrier);
        Shard(job.Count, member, members, out var first, out var count);
        for (var t = first; t < first + count; t++)
        {
            Ops.Taps3(job.Weights.QTaps, qWidth, job.QRaw + (long)(t + ConvTaps - 1) * qWidth, job.QConv + (long)t * qWidth);
            Ops.Taps3(job.Weights.KTaps, kWidth, job.KRaw + (long)(t + ConvTaps - 1) * kWidth, job.KConv + (long)t * kWidth);
            Ops.Taps3(job.Weights.VTaps, vWidth, job.VRaw + (long)(t + ConvTaps - 1) * vWidth, job.VConv + (long)t * vWidth);
        }
        Pool.Barrier(job.Barrier);
        if (member == 0)
        {
            SaveHistory(qHistory, job.QRaw, job.Count, qWidth);
            SaveHistory(kHistory, job.KRaw, job.Count, kWidth);
            SaveHistory(vHistory, job.VRaw, job.Count, vWidth);
        }
    }

    static void AttentionHeads(in LayerJob job, int member, int members)
    {
        var heads = job.Geometry.Heads;
        var kvHeads = job.Geometry.KvHeads;
        var qkDim = job.Geometry.QkDim;
        var vDim = job.Geometry.VDim;
        var half = qkDim / 2;
        var layerKeys = job.KeyCache + (long)job.Layer * job.Capacity * job.Geometry.KeyWidth;
        var layerValues = job.ValueCache + (long)job.Layer * job.Capacity * job.Geometry.ValueWidth;
        var layerKeyScales = job.KeyScale + (long)job.Layer * job.Capacity * kvHeads;
        var layerValueScales = job.ValueScale + (long)job.Layer * job.Capacity * kvHeads;
        Shard(job.Count * (heads + kvHeads), member, members, out var first, out var count);
        for (var u = first; u < first + count; u++)
        {
            var t = u / (heads + kvHeads);
            var h = u % (heads + kvHeads);
            var p = job.Position + t;
            var cos = job.RopeCos + (long)p * half;
            var sin = job.RopeSin + (long)p * half;
            if (h < heads)
            {
                var q = job.QConv + ((long)t * heads + h) * qkDim;
                Ops.ZeroCenteredRms(q, job.Weights.QNorm, qkDim, q);
                Ops.Rope(q, 1, qkDim, cos, sin);
                job.QueryScale[(long)t * heads + h] = Quant.QuantizeInt8(q, qkDim, job.QueryInt8 + ((long)t * heads + h) * qkDim);
            }
            else
            {
                var k = h - heads;
                var kv = job.KConv + ((long)t * kvHeads + k) * qkDim;
                Ops.ZeroCenteredRms(kv, job.Weights.KNorm, qkDim, kv);
                Ops.Rope(kv, 1, qkDim, cos, sin);
                layerKeyScales[(long)p * kvHeads + k] = Quant.QuantizeInt8(kv, qkDim, layerKeys + ((long)p * kvHeads + k) * qkDim);
                var v = job.VConv + ((long)t * kvHeads + k) * vDim;
                layerValueScales[(long)p * kvHeads + k] = Quant.QuantizeInt8(v, vDim, layerValues + ((long)p * kvHeads + k) * vDim);
            }
        }
        if (member == 0) job.Clock.Lap(Stage.AttentionPostProjection);
    }

    static void AttentionCore(in LayerJob job, int member, int members)
    {
        var kvHeads = job.Geometry.KvHeads;
        var batch = new AttentionBatch
        {
            Queries = job.QueryInt8,
            QueryScales = job.QueryScale,
            Keys = job.KeyCache + (long)job.Layer * job.Capacity * job.Geometry.KeyWidth,
            KeyScales = job.KeyScale + (long)job.Layer * job.Capacity * kvHeads,
            Values = job.ValueCache + (long)job.Layer * job.Capacity * job.Geometry.ValueWidth,
            ValueScales = job.ValueScale + (long)job.Layer * job.Capacity * kvHeads,
            Output = job.AttnOut,
            Heads = job.Geometry.Heads,
            KvHeads = kvHeads,
            QkDim = job.Geometry.QkDim,
            VDim = job.Geometry.VDim,
            Position = job.Position,
            Window = job.Geometry.IsGlobal(job.Layer) ? 0 : job.Geometry.SlidingWindow,
            Capacity = job.Capacity,
        };
        batch.Plan(members, job.Count, job.AttnPartials, job.AttnArrivals);
        Shard(batch.TeamTasks, member, members, out var first, out var count);
        for (var i = first; i < first + count; i++) AttentionBatch.RunTask(&batch, i);
        if (member == 0) job.Clock.Lap(Stage.AttentionCore);
    }

    static void AttentionGate(in LayerJob job, int member, int members)
    {
        var outWidth = job.Geometry.OutputWidth;
        Shard(job.Count, member, members, out var first, out var count);
        for (var t = first; t < first + count; t++)
        {
            Ops.GateFakeA8(job.AttnOut + (long)t * outWidth, job.GateOut + (long)t * outWidth, outWidth);
            var slice = job.DimAct.Slice(t);
            Quant.Prepare(job.AttnOut + (long)t * outWidth, 1, outWidth, 2, slice);
        }
        if (member == 0) job.Clock.Lap(Stage.AttentionGate);
    }

    static void AttentionOut(in LayerJob job, int member, int members)
    {
        var batch = new MatMulBatch(job.DimAct, job.Count);
        batch.Add(job.Weights.Out, job.AttnProj, job.Geometry.Dim);
        batch.Plan(members);
        Shard(batch.TaskCount, member, members, out var first, out var count);
        for (var i = first; i < first + count; i++) MatMulBatch.RunTask(&batch, i);
        if (member == 0) job.Clock.Lap(Stage.AttentionOut);
    }

    static void PostNorm(in LayerJob job, int member, int members)
    {
        var dim = job.Geometry.Dim;
        Shard(job.Count, member, members, out var first, out var count);
        for (var t = first; t < first + count; t++)
            Ops.PostNormAdd(job.Residual + (long)t * dim, job.AttnProj + (long)t * dim, job.Weights.PostNorm, job.Weights.AttnGate, dim);
        if (member == 0) job.Clock.Lap(Stage.AttentionPostNorm);
    }

    static void Mlp(in LayerJob job, int member, int members)
    {
        var batch = new MlpBatch
        {
            Weights = job.Mlp,
            PreNorm = job.Weights.PreHada,
            PermA = job.Hyper.PermA,
            PermB = job.Hyper.PermB,
            Residual = job.Residual,
            Dim = job.Geometry.Dim,
            Rows = job.Geometry.HadaRows,
            Cols = job.Geometry.HadaCols,
            Scale = 1f,
        };
        Shard(job.Count, member, members, out var first, out var count);
        for (var t = first; t < first + count; t++) MlpBatch.RunToken(&batch, t);
        if (member == 0) job.Clock.Lap(Stage.Mlp);
    }

    static void Combine(in LayerJob job, int member, int members)
    {
        var dim = job.Geometry.Dim;
        Shard(job.Count, member, members, out var first, out var count);
        for (var t = first; t < first + count; t++)
        {
            var s = job.Stream + (long)t * Lanes * dim;
            var n = job.Next + (long)t * Lanes * dim;
            var x = job.Residual + (long)t * dim;
            var u = job.Input + (long)t * dim;
            var mix = job.ResidualMix + (long)t * Lanes * Lanes;
            for (var i = 0; i < Lanes; i++)
                Ops.CombineDiff4(s, dim, mix[i * 4], mix[i * 4 + 1], mix[i * 4 + 2], mix[i * 4 + 3], x, u, job.PostGate[t * Lanes + i], n + (long)i * dim);
        }
        if (member == 0) job.Clock.Lap(Stage.Combine);
    }

    void RunLayerLegacy(int l, int count, int position)
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
    internal void LogitsForRows(int* rows, int count, float* destination)
    {
        PrepareHead(lastForwardCount - 1);
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
