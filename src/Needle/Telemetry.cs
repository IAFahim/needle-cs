using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
#if NET6_0_OR_GREATER
using System.Diagnostics.Metrics;
#endif

namespace Needle;

[StructLayout(LayoutKind.Explicit, Size = 584)]
public unsafe struct NeedleStats
{
    public const int StageCount = 16;
    public const int BucketCount = 16;
    public const int SchemaVersion = 1;
    public const long MagicNumber = unchecked((long)0x31544154534C444EUL);

    [FieldOffset(0)] public long Magic;
    [FieldOffset(8)] public long Schema;
    [FieldOffset(16)] public long Submitted;
    [FieldOffset(24)] public long Claimed;
    [FieldOffset(32)] public long Completed;
    [FieldOffset(40)] public long Failed;
    [FieldOffset(48)] public long Cancelled;
    [FieldOffset(56)] public long QueueDepth;
    [FieldOffset(64)] public long LastPrefillMicros;
    [FieldOffset(72)] public long LastDecodeMicrosPerToken;
    [FieldOffset(80)] public long PromptTokens;
    [FieldOffset(88)] public long DecodeTokens;
    [FieldOffset(96)] public long ModelPasses;
    [FieldOffset(104)] public long SpeculativeAccepted;
    [FieldOffset(112)] public long ElapsedTicks;
    [FieldOffset(120)] public long PoolRuns;
    [FieldOffset(128)] public long BusyTicks;
    [FieldOffset(136)] public long IdleTicks;
    [FieldOffset(144)] public long MemoryBytes;
    [FieldOffset(152)] public int Tier;
    [FieldOffset(156)] public int Reserved;
    [FieldOffset(160)] public double ConfidenceSum;
    [FieldOffset(168)] public long ConfidenceSamples;
    [FieldOffset(176)] public float ConfidenceMin;
    [FieldOffset(180)] public float ConfidenceMax;
    [FieldOffset(192)] public fixed long StageMicros[16];
    [FieldOffset(320)] public fixed long PrefillBuckets[16];
    [FieldOffset(448)] public fixed long DecodeBuckets[16];
    [FieldOffset(576)] public long LapStamp;
}

public static unsafe class NeedleTelemetry
{
    public static NeedleStats* Allocate()
    {
        var stats = (NeedleStats*)Mem.AlignedAlloc((nuint)sizeof(NeedleStats), 64);
        Mem.Clear(stats, (nuint)sizeof(NeedleStats));
        stats->Magic = NeedleStats.MagicNumber;
        stats->Schema = NeedleStats.SchemaVersion;
        stats->ConfidenceMin = float.MaxValue;
        stats->ConfidenceMax = float.MinValue;
        return stats;
    }

    public static void Free(NeedleStats* stats)
    {
        if (stats != null) Mem.AlignedFree(stats);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void NoteSubmit(NeedleStats* stats)
    {
        if (stats == null) return;
        Interlocked.Increment(ref stats->Submitted);
        Interlocked.Increment(ref stats->QueueDepth);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void NoteClaim(NeedleStats* stats)
    {
        if (stats == null) return;
        Interlocked.Increment(ref stats->Claimed);
        Interlocked.Decrement(ref stats->QueueDepth);
    }

    public static void NoteReply(NeedleStats* stats, in Reply reply)
    {
        if (stats == null) return;
        if (reply.Status == ReplyStatus.Cancelled) Interlocked.Increment(ref stats->Cancelled);
        else if (reply.Status == ReplyStatus.Failed) Interlocked.Increment(ref stats->Failed);
        else Interlocked.Increment(ref stats->Completed);
        var prefillMicros = ToMicros(reply.PrefillTicks);
        var decodeMicros = ToMicros(Math.Max(0, reply.ElapsedTicks - reply.FirstTokenTicks));
        var perToken = reply.DecodeTokens > 0 ? decodeMicros / reply.DecodeTokens : 0;
        Volatile.Write(ref stats->LastPrefillMicros, prefillMicros);
        Volatile.Write(ref stats->LastDecodeMicrosPerToken, perToken);
        stats->PrefillBuckets[BucketFor(prefillMicros)]++;
        stats->DecodeBuckets[BucketFor(perToken)]++;
        Interlocked.Add(ref stats->PromptTokens, Math.Max(0, reply.PromptTokens));
        Interlocked.Add(ref stats->DecodeTokens, Math.Max(0, reply.DecodeTokens));
        Interlocked.Add(ref stats->ElapsedTicks, Math.Max(0, reply.ElapsedTicks));
        var confidence = reply.Confidence;
        stats->ConfidenceSum += confidence;
        Interlocked.Increment(ref stats->ConfidenceSamples);
        if (confidence < stats->ConfidenceMin) stats->ConfidenceMin = confidence;
        if (confidence > stats->ConfidenceMax) stats->ConfidenceMax = confidence;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void NotePasses(NeedleStats* stats, long passes, long speculativeAccepted)
    {
        if (stats == null) return;
        Interlocked.Add(ref stats->ModelPasses, passes);
        Interlocked.Add(ref stats->SpeculativeAccepted, speculativeAccepted);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void NoteWorkerSpan(NeedleStats* stats, long busyTicks, long idleTicks)
    {
        if (stats == null) return;
        Interlocked.Add(ref stats->BusyTicks, busyTicks);
        Interlocked.Add(ref stats->IdleTicks, idleTicks);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void NoteTier(NeedleStats* stats, int tier, long memoryBytes)
    {
        if (stats == null) return;
        Volatile.Write(ref stats->MemoryBytes, memoryBytes);
        Volatile.Write(ref stats->Tier, tier);
    }

    public static long ToMicros(long ticks) => ticks * 1_000_000L / Stopwatch.Frequency;

    public static int BucketFor(long micros)
    {
        if (micros < 128) return 0;
        var bucket = Bits.Log2((ulong)micros) - 6;
        return bucket > NeedleStats.BucketCount - 1 ? NeedleStats.BucketCount - 1 : bucket;
    }

    public static long BucketUpperBound(int bucket) =>
        bucket >= NeedleStats.BucketCount - 1 ? long.MaxValue : 128L << bucket;

    public static double TokensPerSecond(NeedleStats* stats)
    {
        if (stats == null) return 0;
        var tokens = Volatile.Read(ref stats->DecodeTokens);
        var ticks = Volatile.Read(ref stats->ElapsedTicks);
        if (ticks <= 0) return 0;
        return tokens * (double)Stopwatch.Frequency / ticks;
    }

    public static double PassesPerToken(NeedleStats* stats)
    {
        if (stats == null) return 0;
        var tokens = Volatile.Read(ref stats->DecodeTokens);
        if (tokens <= 0) return 0;
        return (double)Volatile.Read(ref stats->ModelPasses) / tokens;
    }

    public static double BusyFraction(NeedleStats* stats)
    {
        if (stats == null) return 0;
        var busy = Volatile.Read(ref stats->BusyTicks);
        var idle = Volatile.Read(ref stats->IdleTicks);
        var total = busy + idle;
        if (total <= 0) return 0;
        return (double)busy / total;
    }

    public static double MeanConfidence(NeedleStats* stats)
    {
        if (stats == null) return 0;
        var samples = Volatile.Read(ref stats->ConfidenceSamples);
        if (samples <= 0) return 0;
        return Volatile.Read(ref stats->ConfidenceSum) / samples;
    }

    public static long PrefillP50(NeedleStats* stats) => Quantile(stats, 0, 0.50);

    public static long PrefillP95(NeedleStats* stats) => Quantile(stats, 0, 0.95);

    public static long DecodeP50(NeedleStats* stats) => Quantile(stats, 1, 0.50);

    public static long DecodeP95(NeedleStats* stats) => Quantile(stats, 1, 0.95);

    static long Quantile(NeedleStats* stats, int table, double fraction)
    {
        if (stats == null) return 0;
        long total = 0;
        for (var i = 0; i < NeedleStats.BucketCount; i++)
            total += table == 0
                ? Volatile.Read(ref stats->PrefillBuckets[i])
                : Volatile.Read(ref stats->DecodeBuckets[i]);
        if (total <= 0) return 0;
        var rank = (long)Math.Ceiling(total * fraction);
        long seen = 0;
        for (var i = 0; i < NeedleStats.BucketCount; i++)
        {
            seen += table == 0
                ? Volatile.Read(ref stats->PrefillBuckets[i])
                : Volatile.Read(ref stats->DecodeBuckets[i]);
            if (seen >= rank) return BucketUpperBound(i);
        }
        return BucketUpperBound(NeedleStats.BucketCount - 1);
    }
}

public readonly unsafe struct TelemetryClock : IStageClock
{
    public readonly NeedleStats* Stats;

    public TelemetryClock(NeedleStats* stats) => Stats = stats;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Begin()
    {
        var stats = Stats;
        if (stats == null) return;
        Volatile.Write(ref stats->LapStamp, Stopwatch.GetTimestamp());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Lap(Stage stage)
    {
        var stats = Stats;
        if (stats == null) return;
        if ((uint)stage >= NeedleStats.StageCount) return;
        var now = Stopwatch.GetTimestamp();
        Volatile.Write(ref stats->StageMicros[(int)stage], NeedleTelemetry.ToMicros(now - Volatile.Read(ref stats->LapStamp)));
        Volatile.Write(ref stats->LapStamp, now);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PoolRun()
    {
        var stats = Stats;
        if (stats == null) return;
        Interlocked.Increment(ref stats->PoolRuns);
    }
}

#if NET6_0_OR_GREATER
public sealed unsafe class NeedleMeter : IDisposable
{
    readonly Meter meter;
    readonly nint address;
    int disposed;

    public NeedleMeter(NeedleStats* stats)
    {
        address = (nint)stats;
        meter = new Meter("Needle.Inference", "1.0");
        _ = meter.CreateObservableCounter("needle.requests.completed", ObserveCompleted, "requests");
        _ = meter.CreateObservableCounter("needle.requests.failed", ObserveFailed, "requests");
        _ = meter.CreateObservableCounter("needle.requests.cancelled", ObserveCancelled, "requests");
        _ = meter.CreateObservableCounter("needle.tokens.decoded", ObserveDecodeTokens, "tokens");
        _ = meter.CreateObservableUpDownCounter("needle.queue.depth", ObserveQueueDepth, "requests");
        _ = meter.CreateObservableGauge("needle.tokens.per_second", ObserveTokensPerSecond, "tokens/s");
        _ = meter.CreateObservableGauge("needle.worker.busy_fraction", ObserveBusyFraction, "ratio");
    }

    public Meter Meter => meter;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) == 0) meter.Dispose();
    }

    NeedleStats* Stats => (NeedleStats*)address;

    long ObserveCompleted() => Volatile.Read(ref Stats->Completed);

    long ObserveFailed() => Volatile.Read(ref Stats->Failed);

    long ObserveCancelled() => Volatile.Read(ref Stats->Cancelled);

    long ObserveDecodeTokens() => Volatile.Read(ref Stats->DecodeTokens);

    int ObserveQueueDepth() => (int)Volatile.Read(ref Stats->QueueDepth);

    double ObserveTokensPerSecond() => NeedleTelemetry.TokensPerSecond(Stats);

    double ObserveBusyFraction() => NeedleTelemetry.BusyFraction(Stats);
}
#endif
