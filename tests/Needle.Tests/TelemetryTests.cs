using System;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Xunit;

namespace Needle.Tests;

public sealed unsafe class TelemetryTests
{
    static Reply SyntheticReply(ReplyStatus status, int decodeTokens, long prefillMillis, long decodeMillis, float confidence)
    {
        var freq = Stopwatch.Frequency;
        var prefillTicks = freq / 1000 * prefillMillis;
        var firstTokenTicks = prefillTicks + freq / 1000;
        var elapsedTicks = firstTokenTicks + freq / 1000 * decodeMillis;
        return new Reply(status, FailureReason.None, ReplyFlags.None, confidence, confidence, confidence,
            1, 10, 5, 6, 18, decodeTokens, prefillTicks, firstTokenTicks, elapsedTicks);
    }

    [Fact]
    public void BlockIsBlittableWithFixedLayout()
    {
        Assert.False(RuntimeHelpers.IsReferenceOrContainsReferences<NeedleStats>());
        Assert.Equal(584, sizeof(NeedleStats));
        Assert.Equal(584, Marshal.SizeOf<NeedleStats>());
        Assert.Equal(0, (int)Marshal.OffsetOf<NeedleStats>(nameof(NeedleStats.Magic)));
        Assert.Equal(8, (int)Marshal.OffsetOf<NeedleStats>(nameof(NeedleStats.Schema)));
        Assert.Equal(16, (int)Marshal.OffsetOf<NeedleStats>(nameof(NeedleStats.Submitted)));
        Assert.Equal(56, (int)Marshal.OffsetOf<NeedleStats>(nameof(NeedleStats.QueueDepth)));
        Assert.Equal(64, (int)Marshal.OffsetOf<NeedleStats>(nameof(NeedleStats.LastPrefillMicros)));
        Assert.Equal(152, (int)Marshal.OffsetOf<NeedleStats>(nameof(NeedleStats.Tier)));
        Assert.Equal(160, (int)Marshal.OffsetOf<NeedleStats>(nameof(NeedleStats.ConfidenceSum)));
        Assert.Equal(192, (int)Marshal.OffsetOf<NeedleStats>(nameof(NeedleStats.StageMicros)));
        Assert.Equal(320, (int)Marshal.OffsetOf<NeedleStats>(nameof(NeedleStats.PrefillBuckets)));
        Assert.Equal(448, (int)Marshal.OffsetOf<NeedleStats>(nameof(NeedleStats.DecodeBuckets)));
        Assert.Equal(576, (int)Marshal.OffsetOf<NeedleStats>(nameof(NeedleStats.LapStamp)));
        Assert.Equal(16, NeedleStats.StageCount);
        Assert.Equal(NeedleStats.StageCount, (int)Stage.Logits + 1);
        Assert.Equal(NeedleStats.BucketCount, 16);
    }

    [Fact]
    public void BucketsCoverTheMicrosecondRange()
    {
        Assert.Equal(0, NeedleTelemetry.BucketFor(0));
        Assert.Equal(0, NeedleTelemetry.BucketFor(127));
        Assert.Equal(1, NeedleTelemetry.BucketFor(128));
        Assert.Equal(1, NeedleTelemetry.BucketFor(255));
        Assert.Equal(2, NeedleTelemetry.BucketFor(256));
        Assert.Equal(4, NeedleTelemetry.BucketFor(2000));
        Assert.Equal(7, NeedleTelemetry.BucketFor(15000));
        Assert.Equal(15, NeedleTelemetry.BucketFor(long.MaxValue));
        Assert.Equal(128L << 2, NeedleTelemetry.BucketUpperBound(2));
    }

    [Fact]
    public void WriterTalliesSubmitClaimAndReply()
    {
        var stats = NeedleTelemetry.Allocate();
        try
        {
            Assert.Equal(NeedleStats.MagicNumber, stats->Magic);
            Assert.Equal(NeedleStats.SchemaVersion, stats->Schema);
            NeedleTelemetry.NoteSubmit(stats);
            NeedleTelemetry.NoteSubmit(stats);
            NeedleTelemetry.NoteSubmit(stats);
            Assert.Equal(3, Volatile.Read(ref stats->Submitted));
            Assert.Equal(3, Volatile.Read(ref stats->QueueDepth));
            NeedleTelemetry.NoteClaim(stats);
            Assert.Equal(1, Volatile.Read(ref stats->Claimed));
            Assert.Equal(2, Volatile.Read(ref stats->QueueDepth));

            NeedleTelemetry.NoteReply(stats, SyntheticReply(ReplyStatus.Called, 40, 15, 80, 0.8f));
            NeedleTelemetry.NoteReply(stats, SyntheticReply(ReplyStatus.Failed, 0, 1, 0, 0.2f));
            NeedleTelemetry.NoteReply(stats, SyntheticReply(ReplyStatus.Cancelled, 10, 5, 20, 0.5f));

            Assert.Equal(1, Volatile.Read(ref stats->Completed));
            Assert.Equal(1, Volatile.Read(ref stats->Failed));
            Assert.Equal(1, Volatile.Read(ref stats->Cancelled));
            Assert.Equal(54, Volatile.Read(ref stats->PromptTokens));
            Assert.Equal(50, Volatile.Read(ref stats->DecodeTokens));
            Assert.Equal(NeedleTelemetry.ToMicros(Stopwatch.Frequency / 1000 * 5),
                Volatile.Read(ref stats->LastPrefillMicros));

            long bucketed = 0;
            for (var i = 0; i < NeedleStats.BucketCount; i++)
                bucketed += Volatile.Read(ref stats->PrefillBuckets[i]);
            Assert.Equal(3, bucketed);

            Assert.Equal(0.5, NeedleTelemetry.MeanConfidence(stats), 6);
            Assert.Equal(0.2f, stats->ConfidenceMin);
            Assert.Equal(0.8f, stats->ConfidenceMax);

            NeedleTelemetry.NotePasses(stats, 100, 36);
            Assert.Equal(2.0, NeedleTelemetry.PassesPerToken(stats), 6);

            var freq = Stopwatch.Frequency;
            NeedleTelemetry.NoteWorkerSpan(stats, freq * 3, freq);
            Assert.Equal(0.75, NeedleTelemetry.BusyFraction(stats), 6);

            NeedleTelemetry.NoteTier(stats, 3, 1L << 30);
            Assert.Equal(3, Volatile.Read(ref stats->Tier));
            Assert.Equal(1L << 30, Volatile.Read(ref stats->MemoryBytes));

            Assert.True(NeedleTelemetry.TokensPerSecond(stats) > 0);
        }
        finally
        {
            NeedleTelemetry.Free(stats);
        }
    }

    [Fact]
    public void QuantilesScanFixedBuckets()
    {
        var stats = NeedleTelemetry.Allocate();
        try
        {
            stats->PrefillBuckets[2] = 90;
            stats->PrefillBuckets[7] = 10;
            Assert.Equal(0, NeedleTelemetry.DecodeP50(stats));
            Assert.Equal(128L << 2, NeedleTelemetry.PrefillP50(stats));
            Assert.Equal(128L << 7, NeedleTelemetry.PrefillP95(stats));
        }
        finally
        {
            NeedleTelemetry.Free(stats);
        }
    }

    [Fact]
    public void ClockWritesStagesThroughExistingHooks()
    {
        var stats = NeedleTelemetry.Allocate();
        try
        {
            var clock = new TelemetryClock(stats);
            IStageClock hooks = clock;
            hooks.Begin();
            hooks.Lap(Stage.Mlp);
            hooks.PoolRun();
            hooks.PoolRun();
            hooks.PoolRun();
            Assert.Equal(3, Volatile.Read(ref stats->PoolRuns));
            Assert.True(Volatile.Read(ref stats->LapStamp) > 0);
            Assert.True(Volatile.Read(ref stats->StageMicros[(int)Stage.Mlp]) >= 0);
        }
        finally
        {
            NeedleTelemetry.Free(stats);
        }
    }

    [Fact]
    public void DisabledWritersAreNullSafe()
    {
        NeedleTelemetry.NoteSubmit(null);
        NeedleTelemetry.NoteClaim(null);
        NeedleTelemetry.NoteReply(null, SyntheticReply(ReplyStatus.Called, 1, 1, 1, 0.5f));
        NeedleTelemetry.NotePasses(null, 1, 0);
        NeedleTelemetry.NoteWorkerSpan(null, 1, 1);
        NeedleTelemetry.NoteTier(null, 0, 0);
        Assert.Equal(0, NeedleTelemetry.TokensPerSecond(null));
        Assert.Equal(0, NeedleTelemetry.PassesPerToken(null));
        Assert.Equal(0, NeedleTelemetry.BusyFraction(null));
        Assert.Equal(0, NeedleTelemetry.MeanConfidence(null));
        Assert.Equal(0, NeedleTelemetry.PrefillP50(null));
        var clock = default(TelemetryClock);
        IStageClock hooks = clock;
        hooks.Begin();
        hooks.Lap(Stage.Logits);
        hooks.PoolRun();
    }

    [Fact]
    public void MeterObservesBlockCounters()
    {
        using var listener = new MeterListener();
        long completed = -1;
        listener.InstrumentPublished = (instrument, l) =>
        {
            if (instrument.Meter.Name == "Needle.Inference") l.EnableMeasurementEvents(instrument);
        };
        listener.SetMeasurementEventCallback<long>((instrument, measurement, _, _) =>
        {
            if (instrument.Name == "needle.requests.completed") completed = measurement;
        });
        listener.Start();
        var stats = NeedleTelemetry.Allocate();
        try
        {
            using var meter = new NeedleMeter(stats);
            NeedleTelemetry.NoteReply(stats, SyntheticReply(ReplyStatus.Called, 40, 15, 80, 0.8f));
            listener.RecordObservableInstruments();
            Assert.Equal(1, completed);
        }
        finally
        {
            NeedleTelemetry.Free(stats);
        }
    }
}
