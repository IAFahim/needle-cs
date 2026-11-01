# Needle telemetry in the Unity Profiler

`NeedleStats` is a 584-byte blittable block in native memory. The host
worker writes it with lock-free counters (`Interlocked` for cross-thread
tallies, single-writer fields for stage times); Burst jobs read it through
a raw pointer with no managed references. Full-request accounting costs
about 35 ns; one stage lap costs about 22 ns.

## Wiring

```csharp
unsafe void Start()
{
    stats = NeedleTelemetry.Allocate();
    NeedleProfiler.Attach(stats);
    // Host records submit/claim/reply via NeedleTelemetry.Note*.
}

void Update()
{
    using (NeedleProfiler.SystemUpdate.Auto())
    {
        NeedleProfiler.Sample(); // once per frame, main thread
        // ... TrySubmit / TryReceive, each wrapped below
    }
}

bool Enqueue(System.ReadOnlySpan<byte> query)
{
    using (NeedleProfiler.Submit.Auto())
        return channel.TrySubmitQuery(query, out _);
}

bool Drain(out ChannelReply reply)
{
    using (NeedleProfiler.Receive.Auto())
        return channel.TryReceive(out reply);
}

void OnDestroy()
{
    NeedleProfiler.Detach();
    NeedleTelemetry.Free(stats);
}
```

Pass a marker by value into a Burst job; read the block directly:

```csharp
[BurstCompile]
struct SubmitJob : IJob
{
    public ProfilerMarker Marker;
    [NativeDisableUnsafePtrRestriction] public unsafe NeedleStats* Stats;
    public void Execute()
    {
        using (Marker.Auto())
        {
            long depth = Stats->QueueDepth;
        }
    }
}
```

## Profiler surface

`Needle.Channel.Submit`, `Needle.Channel.Receive` and `Needle.System`
appear under CPU Usage markers. Eight counters appear under the
user-counters chart: queue depth, requests completed, requests
failed (including cancelled), tokens per second, worker busy fraction,
last prefill us, last decode us per token, and mean confidence.
Per-stage decode us, p50/p95 prefill and decode histograms, pool
runs per token and passes per token stay in the block for Burst-side
reads via `NeedleProfiler.Attached`.

Non-Unity hosts use `NeedleMeter` (`System.Diagnostics.Metrics`,
meter `Needle.Inference`) instead of this bridge.
