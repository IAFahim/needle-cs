# Needle thread budget in Unity games

All three samples (`SquadCommander`, `NpcSample`, `VoiceSample`) run inference
on `NeedleHost`/Whistle worker threads with `ThreadPolicy.Game` and pick the
thread count automatically. The Inspector `Threads` field overrides it
(`0` = automatic, `1`–`4` = forced).

## Defaults

| Setting | Value | Why |
|---|---|---|
| Priority | `BelowNormal` (`ThreadPolicy.Game`) | Unity's job workers, render thread and main thread win every contention |
| Idle park | 50 µs (`GameSpinMicroseconds`) | Workers sleep instead of burning cores between queries |
| Placement | `Any` | No core pinning, so workers float away from busy cores |
| Threads | `max(1, min(4, cores / 2))` | 4 where the box has headroom, 2 on a 4-core laptop, 1 only on single-core |

Worked examples (`cores` = `Environment.ProcessorCount`): 32 cores → 4,
8 cores → 4, 6 cores → 3, 4 cores → 2, 2 cores → 1.

### Why not `cores minus Unity job workers`

`JobsUtility.JobWorkerCount` counts **parked** threads, not busy cores. The
32-core Linux Editor used for the measurements below reports **31** workers,
so the literal subtraction gives `min(4, 32 − 31) = 1` thread — one decode
lane on a machine with 30 idle cores. (In Unity 7000 `JobsUtility` also lives
in `UnityEngine.ManagedKernelModule`, not in an assembly called `Unity.Jobs`,
so the old reflection lookup misses.) Coexistence with Unity comes from the
policy instead — priority, parking and pinning — which the frame-time table
below shows to be sufficient.

## `NEEDLE_PIN=0` in the Editor

`Topology.Select` pins Needle workers only for `PerformanceCores`,
`EfficiencyCores` and `Explicit` placements. The samples use `Game` (`Any`),
so they already float. Sessions built on `ThreadPolicy.Default`
(`PerformanceCores`) pin P-cores — next to the Editor's own import workers and
job system. To turn that off for Default-policy sessions in the Editor:

```sh
export NEEDLE_PIN=0  # before launching the Editor, or
```

```csharp
System.Environment.SetEnvironmentVariable("NEEDLE_PIN", "0"); // before the first session
```

`0` disables all Needle pinning; any other value (or unset) keeps the
policy placement. It has no effect on the samples' Game-policy sessions.

## Measurements

### Main-thread frame cost in the Editor (acceptance)

Two live hosts (ns2.1 build, Game policy, one background query each in
flight), main thread polls `TryReceive` 240 times ~16 ms apart, interleaved
2, 4, 2, 4. Unity 7000.0.0a12, Linux x64, 32 logical cores, `JobWorkerCount`
31, `NEEDLE_PIN` unset, shared dev box (Editor importing, other benches
running — absolute worst-cases are noisy, the interleaving shares it).

| Round | Threads | n | Mean µs | Median µs | p99 µs | Max µs |
|---|---|---|---|---|---|---|
| 1 | 2 | 240 | 3.0 | 1.7 | 10.4 | 245.9 |
| 1 | 4 | 240 | 2.6 | 2.6 | 4.9 | 5.4 |
| 2 | 2 | 240 | 2.7 | 2.6 | 6.1 | 14.5 |
| 2 | 4 | 240 | 2.9 | 2.8 | 7.9 | 8.0 |

Medians are 2–3 µs and p99 stays under 11 µs at **both** counts — under a
thousandth of a 16.7 ms frame. The Update() path is a ring-buffer pop; the
thread count does not move it. Raising the default to 4 where cores allow is
frame-safe; slow runtimes keep 2 through `cores / 2`.

### Worker-latency context (CLI, net10 build)

Uncontended reference (README, quiet 4-core VM): 73–84 ms per call at
4 threads. On the shared box above, interleaved `bench-e2e` rounds
(`--threads 2` vs `--threads 4`, `--reps 2`, golden set, 5 rounds each,
loadavg ~50) gave best-of-bests of ~56–63 ms at 2 threads but ~1.3–2.0 s at
4 threads: with 50 runnable threads fighting over 32 cores, a 4-member
barrier rarely gets 4 cores at once, and the wait amplifies. That run used
`ThreadPolicy.Default` (Normal priority, 300 µs spin, P-core pinning) —
strictly harsher than the samples' Game policy — and still argues for the two
conservative halves of the default: cap at 4, and `cores / 2` so small boxes
never field more lanes than they can feed.

Reproduce:

```sh
out/aot/needle bench-e2e ref/hf/needle3.cact tools/tools.json --threads 2,4 --reps 3 --sets golden
```

## Overrides

- Per-sample `Threads` (`1`–`4`) when you know the game saturates cores
  (e.g. a Burst-heavy simulation holding 6 cores on an 8-core console).
- `FrameBudget.FromMilliseconds(4)` caps one call's decode slice per frame
  (see README); prefill stays uncapped.
- `session.Pause()` runs calls inline while workers sleep (loading screens).
- `NEEDLE_SPIN_US` overrides the park delay for diagnostics only.
