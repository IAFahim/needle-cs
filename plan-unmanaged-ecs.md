# Needle.Inference fully unmanaged: the ECS hot path

Status: plan only, written 2026-11-02. Companion to [plan.md](plan.md) (engine, W01–W30) and [plan-netstandard.md](plan-netstandard.md) (fast on every runtime, N00–N83). Item IDs are `Unn`. This file owns **running inference itself inside Unity's ECS world**: Burst-compiled, no managed objects on the hot path, scheduled by an `ISystem`, writing typed commands straight into entities.

## 0. TL;DR

- **Today only the edges are unmanaged.**
  - `NeedleSystem` (Burst `ISystem`) and `NeedleChannel` (lock-free rings) are unmanaged.
  - The engine behind them is managed classes (`NeedleSession`, `EngineCore`, `Pool`, `Tokenizer`, `TokenTrie`, `Grammar`) on its own OS threads.
  - It answers about **1 reply/s for 64 NPCs on 2 threads** (plan.md §2.9).
- **The move:** make the hot path a set of static methods over blittable structs inside the same `Needle.Inference.dll`, so Burst compiles it when an `ISystem` job calls it.
  - Verified: Burst already compiles calls into our precompiled DLL. The W23 `NeedleSystem` calls `TrySubmitSilent`/`TryReceive` with 0 Burst errors (plan.md §10.3).
  - The managed classes become thin wrappers over the same structs. One implementation, two front doors.
- **What it unlocks**, so you can judge whether it's worth it:

  | Unlock | Why | Size |
  |---|---|---|
  | Throughput for many NPCs | every pending NPC is a lane in one batched decode step; decode is weight-bandwidth-bound (17.5 MB per token), so lanes share the stream; the tool-schema prefix KV is stored once and shared by all lanes | 1 → ≥ 8 replies/s at 64 NPCs, 4 cores (target) |
  | Frame stability | Mode B runs on Unity's job workers with a per-frame token budget and a preemption-tolerant claim scheduler; no extra OS threads fighting the job system | p99 frame impact budgeted, ≤ 0.5 ms at 60 fps (target) |
  | Memory per NPC | shared prefix KV plus suffix-only lane KV | 64 MB per session → ≤ 2 MB per lane |
  | Startup | grammar, prefix token ids and the primed prefix KV baked into BlobAssets | 0.3–0.6 s + prefill → ≤ 50 ms |
  | Typed results in ECS | the grammar sink writes `TCommand` into a `DynamicBuffer` in the same job; no channel copy, no main-thread dispatch; early `ToolSelected` intents | same-frame reaction |
  | Lockstep determinism (opt-in) | Burst `FloatMode.Deterministic` (present in this Burst) | bit-identical NPC decisions across x64 and arm64 |
  | Single-request latency | removes managed transitions and fixed-team stalls only | about equal to the dedicated-thread SIMD engine; **not a headline win** |

- **Day-one spike decides how much Burst work is left (gate G1, §3).** Burst 2.0's compiler contains `System.Runtime.Intrinsics` symbols (E1). If it compiles our existing AVX2/NEON kernels, the whole Burst kernel port in plan-netstandard.md Track B (N31–N41) is cancelled.

## 1. Facts, evidence, hypotheses

| ID | Fact | Evidence |
|---|---|---|
| V1 | Burst compiles code reachable in our precompiled DLL | W23: Burst `NeedleSystem` jobs call `NeedleChannel.TrySubmitSilent`/`TryReceive` (net10 DLL), 0 Burst errors in our files (plan.md §10.3) |
| V2 | The Unity package assembly can call DLL internals | `src/Needle/Needle.csproj`: `<InternalsVisibleTo Include="Needle.Unity" />` |
| V3 | In 7000.0.0a12, Burst is built into the engine (`com.unity.burst` 2.0.0, `"type": "shim"`) | `Library/PackageCache/com.unity.burst@*/package.json` |
| V4 | Engine Burst exposes `IsAvx2Supported`, `IsFmaSupported`, `IsBmi1Supported`, `IsPopcntSupported`, `IsNeonDotProdSupported`, `IsNeonRDMASupported`, `IsNeonCryptoSupported`; **no** AVX-512, AVX-VNNI or I8MM checks | `strings Managed/UnityEngine/UnityEngine.BurstModule.dll` |
| V5 | Engine Burst exposes `FloatMode` (incl. `Deterministic`), `Loop.ExpectVectorized`, `AssumeRange`, `Prefetch`, `NoAlias`, `Hint.Likely`, `Constant.IsConstantExpression`, `SkipLocalsInit` | same |
| V6 | Hot-path data is already pointers | `Cq`, `Activations`, `AttentionTile`, `MlpBatch`, `LayerJob`, `GrammarProgram`, `GrammarState`, `NeedleChannel`, `NeedleStats`, `Reply` are blittable; weights, KV, tokenizer tables and the trie live in arenas |
| V7 | The hot-path blockers are managed wrappers | `Model`, `Vocabulary`, `Tokenizer`, `TokenTrie`, `Grammar` (classes holding pointers); `EngineCore` (class, `SlotView[]`, `Observer` delegate, `Pool`); `Pool` (`Thread[]`, `Monitor`); `NeedleSession`/`BatchedSession` (`byte[]`, `int[]`, `CallSpan[]`, `Regex`, `ToolSet`, `CallResult`); static `byte[]` `Grammar.CallHead`/`CallMid`; `static readonly` runtime state `Tiers.Active` (reads environment variables) |
| V8 | Batched decode exists and is a resumable state machine | `BatchedSession.Lane` + `Phase` enum, `BatchedEngine` ≤ 16 slots, `EngineCore.ForwardBatch` with per-row slots |
| V9 | Chunked attention merges partial softmax segments | `Attention.MergePartials` (peaks plus sums per chunk) |
| V10 | 64-NPC baseline | plan.md §2.9: about 1 reply/s on 2 threads, 0 B main-thread allocations, 84 µs incremental main-thread time (Editor debug codegen) |

| ID | Evidence or hypothesis | Settled by |
|---|---|---|
| E1 | Burst 2.0 understands `System.Runtime.Intrinsics`: `Burst.Compiler.IL.dll` contains the strings `System.Runtime.Intrinsics.X86`, `.Arm`, `Vector128`, `Vector256`, `AdvSimd`, `ToVector256Unsafe`. **Evidence only: which intrinsics compile is unknown** | U01 |
| H1 | IL2CPP players in 7000 accept the net10.0 DLL (the managed cold path) | U02 |
| H2 | A Burst function pointer called from our own `Thread` costs < 100 ns per call | U04 |
| H3 | `IJobFor.ScheduleParallel(width)` does not guarantee `width` concurrent workers, so fixed-team barriers can stall inside jobs | U05 |
| H4 | `Stopwatch.GetTimestamp` is not Burst-compatible (use a caller-supplied clock or a Unity timestamp) | U03 |
| H5 | Burst evaluates `static readonly` primitive arrays initialized with constants, but rejects `Tiers.Active`-style runtime initialization | U03 |
| H6 | Deterministic mode costs ≤ 1.3x versus `FloatMode.Strict` | U45 |

## 2. Target architecture

```
                 managed cold path (any thread, may allocate, may throw)
   Model.Open / ModelLoader ──► ModelData* ─┐   ToolSet baker ──► BlobAsset<NeedleToolSetBlob>
                                             ▼        (grammar tables, prefix ids, primed prefix KV)
 ┌──────────────────────── NeedleWorld (unmanaged, one per ECS world) ─────────────────────────┐
 │ ModelData*  ToolSets[]  PrefixKV[] (shared, read-only)  LanePool (state + suffix KV pages)  │
 │ TaskGraph (claim scheduler)  NeedleStats*  per-lane GrammarState / mask cache / sink        │
 └──────────────▲──────────────────────────────▲───────────────────────────────▲───────────────┘
                │ Mode B: job-sliced           │ Mode A: dedicated threads     │ CoreCLR API
   NeedleStepSystem (ISystem, Burst)     Burst FP on Needle threads       NeedleSession / Host
   IJobFor.ScheduleParallel(width)       + JobsUtility.JobWorkerCount     (thin wrappers over
   K tokens or a time budget per frame     partitioning cores             the same structs)
                │
   NeedleGatherSystem: NeedleAsk entities → free lanes
   NeedleDrainSystem: finished lanes → DynamicBuffer<TCommand>, NeedleOutcome, NeedleIntent
```

**One step, shared by every front door:** `NeedleStep.Run(NeedleWorld*, int worker, in StepBudget)`.

- It advances every active lane by one decode step: gather rows, batched forward, per-lane logits (full or sparse), grammar advance and sink, then the phase transition.
- It loops until the budget runs out.
- It is pure Burst-compatible C#: blittable arguments, no allocation, no exceptions (status codes), no statics with runtime state, no delegates.

**Claim scheduler (`TaskGraph`)** replaces fixed-member sharding (`Shard(tasks, member, members)` in `EngineCore` team stages):

- Each stage publishes N tasks. Any running worker claims the next task with `Interlocked.Increment`. The worker that completes the last task runs the stage's serial epilogue and publishes the next stage.
- A worker that finds no task left spins with `Common.Pause` until the stage generation changes.
- **Deadlock-free under any concurrency.** A claimed task is always being executed by a running worker, and one worker can finish a whole step alone. This is what makes Mode B safe inside the job system (H3).
- It also removes today's failure mode where one preempted pool thread stalls every barrier. The same scheduler backs `Pool` on CoreCLR.

**Two execution modes, same code:**

- **Mode A, dedicated threads (latency first):**
  - Needle owns 1–N OS threads that call the Burst-compiled step through a function pointer.
  - Unity gets the remaining cores through `JobsUtility.JobWorkerCount = cores − 1 − needleThreads`.
- **Mode B, job-sliced (frame first, zero extra threads):**
  - One `ScheduleParallel` per frame from `NeedleStepSystem`. The job runs the claim scheduler for at most `TokensPerFrame` steps or `MicrosecondsPerFrame` of wall time.
  - Lanes resume next frame (the V8 state machine). The `JobHandle` may span frames: the world lives outside ECS chunks, so the safety system never forces a sync point.
  - Results are published at the next frame start.

**Shared-prefix paged KV:**

- The tool-schema prefix (125–200 tokens, plan.md §2.7) is computed once per tool set and stored read-only.
- Each lane stores only its suffix KV pages plus a copy of the small recurrent state (conv history, engram history, pooling).
- Attention runs over two segments (prefix, suffix) and merges them with `MergePartials` (V9).
- Per-lane cost is about 4.8 KB per position (20 layers × 240 B) for the suffix only. That is about 1.2 MB for 256 suffix tokens, versus a full private context today.

**What stays managed (cold path, allowed to allocate and throw):**

- `.cact` validation and `Model.Open`.
- Tool JSON parsing and grammar compilation (`ToolSet`, `Grammar` builder), at bake time.
- The CoreCLR string API (`CallResult`, `ToJson`), `AliasTable` and `CallRestorer` for the JSON path.
- The CLI, eval and benchmarks.

## 3. Gate G1: what the day-one spikes decide

| U01: Burst compiles our `System.Runtime.Intrinsics` kernels | U02: IL2CPP players take the net10 DLL | Consequence |
|---|---|---|
| yes | yes | **Best case.** ECS hot path = net10 DLL + Burst on every platform, AVX2/NEON from the existing kernels. Cancel plan-netstandard.md N31–N41. Keep its Track A and C for non-Burst users only |
| yes | no | Players need the ns2.1 DLL, which has no intrinsics, so Burst only sees scalar code. Keep N31–N41, but call the ported kernels **directly** from the step (no function-pointer table inside Burst) |
| no | either | Keep N31–N41 as direct calls. The ECS step uses `#if`-free dispatch: Burst `IsXSupported` branches in a `BurstKernels` static class that the step calls |
| partial (some intrinsics fail) | either | U01 produces the exact list; port only the failing kernels (most likely the VNNI and AVX-512 tiers, which Burst cannot target anyway per V4) |

## 4. Targets

Measured in a **release player**, never Editor debug codegen. Linux x64 4-core reference box first, then one Android arm64 device. The 64-NPC LoadProof scene (plan.md §2.9) and the game eval set are used throughout.

| Metric | Today | Target |
|---|---|---|
| Replies/s, 64 NPCs, 4 cores | about 1 on 2 threads (V10) | ≥ 8 (after U40 + W20) |
| Single-request latency, Mode A | net10 CLI about 100 ms (plan.md §2.2) | ≤ 1.05x of the dedicated-thread SIMD engine on the same box |
| Main-thread time per frame | 84 µs incremental (Editor debug) | ≤ 20 µs, including gather and drain scheduling |
| Managed allocations per frame | 0 B | 0 B, also with Burst disabled |
| Extra OS threads | host thread + pool workers | 0 in Mode B |
| Frame p99 increase while inferring, Mode B, 60 fps | unmeasured | ≤ 0.5 ms at the default budget; configurable |
| Memory per NPC lane | 64 MB private per `NeedleSession` (plan.md §2.5) | ≤ 2 MB |
| Time to first reply after scene load | 0.3–0.6 s constructor + prefix prefill | ≤ 50 ms (baked blobs) |
| Determinism, opt-in | none | bit-identical token streams Linux x64 ↔ Android arm64 on the eval set |
| Correctness | — | ECS path emits token streams identical to `NeedleSession` (same threads-independent kernels) on every eval prompt |

## 5. Work items

Size: S ≤ 1 day, M ≤ 3 days, L ≤ 2 weeks. "Files" bounds what the item may touch.

### Track S: spikes (days 1–2, decide G1)

| ID | Item | Files | Acceptance | Size |
|---|---|---|---|---|
| U01 | **Burst × our kernels.** In a throwaway asmdef, a `[BurstCompile]` job calls internal `Quant.MatMulRows`, `Quant.Prepare`, `Attention.Run` and `HadamardMlp.Apply` (via V2) on real weights. Record every Burst error; read the Burst Inspector disassembly for `vpmaddubsw`/`vpshufb` (x64) and `sdot`/`tbl` (arm64); compare outputs with the JIT bitwise | `unity/spikes/BurstKernels/*` (deleted after) | A list per kernel: compiles yes/no, vectorized yes/no, bit-identical yes/no, blocking construct named. Settles E1 | S |
| U02 | **IL2CPP × net10 DLL.** Build Linux x64 IL2CPP and Android arm64 IL2CPP players with the Squad sample on the net10 DLL; log tier, `FrameworkDescription`, one reply | build scripts only | Each player runs or the exact IL2CPP error is recorded. Settles H1 | S |
| U03 | **Burst reachability audit.** A minimal struct port of one layer (`ExecuteLayer` stage functions over `LayerJob*`) called from Burst; record every rejected construct (statics, `Stopwatch`, managed arrays, exceptions, `delegate*<>` managed function pointers) | spike files | Blocker list with the fix pattern per construct. Settles H4, H5 | S |
| U04 | **Burst FP from our threads.** Call a Burst-compiled static entry from a plain `Thread` 10⁶ times; measure per-call cost and confirm no Unity-thread requirement | spike | Settles H2 | S |
| U05 | **Job-system concurrency.** `IJobFor.ScheduleParallel(width = workers)` with a claim scheduler over empty tasks, while a Burst-heavy Entities scene runs; measure actual concurrent workers, per-stage overhead, and compare with `Pool.RunTeam` barriers (1.24 µs, plan.md §2.3) | spike | Overhead per stage ≤ 2 µs and no stall when only 1 worker is free. Settles H3 | S |

### Track C: the core becomes unmanaged structs (in `Needle.Inference.dll`)

Every item keeps the public CoreCLR API and all of plan.md §7's gates green. The classes become wrappers.

| ID | Item | Files | Depends | Acceptance | Size |
|---|---|---|---|---|---|
| U10 | **Burst-compatibility rules and a gate.** `[HotPath]` marker attribute; a test that scans IL of every `[HotPath]` method for `newobj` of reference types, `box`, `throw`, managed `ldsfld` of non-readonly or computed statics, `callvirt` on reference types and managed `delegate*` calls; plus a Unity EditMode test that Burst-compiles one entry per hot module | `src/Needle/HotPath.cs`, `tests/Needle.Tests/HotPathTests.cs`, package `Tests/` | U03 | Fails on a deliberately bad method; passes on the ported set | M |
| U11 | **Tier selection without runtime statics.** Replace `Tiers.Active` (`static readonly`, environment variables) on the hot path with direct `IsSupported` checks, which fold to constants in the JIT and in Burst; environment overrides move into an explicit `KernelOverrides` struct that only the managed path reads | `Tiers.cs`, `Scalar.cs` (`Simd`), call sites in `Ops.cs`/`Quant.cs`/`Blocks.cs`/`Gemv.cs` | U03 | `ForcedTierTests` still pass through `KernelOverrides`; no perf change on net10 (`bench` within noise) | M |
| U12 | `ModelData` struct holding everything `Model` holds (geometry, weights, heads, audio pointers, tokenizer blob); `Model` wraps a `ModelData*` | `Model.cs`, `ModelInfo.cs` | U10 | `LoadingTests`, `FormatTests` green; `ModelData` blittable (size assert) | M |
| U13 | `TokenizerData` (shared tables) plus a `TokenizerScratch` per lane; `static int Encode(TokenizerData*, TokenizerScratch*, byte*, int, int*, int capacity, bool leadingSpace)`; `Tokenizer` wraps | `Tokenizer.cs`, `Vocabulary.cs` | U10 | Encodes byte-identical ids on the tokenizer tests; no `Span` or `ReadOnlySpan` in the static entry | M |
| U14 | `TrieData` (shared) plus `MaskCache` per lane (the 64-slot cache, walk states); `static void Allow(...)` | `TokenTrie.cs` | U10 | `MaskedDecodingTests` green | S |
| U15 | Grammar runtime data without managed arrays: `CallHead`/`CallMid` become constant byte sequences (a `fixed` buffer in a static struct, or a switch on length); `Advance` stays generic over `IGrammarSink` | `Grammar.cs` | U10 | `GrammarTests` green; U10 gate passes for `Advance`, `ForcedBytes` | S |
| U16 | **Engine state as structs.** `EngineData` (shared scratch per worker) and `SlotState` (per lane: recurrent state, snapshot, suffix KV pointers) replace the `SlotView[]` and class fields; the batched forward becomes `static void ForwardBatch(EngineData*, SlotState*, ...)`; `Observer` stays on the managed wrapper only | `Engine.cs`, `BatchEngine.cs` | U12, U18 | `BatchedDecodeTests`, `ChunkParityTests`, `ParityTests` green; batched outputs bit-identical to today | L |
| U17 | **Lane state machine as structs.** Port `BatchedSession.Lane` and its phases to `LaneState` (no `ToolSet`, `Grammar` class, `byte[]`, `CallResult`); typed output through `ICallSink<TCommand>` into caller memory; JSON output stays in the managed wrapper; `NeedleStep.Run` drives lanes | `BatchedSession.cs`, new `Hosting/NeedleStep.cs`, `Hosting/LaneState.cs` | U13–U16 | `BatchedDecodeTests` green; a new test proves `NeedleStep` token streams equal `NeedleSession` per prompt | L |
| U18 | **Claim scheduler `TaskGraph`.** Stage publish, atomic claim, last-finisher epilogue, generation spin with pause; used by `Pool` (managed threads) and by jobs; replaces `Shard`-by-member in team stages | `Pool.cs`, new `TaskGraph.cs`, `Engine.cs` (team stage calls) | U10 | Correct with 1..N workers including 1-of-N available; `pool_run_barrier`-equivalent ≤ today; e2e within noise on net10; a test that preempts one worker (sleep inside a task) shows no full-step stall beyond that task | M |
| U19 | **Status codes on the hot path.** Every hot `throw` becomes a `FailureReason` (the enum already covers context exhaustion and budgets); cold-path validation keeps exceptions | `Engine.cs`, `NeedleSession.cs`, `BatchedSession.cs` | U16, U17 | U10 gate shows no `throw` in `[HotPath]` methods | S |
| U1A | **Clock and budget.** `StepBudget { MaxTokens, MaxTicks }` plus a caller-supplied `long* now` source (Unity: a Burst-safe timestamp, verified in U03; CoreCLR: `Stopwatch`); `FrameBudget` maps onto it | `ThreadPolicy.cs`, `NeedleStep.cs` | U03 | Budget respected within one decode step | S |
| U1B | **One implementation.** `NeedleSession`, `BatchedSession` and `NeedleHost` delegate to `NeedleStep` + structs; delete the duplicate decode loops (`Run`, `RunTyped`, `GenerateCalls`, `GenerateCallsTyped`, lane `Advance*`) | `NeedleSession.cs`, `BatchedSession.cs`, `Hosting/NeedleHost.cs` | U17 | Full test suite green; line count of decode logic roughly halves | L |

### Track E: ECS integration (Unity package)

| ID | Item | Files | Depends | Acceptance | Size |
|---|---|---|---|---|---|
| U30 | **`NeedleWorld` singleton and lifecycle.** A managed bootstrap (`ModelLoader` → `ModelData*`) creates the world; an unmanaged singleton component holds `NeedleWorld*`; disposal on world destroy and domain reload; leak check via `UnsafeUtility` leak detection | package `Runtime/Ecs/NeedleWorld*.cs` | U16, U17 | Enter/exit play mode 100 times with 0 leaks and no crash | M |
| U31 | **Mode B: `NeedleStepSystem`.** One `IJobFor.ScheduleParallel` per frame; `TokensPerFrame` and `MicrosecondsPerFrame` budgets on a singleton config; the handle may span frames; publish at the next frame start | package `Runtime/Ecs/NeedleStepSystem.cs` | U18, U1A, U30 | LoadProof runs with 0 extra threads; frame p99 within the budget target (§4) | M |
| U32 | **Mode A: dedicated Burst threads.** Needle threads call the Burst function pointer for `NeedleStep.Run`; `JobsUtility.JobWorkerCount` partitions cores; switchable at runtime against Mode B | package `Runtime/Ecs/NeedleThreads.cs` | U04, U30 | Single-request latency target (§4); clean shutdown | M |
| U33 | **Generator for ECS.** Per `[NeedleTools]` interface emit: an `IBufferElementData` wrapper of `TCommand`, a concrete gather/drain system (no open generic `ISystem`), `RegisterGenericComponentType`/`RegisterGenericJobType` attributes where generics remain, and a Baker that produces `BlobAssetReference<NeedleToolSetBlob>` (grammar tables, prefix token ids). C# 9-safe output (plan-netstandard.md N02) | `src/Needle.Generators/*`, package `Runtime/Ecs/Blob*.cs` | N02, U17 | Both samples work from the interface alone; no hand-written ECS glue per tool set | L |
| U34 | **Typed results into entities.** The drain writes `TCommand` records into the NPC's `DynamicBuffer`, plus `NeedleOutcome { Status, Confidence, CallCount }`; the grammar sink's `ToolSelected` raises an enableable `NeedleIntent { Tool }` before the arguments finish | package `Runtime/Ecs/*`, generated code | U31, U33 | Squad and Npc samples act on buffers; an intent arrives ≥ 1 frame before its outcome on multi-token calls | M |
| U35 | **Baked primed prefix.** An Editor bake runs the prefix once per tool set and stores the KV and recurrent state blob with a hash of model, tool set, system prompt and context (the existing `ExportPrimedPrefix` header rules); `AutoDate` must be off (plan.md §10.5) | package `Editor/NeedlePrefixBaker.cs`, `Engine.cs` export/import | U16, U33 | Time to first reply ≤ 50 ms; a hash mismatch rebakes instead of loading | M |
| U36 | **Text in, without managed strings.** `NeedleAsk.Query` (`FixedString512Bytes`) is tokenized inside the step (U13); UI code converts once at the boundary | package `Runtime/Ecs/*` | U13, U31 | 0 B per ask on the main thread | S |
| U37 | **Profiling.** `ProfilerMarker`s per stage inside Burst; `NeedleStats` (already blittable) surfaced through a system; LoadProof prints replies/s, p50/p99 latency, lanes busy, tokens per frame | package `Runtime/NeedleProfiler.cs`, `Runtime/Ecs/*` | U31 | Markers visible in a player capture | S |

### Track P: performance

| ID | Item | Files | Depends | Acceptance | Size |
|---|---|---|---|---|---|
| U40 | **Shared-prefix paged KV.** Prefix KV stored once per tool set; lane KV holds suffix pages only; attention over two segments merged with `MergePartials`; recurrent state copied from the prefix snapshot at lane start | `Engine.cs`, `Blocks.cs` (`AttentionBatch` two-segment plan) | U16 | Bit-identical or parity-bound vs single-segment; memory per lane ≤ 2 MB; 64 lanes fit in ≤ 160 MB total | L |
| U41 | **Lane policy.** Raise the 16-slot cap (row batches > 16 split into multiple `ForwardBatch` calls); priority and fairness (oldest-first plus a distance or importance hint from the game); admission control when lanes are full | `NeedleStep.cs`, `LaneState.cs` | U17 | No starvation at 64 NPCs; p99 latency reported per priority class | M |
| U42 | **Batched GEMM cost.** Depends on W20 (T = 2..8 at ≤ 1.3x of T=1). Without W20, batching gains are capped by today's 1.9–2.8x (plan.md §2.3) | (W20) | W20 | Replies/s target in §4 | — |
| U43 | **Fused layer per worker.** The whole `ExecuteLayer` stage list as one Burst call per worker per layer, stages driven by `TaskGraph`; supersedes plan-netstandard.md N40 | `Engine.cs`, `TaskGraph.cs` | U16, U18 | Decode µs/token ≥ 10% better than per-stage dispatch, or closed with numbers | M |
| U44 | **Burst-only knobs, proven.** `Loop.ExpectVectorized()` in every row-op loop (a compile-time proof that fails the build when vectorization is lost); `[NoAlias]` only where provably true (`AttentionTile.Output` aliases partials in the chunked path); `Hint.Assume`/`AssumeRange` on loop bounds; `Common.Prefetch` distance tuned per platform; `OptimizeFor.Performance`; safety checks off in release | hot-path files (behind `#if` for the Burst attributes, or via the package wrapper) | U01, U16 | Every annotated loop proven vectorized; decode µs/token delta reported per knob | M |
| U45 | **Deterministic mode (opt-in).** `FloatMode.Deterministic`; our own polynomial `Exp` everywhere on the hot path (no `MathF.Exp` in decode); a golden token-stream test across Linux x64 and Android arm64 | `Ops.cs`, `Scalar.cs`, package config | U01, U16 | Bit-identical streams on the eval set across both platforms; cost ≤ 1.3x (H6) | M |

### Track V: voice (after Track E)

| ID | Item | Files | Depends | Acceptance | Size |
|---|---|---|---|---|---|
| U50 | Whistle encoder and greedy decoder as structs over the same scheduler; microphone → `NativeArray<float>` ring → transcript bytes → `NeedleAsk` in the same frame pipeline; beam search stays managed until proven needed | `Whistle*.cs`, package `Runtime/Ecs/Voice*.cs` | U16, U18, U31 | Push-to-talk to typed command with 0 managed allocations per utterance; transcript equal to `TranscribeUtf8` | L |

### Track Q: quality gates

| ID | Item | Files | Depends | Acceptance | Size |
|---|---|---|---|---|---|
| U60 | Equivalence: the ECS path, `NeedleSession` and `BatchedSession` emit identical token streams for every eval prompt at 1, 2 and 4 workers | `tests/Needle.Tests/*`, package `Tests/` | U17, U31 | 0 mismatches | M |
| U61 | Release-player LoadProof: 64 NPCs, 10 minutes, Mode A and Mode B; frame p50/p99, replies/s, memory | package `Samples~/LoadProof` | U31, U32 | Every §4 row filled with raw JSON attached | M |
| U62 | Soak: 1 hour with play-mode cycling and NPC churn; 0 leaks, flat memory, no stalls longer than the budget | same | U61 | Report attached | S |

## 6. Dependency graph and lanes

```
U01 U02 U03 U04 U05 ── G1 ──┐
                            ▼
U10 ─┬─ U11
     ├─ U12 ─┐
     ├─ U13 ─┤
     ├─ U14 ─┼─ U16 ─ U17 ─┬─ U19 ─ U1B
     ├─ U15 ─┘     ▲       │
     └─ U18 ───────┘       ├─ U30 ─┬─ U31 ─┬─ U34 ─ U36 ─ U37
        U1A ───────────────┘       └─ U32  │
                                    U33 ───┘ (with N02)    U35 (after U16, U33)
U40, U41, U43 after U16/U17;  U44, U45 after U01 + U16;  U50 after U31;  U60–U62 last
```

**File ownership:**

| File | Sequence |
|---|---|
| `Engine.cs` | U16 → U18 (team-stage call sites) → U40 → U43 |
| `Pool.cs`, `TaskGraph.cs` | U18 only |
| `Tiers.cs`, `Scalar.cs` (`Simd`) | U11, then U45 (`Exp` usage) |
| `BatchedSession.cs`, `NeedleStep.cs`, `LaneState.cs` | U17 → U19 → U1B → U41 |
| `NeedleSession.cs` | U19 → U1B (coordinate with plan-netstandard.md N61/N62) |
| `Blocks.cs` | U40 (two-segment attention) only, after plan-netstandard.md N30 if that lands first |
| `src/Needle.Generators/*` | plan-netstandard.md N02 → U33 |
| package `Runtime/Ecs/*` | one file per U3x item |

## 7. Agent protocol

plan.md §8 and plan-netstandard.md §6 apply. On top of them:

1. **No hot-path PR without the U10 gate.** New code on the step path carries `[HotPath]` and must pass the IL scan and the Unity Burst compile test.
2. **Equivalence before speed.** Track C items must keep token streams bit-identical with today's sessions. Speed claims come only after U60 is wired for the item's area.
3. **Release players for numbers.** Editor-debug measurements may guide work but never fill §4 (plan.md §2.9 shows Editor debug codegen at about 6 fps).
4. **Spikes are throwaway.** U01–U05 live under `unity/spikes/` and are deleted once their findings are moved into §1 of this file.
5. **Swarm shape.** The critical path is U03 → U10 → U16 → U17 → U30 → U31, roughly 4–6 weeks of serial work. Up to about 12 implementation items run at once after U10. Put extra agents on U60 equivalence fuzzing, device runs for U02/U45/U61, and instruction-level review of U01's disassembly.

## 8. Risks and open questions

1. **Burst and generics.** The step is generic over `TSink`/`TCommand` per tool set. Burst needs closed instantiations reachable from a concrete job. U33 must emit a concrete job or system per tool set; open generic systems are not an option.
2. **Job-handle lifetime across frames (Mode B).** If Entities or the safety system forces completion at sync points, the slice must finish within a frame. Then `MicrosecondsPerFrame` becomes a hard cap, not a hint. U05/U31 measure this.
3. **Unity alpha churn.** Burst is engine-built-in in 7000 (V3), and the December CoreCLR release may change Burst, IL2CPP or the compile profile. G1 must be re-run on each new Unity alpha (keep U01/U02 as a CI job, not just a one-off spike).
4. **Two front doors, one implementation.** U1B deletes duplicated decode loops. Until it lands, every decode fix must go into both the struct path and the class path; that is the main reason to keep U1B early.
5. **Determinism scope.** Deterministic mode covers Burst-compiled code. The managed cold path (bake) runs once and its outputs are blobs, so it is outside the determinism boundary by design.
6. **Is it worth it for one NPC?** No, not for latency (§0 table, last row). The case rests on many NPCs, frame stability, memory and startup. If the game has one assistant and no crowds, stop after Track S and plan-netstandard.md N00.
