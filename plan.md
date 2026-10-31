# Needle.Inference: game hot-path plan

Status: plan only. Nothing here is implemented yet. Every work item (`Wnn`) has a GitHub issue, so people and local AI agents can pick items up independently. The tracking issue is [#31](https://github.com/IAFahim/needle-cs/issues/31) (learnings plus checklist); work items are [#1](https://github.com/IAFahim/needle-cs/issues/1)–[#30](https://github.com/IAFahim/needle-cs/issues/30).

Last measured: 2026-10-29, on the reference box in [2.2](#22-baseline). The numbers come from instrumented copies of the engine and are reproducible with the harness in W02.

## 0. TL;DR

- **Goal.** `Needle.Inference` becomes the hot path that games call for tool calling, extraction, embeddings and voice to tool call.
  - Zero allocations per call and per frame.
  - Safe to use inside a frame.
  - Attaches to Unity's CoreCLR/.NET 10 release (called "Unity 7" here) through a thin Entities `ISystem` layer.
  - Game code calls **typed C# methods**, not JSON.
  - **Faster than the upstream native C++ engine** on the same machine.
- **Today.** C# runs at parity with native: 92–111 ms per tool call on 4 threads, versus 95–129 ms for upstream `libneedle3.so`.
  - Decode costs 2.03 ms per token. One third of that is the Hadamard MLP running on a single thread with a latency-bound loop.
  - Each token reads 17.5 MB of weights at an effective 9 GB/s.
- **Target.** At most 25 ms per tool call on 4 threads (about 4x faster than native) and at most 70 ms on 1 thread. 0 B allocated per call. The parity suite stays green.
- **How.** Five measured levers, in order of payoff:
  1. Fix the MLP: register blocking and parallel execution (−0.6 ms/token).
  2. Make the matmuls stream at bandwidth (−0.5 ms/token).
  3. Run attention per GQA group (−0.15 ms/token).
  4. Remove serial stages and most of the 86 barriers per token (−0.25 ms/token).
  5. Use jump-forward and lossless prompt-lookup speculation to cut sequential model passes by 46%.
- **Game blockers found.**
  - `Model.Open` is Linux-only, so Windows fails.
  - `new Pool()` pins the calling thread.
  - Workers spin about 10 ms after each request.
  - Every call allocates.
  - The only async API is `Task`-based.
  - The Unity sample is a managed `SystemBase`.
- **Out of scope.** Training, fine-tuning, quantization, export, the `.cact` format, the tokenizer and the model architecture belong to Cactus Compute. We consume their published `.cact` files.

## 1. Scope

| In scope (this repo) | Out of scope (Cactus Compute: `cactus-compute/needle`) |
|---|---|
| Runtime inference of Needle 3 and Whistle `.cact` files | Training, fine-tuning, LoRA, distillation |
| Kernels, scheduling, decoding, grammar, sessions | Quantization recipes, `.cact` export and format design |
| Typed C# tool binding (source generator), zero-allocation API | Model architecture, checkpoints, tokenizer |
| Unmanaged request/reply channel and host worker | Python package, WebAssembly, native C++ engine |
| Unity Entities `ISystem` glue (Unity CoreCLR / .NET 10) | Model accuracy benchmarks |
| Benchmarks, parity tests, CI | |

**Licensing.**
- `needle3` and `whistle` weights are Apache-2.0 (Hugging Face model cards), and so is `cactus-compute/needle`.
- `cactus-compute/cactus`, the C++ engine, uses a custom license: commercial users above $2M funding or revenue need a license. **Never copy code from it.** Work clean-room from `tools/golden.py` and the Apache-2.0 Python architecture.
- This repo has no `LICENSE` yet (W04).

## 2. Learnings

### 2.1 Reproducing the suite

- The projects target `net10.0`, so you need the .NET 10 SDK. The 8.0 SDK fails with `NETSDK1045`.
- `tools/fetch.sh` downloads the model, tokenizer, upstream wheel, Whistle and the audio clips.
- `python3 tools/golden.py` needs `numpy` and `sentencepiece` and takes about 15 minutes of CPU time; it recomputes the full forward pass for every token.
- **Fixture drift (fixed in W01):** the decode budget is now defined once as `DECODE_BUDGET = 40` in `tools/golden.py`, written into `golden.json` as `budget`, and read by `ParityTests`. Fixtures generated before W01 lack the `budget` key and must be regenerated (or patched by adding `"budget":40`).
- `ConfidenceHeadMatchesReference` also needs `python3 tools/golden_confidence.py`. The README mentions neither step.
- After that, 54 of 54 tests pass in Release on AVX2 and with `NEEDLE_SCALAR=1` (2 min 16 s).
- C# greedy decoding matches the numpy reference exactly at 1 and 8 threads.

### 2.2 Baseline

Reference box: 4-core Intel Xeon VM (Sapphire Rapids class, 2.1 GHz); AVX2, FMA, AVX-VNNI, AVX-512 F/BW/VL/VNNI/VBMI/BF16/FP16, AMX; L2 2 MB per core; L3 260 MB. NativeAOT build of the CLI. Query: "what's it like in Lagos right now?" with `tools/tools.json`.

| Engine | Threads | Prefill (18 tok) | Decode | Total per call |
|---|---|---|---|---|
| C# `needle run` (greedy, no grammar) | 1 | 37–39 ms | 2.8–3.5 ms/tok | 152–197 ms |
| C# `needle run` | 2 | 24 ms | 2.2–2.8 ms/tok | 116–150 ms |
| C# `needle run` | 4 | 14.5–15 ms | 1.75–2.03 ms/tok | 92–105 ms |
| C# `needle call` (grammar and confidence) | 4 | 15.8 ms TTFT | 415–440 tok/s | 111 ms |
| Upstream `libneedle3.so` 3.3.3 | default | 710 tok/s | 500–570 tok/s | 95–108 ms (Lagos), 118–129 ms (timer and music) |

### 2.3 Where the time goes

**Bytes per decoded token**, exact from the `.cact` record table:

| Part | Bytes |
|---|---|
| 20 layers x (Q 576, K 96, V 128, Gate 768, Out 768 rows) x 768, 2-bit | 9.53 MB |
| Hyper-connection phi, 24 x 3072 per layer, 4-bit | 0.76 MB |
| Engram key/value, 5 sites x 2 x 768², 2-bit | 1.57 MB |
| Head (tied embedding 8192 x 768, 4-bit) | 3.24 MB |
| Dense fp16 parameters, held as fp32 (MLP factors, conditions, norms, taps) | 2.43 MB |
| **Total** (the engram tables are 18.8 MB, but only about 1 KB is gathered) | **17.5 MB** |

**Decode profile per stage, µs/token** (40 greedy tokens after an 18-token prefill, best of 9):

| Stage | 1 thread | 4 threads | Share at 4 | Note |
|---|---|---|---|---|
| Hadamard MLP (`MlpBatch`) | 589 | **675** | 33% | one task per token, so serial at T=1; does not scale |
| Attention Q/K/V/Gate matmul | 868 | 374 | 18% | 6.4 MB in 374 µs is 17 GB/s |
| Attention out matmul | 420 | 215 | 11% | 3.1 MB in 215 µs is 15 GB/s |
| Attention core (12 heads, int8 KV) | 389 | 202 | 10% | each head re-reads and re-converts its KV head (GQA 6:1) |
| Engram (5 sites) | 251 | 137 | 7% | |
| Head logits 8192 x 768 | 281 | 102 | 5% | 32 GB/s, scales well |
| Hyper: rms, fakeA8 and `Prepare(3072)` | 66 | 77 | 4% | serial |
| Hyper: phi matmuls (24 rows) | 65 | 65 | 3% | serial, on the calling thread |
| Hyper: sigmoid, mix and Sinkhorn | 28 | 29 | 1% | serial |
| Other attention elementwise (norm, conv taps, RoPE, quantization, gate) | 84 | 120 | 6% | |
| Combine lanes, embedding, other | 28 | 31 | 2% | |
| **Total** | **3072** | **2032** | | **86 `Pool.Run` barriers per token** |

Prefill of 18 tokens takes 37.05 ms on 1 thread and 15.06 ms on 4. At 4 threads the biggest parts are the MLP (3.39 ms), Q/K/V/Gate (2.14 ms), the attention core (1.94 ms), out (1.25 ms) and engram (1.22 ms).

**Microbenchmarks:**

| Operation | Cost | Consequence |
|---|---|---|
| `Pool.Run` at 4 threads, empty tasks | 1.24 µs (4 tasks), 2.31 µs (12 tasks) | 86 per token is about 100–200 µs per token |
| `HadamardMlp.Apply` for one token and layer | 30.9 µs | about 210K FMAs at about 20% of AVX2 peak. `Kron` runs one dependent FMA chain per output vector, so it is latency-bound. Its 4 KB `stackalloc` is zeroed (no `SkipLocalsInit`) |
| `Sinkhorn.Normalize` (4x4, 20 iterations) | 0.79 µs | 16 µs per token; vectorizable to about 0.1 µs |
| Softmax over 8192 logits | 27.3 µs | `ArgMaxMasked` does this every step, even in reasoning, where the probability is discarded |
| Scalar `ArgMax` over 8192 | 4.4 µs | vectorizable to under 1 µs |
| Hyper `Prepare(3072)` and phi (24 rows) per layer | 3.79 + 2.47 µs | 125 µs per token, serial |
| `TokenTrie.Allow` per call step | 47.8 µs | about 12% of a 0.4 ms target step |
| GEMM cost at T=4 versus T=1 (`needle bench`) | 1.9–2.8x | speculation and batching only pay off once T=2..8 costs at most 1.3x |

**Savings and the floor we should reach at 4 threads:**

| Fix | Saves (µs/token) |
|---|---|
| Register-blocked, parallel MLP (W12) | about 630 |
| Matmuls at 40 GB/s or more (W19) | about 480 |
| GQA-shared attention (W14) | about 150 |
| Parallel or fused hyper stages and fewer barriers (W13) | about 250 |
| Fused elementwise | about 90 |
| **Floor** | **about 0.43 ms per token** (versus 2.03 ms) |

### 2.4 Token statistics: where decode steps can be skipped

Setup: 14 prompts (4 with the default 3 tools, 10 with a 7-tool game set); 636 generated tokens.

| Phase | Tokens | Fully grammar-forced | Steps with ≤ 16 allowed tokens | Steps with exactly 1 allowed token |
|---|---|---|---|---|
| Reasoning (`<think>`) | 342 (54%) | 0 | 0 | 0 |
| Decision (call or respond) | 14 | 0 | 0 | 0 |
| Call JSON | 280 (44%) | 153 (55%) | 189 (68%) | 0 |

- **Prompt lookup.** For 38.5% of all tokens (245 of 636), looking up the last 1–3 tokens in the prompt or previous output predicts the next token correctly.
- **Sequential model passes needed:**

| Strategy | Passes | Change |
|---|---|---|
| Baseline | 636 | |
| Jump-forward over forced spans | 483 | −24% |
| Lossless prompt-lookup speculation, k=4 | 408 | −36% |
| Both | 342 | −46% |
| Both, k=8 | 331 | −48% |

- **No step ever has exactly one allowed token.** The trie admits every tokenization of a forced span, so "skip logits when the mask is a singleton" never fires. Jump-forward needs grammar-level forced-byte detection.
- **The model misroutes some game commands.** "give Lyra 3 potions" produced `say({"text":"Lyra"})`, so tool-design guidance, confidence gating and an evaluation suite matter (W28).

### 2.5 Memory and allocations

- `Model.Open` adds 45 MB of RSS (the 35 MB mmap plus fp32 copies of fp16 dense tensors).
- Each `NeedleSession` commits about 64 MB of private memory (RSS +12–18 MB). The constructor takes 300–640 ms on 1 thread and rebuilds model-level data, the tokenizer and the trie, for every session. Arenas are sized for 64 KB of input and 64-token chunks.
- One warm `NeedleSession.Complete` allocates about 6.2 KB of managed memory and still mallocs native buffers (`Engine.ConfidenceLogit`, `TokenTrie.Walk`).

### 2.6 Zero-alloc follow-up measurements (F08, #40)

Indicative numbers from the loaded 32-core dev box (JIT, avx2); latency is too noisy there to conclude anything, allocation counts are exact.

- Full-vocabulary softmax over 8192 floats — what the legacy `ArgMaxMasked` ran on every reasoning and response step — costs best 36.2 µs / median 37.2 µs (9 rounds). Same order as the 27 µs reference-box figure; the new masked argmax skips it outside call generation.
- `GrammarState` is 392 B, so the pre-#7 per-step `Mem.Alloc`/`Mem.Free` in `TokenTrie.Walk` was 65 × 392 = 25,480 B (~24.9 KB) per call-generation step; it is now one preallocated per-session buffer.
- `bench-e2e --api span` with `-p:NeedleAllocCounters=true` (1 thread, golden set): median 0 managed B and 0 native allocs per call on both prompts; `--api text` on the same build measures 664 B (Lagos) and 800 B (timer+music) managed per call, so the string wrapper's only per-call cost is the `CallResult` itself.
- `LD_PRELOAD` probes (`tools/malloc_counter.c`, 20 warm + 100 measured span calls): `plain:1` → 0 managed / 0 mallocs; `retrieval:1` → 0 / 0; `plain:4` → 0 / 0 in 3 of 4 runs, with one run showing a single stray libc malloc from CLR background activity (Mem accounting stayed 0 throughout).
- End-to-end latency on the loaded box swings run to run (identical Lagos call: 55–178 ms), so no speedup claim is made here; re-measure on an idle box with `bench-e2e` per #39.

### 2.7 Platform and hosting findings

- **`Model.Open` is Linux-only on net10.0.** It calls `mmap` from `"libc"` with `MAP_PRIVATE|MAP_POPULATE` and passes the `SafeFileHandle` value as an fd. That fails on Windows, and the flag means something else on macOS.
- **`new Pool(n)` pins the calling thread** to the first fast core (`sched_setaffinity(0, …)`). Creating a session on the main thread pins the main thread.
- **Workers spin for about 10 ms** (40,000 x `SpinWait(4)`) before they park, after every request.
- Thread placement exists only on Linux. There is no handling for Windows hybrid cores (CPU sets, priorities), macOS QoS or Android big.LITTLE, and no pause or resume.
- `Geometry.KvWindow` (`kv_window` = 256) is parsed but never used.
- `AutoDate` defaults on and adds a wall-clock line to the prefix (141 tokens instead of 125). With `AutoDate=false` the prefix is identical to native `needle_init` with an empty system prompt.

### 2.8 .NET 10 SIMD facts and platform tiers

- .NET 10 has:
  - `AvxVnni`: VEX encoding, 128/256-bit `VPDPBUSD`.
  - `AvxVnniInt8`/`AvxVnniInt16`: signed x signed, including `V512`, on AVX10.2-class CPUs.
  - `Avx512BW/F/VL/VBMI/VBMI2` and `Avx10v1/v2`.
- **It has no 512-bit `VPDPBUSD`.** A 512-bit int8 dot product costs `VPMADDUBSW + VPMADDWD + VPADDD`.

| Platform | Features | Path today | Opportunity |
|---|---|---|---|
| Intel 12th–14th gen | AVX2 + AVX-VNNI, no AVX-512 | AvxVnni-256 | multi-row GEMV, scheduling |
| AMD Zen 3 | AVX2 | Avx2 3-op path | same |
| AMD Zen 4 | AVX-512 with EVEX VNNI, **no VEX AVX-VNNI** | Avx2 3-op path | Avx512BW path, about 2x |
| AMD Zen 5 | AVX-512 + AVX-VNNI | AvxVnni-256 | measure Avx512BW |
| Steam Deck, PS5, Xbox Series (Zen 2) | AVX2 | Avx2 | same as Zen 3 |
| Apple Silicon, recent Android, Switch-class ARM | NEON, dotprod (i8mm on newer cores) | Neon SDOT | i8mm tier; verify per device |

### 2.9 Unity W23 proofs (2026-10-30, in-Editor 7000.0.0a12, Linux x64)

- **64-NPC load (LoadProof harness, 1.2 s cadence ≈ 50 asks/min/NPC, 120 warmup + 600 measured frames, two runs):** main-thread managed allocations in the per-frame NPC pump bracket: **0 bytes, 0 nonzero frames** both runs. 38–39 replies per run, 100% carrying dispatched tool calls (`MoveTo`/`GiveItem`); 64 asks pending at end (engine does ~1 reply/s on 2 threads vs ~53 asks/s offered — saturation is expected, the Unity side stays flat).
- **Main-thread timing (PlayerLoop timestamp markers around `SimulationSystemGroup`, same runs):** 293.8 µs mean / 595.7 µs max with 64 asks pending; 210.4 µs / 711.9 µs max idle floor (load disabled, system skipped); incremental ≈ 84 µs. The ≤ 20 µs criterion is NOT met in this environment (Editor-debug codegen with safety checks, ~6 fps, inference workers saturating the 4-core box). Method notes: the headless Editor runs with the Unity Profiler disabled and `Profiler.enabled = true` throws NRE, so `Recorder`/`ProfilerRecorder` capture nothing; Entities 1.5 groups expose no `PlayerLoop` children, so no per-system split was obtainable — re-measure in a release player before treating 84 µs as real work.
- **Voice end to end in PlayMode** (`whistle.cact` + `test.wav` from StreamingAssets, `PushToTalk` streaming + `Release`): full 41-word transcript, not truncated. 4-thread `Transcribe` matches upstream exactly ("creating a way of file", "Wi-Fi"); the sample default 1-thread lands on the known beam knife-edge ("grading away a file"). So the transcript is **thread-count-sensitive** — `ThreadCountDoesNotChangeTheTranscript` only pins `record.wav`, not the knife-edge clip; evidence for #38, not a W23 problem. The sample is unchanged (still 1 thread).
- **Repro:** the harness lives in the test project only (`Assets/LoadProof/LoadProof.cs`, dormant `LoadProof` GameObject in `NeedleTest.unity`); rerun = enter PlayMode, read the `LOADPROOF` console line.

## 3. Engineering rules

1. **No comments.** Names, types and data layout carry the meaning. A stranger should learn the domain, not the author's mood.
2. **Data-oriented design.** Use blittable `readonly struct`s, structs of arrays, pointer plus length, and arenas. No object graphs on the hot path.
3. **Pure kernels.** Kernels are static functions from inputs to outputs. Effects live only in the engine and host shells.
4. **Immutable by default.** Use `readonly struct`, `in` parameters, and configuration resolved once (tier, topology).
5. **Total functions on the hot path.** Report results through status enums. Exceptions only at construction and I/O boundaries.
6. **Deterministic.** Identical inputs, tier, thread count and chunking produce identical bits.
7. **No implicit allocation.** 0 managed bytes and 0 native `malloc`s per call after warm-up, enforced by tests.
8. **Every abstraction earns its name.** Deleting a feature shrinks the code without breaking it.
9. **Measure every claim.** Report before and after on the same machine, interleaved A/B, best of at least 5, using the W02 harness.

## 4. Target architecture

```
Unity (CoreCLR/.NET 10)                          Needle.Inference (Unity-agnostic)
-----------------------------                    ----------------------------------------------------------
NpcCommand buffers / NeedleAsk components
NeedleSystem : ISystem [BurstCompile] --TrySubmit/TryReceive-->  NeedleChannel: unmanaged bounded MPMC rings of
     (blittable structs only, no managed calls)                  blittable requests and typed replies, tickets
NeedleBootstrap (managed, once) -------Start/Stop------------>  NeedleHost: worker thread(s), model lifetime, policy
                                                                     |
Game code: [NeedleTools] partial interface                           v
  -> source generator emits schema, grammar blob,          Session core: zero-alloc; UTF-8 in; typed records out;
     argument records, dispatcher                          grammar masks; sparse logits; jump-forward;
                                                           prompt-lookup speculation; confidence
                                                                     |
                                                                     v
                                                           Engine: sequences, KV cache, snapshots, chunk
                                                           checkpoints, batched multi-sequence decode
                                                                     |
                                                                     v
                                                           Layer program: fused phases, persistent worker team,
                                                           about 20 cheap barriers per token, no serial T=1 stages
                                                                     |
                                                                     v
                                                           Kernels: pure, tiered (Scalar | Avx2 | AvxVnni |
                                                           Avx512BW | AvxVnniInt8 | Neon | NeonDot | NeonI8mm)
```

## 5. Typed C# tool calls: the model speaks JSON, C# never sees it

**What cannot change without retraining.** Needle 3 was trained to read JSON-schema tools in the prompt and to write `<tool_call>[{"name":…,"arguments":{…}}]</tool_call>`. A shorter call syntax, such as `move_to(120,45)`, needs a fine-tune, which is Cactus Compute's area. This plan does not depend on it. We can ask upstream later.

**What we change.** JSON exists only inside the token stream. Game code declares and receives typed C# calls. No reflection, no runtime JSON parsing, NativeAOT- and trimming-safe, zero allocation.

### 5.1 Declaration (compile time, W10)

```csharp
[NeedleTools]
public partial interface INpcCommands
{
    [Tool("Move the selected unit to a map position.")] void MoveTo(float x, float y);
    [Tool("Attack a target.")] void Attack(Utf8Text64 target);
    [Tool("Give items to a party member.")] void GiveItem(Item item, int count, Utf8Text32 to);
    [Tool("Change the squad formation.")] void SetFormation(Formation formation = Formation.Line);
}

public enum Item { Potion, Sword, Shield, Arrow, Gold }
public enum Formation { Line, Wedge, Circle, Column }
```

An incremental source generator emits:
- **Schema.** The exact UTF-8 JSON schema as `static ReadOnlySpan<byte> Schema => "…"u8`. Names are snake_case; enums become schema enums; `Utf8TextN` becomes a string with `maxLength = N`; parameters with defaults are optional.
- **Grammar blob.** The compiled grammar program as constant tables, so no runtime JSON parsing or grammar compilation.
- **Argument records.** One blittable record per tool (`MoveToArgs { float X; float Y; }`) plus a tagged union `NpcCommand { Kind; MoveToArgs MoveTo; … }`, for ECS.
- **Dispatcher.** `static void Dispatch<THandler>(ref THandler handler, in NpcCommand command) where THandler : struct, INpcCommands` containing `switch (command.Kind) { case Kind.MoveTo: handler.MoveTo(command.MoveTo.X, command.MoveTo.Y); … }`. The JIT and AOT specialize it over the struct handler, so these are direct calls that can be inlined.

### 5.2 Typed decoding (run time, W11)

- The grammar automaton already knows the tool index (`StepCall` on the closing quote), the property index (`StepObject` phase 2), and every value boundary (pushes and pops of value frames). A generic sink, `Grammar.Advance<TSink>(…, ref TSink sink) where TSink : struct, IGrammarSink`, turns committed bytes into events:
  - `ToolSelected`, `PropertySelected`
  - `StringByte` (already unescaped), `NumberByte`
  - `EnumSelected`, `BoolSelected`
  - `ValueCompleted`, `CallCompleted`
- The mask search uses `NoSink`, so it costs nothing.
- The sink fills the argument record directly:
  - integers by digit accumulation;
  - floats with `Utf8Parser.TryParse` on a small stack buffer;
  - enums by literal index;
  - strings into fixed UTF-8 slots, whose capacity is the schema `maxLength`, so they cannot overflow by construction.
- **Correct by construction.** The grammar only admits tools, properties and types that exist in the C# declaration. Every completed call therefore binds, and `Dispatch` is total.
- **Streaming dispatch.** Each call is delivered when its object closes ("switch to wedge formation **and** follow the captain" starts the formation while `follow` is still decoding). Confidence arrives with the end of the turn. The caller chooses: dispatch on close, or dispatch after confidence.
- **ECS.** The worker writes `NpcCommand` records into the reply ring. A Burst `ISystem` switches on `Kind` and adds components such as `MoveTarget` and `AttackTarget`. No strings or JSON reach game code.
- **Back-compat.** The JSON `CallResult` API stays, rebuilt on top of the same sink.

### 5.3 Fewer sequential decode steps (W15, W16, W17)

| Technique | Lossless? | Measured potential | Dependency |
|---|---|---|---|
| Jump-forward over grammar-forced bytes, fed as one batched chunk with canonical tokenization | Text is identical. Token boundaries may differ from the model's own split, so validate on W28 | −24% passes; 55% of call tokens are forced | grammar forced-byte detection |
| Prompt-lookup speculation: propose the next k tokens from n-gram matches in the prompt and output, verify them in one pass, keep the longest greedy-matching prefix | **Yes**, identical tokens | −36% passes at k=4; 38.5% of tokens are copyable (arguments, reasoning spans, tool names) | engine chunk checkpoint/commit (W17); cheap T=2..8 GEMM (W20) |
| Both | | **−46% passes** | |
| Sparse head: compute only allowed rows when the mask has ≤ 256 tokens; masked log-sum-exp instead of a full softmax | Yes | 76% of call steps have ≤ 256 allowed tokens; saves the 3.24 MB head read and 27 µs of softmax per step | W15 |
| Reasoning budget: cap or skip `<think>` | **No**: changes model behavior | reasoning is 54% of tokens | opt-in only, after W28 shows acceptable accuracy |

The prompt prefix (system and tool schemas) is already primed once per session and restored with `Rewind`. Per call, only the query (10–20 tokens) is prefilled. Shortening tool descriptions saves little and risks accuracy.

## 6. Roadmap

Effort: S is under a day, M is 1–3 days, L is about a week. "Gain" is measured or estimated on the reference box at 4 threads.

| ID | Item | Milestone | Depends on | Effort | Gain or why | Issue |
|---|---|---|---|---|---|---|
| W01 | Fix golden fixture drift, document test setup | M0 Ground truth | | S | suite green out of the box | [#1](https://github.com/IAFahim/needle-cs/issues/1) |
| W02 | Benchmark harness: end-to-end, per-stage profile, kernels, native comparison, allocation gate | M0 | W01 | M | every later PR is measurable | [#2](https://github.com/IAFahim/needle-cs/issues/2) |
| W03 | CI on x64 and arm64: tests, scalar, netstandard2.1, NativeAOT smoke | M0 | W01 | M | regressions caught | [#3](https://github.com/IAFahim/needle-cs/issues/3) |
| W04 | License and provenance | M0 | | S | legal clarity for game shipping | [#4](https://github.com/IAFahim/needle-cs/issues/4) |
| W05 | Cross-platform model loading (Windows, macOS, Android APK, large pages) | M1 Game-safe hosting | | M | **Windows works** | [#5](https://github.com/IAFahim/needle-cs/issues/5) |
| W06 | Game-friendly threading: no foreign pinning, bounded spin, priorities, P/E cores, pause/resume | M1 | | M | no stolen frame time | [#6](https://github.com/IAFahim/needle-cs/issues/6) |
| W07 | Zero-allocation session core (UTF-8 in, records out) | M1 | W02 | L | 0 B per call | [#7](https://github.com/IAFahim/needle-cs/issues/7) |
| W08 | Unmanaged channel and `NeedleHost` worker (MPMC rings, tickets, cancellation) | M1 | W06, W07 | L | Burst-callable hot path | [#8](https://github.com/IAFahim/needle-cs/issues/8) |
| W09 | Session startup and memory: share tokenizer and trie per model, right-size arenas, persist primed prefix, warm-up | M1 | W07 | M | from 64 MB and 0.3–0.6 s per session to ≤ 16 MB and ≤ 50 ms | [#9](https://github.com/IAFahim/needle-cs/issues/9) |
| W10 | Typed tools: `[NeedleTools]` source generator (schema, grammar blob, records, dispatcher) | M2 Typed C# | W07 | L | C# methods instead of JSON | [#10](https://github.com/IAFahim/needle-cs/issues/10) |
| W11 | Typed decoding: grammar event sink fills argument records, streaming dispatch | M2 | W10 | M | no JSON text or parsing | [#11](https://github.com/IAFahim/needle-cs/issues/11) |
| W12 | Hadamard MLP: register-blocked Kronecker, parallel at T=1, fused, no zeroed `stackalloc` | M3 Decode latency | W02 | M | −630 µs/token | [#12](https://github.com/IAFahim/needle-cs/issues/12) |
| W13 | Layer program: about 20 barriers per token, parallel hyper stages, vectorized Sinkhorn, fused elementwise | M3 | W12 | L | −250 µs/token | [#13](https://github.com/IAFahim/needle-cs/issues/13) |
| W14 | Attention core: GQA-shared K/V (6 queries per KV head), int8 P·V | M3 | W02 | M | −150 µs/token | [#14](https://github.com/IAFahim/needle-cs/issues/14) |
| W15 | Masked decoding cost: sparse head rows, masked log-sum-exp, vectorized argmax, memoized grammar masks | M3 | W07 | M | −30 to −130 µs per call step | [#15](https://github.com/IAFahim/needle-cs/issues/15) |
| W16 | Jump-forward over grammar-forced spans | M3 | W15, W28 | M | −24% passes | [#16](https://github.com/IAFahim/needle-cs/issues/16) |
| W17 | Lossless prompt-lookup speculative decoding (engine chunk checkpoint/commit) | M3 | W20 | L | −36% passes (−46% with W16) | [#17](https://github.com/IAFahim/needle-cs/issues/17) |
| W18 | Cheap Reset/Rewind for stateless requests | M3 | W02 | S | per-request reset in µs | [#18](https://github.com/IAFahim/needle-cs/issues/18) |
| W19 | T=1 GEMV at bandwidth: multi-row, precomputed scales, prefetch, tuned splitting | M4 Kernels | W02 | M | −480 µs/token | [#19](https://github.com/IAFahim/needle-cs/issues/19) |
| W20 | Small-T and prefill GEMM: T=2..8 at most 1.3x of T=1; T=16..64 register blocking | M4 | W19 | M | enables W17 and W26; prefill ≤ 6 ms | [#20](https://github.com/IAFahim/needle-cs/issues/20) |
| W21 | Zero-cost tier dispatch and new x86 tiers (Avx512BW for Zen 4, AvxVnniInt8) | M4 | W19 | L | about 2x kernels on Zen 4 | [#21](https://github.com/IAFahim/needle-cs/issues/21) |
| W22 | ARM tiers (dotprod, i8mm) validated on arm64 CI | M4 | W03 | M | mobile and Apple | [#22](https://github.com/IAFahim/needle-cs/issues/22) |
| W23 | Unity package: Entities `ISystem` + Burst glue, samples, model delivery | M5 Unity | W08, W11 | M | about 100 lines of Unity code | [#23](https://github.com/IAFahim/needle-cs/issues/23) |
| W24 | Telemetry: blittable stats block, Unity Profiler bridge | M5 | W08 | S | visibility in game | [#24](https://github.com/IAFahim/needle-cs/issues/24) |
| W25 | AOT platforms: NativeAOT C ABI (`needle_*` compatible) versus Burst kernels | M5 | W07 | L | consoles, iOS, IL2CPP | [#25](https://github.com/IAFahim/needle-cs/issues/25) |
| W26 | Multi-sequence batched decode for many NPCs | M6 Throughput & voice | W13, W20 | L | several NPCs for about the cost of one | [#26](https://github.com/IAFahim/needle-cs/issues/26) |
| W27 | Whistle for games: zero-alloc push-to-talk, streaming, voice to typed call | M6 | W08, W11 | L | voice commands | [#27](https://github.com/IAFahim/needle-cs/issues/27) |
| W28 | Game command evaluation suite: accuracy gate for decoding policies | M7 Quality & parity | W02 | M | guards W16, reasoning caps, tool design | [#28](https://github.com/IAFahim/needle-cs/issues/28) |
| W29 | Parity suite against upstream native: prefix, date, confidence, `KvWindow` | M7 | W02 | M | no silent divergence | [#29](https://github.com/IAFahim/needle-cs/issues/29) |
| W30 | Upstream format tracking: version checks, golden refresh procedure | M7 | W01 | S | safe model updates | [#30](https://github.com/IAFahim/needle-cs/issues/30) |

Suggested order:
1. W01–W06 (blockers and measurement).
2. W12 and W07 in parallel (biggest speed win and the API foundation).
3. W08, W10 and W11 (Unity becomes possible).
4. W13, W14, W19 and W20 (speed).
5. W15–W17.
6. W23–W30.

### Performance targets (reference box, Lagos query, NativeAOT)

| Metric | Today | Target | Native today |
|---|---|---|---|
| Call latency, 4 threads | 92–111 ms | ≤ 25 ms | 95–108 ms |
| Call latency, 1 thread | 152–197 ms | ≤ 70 ms | measure in W02 |
| Decode step, 4 threads | 2.03 ms | ≤ 0.45 ms | about 1.75 ms |
| Sequential passes per call | 1 per token | −46% (W16, W17) | 1 per token |
| Prefill, 18 tokens, 4 threads | 15 ms | ≤ 6 ms | about 25 ms |
| Managed bytes per call after warm-up | about 6.2 KB | 0 | n/a |
| Native mallocs per call | > 0 | 0 | n/a |
| Session memory (private) and constructor time | 64 MB, 0.3–0.6 s | ≤ 16 MB, ≤ 50 ms | n/a |
| Worker spin after a request | about 10 ms per worker | ≤ 50 µs, then park | n/a |

## 7. Verification protocol

Setup on a fresh Linux x64 box:

```bash
curl -sSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh && bash dotnet-install.sh --channel 10.0 --install-dir "$HOME/.dotnet"
```

```bash
export PATH="$HOME/.dotnet:$PATH" DOTNET_CLI_TELEMETRY_OPTOUT=1
```

```bash
tools/fetch.sh && pip install numpy sentencepiece && python3 tools/golden.py 40 && python3 tools/golden_confidence.py
```

Gates for every PR:

```bash
dotnet test tests/Needle.Tests -c Release
```

```bash
NEEDLE_SCALAR=1 dotnet test tests/Needle.Tests -c Release
```

```bash
dotnet build src/Needle -f netstandard2.1 -c Release
```

```bash
dotnet publish src/Needle.Cli -c Release -o out/aot
```

- **Performance PRs** attach W02 harness output before and after the change: end-to-end at 1, 2 and 4 threads, the per-stage profile, kernels and allocations.
- **Parity gates:**
  - `ParityTests`: cosine at least 0.99995 with the same argmax for every chunk and thread combination, and the greedy decode equals the golden.
  - `SessionTests`.
  - `NeonTests`, on arm64 CI or under qemu.
  - From W28 on: the game evaluation suite, with no accuracy regression for any default-on decoding policy.

## 8. How agents pick up work

1. Pick an open issue whose dependencies are closed. Read this file and the issue; both are self-contained.
2. Create a branch named `w<NN>-<slug>`. Touch only the files the issue names, or say why in the PR.
3. Follow [section 3](#3-engineering-rules). Add a test that fails before the change and passes after.
4. Run [section 7](#7-verification-protocol), paste the numbers into the PR, and tick the item in the tracking issue.
5. Update this file when a learning changes.

## 9. Advanced C# techniques, mapped to this repo

| Technique | Where | Why |
|---|---|---|
| `static abstract` members on struct strategies (`IKernelTier`, `IGrammarSink`, `INpcCommands` handlers) | `Quant`, `Ops`, `Blocks`, `Grammar`, typed dispatch | The JIT and AOT specialize per struct, so there are no runtime branches in inner loops and no virtual calls (W11, W21) |
| Incremental source generator | `[NeedleTools]` interfaces become schema, grammar blob, blittable records and dispatcher | Typed C# tools without reflection; NativeAOT- and trimming-safe (W10) |
| `u8` literals and `ReadOnlySpan<byte>` constants | Schemas, prompt fragments, grammar tables | Zero-allocation constant data embedded in the assembly |
| Function pointers `delegate*<…>` chosen once | Tier tables, `Pool.Run` (already used) | One indirect call per batch, never per element |
| `[SkipLocalsInit]` module-wide | Every `stackalloc` (`Kron`, `Prepare`, `MatMulRows`, attention) | `Kron` zeroes 4 KB three times per layer today |
| `[InlineArray(N)]` | `MatMulJobs`, request and reply payloads, grammar frames, `Utf8TextN` | Typed fixed buffers instead of `fixed (… p = &first)` |
| `Utf8Parser`, `IUtf8SpanFormattable`, `Utf8.TryWrite` | Typed decoding, JSON fallback output | Zero-allocation text |
| `SearchValues<byte>` | Tokenizer marker scan | Vectorized scanning |
| `Volatile`/`Interlocked` with explicit linearization points; 128-byte padding | `NeedleChannel` rings, `Pool` control | Lock-free hand-off between Burst code and .NET workers without false sharing (W08) |
| `NativeMemory.AlignedAlloc` plus huge pages (`MADV_HUGEPAGE`, Windows large pages) | Weights, KV, arenas | Fewer TLB misses when streaming 17.5 MB per token (W05) |
| `[UnmanagedCallersOnly(EntryPoint = "needle_complete")]` plus NativeAOT `NativeLib=Shared` | C ABI compatible with `libneedle3.so` | One codebase for IL2CPP and AOT platforms, Burst `DllImport` and Python (W25) |
| `ModuleInitializer`, `AppContext` switches, `runtimeconfig` (TieredPGO, ReadyToRun) | Tier detection; player builds | No first-call hitches (W09) |
| `DOTNET_JitDisasm`, BenchmarkDotNet `DisassemblyDiagnoser` and `MemoryDiagnoser` | Kernel PRs | Prove vectorization, bounds-check removal and 0 B allocated |
| Blittable stats block or `System.Diagnostics.Metrics` | Host telemetry | Unity glue forwards cheap counters to the Profiler (W24) |

## 10. Open questions and decisions

1. **netstandard2.1.** Keep it compiling (scalar) until the Unity CoreCLR release is the minimum; decide in W25.
2. **AOT platforms (consoles, iOS, IL2CPP).** Options are a NativeAOT C-ABI plugin, Burst kernels, or scalar. Prototype in W25.
3. **Unity specifics, verified in-Editor 2026-10-30 (Unity 7000.0.0a12 on Linux x64; Editor only, players unverified):**
   - CoreCLR is the runtime: `FrameworkDescription` `.NET 10.0.10-dev`, `Environment.Version` 10.0.10. Per the Unity 7000 docs CoreCLR ships on Windows x86/x64/Arm64, macOS Arm64 and Linux x64 (and is the default there); every other platform is IL2CPP-only (W25).
   - Intrinsics ARE accelerated in the CoreCLR Editor: `Vector.IsHardwareAccelerated=True`, `Avx2=True`, `AvxVnni=True`; `Avx512BW=False` on the measured box (the VM hides AVX-512; the reference box in §2.2 has it), `AdvSimd=False` (x86_64, as expected).
   - `Interlocked`/`Volatile` in Burst: VERIFIED. `NeedleChannel` uses `Volatile.Read/Write` + `Interlocked.CompareExchange` and the Burst `NeedleSystem` calls `TrySubmitSilent`/`TryReceive` from Burst jobs; Burst 1.8.31 + Entities 1.5.0 compile the package with 0 errors and 0 Burst errors in our files, and 64-NPC round trips flow. Residual: 3 recurring Burst BC0102 internal errors in Entities' own `EntityCommandBufferPlayback.cs`/`EntityCommandBufferDebug.cs` (`DefaultInterpolatedStringHandler`) — a pre-existing Unity-alpha toolchain issue outside our call chain; the game runs.
   - Singleton-handle pattern VERIFIED: blittable `NeedleChannelHandle` (`NeedleChannel` + `nint Identity`) as a singleton component, `RequireForUpdate`, rebind on `Identity` change after host restart. GOTCHA (fixed during the W23 proofs): no C# namespace may be named `X.Unity` in an assembly whose Entities source-generated code qualifies `Unity.*` — `Needle.Unity` (W24 profiler namespace plus the asmdef `rootNamespace`) hijacked `Unity.Entities` resolution in hand code and in the generated `NeedleSystem__JobEntity` wrapper (CS0234). Fix: profiler moved to `namespace Needle`, asmdef `rootNamespace` `Needle.Unity` → `Needle` (assembly name unchanged).
   - Two one-line package compile fixes from the proofs: the missing `using Unity.Collections.LowLevel.Unsafe;` in `NeedleSystem.cs` (CS0246) and the `rootNamespace` above (CS0234).
4. **Jump-forward and reasoning caps** change outputs. Default them on only after W28 passes.
5. **AutoDate.** For games, default to off or a fixed caller-supplied date, so outputs are reproducible and primed prefixes stay valid (W09, W29).
6. **Many NPCs.** Default to stateless commands plus batched decode (W26); per-NPC sessions are opt-in because of memory.
7. **Upstream churn.** `config.json` says `engine_version` 3.1.0 and the wheel is 3.3.3. Fail fast on unknown `.cact` versions (W30).

8. **F08 behavior reviews from the zero-alloc session core (#40).** All four changes are kept; each is test-pinned and `NeedleSession.cs` needs no code change:
   - Empty query now fails with `EmptyInput` instead of running. Keep: an empty prompt decodes to no tokens, so running it only burns a full inference; the string API maps `null` to `""` consistently. Pinned by `SpanApiTests.FailuresReturnStatusesInsteadOfThrowing`.
   - A failed or cancelled turn rewinds to the primed state (`Abandon` → `Reset`). Keep: mid-turn tokens are unrecoverable without per-turn checkpoints (out of scope), and a deterministic known-good state keeps the session usable. Pinned by the `ContextUsed == primed` asserts after budget exhaustion and by the cancel test. Multi-turn callers replay from an explicit `Reset`.
   - Query size is capped at `min(65408, Context*16)` (`MaxInputBytes`). Keep: the tokenizer input buffer is 64 KB (`TokenizerInputBytes − InputOverheadBytes`), the cap scales down with `Context`, and oversize input now fails cleanly with `InputTooLong`. Pinned by the `MaxInputBytes + 1` cases.
   - A lone-surrogate `\uXXXX` escape in generated JSON now restores as U+FFFD. Keep: matches `Encoding.UTF8.GetString` replacement behavior and the JSON recommendation, and prevents emitting invalid UTF-16 that throws on later UTF-8 encoding. Pinned by `CallRestorerTests.LoneSurrogateEscapesBecomeReplacementCharacters`.
9. **F08 allocation gates (#40).** `bench-e2e --api span` exercises the zero-alloc path with reusable scratch buffers, so `--max-managed-bytes 0 --max-native-allocs 0` gates the session instead of the string wrapper (the default stays `text` to keep existing dashboards comparable). CI enforcement (owned by #32) is: build with `-p:NeedleAllocCounters=true`, then `needle bench-e2e <model> <tools> --threads 1,4 --sets all --api span --max-managed-bytes 0 --max-native-allocs 0`. Without the counter build the native gate reports an explicit needs-build violation rather than passing silently; `BenchMachine` prints `alloc-counters=on|off` in every report header. `BenchHarnessTests.ZeroGatesPassOnlyWhenNothingWasAllocated` pins the gate logic in both builds, and `SpanApiModeMatchesTextMode` pins span/text parity through the harness.
10. **F08 test robustness (#40 item 4).** The cancel test now retries (up to 5) until a 30 ms timer lands mid-flight, removing the fast-machine flake while still failing if `RequestCancel` is broken. The 30-minute loaded-box scalar run and CI stability stay with #32.
