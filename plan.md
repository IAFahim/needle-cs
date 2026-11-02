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
| `TokenTrie.Allow` per call step | 2.5 µs cold / 1.9 µs warm on this box (was 47.8 µs pre-W15) | under 1% of a 0.4 ms target step; the harness per-byte figure mixes warm-cache hits with fast-path dilution, see §2.11 |
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
- **CI matrix, all required-green 2026-11-02 (#32).** Tier logs from green run 41375210271: windows-x64 selects Avx2 (Avx2+Fma, no VNNI/512), linux-arm64 and macos-arm64 select NeonDot (AdvSimd+Dp), so NeonTests execute natively on both arm64 legs; macos unit+core also rerun NEEDLE_NODP=1 (heavy excluded for time, fallback covered). Promoted per the two-green-runs rule: windows heavy (green every completed run since 41348708967), macos unit 6/6 + core 3/3 + heavy two straight (41372129451, 41375210271), noavx512 experiment 5 straight. No macos-x64 leg: last stable Intel image (macos-13) retired Dec 2025 (https://endoflife.ai/github-actions-runner-images). No windows-arm64 leg yet (image GA Aug 2026, https://github.blog/changelog/2026-08-20-windows-11-arm64-vs2026-image-generally-available): validate with a one-shard probe (unit, expected-fail + issue link), watch the `needle cpu` tier line and the NeonTests verbose step for two green runs, then promote and extend to core/heavy.
- **Known red, not waived (other owners).** linux-x64/core LongContextStaysCloseToNative + ParityTests.GenerationMatchesReferenceGreedyDecode flakes (41424747474, 41372129451); linux-x64/heavy + linux-arm64/heavy speech/voice flakes SpeechTests.StreamingConverges / VoiceTests.StreamingSpan + PushToTalkRoundTrips (41427532179, 41424747474); macos-heavy ZeroAllocation 5-minute timeouts seen once (41366006614, #40).

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
| AMD Zen 4 | AVX-512 with EVEX VNNI, **no VEX AVX-VNNI** | Avx2 3-op + VBMI permute planes (default since 2026-11-02) | BwDot measured ~0.5x Avx2 — tier kept selectable, full 512-bit expansion open |
| AMD Zen 5 | AVX-512 + AVX-VNNI | AvxVnni-256 | measure Avx512BW |
| Steam Deck, PS5, Xbox Series (Zen 2) | AVX2 | Avx2 | same as Zen 3 |
| Apple Silicon, recent Android, Switch-class ARM | NEON, dotprod (i8mm on newer cores) | Neon SDOT | i8mm tier; verify per device |

#### Zen 4 tier measurements (2026-11-02, Ryzen 5 8500G, 12 threads, idle box)

Method: synthetic `Cq` weights + `Quant.Prepare`d activations, `Quant.MatMulRows` best-of-7 on 1 thread; GB/s = weight bytes / best time (decode GEMV is weight-traffic bound). Temporary xunit sweep, deleted after measuring. All 372 `MatMulTests`/`KernelTests`/`Int8DotTierTests`/`NeonTests` pass on every config (default, `NEEDLE_TIER=avx512|avx2|scalar`, `NEEDLE_NOAVX512=1`) — the VBMI permute planes ran on AMD for the first time.

| Shape | T=1 Avx2+permute | T=1 BwDot+permute | T=1 shuffle+Avx2 | T=1 scalar |
|---|---|---|---|---|
| 768x768 2b (147 KB) | 7.0 µs, 21 GB/s | 20.1 µs, 7.3 GB/s | 7.4 µs, 20 GB/s | 351 µs, 0.4 GB/s |
| 2048x768 2b (393 KB) | 18.8 µs, 21 GB/s | 35.7 µs, 11 GB/s | 19.7 µs, 20 GB/s | 936 µs, 0.4 GB/s |
| 2048x768 4b (786 KB) | 19.6 µs, 40 GB/s | 36.3 µs, 22 GB/s | 22.3 µs, 35 GB/s | 884 µs, 0.9 GB/s |
| 3072x768 4b (1.18 MB) | 30.7 µs, 38 GB/s | 55.1 µs, 21 GB/s | 33.9 µs, 35 GB/s | 1323 µs, 0.9 GB/s |

BwDot is 0.5–0.55x Avx2Dot everywhere (512-bit `madd` is double-pumped on Zen 4, plus the 512 combine + 256 fold); permute planes are ≥ shuffle planes (notably T=4 2b: 23.7 vs 37.8 µs). `Tiers.Best` therefore picks Avx2 over Avx512Bw — `NEEDLE_TIER=avx512` still forces the tier for future work.

Small-T ratios on Avx2+permute, 768x768 2b (T=1/2/4/8 µs: 7.0/14.6/23.7/47.4, i.e. 1x/2.1x/3.4x/6.8x): `Tile4` scales ~linearly with no weight reuse — #20 stays open, target T=2..8 ≤ 1.3x needs the register-blocked T=16..64 tile.

Elementwise/Kron on the same box: Silu-8192 Avx2 2.5–3.4 µs vs Avx512 3.2–4.9 µs (no win); Rms/Dot ~1 µs either way; `Kron` 512-bit (`Blocks.Wide`, tier-independent) beats Avx2 1.6–1.67x (4x1024: 78.7 vs 128.6 µs; 8x2048: 2024.9 vs 3319.7 µs; 32x2048: 6604.6 vs 11013.4 µs) — Wide stays on.

#### Full 512-bit plane expansion (#21, 2026-11-02, same box, idle)

Method: temporary xunit sweep, deleted after measuring. Micro-kernel A/B on the §2.8 shapes: baseline = `PermuteTwoBitPlanes`/`PermuteFourBitPlanes` expand + `Avx2Dot` + int sum per group; wide = `Gemv512Exp` (64B `Vector512` VBMI expand, single 512-bit `VPMADDUBSW`+`VPMADDWD` dot, `Vector512` accumulate, no 256-bit fold). Best-of-7 per side, interleaved, 1 thread, two runs. Exact int equality against the scalar reference on 32 random trials per path before timing. Absolute µs sit above the `MatMulRows` path (no scales/float tail in the loop); the ratio is the comparison.

| Shape | Avx2Dot+permute r1/r2 µs | 512-bit wide r1/r2 µs | Ratio r1/r2 |
|---|---|---|---|
| 768x768 2b (147 KB) | 10.20/9.50 | 10.60/10.50 | 0.96x/0.90x |
| 2048x768 2b (393 KB) | 27.40/25.60 | 28.50/28.40 | 0.96x/0.90x |
| 2048x768 4b (786 KB) | 28.10/27.60 | 36.40/35.70 | 0.77x/0.77x |
| 3072x768 4b (1.18 MB) | 42.10/41.70 | 54.40/53.90 | 0.77x/0.77x |

Verdict: no win — the ≥1.2x integrate criterion is unmet, so no integration patch is proposed and the default stays Avx2+permute. `Gemv512Exp.cs` remains as the measured prototype. Ceiling analysis: Zen 4's datapath is 256 bits wide, so 512-bit integer ops are double-pumped — AMD SOG 57647: "the datapath is 256 bits across all operations ... Due to the reduced instruction and macro-op overhead, expect software to have reduced power and increased performance when using 512-bit operations, compared to 256-bit operations" (https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown/57647_zen4_sog.pdf; implementation background https://chipsandcheese.com/p/amds-zen-4-part-1-frontend-and-execution-engine, measurements https://www.phoronix.com/review/amd-zen4-avx512). uops.info confirms exactly half-rate 512-bit throughput on Zen 4: `VPERMB` ZMM 1.00/cycle, latency 5 (https://uops.info/html-instr/VPERMB_ZMM_ZMM_ZMM.html); `VPMADDUBSW` ZMM 1.00/cycle, latency 3 (https://uops.info/html-instr/VPMADDUBSW_ZMM_ZMM_ZMM.html); `VPMADDWD` ZMM 1.00/cycle, latency 3 (https://uops.info/html-instr/VPMADDWD_ZMM_ZMM_ZMM.html) — i.e. one 512-bit op costs what two 256-bit ops cost (https://stackoverflow.com/questions/74513267/how-to-analyze-the-instructions-pipelining-on-zen4-for-avx-512-packed-double-com). Per 256 activations the wide form therefore spends the same 8 cycles of permute+madd throughput as the 256-bit form by construction, then pays extras the narrow form avoids: 64B loads, a 512-bit shift, and two-half `Vector512.Create` activation assembly (activation mates are group-interleaved 128-wide, so each 512-bit lane gathers from two 256-bit halves). Removing the 256-bit fold did not help, so the fold was never the bottleneck — the double-pumped madd pair is. 4-bit is worse (0.77x) because only two planes fuse to 512 while the activation assembly cost stays. Contrast `Kron`'s 1.6x from 512-bit float FMA: that win comes from halved traffic plus 8 live accumulators hiding FMA latency, not from dot throughput; the GEMV dot is expand-bound at already-saturated 256-bit rates. .NET 10 also has no 512-bit `VPDPBUSD`, so the madd+maddwd pair cost stands on every x86 tier here. Zen 5 runs all three 512-bit ops at 0.50/cycle (full width, same uops.info pages), which could flip the arithmetic — revisit with a Zen 5 box.

#### Real-weight `needle bench` + `bench-e2e` (2026-11-02, same box, idle, serial runs)

Method: `needle bench ref/hf/needle3.cact --threads 1,4` and `bench-e2e ... --threads 1,4 --profile --sets golden` (reps 5), best-of-9 per kernel, `NEEDLE_TIER` forced per row. GB/s = weight bytes / best time. Default tier == forced avx2 within noise (gate 768x768 2b T=1: 7.26 vs 7.23 µs), so the platform-table row stands — no faster default emerged.

| gate_proj 768x768 2b, best µs, T=1/2/4/8/16/32/64 | T=1 GB/s | T-ratios |
|---|---|---|
| default/avx2: 7.26/14.03/24.83/49.41/98.78/197.50/397.26 | 20.3 | 1x/1.93x/3.42x/6.81x |
| avx512 (BwDot): 13.75/26.25/58.28/116.30/234.30/464.04/928.83 | 10.7 | 1x/1.91x/4.24x/8.46x |

head 8192x768 4b T=1 (3.15 MB, decode-realistic): default/avx2 80.9 µs, 38.9 GB/s; avx512 147.7 µs, 21.3 GB/s. Pooled (4 threads) T=1: gate default/avx2 3.2/3.1 µs (45.8/47.1 GB/s), avx512 5.4 µs (27.3 GB/s); head default/avx2 29.4 µs (107 GB/s), avx512 53.1 µs (59.3 GB/s). #20 gate (T=2..8 ≤ 1.3x T=1) is not met on any tier — still needs the register-blocked T=16..64 tile.

`bench-e2e` golden set (2 prompts, prefix 125 tokens, reps 5): 1 thread — prefill 14.7 ms, decode 1146/1150 µs per token (best/med), total 66.7 ms; 4 threads — prefill 6.5 ms, decode 820/828 µs per token, total 43.9 ms; 21.0 Pool.Run per token both. Decode profile T=1 top stages: attention_qkvg 316.3 µs (31%), attention_out 149.7 (14%), attention_core 125.8 (12%), mlp 113.8 (11%), engram 94.2 (9%), logits 87.2 (8%); T=4 top: qkvg 152.5 (20%), mlp 139.4 (19%), hyper_prepare 80.5 (11%).

Small-T T=1..8 sweep, same box, quiet (best-of-9, `NEEDLE_TIER` forced per row; default == avx2). Best µs:

| tier | gate_proj 768x768 2b T=1..8 | ratios | head 8192x768 4b T=1..8 | ratios |
|---|---|---|---|---|
| default (permute+Avx2Dot) | 7.27/14.20/21.24/24.63/31.61/38.89/45.98/49.18 | 1/1.95/2.92/3.39/4.35/5.35/6.32/6.76 | 81.3/157.5/239.6/260.8/341.6/415.4/500.3/527.3 | 1/1.94/2.95/3.21/4.20/5.11/6.15/6.49 |
| NEEDLE_NOAVX512=1 (shuffle+Avx2Dot) | 7.64/14.63/22.46/26.22/33.94/40.21/47.65/51.09 | 1/1.91/2.94/3.43/4.44/5.26/6.24/6.69 | 90.6/164.0/254.1/277.1/368.1/441.4/530.9/554.7 | 1/1.81/2.81/3.06/4.07/4.87/5.86/6.13 |
| avx512 (BwDot+permute) | 13.55/26.17/39.67/60–81/94/107/120/118–161 | 1/1.93/2.93/~5–6/…/…/…/~9–12 | 147.3/283.5/430.3/652–920/…/…/…/1285–1839 | 1/1.92/2.92/~4.4–6.2/…/…/…/~8.7–12.5 |
| scalar | 354/715/1096/1419/1756/2109/2468/2849 | 1/2.02/3.10/4.01/4.96/5.96/6.97/8.05 | 3584/7266/10890/14554/18214/21860/25569/29201 | 1/2.03/3.04/4.06/5.08/6.10/7.13/8.15 |

Findings for #20. (a) The gate (T=2..8 ≤ 1.3x T=1) is unmet on every tier; best is the default tier at 1.95x/3.39x/6.76x (T=2/4/8, gate_proj). T=1 did not regress (gate 7.27 vs 7.25 µs baseline; head 81.3 vs 80.9). (b) Negative result, reverted with no code change: unified single-row Tile3/Tile5 (one expansion shared across 3/5 tokens) measured 3–4% SLOWER than the chunked Tile2+Rows / Tile4+Rows paths (gate T3 21.9 vs 21.2 µs, T5 33.0 vs 31.6). Root cause is ILP, not reuse: a single-row tile keeps only 3–5 live FMA chains against a ~15-cycle expand→madd→cvt→fma dependence, while Pair2/Pair4 keep 4–8 chains fed; the pair-blocked chunking already sits at the 16-ymm-register practical floor. (c) The avx512 tier is 1.9x slower at T=1 AND unstable at T≥4 run to run (gate T4 60 vs 81 µs, T8 118 vs 161 across back-to-back quiet runs; 512-bit double-pumped Tile4) — stays force-selectable, never default. (d) Analytic floor on the 256-bit path: per group-instruction audit gives shared expand E≈7 vs per-(row,token) dot D≈13, so total(T)/total(1) ≥ (E+T·D)/(E+D) ≈ 1.65x/2.9x/5.3x for T=2/4/8; 1.3x at T=4 needs E≥9D, i.e. a memory-bound regime or a VNNI-class dot that .NET 10 cannot emit (no 512-bit VPDPBUSD) — same conclusion as the #20 instruction-floor comment. Next lever for #20 is therefore not a wider small-T tile but the register-blocked T=16..64 prefill path and/or 2-D (rows×tokens) parallel chunking.

GEMV edge-norm wart fixed the same day (`Gemv.cs`: `CorrectionsAtEdge` + the two `Row4`/`Row2` edge branches now sign-extend the half bits via `(short)`, so the fp16 sign lands in the float sign; bit-identical on the non-negative scales real models carry). Parity on the four tiers after the fix: default/avx2/avx512 20/21, scalar 21/21 — the single failure is `LongContextStaysCloseToNative`'s Miles-Davis turn (native 0.8656 vs managed ~0.60, Δ≈0.27 > 0.15/0.25 bounds), which reproduces identically on the pristine base commit, so it is a pre-existing CPU-specific tie amplification, not a regression.
Panel-tile prototype for #20, second negative result (same box, quiet load < 1, serial interleaved A/B, best-of-9 per cell, two rounds; full tables below are best-of-best). New angle from external research: KleidiAI-style activation LHS panel (`learn.arm.com` QD8 GEMM docs: `pack_lhs` once, reuse across all columns) and ik_llama.cpp `iqk_mul_mat` ("reuse the unpacked quants as much as possible"; IQ2_XXS unpack-to-`Q8_0_R8` once then GEMM vs all columns, ~3x prompt-processing gain, PR #515). `Tile4Panel`/`PairPanel4` + `Tile2Panel`/`PairPanel2` (`Gemv.cs`, gated `NEEDLE_PANEL=1`, default path bit-intact): repack the 4 (2) token activation rows once per tile into group-major order, stream the panel contiguously, each token's 4 vectors loaded once per group and reused for both rows via `ApplyPlanesVec` (the old `Pair4`/`Pair2` reload them per row). Single-row tails reuse the existing `Row4`/`Row2`.
Correctness: `MatMulTests` 139/139 plus new `PanelTilesMatchScalarWhenEnabled` 7/7 (T=2..9 incl. odd-row tails, 2b/4b, cols 768/3072) on default/avx512/avx2/scalar/`NEEDLE_NOAVX512=1` with the flag on; `KernelTests`+`Int8DotTierTests` green. Full suite on a committed-only worktree: 658/662 — the 4 `WhistleStageTests` fail on missing `ref/golden/whistle/*/pcm.f32` fixtures (environmental, fail identically without the flag) and 2 `ZeroAllocationTests(threads:4)` failed only as 5-minute timeouts under concurrent load, passing 4/4 serially with the flag in 34 s.
Timing (default tier, best-of-best base vs panel): gate_proj 768x768 2b T=1/2/4/8 = 7.23/14.01/24.64/48.87 vs 7.24/13.89/24.04/47.92 (T=1 +0.1%, T=2/4/8 −0.9%/−2.4%/−1.9%; T=16/32/64 −2.3%/−2.2%/−2.1%); head 8192x768 4b = 81.43/156.50/259.57/521.85 vs 81.70/154.19/259.42/519.86 (−1.5%/−0.1%/−0.4% at T=2/4/8). Panel ratios: gate 1/1.92/3.32/6.62, head 1/1.89/3.17/6.36 — the 1.3x gate still missed by ~2.5x at T=4. The ≤2% deltas sit inside the panel path's own round-to-round spread (gate T4 24.46 vs 24.04), so the verdict is no stable win, not a small win. T=1 unregressed either way (flag off = same code; flag on routes T=1 through the untouched `Rows` path).
avx512 tier: no attribution possible — it flaps ±30–120% between back-to-back identical runs on both old and panel tiles (gate T4 base 80.49→58.18, panel 143.09→63.30; head T4 base 922.99→631.56, panel 625.71→1401.55), i.e. machine AVX-512 frequency state, not code; tier stays non-default and force-selectable for the Bw512 work.
Floor proof, strengthened with measured throughput: uops.info lists Zen 4 `VPMADDUBSW`/`VPMADDWD` YMM at 1 uop and 0.5-cycle throughput (2/cycle), so the 4-plane `Avx2Dot` alone costs ≥4 cycles per 128 MACs per (row, token, group) before the 3 adds, int→float convert, scale broadcast and FMA — T=4 does 4x the dots of T=1 while expansion (already shared across tokens since the two-wide tiles) shrinks to a ~2–3-cycle remainder. The kernel is madd-throughput-bound at ~20 GB/s effective, far from the bandwidth-bound regime where extra tokens ride free; reaching 1.3x at T=4 needs a VNNI-class dot (4x MACs/instr, absent on this Zen 4 and unemittable from .NET 10) or a different machine. Recommendation for #20 stands: stop widening small-T tiles, pursue the register-blocked T=16..64 prefill path and 2-D rows×tokens parallel chunking instead.

#### Layer-team barrier fusion + spin re-decision (2026-11-02, same box, quiet, serial runs)

Method: `bench-e2e --threads 1,4 --reps 5 --sets golden --profile`, interleaved fused-vs-HEAD A/B in identical quiet-box worktrees (only `Engine.cs` differs, 3 lines); best of reps per cell, each 4T cell replicated. 1T (single member, team barriers no-op) is the no-change control.

Change (`src/Needle/Engine.cs`, 3 lines removed, numerics-preserving): dropped the team barriers between same-token-partition elementwise stages — PostNorm→Mlp, Mlp→Combine (all three shard `count` identically, so producer and consumer are the same member) and the EngramBlend→AttentionPrepare barrier on site layers (same argument; the member-0 history save it brackets is read only by the next token, ordered by the RunTeam join). All cross-partition barriers (Prepare→QKVG, Core→Gate, Gate→Out, Out→PostNorm, Gather→MatMul→Blend) and all member-0 publish barriers (history Loads, arrivals zeroing) stay. Team barriers per token: 275 → 230 (15×11 + 5×10 across 20 layers / 5 engram sites); `Pool.Run`/token stays 21.0 (20 layer teams + head, asserted by `BenchHarnessTests`).

Retained deliberately after measurement: the Taps→SaveHistory barrier looks redundant (Save reads `QRaw` tails while other members' `Taps3` reads overlapping `QRaw` windows — read-read) but removing it deterministically breaks `ForwardMatchesReferenceLogits` (chunk-64/threads-8 cosine 0.99973 vs the >0.99995 bound; 3/3 fail without, 4/4 pass with on a clean tree). Same-partition reasoning does not cover that trio — do not fuse it without parity proof. The QKVG→Taps and Heads→Core outer barriers were likewise kept (their inner F1/H1 joins looked sufficient but were not isolated as safe). `RowsPerTask` coarsening was evaluated and rejected: the heterogeneous QKVG batch (576/96/128/768 rows) would idle members (6 tasks over 4 members); the current 3-tasks-per-thread (~12–13 tasks) balances better and homogeneous batches already balance — no change.

Results (mean of the 2 golden prompts, decode µs/token best):

| cell | HEAD | fused | Δ |
|---|---|---|---|
| 1T prefill ms | 14.58 | 14.56 | ~0 (control, as expected) |
| 1T decode | 1142 | 1139 | ~0 (control) |
| 4T prefill ms | 6.64 | 6.45 | −3% best |
| 4T decode | 841 | 812 | −3.4% best (29 µs) |
| Pool.Run/token | 21.0 | 21.0 | unchanged |

#33 spin re-decision at the new barrier count (`NEEDLE_SPIN_US` on the fused binary, reps 5): 4T mean decode best 832 (50 µs) vs 841 (300 µs); 1T 1158 vs 1176 (no workers at 1T — pure noise control). All inside the 812–846 run band: spin explains none of the variance, since dispatches land every ~40 µs during decode and workers never park mid-call under either setting. Decision: keep `DefaultSpinMicroseconds = 300` (`Game` stays 50) — no evidence 50 is free across bursty game gaps, and 300 keeps the #6 wake-storm protection; the frame-budget hint (#33 item 2) already exists (`FrameBudget` + `FrameBudgetHit` wiring, covered by `PoolTests`/`JumpForward`/`Speculative`/`SpanApi`/`Service` tests). No code change for #33.

Design notes: the pool is a two-phase waiter (spin `spinTicks`, then park) and the layer team is a centralized sense barrier over one shared `TeamBarrier` — the small-N regime where centralized barriers win (O(N) on one line for N ≤ 8 members; https://www.cs.rice.edu/~johnmc/papers/tocs91.pdf, https://unseel.com/cs/barrier-synchronization), with spin-then-park set past the context-switch cost (https://www.cs.huji.ac.il/labs/parallel/stud/Tsafrir-MsC.pdf).

### 2.9 Unity W23 proofs (2026-10-30, in-Editor 7000.0.0a12, Linux x64)

- **64-NPC load (LoadProof harness, 1.2 s cadence ≈ 50 asks/min/NPC, 120 warmup + 600 measured frames, two runs):** main-thread managed allocations in the per-frame NPC pump bracket: **0 bytes, 0 nonzero frames** both runs. 38–39 replies per run, 100% carrying dispatched tool calls (`MoveTo`/`GiveItem`); 64 asks pending at end (engine does ~1 reply/s on 2 threads vs ~53 asks/s offered — saturation is expected, the Unity side stays flat).
- **Main-thread timing (PlayerLoop timestamp markers around `SimulationSystemGroup`, same runs):** 293.8 µs mean / 595.7 µs max with 64 asks pending; 210.4 µs / 711.9 µs max idle floor (load disabled, system skipped); incremental ≈ 84 µs. The ≤ 20 µs criterion is NOT met in this environment (Editor-debug codegen with safety checks, ~6 fps, inference workers saturating the 4-core box). Method notes: the headless Editor runs with the Unity Profiler disabled and `Profiler.enabled = true` throws NRE, so `Recorder`/`ProfilerRecorder` capture nothing; Entities 1.5 groups expose no `PlayerLoop` children, so no per-system split was obtainable — re-measure in a release player before treating 84 µs as real work.
- **Voice end to end in PlayMode** (`whistle.cact` + `test.wav` from StreamingAssets, `PushToTalk` streaming + `Release`): full 41-word transcript, not truncated. 4-thread `Transcribe` matches upstream exactly ("creating a way of file", "Wi-Fi"); the sample default 1-thread lands on the known beam knife-edge ("grading away a file"). So the transcript is **thread-count-sensitive** — `ThreadCountDoesNotChangeTheTranscript` only pins `record.wav`, not the knife-edge clip; evidence for #38, not a W23 problem. The sample is unchanged (still 1 thread).
- **Repro:** the harness lives in the test project only (`Assets/LoadProof/LoadProof.cs`, dormant `LoadProof` GameObject in `NeedleTest.unity`); rerun = enter PlayMode, read the `LOADPROOF` console line.

### 2.10 MLP/attention stage work, issues #37 (MLP ≤ 60 µs/tok) and #41 (attention ≤ 50 µs/tok)

Shipped (2026-11-02, `Blocks.cs` + `Ops.cs` only): (a) GQA-shared V accumulate — one V-row read feeds all six query heads of a GQA group (`AccumulateFloatSix`: 6 heads x 2 vecs = 12 chains, inside the 16-ymm budget; the 3-head chunked path loaded and converted every V row twice). Per-head sums keep ascending-s order, so the output is bit-identical to the chunked path. (b) Vectorized `Ops.Softmax` peak reduction (AVX max fold + scalar tail; max is exact, so bits are unchanged). (c) One `Sse.Prefetch0` (4 rows ahead) in the shared-V loop; Kron prefetch skipped (4 KB working set is L1-resident). Research: 10 unrolled FMA chains saturate AVX2/FMA ports ([MicroArchBench](https://github.com/Olajide-Badejo/MicroArchBench)); 4-row GEMV blocking on Zen 3 ([ZenGEMV](https://annals-csis.org/Volume_47/drp/6278.html)); single-KV-shared-across-heads formulation ([GQA](https://arxiv.org/html/2305.13245v3/)); ~5% from prefetch distance tuning ([llama.cpp #768](https://github.com/ggerganov/llama.cpp/pull/768)); FMA latency/throughput tables ([Agner Fog](https://www.agner.org/optimize/instruction_tables.pdf)).

Negative result, reverted: 16-chain `KronAvx2Wide32` (4 rows x 4 vecs, single 32-wide chunk). Isolated micro (`activeRows` 24/32, best of 2000): 0.956x / 1.313x vs the 8-chain `KronAvx2` — doubling the chains doubles the per-iteration FMA work (16 FMAs = 8 cycles at 2/cycle) past the ~5-cycle latency it was meant to hide, so the wider kernel goes throughput-bound and loses; 12 chains (the ~10-chain sweet spot) is unreachable with power-of-2 blocking on 32x32. Removed with no trace (method + dispatch + test deleted).

Method: `bench-e2e --profile` golden set, reps 5, interleaved base/patched alternating order, 5 rounds, best-of-5 per stage per thread count, clean-HEAD worktree + patch (sibling WIP kept out of the tree). Base == HEAD `21f4f79`, patched == base + (a)–(c) above (sweep ran with Wide32 still in: the mlp patched column below carries its ~+3% regression; post-revert mlp is expected ≈ base — the box sat at load 29–40 during the confirmatory window, so no trustworthy re-measure exists yet).

T=1 decode µs/token, best-of-5 (targets: mlp ≤ 60, attention_core ≤ 50 — neither met):

| stage | base | patched | delta |
|---|---|---|---|
| embed | 0.8 | 0.8 | +0.0% |
| hyper_prepare | 40.7 | 41.9 | +2.9% |
| hyper_phi | 27.9 | 29.3 | +5.0% |
| hyper_mix | 11.8 | 12.2 | +3.4% |
| engram | 95.6 | 100.0 | +4.6% |
| attention_prep | 12.2 | 12.6 | +3.3% |
| attention_qkvg | 316.5 | 330.1 | +4.3% |
| attention_post_projection | 25.9 | 27.3 | +5.4% |
| attention_core | 131.0 | 120.5 | −8.0% |
| attention_gate | 14.3 | 14.6 | +2.1% |
| attention_out | 150.0 | 157.6 | +5.1% |
| attention_post_norm | 4.4 | 4.6 | +4.5% |
| cross | 0.0 | 0.0 | — |
| mlp | 123.7 | 128.5 | +3.9% (Wide32 regression, reverted) |
| combine | 7.4 | 7.5 | +1.4% |
| logits | 91.7 | 92.2 | +0.5% |
| unattributed | 3.7 | 4.0 | +8.1% |
| total | 1060.0 | 1090.5 | +2.9% |

T=4 decode µs/token, best-of-5:

| stage | base | patched | delta |
|---|---|---|---|
| embed | 0.8 | 0.8 | +0.0% |
| hyper_prepare | 86.3 | 86.4 | +0.1% |
| hyper_phi | 10.6 | 10.9 | +2.8% |
| hyper_mix | 37.4 | 35.9 | −4.0% |
| engram | 57.8 | 58.3 | +0.9% |
| attention_prep | 19.4 | 19.3 | −0.5% |
| attention_qkvg | 151.4 | 149.5 | −1.3% |
| attention_post_projection | 39.1 | 34.4 | −12.0% |
| attention_core | 60.6 | 58.8 | −3.0% |
| attention_gate | 30.2 | 29.9 | −1.0% |
| attention_out | 58.9 | 58.0 | −1.5% |
| attention_post_norm | 18.0 | 10.3 | −42.8% (small absolute, noisy) |
| cross | 0.0 | 0.0 | — |
| mlp | 146.5 | 150.7 | +2.9% (Wide32 regression, reverted) |
| combine | 10.8 | 10.7 | −0.9% |
| logits | 37.7 | 37.7 | +0.0% |
| unattributed | 4.1 | 4.3 | +4.9% |
| total | 801.8 | 777.5 | −3.0% |

Verdict: no stage hits its target (closest: T=4 attention_core 58.8 vs 50; mlp 146–150 vs 60 needs intra-token parallelism, which is an `Engine.cs` sharding change owned by Barrier13). The uniform ±2–5% drift on untouched stages across T=1/T=4 is load noise, not signal — only attention_core moves the same direction twice (−8.0%/−3.0%). No tier regressed: MatMul/Kernel/Parity green on all four tiers with the final code (default 382/382, `NEEDLE_TIER=scalar` 160/160, `NEEDLE_TIER=avx512` 160/160, `NEEDLE_NOAVX512=1` 160/160, clean worktree). Scalar-epilogue and finer-chunking notes: the Softmax peak is now the only vectorized change standing (exact max, no bit change); `AttentionBatch` chunking is already maxed at decode (`MinChunkPositions` never binds: count ≈ 126 ≥ 2 × 48 with 2 chunks at T=4), so no chunking change shipped.

### 2.11 Grammar-mask cost and Whistle encoder gap, round 5 (issues #39, #38)

**Allow discrepancy, reconciled (2026-11-02, this box, idle, serial runs).** The 47.8 µs plan figure predates W15 (no mask cache, per-step `walkStates` alloc) on the Xeon reference box; the harness's per-byte-position figure mixes warm-cache hits with fast-path dilution. Measured now:

| Source | Cost per step | What it actually times |
|---|---|---|
| `needle bench` structural probes (re-timed same state) | 0.09–0.10 µs | warm-cache hit: hash + full-frame compare + 1 KB mask copy |
| `needle bench` unbounded-string probes | 1.51 µs | string fast path recomputed every call (bypasses the cache) |
| `needle bench` mean of 121 byte positions | 2.51/2.52 µs | partially-evicted cache + fast-path mix, diluted by mid-string states |
| `TokenStepMasksMatchBruteForceAndReportColdVsWarm` (384 genuine greedy token-boundary states, fresh trie) | 2.51 µs cold / 1.90 µs warm | the true per-call-step cost: each state visited once, as in real generation |

Only 36 of the 384 token states hit the fast path, so per-byte positions dilute the structural states roughly 10x — the harness's second bias on top of the warm cache. 3,145,728 token checks pin cold masks to brute-force `Advance` (exact off the string fast path, subset on it, as already pinned); longest token is 16 B, so the 64 B walk budget never truncates. Verdict: the trie is benign, no code change. True cost is ~19x below the old figure: under 1% of a 0.4 ms step, not 12%.

**Crossover re-run, round 5 (same box, `tools/native_crossover.py`).** Numpy `Whistle(cact, quant=set())` (empty set is the float reference; a bare `False` hits a `set()` bug in `golden_whistle.encode`) versus `needle_embed` from `ref/native/libneedle3.so` on identical 16 kHz input; managed frames via `Whistle.Embed` at 4 threads (`/tmp/v01/zdump`, public API only). Test three-way: numpy–native 3.18% / 0.999521, managed–numpy 1.14% / 0.999933, managed–native 3.45% / 0.999436 — cosines match round 2 (0.999528 / 0.999915 / 0.999467) to 4 decimals. Clips: record 2.76%, test 3.18%, hotword 1.62%, long25 1.20%; synthetics: zeros 0.91%, broadband 0.87%, pure tone 33.47%. The quiet-frame lead reproduces: test worst frames are 27/232/231/230/199 (the 27–29 and 230–233 onsets) and quiet-tercile error is 3.55% versus 2.12% mid / 2.44% loud. (My phase-jump narrowband gaps only 1.2% — that construction is less sparse than round 3's 31% probe, not a contradiction; the tone probe carries the sparsity point.) Native is bit-deterministic across embeds (max abs diff 0). No fix landed — the residue is directional and formula-level; the beam-3 default stands. Transcript-level crossover (native frames into the managed decoder at beam 4) was not re-run: the encoder verdict needs no source changes.

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
