# Needle.Inference on netstandard2.1: fast on every runtime

Status: plan only, written 2026-11-02. Companion to [plan.md](plan.md): that file owns the engine (W01–W30); this one owns **making the netstandard2.1 build as fast as the net10.0 build on every runtime Unity ships**. Item IDs are `Nnn` so they never collide with `Wnn`. Where an item overlaps a W item, it links it instead of duplicating it.

## 0. TL;DR

- **The 700-frame latency is a build problem, not a runtime problem.**
  - Unity 7000.0.0a12 *runs* CoreCLR .NET 10 in the Editor. Evidence: `libcoreclr.so` is mapped in the Editor process, and plan.md §10.3 measured `Avx2=True` in-Editor.
  - Unity *compiles* scripts at C# 9 against the .NET Standard API profile (`apiCompatibilityLevel: NET_Standard_2_0`, `Assembly-CSharp.dll` references `netstandard 2.1.0.0`).
  - The netstandard2.1 build defines no `NEEDLE_SIMD`, so `Simd.Enabled` is the constant `false`. All 104 `#if NEEDLE_SIMD` sites fall to `Scalar.*`.
  - The ns2.1 waits call `SpinWait.SpinOnce()` without `-1`. On CoreCLR that calls `Thread.Sleep(1)` once a wait passes 20 spins. With about 260 team barriers per token, that adds millisecond-scale sleeps per token on top of scalar math.
- **Today's fix (N00):** ship the net10.0 DLL to CoreCLR targets and keep the generated code C# 9-safe. That restores the measured 92–111 ms per call in the Editor and in desktop players.
- **Why netstandard2.1 still has to be fast:**
  - Per the Unity 7000 docs (plan.md §10.3), CoreCLR covers Windows, macOS arm64 and Linux x64. Android, iOS, consoles and WebGL are IL2CPP-only.
  - IL2CPP will not light up `System.Runtime.Intrinsics` (hypothesis H4). The ns2.1 IL is the portable artifact.
- **Architecture:** one netstandard2.1 core plus a **kernel table** of unmanaged function pointers, installed at startup by the fastest available provider:
  - CoreCLR: intrinsics provider, the existing AVX2/VNNI/AVX-512/NEON kernels.
  - IL2CPP and Mono players: Burst provider, the same kernels ported 1:1 to `Unity.Burst.Intrinsics`.
  - Anything else: a portable provider (`Vector<T>` plus LUT-GEMV scalar).
  - Phase 2 moves a whole layer per team member into one Burst call.
- **Doing less work multiplies every provider (Track D):**
  - Reasoning tokens are 54% of decode (plan.md §2.4).
  - Jump-forward plus prompt lookup cut passes by 46%.
  - Batched lanes amortize the 17.5 MB weight stream across NPCs.
  - A shipped primed prefix removes tool-prompt prefill at startup.
- **Order:** N00/N02 (today) → N10–N13 (measure, conformance) → N20s (free ns2.1 wins) and N30 (kernel ABI), in parallel → fan out N32–N39, N50s and N60s.

## 1. Facts and hypotheses

Everything in this file is either **V** (verified, evidence given) or **H** (hypothesis, owner item given). No PR may cite an H as a fact.

| ID | Fact | Evidence |
|---|---|---|
| V1 | Editor runtime is CoreCLR .NET 10 | `libcoreclr.so` in `/proc/<editor pid>/maps` (2026-11-02, project `~/Say`); plan.md §10.3: `FrameworkDescription .NET 10.0.10-dev`, `Avx2=True`, `AvxVnni=True` |
| V2 | Script compile profile is C# 9 + .NET Standard | `unity command settings.player get` → `scriptingBackend: CoreCLR`, `apiCompatibilityLevel: NET_Standard_2_0`; compile errors CS8706/CS8929 (static abstract `IToolSet` adapter), CS8773 (`u8` literal), CS7069 (`ReadOnlySpan<>` from `System.Runtime` in the adapter) |
| V3 | ns2.1 build is 100% scalar | `src/Needle/Needle.csproj` defines `NEEDLE_SIMD` for net10.0 only; `Scalar.cs`: `public const bool Enabled = false` under `#else` |
| V4 | ns2.1 waits slept pre-N20 (mechanism of V14) | Pre-N20 `Pool`/`NeedleHost`/`VoiceHost` used `spin.SpinOnce()` → `Thread.Sleep(1)` past 20 spins; N20 `Spinner` replaced all sites (see V12) |
| V5 | ns2.1 `Mem.Clear` was a byte loop pre-N21 | Now `Unsafe.InitBlockUnaligned` ≤ 1 GB slices (`Native.cs`) |
| V6 | ns2.1 `Map` tries mmap first, falls back to copy (N22) | `ModelMemory.Map` ns2.1: `MemoryMappedFile` + `madvise` via `DllImport libc`; `FromBytes` unchanged for APK assets |
| V7 | Generator emits `u8` only when LanguageVersion ≥ 11 (N02) | `Emitter.Emit(..., useUtf8Literals)`; C# 9 path uses `static readonly byte[]`; adapter still conditional on `IToolSet` resolving |
| V8 | Unity's own Mono 6.13 runs headless | `~/Unity/Hub/Editor/7000.0.0a12/Editor/Data/MonoBleedingEdge/bin/mono --version` → 6.13.0; `csc`, `mcs` and the `netstandard.dll` facade are bundled |
| V9 | Package dependencies are remapped | manifest pins Entities 1.5.0, Burst 1.8.31, Collections 2.7.1; 7000.0.0a12 resolves built-ins 6.7.0, 2.0.0, 6.7.0 |
| V10 | Hot-path structs are blittable | `Cq`, `Activations`, `AttentionTile`, `MlpBatch`, `MatMulBatch`, `LayerJob` hold only pointers and scalars (`LayerJob` also holds `TClock`, a struct) |
| V11 | net10 DLL is unreferenceable from Unity scripts | `unity recompile ~/Say` with net10 plugin: `CS1705 System.Runtime 10.0.0.0 > 4.1.2.0`; `apiCompatibilityLevel NET_Standard_2_0` offers no .NET 10 script profile (2026-11-02) |
| V12 | ns2.1 waits no longer sleep (N20) | `Pool`/`NeedleHost`/`VoiceHost` use `Spinner` (`SpinWait` → `Yield`, never `Sleep`); stalled-barrier 3.5 → 1.4 ms/round, hot rendezvous 28 → 0.44 µs (CoreCLR) |
| V13 | ns2.1 on CoreCLR is 9.6–17.5x net10 after free wins, was ~103x at 4T | N12 `bench-ns21.sh` golden set (ref/bench/n12-*.json): CoreCLR AFTER 955 ms T4 / 1196 ms T2 / 1750 ms T1 vs ~100 ms net10; BEFORE 10.3 s T4 (H1 settled in band) |
| V14 | V4 sleeps cost ~11x the residual scalar math at 4 threads | N12: BEFORE inverse scaling T1→T4 29.6→218 ms/tok; sleep ≈ 200.6 ms/tok vs 17.6 scalar; Mono ~8.6x (H2 settled) |
| V15 | CoreCLR Editor: Avx2+AvxVnni, Vector 32 B hw-accelerated, Burst present+enabled | `needle.probe/1` in-Editor: `.NET 10.0.10-dev` X64, `avx2/avxVnni true`, `advSimd/dp false`, `vectorByteCount 32`, `burstEnabled 1`, i9-14900K (H3 CoreCLR half settled; IL2CPP half open) |
| V16 | IL2CPP x64 player: Avx2 false, Vector NOT hw-accelerated (16 B), scalar tier, Burst present but no provider installed | `needle.probe/1` from Linux IL2CPP player 2026-11-03: `.NET 10.0.10-dev` X64, `avx2/avxVnni/advSimd/dp false`, `vectorByteCount 16`, `burstEnabled 1`, `simd scalar`, `provider none`; golden set reproduces, failures=0; T1 1813 ms / T4 704 ms per call (`ref/bench/n41-il2cpp-x64-T{1,4}.json`) — H4 settled for x64, H3 IL2CPP half settled |
| V17 | Burst's own CPU detection reports no AVX on AVX2+VNNI hardware | In-Editor `X86.Avx/Avx2/Fma.Is*Supported` → `False/False/False` (i9-14900K; System.Runtime says Avx2+VNNI true); every Burst lane folds to scalar in Editor and both players despite AVX2 in the AOT set — detection, not target, gates the lanes; provider installs clean and runs correct (parity) at scalar speed |
| V18 | Burst provider activates in CoreCLR player (AVX2 lane compiled in-Editor) | CoreCLR player probe `provider burst+burst+burst+burst-mlp`, `avx2 true`; IL2CPP needed `autoReferenced:true` + explicit refs (linker stripped it) — fixed, `provider burst…` in fresh link (`*-burstlinked.json`) |
| V19 | Phase-2 layer member closed as not worth it | N40: fused entry saves ≤1.1 µs/layer (0.13% of decode); barriers stay; `calli` 1.7 ns |
| V20 | Portable provider complete: LUT-GEMV 2.5–3.6x, Kron 2.4x, attention 1.5x, 20× Vector kernels 3–10x | N50–N54 (bit-exact int, parity floats); N51 1.4–1.6x kept as finding |

| ID | Hypothesis | Settled by |
|---|---|---|
| H1 | ~~ns2.1 on CoreCLR is 10–30x slower per call than net10~~ → V13 | N12 (settled 2026-11-03) |
| H2 | ~~V4 sleeps cost more than the scalar math at 4 threads~~ → V14 | N12 + N20 (settled 2026-11-03) |
| H3 | ~~`Vector<T>` hw-accelerated on CoreCLR but not IL2CPP~~ → V15 + V16 | N10 (settled 2026-11-03: CoreCLR 32 B hw, IL2CPP 16 B not hw) |
| H4 | ~~IL2CPP in 7000 reports `Avx2.IsSupported == false` on x64~~ → V16 (arm64/iOS legs still open) | N10 on an IL2CPP player |
| H5 | ~~Burst FP call < 100 ns~~ → HOLDS (48–65 ns in-Editor) | N31 microbench |
| H6 | ~~Burst compiles package source, not precompiled DLL~~ → HOLDS | N31 (`IsCreated`, 6/6 EditMode) |
| H7 | Burst ports bit-identical to net10 AVX2: HOLDS where testable (N32 1164/1164, N33 111606/111606, N35/N37/N38/N39 bitwise; N31 Softmax maxAbsDiff=0). AVX2-lane device proof pending AOT lane | N13 fixtures + N32–N39 |
| H8 | ~~Burst exposes no AVX-VNNI/AVX-512~~ → HOLDS (X86 = Avx..Ssse3 only) | N31 (reflected installed Burst 2.0.0) |

## 2. Target architecture

```
Needle.Inference.dll  (netstandard2.1, one IL binary for CoreCLR, IL2CPP, Mono)
 ├─ engine, grammar, tokenizer, sessions, host            (logic unchanged)
 └─ Kernels.Active : KernelTable  ── blittable struct of delegate* unmanaged[Cdecl]
        ▲ installed once, before Model.Open; null entry = use the reference
        │
  ┌─────┴──────────────┬────────────────────────┬──────────────────────┬────────────────────┐
  Intrinsics provider   Burst provider           Portable provider       Reference
  Needle.Kernels.Net10  Needle.Kernels.Burst     (inside the core DLL)   Scalar.* (today)
  net10 DLL, loaded by  package source asmdef,   Vector<T> float ops,
  reflection when the   [BurstCompile] FPs,      LUT-GEMV integer
  runtime is .NET 8+    X86 + Arm.Neon paths     matmul (N52)
```

- The **net10.0 build of `Needle.Inference` stays** for NativeAOT, the CLI and servers. Its direct `#if NEEDLE_SIMD` branches stay first, so the table adds zero indirection there. The table only overrides when something is installed explicitly.
- **Provider selection** happens at startup, once:
  1. Explicit `Kernels.Install(...)`.
  2. Intrinsics provider, if `Environment.Version.Major >= 8` and the DLL is present.
  3. Burst provider, which installs itself from `[RuntimeInitializeOnLoadMethod(BeforeSceneLoad)]` and `[InitializeOnLoadMethod]`.
  4. Portable provider.
  5. Reference.

  `NeedleRuntime.Describe()` reports the choice, mirroring `needle cpu`.
- **ABI rules:**
  - Blittable arguments only: pointers to the V10 structs, `int`, `float`, `nint`.
  - No managed objects, strings, spans or delegates cross the table.
  - Every entry does at least about 1 µs of work. Per-element calls are forbidden.
  - Entries are pure: no allocation, no static mutable state, safe from any thread.

### 2.1 Kernel table, first cut

| Entry | Signature (pointers to existing structs) | Today's home | Share of decode (plan.md §2.3, 4T) |
|---|---|---|---|
| `MatMulRows` | `(Cq*, Activations*, int tokens, float* y, int ldy, int rowStart, int rowEnd)` | `Quant.MatMulRows` | about 40% (Q/K/V/Gate, Out, engram, head, phi) |
| `Prepare` | `(float* x, int tokens, int cols, int bits, Activations*)` | `Quant.Prepare` | 4% |
| `GatherRow` | `(Cq*, int row, float* dst)` | `Quant.GatherRow` | embedding, engram gather |
| `AttentionTile` | `(AttentionTile*)` | `Attention.Run` | 10% |
| `MergePartials` | `(float* partials, int count, int heads, float* out)` | `Attention.MergePartials` | split attention |
| `HadamardMlp` | `(MlpBatch*, float* residual, float* scratch)` | `HadamardMlp.Apply` | 33% |
| `RowOps` | one entry per row op: `RmsFakeA8`, `CenteredRmsFakeA8`, `GateFakeA8`, `PostNormAdd`, `CombineDiff4`, `Mix4`, `Combine4`, `Taps3`, `Rope`, `ZeroCenteredRms`, `RmsUnit`, `Softmax`, `Dot`, `AddScaled`, `ScaleAdd`, `MultiplyAccumulate`, `QuantizeInt8`, `FakeA8`, `Sinkhorn4x4` | `Ops`, `Quant` | about 10% combined |
| `ArgMaxMasked`, `BestProbability` | `(float*, uint* mask, int words, int length)`, `(float*, int, float)` | `Ops` | per decode step |
| **Phase 2** `LayerMember` | `(LayerJob*, int member)`: the whole `ExecuteLayer` body, with barriers | `EngineCore.ExecuteLayer` | 1 call per member per layer |

## 3. Targets

Measured with N11 runners, Lagos query plus the 10-prompt game set, warm session, the same `needle.bench-e2e/1` JSON everywhere.

| Runtime | Provider | Call latency, 4 threads | Decode step, 4 threads | Today |
|---|---|---|---|---|
| CoreCLR, net10 DLL (Editor, desktop) | direct SIMD | = CLI ± 10% (W targets apply) | = CLI ± 10% | about 100 ms (CLI) |
| CoreCLR, ns2.1 DLL | intrinsics via table | ≤ 1.10x net10 DLL | ≤ 1.10x | 955 ms T4 / 17.6 ms/tok AFTER (N12); Burst installs clean, scalar lanes (V17); portable LUT 2.5–3.6x T1 (N52) |
| IL2CPP x64 player | Burst | ≤ 1.3x net10 on the same box | ≤ 1.3x | T4 421–704 ms scalar band; provider installs `burst×4+mlp` (burst2 leg) but V17 folds lanes — AVX2 lane pending Burst detection fix |
| IL2CPP arm64 (Android, iOS) | Burst NEON (dotprod or widening) | ≤ 1.3x of the NativeAOT CLI on the same device | ≤ 1.3x | unmeasured |
| Mono (Unity 2022/6000 LTS users) | Burst | ≤ 1.5x net10 | ≤ 1.5x | 1409 ms T4 AFTER (N12 headless Mono 6.13); portable LUT 2.5–3x T1 (N52) |
| Any runtime, no provider | portable | ≤ 4x net10 | ≤ 4x | unmeasured |
| Every runtime, main thread | n/a | 0 B per frame; ≤ 20 µs per frame in a release player | n/a | 0 B (W23) |

Correctness, for every provider:

- Integer kernels (`MatMulRows` dot sums, `QuantizeInt8`, attention int8 dots) are **bit-exact** with `Scalar.*`.
- Float kernels pass `ParityTests` (cosine ≥ 0.99995, same argmax, greedy decode equals golden).
- Burst x86 kernels are additionally checked bitwise against the net10 AVX2 kernels (H7). If H7 fails for a kernel, it falls back to the parity bound and the PR says so.

## 4. Work items

Size: S ≤ 1 day, M ≤ 3 days, L ≤ 2 weeks, for one agent. "Files" is the only set of files the item may touch. Anything else needs a note in the PR (plan.md §8 rule 2).

### Track Z: restore speed today

| ID | Item | Files | Depends | Acceptance | Size |
|---|---|---|---|---|---|
| N00 | Ship the net10.0 DLL to CoreCLR targets (Editor, Standalone Win/Linux/macOS arm64) and keep the generated code C# 9-safe (no `IToolSet` adapter, `byte[]` schema). Revert the single-TFM swap in `tools/pack-unity.sh` | `tools/pack-unity.sh`, `unity/com.needle.inference/Runtime/Plugins/**`, sample `.g.cs` | — | `unity recompile` 0 errors in `~/Say`; `NeedleRuntime.Describe()` (or a temporary log of the active tier) shows Avx2 or AvxVnni; Squad sample latency ≤ 0.15 s on the Editor box | S |
| N01 | Dual-DLL package layout: net10 DLL with plugin platform and define constraints for CoreCLR targets, ns2.1 DLL for the IL2CPP targets, the same GUID scheme, no manual swap. Superseded by the single ns2.1 DLL plus providers once N30/N3x land | plugin `.meta` files, `tools/pack-unity.sh` | N00 | A Linux CoreCLR player and an Android IL2CPP player build from one checkout without edits; each logs its tier | S |
| N02 | Generator emits C# 9-safe code when `LanguageVersion < 11`: `static readonly byte[]` instead of `u8`; the adapter only when `IToolSet` resolves (already true). Check the Roslyn version Unity ships: `ForAttributeWithMetadataName` needs 4.3+, otherwise fall back to `CreateSyntaxProvider`. Ship the generator in the package labeled `RoslynAnalyzer`, and stop checking in sample `.g.cs` | `src/Needle.Generators/*`, package `Editor/` or `Analyzers/` | — | Unity compiles both samples from the interfaces alone; generator tests cover C# 9 and C# 12 output | M |

### Track M: measure first (blocks every performance claim)

| ID | Item | Files | Depends | Acceptance | Size |
|---|---|---|---|---|---|
| N10 | Runtime probe in Unity: `FrameworkDescription`, `Vector.IsHardwareAccelerated`, `Vector<byte>.Count`, intrinsics `IsSupported` (by reflection from ns2.1), Burst presence and target, `ProcessorCount`, `Stopwatch.Frequency`, worker count. As an EditMode test and a player scene that writes JSON to `persistentDataPath` | package `Tests/`, `Samples~/Probe` | — | A table in §1 filled for: Editor Linux, Linux CoreCLR player, Windows CoreCLR player, Android IL2CPP arm64, macOS arm64; settles H3 and H4 | S |
| N11 | One runner per runtime, all emitting `needle.bench-e2e/1` JSON: **(a)** a net10 console host forced to reference the ns2.1 build (`SetTargetFramework=netstandard2.1`), giving CoreCLR plus ns2.1; **(b)** Unity's bundled Mono 6.13 running a harness built with its own `csc` against the `netstandard` facade (V8); **(c)** `unity test` EditMode bench; **(d)** a player bench scene for IL2CPP. One script: `tools/bench-ns21.sh <runtime>` | `samples/BenchNs21/*`, `tools/bench-ns21.sh`, package `Tests/` | — | Each runner reproduces golden-set call outputs and prints JSON; (a) and (b) run headless in CI | M |
| N12 | Baseline: run N11 on every runtime at 1, 2 and 4 threads, plus the stage profile (`IStageClock`) and Pool barrier counts. Fill §3 "Today". **No perf PR in this file merges before N12** | this file | N11 | Table committed with raw JSON under `ref/bench/` (git-ignored, attached to the PR) | S |
| N13 | Kernel conformance suite: for every §2.1 entry, golden and random inputs (edge cases: zero rows, ties for rounding, denormals, the max/min int8, odd tail lengths), the reference output, and a provider comparison (bit-exact for integer, bounded for float). Fixture files generated by the net10 kernels, so Burst can be compared bitwise (H7) | `tests/Needle.Tests/Conformance/*`, `tools/conformance_fixtures.*`, package `Tests/` | N30 for the providers (scaffold now against `Scalar` vs net10 SIMD) | Runs in `dotnet test`, on the N11a host, under Mono (N11b) and in `unity test` | M |

### Track A: free wins in the ns2.1 build

Small, independent, each measured with N11a and N11b before and after.

| ID | Item | Files | Depends | Acceptance | Size |
|---|---|---|---|---|---|
| N20 | No `Sleep(1)` in waits: a `Spinner` (exponential `Thread.SpinWait` up to a cap, then `Thread.Yield()`, never `Sleep`) replaces every `NETSTANDARD2_1` `spin.SpinOnce()` | `Pool.cs`, `Hosting/NeedleHost.cs`, `Hosting/VoiceHost.cs` | N11 | `pool_run_barrier` (ns2.1 on CoreCLR) within 1.5x of net10; e2e before/after attached; settles H2 | S |
| N21 | Fast clear: `Unsafe.InitBlockUnaligned` (the `System.Runtime.CompilerServices.Unsafe` package is already referenced; confirm Unity ships the assembly, otherwise use `Span<byte>.Clear` in ≤ 1 GB slices) | `Native.cs` | N11 | Session constructor time and `Reset` cost reported; byte-identical arena state | S |
| N22 | mmap on ns2.1: `MemoryMappedFile` (in the .NET Standard 2.1 surface; confirm at compile) plus `DllImport` `madvise`; fall back to the read path where mapping fails; Android APK assets stay on `FromBytes` | `ModelMemory.cs`, `Native.cs` (`Paging`) | — | `LoadingTests` pass on the ns2.1 build; RSS and open time reported | S |
| N23 | Rounding without `MathF.Round` calls in scalar hot loops: magic-constant round-to-nearest-even (`(x + 12582912f) - 12582912f`, exact for \|x\| < 2²²; inputs are pre-scaled to ≤ 128) | `Scalar.cs` (`RoundTrip`, `QuantizeInt8`, the `FakeA8` family) | N13 scaffold | Bit-identical to the current code on 10⁷ random values including exact .5 ties; scalar kernel time reported | S |
| N24 | `[module: SkipLocalsInit]` actually applies to the ns2.1 build (the polyfill exists; verify the IL `.locals` flags with `monodis`/`ildasm`, and add the module attribute if missing) | `Polyfills.cs`, `Needle.csproj` | — | IL proof in the PR; `Kron`/`Prepare` stackalloc no longer zeroed | S |
| N25 | Game defaults on slow runtimes: Squad sample threads 2 → `min(4, ProcessorCount/2)`; Game spin policy; document `NEEDLE_PIN=0` in the Editor | package `Samples~/*`, README | N12 | Frame-time impact measured at 2 and 4 threads in the Editor | S |

### Track B: Burst provider (IL2CPP and Mono; the big lever off CoreCLR)

| ID | Item | Files | Depends | Acceptance | Size |
|---|---|---|---|---|---|
| N30 | **Kernel table ABI** in the core: `Needle/Kernels/KernelTable.cs` (struct of `delegate* unmanaged[Cdecl]`), `Kernels.Install`, `NeedleRuntime.Describe()`, and rewiring every §2.1 dispatch site to: SIMD (net10 only), then table entry, then reference. Single owner; these lines are frozen afterwards | `Ops.cs`, `Quant.cs`, `Blocks.cs`, new `Kernels/*` | N12 | net10 `bench-e2e` within noise (≤ 2%); all tests green on net10, `NEEDLE_SCALAR=1` and ns2.1; an empty table behaves exactly as today | L |
| N31 | Burst skeleton: asmdef `Needle.Kernels.Burst` in the package (refs `Unity.Burst`, `allowUnsafeCode`); `[BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Strict)]`; installer fetches `BurstCompiler.CompileFunctionPointer(...).Value` and installs raw pointers; one kernel (`Softmax`) end to end. Settle H5, H6 and H8 | package `Runtime/Kernels/*` | N30 | Conformance passes in `unity test`; FP call overhead microbench reported; Editor and IL2CPP player both use it | M |
| N32 | Burst `MatMulRows` T=1, x86: port `Gemv.Rows`/`Block`/`Single` (`TwoBitPlanes`, `FourBitPlanes`, `Avx2Dot`) | `BurstGemvX86.cs` | N31 | Bit-exact vs Scalar; bitwise vs net10 AVX2 (H7); GB/s within 1.2x of net10 in `bench` | M |
| N33 | Burst T≥2 tiles, x86: `Tile4`, `Tile2`, `Pair4`, `Row4`, `Pair2`, `Row2` | `BurstGemvTileX86.cs` | N32 | As N32, for T = 2..64 | M |
| N34 | Burst `MatMulRows`, Arm: port `Neon.MatMulRows` (`DotBlock`, `DotPair`, `DotTile`, dotprod plus the widening fallback) | `BurstGemvNeon.cs` | N31 | Bit-exact; checked on a dotprod and a non-dotprod device (or `IsNeonDotProdSupported` forced off) | M |
| N35 | Burst `Prepare` + `Hadamard` + `QuantizeInt8` + `Planes2/4`, x86 and Neon | `BurstPrepare.cs` | N31 | Bit-exact | M |
| N36 | Burst attention: `Head`, `GroupAvx2`, `ExponentiateAcrossPositions`, `ValuesFloat` (`AccumulateFloatSix`), `MergePartials`, Neon `AttentionTile`. Skip `GroupVnni`/`ValuesInt8` if H8 holds | `BurstAttention.cs` | N31 | Parity bound; int8 dots exact | L |
| N37 | Burst Hadamard MLP: `KronAvx2`, Neon `Kron`, `ConditionWeights`, `PermuteModulateActivate`, `PermuteScale`, `AccumulateOutput`; inherit W12's register blocking if it has landed | `BurstMlp.cs` | N31 | Parity bound; µs per `HadamardMlp.Apply` within 1.2x of net10 | M |
| N38 | Burst row ops (§2.1 `RowOps` list), x86 and Neon, one file | `BurstRowOps.cs` | N31 | Parity bound; exact where the net10 op is exact | M |
| N39 | Burst `DecodeRow`/`GatherRow`, `ArgMaxMasked`, `BestProbability` | `BurstMisc.cs` | N31 | Exact | S |
| N40 | **Phase 2: Burst layer member.** Port `ExecuteLayer` and its stage functions (`HyperNorm` … `Combine`) as one Burst entry taking `LayerJob*`; barriers via `Interlocked` plus a `Pause` spin loop (Burst `Common.Pause()`, verify the name); no managed calls inside | `BurstLayer.cs`, a `LayerJob` layout freeze in `Engine.cs` (one line: `[StructLayout(Sequential)]` plus a size assert) | N32–N39 | Decode µs/token better than the per-kernel table by ≥ 10%, or the item is closed as "not worth it" with numbers | L |
| N41 | IL2CPP player matrix: Android arm64 (dotprod plus a pre-dotprod device), iOS, Windows IL2CPP x64, Linux IL2CPP x64; Burst AOT settings (x86 target set to include AVX2; arm64 armv8.2 dotprod plus the fallback) | `Samples~/Bench`, docs | N32–N39 | §3 IL2CPP rows filled; zero conformance failures on device | M |

**Burst port cookbook** (System.Runtime.Intrinsics → `Unity.Burst.Intrinsics`). Confirm every name against the Burst version N31 finds installed. Keep the instruction order identical to the net10 kernel; that is what makes H7 achievable.

| net10 | Burst |
|---|---|
| `Avx.LoadVector256` / `Avx.Store` (float, int) | `X86.Avx.mm256_loadu_ps` / `mm256_storeu_ps`, `mm256_loadu_si256` / `mm256_storeu_si256` |
| `Avx2.MultiplyAddAdjacent(byte, sbyte)` (vpmaddubsw) | `X86.Avx2.mm256_maddubs_epi16` |
| `Avx2.MultiplyAddAdjacent(short, short)` (vpmaddwd) | `X86.Avx2.mm256_madd_epi16` |
| `Avx2.Shuffle(byte)` (vpshufb) | `X86.Avx2.mm256_shuffle_epi8` |
| `Fma.MultiplyAdd(a, b, c)` / `MultiplyAddNegated` | `X86.Fma.mm256_fmadd_ps(a, b, c)` / `mm256_fnmadd_ps` |
| `Avx.ConvertToVector256Single(int)` | `X86.Avx.mm256_cvtepi32_ps` |
| `Avx2.ConvertToVector256Int16(sbyte*)` / `Int32(sbyte*)` | `X86.Avx2.mm256_cvtepi8_epi16(mm_loadu_si128)` / `mm256_cvtepi8_epi32(mm_loadl_epi64)` |
| `Avx.HorizontalAdd` / `Avx2.PermuteVar8x32` | `X86.Avx.mm256_hadd_ps` / `X86.Avx2.mm256_permutevar8x32_epi32` (`_ps`) |
| `Vector256.Sum` | explicit `hadd` / extract sequence (keep the net10 reduction order) |
| `AvxVnni.MultiplyWideningAndAdd` | none expected (H8): use the `Avx2Dot` sequence |
| `AdvSimd.LoadVector128` / `Store` | `Arm.Neon.vld1q_f32` / `vst1q_f32`, `vld1q_s8`, `vld1q_u8` |
| `AdvSimd.FusedMultiplyAdd(acc, a, b)` (acc + a·b) | `Arm.Neon.vfmaq_f32(acc, a, b)` |
| `Dp.DotProduct(acc, a, b)` | `Arm.Neon.vdotq_s32(acc, a, b)`, guarded by `IsNeonDotProdSupported` |
| `AdvSimd.VectorTableLookup` | `Arm.Neon.vqtbl1q_u8` / `vqtbl1_u8` (the 4-register forms for 64-byte tables) |
| `AddPairwise` / `AddAcross` / `MaxAcross` | `vpaddq_*` / `vaddvq_s32` / `vmaxvq_f32` |
| `Avx2.IsSupported` etc. | `X86.Avx2.IsAvx2Supported`, `X86.Fma.IsFmaSupported`, `Arm.Neon.IsNeonSupported` (branches fold at Burst compile time) |

Burst rules:

- Use `FloatMode.Strict` and never `Fast`; parity depends on no reassociation.
- Use explicit FMA intrinsics rather than relying on contraction.
- No `[NoAlias]` claims that are untrue (`AttentionTile.Output` aliases `Partials` in the chunked path).
- Pointers in, nothing returned but scalars.

### Track C: portable provider inside the ns2.1 DLL

This is for runtimes with neither CoreCLR nor Burst, and it is the reference for "slow but sane".

| ID | Item | Files | Depends | Acceptance | Size |
|---|---|---|---|---|---|
| N50 | `Vector<T>` row ops and MLP: elementwise, RMS, softmax, dot, `Kron`; enabled only when `Vector.IsHardwareAccelerated`; `Vector<T>` has no byte shuffle, so it does not attempt the matmul unpack | `Kernels/Portable.Vector.cs` | N30 | Parity bound; ≥ 3x vs Scalar for those kernels on N11a | M |
| N51 | Tuned scalar matmul: the 4-row × 2-token blocking of `Gemv.Block`/`Pair2`, one activation load per four rows, register accumulators, unrolled by 4 | `Kernels/Portable.MatMul.cs` | N30 | Bit-exact; ≥ 2x vs `Scalar.MatMulRows` at T=1 and T=4 | M |
| N52 | **LUT-GEMV** (T-MAC style). Per token and 128-group, build `T[j][b] = Σp book[(b >> 2p) & 3] · x[p·32 + j]` (2-bit: 32 × 256 int32 = 32 KB per group) and `T[j][b]` over nibble pairs (4-bit). Each packed weight byte becomes one load and one add instead of four multiply-adds, shifts and masks. Tables are shared by every matrix that consumes the same `Activations` (Q, K, V and Gate share `DimAct`) | `Kernels/Portable.Lut.cs` | N51 | Bit-exact (integer sums); speed vs N51 at T=1 on N11a and N11b; keep or close with numbers | M |
| N53 | Scalar `Kron` register blocking (4 × 4 accumulators), mirroring W12 | `Kernels/Portable.Mlp.cs` | N30 | Parity bound; ≥ 2x vs `Scalar.Kron` | S |
| N54 | Scalar attention: 48-wide int dot unrolled, 4 positions per iteration, the V accumulate shared across the GQA group (as `AccumulateFloatSix`) | `Kernels/Portable.Attention.cs` | N30 | Exact dots, parity bound overall | S |

### Track D: do less work (multiplies every provider)

| ID | Item | Files | Depends | Acceptance | Size |
|---|---|---|---|---|---|
| N60 | Eval jump-forward and prompt lookup (already implemented, `NeedleOptions.JumpForward` and `PromptLookup`, default off) with `needle eval tests/eval --compare`, using the policy names from `EvalPolicies.Settings` | none (data only), then `HostOptions` defaults | — | No accuracy drop on the eval set; passes per call reported; defaults flipped for game hosts only if W28 passes (plan.md §10.4) | S |
| N61 | Reasoning budget for commands: eval `MaxReasoningTokens` ∈ {0, 8, 16, 32, 96}; for 0, fold `thinkEndForced` into the prompt prefill so the decision comes from the prefill logits (saves one pass) | `NeedleSession.cs` (`Reason`, `EncodePrompt`) | — | Accuracy per budget committed; a `ReasoningBudget` host option; reasoning is 54% of decode tokens (plan.md §2.4), so cap 0 is the single largest lever if accuracy holds | M |
| N62 | Stateless fast path: skip the `callClosing`/`responseClosing` feed when the host rewinds after every query. This changes pooled confidence, so recalibrate `RefuseBelow` from the eval curve | `NeedleSession.cs`, `Hosting/NeedleHost.cs` | N61 | One pass fewer per call; calibration error not worse than today | S |
| N63 | Batched host: `NeedleHost` drains up to 8–16 queued requests into `BatchedSession` lanes (W26). Decode is weight-bandwidth-bound (17.5 MB per token), so lanes share the stream | `Hosting/NeedleHost.cs` | N20, W26 | 64-NPC LoadProof: replies/s ≥ 6x single-lane at equal threads; single-request latency ≤ 1.1x | M |
| N64 | Shipped primed prefix: build-time `ExportPrimedPrefix` (Editor menu plus `needle prime`), loaded with the `primedPrefix` constructor; requires `AutoDate=false` (plan.md §10.5) | CLI, package `Editor/`, host options | W09 | Session ready ≤ 50 ms on every runtime, including portable | S |

### Track E: bridge for runtimes with nothing fast

Optional. N00 makes this unnecessary on CoreCLR platforms. Keep it for users on Mono-only Unity LTS versions until Track B lands.

| ID | Item | Files | Depends | Acceptance | Size |
|---|---|---|---|---|---|
| N70 | `needle serve`: NativeAOT process, one JSON request per stdin line, one reply per stdout line; Unity client component | `src/Needle.Cli/Serve.cs`, package `Runtime/OutOfProcess/*` | — | Round trip ≤ net10 latency + 1 ms; child killed when Unity exits | S |
| N71 | Shared-memory `NeedleChannel`: allocate the channel block in a `MemoryMappedFile`; rings are offset-based already, but the wake function pointer in the block is process-local, so replace the wake with polling (or eventfd/futex on Linux, a named event on Windows) | `Hosting/NeedleChannel.cs`, `Hosting/SharedChannel.cs` | N70 | Game-side code unchanged; zero-copy round trip ≤ 20 µs | M |

### Track F: packaging, and readiness for the December CoreCLR release

| ID | Item | Files | Depends | Acceptance | Size |
|---|---|---|---|---|---|
| N80 | Split the package: `com.needle.inference` (core plus MonoBehaviour, no Entities) and `com.needle.inference.entities` (the `ISystem` sample and glue); V9 shows the Entities pin forces built-in remaps | `unity/*` | — | A fresh URP project imports the core package with zero extra dependencies | S |
| N81 | Model delivery per platform: Android StreamingAssets are inside the APK, so use `UnityWebRequest` → `Model.FromBytes`, or copy once to `persistentDataPath` and mmap; iOS and desktop mmap directly (W05 overlap) | package `Runtime/ModelLoader.cs` | N22 | Loads on Android, iOS, Windows, macOS, Linux players | S |
| N82 | Follow Unity's compile profile: when Unity exposes C# 11+ and the .NET 10 reference profile, N02's LanguageVersion check re-enables `u8` and the `IToolSet` adapter with no fork | generator tests | N02 | A test per language version | S |
| N83 | Thread budget versus Unity's job workers: measure frame time with Needle at 1, 2, 4 threads alongside a Burst-heavy Entities scene; pick defaults (BelowNormal priority, threads = `min(4, cores − Unity workers)`) | `Samples~/*`, docs | N12 | Defaults justified by a table of frame time p99 | S |

## 5. Dependency graph and parallel lanes

```
N00 ─ N01                       N02 ─ N82
N10, N11 ─ N12 ─┬─ N20 N21 N22 N23 N24 N25   (Track A, all parallel)
                └─ N30 ─┬─ N31 ─┬─ N32 ─ N33          ┐
                        │       ├─ N34 N35 N36 N37    ├─ N40 ─ N41
                        │       └─ N38 N39            ┘
                        ├─ N50 N51 ─ N52, N53, N54    (Track C, parallel)
                        └─ N13 (providers)            (scaffold starts before N30)
N60 N61 ─ N62    N63 (after N20, W26)    N64 (after W09)
N70 ─ N71        N80 N81 N83              (any time)
```

**File ownership** (one writer at a time; anything else waits or rebases):

| File | Owner | Rule |
|---|---|---|
| `Ops.cs`, `Quant.cs`, `Blocks.cs` dispatch lines | N30 | Frozen after N30 merges; new kernels go into `Kernels/*` |
| `Kernels/Portable.*.cs` | one file per N5x item | — |
| package `Runtime/Kernels/Burst*.cs` | one file per N3x item | `BurstLayer.cs` belongs to N40 only |
| `Pool.cs`, `Hosting/*` | N20, then N63, then N62/N71 | sequential |
| `Native.cs` | N21, then N22 (the `Paging` part) | sequential |
| `ModelMemory.cs` | N22 | — |
| `Scalar.cs` | N23 | Track C must not edit it; Track C adds new files and keeps `Scalar` as the reference |
| `NeedleSession.cs` | N61, then N62 | sequential; coordinate with open W items |
| `src/Needle.Generators/*` | N02 | — |
| `tools/pack-unity.sh`, plugin `.meta` files | N00, then N01 | — |

## 6. Agent protocol

This extends plan.md §8; that section's rules still apply.

1. Pick an item whose dependencies are merged. Branch `n<NN>-<slug>`. Touch only the item's files.
2. **Write the failing check first:** a conformance case, a test, or the benchmark row the item must move.
3. Run every gate:
   - plan.md §7: `dotnet test`, `NEEDLE_SCALAR=1 dotnet test`, the ns2.1 build, the AOT publish.
   - N13 conformance on the N11a host, and under Mono (N11b) for Track A and C items.
   - `unity test` EditMode for Track B items.
4. **Performance PRs attach before/after JSON** from `tools/bench-ns21.sh` for every runtime the item claims to affect, at 1 and 4 threads, plus net10 `bench-e2e` to prove no regression larger than 3% there.
5. A claim without a number is not accepted. An item that does not reach its acceptance is still merged as a finding if it is cheap to keep. Otherwise close it with the numbers, and record the learning in §1 of this file.
6. Update §1 (move an H to V) and §3 (fill "Today") in the same PR that produced the evidence.

**Swarm sizing.** The work is wide but not infinitely wide.

- The critical path is N11 → N12 → N30 → N31 → N32. About 25 implementation items can run at once after N31.
- Put extra agents on roles that make the swarm safe, not on more implementation:
  - **Reproducers:** rerun every claimed benchmark on a second machine before merge. Latency on the shared dev box swings 3x (plan.md §2.6).
  - **Fuzzers:** grow N13 inputs, especially rounding ties, tails and aliasing.
  - **Reviewers:** check the Burst ports instruction by instruction against the net10 kernel (cookbook above).
  - **Device runners:** own one physical device each for N41.
- Never put two agents on the same file (§5).

## 7. First 48 hours

1. **N00** (one agent, S): the Editor is fast again today.
2. **N02** (one agent, M): no more hand-edited `.g.cs`.
3. **N10, N11** (two agents): probes and runners; **N12** as soon as N11 lands.
4. **N20, N21, N23, N24** (four agents, S each): the cheap ns2.1 wins, each with before/after numbers from N11a and N11b.
5. **N30** (one strong agent, L), with **N13** scaffolding in parallel.
6. **N60, N61** (two agents, S/M): eval data only; no default changes until W28 says so.

Then fan out N31 → N32–N39, N50–N54, N63 and N64.

## 8. Risks and open questions

1. **IL2CPP and net10 IL.** If IL2CPP in 7000 accepts the net10 DLL *and* lights up intrinsics (the opposite of H4), Track B shrinks to Mono-only users. N10 decides this before N32 starts. Do not start the Burst ports before N10 reports on an IL2CPP player.
2. **Burst in this alpha.** plan.md §10.3 already saw Burst BC0102 internal errors in Entities code. If Burst function pointers misbehave under CoreCLR or IL2CPP in 7000, fall back to a native C kernel library (the same table, built with clang per platform). Treat that as a new item, not a silent swap.
3. **Parity across providers.** Float kernels that differ by one ULP can flip a greedy argmax on knife-edge prompts (see the Whistle thread-count note, plan.md §2.9). N13 plus `ParityTests` are the guard. A provider that cannot meet the bound is not installed by default.
4. **WebGL.** No threads by default, and wasm SIMD depends on Unity settings. Out of scope until the IL2CPP desktop and mobile rows of §3 are green.
5. **The December CoreCLR release** may change the compile profile, plugin rules or Burst. N82 keeps the generator adaptive; N01 is the only packaging piece that may need rework.
