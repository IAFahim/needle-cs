# IL2CPP bench players (N41): Linux x64 + Android arm64 procedure

Status (2026-11-03): **Linux x64 done, Android open.** `linux-il2cpp` was
installed this session and `BenchBuild.PerformLinuxIl2cpp` built
`out/bench-ns21/BenchNs21` from the live `~/Say` Editor (driven via
`unity command script run`, since `unity build` refuses a project with an
open Editor; `tools/bench-ns21.sh player` now passes the project
positionally after a `--project-path` fix). Two environment shims were
needed on this Omarchy host, both inside the user-owned Unity sysroot
cache (no repo impact, cache re-download restores stock): `clang++`/`ld.lld`
could not start (`libncurses.so.6` absent — only `libncursesw` ships; the
needed symbols are terminfo-only, so DT_NEEDED was repointed at the
present `libtinfo.so.6`), and `ld.lld` needed `libxml2.so.2` (copied the
ICU-free 2.9.9 from Unity's own `linux-x86` sysroot into the llvm
sysroot's `lib/`). N11d closed by these runs; H4 settled for x64 (below).

## Linux x64 results (2026-11-03, i9-14900K, 32 cores)

Runs: golden set, reps 2, warmups 1, model
`~/Say/Assets/StreamingAssets/needle3.cact` (md5-identical to
`ref/hf/needle3.cact`). Archives: `ref/bench/n41-il2cpp-x64-T1.json`,
`ref/bench/n41-il2cpp-x64-T4.json`, `ref/bench/n41-il2cpp-x64-probe.json`.
The bench player now also writes `needle-probe.json` next to `--out`
(`BenchPlayer.WriteProbe`, self-contained mirror of `Samples~/Probe`;
one player build covers bench + probe, no second Probe-scene build).

| | T1 | T4 |
|---|---|---|
| total/call, median | 1813 ms | 704 ms |
| prefill, median | 421 ms @ 47 tps | 160 ms @ 122 tps |
| decode, median | 30.4 ms/tok (34 tps) | 12.0 ms/tok (84 tps) |
| golden calls | `get_weather`; `set_timer,play_music` | same |
| failures / exit | 0 / 0 | 0 / 0 |

Probe (`needle.probe/1`): `frameworkDescription` `.NET 10.0.10-dev`,
`processArchitecture` `X64`, `vectorIsHardwareAccelerated` **false**,
`vectorByteCount` **16**, `avx2` **false**, `avxVnni` false, `advSimd`/`dp`
false, `burstPresent` true (`0.0.0.0`, `burstEnabled` 1),
`processorCount` 32, `simd` `scalar`, `provider` **`none`**
(reference/Scalar — N31 Burst provider has not landed).

H4 verdict (x64): **settled — IL2CPP reports `Avx2.IsSupported == false`
and `Vector<T>` is not hardware-accelerated** (128-bit `Vector<byte>.Count`
of 16, i.e. scalar fallback width). Scalar tier: IL2CPP T1 1813 ms vs
CoreCLR ns2.1 AFTER T1 1750 ms (`ref/bench/n12-after-coreclr.json`),
IL2CPP T4 704 ms vs CoreCLR T4 955 ms — same band, no IL2CPP cliff on
this box. The §3 "≤ 1.3x net10" comparison needs the net10 number on the
same box (not re-measured here); scalar-vs-CoreCLR above is the
reference point N31 will improve on.

## Unblock (one command, ~63 MB)

```sh
unity install-modules --editor-version 7000.0.0a12 -m linux-il2cpp --accept-eula
```

`BenchBuild.PerformLinuxIl2cpp` needs no change: scene path
`Packages/com.needle.inference/Tests/Bench/BenchPlayerScene.unity` resolves
(the `~/Say` project references the package by `file:` path) and it already
sets `ScriptingImplementation.IL2CPP` for `StandaloneLinux64`.

## Linux x64 run (repeat)

```sh
cd /home/i/GitHub/needle-cs
NEEDLE_THREADS=4 NEEDLE_REPS=2 NEEDLE_WARMUPS=1 NEEDLE_SETS=golden \
  tools/bench-ns21.sh player
```

This builds to `out/bench-ns21/BenchNs21` via
`Needle.Bench.Tests.BenchBuild.PerformLinuxIl2cpp`, runs it headless with
`--model/--tools/--game-tools/--out`, and prints `needle.bench-e2e/1` JSON.
Note: with the `~/Say` Editor open, `unity build` refuses; drive the build
through the live Editor (`unity command script run` + `PerformLinuxIl2cpp`)
or close the Editor first. The player writes `needle-probe.json` next to
`--out` itself (see results section), so no separate Probe-scene build.
Archive the result as `ref/bench/n41-il2cpp-x64-T<threads>.json` plus the
co-written probe JSON. Golden-set calls must reproduce
(Lagos → `get_weather`, timer+music → `set_timer`+`play_music`), exit 0.

## Burst-provider leg (N01 player evidence, 2026-11-03)

Both players below were built from the live `~/Say` Editor via
`unity command script run` (`BenchBuild.PerformLinuxIl2cpp` /
`PerformLinuxCoreClr`), then run headless with `--model/--tools/
--game-tools/--sets golden --reps 2 --warmups 1`. Golden-set calls
reproduce on all four runs (failures=0, exit 0). Archives:
`ref/bench/n41-il2cpp-x64-T{1,4}-burst.json`,
`ref/bench/n41-coreclr-x64-T{1,4}-burst.json`, plus
`n41-il2cpp-x64-probe-burst.json` / `n41-coreclr-x64-probe-burst.json`.

| | IL2CPP x64 player | CoreCLR x64 player |
|---|---|---|
| binary | `out/bench-il2cpp/BenchNs21` (fresh 03:56 link from current sources; the `out/bench-ns21/BenchNs21Burst` stub of 03:42 reused a stale pre-Burst `GameAssembly.so`) | `out/bench-ns21/BenchNs21CoreClr` |
| T1 total/call (median) | 1259 / 1459 ms | 1908 / 2220 ms |
| T4 total/call (median) | 493 / 536 ms | 1000 / 1212 ms |
| T4 decode (median) | 9.2 / 8.5 ms/tok | 19.9 / 20.0 ms/tok |
| T4 prefill (median) | 126 / 103 ms | 205 / 193 ms |
| probe `avx2`/`avxVnni` | false / false | **true / true** |
| probe `vectorByteCount` (hw) | 16 (false) | **32 (true)** |
| probe `burstPresent`/`burstEnabled` | true / 1 | true / 1 |
| probe `simd` | scalar | scalar |
| probe `provider` | **`none`** | **`burst+burst+burst+burst-mlp`** |

Verdict: **the Burst provider activates in the CoreCLR player and does
not activate in the IL2CPP player.**

- CoreCLR ships `Needle.Kernels.Burst.dll` (plus `Unity.Burst.dll` and both
  `Needle.Inference` + `Needle.Inference.Net10.dll`) in
  `BenchNs21CoreClr_Data/Managed/`, so the four `BeforeSceneLoad`
  `AutoInstall` roots fire and compose the provider string. (Lane is not
  probed in players — the probe has no lane field; in-Editor the compiled
  reporter says `burstLane=scalar`.)
- IL2CPP strips `Needle.Kernels.Burst.dll`: the player's `Managed/` dir holds
  `Needle.Bench/Inference/Unity.dll` but no Burst assembly. Corroborated on a
  fresh full link from current sources (`out/bench-il2cpp/BenchNs21`,
  `GameAssembly.so` 03:56): `strings GameAssembly.so | grep -ci burst` is 7,
  all `Unity.Burst` runtime plumbing, zero `Needle.BurstKernels` strings, and
  `Managed/` again lacks the assembly, so the four `RuntimeInitializeOnLoads.json`
  roots (`BurstAttention`,
  `BurstGemvNeon`, `BurstKernelInstaller`, `BurstMlpInstaller` `AutoInstall`)
  are registrations without code. Cause: `Needle.Kernels.Burst` is
  `autoReferenced:false` and nothing in the player-reachable graph
  (`Needle.Bench` refs only `Needle.Unity`) statically references it, so
  IL2CPP drops it. Player logs confirm — zero `[needle]` lines and zero
  `Burst kernel install failed` warnings in `il2cpp-burst-T1.log`
  (`grep -c` 0/0; same 0/0 in both CoreCLR logs, where success is silent
  by design): the installer neither ran nor failed, it was absent.
- Note: IL2CPP T4 (493 ms) beats CoreCLR T4 (1000 ms) with `simd` scalar on
  both — same source tree, so the delta is machine-state/codegen (clang
  auto-vectorizes the IL2CPP C++; single-run numbers, and a GiBenchmarks
  load was pinning the box during these runs), not Burst. Needs a
  Reproducer re-run before any claim.

## Burst-provider leg, burst2 (N01 follow-up, 2026-11-03)

Follow-up build `out/bench-ns21/BenchNs21Burst2` fixes the strip and sets the
AOT target, then re-runs golden T1+T4. Archives:
`ref/bench/n41-il2cpp-x64-T{1,4}-burst2.json`,
`n41-il2cpp-x64-probe-burst2.json`. Two repo touches, both in
`Tests/Bench/`: `Needle.Bench.asmdef` gains `Needle.Kernels.Burst` (the
`BenchPlayer.cs` explicit `EnsureInstalled`/`Install` calls are the load-bearing
half — the asmdef reference alone is pruned from the compiled `Needle.Bench`
when no Bench source names a Burst type, verified via
`Assembly.GetReferencedAssemblies()` in-Editor). Burst AOT x86 set to AVX2 in
`~/Say/ProjectSettings/BurstAotSettings_StandaloneLinux64.json`
(`CpuTargetsX64=64`, min=max=6; default was `72` = SSE2|AVX2, min/max unset —
AVX2 was already in the default set, so the baseline-target hypothesis for the
scalar lane was wrong). `BenchBuild.cs` untouched; `Editor green` (recompile clean).

| | burst2 (provider burst x4) | provider-none build | net10 CLI (avx2) |
|---|---|---|---|
| T1 total/call med | 1153 / 1342 ms | 1259 / 1459 ms | 64 / 75 ms |
| T1 decode med | 20.9 / 19.9 ms/tok | 23.3 / 21.8 ms/tok | 1.18 / 1.11 ms/tok |
| T4 total/call med | 428 / 528 ms | 493 / 536 ms | 30 / 39 ms |
| T4 decode med | 8.36 / 8.36 ms/tok (rerun 8.55 / 8.21) | 9.16 / 8.50 ms/tok | 0.63 / 0.67 ms/tok |
| probe `provider` | **`burst+burst+burst+burst-mlp`** | `none` | n/a (direct SIMD) |
| player `[needle]` line | `tier=Scalar provider=burst+burst+burst+burst-mlp table=installed arch=X64 threads=32 burstLane=scalar burstEnabled=1` | absent | n/a |

Verdict: **activation fixed, lane not — no headline.** The provider installs
(managed link proven: `Needle.Kernels.Burst.cpp` in `il2cppOutput/`,
`Managed/Needle.Kernels.Burst.dll` in the staging `BackUpThisFolder`, roots
present, zero install warnings, golden reproduces, failures=0) but the
compiled lane stays `scalar` and T4 decode sits in the scalar band (~8.4 ms
vs net10 ~0.65 ms, ~12.9x; vs provider-none ~8.8 ms, same band — T4 rerun
±3% confirms stability, not a one-off).

- Exact AOT reason: Burst AOT emitted **zero** `Needle.BurstKernels` code in
  three consecutive player builds. `lib_burst_generated.so` holds Entities
  jobs only (fresh AOT library `507a03db…` in `Library/BurstCache`: 12/12
  `NeedleSystem`, 0/12 `BurstSoftmax`/`BurstMlp`/`BurstAttention`/…;
  `BurstCache`-wide grep for `BurstSoftmax` is empty — JIT and AOT alike
  never compiled these methods). The function pointers still resolve (no
  player-side warnings) and execute with managed-gate semantics: IL2CPP
  reports `Avx2.IsSupported == false`, so the un-Burst-folded
  `IsAvx2Supported` gate reads false and the scalar branch runs — correct
  output, scalar speed. Player ILPP *does* process the assembly (entry points
  registered), yet AOT's entry set excludes every kernel; in the Editor
  pipeline ILPP never touches the assembly at all (pre/post bytes identical),
  which is why the Editor JIT logs "`not a known Burst entry point`" and
  falls back scalar there too. Follow-up needs a Runtime-side reproducer
  (e.g. whether a `[BurstCompile]` job in the same assembly AOT-compiles),
  outside this ticket's touch zone.

## Android arm64 run (hardware runs)

1. Install: `unity install-modules --editor-version 7000.0.0a12 \
   -m android -m android-sdk-ndk-tools -m android-open-jdk-17.0.18+8 --accept-eula`
   (Android Build Support ships its IL2CPP toolchain; ~2.2 GB + 1 GB tools).
2. Target: `Android`, `ScriptingImplementation.IL2CPP`, `ARM64` only
   (uncheck ARMv7). Player Settings → Burst AOT: enable, optimization
   level 3. `ApiCompatibilityLevel` stays `.NET Standard 2.1`.
3. Build the same `BenchPlayerScene` (add an `Android` twin of
   `PerformLinuxIl2cpp` in `BenchBuild.cs` switching to
   `BuildTargetGroup.Android`/`BuildTarget.Android` if `unity build --target`
   does not cover it).
4. Model delivery: `StreamingAssets` are inside the APK — load at runtime via
   `UnityWebRequest` → `Model.FromBytes`, or copy once to
   `Application.persistentDataPath` and mmap (N81/N22).
5. Run two devices: one with NEON dotprod (`IsNeonDotProdSupported`), one
   pre-dotprod (force the widening fallback path). Connect with `adb`,
   launch with the same `--model/--tools/--out` args (paths under
   `/sdcard/Android/data/<pkg>/` or `persistentDataPath`), then
   `adb pull <persistentDataPath>/bench-ns21.json ref/bench/n41-il2cpp-arm64-<device>.json`.
   Also pull the `needle.probe/1` JSON (`advSimd`, `dp`, `vectorByteCount`).
6. Acceptance per device: zero conformance failures, golden calls reproduce,
   §3 rows filled (≤ 1.3x net10 on the same box for x64; arm64 vs NativeAOT
   CLI on the same device).
