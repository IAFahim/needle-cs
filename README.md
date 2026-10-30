# needle-cs

[![CI](https://github.com/IAFahim/needle-cs/actions/workflows/ci.yml/badge.svg)](https://github.com/IAFahim/needle-cs/actions/workflows/ci.yml)

Zero-allocation, unsafe C# inference engine for [Needle 3](https://huggingface.co/Cactus-Compute/needle3) (tool calling, extraction, retrieval, embeddings) and [Whistle](https://huggingface.co/Cactus-Compute/whistle) (speech-to-text).

Reads `.cact` model files directly in place with AVX2, ARM64 NEON (SDOT with a widening fallback for CPUs without FEAT_DotProd), and portable scalar backends, selected at runtime per core. Supports NativeAOT, standard .NET (`net10.0`), `netstandard2.1`, and Unity/Burst.

## Status

Early but usable for a showcase: everything below works on Linux x64 today and is covered by 638 passing tests (Linux x64, AVX2 and scalar).

| Area | State |
|---|---|
| Tool calling, extraction, embeddings, voice to tool call | Working, golden-parity tested against a numpy reference |
| Loading models | Portable file mapping (`Model.Open`), `FromBytes`, `FromMemory`, `FromStream`. Verified on Linux; Windows and macOS code exists but has not been run |
| Threading | `ThreadPolicy` presets, the library never pins your calling thread, workers park when idle, `Pause`/`Resume` |
| Speed | Faster than the upstream native engine on the reference VM (73-84 ms vs 95-129 ms per call, 4 threads). A further 3x is planned, see below |
| Per-call allocations | About 6 KB of managed allocations per call today. Zero-allocation API and typed C# calls are planned |
| Voice | Whistle transcription matches the upstream library on record, test and hotword; long25 keeps the queen passage but differs by one short phrase (issue #38). Beam search is on by default (`BeamWidth = 3`), `BeamWidth = 1` gives faster greedy decoding |
| Evaluating tool sets | `needle eval` with 208 game prompts, see below |
| Unity | Sample package exists, the Entities `ISystem` version is planned |

The roadmap lives in [plan.md](plan.md), and every work item is a GitHub issue. Start at the tracking issue [#31](https://github.com/IAFahim/needle-cs/issues/31).

## Try it in two minutes

```sh
tools/fetch.sh
dotnet publish src/Needle.Cli -c Release -o out/aot
```

```sh
NEEDLE_THREADS=4 out/aot/needle call ref/hf/needle3.cact tools/tools.json "what's it like in Lagos right now?" "set a timer for 12 minutes and play some Miles Davis"
```

```sh
NEEDLE_THREADS=4 out/aot/needle run ref/hf/needle3.cact tools/tools.json "what's it like in Lagos right now?" 20
```

The first command prints one JSON result per query: the chosen tools, their arguments, the model's reasoning and a confidence score. The second repeats a query and reports prefill and decode timings. Voice works the same way: `out/aot/needle voice ref/hf/needle3.cact ref/hf/whistle/whistle.cact tools/tools.json ref/audio/record.wav`.

These steps need no golden fixtures. Only `dotnet test` does.

## Features

- **Needle 3 Tool Calling:** Schema-driven byte-level pushdown automaton grammar constraint, multi-turn sessions, confidence scoring, top-K tool retrieval, structured extraction, and 3072-d embeddings.
- **Whistle Speech-to-Text:** 16 kHz WAV resampling, language detection/forcing, keyword biasing, DTW word timestamps, long-audio windowing, streaming, and direct audio-to-tool-call (`VoiceSession`).
- **Portable & Embeddable:** Full scalar fallback for platforms without SIMD, `Model.FromBytes` for platforms without `mmap`, dedicated-thread `Task` wrappers (`NeedleService`, `SpeechService`), and automatic placement of worker threads on the fastest physical cores (Linux; `NEEDLE_PIN=0` disables, `NEEDLE_NODP=1` forces the NEON widening-fallback tier).

### Unmanaged channel (game thread → worker, Burst-callable)

`NeedleChannel` is two lock-free MPMC rings over unmanaged memory (blittable UTF-8 payloads, monotonic tickets, per-ticket cancellation). `NeedleHost` owns the model/session on its own worker thread; submit/receive are pointer math plus `Interlocked`/`Volatile` — no managed calls, no allocation — so Burst-compiled game code can hand off work without touching the managed API:

```csharp
using var host = new NeedleHost();
var channel = host.Start(new HostOptions { ModelPath = "needle3.cact", ToolsJson = tools });
channel.TrySubmitQuery("set a timer for 12 minutes"u8, out var ticket);
// later, e.g. next frame:
if (channel.TryReceive(out var reply) && reply.Ticket == ticket)
    Handle(reply); // reply.Calls / reply.Response are fixed-size UTF-8 buffers
host.Cancel(ticket); // cooperative cancellation per ticket
```

## Quick Start

### Minimal Example

```csharp
using Needle;

const string tools = """
[{"name":"set_timer","description":"Set a countdown timer.",
  "parameters":{"type":"object","properties":{"minutes":{"type":"integer"}},"required":["minutes"]}}]
""";

using var model = Model.Open("needle3.cact");
using var session = new NeedleSession(model, tools, new NeedleOptions { Threads = 4 });

var result = session.Complete("set a timer for 12 minutes");
foreach (var call in result.FunctionCalls)
    Console.WriteLine($"{call.Name} {call.ArgumentsJson}");   // set_timer {"minutes":12}
```

### Tool Calling (multi-turn)

```csharp
using var model = Model.Open("needle3.cact");
using var session = new NeedleSession(model, toolsJson, new NeedleOptions { Threads = 4 });

var result = session.Complete("set a timer for 12 minutes and play some Miles Davis");
foreach (var call in result.FunctionCalls)
    Console.WriteLine($"{call.Name} {call.ArgumentsJson}");

var followUp = session.CompleteWithToolResult("{\"status\":\"ok\"}");
Console.WriteLine(followUp.Response);
```

### Threading in a game

```csharp
var options = new NeedleOptions { Threading = ThreadPolicy.Game.WithThreads(3) };
using var session = new NeedleSession(model, toolsJson, options);
session.Pause();
session.Resume();
```

Creating a session never changes the affinity or priority of the thread that creates it. Worker threads park shortly after finishing work (about 300 microseconds by default, 50 for `ThreadPolicy.Game`) and `Pause()` makes calls run inline on the caller while the workers sleep.

```csharp
var budgeted = new NeedleOptions { Threading = ThreadPolicy.Game.WithThreads(3), FrameBudget = FrameBudget.FromMilliseconds(4) };
```

`FrameBudget` caps the decode work of one call by token count, by milliseconds, or both; an over-budget call fails with `FrameBudgetExceeded` and leaves the session ready for the next frame. Prefill and tool shortlisting run uncapped.

### Speech-to-Text & Voice-to-Tool

```csharp
using var speech = Model.Open("whistle.cact");
using var whistle = new Whistle(speech, threads: 4);

var pcm = Wav.Read("clip.wav");
var transcript = whistle.Transcribe(pcm, keywords: new[] { "Omeprazole" }, wordTimestamps: true);

using var voice = new VoiceSession(model, speech, toolsJson);
var voiceCall = voice.Complete(pcm);
```

## Performance

Measured on a 4-core Sapphire Rapids VM, NativeAOT, 4 threads, "what's it like in Lagos right now?" with the three-tool test set:

| Engine | Per call | Decode |
|---|---|---|
| needle-cs | 73-84 ms | 1.37-1.55 ms/token |
| Upstream `libneedle3.so` 3.3.3 | 95-129 ms | about 1.75 ms/token |

The 4-thread numbers include the faster Hadamard MLP (W12). The detailed profile and the plan to reach 25 ms per call are in [plan.md](plan.md).

### Reference numbers on an i9-14900K, NativeAOT

| Workload | Latency / Speed | Peak RSS (Private) |
|---|---|---|
| Tool calling, 19-prompt / 56-token decode (8 threads) | ~46 ms / query (1365 tok/s decode) | 56 MB (16 MB) |
| Whistle 20s clip (8 threads) | ~60 ms (3900 tok/s decode) | 54 MB (32 MB) |
| Voice-to-tool (4.5s clip) | ~65 ms | 90 MB (34 MB) |

The full parity suite (golden logits, greedy decode, tokenizer) passes on AVX2, NEON (both dot-product tiers, validated under qemu-aarch64), and scalar.

ARM tiers: the NEON backend runs an SDOT (`Dp`) path where FEAT_DotProd exists and a widening fallback (`NEEDLE_NODP=1`) elsewhere; both tiers are validated under qemu-aarch64 (`NeonTests`, including a history-backed `Taps3` check matching production use). .NET 10 exposes `AdvSimd`/`Dp`/`Rdm` but no i8mm class (`SMMLA`/`USMMLA`/`USDOT`), so an i8mm tier is blocked on the runtime. `Sve`/`Sve2` exist but are experimental (`SYSLIB5003`) with no int8 matrix path, and 128-bit SVE2 is roughly NEON-width on target Android cores, so there is deliberately no SVE2 path.

## Build & Test

Requirements: the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) (the 8.0 SDK fails with `NETSDK1045`), Python 3 with `numpy` and `sentencepiece`, and `curl`.

```sh
# 1. Fetch the pinned model, tokenizer, upstream wheel, Whistle and audio clips into ref/ (SHA-256 checked against tools/ref.lock)
tools/fetch.sh

# 2. Generate the golden fixtures (numpy reference forward pass)
pip install numpy sentencepiece soxr
python3 tools/golden.py               # about 15 CPU minutes
python3 tools/golden_confidence.py    # needs the output of golden.py; takes seconds
python3 tools/golden_whistle.py       # Whistle stage goldens; soxr resamples the 48 kHz clips; takes seconds

# 3. Run the 638 tests (under a minute)
dotnet test tests/Needle.Tests -c Release
NEEDLE_SCALAR=1 dotnet test tests/Needle.Tests -c Release
```

The greedy decode budget is `DECODE_BUDGET` in `tools/golden.py` (`python3 tools/golden.py 64` overrides it). The script records the budget it used in `ref/golden/golden.json` and `ParityTests.GenerationMatchesReferenceGreedyDecode` decodes exactly that many tokens, so the fixture and the test cannot drift. It also records the SHA-256 of the model it used, and the tests fail with "fixtures were generated from a different model" if the loaded model differs. To adopt a newer upstream model on purpose, follow "Refreshing the upstream model" in [CONTRIBUTING.md](CONTRIBUTING.md).

The Whistle stage goldens come from a second numpy reference, `tools/golden_whistle.py`. It reads `whistle.cact` and the clips in `ref/audio`, and writes the log-mel features, stem output, encoder layers, encoder output, cross-attention memory and the first decoder logits to `ref/golden/whistle/<clip>/`. `WhistleStageTests` compares the C# stages against them. It needs `pip install numpy sentencepiece soxr` (soxr resamples the 48 kHz clips) and takes seconds. Whistle decodes with beam search (width 3, mean log-probability), which matches the upstream library on all four clips.

```bash
python3 tools/golden_whistle.py
```


```sh
# Build NativeAOT CLI
dotnet publish src/Needle.Cli -c Release

# Build Unity netstandard2.1 DLL
dotnet build src/Needle -f netstandard2.1 -c Release
```

## Licensing and model files

- **This repository's code license is pending.** The owner has not yet chosen one, so there is no `LICENSE` file. See [CONTRIBUTING.md](CONTRIBUTING.md).
- **Model weights are not in this repository.** `needle3.cact` and `whistle.cact` come from the [Needle 3](https://huggingface.co/Cactus-Compute/needle3) and [Whistle](https://huggingface.co/Cactus-Compute/whistle) Hugging Face repositories, whose model cards declare `license: apache-2.0`. `tools/fetch.sh` downloads them into `ref/`, which is git-ignored.
- **Game builds that ship the weights** must comply with the Apache License 2.0 for those files: include its license text and keep the attribution notices. Credits are in [NOTICE](NOTICE).
- **The upstream wheel and `libneedle3.so`** that `tools/fetch.sh` downloads are for local benchmarking only. Never commit or ship them.
- **Clean-room:** no code from [`cactus-compute/cactus`](https://github.com/cactus-compute/cactus), which has a custom non-Apache license, may be used here. Details are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Loading models on mobile and consoles

Files inside an Android APK (StreamingAssets) cannot be memory-mapped. Either read them into a `byte[]` and call `Model.FromBytes`, stream them with `Model.FromStream`, or copy the file once to `persistentDataPath` and call `Model.Open` on that path. iOS can map the file from the bundle path directly.

### First request in a player build

Construct one `NeedleSession` during loading rather than on first input: the first constructor builds the shared per-model vocabulary and prefills the ~140-token tool prefix, while a session restored from an `ExportPrimedPrefix` blob constructs in a few milliseconds with under a megabyte of new commit, so ship the prefix blob or create it on first run and reuse it. Call `WarmUp()` on the loading thread to run every kernel and tier path once, then ship TieredPGO enabled (or NativeAOT/ReadyToRun) so the first real request pays no JIT cost; `bench-e2e` reports session constructor milliseconds and private bytes per set for tracking this budget.

## Benchmarking

One harness covers end-to-end latency, the per-stage profile, kernels, allocations and the upstream native library. Build the CLI first (`dotnet publish src/Needle.Cli -c Release -o out/aot`).

```sh
# End to end: the two golden prompts (tools.json) plus a 10-prompt game set with 7 tools (tools/game_tools.json)
out/aot/needle bench-e2e ref/hf/needle3.cact tools/tools.json --threads 1,2,4 --reps 5

# Same run as JSON (CI artifact), with the per-stage profile of the first prompt
out/aot/needle bench-e2e ref/hf/needle3.cact tools/tools.json --threads 4 --reps 9 --profile --json > e2e.json

# Kernels: every CQ matmul shape at T=1,2,4,8,16,64 (serial and pooled), Prepare, HadamardMlp, Sinkhorn, softmax/argmax, Pool.Run barrier, TokenTrie.Allow
out/aot/needle bench ref/hf/needle3.cact --threads 4

# Upstream libneedle3.so through ctypes on the same prompts and the same prefix (never commit the wheel or the .so)
python3 tools/native_bench.py ref/hf/wheel ref/hf/needle3.cact tools/tools.json --json > native.json
```

- `bench-e2e` runs `NeedleSession` with `AutoDate = false` and an empty system prompt, so the prefix is identical to upstream `needle_init(system="")`. Tool retrieval is off. It reports prefill ms, decode us/token, TTFT, total ms and token counts as best and median over `--reps`, plus managed bytes per call (`GC.GetTotalAllocatedBytes(true)` delta, which covers worker threads) and native allocations per call.
- `--profile` drives a greedy decode (40 tokens after the prefill) on an engine instantiated with a real stage clock and prints embed, hyper (prepare, phi, mix), engram, attention (prep, QKVG, post-projection, core, gate, out, post-norm), cross, MLP, combine and logits in us/token, the `Pool.Run` count per token, and the same split for the prefill. Without `--profile` the engine uses `NoClock`, an empty struct whose calls are inlined away.
- Native allocation counts need a build with the counters compiled in: `dotnet publish src/Needle.Cli -c Release -p:NeedleAllocCounters=true -o out/aot-counters`. Otherwise the field is `null` (JSON) or `n/a` (text).
- Allocation gate: `--max-managed-bytes N` and `--max-native-allocs N` make the command exit with status 1 when the median per call exceeds the limit.
- Kernel GB/s is weight bytes per call over the best time, with the weights cycled across layers; GMAC/s counts multiply-accumulates (Kron only for `hadamard_mlp_apply`). The `Pool.Run` barrier rows need an idle machine: if workers do not get a core, the calling thread drains the empty tasks alone and the figure collapses.

### A/B method

Every performance claim compares two builds on the same machine, interleaved, best of at least 5 rounds:

```sh
python3 tools/bench_ab.py --rounds 6 \
  --a "out/base/needle bench-e2e ref/hf/needle3.cact tools/tools.json --threads 4 --reps 5 --sets golden --json" \
  --b "out/aot/needle bench-e2e ref/hf/needle3.cact tools/tools.json --threads 4 --reps 5 --sets golden --json"
```

`bench_ab.py` alternates A,B / B,A rounds, then reports the best of the per-round bests and the median of the per-round medians for both variants, with the percentage change. Expect about 15% noise on a shared machine; trust the best-of figures and repeat when A and B differ by less than the spread between rounds. `tools/native_bench.py --json` produces the same schema, so a native run can be passed as `--b` against a C# `--a` (its `threads` is `null`, which matches every thread count). The upstream C API reports tokens per second and no token counts or TTFT, so the native JSON carries `prefill_tps`, `decode_tps`, `decode_us_per_token` and a wall-clock `total_ms` around `needle_complete`; `bench-e2e` emits the same `prefill_tps` and `decode_tps` fields for a like-for-like comparison. The library spins its own worker threads and degrades badly on a loaded machine, so run it on an idle one.

### Large pages (`NEEDLE_HUGEPAGES`)

`NEEDLE_HUGEPAGES=1` is an opt-in Linux hint for weight and arena memory. When set, `Model.Open` advises `MADV_HUGEPAGE` on the mapped weights and every `Arena` of 2 MiB or more is 2 MiB-aligned and advised `MADV_HUGEPAGE`; smaller arenas keep the 64-byte alignment. Unset or `0` disables it with zero behavior change. Any other OS is a graceful no-op: Windows would need `VirtualAlloc` with `MEM_LARGE_PAGES` plus the `SeLockMemoryPrivilege` privilege, which is not implemented, and the flag is never consulted there.

Measured on a loaded Linux box (transparent hugepages `always`, so anonymous memory already collapses without the flag): `Model.Open` time is unchanged (page-cache resident either way), and two interleaved 6-round A/B runs of `bench-e2e --threads 4` disagree on decode (best-of-bests -5.7% then +0.5%, medians +2.1% then -10.4%), i.e. inside round-to-round spread and short of the 3% bar in issue #36. `perf stat -e dTLB-load-misses` swings more run-to-run than off-vs-on. Verdict: keep the flag as a harmless opt-in (it may matter on `madvise`-mode kernels), default off, no further work until a reference-box measurement shows TLB pressure. Re-run the A/B above with `--b "env NEEDLE_HUGEPAGES=1 ..."` to re-check.

## Evaluating your tool set

`needle eval` replays hand-written player commands through a real `NeedleSession` (reset between prompts) and scores the calls against expected values. Use it to choose a refusal threshold, to check that a tool set is usable, and to gate runtime policies (`MaxReasoningTokens`, `AutoDate`, `RetrievalTopK`, and later jump-forward) before turning them on.

```sh
# The shipped suite: 208 prompts over 5 game tool sets (tests/eval/*.jsonl, tools in tests/eval/tools/*.json)
out/aot/needle eval ref/hf/needle3.cact tests/eval --policy AutoDate=false RetrievalTopK=64

# Your own tool set: one JSON line per prompt, tools in <dir>/tools/<id>.json
out/aot/needle eval ref/hf/needle3.cact mygame/eval --json > eval.json

# Side by side: --baseline settings versus --policy settings, with the per-prompt regressions and improvements
out/aot/needle eval ref/hf/needle3.cact tests/eval --baseline AutoDate=false --policy AutoDate=false MaxReasoningTokens=32 --max-drop 0.02
```

Dataset line: `{"tools":"rts","query":"move to 120, 45","expected":[{"name":"move_to","args":{"x":120,"y":45}}],"tags":["ci"]}`. An empty `expected` means the prompt is out of domain and must produce no call. Calls are compared in order, and unexpected extra arguments fail the call. Argument matchers: a bare value is an exact match, `{"near":100,"tol":2}` a number within a tolerance, `{"contains":"orc"}` a case-insensitive substring, `{"oneOf":["line","wedge"]}` an enum choice, `{"any":true}` any present value. Options: `--tag ci` selects the 20-prompt subset that `EvalTests` runs, `--limit N` takes a round-robin sample over tool sets, `--json` writes every case, `--verbose` lists all misses, `--min-accuracy f` and `--max-drop f` set the exit status. New runtime policies register as one `PolicySetting` line in `src/Needle.Cli/EvalPolicies.cs`.

How to read the report:

- **exact** is the fraction of in-domain prompts whose complete call list was right and was not refused. **tool** and **args** score the model output before refusal, so they show routing and extraction quality on their own. **no-call** is the fraction of out-of-domain prompts that produced no executed call.
- **calibration** groups every reply that carries a call by its confidence. A usable threshold sits where accuracy in the bin clearly beats the bins below it. The model is monotone but not sharp: in the reference run, calls under 0.1 are right 7% of the time and calls above 0.5 are right 86% to 100% of the time.
- **RefuseBelow curve** re-scores the same replies at each threshold. *Coverage* is the share of in-domain prompts that still execute, *precision* the share of executed calls that are exactly right, *false-call* the share of out-of-domain prompts that still execute a call, and *wrong executed* the count of executed calls that are wrong. Pick the row whose wrong-executed count your game can tolerate.

Reference run (needle3, 4 threads, `AutoDate=false RetrievalTopK=64`, greedy, 208 prompts): exact 65.9%, tool 77.2%, args 72.7%, no-call 78.1%, median 200 ms per prompt. By set: chains 73.5%, dialogue 60.6%, rpg 47.2%, rts 63.9%, settings 83.8%. The default threshold 0.1 gives the best overall accuracy (67.8%) but still executes 40 wrong calls; 0.3 executes 22 (precision 82%, coverage 68%) and 0.5 executes 7 (precision 92%, coverage 47%). For commands where a wrong action is worse than a repeated prompt, start at `RefuseBelow = 0.3` and raise it for destructive commands.

### Tool design tips

These come from the misses in the suite above; re-measure on your own tools.

- **Keep the set small enough to skip retrieval.** With more tools than `RetrievalTopK` (5 by default) the session embeds the query and re-primes a 5-tool prefix per prompt. On the 8-tool sets this lost 6.8 points of exact accuracy (65.9% to 59.1%) and tripled median latency (about 160 ms to 520 ms). Set `RetrievalTopK` to at least your tool count, or split tools by game state and keep fewer than 6 per session.
- **Name tools with the verb the player says.** `set_formation` for "switch to wedge" worked, but "heal Lyra" became `recruit_member(Lyra)` at confidence 0.75 and "revive Elric" became `equip_item`, because spells hid behind the generic `cast_spell(spell=...)`. Prefer one tool per spoken verb (`heal`, `revive`), and avoid several tools that all read as "do something to a party member" (`give_item`, `use_item`, `equip_item`).
- **Use enums for closed vocabularies and booleans or integers for toggles.** The settings set (enums, booleans, slots) scored 83.8% exact, the rpg set with free-text item and member names 47.2%. Free-text arguments are where the model copies the wrong word (`give_item(to="arrow")`).
- **Do not require arguments the player will not say.** "light a torch" has no target, and the model invents one. Default the argument in game code or give the tool a separate no-target variant.
- **Expect long chains to drift.** Two-call sentences are mostly right, but "move to 1 1, then 2 2, then 3 3" came back as (1,2),(2,3) and "go to 10 10, then 20 20" as a single `move_to(10,20)`. Split three-command sentences in the UI or execute and re-prompt.
- **A catch-all tool attracts out-of-domain prompts.** With a `say(text)` tool in the chains set, "tell me a joke" became `say("joke")`; in the settings set the same prompt became `toggle_subtitles(true)`. Out-of-domain prompts executed a call 22% of the time at threshold 0.1 and 0% at 0.5 and above, so a higher threshold is the fix, not tool descriptions.
- **Do not cap or skip reasoning.** `MaxReasoningTokens=32` cost 8.5 points and `0` cost 31 points of exact accuracy.
- **Pin `AutoDate=false` for tests.** The date and time change the prompt, and 14 prompts flipped between right and wrong (8 lost, 6 gained, no net change) when it was on. Leave it on only if your tools need the date.
