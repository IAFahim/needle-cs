# needle-cs

Zero-allocation, unsafe C# inference engine for [Needle 3](https://huggingface.co/Cactus-Compute/needle3) (tool calling, extraction, retrieval, embeddings) and [Whistle](https://huggingface.co/Cactus-Compute/whistle) (speech-to-text).

Reads `.cact` model files directly in place with AVX2, ARM64 NEON (SDOT with a widening fallback for CPUs without FEAT_DotProd), and portable scalar backends, selected at runtime per core. Supports NativeAOT, standard .NET (`net10.0`), `netstandard2.1`, and Unity/Burst.

## Features

- **Needle 3 Tool Calling:** Schema-driven byte-level pushdown automaton grammar constraint, multi-turn sessions, confidence scoring, top-K tool retrieval, structured extraction, and 3072-d embeddings.
- **Whistle Speech-to-Text:** 16 kHz WAV resampling, language detection/forcing, keyword biasing, DTW word timestamps, long-audio windowing, streaming, and direct audio-to-tool-call (`VoiceSession`).
- **Portable & Embeddable:** Full scalar fallback for platforms without SIMD, `Model.FromBytes` for platforms without `mmap`, dedicated-thread `Task` wrappers (`NeedleService`, `SpeechService`), and automatic placement of worker threads on the fastest physical cores (Linux; `NEEDLE_PIN=0` disables, `NEEDLE_NODP=1` forces the NEON widening-fallback tier).

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

### Speech-to-Text & Voice-to-Tool

```csharp
using var speech = Model.Open("whistle.cact");
using var whistle = new Whistle(speech, threads: 4);

var pcm = Wav.Read("clip.wav");
var transcript = whistle.Transcribe(pcm, keywords: new[] { "Omeprazole" }, wordTimestamps: true);

using var voice = new VoiceSession(model, speech, toolsJson);
var voiceCall = voice.Complete(pcm);
```

## Performance (i9-14900K, NativeAOT)

| Workload | Latency / Speed | Peak RSS (Private) |
|---|---|---|
| Tool calling, 19-prompt / 56-token decode (8 threads) | ~46 ms / query (1365 tok/s decode) | 56 MB (16 MB) |
| Whistle 20s clip (8 threads) | ~60 ms (3900 tok/s decode) | 54 MB (32 MB) |
| Voice-to-tool (4.5s clip) | ~65 ms | 90 MB (34 MB) |

The full parity suite (golden logits, greedy decode, tokenizer) passes on AVX2, NEON (both dot-product tiers, validated under qemu-aarch64), and scalar.

## Build & Test

```sh
# Fetch models & fixtures, then run tests
tools/fetch.sh
dotnet test tests/Needle.Tests -c Release
NEEDLE_SCALAR=1 dotnet test tests/Needle.Tests -c Release

# Build NativeAOT CLI
dotnet publish src/Needle.Cli -c Release

# Build Unity netstandard2.1 DLL
dotnet build src/Needle -f netstandard2.1 -c Release
```

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
