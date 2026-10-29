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

Requirements: the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) (the 8.0 SDK fails with `NETSDK1045`), Python 3 with `numpy` and `sentencepiece`, and `curl`.

```sh
# 1. Fetch the model, tokenizer, upstream wheel, Whistle and audio clips into ref/
tools/fetch.sh

# 2. Generate the golden fixtures (numpy reference forward pass)
pip install numpy sentencepiece
python3 tools/golden.py               # about 15 CPU minutes
python3 tools/golden_confidence.py    # needs the output of golden.py; takes seconds

# 3. Run the 54 tests (about 2 minutes each)
dotnet test tests/Needle.Tests -c Release
NEEDLE_SCALAR=1 dotnet test tests/Needle.Tests -c Release
```

The greedy decode budget is `DECODE_BUDGET` in `tools/golden.py` (`python3 tools/golden.py 64` overrides it). The script records the budget it used in `ref/golden/golden.json` and `ParityTests.GenerationMatchesReferenceGreedyDecode` decodes exactly that many tokens, so the fixture and the test cannot drift.

```sh
# Build NativeAOT CLI
dotnet publish src/Needle.Cli -c Release

# Build Unity netstandard2.1 DLL
dotnet build src/Needle -f netstandard2.1 -c Release
```
