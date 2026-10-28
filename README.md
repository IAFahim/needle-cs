# needle-cs

Everything Needle 3 and Whistle do at inference time, in C#: tool calling with grammar-constrained output, a calibrated confidence score, multi-turn conversations with tool results, structured extraction, tool retrieval, text embeddings, and speech-to-text with language detection, keyword biasing, word timestamps, long audio and streaming. Training stays with the Python package.

The core is unsafe C#: the `.cact` files are read in place, every hot path is a static function over raw pointers, and nothing is allocated after load. It runs as a NativeAOT binary or an ordinary .NET library, on AVX2 machines at full speed and on anything else through a scalar backend. A `netstandard2.1` build and a small Unity package are included for Mono/IL2CPP and for Burst jobs.

Models: [Cactus-Compute/needle3](https://huggingface.co/Cactus-Compute/needle3) (121M, 35 MB) and [Cactus-Compute/whistle](https://huggingface.co/Cactus-Compute/whistle) (17 MB).

## Numbers

i9-14900K, Linux, NativeAOT, pinned to the P-cores. This machine was shared with other projects' test runs for most of the session, so timings from a loaded moment are marked; compare ratios, not last digits.

**Tool calling** (3 tools, 18-token query, ~40 generated tokens, tools prefix cached):

| threads | query | decode | decode tok/s |
|---:|---:|---:|---:|
| 1 | 90.7 ms | 1.64 ms/token | 609 |
| 4 | 37.9 ms | 0.70 ms/token | 1439 |
| 8 | 29.7 ms | 0.555 ms/token | 1802 |

With grammar, reasoning and confidence (`NeedleSession`, 4–8 threads): about 43–48 ms per query, 6–8 ms to the first token, ready in about 50 ms. The 8-thread row above was measured on a quiet machine; the later 8-thread figure of ~0.9 ms/token was taken while three unrelated processes held three cores.

**Speech** (8 threads):

| clip | time |
|---|---:|
| 4.5 s, 48 kHz stereo (resampled) | 17–25 ms |
| 20 s | 61–66 ms (encode 43 ms) |
| 75.7 s, three windows | 306 ms |
| streaming, one pass per second of audio | 6–9 ms |

Audio to tool call, 4.5 s clip, both models: 65 ms.

**Memory** (peak RSS, of which most is the shared file mapping): tool calling 56 MB (16 MB private); speech on 20 s 54 MB (32 MB private); both models together 90 MB (34 MB private). The scalar backend, for machines without AVX2, decodes at about 7.6 ms/token (131 tok/s).

Not measured: the vendor engine. `tools/compare.py` runs `libneedle3.so` and this build on the same task; I did not run the vendor binary while building this.

## Use

```csharp
using var model = Model.Open("needle3.cact");
using var session = new NeedleSession(model, toolsJson, new NeedleOptions { Threads = 4, System = "…" });

var result = session.Complete("set a timer for 12 minutes and play some Miles Davis");
foreach (var call in result.FunctionCalls) Console.WriteLine($"{call.Name} {call.ArgumentsJson}");
Console.WriteLine(result.ToJson());                        // success, function_calls, suppressed_calls, reasoning, response, confidence, timings

var answer = session.CompleteWithToolResult("{\"temp_c\":27}");   // plain-language reply in answer.Response
session.Reset();                                           // start a new conversation, keep the tools
```

`Complete` takes one turn. Context carries across turns until `Reset()`, as in the Python package. Tool schemas may be OpenAI-wrapped, camelCase or snake_case; names are aliased for the model and restored in the output. More than five tools are narrowed to the five most relevant per query by embedding similarity (`RetrievalTopK`). `Extraction.ToolsJson(schema)` turns a JSON schema into a one-tool set for structured extraction. `session.Embed(text)` returns a unit-norm 3072-float vector.

```csharp
using var speech = Model.Open("whistle.cact");
using var whistle = new Whistle(speech, threads: 4);
var pcm = Wav.Read("clip.wav");                            // any WAV, mono-mixed and resampled to 16 kHz
var text = whistle.Transcribe(pcm, language: null, keywords: new[] { "Omeprazole" }, wordTimestamps: true);
var live = whistle.Stream();                               // live.Process(chunk) … live.Stop()
var voice = new VoiceSession(needleModel, speech, toolsJson);   // audio in, tool call out
```

`NeedleService`, `SpeechService` and `InferenceThread` run a model on its own thread with `Task` APIs, for game loops and UI threads. Models are not thread-safe; use one thread per model.

CLI: `needle call|transcribe|stream|voice|embed|generate|run|verify|bench`.

## How the grammar works

The tool schemas compile once into a flat native table. A byte-level pushdown automaton with a fixed 400-byte copyable state accepts exactly the JSON the schemas allow: tool names, property names, required fields, enums, `const`, nullable unions, arrays with bounds, nested objects, numbers, booleans, string length. A trie over the model's vocabulary turns the automaton state into an allowed-token mask, with a precomputed fast path for free-text string interiors. The model chooses freely inside the mask, so a call is always well-formed and never invents a tool or a key. After a tool result the model may instead answer in prose; the decoder lets it choose.

A property test walks 150 random paths through the real 8192-token vocabulary and checks every result parses and conforms to the schemas.

## The hot path

A 2-bit weight is `(codebook[index] · groupNorm) @ H`, with `H` the normalised 128-point Walsh–Hadamard matrix, so `W·x = Σ groupNorm · codebook · (H·x)`. Rotate the activation once; never rebuild weights.

1. `Prepare`: rotate each 128-group, quantise to int8 with its own scale, de-interleave so one `vpshufb` expands 32 packed indices.
2. `MatMul`: shuffle the packed 2- or 4-bit indices through a 16-byte codebook, then `vpdpbusd` (or `vpmaddubsw` + `vpmaddwd`). One vector dot per row removes the zero point.
3. KV cache and attention are int8; heads run in parallel on a lock-free pool.
4. The confidence head pools hidden states with an online softmax as tokens are fed, so no per-token cell is stored.
5. Whistle runs a log-mel frontend, a blocked three-stage conv stem and the same lane-mixing encoder, then reuses the decoder engine with cross-attention.

| file | role |
|---|---|
| `Model.cs`, `Audio.cs` | `.cact` loader, typed pointer structs, probe heads, speech tensors |
| `Quant.cs`, `Ops.cs`, `Blocks.cs` | AVX2 kernels, each with a scalar twin in `Scalar.cs` |
| `Engine.cs`, `Pool.cs` | batched forward, KV state, cross-attention, worker pool |
| `Grammar.cs`, `TokenTrie.cs`, `Json.cs`, `ToolSet.cs` | schema → constraint |
| `NeedleSession.cs`, `Voice.cs` | conversations, envelope, retrieval, speech → tools |
| `Whistle.cs`, `Whistle.Decode.cs`, `Frontend.cs`, `Wav.cs` | speech |
| `Tokenizer.cs` | SentencePiece-BPE over the embedded vocabulary |

## Unity and Burst

`dotnet build src/Needle -f netstandard2.1` produces a 120 KB `Needle.Inference.dll` for Unity: no intrinsics, no `Half`, no `NativeMemory`, models loadable from a byte array (`Model.FromBytes`) when memory mapping isn't available. Drop it in `unity/com.needle.inference/Runtime/Plugins/`.

- **Managed use, any platform.** `NeedleService` and `SpeechService` give `Task` APIs on a dedicated thread. `Samples~/EcsAgent` shows an ECS system that turns a `NeedleRequest` component into a `NeedleResult` without blocking a frame.
- **Burst.** The kernels are static functions over pointers and blittable structs, with no managed objects on their path. `Runtime/NeedleBurstJobs.cs` wraps `Quant.MatMulRows` and `Quant.Prepare` as `IJobParallelFor` jobs; the assembly name `Needle.Unity` is granted `InternalsVisibleTo`. For one inference per entity, run a single-threaded engine inside a job and let ECS provide the parallelism.

**Not verified:** there is no Unity install here. The netstandard2.1 build compiles and its scalar code is exercised by the whole test suite, but the Unity package, the Burst jobs and the ECS sample were written against the API and have never been opened in the Editor. Expect to adjust asmdef references and Burst's restrictions on first import.

## Verifying

```sh
tools/fetch.sh
python3 -m venv .venv && .venv/bin/pip install numpy sentencepiece
.venv/bin/python tools/golden.py 40 && .venv/bin/python tools/golden_confidence.py   # ~13 min
dotnet test tests/Needle.Tests -c Release                    # 37 tests, SIMD backend
NEEDLE_SCALAR=1 dotnet test tests/Needle.Tests -c Release    # same suite on the scalar backend
```

Tests cover: logit, per-layer and generation parity with an independent numpy reference; the confidence head against its reference; tokenizer ids against SentencePiece; grammar acceptance and the random-walk property; tool conversations, aliasing, enums, extraction and retrieval; Whistle transcripts, forced languages, word times, long audio, keywords and streaming at 1 and 4 threads; the pool under 50,000 back-to-back jobs; the services; and `Model.FromBytes`.

## Differences from the vendor engine

The vendor engine is closed, so these are re-derivations from the Python package and the public notes:

- **Confidence** is the minimum of the probe-head probability and the geometric-mean token probability of the call, with calls under 0.1 moved to `suppressed_calls`. The vendor's exact calibration and its grounding gates (negated verbs, invented enum values, unmentioned places) are not reproduced, so a call it would withhold may be returned here with a moderate score. Both components are in the JSON as `confidence_head` and `confidence_decode`.
- **Word times** come from cross-attention alignment (column-normalised, median-filtered, open-ended DTW). They are plausible and ordered but untested against ground truth.
- **Keyword biasing** adds a logit bonus to tokens that continue a keyword's spelling; it nudges rather than forces.
- The 256-token KV window of the exported header is not applied; attention is full up to the 1024-token local window, as in training.
- Whistle's silence rule is an RMS floor plus the model's own end-of-text.
- Not implemented: the router and embedding heads (the shipped model has neither), ladder-depth selection at runtime, and big-endian hosts.
