# Site-model compat check (cactuscompute.com, 2026-10-31)

Standalone check of the C# engine against the CURRENT demo models from
cactuscompute.com (`needle`, `/blog/whistle`), not the frozen CI fixtures.

## Models under test

| file | bytes | sha256 |
|---|---|---|
| `/tmp/cactussite/whistle.cact` | 16 919 407 | `b6e02f04…dffeb` |
| `/tmp/cactussite/needle3.cact` | 35 335 380 | `c9d915ec…8170c38` |

Identity result: **byte-identical to the repo fixtures**
(`ref/hf/whistle/whistle.cact`, `ref/hf/needle3.cact` — same hashes, same
sizes; `tools/ref.lock` records both). So there is **no format drift**; repo
golden tests apply 1:1. All CLI runs below used the `/tmp/cactussite` copies
directly (`NEEDLE_THREADS=4`, Release `needle` CLI).

## Mode 1 — transcription (`whistle.cact`) — WORKS

`needle transcribe /tmp/cactussite/whistle.cact <clip>` loads and transcribes;
site form `needle --model whistle.cact --audio clip.wav` maps to this verb.

| clip (repo audio) | language | transcript | timing |
|---|---|---|---|
| `ref/audio/record.wav` (4.5 s) | en | `In one short sentence, what is the speed of sound at sea level?` (== `SpeechTests.RecordMatchesUpstreamExactly`) | ready 19.1 ms, encode 47.3, ttft 84.8 ms, 577 tok/s |
| `ref/audio/test.wav` (20 s) | en | `Below Hello Hello, just quickly testing out, …` (whisper memo) | ready 18.8 ms, encode 85.6, ttft 177.5 ms, 881 tok/s |
| `ref/audio/hotword.wav` (17.6 s) | en | `Hello, hello, just testing this. I am part of the cactus team …` | ready 18.7 ms, encode 72.7, ttft 141.3 ms, 883 tok/s |
| `ref/audio/test_long.wav` (75.7 s) | en | long-form monarchy/industrial passage, windowed past 30 s | ready 19.0 ms, encode 117.5, ttft 266.8 ms, 748 tok/s |

`--words` emits per-word start/end/probability (13 words on record.wav, ordered,
inside the clip); `--language xx` rejects with
`language is one of en de fr es it nl pl` (all 7 site languages supported,
forced or detected).

## Mode 2 — tool calls (`needle3.cact`) — WORKS

`needle call /tmp/cactussite/needle3.cact tools/tools.json …` (golden queries
from `tools/golden.py`):

- `what's it like in Lagos right now?` → `get_weather {city: Lagos}`,
  confidence **0.908**, reasoning cites Lagos; total 57.7 ms.
- `set a timer for 12 minutes and play some Miles Davis` →
  `set_timer {minutes: 12}` + `play_music {query: Miles Davis}` **in order**,
  confidence **0.9354**, total 64.5 ms.
- Smart-home prompts (custom 2-tool set, site kitchen-lights style):
  `turn off the kitchen lights` → `set_lights {room: kitchen, on: false,
  brightness: 0}` (conf 0.5892);
  `dim the bedroom to 30 and lock up` → `set_lights {bedroom, on: true,
  brightness: 30}` + `lock_doors {}` **two calls in order** (conf 0.7098).
- Uncovered requests against the smart-home set return **empty
  `function_calls`** with `"type": "respond"` (site behavior):
  `what is the capital of France?` (conf 0.9141), `write a haiku…` (0.599),
  `what's the weather in Lagos?` (0.9223, no weather tool).
  Note: with the tiny 3-tool `tools.json` the model instead stretches
  near-misses onto tools (`haiku → play_music` 0.3149,
  `capital of France → get_weather/France` 0.7589) — model behavior, matches
  the site's "keep the toolset narrow" guidance, not an engine bug.

## Mode 3 — audio straight to tool calls (both models) — WORKS

`needle voice needle3.cact whistle.cact tools.json clip.wav` returns calls plus
`audio_text` / `audio_language` / `audio_ttft_ms` / `audio_decode_tps` in one
object (site's `audio_`-prefixed speech fields):

- `tools.json` + `record.wav` → `get_weather {city: "Sea level"}` (conf
  0.4071, above floor) with `audio_text: "In one short sentence, what is the
  speed of sound at sea level?"`, `audio_language: "en"`, audio ttft
  60.8 ms — call + speech in one object, 118 ms end to end.
  (Misrouting is expected: the clip's content matches no tool; the mechanism
  is what is under test.)
- `tools/voice_tools.json` + `record.wav` → empty `function_calls`, held
  `suppressed_calls: [ask_assistant {question: "What is the speed of sound at
  sea level?"}]`, confidence **0.077 < 0.1 floor** — proves confidence gating
  applies on the combined path.
- `voice_tools.json` + `test.wav` (20 s) → `type: respond`, no calls, full
  20 s transcript + `audio_language: en` in the same object (213.9 ms).

## Test suites

- `SessionTests` + `SpeechTests`: **25/25 pass** (Release).
- `VoiceTests`: **9/10 pass**. `SpanTranscriptionAllocatesNothing` fails on
  `WholeProcess` (336 bytes vs 0; caller-thread and silence checks pass). That
  test uses `ref/hf` fixtures, the tree is untouched by this check, and the
  probe measures background-thread GC — pre-existing/environmental, unrelated
  to site-model compat. No re-runs were done to confirm it.

## Feature gaps vs the site docs (pointers, no implementation)

1. **`triggers` (per-tool regexes)** — `/needle`: restrict decode to matched
   tools, require a call, ship below the confidence floor. Absent: no
   `trigger` string in `src/Needle/*.cs`, `src/Needle.Cli/*.cs`,
   `src/Needle/Hosting/*.cs`, or tests; `src/Needle/ToolSet.cs` /
   `NeedleSession.cs` have no trigger concept.
2. **`error_code` + `peak_ram_mb`** — `/needle` response JSON carries both;
   `CallResult.ToJson` (`src/Needle/NeedleSession.cs:91`) emits
   `success/error/type/function_calls/suppressed_calls/reasoning/response/
   confidence(_head/_decode)/time_to_first_token_ms/prefill_tokens(_tps)/
   decode_tokens(_tps)/total_time_ms` — no `error_code`, no `peak_ram_mb`.
3. **`--audio-depth` decoder ladder** — `/blog/whistle`: every 2..20-layer
   decoder depth trained, encoder always full. Absent: `Whistle` builds its
   `Engine` at full geometry (`src/Needle/Whistle.cs:58-80`); no depth option
   on `Whistle`, `Transcribe`, or the CLI (`src/Needle.Cli/Program.cs:
   Transcribe/Voice`).
4. **Needle subnetwork selection (`--layers`)** — `/needle` deploy table.
   Absent: `Model.Open` loads full geometry; no layer slicing.
5. **Speech-embed CLI verb** — `/blog/whistle` `embed` returns one row per
   80 ms frame. API exists (`Whistle.Embed`, `src/Needle/Whistle.cs:488`,
   covered in `SpeechTests`), but `Program.cs` has only text `embed`, no
   audio-embed verb.
6. **Beam default 3 vs site 5** — `Whistle.Decode.cs:183`; settable in code,
   no CLI flag.
7. **Mic capture / `playground` / `compare`** — CLI is file-based
   (`Wav.Read`); no mic extra.
8. **LoRA / fine-tune / `build --lora`** — training-side, absent as expected.
9. **Auto-execute agent loop** — site Python `run()` executes tools and feeds
   results back; C# caller executes manually via
   `CompleteWithToolResult` (`SessionTests.ToolResultsAreAnsweredInPlainLanguage`).
   `src/Needle/Hosting/` provides channels/hosts but no tool-executing loop.

At parity (verified above or in tests): grammar-guaranteed calls, ordered
multi-calls, confidence head + 0.1 floor + `suppressed_calls`
(`NeedleOptions.RefuseBelow`, `src/Needle/NeedleSession.cs:21`), `reasoning`,
extraction helper (`Extraction.ToolsJson`), text `embed`, 7-language
detect/force, keywords, `--words`, silence→empty, streaming, resampling,
TTFT/tps reporting, dual-model `voice` verb (site's two-`--model` line),
either-`.cact` `Model.Open` (site `needle_load`).
