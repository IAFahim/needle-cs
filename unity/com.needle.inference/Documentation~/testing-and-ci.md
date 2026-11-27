# Testing the Unity package and running its CI

The package's tests run in the Unity Editor through the Unity Test Framework. `tools/unity-ci.sh`
runs them from a terminal and is what `.github/workflows/unity.yml` calls on the self-hosted Unity
runner, so a run by hand and a CI run do the same thing.

## What is tested

| Assembly | Mode | Needs the model | What it checks |
|---|---|---|---|
| `Needle.Unity.Tests` (`Tests/Runtime`) | EditMode | some tests | The two plugin DLLs carry the same source stamp, a CoreCLR Editor runs the `net10.0` kernels, the kernel table is installed, `ModelLoader` paths and errors, the unmanaged channel rings, the Burst `NeedleSystem` in a private `World`, and end to end: tool calls from a session and from `NeedleHost`, zero managed allocation on the span API, Whistle transcribing `record.wav`. |
| `Needle.Unity.PlayMode.Tests` (`Tests/PlayMode`) | PlayMode | yes | `NeedleHost` answers while frames keep ticking, and submitting plus polling costs the main thread under 1 ms in every frame (`NEEDLE_FRAME_BUDGET_MS` changes the budget). The game-path bench asks the game tool set `NEEDLE_REPS` times over and fails when an ask fails or when submitting or polling allocates on the main thread; it reports main-thread microseconds per ask, frame-time spikes while the model runs and engine latency against the Unity targets of the 4.x plan (`gamepath-playmode.json`). |
| `Needle.Conformance.Tests`, `Needle.Kernels.Burst.Tests`, `Needle.Probe.Tests` | EditMode | no | The Burst kernels against the managed reference kernels, and the runtime probe. |
| `Needle.Bench.Tests` | EditMode | yes | The golden-set bench inside the Editor; writes `needle.bench-e2e/1` JSON. |

Tests that need the weights are Ignored when the files are missing, so a project that lists the
package in `testables` without them still passes. `NEEDLE_REQUIRE_MODEL=1` (set in CI) turns a
missing file into a failure. Every test assembly has `defineConstraints: UNITY_INCLUDE_TESTS`, so
none of them compiles in a game that installs the package.

The tests find the files in a needle-cs checkout above the Unity project (the case for
`unity/NeedleDemo`), in its `ref/` folder, which `tools/fetch.sh` fills. Each path can be set
instead: `NEEDLE_REPO`, `NEEDLE_MODEL`, `NEEDLE_WHISTLE`, `NEEDLE_TOOLS`, `NEEDLE_GAME_TOOLS`,
`NEEDLE_AUDIO_DIR`. `NEEDLE_THREADS` sets the thread count (default 4).

## Running it by hand

```bash
tools/fetch.sh                      # once: model, Whistle and clips into ref/
tools/unity-ci.sh lint              # no Unity needed: metas, GUIDs, tracked DLLs, C# 9 syntax, DLL stamps
tools/unity-ci.sh fresh-install     # the package as a Git-URL install sees it, in an empty project
tools/unity-ci.sh editmode
tools/unity-ci.sh playmode
tools/unity-ci.sh bench-editmode
tools/unity-ci.sh player il2cpp     # builds the bench player, runs it, writes JSON
tools/unity-ci.sh player coreclr
tools/unity-ci.sh all               # everything, in that order
```

Results, logs and bench JSON go to `out/unity/`. The script fails when a test fails, when no test
ran, when the log has a compile error, or when Unity exits before writing results. A run whose
Library was cold (the first run, or the first after a manifest change) can fail checks that need
Burst-compiled code, so a failing test run is repeated once and the second run decides; the first
run's files are kept as `<step>-cold.xml` and `.log`.

`NEEDLE_TEST_FILTER` runs only the tests whose full names match (Unity's `-testFilter`), for
example `NEEDLE_TEST_FILTER=Needle.Unity.PlayMode.Tests.GrammarRegressionTests tools/unity-ci.sh playmode`.
The grammar regression test asks every game query `NEEDLE_REPRO_ROUNDS` times (default 3) through a
host with retrieval on and fails on any grammar failure (issue #74).

It uses `UNITY_EDITOR` when set, else the editor named in
`unity/NeedleDemo/ProjectSettings/ProjectVersion.txt` if it is installed under
`~/Unity/Hub/Editor`, else the newest Unity 7 editor there. Close any Editor that has
`unity/NeedleDemo` open first.

`fresh-install` packs only the files git tracks into a tarball, which Unity treats as immutable
like a Git URL, and imports it into an empty project. It catches what a `file:` install hides:
an untracked or git-ignored DLL, a file without a `.meta`, a test assembly that compiles outside
`testables` (#56).

## The plugin DLLs

`Runtime/Plugins/netstandard2.1/Needle.Inference.dll` and
`Runtime/Plugins/net10.0/Needle.Inference.Net10.dll` are built from `src/Needle` by
`tools/pack-unity.sh` and committed. The build is deterministic and stamps each DLL with a hash of
the sources (`AssemblyMetadata("Needle.SourceStamp")`). After any change under `src/Needle`, run
`tools/pack-unity.sh` and commit both DLLs; `tools/pack-unity.sh --check` (run by CI) fails while
they are stale.

The source generator ships the same way: `Analyzers/Needle.Generators.dll` is built from
`src/Needle.Generators` by the same script and stamped with a hash of the generator's own sources
(`gen-sha256:`), so a change there without a rebuild also fails the check.

## Comparing with the native library

`tools/native_bench.py` runs the upstream `libneedle3.so` from the wheel `tools/fetch.sh` downloads
on the same prompt sets and writes the same JSON schema. `tools/unity-bench-compare.py` puts the
reports side by side, with each run's speed relative to the native library:

```bash
python3 tools/native_bench.py ref/hf/wheel ref/hf/needle3.cact tools/tools.json --sets all --json > out/unity/bench-native.json
python3 tools/unity-bench-compare.py --native out/unity/bench-native.json \
  editor=out/unity/bench-editmode.json il2cpp=out/unity/bench-player-il2cpp.json coreclr=out/unity/bench-player-coreclr.json
```

The native library runs with its own default thread count; the Unity runs use `NEEDLE_THREADS`.
Compare runs from one machine only, and only while nothing else runs on it: another Unity or
GitHub runner job on the same PC skews a bench by tens of percent, and the native library, which
spins its own worker threads, far more. CI therefore benches the native library first, on the idle
machine, and again after the Unity runs (`native-after` in the table); when the two disagree, the
machine was busy and the speedups are not trustworthy. The CPU load and the busiest processes are
printed before each bench and recorded in `load.txt`; when the 1-minute load was over a quarter of
the cores before any bench, the run summary says the numbers are not a fair comparison.

The IL2CPP player's toolchain needs `libncurses.so.6` and `libxml2.so.2`, which Arch-based systems
(Omarchy included) do not ship under those names. `tools/unity-ci.sh player il2cpp` then links the
system's `libtinfo.so.6` and a 64-bit `libxml2` from Unity's own sysroots into
`out/unity/il2cpp-shims/` (preferring a copy whose own libraries are all present) and puts that folder
on the build's library path, so nothing outside the checkout changes. It then asks `ldd` what the
cached `clang` and `ld.lld` still cannot find, such as ICU 70 (`libicui18n.so.70`) on a system with a
newer ICU, and links those from Unity's sysroot too, a few passes deep; a library that is in neither
place is named in the log. The cached toolchain only exists after Unity has unpacked it once, so on a
fresh machine the first IL2CPP build can fail where the second succeeds. When all of that is not
enough, the manual fix in `il2cpp-players.md` still applies.

The `simd` field of a report names the tier of the kernels that ran and their provider. In the
Editor and a CoreCLR player that is the `net10.0` sidecar; an IL2CPP player has no sidecar and runs
the Burst or `netstandard2.1` kernels.

## The CI workflow

`.github/workflows/unity.yml` has two jobs:

- **Package layout and DLL stamps** on a GitHub-hosted runner: `tools/unity-package-lint.py` and
  `tools/pack-unity.sh --check`. It needs no Unity and runs for every pull request, forks included.
- **Unity Editor, players and bench** on a self-hosted runner with the labels `self-hosted`,
  `Linux`, `X64` and `unity`: fetch, lint, fresh install, EditMode, PlayMode, the Editor bench, the
  IL2CPP and CoreCLR player benches, the native bench before and after them, and a comparison
  table in the run summary. A failed player build prints the errors from its build log.
  Logs, NUnit XML and bench JSON are uploaded as the `unity-<run>` artifact. It skips pull requests
  from forks, runs one at a time, and is never cancelled halfway.

It runs on pushes to `master` and on pull requests that touch the package, `src/Needle` or these
tools, and by hand from Actions → Unity → Run workflow (thread counts, repetitions, players on or off).

## Setting up the self-hosted runner

On a Linux x64 PC with Unity:

1. **Register a runner for needle-cs.** A runner registered to another repository of a personal
   account cannot take needle-cs jobs, so add one in its own folder, for example
   `~/actions-runner-needle-cs`: needle-cs → Settings → Actions → Runners → New self-hosted runner →
   Linux x64, then its download and `config.sh` steps, with the label `unity`. If the PC already
   has a runner for another repository, copy that runner's `.env` and `.path` into the new folder
   first. Install it as a service that runs as the user who uses Unity Hub:
   `sudo ./svc.sh install <user>`, then `sudo ./svc.sh start`.
2. **Fork pull requests.** needle-cs is public: Settings → Actions → General → "Fork pull request
   workflows from outside collaborators" → "Require approval for all outside collaborators". The
   workflow also skips the self-hosted job for forks.
3. **Unity.** `7000.0.0a16` (the demo's version), or the newest Unity 7 editor already on the PC,
   with Linux Build Support (IL2CPP) for the IL2CPP player
   (`unity install-modules --editor-version <version> --module linux-il2cpp --accept-eula --yes`
   adds it; `tools/unity-ci.sh player il2cpp` stops with that command when the module is
   missing). The CoreCLR player needs nothing extra. On Arch Linux (Omarchy included) the IL2CPP
   build also needs the `libncurses`/`libxml2` shims in `il2cpp-players.md`, and needs them again
   whenever Unity downloads a fresh copy of its IL2CPP toolchain.
4. **License.** Sign in to Unity Hub once as the runner's user and activate a license; batch runs
   reuse it. Only to activate from the workflow instead, add the repository secrets `UNITY_SERIAL`,
   `UNITY_EMAIL` and `UNITY_PASSWORD`.
5. **Runner environment.** In the runner folder's `.env`, then restart the service:
   `NEEDLE_REF_CACHE=/home/<user>/needle-ref` (a persistent folder for the model files and the
   demo's Library) and, optionally, `UNITY_EDITOR=/home/<user>/Unity/Hub/Editor/<version>/Editor/Unity`.
6. **Packages.** `git`, `curl`, `bash`, `tar` and `python3`. The Unity job needs no .NET SDK.
7. **Models.** Each run calls `tools/fetch.sh` into `NEEDLE_REF_CACHE`, which downloads only missing
   or changed files, so the PC must reach huggingface.co. The first run downloads several GB.
8. **Check.** The runner shows as Idle under Settings → Runners; then Actions → Unity → Run workflow.
