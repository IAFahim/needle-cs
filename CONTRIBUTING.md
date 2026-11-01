# Contributing

Read `plan.md` first (it is not on every branch yet; see tracking issue [#31](https://github.com/IAFahim/needle-cs/issues/31)). Every work item has a GitHub issue, and the plan and the issue are self-contained.

## License status

This repository has no `LICENSE` file yet. Choosing the license for the repository's own code is the owner's decision, and it is pending. Until a `LICENSE` file is added, do not assume any license for the code, and do not add a `LICENSE` file in a pull request.

Third-party materials and their licenses are listed in [NOTICE](NOTICE).

## Provenance and the clean-room rule

Three upstream sources are involved, and they have different licenses.

| Source | License | May we use it? |
|---|---|---|
| `Cactus-Compute/needle3` and `Cactus-Compute/whistle` weights (Hugging Face) | `apache-2.0` per the model cards | Download and load them. Do not commit them. |
| [`cactus-compute/needle`](https://github.com/cactus-compute/needle) (Python package and architecture) | Apache License 2.0 | Yes. Derive behavior from it. |
| [`cactus-compute/cactus`](https://github.com/cactus-compute/cactus) (C++ engine and kernels) | Custom license, not Apache-2.0 | No. Never copy code from it. |

The custom license of `cactus-compute/cactus` grants permission only to individuals using it for personal, educational, research or non-commercial purposes, to organizations with less than 2,000,000 USD in total funding and less than 2,000,000 USD in gross annual revenue, to educational institutions and students, and to registered non-profit organizations. Anyone else needs a separate commercial license from Cactus Compute, Inc. Read the file itself for the exact terms.

Rules:

- Work clean-room. Do not copy, translate, paraphrase line by line, or port code from `cactus-compute/cactus`, including its kernels, tests and comments.
- Derive behavior from `tools/golden.py` and from the Apache-2.0 Python architecture in `cactus-compute/needle`.
- If you have read the `cactus-compute/cactus` source closely, say so in the pull request so the reviewer can judge the risk.
- Never commit, package or ship the upstream wheel (`cactus_needle-*.whl`) or the native library `libneedle3.so` it contains. They are for local benchmarking only (W02).
- Never commit or redistribute model files (`*.cact`), tokenizers, `config.json` from the model repository, or the downloaded audio clips. They live in `ref/`, which is git-ignored.
- Keep `ref/` ignored. Do not use `git add -f` on anything under it.

## Code style

These rules come from `plan.md` section 3.

1. No comments. Names, types and data layout carry the meaning.
2. Data-oriented design. Use blittable `readonly struct`s, structs of arrays, pointer plus length, and arenas. No object graphs on the hot path.
3. Pure kernels. Kernels are static functions from inputs to outputs. Effects live only in the engine and host shells.
4. Immutable by default. Use `readonly struct`, `in` parameters, and configuration resolved once.
5. Total functions on the hot path. Report results through status enums. Exceptions only at construction and I/O boundaries.
6. Deterministic. Identical inputs, tier, thread count and chunking produce identical bits.
7. Zero allocation on the hot path. 0 managed bytes and 0 native `malloc`s per call after warm-up, enforced by tests.
8. Every abstraction earns its name. Deleting a feature shrinks the code without breaking it.
9. Measure every claim. Report before and after on the same machine, interleaved A/B, best of at least 5, using the W02 harness.

## Test setup

The projects target `net10.0`, so you need the .NET 10 SDK on a Linux x64 machine.

```bash
curl -sSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh && bash dotnet-install.sh --channel 10.0 --install-dir "$HOME/.dotnet"
```

```bash
export PATH="$HOME/.dotnet:$PATH" DOTNET_CLI_TELEMETRY_OPTOUT=1
```

```bash
tools/fetch.sh && pip install numpy sentencepiece && python3 tools/golden.py 40 && python3 tools/golden_confidence.py
```

## Refreshing the upstream model

Cactus Compute owns the model weights and the `.cact` format. `tools/ref.lock` pins every downloaded file by SHA-256 and size, and pins the Hugging Face commits of `Cactus-Compute/needle3` and `Cactus-Compute/whistle` and the `cactus-compute/cactus` commit that hosts the audio clips. `tools/fetch.sh` downloads exactly those revisions and fails with "upstream changed" when a file does not match its pin. Nothing changes silently.

Everyday commands:

```bash
tools/fetch.sh
tools/fetch.sh --verify
```

`tools/fetch.sh` skips files that already match the lock, so it is safe to rerun. `--verify` downloads nothing and checks the files already in `ref/` (the test suite runs it).

Refresh procedure, only when you deliberately adopt a new upstream model:

1. Run `tools/fetch.sh --update-lock`. It resolves the newest upstream revisions, downloads those files into `ref/`, and rewrites `tools/ref.lock`. Read the diff of the lock: it shows which files and revisions changed.
2. Run `python3 tools/golden.py 40` and `python3 tools/golden_confidence.py`. `golden.json` records the model SHA-256 and the lock SHA-256, and the tests refuse fixtures generated from a different model ("fixtures were generated from a different model; regenerate with tools/golden.py"). An old `golden.json` without the field is accepted, so regenerate it once.
3. Run `dotnet test tests/Needle.Tests -c Release` and `NEEDLE_SCALAR=1 dotnet test tests/Needle.Tests -c Release`. A new format variant fails at load with a message naming the found and the supported values (`ModelInfo` in `src/Needle/ModelInfo.cs`, `Model.Info` for the loaded model). Extend `Model` only after reading what changed upstream. `FormatTests` pins the expected `ModelInfo` of the current model, so update those numbers on purpose.
4. Rerun the W02 benchmarks (`needle bench-e2e`, `tools/native_bench.py`) on the reference machine and record before and after numbers.
5. Run the evaluation suite and the parity checks: `needle eval ref/hf/needle3.cact tests/eval` with a baseline from the old model (`--baseline`, `--max-drop`) for W28, and the upstream native comparison for W29.
6. Update the numbers in `plan.md` section 2 and the model facts in the README, then commit `tools/ref.lock` together with the code and test changes.

`ref/` stays git-ignored. Never commit the lock-listed files themselves.

## Pull request checklist

From `plan.md` section 8 and section 7.

- [ ] The branch is named `w<NN>-<slug>` and touches only the files the issue names, or the pull request says why.
- [ ] The code follows the code style above.
- [ ] A test fails before the change and passes after it.
- [ ] No code, comments or tests were copied from `cactus-compute/cactus`.
- [ ] No model files, tokenizers, wheels or `libneedle3.so` are committed.
- [ ] The test suite passes:

```bash
dotnet test tests/Needle.Tests -c Release
```

- [ ] The scalar run passes:

```bash
NEEDLE_SCALAR=1 dotnet test tests/Needle.Tests -c Release
```

- [ ] The `netstandard2.1` build succeeds:

```bash
dotnet build src/Needle -f netstandard2.1 -c Release
```

- [ ] The NativeAOT publish succeeds:

```bash
dotnet publish src/Needle.Cli -c Release -o out/aot
```

- [ ] Performance pull requests attach W02 harness output before and after: end-to-end at 1, 2 and 4 threads, the per-stage profile, kernels and allocations.
- [ ] The numbers are pasted into the pull request and the item is ticked in the tracking issue [#31](https://github.com/IAFahim/needle-cs/issues/31).
- [ ] `plan.md` is updated if a learning changed.
