# Checkpoint: resume here

Written 2026-10-29 ~20:45 UTC. Designated branch: `claude/zealous-mayer-9vx4qp` (everything merged is pushed). Do not open a PR or push other branches unless the user says so. Delete the `checkpoint/` directory (and mention it in the commit) once all items below are merged.

## Merged and pushed (123 tests pass, netstandard2.1 clean)
W01 #1, W03 #3, W04 #4, W05 #5, W06 #6, W12 #12, W02 #2, W07 #7 (all closed). Follow-ups filed: #32 F01 CI on other OSes, #33 F02 spin default, #34 F03 LICENSE (owner decision), #35 F04 fresh-clone check, #36 F05 large pages, #37 F06 MLP remaining, #39 F07 idle-box bench, #40 F08 zero-alloc follow-ups. Tracking issue #31; plan in `plan.md`.

## In flight (committed in local worktrees of the old container; saved here as patches)
Apply with `git apply --3way checkpoint/patches/<name>.patch` on top of the designated branch (patches are `git diff base...branch`; w14/w19 were already merged with the branch at 8b25557, v01 is based on master 1893f77 and will need conflict resolution in Whistle*.cs/Scalar.cs/Frontend.cs).

1. `w14-attention-gqa.patch` (#14): GQA-shared attention core in Blocks.cs/Scalar/Neon. Committed, test and benchmark status UNKNOWN (agent was cut off by the rate limit while waiting on runs). Next: apply, build, run `dotnet test tests/Needle.Tests -c Release`, `NEEDLE_SCALAR=1`, netstandard2.1 build, then `needle bench-e2e ref/hf/needle3.cact tools/tools.json --threads 4 --reps 5 --profile` before/after for `attention_core` (baseline ~200 us/token decode; target <=50), prefill attention >=3x faster. Add the KernelTests GQA 6:1 / 1:1 / window / probabilities cases if missing.
2. `w19-gemv-bandwidth.patch` (#19): 4-token tile kernel sharing plane expansion, multi-row GEMV, precomputed factors in Quant.cs/Model.cs. Last commit e264b75; agent was waiting on the scalar test run. Same gates plus `NEEDLE_NOAVX512=1`; targets >=35 GB/s aggregate at 4 threads for QKVG/Out, >=15 GB/s single thread; report honestly if missed.
3. `v01-whistle-accuracy-2.patch` (#38 voice): WIP. Found: (a) upstream's log-mel is nearly gain invariant => use a relative log floor (1e-8 x clip peak mel energy), implemented in Frontend.cs/Whistle.cs; (b) `Scalar.HalfToSingle` mishandled fp16 subnormals (fixed); (c) upstream does not fake-quantize (A8) before conv pw2 (removed) and probably nowhere in the encoder; (d) added `StageObserver` (zero cost) and `tools/golden_whistle.py` (numpy reference from the Apache `cactus-compute/needle` whistle code, clean-room: never read `cactus-compute/cactus`). Result: encoder cosine vs upstream on long25 0.99923 -> 0.99979; record.wav and test.wav transcripts match upstream; hotword ("Captist" vs "cactus") and long25 ("victory" vs "Victoria", missing "and her husband Albert") still differ. Next steps: (1) remove the remaining encoder `Quant.FakeA8` sites in Whistle.cs (stem, hyper route, attention input, q/k/v, attention output, pw1) one at a time, measuring against upstream (`checkpoint/v01-scratch/ab.py`-style sweeps; the float reference reached cosine 0.99993 on long25 = target); (2) re-check decoder sites; use native encoder output fed into our decoder to separate encoder from decoder; compare logit margins at the divergence tokens; (3) reduce matmul noise from the 6-bit byte codebook (`Codebook.Bias = 63`) only minimally (shared with W19); (4) regenerate Whistle goldens (`python3 tools/golden_whistle.py`, outputs `ref/golden/whistle/<clip>/`), add the README Build&Test paragraph, add `WhistleStageTests` and an end-to-end transcript test with the upstream strings; (5) full gates and encode-ms before/after for `ref/audio/test.wav` at 4 threads. Scratch scripts are in `checkpoint/v01-scratch/` (upstream black-box scripts native_stt.py, native_enc.py, native_voice.py, native_bench.py; probe Program.cs).

## Remaining wave 2 (not started)
#18 W18 cheap Reset/Rewind (Engine.cs), #28 W28 game command eval suite (tests/eval + `needle eval` in the CLI), #30 W30 upstream format tracking (fetch.sh pins, Model.cs version info). Start these after #14/#19 merge (they touch Engine.cs/Model.cs/CLI).

## Environment restore (fresh container)
```bash
git clone https://github.com/IAFahim/needle-cs && cd needle-cs && git checkout claude/zealous-mayer-9vx4qp
curl -sSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh && bash /tmp/dotnet-install.sh --channel 10.0 --install-dir "$HOME/.dotnet"
export PATH="$HOME/.dotnet:$PATH" DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
tools/fetch.sh && pip install numpy sentencepiece
python3 tools/golden.py && python3 tools/golden_confidence.py   # ~15 CPU minutes; writes budget 40
```
Upstream wheel for black-box comparison: `ref/hf/wheel/*.whl` (extract `needle/libneedle3.so`; C API in `checkpoint/v01-scratch/native_*.py`; audio needs text=NULL).

## Working method that worked
Sonnet subagents (`model: sonnet`, `isolation: worktree`, background), max 2-3 at once because five at once hit the account usage limit twice. Worktrees branch from `master`, so tell each agent to `git merge claude/zealous-mayer-9vx4qp` first. Merge their branches here, resolve conflicts, run the full test suite, push this branch only, comment on and close finished issues, file follow-ups for anything unverified. Timings are noisy when agents run in parallel: interleave A/B, best-of-N.

## Decisions waiting on the user
LICENSE choice (#34); whether to open a PR into `master`; #33 spin default (300 us chosen over 50 us); #40 behavior changes of the span API.
