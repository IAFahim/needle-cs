# Checkpoint: resume here

Written 2026-10-29 ~20:45 UTC. Designated branch: `claude/zealous-mayer-9vx4qp` (everything merged is pushed). Do not open a PR or push other branches unless the user says so. Delete the `checkpoint/` directory (and mention it in the commit) once all items below are merged.

## Merged and pushed (179 tests pass in default, NEEDLE_SCALAR=1 and NEEDLE_NOAVX512=1 tiers; netstandard2.1 clean)
W01 #1, W03 #3, W04 #4, W05 #5, W06 #6, W12 #12, W02 #2, W07 #7, W14 #14, W19 #19 (all closed). Follow-ups filed: #32 F01 CI on other OSes, #33 F02 spin default, #34 F03 LICENSE (owner decision), #35 F04 fresh-clone check, #36 F05 large pages, #37 F06 MLP remaining, #39 F07 idle-box bench, #40 F08 zero-alloc follow-ups, #41 F09 attention remaining, #42 F10 GEMV remaining. Tracking issue #31; plan in `plan.md`. Current speed (loaded box, 4 threads): decode ~1.32 ms/token, ~74 ms per call.

## In flight (running as Sonnet subagents in local worktrees; check `git branch` for their branches)
- `v01-whistle-accuracy-2` (#38 voice): WIP patch saved in `checkpoint/patches/v01-whistle-accuracy-2.patch` (base master 1893f77; needs `git merge claude/zealous-mayer-9vx4qp` and conflict resolution in Whistle*.cs/Scalar.cs/Frontend.cs/Quant.cs). Findings: upstream's log-mel is nearly gain invariant => relative log floor (1e-8 x clip peak mel energy) implemented; `Scalar.HalfToSingle` fp16 subnormal bug fixed; upstream does not fake-quantize (A8) before conv pw2 (removed) and probably nowhere in the encoder; `StageObserver` (zero cost) and `tools/golden_whistle.py` (numpy reference from the Apache `cactus-compute/needle` whistle code; clean-room: never read `cactus-compute/cactus`) added. Encoder cosine vs upstream on long25 0.99923 -> 0.99979. record.wav/test.wav match upstream; hotword ("Captist" vs "cactus") and long25 ("victory" vs "Victoria") still differ. Next steps: (1) remove remaining encoder `Quant.FakeA8` sites in Whistle.cs one at a time, measuring against upstream (`checkpoint/v01-scratch/ab.py`-style sweeps; float reference reaches cosine 0.99993 on long25 = target); (2) re-check decoder sites, native encoder output fed to our decoder, logit margins at divergence tokens; (3) goldens (`python3 tools/golden_whistle.py` -> `ref/golden/whistle/<clip>/`), README paragraph, `WhistleStageTests`, end-to-end transcript test with the upstream strings; (4) gates + encode-ms before/after for ref/audio/test.wav at 4 threads.
- `w18-cheap-rewind` (#18) and `w28-eval-suite` (#28): just started; if their branches are missing after a container reset, restart them from the issues (prompts: see issue text; merge `claude/zealous-mayer-9vx4qp` first).

## Remaining wave 2 (not started)
#30 W30 upstream format tracking (fetch.sh pins, Model.cs version info). Start after W18 merges (both touch Model/Engine areas lightly).

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
