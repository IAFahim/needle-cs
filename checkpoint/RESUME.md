# Checkpoint: resume here

Written 2026-10-29 ~20:45 UTC. Designated branch: `claude/zealous-mayer-9vx4qp` (everything merged is pushed). Do not open a PR or push other branches unless the user says so. Delete the `checkpoint/` directory (and mention it in the commit) once all items below are merged.

## Merged and pushed (179 tests pass in default, NEEDLE_SCALAR=1 and NEEDLE_NOAVX512=1 tiers; netstandard2.1 clean)
W01 #1, W03 #3, W04 #4, W05 #5, W06 #6, W12 #12, W02 #2, W07 #7, W14 #14, W19 #19 (all closed). Follow-ups filed: #32 F01 CI on other OSes, #33 F02 spin default, #34 F03 LICENSE (owner decision), #35 F04 fresh-clone check, #36 F05 large pages, #37 F06 MLP remaining, #39 F07 idle-box bench, #40 F08 zero-alloc follow-ups, #41 F09 attention remaining, #42 F10 GEMV remaining. Tracking issue #31; plan in `plan.md`. Current speed (loaded box, 4 threads): decode ~1.32 ms/token, ~74 ms per call.

## Also merged and pushed after the first checkpoint
W18 #18 (O(1) Rewind, ~1.5 ns) and W28 #28 (eval suite, `needle eval`, tests/eval; merge pending push and full test verification) are merged on the branch. Issues #18 and #28 still need verification comments and closing once the full suite is green on the merged tree: after the fixtures exist run default, `NEEDLE_SCALAR=1` (takes ~20 min, run it last) and `NEEDLE_NOAVX512=1`, plus the netstandard2.1 build. W28 baseline to record in plan.md: exact 65.9% on 208 hand-written game prompts (AutoDate=false RetrievalTopK=64), tool routing 77.2%, no-call correctness 78.1%; per set exact: chains 73.5, dialogue 60.6, rpg 47.2, rts 63.9, settings 83.8; calibration: conf>=0.5 gives 86-100% accuracy, <0.1 gives 7%; recommended RefuseBelow 0.3 for games (0.5+ for destructive commands); default RetrievalTopK=5 costs -6.8 pp exact and ~3x latency on the 7-tool sets; MaxReasoningTokens=32 costs -8.5 pp, 0 costs -31 pp.

## LESSON: never commit a `ref` symlink
Subagent worktrees symlink `ref -> /home/user/needle-cs/ref`. W18 committed that symlink; merging it replaced the real `ref/` directory (models, goldens) with a self-referencing symlink. Fix applied: `/ref` added to .gitignore, symlink removed from the merge. When merging agent branches always check `git ls-tree -r <branch> --name-only | grep -x ref`. To rebuild `ref/`: `tools/fetch.sh` (5 s) then `python3 tools/golden.py && python3 tools/golden_confidence.py` (~15 CPU minutes), then `python3 tools/golden_whistle.py` for the Whistle goldens (V01 branch).

## In flight (Sonnet subagent in a local worktree; check `git branch`)
- `v01-whistle-accuracy-2` (#38 voice): commits 73f4d8a, 847efaa (+ more); base master, already merged once with the designated branch. WIP patch (older) in `checkpoint/patches/v01-whistle-accuracy-2.patch`. Findings: upstream's log-mel is nearly gain invariant => relative log floor (1e-8 x clip peak mel energy) implemented; `Scalar.HalfToSingle` fp16 subnormal bug fixed; upstream does not fake-quantize (A8) in the encoder (being removed site by site); `StageObserver` and `tools/golden_whistle.py` (numpy reference from the Apache `cactus-compute/needle` whistle code; clean-room: never read `cactus-compute/cactus`). Encoder cosine vs upstream on long25 0.99923 -> 0.99979 before the A8 removals. record.wav/test.wav match upstream; hotword ("Captist" vs "cactus") and long25 ("victory" vs "Victoria") still differ at last check. Remaining: finish A8 removals measured against upstream (`checkpoint/v01-scratch/ab.py`-style sweeps), decoder-site check, Whistle goldens + `WhistleStageTests` + end-to-end transcript test with upstream strings, README paragraph, gates, encode-ms before/after.

## Remaining wave 2 (not started)
#30 W30 upstream format tracking (fetch.sh pins, Model.cs version info). Can start now.

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
