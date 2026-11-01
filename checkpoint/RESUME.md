# Checkpoint: resume here

Updated 2026-10-30 ~09:30 UTC. Designated branch: `claude/zealous-mayer-9vx4qp` (pushed; PR #43 into `master` open). All three slices merged: CI whistle goldens, fp16-subnormal policy, voice test.wav fix. A full `NEEDLE_SCALAR=1` run on the final tree is in flight (log /tmp/scalar-final.log). Delete `checkpoint/` when you no longer need it.

## State
Wave 1+2 merged (#1-#7, #12, #14, #18, #19, #28, #30 closed), plus: Apache-2.0 LICENSE (#34 closed), CI whistle goldens (fixtures job runs `golden_whistle.py`, cache key ref-v2), fp16-subnormal policy (needle3 IEEE-exact, whistle body keeps native shift-trick decode, confidence window 0.15→0.25 with realization table in 27aed3f), voice test.wav fixed by `BeamWidth` 4→3 (6293e1d). Full suite: 220/220 (default and `NEEDLE_NOAVX512=1`), netstandard2.1 clean. #38 stays open: residual encoder gap vs native at test frames 27-29/230-233 (~0.5-0.7%, engine numerics, not front end — floor scale, hann, norm, A8 layouts all ruled out); closing it is what would make beam-4 safe too. Known pre-existing: `NEEDLE_SCALAR=1` hotword phrases fail on pristine c0ce0f9 too, `HadamardMlpTests.MatchesDoublePrecisionReference` flaky ~1/8.

## Do locally (in this order)
```bash
git clone https://github.com/IAFahim/needle-cs && cd needle-cs && git checkout claude/zealous-mayer-9vx4qp
curl -sSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh && bash dotnet-install.sh --channel 10.0   # .NET 10 SDK
tools/fetch.sh && pip install numpy sentencepiece soxr
python3 tools/golden.py && python3 tools/golden_confidence.py && python3 tools/golden_whistle.py   # ~15 min, then seconds
dotnet test tests/Needle.Tests -c Release
NEEDLE_SCALAR=1 dotnet test tests/Needle.Tests -c Release     # ~22 min, run last
dotnet publish src/Needle.Cli -c Release -o out/aot
out/aot/needle call ref/hf/needle3.cact tools/tools.json "what's it like in Lagos right now?"     # demo
out/aot/needle eval ref/hf/needle3.cact tests/eval                                                 # game-prompt accuracy
```
Then pick work from the issue board. Tracking issue #31. Order I recommend:
1. #38 voice: `test.wav` regressed (2 words differ from upstream) while hotword and long clips improved. Find the remaining 2-3% encoder error on test.wav (front end likely), see the issue comment for the list.
2. #39 + the other idle-box measurements (#37, #41, #42): run `needle bench-e2e --profile` and `needle bench` on an idle machine and record numbers in plan.md.
3. #32 open a PR into `master` to run CI on Windows/macOS/arm64 (workflow already exists), fix what fails.
4. #34 choose a LICENSE (owner decision) - blocks shipping.
5. Then the big roadmap items in plan.md: #8 channel/host, #9 session startup memory, #10/#11 typed C# tools (source generator), #13 layer program (fewer barriers), #15-#17 decoding speedups (sparse head, jump-forward, speculation), #20-#27.

## Lessons
- Subagent worktrees create a `ref -> /home/user/needle-cs/ref` symlink. NEVER commit it: a merge once replaced the real `ref/` directory (models, fixtures). `/ref` is now in .gitignore; before merging an agent branch check `git ls-tree -r <branch> --name-only | grep -x ref`. Rebuild `ref/` with the commands above.
- Running 4-5 Sonnet agents at once exhausted the usage limit twice; 2-3 at a time works. Worktrees branch from `master`: tell agents to `git merge claude/zealous-mayer-9vx4qp` first.
- Timings are meaningless while agents run in parallel; use an idle machine and interleaved A/B, best of N.

## Decisions waiting on the user
Decided with owner: Apache-2.0 (#34, done), PR into `master` opened (#43). Whistle default changed 4→3 with evidence (w=2 fails hotword, w≥4 fails test, w=3 passes all four clips; 0.0032 nats/token near-tie at beam-4). Still open: #33 spin default (300 us chosen over 50 us); #40 behavior changes of the span API.
