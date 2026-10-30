# Checkpoint: resume here

Updated 2026-10-30 ~01:35 UTC. Designated branch: `claude/zealous-mayer-9vx4qp`; everything below is merged and pushed there. No PR has been opened and no other branch has been pushed. Delete `checkpoint/` when you no longer need it.

## State
Wave 1 and wave 2 are merged: #1-#7, #12, #14, #18, #19, #28, #30 (closed). #38 (voice) is partly fixed and still open. Full suite on the merged tree: 216 tests pass (default and `NEEDLE_NOAVX512=1`), netstandard2.1 build clean; a `NEEDLE_SCALAR=1` run was started at ~01:30 UTC (log was /tmp/scalar.log, takes ~22 min) - re-run it to confirm. Speed on the 4-core reference VM at 4 threads: ~74 ms per call, decode ~1.3 ms/token (upstream native 95-129 ms).

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
LICENSE (#34); whether to open a PR into `master`; #33 spin default (300 us chosen over 50 us); #40 behavior changes of the span API; Whistle default `BeamWidth` 4 (better accuracy, ~2x slower transcription) vs 1.
