# N60/N61 eval results: reasoning budget sweep (2026-11-02/03)

All runs: `needle eval ref/hf/needle3.cact tests/eval --threads 8`
(208 prompts: chains 39, dialogue 39, rpg 43, rts 43, settings 44),
post-N61 binary (fold + `ReasoningBudget` alias), policy names from
`EvalPolicies.Settings` (`MaxReasoningTokens=N`).

## N61: accuracy per reasoning budget

| budget | overall | exact | tool | args | no-call | refused | failed | false-call | mean decode tok |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.361 | 0.273 | 0.344 | 0.302 | 0.844 | 17 | 11 | 4 | 204.4 |
| 8 | 0.183 | 0.040 | 0.051 | 0.029 | 0.969 | 1 | 0 | 1 | 129.6 |
| 16 | 0.255 | 0.131 | 0.130 | 0.096 | 0.938 | 4 | 5 | 0 | 139.2 |
| 32 | 0.447 | 0.392 | 0.400 | 0.357 | 0.750 | 32 | 22 | 7 | 53.4 |
| 96 (default) | 0.471 | 0.432 | 0.479 | 0.460 | 0.688 | 32 | 34 | 8 | 41.2 |

Per-set overall accuracy (budget 0 / 8 / 16 / 32 / 96):

| set | n | 0 | 8 | 16 | 32 | 96 |
|---|---|---|---|---|---|---|
| chains | 39 | 0.282 | 0.154 | 0.154 | 0.179 | 0.282 |
| dialogue | 39 | 0.308 | 0.179 | 0.231 | 0.462 | 0.487 |
| rpg | 43 | 0.302 | 0.163 | 0.209 | 0.349 | 0.395 |
| rts | 43 | 0.326 | 0.233 | 0.233 | 0.512 | 0.465 |
| settings | 44 | 0.568 | 0.182 | 0.432 | 0.705 | 0.705 |

Outcome mix (Called / Refused / Responded / Failed):
96: 117 / 46 / 11 / 34. 32: 99 / 45 / 42 / 22.
16: 32 / 7 / 164 / 5. 8: 10 / 3 / 195 / 0. 0: 66 / 19 / 112 / 11.

Readings:

- **Budget 32 is indistinguishable from the default.** 44.7% vs 47.1%
  overall sits inside the run-to-run noise floor (three default runs:
  47.1 / 44.7 / 43.8, see N60 below). Matched-call analysis
  (90 cases Called with identical output at 96 and 32) shows a mean
  decode delta of 0.1 tokens: the 32 cap essentially never binds.
- **Budgets 16/8 collapse, and 8 is worse than 0.** Truncated thinking
  flips calls into long Respond paths (195/208 Responded at 8), so mean
  decode tokens *rise* (129.6 at 8 vs 41.2 at 96). Zero thinking keeps a
  clean prefill decision (exact 27.3%, above 8 and 16).
- **Cap 0 is NOT a latency win on this set**, contra the "largest lever"
  hope: it saves all reasoning passes plus one folded prefill pass, but
  mean decode tokens grow 5x (204.4 vs 41.2, responses hit the 384 cap)
  and overall accuracy drops 11 pts (47.1 -> 36.1).
- **No default changes.** Default stays `ReasoningBudget = 96`
  (`MaxReasoningTokens` alias, same value); jump-forward and
  prompt-lookup stay default-off until W28 passes.

## Reasoning share of decode: 54% CONFIRMED

36 cases Called with byte-identical output at budgets 96 and 0:
decode(96) - decode(0) = reasoning tokens (call tokens cancel).
Mean reasoning 21.5 tokens on mean decode 39.2 = **54.7%**,
vs 53.8% (342/636) in plan.md §2.4. Confirmed on this set.

## N60: jump-forward + prompt lookup, `--compare` identity gate

`needle eval ... --compare` (baseline and candidate both defaults;
neither `JumpForward` nor `PromptLookup` is in `EvalPolicies.Settings`,
so both stay off): baseline 44.7%, candidate 43.8%, **8 flips / 208**.
All flips are 8-thread greedy nondeterminism (cf. ChunkParityTests:
multi-threaded chunk-vs-sequential drift ~0.36 abs logit), not a code
difference. Noise floor for this set at 8 threads: about ±1.7 pts
overall (47.1 / 44.7 / 43.8 across three default runs).
No accuracy drop to report (nothing enabled); no defaults flipped.

Passes per call (both speculative paths off, so every decode token is
exactly one engine pass; prefill is one chunked pass, queries stay
under 64 tokens): passes/call ≈ mean-decode-tokens + 4 fixed passes
(prefill 1 + think-tail 1 + ToolCallStart 1 + closing 1).
At the default: 41.2 + 4 ≈ **45 passes/call**, of which ~54% (≈22) are
reasoning. Budget 0 with the N61 fold skips the think-tail pass
(+3 instead of +4) but the response blowup above dominates.

## Code change (src/Needle/NeedleSession.cs only)

- `NeedleOptions.ReasoningBudget` (default 96): the host-facing budget
  knob. `MaxReasoningTokens` is kept as a pass-through alias (eval
  policy name, existing hosts/tests) with the same default; setting
  either sets the same value.
- Budget <= 0 folds `thinkEndForced` (`</think>` + newline) into the
  `EncodePrompt` prefill chunk, so the call/respond decision ArgMax
  reads the prefill logits directly and `Reason` skips its separate
  tail feed: one engine pass saved per call. Same token sequence either
  way; single-threaded chunk feeds are bit-identical to split feeds
  (ChunkParityTests), multi-threaded drift is covered by this eval gate.
- `Reason` loop bound uses the same budget (alias: identical value).
  `BatchedSession` is untouched: at budget 0 it still pays the separate
  tail pass (port when the batched host lands it).
