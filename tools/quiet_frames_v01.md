# Issue #38, round 3: quiet frames (V01QuietFrames)

Branch `v01-quiet-frames` (at origin tip `9b8feae`). **No code changes made** — nothing
was convicted to the standard required for a fix. Worktree is clean; all probes live in
`/tmp/v01q/*.py` (native side: `/tmp/v01/libneedle3.so` + `/tmp/v01/native_embed.py`).

Reference: `Whistle(cact, quant=False)` float numpy vs `needle_embed` native encoder.
C# stage dumps via `/tmp/v01q/zdump` (Observer, `needle` assembly name) on this branch.

## Headline numbers (numpy float vs native, rel err / mean cosine)

| probe | gap | cos |
|---|---|---|
| test.wav | 2.56% | 0.999561 |
| record.wav | 2.05% | 0.999750 |
| hotword.wav | 1.29% | 0.999883 |
| zeros (any length) | 1.2% | 0.99992 |
| broadband noise | 1.5% | 0.99985 |
| narrowband noise (440 Hz, Q~1) | 31% | 0.945 |
| pure tone (any 80–6800 Hz, any level) | 11–30% | 0.93–0.97 |
| chirp / AM tone | 14–19% | 0.98 |

C# and numpy agree with each other everywhere (~1%, fully explained by C#'s int8
qkv + quantized stem), while **both** differ from native — so the delta is in the
shared reverse-engineered formulae/constants or their operating point, not in C# codegen.

## Convictions (positive findings)

1. **The gap is sparsity-selective, not energy-selective.** Narrowband noise (random
   phase, no temporal coherence) gaps 20–31% while broadband gaps 1.5% at the same RMS.
   Pure tones gap 11–30% uniformly across frequency and across 100x level (27.7–27.9%,
   perfectly level-invariant). Prior "low-energy" framing is imprecise: worst test frame
   f27 (11.9%) is moderate-energy onset; deep-silence f100/f147/f177 gap <1% while
   tail-silence f230–236 gaps ~8%. What matters is spectral sparsity (many floored mel
   bins + a few large normalized outliers), not loudness.
2. **Suppressing large normalized-mel outliers moves toward native, robustly.**
   Post-norm hard clip at ±4..6: narrowband 31→13%, tone 24→17%, chirp 18→8%,
   test 2.56→2.39% (cos 0.999561→0.999631), record/hotword/zeros neutral-or-better.
   Never meaningfully hurts any probe. Noiseless clip beats int8-with-scale (rounding
   noise hurts test), so the mechanism is saturation-like, not quantization noise.
   Soft (tanh) saturation trades test for narrowband — not the shape. Residual on
   synthetics (~9–13%) is large; this is a lead, NOT a conviction — **not shipped**.
3. **Test.wav anatomy (clip-5 on/off):** f27–29 onset is outlier-driven (clip fixes it,
   drops out of top-5); tail f230–236 + f144 persist (~8%, mel max only 3.3–3.9, no
   outliers) — a second, non-outlier phenomenon at speech↔silence transitions.
   Isolated last-3s still gaps 5.9% (transition at its frame 17), so it is local to
   transitions, not far-context. Leading silence gaps less than trailing silence.
4. **Stream grows ~35x through the 8 layers** (rms 6→227) in both implementations;
   final zc removes scale. Late-layer direction dominates; small early differences amplify.

## Exonerated this round (do not re-test)

Gain scaling ×0.01–×10: gap constant (2.38–2.56%), worst frame always f27 —
denormal/FTZ dead; both sides gain-invariant (numpy self-cos exactly 1.0, native 0.9999).
VAD/trimming: ±2 s silence pads give +25 frames on both, no alignment shift, same-size
response — dead. Precision: full-float64 golden clone is bit-near-identical in gap
(1.2130% vs 1.2138% zeros; 2.5559% test) — all rounding/rsqrt/exp/FMA/order theories dead.
Weights: int8/6-bit quant and ±1–3% global scale both strictly worse — exact decode stands.
Structure (all strictly worse unless noted): LN-vs-RMS (40% — RMS confirmed), GELU,
lane combine (mean confirmed up to scale) / lane rotation (destroys), gate removals,
0.5-scale changes, cond variants, hyper-offset zeroing, GLU variants, hpost
(2.0 is a local minimum; 1.9/2.1 worse), norm axes (frame/global/whisper-style all far
worse), mel eps (0–1e-2 neutral-or-worse), floor scale 1e-9..1e-6 and max-form,
variance prior (helps narrowband, hurts test — not truth), RMS denominator (46%),
pre-cap depths, energy caps, conv gate placement (both orders tested), stem phase
(global odd-phase: test 68%), per-layer deep clip (300% — deep range is real),
final tanh (destroys), fp16 attention scores/probs/output (no effect to 4 decimals),
attention scale ±1% (trades zeros vs test), RoPE interleaved (85% — NeoX confirmed)
and base 1e4/5e5 (far worse), hard-sigmoid/hard-swish gates (far worse),
q-norm placement (identical by rotation math), additive positional offset
(corr 0.01, subtracting worsens), log-base/magnitude-vs-power (scale, absorbed by norm).

## Open (blocking cos ≥ 0.9999 on test.wav)

- Mechanism behind the outlier suppression (fixed-point frontend? saturating norm?) and
  the ~13% synthetic residual after clip.
- The transition/tail component (silent frames next to speech, no mel outliers, ~8%).
- The dense-input floor (~1.2–1.5% on zeros/broadband/hotword): deterministic,
  input-dependent, non-additive, formula-level (f64-stable), present with mel=0 and
  trivial attention. Hypotheses remaining: hyper/MLP bias-path formula nuance,
  attention temperature/structure nuance, or an untested frontend edge (tail padding).
- BeamWidth re-check on all four clips was skipped (no fix landed).
