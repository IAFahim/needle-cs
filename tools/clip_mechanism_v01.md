# Issue #38, round 4: clip mechanism (V01ClipMechanism)

Branch `v01-clip-mechanism` (at origin tip `ecd4b1d`). **No code changes made** — neither
lead convicted a mechanism to the standard required for a fix. All probes live in `/tmp/v01c/`
(`a2_outlier.py`, `b_trans.py`, `c_step.py`, `d_floor.py`, `e_localize.py`, `f_win.py`,
`g_fit.py`, `h_const.py`, `i_stem.py`; native side `/tmp/v01/libneedle3.so` + `nat.py`).
Reference: `Whistle(cact, quant=False)` float numpy vs `needle_embed` native encoder.

Prior state (round 3): test.wav 2.56% / cos 0.999561; dense floor ~1.2% (zeros/broadband);
clip-at-4..6 helps sparse probes + test (2.56->2.39%) but leaves ~13% synthetic residual;
tail/transition component f230-236/f144 (~8%, no mel outliers).

## Lead (a): outlier-driven onset — VERDICT: no mechanism convicted

Synthetic outlier battery (2 s clips, 25 enc frames): zeros, broadband, burst (single-frame
loud 1 kHz pip, mel max ~14), impulse (single-sample spike), pure tone 440 Hz, step, narrowband.

Base gaps: zeros 1.18%, broad 1.13%, impulse 2.25%, burst 10.69%, tone 27.79%,
narrow 24-31%, step 18.69%. (C#==numpy proxy per round 3; C# not re-run on synthetics.)

- **Hard clip of normalized mel (pre-mixing): helps tone/narrow/test, neutral on
  broad/zeros/burst/step, HURTS impulse** (2.25% -> 8-10% at clip 3-5). Tone 27.8->12.1
  (clip 4), test 2.56->2.39 (clip 4/6), narrow 24.3->~15. Burst 10.69->10.5 (nothing).
  The one probe that should love a clip (impulse = single-frame all-bin outlier) hates it:
  native does NOT apply a global post-norm clip/saturation. Not shippable.
- **Tanh soft saturation: wrong shape.** Helps tone weakly, WRECKS steps (18.7->29.9 at
  T=3) and hurts test at small T. Dead.
- **Fixed-point sims: dead.** int8 post-norm (scales 1/32..1/8), int16 post-norm
  (1/64..1/16), int16 logmel (0.01..0.1), logmel range-cap+quantize: all worse-or-equal
  to noiseless clip; rounding noise hurts test. Saturation-without-noise is the shape,
  but no saturation point reproduces native.
- **Row-norm (L2) clip per frame: same pattern as value clip** (tone 27.8->10.8 at C=24,
  impulse 2.25->9.9, burst/step unchanged). "Clipping norm" does not reproduce native either.
- **Output-side fitting is completely negative** (`g_fit.py`): at every bad frame of
  burst/tone/test, best-fit scalar gain, hard clip (T=2..20) and tanh (T=2..14) applied to
  OUR encoder vector leave residual ≈ base (e.g. test f27: 11.9->11.7%; burst f0:
  12.5->12.1%). |native|/|numpy| ≈ 0.95-1.04 everywhere on test. The error is a DIRECTION
  difference in 512-space, not a magnitude/saturation difference at the output.
- **Stem-internal clip (post-mixing saturation): nothing new** (`i_stem.py`). Clip after
  stem dw1/dw2 = no-op (values < 4); after pw1 (T=8) ≈ mel-clip on tone (11.9%), worse on
  test (2.43 vs 2.39); stemout (T=4) weaker (tone 18.8, test 2.44). A fixed-point/saturating
  stem does not reproduce native beyond what mel-clip already does.

## Lead (b): transition-local tail — VERDICT: amplified, not a separate mechanism

- Sustained steps (silence<->loud noise) reproduce a HUGE gap with zero mel outliers
  (mel max 1.3): stepup 14-21%, stepdn 8.6%, ramp 16% (single frames to 73-156%).
  Brief events are fine: 5-frame tone pip 1.19%, 10-frame silence gap 2.40%.
- **Conv padding EXONERATED again, decisively**: stem/encoder edge/reflect -> 88-100%
  (destroys everything). Zero-pad confirmed. Causal (left-only) pads worse (stepup 21.8->
  39.7/111/112%). Phase/placement dead.
- **Streaming/state-carry EXONERATED**: append-B-after-A / prepend-B-before-A change the
  shared prefix identically in native and numpy (42.84 vs 42.97%; 71.04 vs 70.83%).
  Both equally (non-)causal. Gap on shared region barely moves with added context
  (1.01->1.21%). No chunking: step-position sweep (boundary at mel 50/100/150) shows spikes
  TRACK the boundary, never at absolute positions.
- **Norm statistics EXONERATED again, comprehensively**: per-bin floor (1e-6..1e-10,
  add/max form) wrecks test (2.56->8-11%) while helping sparse probes — a trade, not truth.
  Per-frame floor replicates dead (test 16%). Sliding-window (25..400), causal (100/200),
  block (100/200) stats all worse than global, monotone toward global. Global per-bin
  mean/std stands.
- **Anatomy** (`e_localize.py`, 4 s clips): abrupt tone 23.5% -> faded (0.5 s raised-cosine)
  2.30%. The tone gap is essentially ALL onset/offset transient; sustained tone ≈ 2.3-2.6%.
  Burst gap is position-independent in magnitude (22-25% wherever placed) but worst frames
  track the burst. test.wav excluding 27-29/230-233: 2.365% (bulk 30-229: 2.27%).
- **Output-side fitting negative on steps too** (direction differs, magnitudes match).
- Picture: a SMALL frontend delta at transients, broadcast globally by knife-edge attention
  (spikes 40-150% at boundaries, chaotic across noise realizations) plus a ~2.3% sustained
  residue. No separate transition-local formula found; the tail (f230-236) is the same
  onset physics at speech->silence edges.

## Constant-input structure (`h_const.py`)

Zeros (mel exactly 0) and faded-tone plateaus: numpy and native outputs have identical
second-order stats (TV/rms matches to 4 decimals) but differ 1.2-2.6% per frame; the mid
difference is half constant-vector, half per-frame-varying; per-dim gain 1.00 ± 0.01.
Rules out DC-offset/gain mechanisms; consistent with a small rotation/phase-like
difference (attention re-weighting from a small early delta).

## Exonerated this round (do not re-test)

Per-bin relative floor; per-frame floor (replicated); sliding/causal/block norm windows;
causal conv padding; streaming/chunked encoder (causality test); output-side gain/clip/tanh
(any frame); stem-internal saturation (dw1/dw2 no-ops, pw1/stemout ≤ mel-clip); int8/int16
fixed-point frontend (logmel and post-norm); row-norm clip; tanh saturation.

## Open (blocking cos ≥ 0.9999 on test.wav, beam-4 upstream transcripts)

- The ~2.3% sustained residue on speech/faded-tone (dense floor 1.2% + ~1% sparse
  amplification): formula-level, directional, magnitude-neutral. Not localized to any stage
  testable through the PCM-only native API.
- The transient delta itself: invisible steady-state, large at onsets/offsets; all
  standard frontend knobs (window, placement, FFT length, floor, compression,
  pre-emphasis, stats, padding) exonerated across rounds 1-4.
- Beam re-check skipped (no fix landed): default BeamWidth=3 still passes all four clips;
  beam 4 still flips test.wav ("grading/white file"). The encoder gap fully explains the
  flip (round 2 crossover stands).
