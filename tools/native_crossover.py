#!/usr/bin/env python3
"""#38 round 5: crossover re-run (native .so vs numpy reference) on this box.

Reproduces the round-2 crossover numbers with the Apache-2.0 numpy reference
(`Whistle(cact, quant=False)` from tools/golden_whistle.py) against upstream
`needle_embed` from the local wheel, on identical 16 kHz input.

Per clip/probe it reports: overall relative error, mean frame cosine, the 5
worst frames, and error by native-RMS tercile (the quiet-frame lead says quiet
frames err more). It also re-checks the round-3 sparsity pattern (narrowband
vs broadband noise at equal RMS) and native determinism (embed twice).

Usage:
    python3 tools/native_crossover.py [--clips record,test,hotword,long25]
                                      [--out report.json]
"""

import argparse
import ctypes
import glob
import json
import os
import sys
import tempfile
import wave
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import Cact  # noqa: E402
from golden_whistle import CLIPS, SAMPLE_RATE, Whistle, read_wav  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WHEEL_DIR = os.path.join(ROOT, "ref/hf/wheel")
PREFERRED_LIB = os.path.join(ROOT, "ref/native/libneedle3.so")
CACT = os.path.join(ROOT, "ref/hf/whistle/whistle.cact")
DIM = 512


def locate_library():
    if os.path.isfile(PREFERRED_LIB):
        return PREFERRED_LIB
    wheels = sorted(glob.glob(os.path.join(WHEEL_DIR, "*.whl")))
    if not wheels:
        sys.exit(f"no .whl in {WHEEL_DIR} and no {PREFERRED_LIB}")
    target = os.path.join(tempfile.mkdtemp(prefix="needle-native-"), "libneedle3.so")
    with zipfile.ZipFile(wheels[-1]) as archive:
        member = next(n for n in archive.namelist() if n.endswith("libneedle3.so"))
        with archive.open(member) as src, open(target, "wb") as dst:
            dst.write(src.read())
    return target


def bind(path):
    lib = ctypes.CDLL(os.path.abspath(path))
    lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
    lib.needle_load.restype = ctypes.c_int
    lib.needle_embed.argtypes = [ctypes.c_char_p,
                                 ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.needle_embed.restype = ctypes.c_int
    return lib


def native_frames(lib, pcm):
    pcm = np.ascontiguousarray(pcm, np.float32)
    capacity = (len(pcm) // 160 + 8) * DIM + DIM
    out = np.zeros(capacity, np.float32)
    f32 = ctypes.POINTER(ctypes.c_float)
    got = lib.needle_embed(None, pcm.ctypes.data_as(f32), len(pcm),
                           out.ctypes.data_as(f32), capacity)
    if got <= 0:
        sys.exit(f"needle_embed failed (rc={got}) on {len(pcm)} samples")
    return out[:got].reshape(got // DIM, DIM).copy()


def frame_cosine(a, b):
    denom = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return (a * b).sum(-1) / np.maximum(denom, 1e-30)


def compare(name, ref, nat):
    assert ref.shape == nat.shape, f"{name}: shape {ref.shape} vs {nat.shape}"
    diff = ref - nat
    rel = float(np.sqrt((diff * diff).sum()) / max(np.sqrt((nat * nat).sum()), 1e-30))
    cos = frame_cosine(ref, nat)
    per_frame = np.sqrt((diff * diff).sum(-1)) / np.maximum(np.sqrt((nat * nat).sum(-1)), 1e-30)
    rms = np.sqrt((nat * nat).mean(-1))
    order = np.argsort(rms)
    thirds = np.array_split(order, 3)
    entry = {
        "frames": int(ref.shape[0]),
        "rel_err_pct": 100.0 * rel,
        "mean_cosine": float(cos.mean()),
        "min_cosine": float(cos.min()),
        "worst_frames": [int(i) for i in np.argsort(-per_frame)[:5]],
        "worst_frame_err_pct": [100.0 * float(per_frame[i]) for i in np.argsort(-per_frame)[:5]],
        "quiet_tercile_err_pct": 100.0 * float(per_frame[thirds[0]].mean()),
        "mid_tercile_err_pct": 100.0 * float(per_frame[thirds[1]].mean()),
        "loud_tercile_err_pct": 100.0 * float(per_frame[thirds[2]].mean()),
    }
    return entry


def synth_probes(sr, seconds=2, seed=7):
    rng = np.random.default_rng(seed)
    n = seconds * sr
    t = np.arange(n, dtype=np.float64) / sr
    white = rng.standard_normal(n).astype(np.float32)
    phase_jumps = np.repeat(rng.uniform(0, 2 * np.pi, n // 128 + 1), 128)[:n]
    narrow = np.sin(2 * np.pi * 440.0 * t + phase_jumps).astype(np.float32)
    tone = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    zeros = np.zeros(n, np.float32)
    return {"zeros": zeros, "broadband": white, "narrowband": narrow, "tone440": tone}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", default=",".join(CLIPS),
                        help="comma-separated subset of record,test,hotword,long25")
    parser.add_argument("--managed", default=None,
                        help="managed encoder dump (raw frames x 512 float32, row-major) for one clip; "
                             "adds managed-vs-numpy and managed-vs-native rows for that clip")
    parser.add_argument("--managed-clip", default="test",
                        help="which clip the managed dump corresponds to")
    parser.add_argument("--synth", action="store_true", help="also run synthetic probes")
    parser.add_argument("--out", default=None, help="write JSON report here")
    options = parser.parse_args()

    lib = bind(locate_library())
    with open(CACT, "rb") as handle:
        blob = handle.read()
    if lib.needle_load(blob, len(blob)) < 0:
        sys.exit("needle_load failed on " + CACT)
    cact = Cact(CACT)
    # Empty quant set == fully float reference: golden_whistle.encode's
    # `set(self.quant)` path throws on a bare False, and an empty set
    # disables every fake-quant site identically.
    reference = Whistle(cact, quant=set())
    report = {"clips": {}, "synth": {}, "native_determinism": {}}
    managed = None
    if options.managed:
        managed = np.fromfile(options.managed, np.float32).reshape(-1, DIM).astype(np.float64)
    pcm_cache = {}
    for name in [c for c in options.clips.split(",") if c.strip()]:
        filename, limit = CLIPS[name]
        pcm = read_wav(os.path.join(ROOT, "ref/audio", filename), limit)
        pcm_cache[name] = pcm
        ref = reference.encode(pcm)["encoder"].astype(np.float64)
        nat = native_frames(lib, pcm).astype(np.float64)
        report["clips"][name] = compare(name, ref, nat)
    if managed is not None:
        name = options.managed_clip
        if name not in pcm_cache:
            filename, limit = CLIPS[name]
            pcm_cache[name] = read_wav(os.path.join(ROOT, "ref/audio", filename), limit)
        clip_ref = reference.encode(pcm_cache[name])["encoder"].astype(np.float64)
        clip_nat = native_frames(lib, pcm_cache[name]).astype(np.float64)
        report["managed_vs_numpy"] = compare("managed_vs_numpy", clip_ref, managed)
        report["managed_vs_native"] = compare("managed_vs_native", clip_nat, managed)

    test_pcm = pcm_cache.get("test")
    if test_pcm is None:
        test_pcm = read_wav(os.path.join(ROOT, "ref/audio", "test.wav"), None)
    twice = [native_frames(lib, test_pcm) for _ in range(2)]
    report["native_determinism"] = {
        "max_abs_diff": float(np.abs(twice[0] - twice[1]).max()),
        "frames": int(twice[0].shape[0]),
    }

    if options.synth:
        probes = synth_probes(SAMPLE_RATE)
        scale = float(np.abs(test_pcm).mean())
        for name, probe in probes.items():
            if name != "zeros":
                probe = probe * (scale / max(float(np.abs(probe).mean()), 1e-9))
            ref = reference.encode(probe)["encoder"].astype(np.float64)
            nat = native_frames(lib, probe).astype(np.float64)
            report["synth"][name] = compare(name, ref, nat)

    for name, entry in list(report["clips"].items()) + list(report["synth"].items()):
        print(f"{name:<10} frames={entry['frames']:<5} rel={entry['rel_err_pct']:.2f}% "
              f"cos={entry['mean_cosine']:.6f} worst={entry['worst_frames']} "
              f"terciles q/m/l={entry['quiet_tercile_err_pct']:.2f}/"
              f"{entry['mid_tercile_err_pct']:.2f}/{entry['loud_tercile_err_pct']:.2f}%")
    for name in ("managed_vs_numpy", "managed_vs_native"):
        if name in report:
            entry = report[name]
            print(f"{name:<16} frames={entry['frames']:<5} rel={entry['rel_err_pct']:.2f}% "
                  f"cos={entry['mean_cosine']:.6f} worst={entry['worst_frames']}")
    det = report["native_determinism"]
    print(f"native embed twice: max abs diff {det['max_abs_diff']:.3e} over {det['frames']} frames")
    if options.out:
        with open(options.out, "w") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
