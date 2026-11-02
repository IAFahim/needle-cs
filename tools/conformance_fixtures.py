#!/usr/bin/env python3
"""Conformance fixture spec for Needle.Inference kernel table (plan-netstandard.md N13).

Layout mirrors plan-netstandard.md section 2.1: one entry per kernel-table row
(MatMulRows, Prepare, GatherRow, AttentionTile, MergePartials, HadamardMlp,
RowOps list, ArgMaxMasked, BestProbability). Two jobs:

  generate --out <dir>   write inputs.json: deterministic input spec (seeds,
                         shapes, edge vectors). The C# suite in
                         tests/Needle.Tests/Conformance/ carries the same
                         numbers compiled in, so tests run with no files
                         present (dotnet test, N11a host, Mono N11b harness,
                         unity test). The JSON is the recorded spec.
  check --dir <dump>      validate a CONFORMANCE_DUMP directory produced by
                         `CONFORMANCE_DUMP=<dir> dotnet test --filter
                         Conformance`: every manifest entry has its .bin,
                         byte length matches, FNV-1a 64 of the bytes matches.
                         The .bin files hold net10 SIMD kernel outputs, so a
                         Burst provider port can be compared bitwise (H7).

Golden math truths printed by `golden` are embedded in both the xunit suite
and the Unity EditMode mirror (package Tests/Conformance/). They are pure
float32 (struct) evaluations, portable across runtimes.
"""

import argparse
import json
import math
import os
import struct
import sys

ENTRIES = {
    "MatMulRows": {"seeds": [0x1234, 0x5678, 0x9ABC], "bits": [2, 4],
                   "shapes": [[4, 128], [3, 128], [5, 768], [7, 384], [97, 768]],
                   "tokens": [0, 1, 2, 3, 5, 8], "edges": ["zero_rows", "odd_tails", "int8_min_max"]},
    "Prepare": {"seeds": [0x2458], "cols": [128, 768], "tokens": [1, 3], "bits": [2, 4],
                "edges": ["zeros", "denormals", "rounding_ties", "odd_tails"]},
    "GatherRow": {"seeds": [0xACE1], "bits": [2, 4], "cols": [128, 768],
                  "edges": ["first_row", "last_row", "zero_norm"]},
    "AttentionTile": {"seeds": [0xBEEF], "heads": [1, 2, 6], "counts": [1, 2, 4, 45],
                      "firsts": [0, 13, 41], "edges": ["count_1", "full_span", "tail_span", "int8_min_max"]},
    "MergePartials": {"seeds": [0xF001], "counts": [1, 2, 3], "heads": [1, 6],
                      "edges": ["count_1", "count_2_fast_path"]},
    "HadamardMlp": {"seeds": [0x1A5E], "dims": [128], "rows": [4], "cols": [32],
                    "edges": ["identity_perm", "rotated_perm"]},
    "RowOps": {"seeds": [0x0DD5], "lengths": [0, 1, 7, 8, 9, 13, 16, 17, 31, 32, 33, 64, 100, 768, 8192],
               "ops": ["RmsFakeA8", "CenteredRmsFakeA8", "GateFakeA8", "PostNormAdd",
                       "CombineDiff4", "Mix4", "Combine4", "Taps3", "Rope",
                       "ZeroCenteredRms", "RmsUnit", "Softmax", "Dot", "AddScaled",
                       "ScaleAdd", "MultiplyAccumulate", "QuantizeInt8", "FakeA8", "Sinkhorn4x4"],
               "edges": ["zero_length", "zeros", "denormals", "rounding_ties", "int8_min_max", "odd_tails"]},
    "ArgMaxMasked": {"seeds": [0xA969], "lengths": [0, 1, 31, 32, 33, 64, 100, 8192],
                     "edges": ["empty_mask", "single_bit", "full_words", "sparse", "ties", "odd_tail"]},
    "BestProbability": {"seeds": [0xBE57], "counts": [1, 7, 8, 9, 8192],
                        "edges": ["count_1", "denormals", "wide_range"]},
}


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]
def softmax(xs):
    peak = max(xs)
    exps = [f32(math.exp(f32(x - peak))) for x in xs]
    total = 0.0
    for e in exps:
        total += e
    return [f32(e / total) for e in exps]


def golden():
    out = {}
    out["Softmax_0_0_0"] = softmax([0.0, 0.0, 0.0])
    out["Softmax_1_2_3"] = softmax([1.0, 2.0, 3.0])
    out["Softmax_tie_first_wins"] = 0
    bp = [1.0, 2.0, 3.0]
    total = sum(f32(math.exp(f32(v - 3.0))) for v in bp)
    out["BestProbability_1_2_3"] = f32(1.0 / total)
    inv = f32(1.0 / f32(math.sqrt(f32(f32(3.0 * 3.0 + 4.0 * 4.0) / 2.0 + 1e-6))))
    out["RmsUnit_3_4"] = [f32(3.0 * inv), f32(4.0 * inv)]
    out["Dot_1to8_8to1"] = float(sum(a * b for a, b in zip(range(1, 9), range(8, 0, -1))))
    out["Sinkhorn_zeros_4x4"] = [0.25] * 16
    out["QuantizeInt8_scale_neg4_0_4"] = f32(4.0 / 127.0)
    out["QuantizeInt8_bytes_neg4_0_4"] = [-127, 0, 127]
    out["FakeA8_tie_0_5"] = 0.0
    out["FakeA8_tie_2_5"] = 2.0
    out["ArgMaxMasked_single_bit_word1_bit5"] = 37
    out["ArgMaxMasked_empty"] = -1
    print(json.dumps(out, indent=2))


def generate(path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "inputs.json"), "w") as f:
        json.dump({"entries": ENTRIES, "version": 1}, f, indent=2)
    print("wrote %s" % os.path.join(path, "inputs.json"))


def fnv1a(data):
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def check(dump):
    with open(os.path.join(dump, "manifest.json")) as f:
        manifest = json.load(f)
    failures = 0
    for entry in manifest["entries"]:
        blob = os.path.join(dump, entry["name"] + ".bin")
        if not os.path.exists(blob):
            print("missing %s" % blob)
            failures += 1
            continue
        with open(blob, "rb") as f:
            data = f.read()
        if len(data) != entry["bytes"]:
            print("size %s: %d != %d" % (entry["name"], len(data), entry["bytes"]))
            failures += 1
        if fnv1a(data) != int(entry["fnv1a"], 16):
            print("hash %s" % entry["name"])
            failures += 1
    print("%d entries, %d failures" % (len(manifest["entries"]), failures))
    return 1 if failures else 0


def main(argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--out", default="ref/conformance")
    chk = sub.add_parser("check")
    chk.add_argument("--dir", required=True)
    sub.add_parser("golden")
    args = parser.parse_args(argv[1:])
    if args.command == "generate":
        generate(args.out)
        return 0
    if args.command == "check":
        return check(args.dir)
    golden()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
