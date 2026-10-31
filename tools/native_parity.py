#!/usr/bin/env python3
"""Native-vs-managed parity matrix (W29).

Runs upstream libneedle3.so and the managed `needle call` CLI on identical
prefixes (empty system, and one fixed date string) over the W28 prompt sets,
then reports where calls, arguments, confidence and reasoning differ.

The native library comes from the local upstream wheel (ref/hf/wheel); it is
never committed. Managed runs use one fresh process per query, mirroring the
native reset between queries.

Usage:
    python3 tools/native_parity.py [--out report.json] [--markdown report.md]
"""
import argparse
import ctypes
import glob
import json
import os
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WHEEL_DIR = os.path.join(ROOT, "ref/hf/wheel")
MODEL = os.path.join(ROOT, "ref/hf/needle3.cact")
GOLDEN_TOOLS = os.path.join(ROOT, "tools/tools.json")
GAME_TOOLS = os.path.join(ROOT, "tools/game_tools.json")
NEEDLE = os.path.join(ROOT, "src/Needle.Cli/bin/Release/net10.0/needle")

GOLDEN_QUERIES = [
    "what's it like in Lagos right now?",
    "set a timer for 12 minutes and play some Miles Davis",
]
GAME_QUERIES = [
    "move to 120, 45",
    "attack the orc chieftain",
    "give Lyra 3 potions",
    "switch to wedge formation and follow the captain",
    "say hello traveler, the bridge is out",
    "stop everything",
    "give 50 gold to Borin then move to 10, 20",
    "follow Mira",
    "attack the watchtower near the river",
    "form a circle",
]
SYSTEMS = ["", "date: 2026-10-26 Mon 09:41"]

OUTPUT_CAPACITY = 1 << 16


def locate_library():
    direct = os.path.join(WHEEL_DIR, "libneedle3.so")
    if os.path.isfile(direct):
        return direct
    wheels = sorted(glob.glob(os.path.join(WHEEL_DIR, "*.whl")))
    if not wheels:
        sys.exit(f"no upstream wheel in {WHEEL_DIR}; run tools/fetch.sh")
    target = os.path.join(tempfile.gettempdir(), "needle-cs-native", "libneedle3.so")
    if not os.path.isfile(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with zipfile.ZipFile(wheels[-1]) as archive:
            member = next(n for n in archive.namelist() if n.endswith("libneedle3.so"))
            with archive.open(member) as source, open(target, "wb") as out:
                out.write(source.read())
    return target


def bind(path):
    lib = ctypes.CDLL(os.path.abspath(path))
    lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
    lib.needle_load.restype = ctypes.c_int
    lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.needle_init.restype = ctypes.c_int
    lib.needle_complete.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
                                    ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.needle_complete.restype = ctypes.c_int
    lib.needle_reset.argtypes = []
    lib.needle_reset.restype = None
    return lib


def native_run(lib, system, tools_json, query):
    lib.needle_reset()
    buffer = ctypes.create_string_buffer(OUTPUT_CAPACITY)
    status = lib.needle_complete(query.encode(), None, 0, 256, buffer, len(buffer))
    if status < 0:
        return {"error": buffer.value.decode(errors="replace")}
    raw = buffer.value.decode()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": raw[:200]}
    calls = result.get("function_calls") or []
    return {
        "names": ",".join(c["name"] for c in calls),
        "arguments": [c.get("arguments") for c in calls],
        "confidence": result.get("confidence"),
        "reasoning": result.get("reasoning", ""),
        "response": result.get("response", ""),
    }


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def managed_run(system, tools_path, query, threads):
    env = dict(os.environ, NEEDLE_THREADS=str(threads))
    env.pop("NEEDLE_SYSTEM", None)
    env.pop("NEEDLE_DEBUG", None)
    if system == "":
        args = [NEEDLE, "call", MODEL, tools_path, "--no-date", "--no-retrieval", query]
    else:
        env["NEEDLE_SYSTEM"] = system
        args = [NEEDLE, "call", MODEL, tools_path, "--no-retrieval", query]
    process = subprocess.run(args, env=env, capture_output=True, text=True)
    if process.returncode != 0:
        return {"error": (process.stderr or process.stdout)[-500:]}
    prefix = None
    for line in process.stderr.splitlines():
        if "prefix" in line:
            prefix = int(line.split("prefix")[1].split("tokens")[0])
    try:
        result = json.loads(process.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": process.stdout[-500:]}
    calls = result.get("function_calls") or []
    return {
        "prefix_tokens": prefix,
        "names": ",".join(c["name"] for c in calls),
        "arguments": [c.get("arguments") for c in calls],
        "confidence": result.get("confidence"),
        "head": result.get("confidence_head"),
        "decode": result.get("confidence_decode"),
        "reasoning": result.get("reasoning", ""),
        "response": result.get("response", ""),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="", help="write JSON report here")
    parser.add_argument("--markdown", default="", help="write markdown report here")
    parser.add_argument("--threads", type=int, default=4)
    options = parser.parse_args()

    lib = bind(locate_library())
    with open(MODEL, "rb") as handle:
        data = handle.read()
    if lib.needle_load(data, len(data)) < 0:
        sys.exit("needle_load refused the model bytes")

    rows = []
    for tools_path, queries in ((GOLDEN_TOOLS, GOLDEN_QUERIES), (GAME_TOOLS, GAME_QUERIES)):
        with open(tools_path, "rb") as handle:
            tools_json = handle.read()
        for system in SYSTEMS:
            prefix_tokens = lib.needle_init(system.encode(), tools_json, None)
            for query in queries:
                expected = native_run(lib, system, tools_json, query)
                actual = managed_run(system, tools_path, query, options.threads)
                same_calls = expected.get("names") == actual.get("names")
                same_args = same_calls and canonical(expected.get("arguments")) == canonical(actual.get("arguments"))
                head, decode = actual.get("head"), actual.get("decode")
                rows.append({
                    "tools": os.path.basename(tools_path),
                    "system": system if system else "<empty>",
                    "query": query,
                    "prefix_tokens": {"native": prefix_tokens, "managed": actual.get("prefix_tokens"),
                                      "match": prefix_tokens == actual.get("prefix_tokens")},
                    "names": {"native": expected.get("names"), "managed": actual.get("names"), "match": same_calls},
                    "arguments": {"native": expected.get("arguments"), "managed": actual.get("arguments"), "match": same_args},
                    "confidence": {
                        "native": expected.get("confidence"), "managed": actual.get("confidence"),
                        "head": head, "decode": decode,
                        "delta": (None if expected.get("confidence") is None or actual.get("confidence") is None
                                  else abs(expected["confidence"] - actual["confidence"])),
                        "decode_ge_head": None if head is None or decode is None else decode >= head,
                    },
                    "reasoning_match": expected.get("reasoning") == actual.get("reasoning"),
                    "reasoning": {"native": expected.get("reasoning"), "managed": actual.get("reasoning")},
                })

    mismatched_prefix = [r for r in rows if not r["prefix_tokens"]["match"]]
    mismatched_calls = [r for r in rows if not r["names"]["match"]]
    mismatched_args = [r for r in rows if r["names"]["match"] and not r["arguments"]["match"]]
    worst_confidence = max((r for r in rows if r["confidence"]["delta"] is not None),
                           key=lambda r: r["confidence"]["delta"])
    reasoning_diffs = [r for r in rows if not r["reasoning_match"]]
    report = {
        "sets": ["golden", "game"],
        "systems": ["<empty>" if not s else s for s in SYSTEMS],
        "mismatched_prefix": len(mismatched_prefix),
        "mismatched_calls": len(mismatched_calls),
        "mismatched_arguments": len(mismatched_args),
        "worst_confidence_delta": {"delta": worst_confidence["confidence"]["delta"],
                                   "query": worst_confidence["query"],
                                   "system": worst_confidence["system"]},
        "reasoning_diffs": len(reasoning_diffs),
        "rows": rows,
    }
    if mismatched_prefix:
        for row in mismatched_prefix:
            print(f"PREFIX MISMATCH {row['tools']} [{row['system']}]: native={row['prefix_tokens']['native']} managed={row['prefix_tokens']['managed']}",
                  file=sys.stderr)
        sys.exit(f"{len(mismatched_prefix)} prefix mismatches; fix prefixes before comparing outputs")
    lines = ["# Native parity (W29)", "",
             f"rows={len(rows)} call mismatches={len(mismatched_calls)} "
             f"argument mismatches={len(mismatched_args)} reasoning diffs={len(reasoning_diffs)}",
             f"worst confidence delta={worst_confidence['confidence']['delta']:.4f} "
             f"({worst_confidence['query']!r} [{worst_confidence['system']}])", "",
             "| tools | system | query | calls | args | conf delta | decode>=head | reasoning |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        cells = [row["tools"], row["system"], row["query"][:42],
                 "same" if row["names"]["match"] else
                 f"N:{row['names']['native']} M:{row['names']['managed']}",
                 "same" if row["arguments"]["match"] else "DIFFER",
                 "n/a" if row["confidence"]["delta"] is None else f"{row['confidence']['delta']:.4f}",
                 "n/a" if row["confidence"]["decode_ge_head"] is None
                 else str(row["confidence"]["decode_ge_head"]),
                 "same" if row["reasoning_match"] else "differ"]
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines)

    if options.out:
        with open(options.out, "w") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    if options.markdown:
        with open(options.markdown, "w") as handle:
            handle.write(text + "\n")
    print(text)
    for row in mismatched_calls + mismatched_args:
        print(f"\n### {row['query']!r} [{row['system']}]")
        print(f"  native reasoning: {row['reasoning']['native']}")
        print(f"  managed reasoning: {row['reasoning']['managed']}")
        print(f"  native arguments: {json.dumps(row['arguments']['native'], ensure_ascii=False)}")
        print(f"  managed arguments: {json.dumps(row['arguments']['managed'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
