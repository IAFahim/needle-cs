#!/usr/bin/env python3
import argparse
import ctypes
import glob
import json
import os
import platform
import statistics
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARY_NAME = "libneedle3.so"
OUTPUT_CAPACITY = 1 << 16

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

SAMPLED = ("prefill_tps", "decode_tps", "decode_us_per_token", "total_ms")
HIGHER_IS_BETTER = ("prefill_tps", "decode_tps")


def locate_library(source, extract_dir):
    if os.path.isfile(source) and source.endswith(".so"):
        return source
    wheels = [source] if source.endswith(".whl") else sorted(glob.glob(os.path.join(source, "*.whl")))
    if not wheels:
        sys.exit(f"no .whl found in {source}")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(wheels[-1]) as archive:
        member = next((n for n in archive.namelist() if n.endswith(LIBRARY_NAME)), None)
        if member is None:
            sys.exit(f"{LIBRARY_NAME} not found in {wheels[-1]}")
        target = os.path.join(extract_dir, LIBRARY_NAME)
        with archive.open(member) as source_file, open(target, "wb") as target_file:
            target_file.write(source_file.read())
    return target


def bind(path):
    lib = ctypes.CDLL(os.path.abspath(path))
    lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
    lib.needle_load.restype = ctypes.c_int
    lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.needle_init.restype = ctypes.c_int
    lib.needle_complete.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.needle_complete.restype = ctypes.c_int
    lib.needle_reset.argtypes = []
    lib.needle_reset.restype = None
    return lib


def complete(lib, buffer, query, max_tokens):
    lib.needle_reset()
    started = time.perf_counter()
    status = lib.needle_complete(query.encode(), None, 0, max_tokens, buffer, len(buffer))
    wall_ms = (time.perf_counter() - started) * 1000.0
    if status < 0:
        return None, wall_ms
    return json.loads(buffer.value.decode()), wall_ms


def spread(values, higher_is_better=False):
    if not values:
        return {"best": None, "median": None}
    return {"best": max(values) if higher_is_better else min(values), "median": statistics.median(values)}


def measure_set(lib, name, tools_path, queries, reps, warmups, max_tokens):
    with open(tools_path, "rb") as handle:
        tools = handle.read()
    prefix_tokens = lib.needle_init(b"", tools, None)
    buffer = ctypes.create_string_buffer(OUTPUT_CAPACITY)
    prompts = [{"query": q, "calls": "", "failures": 0, "samples": {m: [] for m in SAMPLED}} for q in queries]
    for _ in range(warmups):
        for query in queries:
            complete(lib, buffer, query, max_tokens)
    for _ in range(reps):
        for prompt in prompts:
            result, wall_ms = complete(lib, buffer, prompt["query"], max_tokens)
            if result is None or not result.get("success", True) or not result.get("prefill_tps") or not result.get("decode_tps"):
                prompt["failures"] += 1
                continue
            calls = [c["name"] for c in result.get("function_calls", [])]
            prompt["calls"] = ",".join(calls) if calls else "respond"
            prompt["samples"]["prefill_tps"].append(result["prefill_tps"])
            prompt["samples"]["decode_tps"].append(result["decode_tps"])
            prompt["samples"]["decode_us_per_token"].append(1e6 / result["decode_tps"])
            prompt["samples"]["total_ms"].append(wall_ms)
    entries = []
    for prompt in prompts:
        entry = {"query": prompt["query"], "calls": prompt["calls"], "failures": prompt["failures"],
                 "prefill_tokens": None, "decode_tokens": None, "prefill_ms": None, "ttft_ms": None,
                 "managed_bytes": None, "native_allocs": None}
        for metric in SAMPLED:
            entry[metric] = spread(prompt["samples"][metric], metric in HIGHER_IS_BETTER)
        entries.append(entry)
    summary = {}
    for metric in SAMPLED:
        rows = [e[metric] for e in entries if e[metric]["best"] is not None]
        summary[metric] = {
            "best": statistics.fmean(r["best"] for r in rows) if rows else None,
            "median": statistics.fmean(r["median"] for r in rows) if rows else None,
        }
    return {"name": name, "tools": len(json.loads(tools)), "prefix_tokens": prefix_tokens, "prompts": entries, "summary": summary}


def render(report):
    lines = [f"native bench {report['library']} | reps {report['reps']} | total ms is wall time around needle_complete"]
    for run in report["runs"]:
        for group in run["sets"]:
            lines.append("")
            lines.append(f"{group['name']} set | {group['tools']} tools | prefix {group['prefix_tokens']} tokens | threads: library default")
            lines.append(f"{'query':<48}{'prefill tok/s':>20}{'decode tok/s':>20}{'decode us/tok':>20}{'total ms':>20}")
            lines.append(f"{'':<48}{'best':>10}{'med':>10}{'best':>10}{'med':>10}{'best':>10}{'med':>10}{'best':>10}{'med':>10}")
            for p in group["prompts"]:
                lines.append(row(p["query"], p))
            lines.append(row("mean of prompts", group["summary"]))
    return "\n".join(lines)


def row(label, metrics):
    cells = []
    for metric in ("prefill_tps", "decode_tps", "decode_us_per_token", "total_ms"):
        for key in ("best", "median"):
            value = metrics[metric][key]
            cells.append(f"{float('nan') if value is None else value:>10.1f}")
    return f"{label[:47]:<48}" + "".join(cells)


def main():
    parser = argparse.ArgumentParser(description="Run upstream libneedle3.so through ctypes on the bench-e2e prompt sets")
    parser.add_argument("library", help="wheel directory, .whl file or libneedle3.so")
    parser.add_argument("model", help="needle3.cact")
    parser.add_argument("tools", help="tools.json used for the golden prompts")
    parser.add_argument("--game-tools", default=os.path.join(HERE, "game_tools.json"))
    parser.add_argument("--sets", choices=("golden", "game", "all"), default="all")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--extract-dir", default=os.path.join(HERE, "..", "ref", "native"))
    parser.add_argument("--json", action="store_true")
    options = parser.parse_args()

    library = locate_library(options.library, options.extract_dir)
    lib = bind(library)
    with open(options.model, "rb") as handle:
        weights = handle.read()
    if lib.needle_load(weights, len(weights)) < 0:
        sys.exit("needle_load failed")

    groups = []
    if options.sets in ("golden", "all"):
        groups.append(("golden", options.tools, GOLDEN_QUERIES))
    if options.sets in ("game", "all"):
        groups.append(("game", options.game_tools, GAME_QUERIES))
    sets = [measure_set(lib, name, path, queries, options.reps, options.warmup, options.max_tokens) for name, path, queries in groups]

    report = {
        "schema": "needle.bench-e2e/1",
        "engine": "native",
        "library": os.path.basename(library),
        "model": options.model,
        "reps": options.reps,
        "machine": {"os": platform.platform(), "arch": platform.machine(), "logical_cores": os.cpu_count(), "runtime": "python " + platform.python_version()},
        "runs": [{"threads": None, "sets": sets}],
    }
    print(json.dumps(report, indent=2) if options.json else render(report))


if __name__ == "__main__":
    main()
