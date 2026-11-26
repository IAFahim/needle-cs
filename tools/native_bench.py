#!/usr/bin/env python3
import argparse
import ctypes
import datetime
import glob
import json
import os
import platform
import statistics
import subprocess
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

LIBRARY_NAMES = ("libneedle3.so", "libcactus.so", "libcactus.dylib")

def locate_library(source, extract_dir):
    if os.path.isfile(source) and (source.endswith(".so") or source.endswith(".dylib")):
        return source
    if os.path.isdir(source):
        for name in LIBRARY_NAMES:
            if os.path.isfile(os.path.join(source, name)):
                return os.path.join(source, name)
    wheels = [source] if source.endswith(".whl") else sorted(glob.glob(os.path.join(source, "*.whl")))
    if not wheels:
        sys.exit(f"no .whl or engine library found in {source}")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(wheels[-1]) as archive:
        member = next((n for n in archive.namelist() if n.split("/")[-1] in LIBRARY_NAMES), None)
        if member is None:
            sys.exit(f"no engine library found in {wheels[-1]}")
        target = os.path.join(extract_dir, member.split("/")[-1])
        with archive.open(member) as source_file, open(target, "wb") as target_file:
            target_file.write(source_file.read())
    return target

class Engine:
    def __init__(self, path):
        lib = ctypes.CDLL(os.path.abspath(path))
        self.prefix = "cactus_" if hasattr(lib, "cactus_complete") else "needle_"
        self.abi = "cactus 4.x" if self.prefix == "cactus_" else "needle 3.x"
        text, integer = ctypes.c_char_p, ctypes.c_int
        signatures = {
            "load": ([text, ctypes.c_uint64], integer),
            "init": ([text, text, text], integer),
            "complete": ([text, ctypes.POINTER(ctypes.c_float), integer, integer, text, integer], integer),
            "reset": ([], None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(lib, self.prefix + name)
            function.argtypes, function.restype = arguments, result
            setattr(self, name, function)
        self.set_threads = None
        if hasattr(lib, self.prefix + "set_threads"):
            self.set_threads = getattr(lib, self.prefix + "set_threads")
            self.set_threads.argtypes, self.set_threads.restype = [integer], None

def bind(path):
    return Engine(path)

def complete(lib, buffer, query, max_tokens):
    lib.reset()
    started = time.perf_counter()
    status = lib.complete(query.encode(), None, 0, max_tokens, buffer, len(buffer))
    wall_ms = (time.perf_counter() - started) * 1000.0
    if status < 0:
        return None, wall_ms
    return json.loads(buffer.value.decode()), wall_ms

def calls_json(result):
    calls = [{"name": c.get("name"), "arguments": c.get("arguments") or {}} for c in result.get("function_calls") or []]
    return json.dumps(calls, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def spread(values, higher_is_better=False):
    if not values:
        return {"best": None, "median": None}
    return {"best": max(values) if higher_is_better else min(values), "median": statistics.median(values)}

def measure_set(lib, name, tools_path, queries, reps, warmups, max_tokens, system):
    with open(tools_path, "rb") as handle:
        tools = handle.read()
    prefix_tokens = lib.init(system.encode("utf-8"), tools, None)
    buffer = ctypes.create_string_buffer(OUTPUT_CAPACITY)
    prompts = [{"query": q, "calls": "", "calls_json": None, "failures": 0, "errors": [], "prefill_tokens": None,
                "decode_tokens": None, "token_counts": set(), "samples": {m: [] for m in SAMPLED}} for q in queries]
    for _ in range(warmups):
        for query in queries:
            complete(lib, buffer, query, max_tokens)
    for _ in range(reps):
        for prompt in prompts:
            result, wall_ms = complete(lib, buffer, prompt["query"], max_tokens)
            if result is None or result.get("success", True) is False:
                prompt["failures"] += 1
                prompt["errors"].append("status < 0" if result is None else str(result.get("error")))
                continue
            calls = [c["name"] for c in result.get("function_calls", [])]
            prompt["calls"] = ",".join(calls) if calls else "respond"
            prompt["calls_json"] = calls_json(result)
            prompt["prefill_tokens"] = result.get("prefill_tokens")
            prompt["decode_tokens"] = result.get("decode_tokens")
            prompt["token_counts"].add(result.get("decode_tokens"))
            decode_tps = result.get("decode_tps") or 0.0
            prompt["samples"]["prefill_tps"].append(result.get("prefill_tps") or 0.0)
            prompt["samples"]["decode_tps"].append(decode_tps)
            if decode_tps > 0:
                prompt["samples"]["decode_us_per_token"].append(1e6 / decode_tps)
            prompt["samples"]["total_ms"].append(wall_ms)
    entries = []
    for prompt in prompts:
        entry = {"query": prompt["query"], "calls": prompt["calls"], "calls_json": prompt["calls_json"],
                 "failures": prompt["failures"], "errors": prompt["errors"][:3],
                 "prefill_tokens": prompt["prefill_tokens"], "decode_tokens": prompt["decode_tokens"],
                 "decode_tokens_varied": len(prompt["token_counts"]) > 1, "prefill_ms": None, "ttft_ms": None,
                 "managed_bytes": None, "native_allocs": None}
        for metric in SAMPLED:
            entry[metric] = spread(prompt["samples"][metric], metric in HIGHER_IS_BETTER)
        entries.append(entry)
    summary = {"failed_prompts": [e["query"] for e in entries if e["failures"]]}
    for metric in SAMPLED:
        rows = [e[metric] for e in entries if e[metric]["best"] is not None]
        summary[metric] = {
            "best": statistics.fmean(r["best"] for r in rows) if rows else None,
            "median": statistics.fmean(r["median"] for r in rows) if rows else None,
            "prompts": len(rows),
        }
    return {"name": name, "tools": len(json.loads(tools)), "prefix_tokens": prefix_tokens, "system": system,
            "prompts": entries, "summary": summary}

def render(report):
    lines = [f"native bench {report['library']} | reps {report['reps']} | total ms is wall time around needle_complete"]
    for run in report["runs"]:
        for group in run["sets"]:
            lines.append("")
            threads = "library default" if run["threads"] is None else run["threads"]
            lines.append(f"{group['name']} set | {group['tools']} tools | prefix {group['prefix_tokens']} tokens | threads: {threads}")
            lines.append(f"{'query':<48}{'ptok':>6}{'dtok':>6}{'prefill tok/s':>20}{'decode tok/s':>20}{'decode us/tok':>20}{'total ms':>20}  calls")
            lines.append(f"{'':<60}{'best':>10}{'med':>10}{'best':>10}{'med':>10}{'best':>10}{'med':>10}{'best':>10}{'med':>10}")
            for p in group["prompts"]:
                tokens = f"{str(p['prefill_tokens']):>6}{str(p['decode_tokens']) + ('~' if p.get('decode_tokens_varied') else ''):>6}"
                note = f"  FAILED {p['failures']}x: {'; '.join(p.get('errors') or [])}" if p["failures"] else ""
                lines.append(row(p["query"], p, tokens) + "  " + (p.get("calls_json") or "") + note)
            failed = group["summary"].get("failed_prompts") or []
            lines.append(row("mean of prompts" + (f" (excl. {len(failed)} failed)" if failed else ""), group["summary"], " " * 12))
    return "\n".join(lines)

def row(label, metrics, tokens=""):
    cells = []
    for metric in ("prefill_tps", "decode_tps", "decode_us_per_token", "total_ms"):
        for key in ("best", "median"):
            value = metrics[metric][key]
            cells.append(f"{float('nan') if value is None else value:>10.1f}")
    return f"{label[:47]:<48}" + tokens + "".join(cells)

def date_fact():
    return "date: " + datetime.datetime.now().strftime("%Y-%m-%d %a %H:%M")

def run_child(options, threads):
    """One process per thread count: upstream's Python calls set_threads before load, so we do too."""
    command = [sys.executable, os.path.abspath(__file__), options.library, options.model, options.tools,
               "--game-tools", options.game_tools, "--sets", options.sets, "--reps", str(options.reps),
               "--warmup", str(options.warmup), "--max-tokens", str(options.max_tokens),
               "--extract-dir", options.extract_dir, "--json", "--threads", str(threads), "--system", options.system]
    if options.threads_after_load:
        command.append("--threads-after-load")
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stderr[-4000:], file=sys.stderr)
        sys.exit(f"threads {threads}: child failed")
    return json.loads(completed.stdout)

def main():
    parser = argparse.ArgumentParser(description="Run an upstream engine (libneedle3.so 3.x or libcactus 4.x) through ctypes on the bench-e2e prompt sets")
    parser.add_argument("library", help="wheel directory, .whl file, engine directory, libneedle3.so or libcactus.so")
    parser.add_argument("model", help="needle3.cact")
    parser.add_argument("tools", help="tools.json used for the golden prompts")
    parser.add_argument("--game-tools", default=os.path.join(HERE, "game_tools.json"))
    parser.add_argument("--sets", choices=("golden", "game", "all"), default="all")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--extract-dir", default=os.path.join(HERE, "..", "ref", "native"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--threads", type=int, nargs="*", default=None,
                        help="thread counts to sweep, one process each, set before load (needs set_threads; 0 is the engine's automatic count)")
    parser.add_argument("--threads-after-load", action="store_true",
                        help="diagnostic: call set_threads after load in one process, the way this script used to")
    parser.add_argument("--system", default="", help="system text for init; 'auto-date' is the package default (date fact only)")
    options = parser.parse_args()
    if options.system == "auto-date":
        options.system = date_fact()

    library = locate_library(options.library, options.extract_dir)
    machine = {"os": platform.platform(), "arch": platform.machine(), "logical_cores": os.cpu_count(), "runtime": "python " + platform.python_version()}
    if options.threads and len(options.threads) > 1 and not options.threads_after_load:
        children = [run_child(options, threads) for threads in options.threads]
        report = dict(children[0])
        report["runs"] = [run for child in children for run in child["runs"]]
        report["process_per_thread_count"] = True
        print(json.dumps(report, indent=2) if options.json else render(report))
        return

    lib = bind(library)
    sweep = options.threads or [None]
    if not options.threads_after_load and sweep[0] is not None:
        if lib.set_threads is None:
            sys.exit(f"{lib.abi} has no set_threads")
        lib.set_threads(sweep[0])
    with open(options.model, "rb") as handle:
        weights = handle.read()
    if lib.load(weights, len(weights)) < 0:
        sys.exit("engine load failed")

    groups = []
    if options.sets in ("golden", "all"):
        groups.append(("golden", options.tools, GOLDEN_QUERIES))
    if options.sets in ("game", "all"):
        groups.append(("game", options.game_tools, GAME_QUERIES))
    runs = []
    for threads in sweep:
        if threads is not None and options.threads_after_load:
            if lib.set_threads is None:
                sys.exit(f"{lib.abi} has no set_threads")
            lib.set_threads(threads)
        sets = [measure_set(lib, name, path, queries, options.reps, options.warmup, options.max_tokens, options.system) for name, path, queries in groups]
        runs.append({"threads": threads, "threads_set": None if threads is None else ("after load" if options.threads_after_load else "before load"), "sets": sets})

    report = {
        "schema": "needle.bench-e2e/1",
        "engine": "native",
        "library": os.path.basename(library),
        "abi": lib.abi,
        "model": options.model,
        "reps": options.reps,
        "machine": machine,
        "runs": runs,
    }
    print(json.dumps(report, indent=2) if options.json else render(report))

if __name__ == "__main__":
    main()
