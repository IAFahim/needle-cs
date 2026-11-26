#!/usr/bin/env python3
"""Score upstream libcactus 4.x on the six 4.x acceptance suites through the cactus_* C ABI.

The suites and the scoring rule come from the Apache-2.0 cactus-compute 4.2.0
package (cactus/environments/_harness.py, cactus/agent/text.py,
cactus/agent/grounding.py); see tests/eval/environments/NOTICE. The driver
mirrors cactus.Needle: set_threads before load, init(system, tools, NULL) with
the auto_date prefix on the system text, then reset + complete per case, and
the same client-side year grounding the package applies to every response.

usage: python3 tools/eval_native.py <libcactus.so|dir> <needle3.cact> [--suite all|name ...] [--threads N]
                                    [--date "YYYY-MM-DD Ddd HH:MM"] [--min-confidence F] [--json out.json]
"""
import argparse
import ctypes
import datetime
import json
import os
import platform
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from native_bench import locate_library  # noqa: E402
import eval_rule  # noqa: E402

OUTPUT_CAPACITY = 65536


def bind(path):
    lib = ctypes.CDLL(os.path.abspath(path))
    if not hasattr(lib, "cactus_complete"):
        sys.exit(f"{path} has no cactus_* ABI (4.x)")
    text, integer = ctypes.c_char_p, ctypes.c_int
    signatures = {
        "load": ([text, ctypes.c_uint64], integer),
        "init": ([text, text, text], integer),
        "complete": ([text, ctypes.POINTER(ctypes.c_float), integer, integer, text, integer], integer),
        "set_threads": ([integer], None),
        "reset": ([], None),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(lib, "cactus_" + name)
        function.argtypes, function.restype = arguments, result
    if hasattr(lib, "cactus_last_error"):
        lib.cactus_last_error.argtypes, lib.cactus_last_error.restype = [], text
    return lib


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("library")
    parser.add_argument("model")
    parser.add_argument("--suite", nargs="+", default=["all"])
    parser.add_argument("--dir", default=os.path.join(HERE, "..", "tests", "eval", "environments"))
    parser.add_argument("--threads", type=int, default=0, help="0 is the engine's automatic count (upstream's default)")
    parser.add_argument("--date", default=None, help="date fact body, e.g. '2026-11-26 Thu 09:30'; default is now")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--extract-dir", default=os.path.join(HERE, "..", "ref", "native4"))
    parser.add_argument("--json", help="write the per-case report here")
    parser.add_argument("--quiet", action="store_true")
    options = parser.parse_args()

    date = options.date or datetime.datetime.now().strftime("%Y-%m-%d %a %H:%M")
    suites = eval_rule.load_suites(options.dir, options.suite)
    library = locate_library(options.library, options.extract_dir)
    lib = bind(library)
    lib.cactus_set_threads(options.threads)
    with open(options.model, "rb") as handle:
        weights = handle.read()
    if lib.cactus_load(weights, len(weights)) < 0:
        sys.exit("cactus_load failed")
    buffer = ctypes.create_string_buffer(OUTPUT_CAPACITY)

    report = {"schema": "needle.eval-4x/1", "engine": "upstream-4.x", "library": os.path.basename(library),
              "model": os.path.basename(options.model), "threads": options.threads, "date": date,
              "min_confidence": options.min_confidence,
              "machine": {"os": platform.platform(), "arch": platform.machine(), "logical_cores": os.cpu_count()},
              "suites": []}
    for suite in suites:
        system = eval_rule.with_date_fact(suite["system"], date)
        if lib.cactus_init(system.encode("utf-8"), suite["tools_bytes"], None) < 0:
            sys.exit(f"cactus_init failed for {suite['suite']}")
        cases = []
        for case in suite["cases"]:
            lib.cactus_reset()
            started = time.perf_counter()
            rc = lib.cactus_complete(case["query"].encode("utf-8"), None, 0, options.max_tokens, buffer, len(buffer))
            wall_ms = (time.perf_counter() - started) * 1000.0
            raw = buffer.value.decode("utf-8", "replace")
            if rc < 0:
                response = {"success": False, "error": raw or f"cactus_complete returned {rc}", "function_calls": []}
            else:
                response = json.loads(raw)
                eval_rule.annotate_ungrounded(response, suite["tools"], eval_rule.source_years(case["query"]), system, case["query"])
            cases.append(eval_rule.score_case(case, response, options.min_confidence, wall_ms))
        result = eval_rule.summarize(suite["suite"], cases)
        report["suites"].append(result)
        if not options.quiet:
            print(eval_rule.render_suite(result), file=sys.stderr)
    report["overall"] = eval_rule.overall(report["suites"])
    print(eval_rule.render_table([report]))
    if options.json:
        with open(options.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, ensure_ascii=False)
    return 0 if report["overall"]["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
