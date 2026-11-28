#!/usr/bin/env python3
"""Interleave several bench-e2e JSON producers and print an honest markdown comparison.

Each --engine command prints needle.bench-e2e/1 JSON (needle bench-e2e --json or
tools/native_bench.py --json). Rounds alternate the engine order. For every
(threads, set, prompt) the per-round best is kept and the best over rounds is
reported. Means are taken over the prompts every engine completed; failed
prompts are listed, never dropped silently. Decoded token counts and the
returned calls (normalized JSON) are compared across engines and every
mismatch is listed, and a matched-work table restricts the timing to prompts
where every engine decoded the same number of tokens and returned the same calls.
"""
import argparse
import json
import statistics
import subprocess
import sys

METRICS = ("total_ms", "decode_us_per_token", "prefill_tps")
HIGHER_IS_BETTER = ("prefill_tps",)
THREAD_ORDER = {"auto": 1000, "lib default": 1001}


def run_once(command):
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stderr[-4000:], file=sys.stderr)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"failed: {' '.join(command)} (exit {completed.returncode})")


def thread_label(threads):
    if threads is None:
        return "lib default"
    if threads == 0:
        return "auto"
    return str(threads)


def thread_key(label):
    return THREAD_ORDER.get(label, int(label) if label.isdigit() else 999)


def fold(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: fold(v) for k, v in value.items()}
    if isinstance(value, list):
        return [fold(v) for v in value]
    return value


def normalize_calls(text):
    if text is None or text == "":
        return None
    try:
        calls = json.loads(text)
    except json.JSONDecodeError:
        return "unparseable: " + text
    calls = [{"name": c.get("name"), "arguments": c.get("arguments") or {}} for c in calls]
    return json.dumps(sorted((fold(c) for c in calls), key=lambda c: json.dumps(c, sort_keys=True)), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def best_of(values, metric):
    if not values:
        return None
    return max(values) if metric in HIGHER_IS_BETTER else min(values)


def collect(samples, name, document, single_row=None):
    for run in document["runs"]:
        label = single_row or thread_label(run["threads"])
        for group in run["sets"]:
            for prompt in group["prompts"]:
                key = (label, group["name"], prompt["query"])
                entry = samples[name].setdefault(key, {"metrics": {m: [] for m in METRICS}, "failures": 0, "errors": [],
                                                       "decode_tokens": set(), "prefill_tokens": set(), "calls": set(), "varied": False})
                entry["failures"] += prompt.get("failures", 0)
                errors = prompt.get("errors") or []
                entry["errors"].extend([errors] if isinstance(errors, str) else [e for e in errors if e])
                if prompt.get("decode_tokens_varied"):
                    entry["varied"] = True
                ran = (prompt.get("total_ms") or {}).get("best") is not None
                for metric in METRICS:
                    value = (prompt.get(metric) or {}).get("best")
                    if value is not None:
                        entry["metrics"][metric].append(value)
                if ran and prompt.get("decode_tokens") is not None:
                    entry["decode_tokens"].add(prompt["decode_tokens"])
                if ran and prompt.get("prefill_tokens") is not None:
                    entry["prefill_tokens"].add(prompt["prefill_tokens"])
                calls = normalize_calls(prompt.get("calls_json")) if ran else None
                if calls is not None:
                    entry["calls"].add(calls)


def fmt(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def ratio(base, value, metric):
    if base is None or value is None or base == 0 or value == 0:
        return "n/a"
    return f"{(value / base if metric in HIGHER_IS_BETTER else base / value):.2f}x"


def one(values):
    return next(iter(values)) if len(values) == 1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", action="append", nargs=2, metavar=("NAME", "COMMAND"), required=True,
                        help="engine name and a command line printing bench JSON; repeat for each engine, the first is the baseline")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--title", default="")
    parser.add_argument("--detail-threads", default="1", help="thread row for the per-prompt table")
    parser.add_argument("--single-row", help="label every engine's (single) thread configuration with this row name, "
                        "for product-default comparisons where the engines' defaults differ")
    parser.add_argument("--json", help="write the raw samples here")
    options = parser.parse_args()
    if options.single_row:
        options.detail_threads = options.single_row

    engines = [(name, command.split()) for name, command in options.engine]
    names = [name for name, _ in engines]
    samples = {name: {} for name in names}
    for round_index in range(options.rounds):
        order = engines if round_index % 2 == 0 else list(reversed(engines))
        for name, command in order:
            print(f"round {round_index + 1}/{options.rounds}: {name}", file=sys.stderr)
            collect(samples, name, run_once(command), options.single_row)

    first_seen = {}
    for n in names:
        for k in samples[n]:
            first_seen.setdefault(k, len(first_seen))
    keys = sorted(first_seen, key=lambda k: (thread_key(k[0]), k[1], first_seen[k]))
    rows = sorted({(k[0], k[1]) for k in keys}, key=lambda r: (thread_key(r[0]), r[1]))
    queries = {}
    for label, set_name, query in keys:
        bucket = queries.setdefault((label, set_name), [])
        if query not in bucket:
            bucket.append(query)

    def entry(name, label, set_name, query):
        return samples[name].get((label, set_name, query))

    def completed(name, label, set_name, query):
        e = entry(name, label, set_name, query)
        return e is not None and bool(e["metrics"]["total_ms"])

    def present(name, label, set_name):
        return any(entry(name, label, set_name, q) is not None for q in queries[(label, set_name)])

    header = "| threads | set | prompts | " + " | ".join(names) + " | " + " | ".join(f"{n} vs {names[0]}" for n in names[1:]) + " |"
    rule = "|---|---|---|" + "---|" * (2 * len(names) - 1)
    lines = []
    if options.title:
        lines += [f"## {options.title}", ""]
    lines.append(f"Best of {options.rounds} interleaved rounds per prompt; means over the prompts that every engine present on the row completed "
                 "(column `prompts`). Speedup is the baseline's time over each engine's time. `n/a`: the engine has no such thread row "
                 "(needle-cs has no automatic thread count yet; `lib default` is an engine run without set_threads).")
    lines.append("")

    matched_rows = []
    for metric in METRICS:
        lines += [f"### {metric}", "", header, rule]
        for label, set_name in rows:
            engines_here = [n for n in names if present(n, label, set_name)]
            common = [q for q in queries[(label, set_name)] if all(completed(n, label, set_name, q) for n in engines_here)]
            values = []
            for n in names:
                per_prompt = [best_of(entry(n, label, set_name, q)["metrics"][metric], metric) for q in common] if n in engines_here else []
                values.append(statistics.fmean(per_prompt) if common and per_prompt and None not in per_prompt else None)
            lines.append(f"| {label} | {set_name} | {len(common)}/{len(queries[(label, set_name)])} | " + " | ".join(fmt(v) for v in values)
                         + " | " + " | ".join(ratio(values[0], v, metric) for v in values[1:]) + " |")
            if metric == "total_ms" and len(engines_here) > 1:
                def same(field, q):
                    found = {one(entry(n, label, set_name, q)[field]) for n in engines_here}
                    return len(found) == 1 and None not in found
                matched = [q for q in common if same("decode_tokens", q) and same("calls", q)]
                matched_values = [statistics.fmean(best_of(entry(n, label, set_name, q)["metrics"]["total_ms"], "total_ms") for q in matched)
                                  if n in engines_here and matched else None for n in names]
                matched_rows.append(f"| {label} | {set_name} | {len(matched)}/{len(queries[(label, set_name)])} | " + " | ".join(fmt(v) for v in matched_values)
                                    + " | " + " | ".join(ratio(matched_values[0], v, "total_ms") for v in matched_values[1:]) + " |")
        lines.append("")

    lines += ["### total_ms over matched work", "",
              "Only prompts where every engine on the row decoded the same number of tokens and returned the same calls.", "", header, rule]
    lines += matched_rows or ["| (no row with two engines) |"]
    lines.append("")

    detail = options.detail_threads
    lines += [f"### Per prompt at threads {detail}: decoded tokens, best total ms", "",
              "| set | query | " + " | ".join(f"{n} dtok | {n} ms" for n in names) + " | same tokens | same calls |",
              "|---|---|" + "---|---|" * len(names) + "---|---|"]
    for label, set_name in rows:
        if label != detail:
            continue
        for q in queries[(label, set_name)]:
            cells, tokens, calls = [], set(), set()
            for n in names:
                e = entry(n, label, set_name, q)
                if e is None:
                    cells.append("n/a | n/a")
                    continue
                dtok = "/".join(str(t) for t in sorted(e["decode_tokens"])) or "-"
                if e["varied"]:
                    dtok += "~"
                ms = best_of(e["metrics"]["total_ms"], "total_ms")
                cells.append(f"{dtok} | {fmt(ms) if ms is not None else 'FAILED'}")
                tokens.add(tuple(sorted(e["decode_tokens"])))
                calls.add(tuple(sorted(e["calls"])))
            lines.append(f"| {set_name} | {q} | " + " | ".join(cells) + f" | {'yes' if len(tokens) == 1 else 'NO'} | {'yes' if len(calls) == 1 else 'NO'} |")
    lines.append("")

    mismatches, failures = [], []
    for label, set_name, q in keys:
        present_names = [n for n in names if entry(n, label, set_name, q) is not None]
        ran = [n for n in present_names if completed(n, label, set_name, q)]
        calls = {n: entry(n, label, set_name, q)["calls"] for n in ran}
        tokens = {n: entry(n, label, set_name, q)["decode_tokens"] for n in ran}
        if len({tuple(sorted(v)) for v in calls.values()}) > 1 or any(len(v) > 1 for v in calls.values()):
            mismatches.append(f"- calls differ, threads {label} {set_name} \"{q}\": " + "; ".join(f"{n} `{' | '.join(sorted(v)) or 'none'}`" for n, v in calls.items()))
        if len({tuple(sorted(v)) for v in tokens.values()}) > 1:
            mismatches.append(f"- decoded tokens differ, threads {label} {set_name} \"{q}\": " + ", ".join(f"{n} {'/'.join(map(str, sorted(v))) or '-'}" for n, v in tokens.items()))
        for n in present_names:
            e = entry(n, label, set_name, q)
            if e["failures"]:
                failures.append(f"- {n} threads {label} {set_name} \"{q}\": {e['failures']} failed calls"
                                + (f" ({'; '.join(sorted(set(e['errors']))[:3])})" if e["errors"] else ""))
    lines += [f"### Mismatches between engines: {len(mismatches)}", ""] + (mismatches or ["none"]) + [""]
    lines += [f"### Failed prompts: {len(failures)}", ""] + (failures or ["none"]) + [""]
    print("\n".join(lines))

    if options.json:
        with open(options.json, "w") as handle:
            json.dump({n: {"|".join(k): {"metrics": v["metrics"], "failures": v["failures"], "decode_tokens": sorted(v["decode_tokens"]),
                                         "prefill_tokens": sorted(v["prefill_tokens"]), "calls": sorted(v["calls"])}
                           for k, v in samples[n].items()} for n in names}, handle, indent=2)


if __name__ == "__main__":
    main()
