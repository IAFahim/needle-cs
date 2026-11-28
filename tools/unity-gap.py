#!/usr/bin/env python3
"""Splits each needle.bench-e2e/1 report's time against the native baseline into its parts.

    python3 tools/unity-gap.py --native out/gap/native.json label=report.json ...

For every report, prompt set and thread count it prints the mean over prompts (medians of the
timed repetitions) of: prompt tokens, decoded tokens, prefill ms, decode us per token, decode ms,
the rest of the call (total minus prefill minus decode: sampling, grammar, parsing, the call into
the engine), and the total. Then it splits the speedup over native, native total / this total,
into what prefill, decode per token, the number of decoded tokens and the rest each contribute,
so a 2x can be read as, say, "decode per token 1.6x, fewer tokens 1.2x".

Only reports, never fails: a busy machine makes single runs noisy.
"""
import argparse
import json
import math
import pathlib
import statistics


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def mid(value):
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("median")
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def prompt_parts(prompt):
    total = mid(prompt.get("total_ms"))
    if total is None:
        return None
    prefill_tokens = prompt.get("prefill_tokens") or 0
    decode_tokens = prompt.get("decode_tokens") or 0
    prefill_ms = mid(prompt.get("prefill_ms"))
    if prefill_ms is None:
        tps = mid(prompt.get("prefill_tps"))
        prefill_ms = prefill_tokens / tps * 1000.0 if tps and prefill_tokens else None
    us_per_token = mid(prompt.get("decode_us_per_token"))
    if us_per_token is None:
        tps = mid(prompt.get("decode_tps"))
        us_per_token = 1e6 / tps if tps else None
    decode_ms = us_per_token * decode_tokens / 1000.0 if us_per_token is not None else None
    rest = None
    if prefill_ms is not None and decode_ms is not None:
        rest = total - prefill_ms - decode_ms
    return {"prefill_tokens": prefill_tokens, "decode_tokens": decode_tokens, "prefill_ms": prefill_ms,
            "us_per_token": us_per_token, "decode_ms": decode_ms, "rest_ms": rest, "total_ms": total}


def mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.fmean(values) if values else None


def sets_of(report):
    for run in report.get("runs", []):
        threads = run.get("threads")
        label = "default" if threads is None else ("auto" if threads == 0 else str(threads))
        for group in run.get("sets", []):
            rows = [p for p in (prompt_parts(x) for x in group.get("prompts", [])) if p]
            if rows:
                yield label, group["name"], {k: mean(rows, k) for k in rows[0]}


def fmt(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def ratio(a, b):
    return f"{a / b:.2f}x" if a and b else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--native", required=True)
    parser.add_argument("runs", nargs="+", help="label=path")
    options = parser.parse_args()

    native = {name: parts for _, name, parts in sets_of(load(options.native))}
    lines = ["| run | threads | set | prompt tok | decoded tok | prefill ms | decode us/tok | decode ms | rest ms | total ms | vs native | prefill x | per-token x | token-count x |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, parts in native.items():
        lines.append(f"| native | default | {name} | {fmt(parts['prefill_tokens'], 0)} | {fmt(parts['decode_tokens'], 1)} | {fmt(parts['prefill_ms'])} | "
                     f"{fmt(parts['us_per_token'], 0)} | {fmt(parts['decode_ms'])} | {fmt(parts['rest_ms'])} | {fmt(parts['total_ms'])} | 1.00x | | | |")
    for item in options.runs:
        label, _, path = item.partition("=")
        if not pathlib.Path(path).is_file():
            lines.append(f"| {label} | missing | | | | | | | | | | | | |")
            continue
        for threads, name, parts in sets_of(load(path)):
            base = native.get(name)
            if not base:
                continue
            lines.append(
                f"| {label} | {threads} | {name} | {fmt(parts['prefill_tokens'], 0)} | {fmt(parts['decode_tokens'], 1)} | {fmt(parts['prefill_ms'])} | "
                f"{fmt(parts['us_per_token'], 0)} | {fmt(parts['decode_ms'])} | {fmt(parts['rest_ms'])} | {fmt(parts['total_ms'])} | "
                f"{ratio(base['total_ms'], parts['total_ms'])} | {ratio(base['prefill_ms'], parts['prefill_ms'])} | "
                f"{ratio(base['us_per_token'], parts['us_per_token'])} | {ratio(base['decode_tokens'], parts['decode_tokens'])} |")
    lines += ["", "prefill x, per-token x and token-count x are native over this run; total ms = prefill + decode + rest."]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
