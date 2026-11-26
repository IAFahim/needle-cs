#!/usr/bin/env python3
"""Puts needle.bench-e2e/1 reports side by side as a Markdown table.

    python3 tools/unity-bench-compare.py --native out/unity/bench-native.json \
        editor=out/unity/bench-editmode.json il2cpp=out/unity/bench-player-il2cpp.json

Every report must come from the same machine. For each prompt set and thread count it prints the
mean over prompts of the median total time, time to first token and decode speed, and how many
times faster than the native library each run is (native total ms / this run's total ms). It only
reports; it never fails on speed, because a shared or busy runner makes a single run noisy.
"""
import argparse
import json
import pathlib
import sys

METRICS = [("total_ms", "total ms", False), ("ttft_ms", "first token ms", False), ("decode_tps", "decode tok/s", True)]


def load(path):
    report = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if report.get("schema") != "needle.bench-e2e/1":
        sys.exit(f"{path}: not a needle.bench-e2e/1 report")
    return report


def median(summary, metric):
    value = (summary.get(metric) or {}).get("median")
    return None if value is None else float(value)


def failures(group):
    return sum(int(p.get("failures") or 0) for p in group.get("prompts", []))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--native", help="report from tools/native_bench.py --json")
    parser.add_argument("--gamepath", help="needle.unity-gamepath/1 report from the PlayMode game-path bench")
    parser.add_argument("runs", nargs="+", help="label=path of a needle.bench-e2e/1 report")
    options = parser.parse_args()

    native = {}
    lines = []
    if options.native and pathlib.Path(options.native).is_file():
        report = load(options.native)
        for run in report["runs"]:
            for group in run["sets"]:
                native[group["name"]] = group["summary"]
        lines.append(f"Native baseline: {report.get('library', 'native')} (its own default threads).")
    elif options.native:
        lines.append(f"No native baseline at {options.native}; speedups are left out.")

    machine = None
    lines += ["", "| run | set | threads | total ms | first token ms | decode tok/s | vs native | failures |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, summary in native.items():
        cells = [f"{median(summary, m):.1f}" if median(summary, m) is not None else "n/a" for m, _, _ in METRICS]
        lines.append(f"| native | {name} | default | {' | '.join(cells)} | 1.00x | |")

    for item in options.runs:
        label, _, path = item.partition("=")
        if not path:
            label, path = pathlib.Path(item).stem, item
        if not pathlib.Path(path).is_file():
            lines.append(f"| {label} | missing | | | | | | |")
            continue
        report = load(path)
        machine = machine or report.get("machine")
        for run in report["runs"]:
            for group in run["sets"]:
                summary = group["summary"]
                cells = [f"{median(summary, m):.1f}" if median(summary, m) is not None else "n/a" for m, _, _ in METRICS]
                ours, base = median(summary, "total_ms"), median(native.get(group["name"], {}), "total_ms")
                speedup = f"{base / ours:.2f}x" if ours and base else "n/a"
                lines.append(f"| {label} | {group['name']} | {run.get('threads') or 'default'} | {' | '.join(cells)} | {speedup} | {failures(group)} |")

    if machine:
        lines += ["", f"Machine: {machine.get('os')}, {machine.get('arch')}, {machine.get('logical_cores')} logical cores, "
                      f"SIMD {machine.get('simd', 'n/a')}, runtime {machine.get('runtime')}."]
    if options.gamepath:
        lines += [""] + gamepath(options.gamepath)
    print("\n".join(lines))


def gamepath(path):
    """The game path in the Editor against the Unity targets of the 4.x plan."""
    if not pathlib.Path(path).is_file():
        return [f"No game-path report at {path}."]
    report = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if report.get("schema") != "needle.unity-gamepath/1":
        return [f"{path}: not a needle.unity-gamepath/1 report."]

    def p(name, key="p50"):
        value = (report.get(name) or {}).get(key)
        return "n/a" if value is None else f"{value:g}"

    targets = report.get("targets", {})

    def met(name):
        return "met" if (targets.get(name) or {}).get("met") else "not met"

    return [
        f"Game path in the Editor, {report.get('asks')} asks at {report.get('threads')} threads, kernels: {report.get('kernels')}.",
        "",
        "| measure | p50 | p99 | max | target |",
        "|---|---:|---:|---:|---|",
        f"| main-thread us per ask | {p('main_thread_us_per_ask')} | {p('main_thread_us_per_ask', 'p99')} | {p('main_thread_us_per_ask', 'max')} | p99 at most 20 us: {met('main_thread_us_per_ask_p99')} |",
        f"| main-thread managed bytes | {report.get('main_thread_alloc_bytes')} | | | zero: {met('zero_main_thread_alloc')} |",
        f"| frame work idle, ms | {p('frame_work_idle_ms')} | {p('frame_work_idle_ms', 'p99')} | {p('frame_work_idle_ms', 'max')} | |",
        f"| frame work while inferring, ms | {p('frame_work_busy_ms')} | {p('frame_work_busy_ms', 'p99')} | {p('frame_work_busy_ms', 'max')} | p99 within 1.1x of idle: {met('frame_work_p99_busy_over_idle')} |",
        f"| engine ms per ask | {p('engine_ms')} | {p('engine_ms', 'p99')} | {p('engine_ms', 'max')} | |",
        f"| frames per ask | {p('frames_per_ask')} | {p('frames_per_ask', 'p99')} | {p('frames_per_ask', 'max')} | |",
    ]


if __name__ == "__main__":
    main()
