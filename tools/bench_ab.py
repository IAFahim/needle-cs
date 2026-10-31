#!/usr/bin/env python3
import argparse
import json
import shlex
import statistics
import subprocess
import sys

METRICS = ("prefill_ms", "decode_us_per_token", "ttft_ms", "total_ms", "prefill_tps", "decode_tps")
HIGHER_IS_BETTER = ("prefill_tps", "decode_tps")


def run_once(command):
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def summaries(document):
    for run in document["runs"]:
        for group in run["sets"]:
            for metric in METRICS:
                summary = group["summary"].get(metric)
                if summary and summary["best"] is not None:
                    yield (run["threads"], group["name"], metric), summary


def collect(commands, rounds):
    samples = {name: {} for name in commands}
    names = list(commands)
    for round_index in range(rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for name in order:
            print(f"round {round_index + 1}/{rounds}: {name}", file=sys.stderr)
            for key, summary in summaries(run_once(commands[name])):
                samples[name].setdefault(key, []).append(summary)
    return samples


def reduce(metric, rounds):
    pick = max if metric in HIGHER_IS_BETTER else min
    best = pick(entry["best"] for entry in rounds)
    median = statistics.median(entry["median"] for entry in rounds)
    return best, median


def find(samples, key):
    return samples.get(key) or samples.get((None, key[1], key[2]))


def percent(b, a):
    return (b / a - 1.0) * 100.0 if a else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Interleaved A/B comparison of two bench-e2e JSON producers")
    parser.add_argument("--a", required=True, help="command line printing bench JSON for variant A (baseline)")
    parser.add_argument("--b", required=True, help="command line printing bench JSON for variant B")
    parser.add_argument("--rounds", type=int, default=5, help="alternating rounds per variant (at least 5 for a claim)")
    parser.add_argument("--json", action="store_true", help="print the comparison as JSON")
    options = parser.parse_args()

    commands = {"A": shlex.split(options.a), "B": shlex.split(options.b)}
    samples = collect(commands, options.rounds)

    rows = []
    for key in samples["A"]:
        a_rounds = samples["A"][key]
        b_rounds = find(samples["B"], key)
        if not b_rounds:
            continue
        a_best, a_median = reduce(key[2], a_rounds)
        b_best, b_median = reduce(key[2], b_rounds)
        rows.append({
            "threads": key[0], "set": key[1], "metric": key[2],
            "a_best": a_best, "a_median": a_median, "b_best": b_best, "b_median": b_median,
            "best_delta_percent": percent(b_best, a_best), "median_delta_percent": percent(b_median, a_median),
        })

    if options.json:
        print(json.dumps({"rounds": options.rounds, "rows": rows}, indent=2))
        return
    print(f"rounds {options.rounds}, order alternates A,B / B,A; best = min over rounds of per-round best, median = median of per-round medians")
    print(f"{'threads':>7} {'set':<8} {'metric':<20} {'A best':>10} {'B best':>10} {'delta':>8} {'A median':>10} {'B median':>10} {'delta':>8}")
    for row in rows:
        print(f"{str(row['threads']):>7} {row['set']:<8} {row['metric']:<20} {row['a_best']:>10.2f} {row['b_best']:>10.2f} {row['best_delta_percent']:>+7.1f}% "
              f"{row['a_median']:>10.2f} {row['b_median']:>10.2f} {row['median_delta_percent']:>+7.1f}%")


if __name__ == "__main__":
    main()
