#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import sys

METRICS = ("total_ms", "decode_us_per_token", "prefill_tps")
HIGHER_IS_BETTER = ("prefill_tps",)

def run_once(command):
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"failed: {' '.join(command)}")
    return json.loads(completed.stdout)

def summaries(document):
    for run in document["runs"]:
        for group in run["sets"]:
            for metric in METRICS:
                summary = group["summary"].get(metric)
                if summary and summary.get("best") is not None:
                    yield (run["threads"], group["name"], metric), summary["best"]

def main():
    parser = argparse.ArgumentParser(description="Interleave several bench-e2e JSON producers and print a markdown comparison")
    parser.add_argument("--engine", action="append", nargs=2, metavar=("NAME", "COMMAND"), required=True,
                        help="engine name and a command line printing bench JSON; repeat for each engine, the first is the baseline")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--json", help="write the raw samples here")
    options = parser.parse_args()

    engines = [(name, command.split()) for name, command in options.engine]
    samples = {name: {} for name, _ in engines}
    for round_index in range(options.rounds):
        order = engines if round_index % 2 == 0 else list(reversed(engines))
        for name, command in order:
            print(f"round {round_index + 1}/{options.rounds}: {name}", file=sys.stderr)
            for key, value in summaries(run_once(command)):
                samples[name].setdefault(key, []).append(value)

    def best(name, key):
        values = samples[name].get(key)
        if not values:
            return None
        return max(values) if key[2] in HIGHER_IS_BETTER else min(values)

    names = [name for name, _ in engines]
    lines = [f"Best of {options.rounds} interleaved rounds, mean over each set's prompts. Speedup is the baseline's time over each engine's time.", ""]
    for metric in METRICS:
        lines.append(f"### {metric}")
        lines.append("")
        lines.append("| threads | set | " + " | ".join(names) + " | " + " | ".join(f"{n} speedup vs {names[0]}" for n in names[1:]) + " |")
        lines.append("|---|---|" + "---|" * (2 * len(names) - 1))
        keys = sorted({(k[0] if k[0] is not None else 0, k[1]) for n in names for k in samples[n] if k[2] == metric})
        keys = [k for k in keys if k[0] != 0] + [k for k in keys if k[0] == 0]
        for threads, set_name in keys:
            row = []
            values = []
            for n in names:
                v = (best(n, (0, set_name, metric)) or best(n, (None, set_name, metric))) if threads == 0 else best(n, (threads, set_name, metric))
                values.append(v)
                row.append("n/a" if v is None else f"{v:.1f}")
            ratios = []
            for v in values[1:]:
                base = values[0]
                if v is None or base is None or v == 0 or base == 0:
                    ratios.append("n/a")
                else:
                    ratio = v / base if metric in HIGHER_IS_BETTER else base / v
                    ratios.append(f"{ratio:.2f}x")
            lines.append(f"| {threads or 'default'} | {set_name} | " + " | ".join(row) + " | " + " | ".join(ratios) + " |")
        lines.append("")
    print("\n".join(lines))
    if options.json:
        with open(options.json, "w") as handle:
            json.dump({n: {"|".join(map(str, k)): v for k, v in samples[n].items()} for n in names}, handle, indent=2)

if __name__ == "__main__":
    main()
