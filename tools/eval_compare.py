#!/usr/bin/env python3
"""Compare `needle eval --suite` and tools/eval_native.py reports on the 4.x acceptance suites.

Prints a markdown per-suite table for every engine, re-scores each report with
the Python rule (tools/eval_rule.py) to cross-check the C# scorer, and lists
every case where the engines' scored calls differ, with a best-guess cause.

usage: python3 tools/eval_compare.py needle-cs.json upstream.json [...] [--out report.md]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_rule  # noqa: E402


def keys(calls):
    return sorted(eval_rule.call_key(c) for c in calls)


def rescore(report):
    mismatches = []
    for suite in report["suites"]:
        for case in suite["cases"]:
            got = [] if case.get("gated") else case["raw_calls"]
            ok = not case.get("engine_failed") and eval_rule.calls_equal(got, case["want"])
            if ok != case["ok"]:
                mismatches.append(f"{report['engine']} {suite['suite']}: \"{case['query']}\" scorer says {case['ok']}, Python rule says {ok}")
    return mismatches


def cause(a, b):
    if a.get("success", True) is False or b.get("success", True) is False:
        return "truncation (success false)"
    if keys(a["raw_calls"]) == keys(b["raw_calls"]):
        if a["negation"] != b["negation"]:
            return "negation"
        if bool(a["ungrounded"]) != bool(b["ungrounded"]):
            return "grounding"
        return "confidence gate"
    if a.get("repaired") or b.get("repaired"):
        return "repair"
    names_a = sorted(c["name"] for c in a["raw_calls"])
    names_b = sorted(c["name"] for c in b["raw_calls"])
    if not names_a or not names_b:
        if a.get("suppressed_calls") or b.get("suppressed_calls"):
            return "confidence (suppressed)"
        return "triggers (call vs no call)"
    if names_a != names_b:
        return "tool choice / call count"
    differing = set()
    for ca, cb in zip(sorted(a["raw_calls"], key=eval_rule.call_key), sorted(b["raw_calls"], key=eval_rule.call_key)):
        args_a, args_b = ca.get("arguments") or {}, cb.get("arguments") or {}
        differing |= {k for k in set(args_a) | set(args_b) if eval_rule.call_key({"v": args_a.get(k)}) != eval_rule.call_key({"v": args_b.get(k)})}
    if any("time" in k or "date" in k or "duration" in k for k in differing):
        return "dates/times"
    return "arguments (" + ", ".join(sorted(differing)) + ")"


def brief(case):
    text = json.dumps(case["got"], ensure_ascii=False, separators=(",", ":"))
    flags = []
    if case.get("gated"):
        flags.append(f"gated:{case['gated']} raw={json.dumps(case['raw_calls'], ensure_ascii=False, separators=(',', ':'))}")
    if case.get("negation"):
        flags.append("negation")
    if case.get("ungrounded"):
        flags.append("ungrounded=" + ",".join(case["ungrounded"]))
    if case.get("repaired"):
        flags.append("repaired")
    if case.get("suppressed_calls"):
        flags.append("suppressed=" + json.dumps(case["suppressed_calls"], ensure_ascii=False, separators=(",", ":")))
    if case.get("confidence") is not None:
        flags.append(f"conf {case['confidence']:.2f}")
    if not case.get("success", True):
        flags.append(f"success false: {case.get('error')}")
    return (text + (" (" + "; ".join(flags) + ")" if flags else "")).replace("|", "\\|")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--out")
    options = parser.parse_args()
    reports = []
    for path in options.reports:
        with open(path, encoding="utf-8") as handle:
            reports.append(json.load(handle))

    lines = ["### 4.x acceptance suites (rule: passed >= round(0.9 n), no failed critical case)", ""]
    lines.append(" · ".join(f"{r['engine']}: threads {r['threads']}, date fact `{r['date']}`, confidence gate {r['min_confidence']}" for r in reports))
    lines.append("")
    lines.append(eval_rule.render_table(reports))
    lines.append("")

    mismatches = [m for r in reports for m in rescore(r)]
    lines.append("Scorer cross-check (every report re-scored with tools/eval_rule.py): " + ("all agree" if not mismatches else f"{len(mismatches)} mismatches"))
    lines.extend("- " + m for m in mismatches)
    lines.append("")

    if len(reports) >= 2:
        a, b = reports[0], reports[1]
        rows, causes = [], {}
        for suite_a in a["suites"]:
            suite_b = next((s for s in b["suites"] if s["suite"] == suite_a["suite"]), None)
            if suite_b is None:
                continue
            for ca, cb in zip(suite_a["cases"], suite_b["cases"]):
                if keys(ca["got"]) == keys(cb["got"]) and ca["ok"] == cb["ok"]:
                    continue
                why = cause(ca, cb)
                causes[why.split(" (")[0]] = causes.get(why.split(" (")[0], 0) + 1
                rows.append(f"| {suite_a['suite']} | {ca['query']}{' (critical)' if ca['critical'] else ''} | {'ok' if ca['ok'] else 'FAIL'} {brief(ca)} | {'ok' if cb['ok'] else 'FAIL'} {brief(cb)} | {why} |")
        lines.append(f"### Cases where {a['engine']} and {b['engine']} disagree: {len(rows)}")
        lines.append("")
        if causes:
            lines.append("By cause: " + ", ".join(f"{k} {v}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])))
            lines.append("")
            lines.append(f"| suite | query | {a['engine']} | {b['engine']} | likely cause |")
            lines.append("|---|---|---|---|---|")
            lines.extend(rows)
        lines.append("")
        tokens = []
        for suite_a in a["suites"]:
            suite_b = next((s for s in b["suites"] if s["suite"] == suite_a["suite"]), None)
            if suite_b is None:
                continue
            def total(s, k, digits=0):
                values = [c.get(k) for c in s["cases"]]
                return "n/a" if any(v is None for v in values) else f"{sum(values):.{digits}f}"
            tokens.append(f"| {suite_a['suite']} | {total(suite_a, 'decode_tokens')} | {total(suite_b, 'decode_tokens')} | {total(suite_a, 'wall_ms')} | {total(suite_b, 'wall_ms')} |")
        lines.append(f"Decoded tokens and wall time per suite (sum over 32 cases; informational, see the speed benchmark for timing; the 4.x envelope reports no token counts)")
        lines.append("")
        lines.append(f"| suite | {a['engine']} decode tok | {b['engine']} decode tok | {a['engine']} ms | {b['engine']} ms |")
        lines.append("|---|---|---|---|---|")
        lines.extend(tokens)

    text = "\n".join(lines)
    print(text)
    if options.out:
        with open(options.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
