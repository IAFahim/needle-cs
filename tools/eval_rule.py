"""Shared scoring for the 4.x acceptance suites (tests/eval/environments).

The rule and the date/grounding helpers are derived from the Apache-2.0
cactus-compute 4.2.0 package: cactus/environments/_harness.py (run_tests,
_fold, _key) and cactus/agent/grounding.py (_with_date_fact, _source_years,
_relative_cue, _licensed_years, _walk_grounding, _annotate_ungrounded).
`needle eval --suite` implements the same rule in C# (src/Needle.Cli/EvalSuites.cs).
"""
import json
import os
import re

SUITES = ("smart_home", "media_player", "productivity", "wearable", "kitchen_appliance", "data_capture")


def load_suites(directory, names):
    if not names or "all" in names:
        names = SUITES
    suites = []
    for name in names:
        if name not in SUITES:
            raise SystemExit(f"unknown suite {name}; expected one of {', '.join(SUITES)} or all")
        with open(os.path.join(directory, name + ".json"), encoding="utf-8") as handle:
            suite = json.load(handle)
        with open(os.path.join(directory, name + ".tools.json"), "rb") as handle:
            suite["tools_bytes"] = handle.read()
        suite["tools"] = json.loads(suite["tools_bytes"])
        suites.append(suite)
    return suites


_ISO_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATE_KEY = re.compile(r"(?<![A-Za-z])date:")


def with_date_fact(system, date):
    """cactus.agent.grounding._with_date_fact with the clock pinned to `date` ("%Y-%m-%d %a %H:%M")."""
    if _DATE_KEY.search(system) or _ISO_STAMP.search(system):
        return system
    fact = "date: " + date
    if not system.strip():
        return fact
    if system.lstrip().startswith("{"):
        return system
    return fact + "; " + system


_MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
           r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
           r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_YEAR_PATTERNS = [
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}[\s,]+(\d{{1,4}})(?![0-9A-Za-z]|:\d|\s*[ap]\.?m\b)",
    rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s+(\d{{1,4}})(?![0-9A-Za-z]|:\d|\s*[ap]\.?m\b)",
    rf"\b{_MONTHS}[\s,]+(\d{{3,4}})(?![0-9A-Za-z])",
    r"\byear\s+(\d{1,4})(?![0-9A-Za-z])",
    r"(?<![0-9])(\d{4})(?=[-/]\d{1,2}[-/]\d{1,2}(?![0-9]))",
]
_RELATIVE_CUE = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|next|this|coming|now|in \d+ (?:days?|weeks?|months?|years?)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)


def source_years(text):
    lowered = (text or "").lower()
    return {int(m.group(1)) for pattern in _YEAR_PATTERNS for m in re.finditer(pattern, lowered)}


def _licensed_years(seen, system, relative):
    years = set(seen)
    if years and system and relative:
        years |= source_years(system)
    return years


def _resolve_ref(node, root):
    seen = set()
    while isinstance(node, dict) and "$ref" in node:
        if node["$ref"] in seen:
            break
        seen.add(node["$ref"])
        target = root
        for part in node["$ref"].removeprefix("#/").split("/"):
            target = target.get(part.replace("~1", "/").replace("~0", "~"), {})
        if target is node:
            break
        node = target
    return node


def _walk_grounding(schema, arguments, years):
    root = schema.get("parameters", schema)
    failures = set()

    def walk(value, node, path):
        node = _resolve_ref(node, root)
        variants = node.get("anyOf") or node.get("oneOf") or []
        concrete = [v for v in variants if _resolve_ref(v, root).get("type") != "null"]
        if len(concrete) == 1:
            node = _resolve_ref(concrete[0], root)
        if node.get("format") in ("date", "date-time") and isinstance(value, str):
            match = re.match(r"^(\d{4})-", value)
            if match and years and int(match.group(1)) not in years:
                failures.add(path)
            return
        if isinstance(value, dict):
            properties = node.get("properties", {})
            for key, item in value.items():
                if key in properties:
                    walk(item, properties[key], f"{path}.{key}" if path else key)
        elif isinstance(value, list) and "items" in node:
            for index, item in enumerate(value):
                walk(item, node["items"], f"{path}[{index}]")

    walk(arguments, root, "")
    return failures


def annotate_ungrounded(response, tool_schemas, seen_years, system, text):
    calls = response.get("function_calls") or []
    years = _licensed_years(seen_years, system, bool(text) and _RELATIVE_CUE.search(text) is not None)
    if not calls or not years:
        return
    schemas = {entry.get("name"): entry for entry in tool_schemas}
    found = []
    for call in calls:
        schema = schemas.get(call.get("name"))
        if schema is not None:
            found.extend(f"{call['name']}.{path}" for path in sorted(_walk_grounding(schema, call.get("arguments") or {}, years)))
    if found:
        validation = response.setdefault("validation", {}) or {}
        response["validation"] = validation
        existing = list(validation.get("ungrounded") or [])
        validation["ungrounded"] = existing + [n for n in found if n not in existing]


def _fold(value):
    if isinstance(value, str):
        return value.casefold()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _fold(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_fold(v) for v in value]
    return value


def call_key(call):
    return json.dumps(_fold(call), sort_keys=True)


def calls_equal(a, b):
    return sorted(call_key(c) for c in a) == sorted(call_key(c) for c in b)


def _first(response, *names):
    for name in names:
        if response.get(name) is not None:
            return response[name]
    return None


def score_case(case, response, min_confidence, wall_ms):
    raw = response.get("function_calls") or []
    validation = response.get("validation") or {}
    confidence = response.get("confidence")
    got = raw
    gated = None
    if got and (validation.get("ungrounded") or validation.get("negation")):
        got, gated = [], "validation"
    if got and (confidence if confidence is not None else 0.0) < min_confidence:
        got, gated = [], "confidence"
    return {
        "query": case["query"], "category": case["category"], "critical": case["critical"], "want": case["calls"],
        "raw_calls": [{"name": c.get("name"), "arguments": c.get("arguments") or {}} for c in raw],
        "suppressed_calls": response.get("suppressed_calls") or [],
        "got": [{"name": c.get("name"), "arguments": c.get("arguments") or {}} for c in got], "gated": gated,
        "ungrounded": list(validation.get("ungrounded") or []), "negation": bool(validation.get("negation")),
        "repaired": bool(validation.get("repaired")), "confidence": confidence,
        "success": response.get("success", True) is not False, "error": response.get("error"),
        "prefill_tokens": _first(response, "prefill_tokens", "prompt_tokens"),
        "decode_tokens": _first(response, "decode_tokens", "completion_tokens", "generated_tokens", "tokens_generated"),
        "engine_ms": response.get("total_time_ms"), "wall_ms": round(wall_ms, 2),
        # The package's harness reads only function_calls, validation and confidence; an envelope with
        # success false still scores on its (empty) calls. Only a failed cactus_complete (rc < 0), where
        # the package raises, counts as a failure.
        "engine_failed": bool(response.get("_engine_failed")),
        "ok": not response.get("_engine_failed") and calls_equal(got, case["calls"]),
    }


def summarize(name, cases):
    passed = sum(c["ok"] for c in cases)
    critical = [c["query"] for c in cases if c["critical"] and not c["ok"]]
    need = round(0.9 * len(cases))
    return {"suite": name, "cases": cases, "passed": passed, "n": len(cases), "need": need,
            "critical_failures": len(critical), "critical_failed": critical,
            "pass": passed >= need and not critical}


def overall(suites):
    return {"passed": sum(s["passed"] for s in suites), "n": sum(s["n"] for s in suites),
            "suites_passed": sum(s["pass"] for s in suites), "suites": len(suites),
            "all_passed": all(s["pass"] for s in suites)}


def render_suite(result):
    lines = [result["suite"]]
    for c in result["cases"]:
        if not c["ok"]:
            lines.append(f"FAIL [{c['category']}{', critical' if c['critical'] else ''}] {c['query']}")
            lines.append(f"  want {json.dumps(c['want'])}")
            lines.append(f"  got  {json.dumps(c['got'])}" + (f" (raw {json.dumps(c['raw_calls'])}, gated by {c['gated']})" if c["gated"] else ""))
    lines.append(f"{result['passed']}/{result['n']} passed, {result['critical_failures']} critical failures")
    return "\n".join(lines)


def render_table(reports):
    names = [r["engine"] for r in reports]
    lines = ["| suite | " + " | ".join(f"{n} passed | {n} critical fails | {n} verdict" for n in names) + " |",
             "|---|" + "---|---|---|" * len(names)]
    for index, suite in enumerate(reports[0]["suites"]):
        cells = []
        for report in reports:
            s = next((x for x in report["suites"] if x["suite"] == suite["suite"]), None)
            cells.append("n/a | n/a | n/a" if s is None else f"{s['passed']}/{s['n']} | {s['critical_failures']} | {'PASS' if s['pass'] else 'FAIL'}")
        lines.append(f"| {suite['suite']} | " + " | ".join(cells) + " |")
    cells = [f"{r['overall']['passed']}/{r['overall']['n']} | | {r['overall']['suites_passed']}/{r['overall']['suites']} suites" for r in reports]
    lines.append("| **all** | " + " | ".join(cells) + " |")
    return "\n".join(lines)
