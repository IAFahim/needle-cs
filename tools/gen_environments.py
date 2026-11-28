#!/usr/bin/env python3
"""Regenerate tests/eval/environments from the Apache-2.0 cactus-compute package.

The six acceptance suites (TOOLS, SYSTEM, TEST_CASES) live in
cactus/environments/*.py of the cactus-compute wheel. This script loads each
module against a stub `cactus` package that carries only the package's own
schema builder (cactus/agent/tools.py), so no engine, network or telemetry
code runs, and writes:

  <suite>.tools.json  the exact bytes upstream hands to cactus_init:
                      json.dumps([...schemas]) with default separators
  <suite>.json        {"suite", "source", "system", "cases": [...]}

usage: python3 -I tools/gen_environments.py <extracted-wheel-dir> [out-dir]
"""
import importlib.util
import json
import os
import sys
import types

SUITES = ("smart_home", "media_player", "productivity", "wearable", "kitchen_appliance", "data_capture")


def stub_package(root):
    spec = importlib.util.spec_from_file_location("cactus_schema_tools", os.path.join(root, "cactus", "agent", "tools.py"))
    tools = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tools
    spec.loader.exec_module(tools)
    cactus = types.ModuleType("cactus")
    cactus.tool, cactus.Field = tools.tool, tools.Field
    cactus.__path__ = []
    environments = types.ModuleType("cactus.environments")
    environments.__path__ = []
    harness = types.ModuleType("cactus.environments._harness")
    environments._harness = harness
    cactus.environments = environments
    sys.modules.update({"cactus": cactus, "cactus.environments": environments, "cactus.environments._harness": harness})
    return tools


def main():
    root = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "eval", "environments")
    with open(os.path.join(root, "cactus", "__init__.py")) as handle:
        version = next(line.split('"')[1] for line in handle if line.startswith("__version__"))
    tools = stub_package(root)
    os.makedirs(out, exist_ok=True)
    for suite in SUITES:
        spec = importlib.util.spec_from_file_location(f"cactus.environments.{suite}", os.path.join(root, "cactus", "environments", suite + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        schemas = [getattr(fn, "_cactus_tool", None) or tools.build_schema(fn) for fn in module.TOOLS]
        with open(os.path.join(out, suite + ".tools.json"), "w", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(schemas))
        cases = [{"query": c["query"], "calls": c["calls"], "category": c["category"], "critical": bool(c.get("critical", False))}
                 for c in module.TEST_CASES]
        document = {"suite": suite, "source": f"cactus-compute {version} cactus/environments/{suite}.py (Apache-2.0)",
                    "system": module.SYSTEM, "cases": cases}
        with open(os.path.join(out, suite + ".json"), "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=1, ensure_ascii=False)
            handle.write("\n")
        print(f"{suite}: {len(schemas)} tools, {len(cases)} cases, {sum(c['critical'] for c in cases)} critical")


if __name__ == "__main__":
    main()
