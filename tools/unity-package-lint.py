#!/usr/bin/env python3
"""Checks unity/com.needle.inference the way a Git-URL install sees it, without Unity.

An immutable (Git or tarball) package install drops every file that has no .meta and every file
git does not track, so a package can work from a file: path and break for everyone else (#56).
This script fails when:
  - a file or folder Unity imports has no .meta, or a .meta has nothing next to it
  - two .meta files share a GUID
  - a file Unity needs (the plugin DLLs above all) is git-ignored or untracked
  - a test assembly lacks defineConstraints UNITY_INCLUDE_TESTS, so it would compile in every project
  - two asmdefs share a name, or an asmdef is not valid JSON
  - package C# uses syntax newer than the C# 9 Unity 7 compiles with
Run from anywhere: python3 tools/unity-package-lint.py
"""
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "unity" / "com.needle.inference"
REQUIRED_DLLS = [
    "Runtime/Plugins/netstandard2.1/Needle.Inference.dll",
    "Runtime/Plugins/net10.0/Needle.Inference.Net10.dll",
]
# Syntax Unity 7's C# 9 compiler rejects, with what to write instead.
NEWER_SYNTAX = [
    (re.compile(r'"[^"\n]*"u8'), 'u8 string literal (C# 11); use Encoding.UTF8.GetBytes'),
    (re.compile(r'^\s*namespace\s+[\w.]+\s*;', re.M), 'file-scoped namespace (C# 10); use a block'),
    (re.compile(r'"""'), 'raw string literal (C# 11)'),
    (re.compile(r'\brequired\s+(public|internal|private|protected)?\s*\w'), 'required member (C# 11)'),
    (re.compile(r'^\s*global\s+using\b', re.M), 'global using (C# 10)'),
]

problems = []


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)


def unity_sees(rel):
    """True when Unity imports this path: no part ends in ~ or starts with a dot."""
    return not any(part.endswith("~") or part.startswith(".") for part in rel.parts)


def main():
    if not PACKAGE.is_dir():
        sys.exit(f"no package at {PACKAGE}")

    listed = git("ls-files", "--cached", "--others", "--exclude-standard", "--", str(PACKAGE))
    tracked = git("ls-files", "--cached", "--", str(PACKAGE))
    if listed.returncode != 0:
        sys.exit("unity-package-lint needs a git checkout: " + listed.stderr.strip())
    files = {pathlib.Path(p).relative_to(PACKAGE.relative_to(ROOT)) for p in listed.stdout.splitlines()}
    tracked_files = {pathlib.Path(p).relative_to(PACKAGE.relative_to(ROOT)) for p in tracked.stdout.splitlines()}
    untracked = files - tracked_files

    assets = set()
    metas = set()
    for rel in files:
        if not (PACKAGE / rel).exists():
            continue  # deleted in the working tree
        if rel.suffix == ".meta":
            metas.add(rel)
        else:
            assets.add(rel)
            for parent in rel.parents:
                if parent != pathlib.Path("."):
                    assets.add(parent)

    for rel in sorted(assets):
        if not unity_sees(rel) or rel == pathlib.Path("package.json.meta"):
            continue
        meta = rel.with_name(rel.name + ".meta")
        if meta not in metas:
            problems.append(f"{rel}: no .meta, so a Git-URL install drops it")
    for meta in sorted(metas):
        asset = meta.with_name(meta.name[: -len(".meta")])
        if unity_sees(meta) and asset not in assets:
            problems.append(f"{meta}: orphan .meta (nothing named {asset.name} next to it)")

    guids = {}
    for meta in sorted(metas):
        text = (PACKAGE / meta).read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^guid:\s*([0-9a-f]{32})\s*$", text, re.M)
        if not match:
            problems.append(f"{meta}: no 32-hex guid line")
            continue
        guid = match.group(1)
        if guid in guids:
            problems.append(f"{meta}: guid {guid} is also used by {guids[guid]}")
        guids[guid] = meta

    for dll in REQUIRED_DLLS:
        rel = pathlib.Path(dll)
        if not (PACKAGE / rel).is_file():
            problems.append(f"{dll}: missing; run tools/pack-unity.sh")
        elif git("check-ignore", "-q", str(PACKAGE / rel)).returncode == 0:
            problems.append(f"{dll}: git-ignored, so a Git-URL install has no plugin (#56)")
        elif rel in untracked:
            problems.append(f"{dll}: not committed yet; git add it with its .meta")

    names = {}
    for rel in sorted(a for a in assets if a.suffix == ".asmdef" and unity_sees(a)):
        try:
            asmdef = json.loads((PACKAGE / rel).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            problems.append(f"{rel}: not valid JSON ({error})")
            continue
        name = asmdef.get("name", "")
        if name in names:
            problems.append(f"{rel}: assembly name {name} is also used by {names[name]}")
        names[name] = rel
        references = asmdef.get("references", []) + asmdef.get("precompiledReferences", [])
        is_test = any("TestRunner" in r or r == "nunit.framework.dll" for r in references) or \
            "TestAssemblies" in asmdef.get("optionalUnityReferences", [])
        if (is_test or name.endswith(".Tests")) and "UNITY_INCLUDE_TESTS" not in asmdef.get("defineConstraints", []):
            problems.append(f"{rel}: test assembly {name} without defineConstraints UNITY_INCLUDE_TESTS compiles in every project that installs the package (#56)")

    for rel in sorted(a for a in assets if a.suffix == ".cs"):
        text = (PACKAGE / rel).read_text(encoding="utf-8", errors="replace")
        for pattern, what in NEWER_SYNTAX:
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                where = "sample (compiled once imported)" if not unity_sees(rel) else "package"
                problems.append(f"{rel}:{line}: {what}; Unity 7 compiles {where} code at C# 9")

    for problem in problems:
        print(f"unity-package-lint: {problem}", file=sys.stderr)
    print(f"unity-package-lint: {len(assets)} assets, {len(metas)} metas, {len(names)} asmdefs, {len(problems)} problems", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
