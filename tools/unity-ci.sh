#!/usr/bin/env bash
# Runs the Unity side of needle-cs from the command line, the way .github/workflows/unity.yml does
# on the self-hosted Unity runner. Every step writes its log, NUnit XML and bench JSON under
# out/unity/ and exits non-zero when a test fails, when no test ran, or when Unity itself fails.
#
#   tools/unity-ci.sh lint                 package checks that need no Unity (metas, GUIDs, DLL stamps)
#   tools/unity-ci.sh editmode             EditMode tests of com.needle.inference in unity/NeedleDemo
#   tools/unity-ci.sh playmode             PlayMode tests (the host answering while frames tick)
#   tools/unity-ci.sh fresh-install        imports only the git-tracked package into an empty project
#   tools/unity-ci.sh bench-editmode       golden-set bench inside the Editor, needle.bench-e2e/1 JSON
#   tools/unity-ci.sh player il2cpp|coreclr  builds the bench player and runs it, needle.bench-e2e/1 JSON
#   tools/unity-ci.sh all                  everything above, in that order
#
# Environment:
#   UNITY_EDITOR          path to the Unity binary (default: ~/Unity/Hub/Editor/<ProjectVersion>/Editor/Unity)
#   NEEDLE_REF_CACHE      folder that tools/fetch.sh fills once and later runs reuse; ref/ is linked to it
#   NEEDLE_UNITY_LIBRARY  folder kept between runs for the demo project's Library
#                         (default: $NEEDLE_REF_CACHE/unity-library when NEEDLE_REF_CACHE is set)
#   NEEDLE_REQUIRE_MODEL  1 turns a missing model file into a failure instead of an Ignored test
#   NEEDLE_THREADS, NEEDLE_REPS, NEEDLE_WARMUPS, NEEDLE_SETS  passed to the benches
#   UNITY_SERIAL, UNITY_EMAIL, UNITY_PASSWORD  only used when the Editor has no license yet
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
project="$root/unity/NeedleDemo"
out="$root/out/unity"
mkdir -p "$out"

say() { printf '[unity-ci] %s\n' "$*" >&2; }
die() { say "$*"; exit 1; }

editor_version() { sed -n 's/^m_EditorVersion: *//p' "$project/ProjectSettings/ProjectVersion.txt"; }

find_editor() {
  if [ -n "${UNITY_EDITOR:-}" ]; then
    [ -x "$UNITY_EDITOR" ] || die "UNITY_EDITOR=$UNITY_EDITOR is not an executable."
    echo "$UNITY_EDITOR"; return
  fi
  # The project names 7000.0.0a16, but newer Unity 7 alphas open it too; take that exact one
  # when it is installed, otherwise the newest Unity 7 editor under the Hub folder.
  local version candidate newest
  version="$(editor_version)"
  candidate="$HOME/Unity/Hub/Editor/$version/Editor/Unity"
  if [ -x "$candidate" ]; then echo "$candidate"; return; fi
  newest="$(ls -1 "$HOME/Unity/Hub/Editor" 2>/dev/null | grep '^7' | sort -V | tail -n 1 || true)"
  candidate="$HOME/Unity/Hub/Editor/$newest/Editor/Unity"
  if [ -n "$newest" ] && [ -x "$candidate" ]; then say "Unity $version is not installed; using $newest."; echo "$candidate"; return; fi
  die "No Unity 7 editor under ~/Unity/Hub/Editor. Install one with Unity Hub (or: unity install 7) or set UNITY_EDITOR."
}

link_cache() {
  # ref/ holds the weights and clips (several GB); keep one copy outside the checkout.
  if [ -n "${NEEDLE_REF_CACHE:-}" ]; then
    mkdir -p "$NEEDLE_REF_CACHE"
    if [ -d "$root/ref" ] && [ ! -L "$root/ref" ]; then
      cp -an "$root/ref/." "$NEEDLE_REF_CACHE/" 2>/dev/null || true
      rm -rf "$root/ref"
    fi
    ln -sfn "$NEEDLE_REF_CACHE" "$root/ref"
  fi
  # The demo's Library takes minutes to import from scratch; checkout's clean step would delete it.
  local library="${NEEDLE_UNITY_LIBRARY:-${NEEDLE_REF_CACHE:+$NEEDLE_REF_CACHE/unity-library}}"
  if [ -n "$library" ]; then
    mkdir -p "$library"
    if [ -d "$project/Library" ] && [ ! -L "$project/Library" ]; then rm -rf "$project/Library"; fi
    ln -sfn "$library" "$project/Library"
  fi
}

ensure_license() {
  # A machine where someone has signed in to Unity Hub already has a license; nothing to do.
  [ -n "${UNITY_SERIAL:-}" ] || return 0
  local editor="$1"
  if ls "$HOME/.local/share/unity3d/Unity/"*.ulf >/dev/null 2>&1 || ls "$HOME/.config/unity3d/Unity/licenses/"* >/dev/null 2>&1; then
    return 0
  fi
  say "No Unity license on this machine; activating with UNITY_SERIAL."
  "$editor" -batchmode -nographics -quit -logFile "$out/activate.log" \
    -serial "$UNITY_SERIAL" -username "${UNITY_EMAIL:?UNITY_EMAIL is needed with UNITY_SERIAL}" \
    -password "${UNITY_PASSWORD:?UNITY_PASSWORD is needed with UNITY_SERIAL}" \
    || die "Unity activation failed; see out/unity/activate.log."
}

# Reads an NUnit 3 result file and fails on missing results, zero tests or any failure.
check_results() {
  local xml="$1" label="$2"
  [ -s "$xml" ] || die "$label: Unity wrote no test results ($xml). See the log next to it."
  python3 - "$xml" "$label" <<'PY'
import sys, xml.etree.ElementTree as ET
path, label = sys.argv[1], sys.argv[2]
run = ET.parse(path).getroot()
get = lambda k: int(run.get(k, "0"))
total, passed, failed, skipped = get("total"), get("passed"), get("failed"), get("skipped")
print(f"[unity-ci] {label}: {total} tests, {passed} passed, {failed} failed, {skipped} skipped", file=sys.stderr)
for case in run.iter("test-case"):
    result = case.get("result")
    if result == "Failed":
        message = case.findtext("failure/message", "").strip().splitlines()
        print(f"[unity-ci]   FAILED {case.get('fullname')}: {message[0] if message else ''}", file=sys.stderr)
    elif result == "Skipped":
        message = case.findtext("reason/message", "").strip().splitlines()
        print(f"[unity-ci]   skipped {case.get('fullname')}: {message[0] if message else ''}", file=sys.stderr)
if total == 0:
    print(f"[unity-ci] {label}: no tests ran; is com.needle.inference in the project's testables?", file=sys.stderr)
    sys.exit(3)
sys.exit(1 if failed else 0)
PY
}

refuse_open_project() {
  # A second Editor on the same project fights over Library/ and the results files.
  if pgrep -af 'Unity' 2>/dev/null | grep -F -- "$1" | grep -vq 'unity-ci.sh'; then
    die "An Editor already has $1 open; close it first."
  fi
}

# unity_tests <label> <EditMode|PlayMode> <assemblies> <project>: one Unity run, results checked.
unity_tests() {
  local label="$1" platform="$2" assemblies="$3" path="$4"
  # Returns 1 when tests failed (worth a second run) and exits 3 when the run is broken.
  local editor status=0
  editor="$(find_editor)" || exit 3
  ensure_license "$editor" || exit 3
  refuse_open_project "$path" || exit 3
  rm -f "$out/$label.xml"
  say "$label: $editor -runTests -testPlatform $platform ($assemblies)"
  # No -quit with -runTests: the test runner quits by itself.
  "$editor" -batchmode -nographics -projectPath "$path" \
    -runTests -testPlatform "$platform" -assemblyNames "$assemblies" \
    -testResults "$out/$label.xml" -logFile "$out/$label.log" || status=$?
  # Unity exits 2 when tests failed, and 1 when compiling, the license or the run itself broke.
  if grep -q 'error CS[0-9]' "$out/$label.log" 2>/dev/null; then
    grep 'error CS[0-9]' "$out/$label.log" | sort -u | head -n 20 >&2
    say "$label: compile errors; see out/unity/$label.log."; exit 3
  fi
  if [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then
    tail -n 60 "$out/$label.log" >&2 || true
    say "$label: Unity exited with $status before finishing the tests; see out/unity/$label.log."; exit 3
  fi
  [ -s "$out/$label.xml" ] || { say "$label: Unity wrote no test results; see out/unity/$label.log."; exit 3; }
  check_results "$out/$label.xml" "$label"
}

# run_tests <label> <EditMode|PlayMode> <assemblies> [project]
# A cold Library (first run, or the first after a manifest change) can fail the checks that need
# Burst-compiled code, because Burst compiles in the background. A failing run is therefore run
# once more and the second run decides; the first run's files are kept as <label>-cold.*.
run_tests() {
  local label="$1" platform="$2" assemblies="$3" path="${4:-$project}"
  local rc=0
  (unity_tests "$label" "$platform" "$assemblies" "$path") || rc=$?
  [ "$rc" -eq 0 ] && return 0
  [ "$rc" -eq 1 ] || exit 1
  mv -f "$out/$label.xml" "$out/$label-cold.xml"
  mv -f "$out/$label.log" "$out/$label-cold.log"
  say "$label: failed on the first run; running once more with a warm Library."
  (unity_tests "$label" "$platform" "$assemblies" "$path") || exit 1
}

cmd_lint() {
  python3 "$root/tools/unity-package-lint.py"
  bash "$root/tools/pack-unity.sh" --check
}

cmd_editmode() {
  link_cache
  run_tests editmode EditMode "Needle.Unity.Tests;Needle.Conformance.Tests;Needle.Kernels.Burst.Tests;Needle.Probe.Tests"
}

cmd_playmode() {
  link_cache
  NEEDLE_GAMEPATH_OUT="$out/gamepath-playmode.json" run_tests playmode PlayMode "Needle.Unity.PlayMode.Tests"
}

cmd_fresh_install() {
  # Imports exactly what a git-URL install gets: the tracked files of the package and nothing the
  # working tree has on top (the cause of #56). It goes in as a tarball, which Unity treats as
  # immutable like a Git URL, so a file without a .meta is dropped instead of getting a new one.
  # The package's own tests then check its DLLs and kernels.
  local fresh="$out/fresh-install"
  rm -rf "$fresh"
  mkdir -p "$fresh/stage/package" "$fresh/project/Packages" "$fresh/project/Assets"
  (cd "$root/unity/com.needle.inference" && git ls-files -z -- . | tar --null -T - -cf - | tar -xf - -C "$fresh/stage/package")
  tar -czf "$fresh/com.needle.inference.tgz" -C "$fresh/stage" package
  cat > "$fresh/project/Packages/manifest.json" <<JSON
{
  "dependencies": {
    "com.needle.inference": "file:$fresh/com.needle.inference.tgz",
    "com.unity.test-framework": "1.9.0"
  },
  "testables": ["com.needle.inference"]
}
JSON
  NEEDLE_REQUIRE_MODEL=0 run_tests fresh-install EditMode "Needle.Unity.Tests" "$fresh/project"
  if grep -E "error CS[0-9]+|Assembly .* will not be loaded|has no meta file" "$out/fresh-install.log" >&2; then
    die "fresh-install: the package compiled with errors or missing metas in a clean project."
  fi
}

cmd_bench_editmode() {
  link_cache
  NEEDLE_REQUIRE_MODEL=1 NEEDLE_BENCH_OUT="$out/bench-editmode.json" \
    run_tests bench-editmode EditMode "Needle.Bench.Tests"
  [ -s "$out/bench-editmode.json" ] || die "bench-editmode: no needle.bench-e2e/1 JSON was written."
  grep -q 'needle.bench-e2e/1' "$out/bench-editmode.json" || die "bench-editmode: the JSON is not needle.bench-e2e/1."
}

# The lines of a player build log that say why the build failed. The log ends with shutdown
# noise, so its tail rarely does; this prints the compiler, IL2CPP and build-step errors instead.
build_errors() {
  local log="$1"
  [ -f "$log" ] || { say "no build log at $log"; return 0; }
  say "errors from $log:"
  grep -n -E "error CS[0-9]+|error:|Error:|\[error\]|il2cpp.*(fail|error)|BuildFailedException|Building .* failed|Build Finished, Result: Failure|Exception:|undefined reference|cannot find|No such file|error while loading shared libraries" "$log" \
    | grep -v -E "Error: write EPIPE|Failed to extract CoreCLR stacktrace" | head -n 80 >&2 || true
  grep -n -E "^##utp:.*\"success\":false" "$log" | cut -c1-2000 >&2 || true
  if grep -q -E "error while loading shared libraries|libncurses\.so\.6|libxml2\.so\.2" "$log"; then
    say "IL2CPP's clang or lld could not start: a library it needs is missing on this machine."
    say "On Arch-based systems see unity/com.needle.inference/Documentation~/il2cpp-players.md (libncurses and libxml2)."
  fi
}

cmd_player() {
  local backend="${1:-}" method editor status=0
  case "$backend" in
    il2cpp) method="Needle.Bench.Tests.BenchBuild.PerformLinuxIl2cpp" ;;
    coreclr) method="Needle.Bench.Tests.BenchBuild.PerformLinuxCoreClr" ;;
    *) die "usage: tools/unity-ci.sh player il2cpp|coreclr" ;;
  esac
  link_cache
  editor="$(find_editor)"
  ensure_license "$editor"
  if [ "$backend" = il2cpp ]; then
    # Only a positive answer counts: if the folder layout is not what this expects, the build runs
    # and its own errors decide.
    local support
    support="$(dirname "$editor")/Data/PlaybackEngines/LinuxStandaloneSupport"
    if [ -d "$support" ] && ! find "$support" -maxdepth 3 -iname '*il2cpp*' | grep -q .; then
      die "player il2cpp: $editor has no Linux IL2CPP player under $support. Add the module: unity install-modules --editor-version $(basename "$(dirname "$(dirname "$editor")")") --module linux-il2cpp"
    fi
  fi
  local player="$out/player-$backend/BenchNs21"
  rm -rf "$out/player-$backend"
  say "player $backend: building with $method"
  NEEDLE_PLAYER_OUT="$player" "$editor" -batchmode -nographics -quit -projectPath "$project" \
    -buildTarget Linux64 -executeMethod "$method" -logFile "$out/player-$backend-build.log" || status=$?
  if [ "$status" -ne 0 ] || [ ! -x "$player" ]; then
    build_errors "$out/player-$backend-build.log"
    die "player $backend: build failed (exit $status); see out/unity/player-$backend-build.log."
  fi
  local json="$out/bench-player-$backend.json"
  "$player" -batchmode -nographics \
    --model "$root/ref/hf/needle3.cact" --tools "$root/tools/tools.json" --game-tools "$root/tools/game_tools.json" \
    --threads "${NEEDLE_THREADS:-4}" --reps "${NEEDLE_REPS:-2}" --warmups "${NEEDLE_WARMUPS:-1}" --sets "${NEEDLE_SETS:-golden}" \
    --out "$json" -logFile "$out/player-$backend.log" || status=$?
  [ -s "$json" ] || { tail -n 60 "$out/player-$backend.log" >&2 || true; die "player $backend: no bench JSON (exit $status)."; }
  grep -q 'needle.bench-e2e/1' "$json" || die "player $backend: the JSON is not needle.bench-e2e/1."
  [ "$status" -eq 0 ] || die "player $backend: the golden set did not reproduce (exit $status); see $json."
  say "player $backend: wrote $json"
}

case "${1:-}" in
  lint) cmd_lint ;;
  editmode) cmd_editmode ;;
  playmode) cmd_playmode ;;
  fresh-install) cmd_fresh_install ;;
  bench-editmode) cmd_bench_editmode ;;
  player) shift; cmd_player "${1:-}" ;;
  all)
    cmd_lint
    cmd_fresh_install
    cmd_editmode
    cmd_playmode
    cmd_bench_editmode
    cmd_player il2cpp
    cmd_player coreclr
    ;;
  *) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
