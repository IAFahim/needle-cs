#!/usr/bin/env bash
# N11: one runner per runtime, all emitting needle.bench-e2e/1 JSON.
#   tools/bench-ns21.sh coreclr  -- CoreCLR + ns2.1 (net10 host pinned to the
#                                  netstandard2.1 Needle build). Headless.
#   tools/bench-ns21.sh mono     -- Unity's bundled Mono 6.13 running a harness
#                                  built with its own csc against the netstandard
#                                  facade. Headless.
#   tools/bench-ns21.sh editmode -- unity test EditMode bench in the Editor.
#   tools/bench-ns21.sh player   -- IL2CPP bench player: build + run headless.
# Every leg reproduces the golden-set call outputs (Lagos -> get_weather,
# timer+music -> set_timer,play_music) and prints the JSON to stdout.
# Exit code: failure count from the runner (0 = reproduced + finite numbers).
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"

model="${NEEDLE_MODEL:-$root/ref/hf/needle3.cact}"
tools="${NEEDLE_TOOLS:-$root/tools/tools.json}"
game_tools="${NEEDLE_GAME_TOOLS:-$root/tools/game_tools.json}"
threads="${NEEDLE_THREADS:-4}"
reps="${NEEDLE_REPS:-2}"
warmups="${NEEDLE_WARMUPS:-1}"
sets="${NEEDLE_SETS:-golden}"
out="${NEEDLE_OUT:-}"
project="${SAY_PROJECT:-$HOME/Say}"
mode="${1:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    coreclr|mono|editmode|player) mode="$1"; shift ;;
    --model) model="$2"; shift 2 ;;
    --tools) tools="$2"; shift 2 ;;
    --game-tools) game_tools="$2"; shift 2 ;;
    --threads) threads="$2"; shift 2 ;;
    --reps) reps="$2"; shift 2 ;;
    --warmups) warmups="$2"; shift 2 ;;
    --sets) sets="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --project) project="$2"; shift 2 ;;
    *) echo "bench-ns21: unknown argument $1" >&2; exit 2 ;;
  esac
done

case "$mode" in
  coreclr|"")
    dotnet build "$root/samples/BenchNs21/BenchNs21.csproj" -c Release \
      -o "$root/out/bench-ns21/coreclr" --nologo -v q >&2
    exe="$root/out/bench-ns21/coreclr/bench-ns21"
    if [ -z "$out" ]; then
      exec "$exe" "$model" "$tools" --threads "$threads" --reps "$reps" \
        --warmups "$warmups" --sets "$sets" --game-tools "$game_tools"
    else
      "$exe" "$model" "$tools" --threads "$threads" --reps "$reps" \
        --warmups "$warmups" --sets "$sets" --game-tools "$game_tools" | tee "$out"
    fi
    ;;
  mono)
    mono_root="$HOME/Unity/Hub/Editor/7000.0.0a12/Editor/Data/MonoBleedingEdge"
    mono_bin="$mono_root/bin/mono"
    csc="$mono_root/lib/mono/4.5/csc.exe"
    facades="$mono_root/lib/mono/unityjit-linux/Facades"
    if [ ! -x "$mono_bin" ]; then echo "bench-ns21: no mono at $mono_bin" >&2; exit 2; fi
    dotnet build "$root/src/Needle/Needle.csproj" -c Release -f netstandard2.1 --nologo -v q >&2
    ns21_dll="$root/src/Needle/bin/Release/netstandard2.1/Needle.Inference.dll"
    mkdir -p "$root/out/bench-ns21/mono"
    # System.Runtime.CompilerServices.Unsafe ships with the ns2.1 build; Mono's
    # GAC does not provide it, so stage it next to the harness.
    unsafe_dll="$(find "$HOME/.nuget/packages/system.runtime.compilerservices.unsafe" -name "System.Runtime.CompilerServices.Unsafe.dll" -path "*netstandard*" 2>/dev/null | head -1 || true)"
    if [ -n "${unsafe_dll:-}" ]; then cp -f "$unsafe_dll" "$root/out/bench-ns21/mono/"; fi
    cp -f "$ns21_dll" "$root/out/bench-ns21/mono/"
    # Unity script profile: the unityjit-linux mscorlib + System.dll pair plus
    # every netstandard facade (Stopwatch/Process live in System.dll here, not
    # mscorlib; /lib alone does not resolve type forwarders, so pass each /r).
    profile="$mono_root/lib/mono/unityjit-linux"
    refs="/r:$profile/mscorlib.dll /r:$profile/System.dll /r:$profile/System.Core.dll"
    for f in "$facades"/*.dll; do refs="$refs /r:$f"; done
    # shellcheck disable=SC2086
    "$mono_bin" "$csc" /nologo /nostdlib /noconfig /t:exe \
      "/out:$root/out/bench-ns21/mono/bench-mono.exe" \
      $refs \
      "/r:$root/out/bench-ns21/mono/Needle.Inference.dll" \
      "$root/unity/com.needle.inference/Tests/Bench/BenchDriver.cs" \
      "$root/samples/BenchNs21/BenchArgs.cs" \
      "$root/samples/BenchNs21/MonoMain.cs"
    (cd "$root/out/bench-ns21/mono" && if [ -z "$out" ]; then
      exec "$mono_bin" bench-mono.exe "$model" "$tools" --threads "$threads" --reps "$reps" \
        --warmups "$warmups" --sets "$sets" --game-tools "$game_tools"
    else
      "$mono_bin" bench-mono.exe "$model" "$tools" --threads "$threads" --reps "$reps" \
        --warmups "$warmups" --sets "$sets" --game-tools "$game_tools" | tee "$out"
    fi)
    ;;
  editmode)
    NEEDLE_MODEL="$model" NEEDLE_TOOLS="$tools" NEEDLE_GAME_TOOLS="$game_tools" \
    NEEDLE_THREADS="$threads" NEEDLE_REPS="$reps" NEEDLE_WARMUPS="$warmups" NEEDLE_SETS="$sets" \
    unity test --project-path "$project" --mode EditMode --filter "Needle.Bench"
    ;;
  player)
    player_out="${NEEDLE_PLAYER_OUT:-$root/out/bench-ns21/BenchNs21}"
    NEEDLE_PLAYER_OUT="$player_out" \
    unity build "$project" \
      --execute-method "Needle.Bench.Tests.BenchBuild.PerformLinuxIl2cpp" \
      --target StandaloneLinux64 --output-path "$player_out"
    if [ -z "$out" ]; then out="$root/out/bench-ns21/player.json"; fi
    "$player_out" -batchmode -nographics \
      --model "$model" --tools "$tools" --game-tools "$game_tools" \
      --threads "$threads" --reps "$reps" --warmups "$warmups" --sets "$sets" \
      --out "$out" -logFile "$root/out/bench-ns21/player.log"
    cat "$out"
    ;;
  *)
    echo "usage: tools/bench-ns21.sh <coreclr|mono|editmode|player> [options]" >&2
    exit 2
    ;;
esac
