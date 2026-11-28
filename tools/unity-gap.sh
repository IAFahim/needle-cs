#!/usr/bin/env bash
# Measures where Unity's speedup over the native library goes, on the self-hosted Unity runner.
# Run by the profile job of .github/workflows/unity.yml (workflow_dispatch with profile ticked).
# It installs nothing and changes nothing outside the checkout: it runs the native library, a
# prebuilt needle-cs CLI (out/cli/jit, out/cli/aot) and the two Unity bench players, and writes
# everything under out/gap/.
#
#   tools/unity-gap.sh machine     CPU topology, frequencies, what each build picks
#   tools/unity-gap.sh native-threads  how many threads the native library runs by default
#   tools/unity-gap.sh engines     native vs plain .NET needle-cs, interleaved rounds (engine ceiling)
#   tools/unity-gap.sh stages      needle-cs per-stage profile at 2 and 4 threads
#   tools/unity-gap.sh players     native vs CoreCLR player (net10 sidecar) vs CoreCLR player with the
#                                  Burst kernels vs IL2CPP player, interleaved rounds, plus thread sweeps
#   tools/unity-gap.sh perf        perf samples of the IL2CPP player and the native library, if perf works
set -uo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
out="$root/out/gap"
mkdir -p "$out"
rounds="${NEEDLE_GAP_ROUNDS:-3}"
reps="${NEEDLE_REPS:-3}"
model="$root/ref/hf/needle3.cact"
tools="$root/tools/tools.json"
game_tools="$root/tools/game_tools.json"
jit="$root/out/cli/jit/needle"
aot="$root/out/cli/aot/needle"

say() { printf '[unity-gap] %s\n' "$*" >&2; }

native() { python3 tools/native_bench.py ref/hf/wheel "$model" "$tools" --sets all --reps "$reps" --warmup 1 --json "$@"; }

load_line() { say "load $(cut -d' ' -f1-3 /proc/loadavg) before $1"; }

cmd_machine() {
  lscpu
  echo
  lscpu -e=CPU,CORE,SOCKET,ONLINE,MAXMHZ,MINMHZ 2>/dev/null || lscpu -e
  echo
  printf 'cpu capacity max_freq governor epp prefcore_rank highest_perf siblings\n'
  for d in /sys/devices/system/cpu/cpu[0-9]*; do
    c="${d##*/cpu}"
    printf '%s %s %s %s %s %s %s %s\n' "$c" \
      "$(cat "$d/cpu_capacity" 2>/dev/null || echo -)" \
      "$(cat "$d/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo -)" \
      "$(cat "$d/cpufreq/scaling_governor" 2>/dev/null || echo -)" \
      "$(cat "$d/cpufreq/energy_performance_preference" 2>/dev/null || echo -)" \
      "$(cat "$d/cpufreq/amd_pstate_prefcore_ranking" 2>/dev/null || echo -)" \
      "$(cat "$d/cpufreq/amd_pstate_highest_perf" 2>/dev/null || echo -)" \
      "$(cat "$d/topology/thread_siblings_list" 2>/dev/null || echo -)"
  done | sort -n
  echo
  grep -H . /sys/devices/system/cpu/cpufreq/boost /sys/devices/system/cpu/amd_pstate/status 2>/dev/null || true
  echo "perf: $(command -v perf || echo none), perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null)"
  echo "dotnet: $(command -v dotnet || echo none)"
  echo "loadavg: $(cat /proc/loadavg)"
  free -g || true
  for build in "$jit" "$aot"; do
    [ -x "$build" ] || { echo "$build: missing"; continue; }
    echo "== $build cpu"; "$build" cpu || true
  done
}

cmd_native_threads() {
  # The 3.3.3 library has no set_threads; watch what it runs on its own.
  python3 tools/native_bench.py ref/hf/wheel "$model" "$tools" --sets game --reps 40 --warmup 1 --json > /dev/null &
  local pid=$!
  sleep 12
  echo "native threads in the process: $(ps -o nlwp= -p "$pid" 2>/dev/null)"
  grep -E 'Cpus_allowed_list|Threads' "/proc/$pid/status" 2>/dev/null || true
  echo "per-thread CPU over 3 s (psr = core it ran on last):"
  for _ in 1 2 3; do ps -L -o tid=,psr=,pcpu=,comm= -p "$pid" 2>/dev/null | sort -k3 -nr | head -n 16; echo; sleep 1; done
  if command -v python3 >/dev/null && [ -d "/proc/$pid/task" ]; then
    echo "per-thread CPU seconds:"
    for t in /proc/$pid/task/*; do awk -v t="${t##*/}" '{ printf "%s utime=%.2fs cpu=%s\n", t, ($14+$15)/100, $39 }' "$t/stat"; done | sort -k2 -t= -nr | head -n 16
  fi
  wait "$pid" || true
}

cmd_native_set_threads() {
  # 4.x-style set_threads, if this build happens to export it.
  python3 tools/native_bench.py ref/hf/wheel "$model" "$tools" --sets golden --reps 1 --threads 4 --json > /dev/null 2>"$out/native-set-threads.err" \
    && echo yes || { cat "$out/native-set-threads.err" >&2; echo no; }
}

cmd_engines() {
  [ -x "$jit" ] || { say "no JIT CLI at $jit"; return 1; }
  local engines=(--engine "native-3.3.3" "python3 tools/native_bench.py ref/hf/wheel $model $tools --json --reps $reps")
  engines+=(--engine "cs-jit" "$jit bench-e2e $model $tools --threads 1,2,3,4,6,0 --reps $reps --json")
  engines+=(--engine "cs-jit nopin" "env NEEDLE_PIN=0 $jit bench-e2e $model $tools --threads 2,4,6 --reps $reps --json")
  engines+=(--engine "cs-jit avx2-only" "env NEEDLE_NOEVEX=1 NEEDLE_NOVNNI=1 NEEDLE_NOAVX512=1 $jit bench-e2e $model $tools --threads 2,4 --reps $reps --json")
  engines+=(--engine "cs-jit tiered" "env DOTNET_TieredCompilation=1 DOTNET_TieredPGO=1 $jit bench-e2e $model $tools --threads 4 --reps $reps --json")
  [ -x "$aot" ] && engines+=(--engine "cs-aot" "$aot bench-e2e $model $tools --threads 2,4,0 --reps $reps --json")
  load_line engines
  python3 tools/bench_engines.py --rounds "$rounds" --detail-threads 4 --json "$out/engines.json" \
    --title "Fahim's PC: native 3.3.3 (its default threads) vs plain .NET needle-cs, interleaved rounds" "${engines[@]}" | tee "$out/engines.md"
  # Single reports in bench-e2e form for the gap table.
  native > "$out/plain-native.json"
  "$jit" bench-e2e "$model" "$tools" --threads 2,4 --reps "$reps" --json > "$out/plain-jit.json"
  env NEEDLE_NOEVEX=1 NEEDLE_NOVNNI=1 NEEDLE_NOAVX512=1 "$jit" bench-e2e "$model" "$tools" --threads 4 --reps "$reps" --json > "$out/plain-jit-avx2.json"
  python3 tools/unity-gap.py --native "$out/plain-native.json" "cs-jit=$out/plain-jit.json" "cs-jit avx2-only=$out/plain-jit-avx2.json" | tee "$out/plain-gap.md"
}

cmd_stages() {
  [ -x "$jit" ] || return 1
  for t in 2 4; do
    load_line "stages $t"
    echo "== needle-cs JIT, $t threads, per-stage profile"
    "$jit" bench-e2e "$model" "$tools" --sets golden --threads "$t" --reps 5 --profile | tail -n 80
    echo "== same, Burst-like kernels (AVX2 + FMA only)"
    env NEEDLE_NOEVEX=1 NEEDLE_NOVNNI=1 NEEDLE_NOAVX512=1 "$jit" bench-e2e "$model" "$tools" --sets golden --threads "$t" --reps 5 --profile | tail -n 80
  done
}

player_run() {
  # player_run <backend> <out.json> <threads> [env...]
  local backend="$1" json="$2" threads="$3"; shift 3
  local player="$root/out/unity/player-$backend/BenchNs21"
  [ -x "$player" ] || { say "no $backend player at $player"; return 1; }
  env "$@" "$player" -batchmode -nographics --model "$model" --tools "$tools" --game-tools "$game_tools" \
    --threads "$threads" --reps "$reps" --warmups 1 --sets all --out "$json" -logFile "${json%.json}.log" || true
  grep -h -E '^\[needle\]|burst explicit' "${json%.json}.log" 2>/dev/null | head -n 3 >&2 || true
}

cmd_players() {
  for backend in coreclr il2cpp; do
    [ -x "$root/out/unity/player-$backend/BenchNs21" ] || { NEEDLE_SETS=golden NEEDLE_REPS=1 tools/unity-ci.sh player "$backend" || say "player $backend: build failed"; }
  done
  local r
  for r in $(seq 1 "$rounds"); do
    load_line "players round $r"
    native > "$out/r$r-native.json"
    player_run coreclr "$out/r$r-coreclr.json" 4
    player_run coreclr "$out/r$r-coreclr-burst.json" 4 NEEDLE_FORCE_BURST=1
    player_run il2cpp "$out/r$r-il2cpp.json" 4
    "$jit" bench-e2e "$model" "$tools" --threads 4 --reps "$reps" --json > "$out/r$r-jit.json" 2>/dev/null || true
    echo "### Round $r"
    python3 tools/unity-gap.py --native "$out/r$r-native.json" \
      "plain .NET=$out/r$r-jit.json" "coreclr player=$out/r$r-coreclr.json" \
      "coreclr + burst kernels=$out/r$r-coreclr-burst.json" "il2cpp player=$out/r$r-il2cpp.json" | tee "$out/r$r-gap.md"
    echo
  done
  load_line "player thread sweeps"
  native > "$out/sweep-native.json"
  player_run il2cpp "$out/sweep-il2cpp.json" 1,2,3,4,6
  player_run il2cpp "$out/sweep-il2cpp-nopin.json" 2,4,6 NEEDLE_PIN=0
  player_run coreclr "$out/sweep-coreclr.json" 1,2,3,4,6
  player_run coreclr "$out/sweep-coreclr-burst.json" 2,4 NEEDLE_FORCE_BURST=1
  echo "### Thread sweeps"
  python3 tools/unity-gap.py --native "$out/sweep-native.json" \
    "il2cpp=$out/sweep-il2cpp.json" "il2cpp nopin=$out/sweep-il2cpp-nopin.json" \
    "coreclr=$out/sweep-coreclr.json" "coreclr + burst=$out/sweep-coreclr-burst.json" | tee "$out/sweep-gap.md"
  for f in "$out"/*.log; do grep -h 'needle.probe/1' "$f" 2>/dev/null | head -n 1; done | sort -u | head -n 4
}

cmd_perf() {
  if ! command -v perf >/dev/null; then say "perf is not installed; skipping"; return 0; fi
  local player="$root/out/unity/player-il2cpp/BenchNs21"
  if [ -x "$player" ]; then
    perf record -F 1999 -o "$out/il2cpp.perf.data" -- "$player" -batchmode -nographics --model "$model" --tools "$tools" \
      --game-tools "$game_tools" --threads 4 --reps 3 --warmups 1 --sets all --out "$out/perf-il2cpp.json" -logFile "$out/perf-il2cpp.log" \
      && perf report -i "$out/il2cpp.perf.data" --stdio --no-children --sort dso,symbol 2>/dev/null | grep -v '^#' | grep -v '^$' | head -n 70 \
      || say "perf record on the IL2CPP player failed"
  fi
  perf record -F 1999 -o "$out/native.perf.data" -- python3 tools/native_bench.py ref/hf/wheel "$model" "$tools" --sets all --reps 3 --json > /dev/null \
    && perf report -i "$out/native.perf.data" --stdio --no-children --sort dso,symbol 2>/dev/null | grep -v '^#' | grep -v '^$' | head -n 40 \
    || say "perf record on the native library failed"
  rm -f "$out"/*.perf.data
}

case "${1:-}" in
  machine) cmd_machine ;;
  native-threads) cmd_native_threads; echo "set_threads exported: $(cmd_native_set_threads)" ;;
  engines) cmd_engines ;;
  stages) cmd_stages ;;
  players) cmd_players ;;
  perf) cmd_perf ;;
  *) echo "usage: tools/unity-gap.sh machine|native-threads|engines|stages|players|perf" >&2; exit 2 ;;
esac
