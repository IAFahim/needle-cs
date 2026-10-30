#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
ref="${NEEDLE_REF_DIR:-$root/ref}"
lock="${NEEDLE_REF_LOCK:-$root/tools/ref.lock}"
guidance='see CONTRIBUTING.md, section "Refreshing the upstream model"'

files=(
  "needle3|needle3.cact|hf/needle3.cact"
  "needle3|config.json|hf/config.json"
  "needle3|tokenizer/tokenizer.model|hf/tokenizer/tokenizer.model"
  "needle3|python/cactus_needle-3.3.3-py3-none-manylinux2014_x86_64.whl|hf/wheel/cactus_needle-3.3.3-py3-none-manylinux2014_x86_64.whl"
  "whistle|whistle.cact|hf/whistle/whistle.cact"
  "cactus|test.wav|audio/test.wav"
  "cactus|test_long.wav|audio/test_long.wav"
  "cactus|hotword.wav|audio/hotword.wav"
  "cactus|record.wav|audio/record.wav"
)

usage() {
  echo "usage: tools/fetch.sh [--verify | --update-lock]" >&2
  echo "  (no option)    download the files pinned in tools/ref.lock into ref/ and verify their SHA-256" >&2
  echo "  --verify       check the files already in ref/ against tools/ref.lock, download nothing" >&2
  echo "  --update-lock  download the newest upstream revisions and rewrite tools/ref.lock" >&2
  exit 2
}

fail() {
  echo "$*" >&2
  exit 1
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

base_url() {
  case "$1" in
    needle3) echo "https://huggingface.co/Cactus-Compute/needle3/resolve/$2" ;;
    whistle) echo "https://huggingface.co/Cactus-Compute/whistle/resolve/$2" ;;
    cactus) echo "https://raw.githubusercontent.com/cactus-compute/cactus/$2/cactus-engine/tests/assets" ;;
    *) fail "unknown source $1" ;;
  esac
}

latest_revision() {
  case "$1" in
    needle3 | whistle) curl -fsS "https://huggingface.co/api/models/Cactus-Compute/$1" | grep -o '"sha":"[0-9a-f]\{40\}"' | head -1 | cut -d'"' -f4 ;;
    cactus) curl -fsS "https://github.com/cactus-compute/cactus.git/info/refs?service=git-upload-pack" | grep -a -o '[0-9a-f]\{40\} HEAD' | head -1 | cut -d' ' -f1 ;;
  esac
}

revision_in() {
  awk -v source="$2" '$1 == "revision" && $2 == source { print $3 }' "$1"
}

locked_sha() {
  awk -v path="$1" '$1 == "file" && $6 == path { print $2 }' "$lock"
}

check_lock_covers_files() {
  [ -f "$lock" ] || fail "lock file $lock is missing; create it with: tools/fetch.sh --update-lock"
  for entry in "${files[@]}"; do
    IFS='|' read -r source remote local_path <<<"$entry"
    [ -n "$(locked_sha "$local_path")" ] || fail "lock file $lock has no entry for $local_path; $guidance"
    [ -n "$(revision_in "$lock" "$source")" ] || fail "lock file $lock has no revision for $source; $guidance"
  done
}

mismatch_message() {
  echo "upstream changed: $1 has SHA-256 $2 but $lock pins $3; $guidance" >&2
}

download() {
  mkdir -p "$(dirname "$2")"
  curl -fsSL -o "$2.part" "$1" || { rm -f "$2.part"; fail "download failed: $1"; }
}

verify_all() {
  check_lock_covers_files
  local bad=0
  for entry in "${files[@]}"; do
    IFS='|' read -r source remote local_path <<<"$entry"
    local expected actual
    expected="$(locked_sha "$local_path")"
    if [ ! -f "$ref/$local_path" ]; then
      echo "missing $ref/$local_path; run tools/fetch.sh" >&2
      bad=1
      continue
    fi
    actual="$(sha256_of "$ref/$local_path")"
    if [ "$actual" != "$expected" ]; then
      mismatch_message "$ref/$local_path" "$actual" "$expected"
      bad=1
    fi
  done
  [ "$bad" = 0 ] || exit 1
  echo "ref files match $lock"
}

fetch_all() {
  check_lock_covers_files
  for entry in "${files[@]}"; do
    IFS='|' read -r source remote local_path <<<"$entry"
    local expected actual destination="$ref/$local_path"
    expected="$(locked_sha "$local_path")"
    if [ -f "$destination" ] && [ "$(sha256_of "$destination")" = "$expected" ]; then
      echo "ok $local_path"
      continue
    fi
    download "$(base_url "$source" "$(revision_in "$lock" "$source")")/$remote" "$destination"
    actual="$(sha256_of "$destination.part")"
    if [ "$actual" != "$expected" ]; then
      rm -f "$destination.part"
      mismatch_message "downloaded $local_path" "$actual" "$expected"
      exit 1
    fi
    mv "$destination.part" "$destination"
    echo "fetched $local_path"
  done
}

update_lock() {
  local staging
  staging="$(mktemp)"
  for source in needle3 whistle cactus; do
    local revision
    revision="$(latest_revision "$source")"
    [ -n "$revision" ] || fail "could not resolve the newest revision of $source"
    echo "revision $source $revision" >>"$staging"
  done
  for entry in "${files[@]}"; do
    IFS='|' read -r source remote local_path <<<"$entry"
    local destination="$ref/$local_path"
    download "$(base_url "$source" "$(revision_in "$staging" "$source")")/$remote" "$destination"
    mv "$destination.part" "$destination"
    echo "file $(sha256_of "$destination") $(wc -c <"$destination" | tr -d ' ') $source $remote $local_path" >>"$staging"
  done
  mv "$staging" "$lock"
  chmod 644 "$lock"
  echo "wrote $lock; regenerate the golden fixtures next ($guidance)"
}

case "${1:-}" in
  "") fetch_all ;;
  --verify) verify_all ;;
  --update-lock) update_lock ;;
  *) usage ;;
esac
