#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
ref="${NEEDLE_REF_DIR:-$root/ref}"
lock="${NEEDLE_REF_LOCK:-$root/tools/ref.lock}"
tag="${1:-manylinux2014_x86_64}"

revision="$(awk '$1 == "revision" && $2 == "engine4" { print $3 }' "$lock")"
version="$(awk '$1 == "manifest" && $2 == "engine4" { print $3 }' "$lock")"
manifest_sha="$(awk '$1 == "manifest" && $2 == "engine4" { print $4 }' "$lock")"
[ -n "$revision" ] && [ -n "$version" ] && [ -n "$manifest_sha" ] || { echo "ref.lock has no engine4 pin" >&2; exit 1; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

base="https://huggingface.co/Cactus-Compute/cactus/resolve/$revision/releases/$version"
out="$ref/native4"
mkdir -p "$out"
curl -fsSL -o "$out/manifest.json" "$base/manifest.json"
[ "$(sha256_of "$out/manifest.json")" = "$manifest_sha" ] || { echo "engine 4 manifest checksum mismatch" >&2; exit 1; }

wheel="python/cactus_runtime-$version-py3-none-$tag.whl"
read -r expected size < <(python3 -I -c 'import json,sys; e=json.load(open(sys.argv[1]))["files"][sys.argv[2]]; print(e["sha256"], e["size"])' "$out/manifest.json" "$wheel")
local_wheel="$out/$(basename "$wheel")"
if [ ! -f "$local_wheel" ] || [ "$(sha256_of "$local_wheel")" != "$expected" ]; then
  curl -fsSL -o "$local_wheel.part" "$base/$wheel"
  [ "$(sha256_of "$local_wheel.part")" = "$expected" ] && [ "$(wc -c <"$local_wheel.part" | tr -d ' ')" = "$size" ] || { rm -f "$local_wheel.part"; echo "engine 4 wheel checksum mismatch" >&2; exit 1; }
  mv "$local_wheel.part" "$local_wheel"
fi
python3 -I -c 'import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); m=next(n for n in z.namelist() if n.endswith("libcactus.so") or n.endswith("libcactus.dylib")); open(sys.argv[2]+"/"+m.split("/")[-1],"wb").write(z.read(m))' "$local_wheel" "$out"
echo "engine $version ready in $out"
