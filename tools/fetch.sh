#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
needle=https://huggingface.co/Cactus-Compute/needle3/resolve/main
whistle=https://huggingface.co/Cactus-Compute/whistle/resolve/main
clips=https://raw.githubusercontent.com/cactus-compute/cactus/main/cactus-engine/tests/assets
mkdir -p "$root/ref/hf/tokenizer" "$root/ref/hf/wheel" "$root/ref/hf/whistle" "$root/ref/audio"
curl -fL -o "$root/ref/hf/needle3.cact" "$needle/needle3.cact"
curl -fL -o "$root/ref/hf/config.json" "$needle/config.json"
curl -fL -o "$root/ref/hf/tokenizer/tokenizer.model" "$needle/tokenizer/tokenizer.model"
curl -fL -o "$root/ref/hf/wheel/cactus_needle-3.3.3-py3-none-manylinux2014_x86_64.whl" "$needle/python/cactus_needle-3.3.3-py3-none-manylinux2014_x86_64.whl"
curl -fL -o "$root/ref/hf/whistle/whistle.cact" "$whistle/whistle.cact"
for clip in test.wav test_long.wav hotword.wav record.wav; do curl -fL -o "$root/ref/audio/$clip" "$clips/$clip"; done
