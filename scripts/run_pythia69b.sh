#!/usr/bin/env bash
# Full experiment on a CUDA node (e.g. Grid'5000): Pythia-6.9b, 20 checkpoints.
# Each checkpoint download is ~14 GB; set HF_HOME to a large scratch disk.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

uv run python -m wsd_probe all \
  --model EleutherAI/pythia-6.9b \
  --batch-size 8 \
  --max-length 128 \
  --n-permutations 1000 \
  "$@"
