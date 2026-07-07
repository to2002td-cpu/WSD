#!/usr/bin/env bash
# Payload executed on the reserved node by OAR (see submit_oar.sh).
# Runs the full pipeline; any extra arguments are forwarded to `wsd_probe all`.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

echo "OAR job ${OAR_JOB_ID:-?} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  || echo "WARNING: no GPU visible"

# Node-local scratch for the HF cache: pythia-6.9b revisions are ~14 GB each
# and would blow the NFS home quota. Combined with --purge-cache below, disk
# usage stays bounded to one checkpoint at a time.
export HF_HOME="${HF_HOME:-/tmp/$USER/hf_cache}"
mkdir -p "$HF_HOME"

# uv is not in the standard Grid'5000 environment.
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || {
    curl -LsSf https://astral.sh/uv/install.sh | sh
  }
fi

uv sync

exec uv run python -m wsd_probe all \
  --model EleutherAI/pythia-6.9b \
  --purge-cache \
  --n-permutations 1000 \
  "$@"
