#!/bin/bash
# Pool target-word hidden states from Pythia checkpoints.
# Submit with:  oarsub -S ./scripts/extract.sh
#
# Extraction is resumable per checkpoint (each step*.npz is skipped if present),
# so if the walltime is hit mid-run, re-submit to continue with the next
# checkpoint. One .npz over the full top-100 corpus is large (~tens of GB); make
# sure storage_root points at group storage with room for ~10 of them.
#OAR -q production
#OAR -l {gpu_mem > 20000}/host=1/gpu=1,walltime=24:0:0
#OAR -O .extract.logs
#OAR -E .extract.errors
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

echo "OAR job ${OAR_JOB_ID:-?} on $(hostname) -- extract (GPU)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  || echo "WARNING: no GPU visible"

# Persistent HF cache on group storage (reused across jobs; see extract.cache_dir).
export HF_HOME="${HF_HOME:-/home/tderrien/storage/cache/huggingface}"
mkdir -p "$HF_HOME"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync
exec uv run python -m src extract "$@"
