#!/bin/bash
# Stage 2 (GPU): pool hidden states. Resumable per checkpoint; re-submit to
# continue. Usage:  oarsub -S "./scripts/extract.sh --model pythia-6.9b"
#OAR -q production
#OAR -l {gpu_mem > 20000}/host=1/gpu=1,walltime=24:0:0
#OAR -O .extract.logs
#OAR -E .extract.errors
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

echo "OAR job ${OAR_JOB_ID:-?} on $(hostname) -- extract (GPU)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  || echo "WARNING: no GPU visible"

export HF_HOME="${HF_HOME:-/home/tderrien/storage/cache/huggingface}"   # persistent HF cache
mkdir -p "$HF_HOME"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync
exec uv run python -m src extract "$@"
