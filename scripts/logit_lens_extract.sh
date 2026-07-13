#!/bin/bash
# Stage 1 (GPU, the long one): logit-lens logprobs for every checkpoint x layer.
# Resumable per checkpoint -- re-submit to continue where a previous run left off.
# Usage:  oarsub -S "./scripts/logit_lens_extract.sh"
#OAR -q production
#OAR -l {gpu_mem > 20000}/host=1/gpu=1,walltime=48:0:0
#OAR -O .logit_lens_extract.logs
#OAR -E .logit_lens_extract.errors
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

echo "OAR job ${OAR_JOB_ID:-?} on $(hostname) -- logit_lens extract (GPU)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  || echo "WARNING: no GPU visible"

# HF cache defaults to node-local /tmp (see logit_lens/config.py:hf_cache_dir) so a
# ~14GB Pythia snapshot never threatens the small NFS home quota.
mkdir -p "${TMPDIR:-/tmp}"
df -h "${TMPDIR:-/tmp}" || true

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync --extra logit-lens
exec uv run python -m logit_lens extract "$@"
