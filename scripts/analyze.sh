#!/bin/bash
# Stage 3 (CPU): k-NN purity over the corpus + aggregate. Idempotent.
# Usage:  oarsub -S "./scripts/analyze.sh --model pythia-6.9b"   (add --pca for PCA grids)
#OAR -q production
#OAR -l host=1/core=16,walltime=12:0:0
#OAR -O .analyze.logs
#OAR -E .analyze.errors
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

echo "OAR job ${OAR_JOB_ID:-?} on $(hostname) -- analyze (purity, CPU)"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync --extra plot
exec uv run python scripts/analyze_all.py "$@"
