#!/bin/bash
# Sense-separation analysis over the whole corpus: pairwise MMD kernel two-sample
# test per word, then a corpus-level aggregate heatmap. CPU only; reads the
# shared extracted embeddings. Idempotent (per-word outputs overwrite), so a
# re-submit after a timeout simply recomputes.
# Submit with:  oarsub -S ./scripts/analyze.sh
#     add KDE:  oarsub -S "./scripts/analyze.sh --kde"
#
#OAR -q production
#OAR -l host=1/core=16,walltime=12:0:0
#OAR -O .analyze.logs
#OAR -E .analyze.errors
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

echo "OAR job ${OAR_JOB_ID:-?} on $(hostname) -- analyze (MMD, CPU)"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync --extra plot
exec uv run python scripts/analyze_all.py "$@"
