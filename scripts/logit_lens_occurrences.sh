#!/bin/bash
# Stage 0 (CPU, quick): SemCor -> occ_df.pkl / targets_df.csv. Idempotent -- safe
# to re-submit, it just reuses the cached tables.
# Usage:  oarsub -S "./scripts/logit_lens_occurrences.sh"
#OAR -q production
#OAR -l host=1/core=4,walltime=2:0:0
#OAR -O .logit_lens_occurrences.logs
#OAR -E .logit_lens_occurrences.errors
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

echo "OAR job ${OAR_JOB_ID:-?} on $(hostname) -- logit_lens occurrences (CPU)"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync --extra logit-lens
exec uv run python -m logit_lens occurrences "$@"
