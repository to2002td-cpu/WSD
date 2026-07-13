#!/bin/bash
# Stage 2 (CPU): merge per-checkpoint pickles into a tidy word x layer x step
# summary. Idempotent -- cheap to re-run after `extract` picks up more checkpoints.
# Usage:  oarsub -S "./scripts/logit_lens_aggregate.sh"
#OAR -q production
#OAR -l host=1/core=4,walltime=1:0:0
#OAR -O .logit_lens_aggregate.logs
#OAR -E .logit_lens_aggregate.errors
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

echo "OAR job ${OAR_JOB_ID:-?} on $(hostname) -- logit_lens aggregate (CPU)"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync --extra logit-lens
exec uv run python -m logit_lens aggregate "$@"
