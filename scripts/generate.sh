#!/bin/bash
# Synthesize sentences via the chat API.
# Submit with:  oarsub -S ./scripts/generate.sh
#
#OAR -q production
#OAR -l host=1/core=8,walltime=24:0:0
#OAR -O .generate.logs
#OAR -E .generate.errors
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

echo "OAR job ${OAR_JOB_ID:-?} on $(hostname) -- generate (CPU)"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync
exec uv run python -m src generate "$@"
