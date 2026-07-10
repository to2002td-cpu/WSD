#!/usr/bin/env bash
# Analysis: pairwise sense MMD kernel two-sample test + separation heatmap.
#   ./scripts/mmd.sh bank --pos n --only bank.n
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
uv sync --extra plot
exec uv run python -m src mmd "$@"
