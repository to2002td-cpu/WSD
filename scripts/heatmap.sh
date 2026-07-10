#!/usr/bin/env bash
# Analysis: per-word UMAP density (thermal) grid.
#   ./scripts/heatmap.sh bank --pos n --only bank.n
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
uv sync --extra plot
exec uv run python -m src heatmap "$@"
