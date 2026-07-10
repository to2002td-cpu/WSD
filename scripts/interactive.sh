#!/usr/bin/env bash
# Analysis: interactive Plotly UMAP (self-contained HTML).
#   ./scripts/interactive.sh bank --pos n --only bank.n
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
uv sync --extra plot
exec uv run python -m src interactive "$@"
