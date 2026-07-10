#!/usr/bin/env bash
# Analysis: per-word UMAP scatter + per-sense KDE grid.
#   ./scripts/density.sh bank --pos n --only bank.n
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
uv sync --extra plot
exec uv run python -m src density "$@"
