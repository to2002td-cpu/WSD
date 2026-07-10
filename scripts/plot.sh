#!/usr/bin/env bash
# Analysis: per-word UMAP grid (local).
#   ./scripts/plot.sh bank --pos n
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
uv sync --extra plot
exec uv run python -m src plot "$@"
