#!/usr/bin/env bash
# Per-word per-condition PCA scatter grid (PDF + PNG).
#   ./scripts/pca.sh bank --model pythia-6.9b
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
uv sync --extra plot
exec uv run python -m src pca "$@"
