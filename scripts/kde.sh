#!/usr/bin/env bash
# Analysis: interactive per-layer KDE slider over checkpoints (self-contained HTML).
#   ./scripts/kde.sh bank --pos n --only bank.n
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
uv sync --extra plot
exec uv run python -m src kde "$@"
