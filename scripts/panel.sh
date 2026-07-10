#!/usr/bin/env bash
# Analysis: single large publication panel (one layer + checkpoint).
#   ./scripts/panel.sh bank --pos n --only bank.n --layer 16 --step 143000
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
uv sync --extra plot
exec uv run python -m src panel "$@"
