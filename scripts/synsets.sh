#!/usr/bin/env bash
# Build synsets.json from lemmas.txt
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
uv sync
exec uv run python -m src synsets "$@"
