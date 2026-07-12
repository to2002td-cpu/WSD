#!/usr/bin/env bash
# Build synsets.json from the lemma list.  ./scripts/synsets.sh --lemmas nouns_top50
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
uv sync
exec uv run python -m src synsets "$@"
