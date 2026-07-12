#!/usr/bin/env bash
# Push generated sentences to a HF dataset repo.
#   HF_TOKEN=hf_... ./scripts/push.sh --repo-id you/wsd-sentences
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
uv sync
exec uv run python -m src push "$@"
