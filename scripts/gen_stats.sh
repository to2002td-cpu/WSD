#!/usr/bin/env bash
# Live stats on the generated corpus: progress toward per_synset targets,
# per-lemma / per-(sense, style) balance, and a live generation rate.
#   ./scripts/gen_stats.sh --lemmas top100Lemmas --model pythia-6.9b
#   ./scripts/gen_stats.sh --lemmas top100Lemmas --watch 15    # refresh every 15s
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

watch_secs=""
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch) watch_secs="${2:-10}"; shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done

if [[ -n "$watch_secs" ]]; then
  while true; do
    clear
    date
    echo
    uv run python scripts/gen_stats.py "${args[@]}"
    sleep "$watch_secs"
  done
else
  uv run python scripts/gen_stats.py "${args[@]}"
fi
