#!/usr/bin/env bash
# Chain the logit-lens pipeline as OAR jobs (each waits on the previous via -a):
#   occurrences (CPU) -> extract (GPU, overnight) -> aggregate (CPU)
#
#   ./scripts/logit_lens_submit.sh [all|occurrences|extract|aggregate] -- [passthrough args]
#   ./scripts/logit_lens_submit.sh all                       # full chain
#   ./scripts/logit_lens_submit.sh extract -- --force        # just re-run extraction
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

stage="${1:-all}"; shift || true
[[ "${1:-}" == "--" ]] && shift || true
args=("$@")

submit() { oarsub "$@" | sed -n 's/^OAR_JOB_ID=//p'; }
want() { [[ "$stage" == "all" || "$stage" == "$1" ]]; }
job() { printf '%s ' "$@"; }

occ="" ext=""
if want occurrences; then
  occ=$(submit -S "$(job ./scripts/logit_lens_occurrences.sh "${args[@]}")")
  echo "stage 0 occurrences -> job $occ"
fi
if want extract; then
  dep=(); [[ -n "$occ" ]] && dep=(-a "$occ")
  ext=$(submit "${dep[@]}" -S "$(job ./scripts/logit_lens_extract.sh "${args[@]}")")
  echo "stage 1 extract     -> job $ext ${occ:+(after $occ)}"
fi
if want aggregate; then
  dep=(); [[ -n "$ext" ]] && dep=(-a "$ext")
  agg=$(submit "${dep[@]}" -S "$(job ./scripts/logit_lens_aggregate.sh "${args[@]}")")
  echo "stage 2 aggregate   -> job $agg ${ext:+(after $ext)}"
fi
echo "Track with: oarstat -u"
