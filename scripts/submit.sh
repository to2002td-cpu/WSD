#!/usr/bin/env bash
# Chain the pipeline as OAR jobs (each waits on the previous via -a):
#   generate (CPU) -> extract (GPU) -> analyze (CPU)
#
#   ./scripts/submit.sh [all|generate|extract|analyze|gen-ext] -- [passthrough args]
#   ./scripts/submit.sh all -- --model pythia-6.9b --lemmas nouns_top50
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

stage="${1:-all}"; shift || true
[[ "${1:-}" == "--" ]] && shift || true
args=("$@")                                   # forwarded to every stage

storage=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/default.yaml')).get('storage_root'))")
echo "storage_root = ${WSD_STORAGE_ROOT:-$storage}"

submit() { oarsub "$@" | sed -n 's/^OAR_JOB_ID=//p'; }
want() { [[ "$stage" == "all" || "$stage" == "$1" || "$stage" == "$2" ]]; }
job() { printf '%s ' "$@"; }                  # build the "script args..." command string

gen="" ext=""
if want generate gen-ext; then
  gen=$(submit -S "$(job ./scripts/generate.sh "${args[@]}")")
  echo "stage 1 generate -> job $gen"
fi
if want extract gen-ext; then
  dep=(); [[ -n "$gen" ]] && dep=(-a "$gen")
  ext=$(submit "${dep[@]}" -S "$(job ./scripts/extract.sh "${args[@]}")")
  echo "stage 2 extract  -> job $ext ${gen:+(after $gen)}"
fi
if want analyze _; then
  dep=(); [[ -n "$ext" ]] && dep=(-a "$ext")
  ana=$(submit "${dep[@]}" -S "$(job ./scripts/analyze.sh "${args[@]}")")
  echo "stage 3 analyze  -> job $ana ${ext:+(after $ext)}"
fi
echo "Track with: oarstat -u"
