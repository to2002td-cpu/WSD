#!/usr/bin/env bash
# Submit the pipeline as chained OAR jobs, each starting only once the previous
# one succeeds (OAR dependency -a), so a reservation is never held while an
# earlier stage runs:
#
#   generate (CPU, API)  ->  extract (GPU)  ->  analyze (CPU, purity + aggregate)
#
#   ./scripts/submit.sh            # all three, chained
#   ./scripts/submit.sh generate   # only stage 1
#   ./scripts/submit.sh extract    # only stage 2
#   ./scripts/submit.sh analyze    # only stage 3
#   ./scripts/submit.sh gen-ext    # generate + extract (no analysis)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

stage="${1:-all}"

storage=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml')).get('storage_root'))")
echo "storage_root = ${WSD_STORAGE_ROOT:-$storage}"

submit() { oarsub "$@" | sed -n 's/^OAR_JOB_ID=//p'; }

want() { [[ "$stage" == "all" || "$stage" == "$1" || "$stage" == "$2" ]]; }

gen="" ext=""
if want generate gen-ext; then
  gen=$(submit -S ./scripts/generate.sh)
  echo "stage 1 generate -> job $gen"
fi

if want extract gen-ext; then
  if [[ -n "$gen" ]]; then
    ext=$(submit -a "$gen" -S ./scripts/extract.sh)
    echo "stage 2 extract  -> job $ext (after $gen)"
  else
    ext=$(submit -S ./scripts/extract.sh)
    echo "stage 2 extract  -> job $ext"
  fi
fi

if want analyze _; then
  if [[ -n "$ext" ]]; then
    ana=$(submit -a "$ext" -S ./scripts/analyze.sh)
    echo "stage 3 analyze  -> job $ana (after $ext)"
  else
    ana=$(submit -S ./scripts/analyze.sh)
    echo "stage 3 analyze  -> job $ana"
  fi
fi

echo "Track with: oarstat -u"
