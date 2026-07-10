#!/usr/bin/env bash
# Submit the two OAR jobs: generate (CPU), then extract (GPU) once it succeeds.
# The GPU job carries an OAR dependency (-a) so the reservation is not held
# while the API generation runs.
#
#   ./scripts/submit.sh            # both, extract after generate
#   ./scripts/submit.sh generate   # only stage 1
#   ./scripts/submit.sh extract    # only stage 2
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

stage="${1:-all}"

storage=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml')).get('storage_root'))")
echo "storage_root = ${WSD_STORAGE_ROOT:-$storage}"

submit() { oarsub "$@" | sed -n 's/^OAR_JOB_ID=//p'; }

gen=""
if [[ "$stage" == "all" || "$stage" == "generate" ]]; then
  gen=$(submit -S ./scripts/generate.sh)
  echo "stage 1 generate -> job $gen"
fi

if [[ "$stage" == "all" || "$stage" == "extract" ]]; then
  if [[ -n "$gen" ]]; then
    ext=$(submit -a "$gen" -S ./scripts/extract.sh)
    echo "stage 2 extract  -> job $ext (after $gen)"
  else
    ext=$(submit -S ./scripts/extract.sh)
    echo "stage 2 extract  -> job $ext"
  fi
fi

echo "Track with: oarstat -u"
