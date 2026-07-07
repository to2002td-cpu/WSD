#!/usr/bin/env bash
# Submit the full pythia-6.9b experiment as an OAR job (Grid'5000).
#
#   ./scripts/submit_oar.sh                       # defaults: 1 GPU, 24h, production queue
#   ./scripts/submit_oar.sh --batch-size 16       # extra args go to `wsd_probe all`
#
# Tunables (environment variables):
#   OAR_QUEUE      queue name                 (default: production)
#   OAR_RESOURCES  -l resource expression     (default: host=1/gpu=1,walltime=24:00:00)
#   OAR_PROPERTY   -p property filter, e.g. "gpu_model LIKE 'A100%'" (default: none)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

QUEUE="${OAR_QUEUE:-production}"
RESOURCES="${OAR_RESOURCES:-host=1/gpu=1,walltime=24:00:00}"
PROPERTY="${OAR_PROPERTY:-}"

mkdir -p oar_logs
chmod +x scripts/oar_job.sh

args=(
  -n wsd-probe
  -q "$QUEUE"
  -l {ressources.gpu_mem>20000}host=1/gpu=1,walltime=24:00:00}
  -O "oar_logs/wsd-probe.%jobid%.out"
  -E "oar_logs/wsd-probe.%jobid%.err"
)
[ -n "$PROPERTY" ] && args+=(-p "$PROPERTY")

oarsub "${args[@]}" "$PWD/scripts/oar_job.sh $*"

echo
echo "Follow up with:"
echo "  oarstat -u                # job state"
echo "  tail -f oar_logs/wsd-probe.<jobid>.out   # progress bars / logs"
echo "  oardel <jobid>            # cancel"
