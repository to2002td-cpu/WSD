#!/bin/bash
#OAR -q abaca
#OAR -l {gpu_mem>32000}/gpu=1,walltime=10:00:00
#OAR -p gpu_model NOT IN ('Tesla P100-PCIE-16GB', 'Quadro P6000')
#OAR -n annotate_part
#OAR -O annotate_part.%jobid%.out
#OAR -E annotate_part.%jobid%.err

set -e

part=$1
if [ -z "$part" ]; then
  echo "usage: pass the part number, e.g. oarsub -S './job_annotate_part.sh 0'"
  exit 1
fi

workdir=$HOME/consec_wsd
cd "$workdir"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate consec_wsd

export CONSEC_REPO=$workdir/consec
export TRANSFORMERS_OFFLINE=1

python annotate.py --occurrences big_annotation/occ_part_${part}.jsonl --out big_annotation/annotations_part_${part}.jsonl --checkpoint consec_semcor.ckpt --device 0
