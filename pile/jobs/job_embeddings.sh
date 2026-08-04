#!/bin/bash
#OAR -l gpu=1,walltime=8:00:00
#OAR -p gpu_model = 'Tesla V100-SXM2-32GB'
#OAR -n pile_embeddings
#OAR -O pile_emb.%jobid%.out
#OAR -E pile_emb.%jobid%.err

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

steps=$1
if [ -z "$steps" ]; then
  echo "usage: oarsub -S './pile_geometry/job_embeddings.sh 0,512,1000'"
  exit 1
fi

workdir=$HOME/consec_wsd
cd "$workdir"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wsd_env

storage=/srv/storage/linkmedia@storage2.rennes.grid5000.fr/gdelicat

python pile_geometry/extract_embeddings.py \
  --dataset pile_dataset.jsonl \
  --out-dir $storage/wsd_geometry/embeddings/pythia-6.9b \
  --steps "$steps" \
  --layers 1,8,16,24,32 \
  --batch-size 16 \
  --max-length 256 \
  --cache-dir $storage/hf_cache/hub
