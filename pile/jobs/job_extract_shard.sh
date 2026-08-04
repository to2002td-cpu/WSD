#!/bin/bash
#OAR -l core=2,walltime=12:00:00
#OAR -n extract_shard
#OAR -O extract_shard.%jobid%.out
#OAR -E extract_shard.%jobid%.err

set -e

shard=$1
if [ -z "$shard" ]; then
  echo "usage: pass the shard name, e.g. 00"
  exit 1
fi

workdir=$HOME/consec_wsd
cd "$workdir"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate extract_env

python extract.py --targets top100Lemmas.txt --out big_annotation/occ_shard_${shard}.jsonl --source local --pile-dir pile_shards/train/${shard}.jsonl.zst --cap 34000 --max-docs 200000 --n-process 1
