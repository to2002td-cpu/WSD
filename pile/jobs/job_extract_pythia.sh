#!/bin/bash
#OAR -l core=2,walltime=12:00:00
#OAR -n extract_pythia
#OAR -O extract_pythia.%jobid%.out
#OAR -E extract_pythia.%jobid%.err

set -e

workdir=$HOME/consec_wsd
cd "$workdir"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate extract_env

python extract.py --targets top100Lemmas.txt \
  --out pythia_occurrences.jsonl \
  --source pythia \
  --bin-path /srv/storage/linkmedia@storage2.rennes.fr/gdelicat/pythia_bin/document-00000-of-00020.bin \
  --tokenizer $HOME/pythia_repo/utils/20B_tokenizer.json \
  --max-docs 200000 --cap 34000 --n-process 1
