#!/bin/bash
#SBATCH --job-name=flag_embed
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# optional: raw-text SBERT on CPU (gen_texts.py skips it if the file exists)
set -eo pipefail
source scripts/common.sh
python embed_raw.py
