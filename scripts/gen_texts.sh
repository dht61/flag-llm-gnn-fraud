#!/bin/bash
#SBATCH --job-name=flag_gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
# SBERT embeddings + Llama-3-8B disc/res texts for all nodes (~45 min on one A100).
# LIMIT=200 sbatch scripts/gen_texts.sh  -> quick test on the first 200 nodes
set -eo pipefail
source scripts/common.sh
python gen_texts.py --limit ${LIMIT:-0}
