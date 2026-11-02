#!/bin/bash
#SBATCH --job-name=flag_w2v
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
source scripts/common.sh
python w2v_features.py dump
python w2v_features.py train
