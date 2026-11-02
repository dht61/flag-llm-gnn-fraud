#!/bin/bash
#SBATCH --job-name=flag_prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
source scripts/common.sh
python prep_data.py --ratio implied --train_ratio 0.4
python prep_data.py --ratio 1:10 --train_ratio 0.4   # the stated ratio, for comparison
