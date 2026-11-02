#!/bin/bash
#SBATCH --job-name=flag_ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
# FLAG* fine-tuning, one (dataset, seed) per task from grids/ft.txt. 20 min - 1.5 h each.
set -eo pipefail
source scripts/common.sh
read -r DS SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" grids/ft.txt)
python finetune_llm.py all --dataset $DS --seed $SEED
