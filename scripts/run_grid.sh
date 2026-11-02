#!/bin/bash
#SBATCH --job-name=flag_grid
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%A_%a.out
# one line of grids/$GRID.txt per task ("dataset backbone variant seed"), 5 inits each. Finished runs are skipped.
#   sbatch --array=0-199%16 --export=ALL,GRID=phase2 scripts/run_grid.sh
#   sbatch --array=0-34 --export=ALL,GRID=star_ig,EXTRA="--disc_tag _ft_s{seed}" scripts/run_grid.sh
set -eo pipefail
source scripts/common.sh
read -r DS BB VAR SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" grids/$GRID.txt)
for INIT in 0 1 2 3 4; do
  python run_flag.py --dataset $DS --backbone $BB --variant $VAR --seed $SEED --init $INIT ${EXTRA:-} \
    --out results/main/$DS/$BB/$VAR/s${SEED}_i${INIT} | grep -E "^\[result\]|^\[skip\]"
done
