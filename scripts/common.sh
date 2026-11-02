# sourced by the job scripts (run them from the repo root)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flag
export FLAG_DATA=${FLAG_DATA:-$PWD/data}
export DGLBACKEND=pytorch PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p logs
