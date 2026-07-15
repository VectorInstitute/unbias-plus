#!/bin/bash
#SBATCH --job-name=unbias_recipe_sft
#SBATCH --output=logs/recipe_sft_%j.out
#SBATCH --error=logs/recipe_sft_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# Generic launcher for any recipe. Pass the recipe name (and any extra flags)
# as arguments, e.g.:
#   sbatch launch.slurm bias_weighted
#   sbatch launch.slurm baseline --train-samples 20 --epochs 1 --load-in-4bit
#
# The first argument is the recipe name; anything after it is forwarded to
# `python -m recipes.train` verbatim.

set -euo pipefail

RECIPE="${1:?Usage: sbatch launch.slurm <recipe> [extra train.py flags]}"
shift || true
EXTRA_ARGS=("$@")

# Resolve paths relative to this script so nothing machine-specific is hardcoded.
# Layout: <project-root>/training/recipes/launch.slurm
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "${SCRIPT_DIR}")"
PROJECT_ROOT="$(dirname "${TRAINING_DIR}")"

# Run from the training/ directory so `recipes` resolves as a package.
cd "${TRAINING_DIR}"

mkdir -p logs

# Configurable via env; override to match your cluster (e.g. `export CUDA_MODULE=...`).
module load "${CUDA_MODULE:-cuda/12.6}"
source "${VENV_PATH:-${PROJECT_ROOT}/.venv}/bin/activate"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
# Point the HF cache somewhere with quota; set HF_HOME before submitting if needed.
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"

echo "============================================"
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "Recipe     : ${RECIPE}"
echo "Extra args : ${EXTRA_ARGS[*]:-(none)}"
echo "Start time : $(date)"
echo "============================================"

accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    --main_process_port=29500 \
    -m recipes.train --recipe "${RECIPE}" "${EXTRA_ARGS[@]}"

EXIT_CODE=$?

echo "============================================"
echo "End time   : $(date)"
echo "Exit code  : ${EXIT_CODE}"
echo "============================================"

exit $EXIT_CODE
