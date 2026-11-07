#!/bin/bash
#SBATCH --job-name=bc_1529545_unseen
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-9%2
#SBATCH --output=logs/unified_1529545_unseen_%A_%a.out
#SBATCH --error=logs/unified_1529545_unseen_%A_%a.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

modes=(mean exhaustive)
mode_index=$((SLURM_ARRAY_TASK_ID / 5))
fold=$((SLURM_ARRAY_TASK_ID % 5))
seed=$((42 + fold))

python unified_benchmark.py \
    --config configs/unified_benchmark_task1529545_matched.yaml \
    --protocol unseen_subject \
    --data-mode "${modes[$mode_index]}" \
    --fold "$fold" \
    --seed "$seed"
