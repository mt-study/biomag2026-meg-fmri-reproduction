#!/bin/bash
#SBATCH --job-name=bc_legacy_smoke
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=logs/unified_legacy_smoke_%A_%a.out
#SBATCH --error=logs/unified_legacy_smoke_%A_%a.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

modes=(mean exhaustive)

python unified_benchmark.py \
    --config configs/unified_benchmark_legacy_matched.yaml \
    --protocol all_subjects \
    --data-mode "${modes[$SLURM_ARRAY_TASK_ID]}" \
    --seed 42 \
    --encoder-epochs 1 \
    --fusion-epochs 1 \
    --checkpoint-root checkpoints/unified_benchmark_legacy_matched_smoke \
    --result-root results/unified_benchmark_legacy_matched_smoke
