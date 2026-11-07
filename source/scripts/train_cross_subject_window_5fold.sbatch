#!/bin/bash
#SBATCH --job-name=bc_unseen_win
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-9%2
#SBATCH --output=logs/cross_subject_window_5fold_%A_%a.out
#SBATCH --error=logs/cross_subject_window_5fold_%A_%a.err

set -eo pipefail
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

# Ten runs: 2 s and 4 s MEG+text windows x five subject-disjoint folds.
windows=(2 4)
window_seconds="${windows[$((SLURM_ARRAY_TASK_ID / 5))]}"
fold=$((SLURM_ARRAY_TASK_ID % 5))

echo "[strict unseen-subject] window=${window_seconds}s fold=${fold}; sections 1-6 train, 7 select, 8-9 unseen test"
python train_cross_subject_generalization.py \
    --config "configs/cross_subject_window_5fold_w${window_seconds}.yaml" \
    --fold "${fold}"
