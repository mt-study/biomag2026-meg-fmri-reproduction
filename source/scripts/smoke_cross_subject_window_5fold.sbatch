#!/bin/bash
#SBATCH --job-name=bc_unseen_win_smoke
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/cross_subject_window_smoke_%j.out
#SBATCH --error=logs/cross_subject_window_smoke_%j.err

set -eo pipefail
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

# One epoch per stage on fold 0 of both conditions. Writes only under
# checkpoints/results roots suffixed with smoke_, never the real ones.
for window_seconds in 2 4; do
    echo "[smoke] window=${window_seconds}s fold=0"
    python train_cross_subject_generalization.py \
        --config "configs/cross_subject_window_5fold_w${window_seconds}.yaml" \
        --fold 0 \
        --encoder-epochs 1 \
        --fusion-epochs 1 \
        --max-train-windows 64 \
        --checkpoint-root "checkpoints/smoke_cross_subject_window_5fold/w${window_seconds}" \
        --result-root "results/smoke_cross_subject_window_5fold/w${window_seconds}"
done
