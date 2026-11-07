#!/bin/bash
#SBATCH --job-name=bc_mtwin
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/meg_text_window_ablation_%A_%a.out
#SBATCH --error=logs/meg_text_window_ablation_%A_%a.err

set -eo pipefail
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

windows=(2 4 6 8)
window_index=$((SLURM_ARRAY_TASK_ID / 3))
seed=$((42 + SLURM_ARRAY_TASK_ID % 3))
window_seconds="${windows[$window_index]}"
condition="w${window_seconds}"

echo "[MEG+text window] condition=${condition} seed=${seed}; fMRI delays=2/4/6/8s"
python unified_benchmark.py     --config configs/meg_text_window_ablation.yaml     --protocol all_subjects     --data-mode exhaustive     --seed "${seed}"     --meg-text-window-seconds "${window_seconds}"     --checkpoint-root "checkpoints/meg_text_window_ablation/${condition}"     --result-root "results/meg_text_window_ablation/${condition}"