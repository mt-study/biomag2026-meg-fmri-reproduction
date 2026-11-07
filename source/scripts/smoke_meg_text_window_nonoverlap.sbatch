#!/bin/bash
#SBATCH --job-name=bc_mtwin_no_smoke
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/meg_text_window_nonoverlap_smoke_%j.out
#SBATCH --error=logs/meg_text_window_nonoverlap_smoke_%j.err

set -eo pipefail
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

for window_seconds in 2 4 6 8; do
  condition="w${window_seconds}"
  cache_dir="cache/meg_text_window_nonoverlap/${condition}_stride${window_seconds}"
  echo "[smoke] ${condition}: non-overlapping ${window_seconds}s grid; one encoder/fusion epoch"
  python unified_benchmark.py \
    --config configs/meg_text_window_nonoverlap.yaml \
    --protocol all_subjects \
    --data-mode exhaustive \
    --seed 42 \
    --window-cache "${cache_dir}" \
    --meg-text-window-seconds "${window_seconds}" \
    --max-train-windows 64 \
    --encoder-epochs 1 \
    --fusion-epochs 1 \
    --checkpoint-root "checkpoints/smoke_meg_text_window_nonoverlap/${condition}" \
    --result-root "results/smoke_meg_text_window_nonoverlap/${condition}"
done
