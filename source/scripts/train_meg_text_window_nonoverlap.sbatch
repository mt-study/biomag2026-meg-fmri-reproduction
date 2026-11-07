#!/bin/bash
#SBATCH --job-name=bc_mtwin_no
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/meg_text_window_nonoverlap_%A_%a.out
#SBATCH --error=logs/meg_text_window_nonoverlap_%A_%a.err

set -eo pipefail
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

# Each condition uses its own non-overlapping start grid. Candidate counts are
# therefore expected to decrease as the window grows: 2s > 4s > 6s > 8s.
windows=(2 4 6 8)
window_index=$((SLURM_ARRAY_TASK_ID / 3))
seed=$((42 + SLURM_ARRAY_TASK_ID % 3))
window_seconds="${windows[$window_index]}"
condition="w${window_seconds}"

cache_dir="cache/meg_text_window_nonoverlap/${condition}_stride${window_seconds}"

echo "[MEG+text non-overlap] condition=${condition} seed=${seed}; stride=${window_seconds}s; fMRI delays=2/4/6/8s"
python unified_benchmark.py \
  --config configs/meg_text_window_nonoverlap.yaml \
  --protocol all_subjects \
  --data-mode exhaustive \
  --seed "${seed}" \
  --window-cache "${cache_dir}" \
  --meg-text-window-seconds "${window_seconds}" \
  --checkpoint-root "checkpoints/meg_text_window_nonoverlap/${condition}" \
  --result-root "results/meg_text_window_nonoverlap/${condition}"
