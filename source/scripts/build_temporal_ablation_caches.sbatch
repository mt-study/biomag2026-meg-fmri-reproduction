#!/bin/bash
#SBATCH --job-name=bc_time_cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/temporal_cache_%A_%a.out
#SBATCH --error=logs/temporal_cache_%A_%a.err

set -eo pipefail
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

windows=(1 2 4)
labels=(w1_dall w2_dall w4_dall)
window_seconds="${windows[$SLURM_ARRAY_TASK_ID]}"
label="${labels[$SLURM_ARRAY_TASK_ID]}"

echo "[temporal cache] condition=${label} window=${window_seconds}s stride=4s bounds=4s+8s text_presence=1s"
python scripts/build_subject_pair_window_cache.py     --config configs/temporal_ablation_cache.yaml     --window-seconds "${window_seconds}"     --window-cache "cache/temporal_ablation/${label}"