#!/bin/bash
#SBATCH --job-name=bc_mtwin_no_cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/meg_text_window_nonoverlap_cache_%j.out
#SBATCH --error=logs/meg_text_window_nonoverlap_cache_%j.err

set -eo pipefail
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

for window_seconds in 2 4 6 8; do
  cache_dir="cache/meg_text_window_nonoverlap/w${window_seconds}_stride${window_seconds}"
  echo "[cache] MEG/text=${window_seconds}s, stride=${window_seconds}s, fMRI delays=2/4/6/8s"
  python scripts/build_subject_pair_window_cache.py \
    --config configs/meg_text_window_nonoverlap_cache.yaml \
    --window-seconds "${window_seconds}" \
    --stride-seconds "${window_seconds}" \
    --grid-window-bound-seconds "${window_seconds}" \
    --grid-delay-bound-seconds 8 \
    --grid-text-presence-seconds "${window_seconds}" \
    --text-window-variants "${window_seconds}" \
    --window-cache "${cache_dir}"
done
