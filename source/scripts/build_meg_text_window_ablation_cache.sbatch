#!/bin/bash
#SBATCH --job-name=bc_mtwin_cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/meg_text_window_cache_%j.out
#SBATCH --error=logs/meg_text_window_cache_%j.err

set -eo pipefail
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

echo "[shared cache] MEG=8s, text variants=2/4/6/8s, stride=2s, fMRI delays fixed"
python scripts/build_subject_pair_window_cache.py     --config configs/meg_text_window_ablation_cache.yaml