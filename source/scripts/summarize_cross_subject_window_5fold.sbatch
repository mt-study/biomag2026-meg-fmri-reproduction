#!/bin/bash
#SBATCH --job-name=bc_unseen_win_sum
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/cross_subject_window_summary_%j.out
#SBATCH --error=logs/cross_subject_window_summary_%j.err

set -eo pipefail
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

python summarize_cross_subject_window_ablation.py
