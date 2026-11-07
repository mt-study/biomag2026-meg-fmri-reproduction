#!/bin/bash
#SBATCH --job-name=window_cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/window_cache_%j.out
#SBATCH --error=logs/window_cache_%j.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907

source ~/.bashrc
conda activate biomag_0530

python scripts/build_window_cache.py --config configs/default.yaml
