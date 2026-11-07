#!/bin/bash
#SBATCH --job-name=meg_stats
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/meg_stats_%j.out
#SBATCH --error=logs/meg_stats_%j.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907

source ~/.bashrc
conda activate biomag_0530

python scripts/build_meg_stats.py --config configs/default.yaml
