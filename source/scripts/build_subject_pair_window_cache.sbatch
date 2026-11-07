#!/bin/bash
#SBATCH --job-name=spair_cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/subject_pair_cache_%j.out
#SBATCH --error=logs/subject_pair_cache_%j.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907

source ~/.bashrc
conda activate biomag_0530

python scripts/build_subject_pair_window_cache.py --config configs/subject_pair.yaml
