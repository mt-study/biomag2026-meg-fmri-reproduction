#!/bin/bash
#SBATCH --job-name=text_cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/text_cache_%j.out
#SBATCH --error=logs/text_cache_%j.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907

source ~/.bashrc
conda activate biomag_0530

python scripts/build_text_cache.py --config configs/default.yaml
