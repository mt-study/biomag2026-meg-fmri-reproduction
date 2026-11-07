#!/bin/bash
#SBATCH --job-name=brainclip_cc
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
# Note: no --mem here (see scripts/smoke_test.sbatch for why).
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907

source ~/.bashrc
conda activate biomag_0530

python train.py --config configs/default.yaml
