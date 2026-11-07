#!/bin/bash
#SBATCH --job-name=brainclip_eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907

source ~/.bashrc
conda activate biomag_0530

python eval_alignment.py --config configs/default.yaml --checkpoint checkpoints/best.pt
