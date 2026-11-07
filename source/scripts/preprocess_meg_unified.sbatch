#!/bin/bash
#SBATCH --job-name=meg_unify
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=1-20
#SBATCH --output=logs/meg_unify_%A_%a.out
#SBATCH --error=logs/meg_unify_%A_%a.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907

source ~/.bashrc
conda activate biomag_0530

SUBJECT=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
python scripts/preprocess_meg_unified.py --config configs/default.yaml --subject "$SUBJECT"
