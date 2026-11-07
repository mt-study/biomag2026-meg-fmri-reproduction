#!/bin/bash
#SBATCH --job-name=sec_exact_enc
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/section_matched_encoder_%A_%a.out
#SBATCH --error=logs/section_matched_encoder_%A_%a.err
set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
splits=(holdout_45 holdout_12)
python section_matched_0908/verify_sources.py
python section_matched_0908/train_encoder.py \
  --config "section_matched_0908/${splits[$SLURM_ARRAY_TASK_ID]}_encoder.yaml" \
  --protocol all_subjects --data-mode exhaustive --seed 42
