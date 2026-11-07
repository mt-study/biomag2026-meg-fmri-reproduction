#!/bin/bash
#SBATCH --job-name=bc_unseen_attn
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/cross_subject_attention_5fold_%A_%a.out
#SBATCH --error=logs/cross_subject_attention_5fold_%A_%a.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530
python train_cross_subject_attention_generalization.py \
    --config configs/cross_subject_attention_5fold.yaml \
    --fold "$SLURM_ARRAY_TASK_ID"
