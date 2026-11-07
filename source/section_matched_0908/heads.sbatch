#!/bin/bash
#SBATCH --job-name=sec_exact_head
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-11%1
#SBATCH --output=logs/section_matched_heads_%A_%a.out
#SBATCH --error=logs/section_matched_heads_%A_%a.err
set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
splits=(holdout_45 holdout_12)
modes=(mlp attention)
split="${splits[$((SLURM_ARRAY_TASK_ID / 6))]}"
mode="${modes[$(((SLURM_ARRAY_TASK_ID % 6) / 3))]}"
seed=$((42 + SLURM_ARRAY_TASK_ID % 3))
config="section_matched_0908/${split}_${mode}.yaml"
checkpoint_dir="checkpoints/section_matched_0908/${split}/${mode}/s${seed}"
python section_matched_0908/verify_sources.py
python section_matched_0908/export_result.py --config "$config" --seed "$seed" --check-lineage
python train_subject_pair_exhaustive.py \
  --config "$config" --seed "$seed" --checkpoint-dir "$checkpoint_dir"
python section_matched_0908/export_result.py --config "$config" --seed "$seed" \
  --checkpoint "$checkpoint_dir/best.pt" \
  --output "results/section_matched_0908/${split}/${mode}/s${seed}/result.json"
