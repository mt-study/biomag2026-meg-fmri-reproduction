#!/bin/bash
#SBATCH --job-name=bc_joint_ft_mlp
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/unified_joint_finetune_legacy_mlp_%A_%a.out
#SBATCH --error=logs/unified_joint_finetune_legacy_mlp_%A_%a.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

seed=$((42 + SLURM_ARRAY_TASK_ID))
checkpoint_dir="checkpoints/unified_joint_finetune_legacy_mlp/s$seed"

echo "[fixed initialization] checkpoints/unified_benchmark_task1529545_matched/all_subjects/exhaustive/seed_42/encoder_best.pt"
echo "[joint fine-tuning] legacy MLP hidden=512 objective=joint_text_only seed=$seed"
python train_subject_pair_exhaustive.py \
    --config configs/unified_joint_finetune_legacy_mlp.yaml \
    --seed "$seed" \
    --checkpoint-dir "$checkpoint_dir"
