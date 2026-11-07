#!/bin/bash
#SBATCH --job-name=bc_sp_headcmp
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/exhaustive_head_compare_%A_%a.out
#SBATCH --error=logs/exhaustive_head_compare_%A_%a.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

if [ "$SLURM_ARRAY_TASK_ID" -le 2 ]; then
    mode=attention
    seed=$((42 + SLURM_ARRAY_TASK_ID))
    config=configs/subject_pair_exhaustive_attention.yaml
    checkpoint_dir=checkpoints/subject_pair_exhaustive_attention/s$seed
else
    mode=mlp
    seed=$((40 + SLURM_ARRAY_TASK_ID))
    config=configs/subject_pair_exhaustive_frozen.yaml
    checkpoint_dir=checkpoints/subject_pair_exhaustive_mlp_replicates/s$seed
fi

echo "[comparison] mode=$mode seed=$seed"
python train_subject_pair_exhaustive.py --config "$config" --seed "$seed" --checkpoint-dir "$checkpoint_dir"
