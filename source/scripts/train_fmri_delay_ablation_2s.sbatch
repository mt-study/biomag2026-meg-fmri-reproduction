#!/bin/bash
#SBATCH --job-name=bc_fmri_tr2s
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-8%2
#SBATCH --output=logs/fmri_delay_ablation_2s_%A_%a.out
#SBATCH --error=logs/fmri_delay_ablation_2s_%A_%a.err

set -eo pipefail
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

labels=(d2 d2_4 d2_4_6)
delay_args=("2" "2 4" "2 4 6")
condition_index=$((SLURM_ARRAY_TASK_ID / 3))
seed=$((42 + SLURM_ARRAY_TASK_ID % 3))
label="${labels[$condition_index]}"
delays="${delay_args[$condition_index]}"

echo "[2s fMRI delay ablation] condition=${label} delays=${delays}; seed=${seed}"
python unified_benchmark.py \
  --config configs/meg_text_window_nonoverlap.yaml \
  --protocol all_subjects \
  --data-mode exhaustive \
  --seed "${seed}" \
  --window-cache cache/meg_text_window_nonoverlap/w2_stride2 \
  --meg-text-window-seconds 2 \
  --fmri-delay-subset-seconds ${delays} \
  --checkpoint-root "checkpoints/fmri_delay_ablation_2s/${label}" \
  --result-root "results/fmri_delay_ablation_2s/${label}"
