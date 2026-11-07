#!/bin/bash
#SBATCH --job-name=bc_l0_fmri_tr2s
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-11%2
#SBATCH --output=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/paper_tables_lambda0_20260917/logs/fmri_delay_ablation_2s_%A_%a.out
#SBATCH --error=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/paper_tables_lambda0_20260917/logs/fmri_delay_ablation_2s_%A_%a.err

set -eo pipefail

SOURCE_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
RUN_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/paper_tables_lambda0_20260917
CONFIG_PATH=${RUN_ROOT}/configs/fmri_delay_ablation_2s.yaml

cd "${SOURCE_ROOT}"
source ~/.bashrc
conda activate biomag_0530

labels=(d2 d2_4 d2_4_6 d2_4_6_8)
delay_args=("2" "2 4" "2 4 6" "2 4 6 8")
condition_index=$((SLURM_ARRAY_TASK_ID / 3))
seed=$((42 + SLURM_ARRAY_TASK_ID % 3))
label="${labels[$condition_index]}"
delays="${delay_args[$condition_index]}"

echo "[lambda0 2s fMRI delay ablation] condition=${label} delays=${delays}; seed=${seed}"
python unified_benchmark.py \
  --config "${CONFIG_PATH}" \
  --protocol all_subjects \
  --data-mode exhaustive \
  --seed "${seed}" \
  --window-cache cache/meg_text_window_nonoverlap/w2_stride2 \
  --meg-text-window-seconds 2 \
  --fmri-delay-subset-seconds ${delays} \
  --checkpoint-root "${RUN_ROOT}/checkpoints/fmri_delay_ablation_2s/${label}" \
  --result-root "${RUN_ROOT}/results/fmri_delay_ablation_2s/${label}"
