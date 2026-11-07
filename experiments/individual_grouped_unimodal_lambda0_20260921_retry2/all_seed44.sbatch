#!/bin/bash
#SBATCH --job-name=bc_l0_ind_all44_r2
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/individual_grouped_unimodal_lambda0_20260921_retry2/logs/all_seed44_%j.out
#SBATCH --error=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/individual_grouped_unimodal_lambda0_20260921_retry2/logs/all_seed44_%j.err

set -eo pipefail

SOURCE_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
CONFIG_PATH=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/paper_tables_lambda0_20260917/configs/unified_benchmark_task1529545_matched.yaml
RUN_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/individual_grouped_unimodal_lambda0_20260921_retry2

cd "${SOURCE_ROOT}"
source ~/.bashrc
conda activate biomag_0530

echo "[resource] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
echo "[individual-grouped lambda0 retry2] protocol=all_subjects data_mode=exhaustive seed=44"

python unified_benchmark.py \
  --config "${CONFIG_PATH}" \
  --protocol all_subjects \
  --data-mode exhaustive \
  --seed 44 \
  --checkpoint-root "${RUN_ROOT}/checkpoints" \
  --result-root "${RUN_ROOT}/results"
