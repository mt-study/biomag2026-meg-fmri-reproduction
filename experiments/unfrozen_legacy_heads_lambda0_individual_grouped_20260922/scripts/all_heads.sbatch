#!/bin/bash
#SBATCH --job-name=bc_l0_unf_all
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/unfrozen_legacy_heads_lambda0_individual_grouped_20260922/logs/all_%A_%a.out
#SBATCH --error=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/unfrozen_legacy_heads_lambda0_individual_grouped_20260922/logs/all_%A_%a.err

set -eo pipefail

SOURCE_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
RUN_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/unfrozen_legacy_heads_lambda0_individual_grouped_20260922
CONFIG_PATH=${RUN_ROOT}/config.yaml
ENCODER_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/individual_grouped_unimodal_lambda0_20260920_retry1
ENCODER_CHECKPOINT=${ENCODER_ROOT}/checkpoints/all_subjects/exhaustive/seed_42/encoder_best.pt
ENCODER_RESULT=${ENCODER_ROOT}/results/all_subjects/exhaustive/seed_42/result.json

task=${SLURM_ARRAY_TASK_ID}
if (( task < 3 )); then
  fusion_mode=mlp
  head_seed=$((42 + task))
else
  fusion_mode=attention
  head_seed=$((42 + task - 3))
fi

OUTPUT_KEY=all_subjects/fixed_encoder_seed_42/${fusion_mode}/head_seed_${head_seed}

cd "${SOURCE_ROOT}"
source ~/.bashrc
conda activate biomag_0530

echo "[resource] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
echo "[task] protocol=all_subjects encoder_seed=42 fusion_mode=${fusion_mode} head_seed=${head_seed}"

PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}" python "${RUN_ROOT}/train_legacy_head_unfrozen.py" \
  --source-root "${SOURCE_ROOT}" \
  --config "${CONFIG_PATH}" \
  --protocol all_subjects \
  --encoder-seed 42 \
  --encoder-checkpoint "${ENCODER_CHECKPOINT}" \
  --encoder-result "${ENCODER_RESULT}" \
  --fusion-mode "${fusion_mode}" \
  --head-seed "${head_seed}" \
  --checkpoint-dir "${RUN_ROOT}/checkpoints/${OUTPUT_KEY}" \
  --result-dir "${RUN_ROOT}/results/${OUTPUT_KEY}"
