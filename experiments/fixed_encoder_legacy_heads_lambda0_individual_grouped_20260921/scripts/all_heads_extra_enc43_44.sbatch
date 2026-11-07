#!/bin/bash
# Supplement the All-participant frozen legacy-head benchmark with encoder
# seeds 43 and 44.  Existing seed-42 outputs are intentionally untouched.
#SBATCH --job-name=bc_l0_leg_allx2
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/fixed_encoder_legacy_heads_lambda0_individual_grouped_20260921/logs/all_extra_%A_%a.out
#SBATCH --error=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/fixed_encoder_legacy_heads_lambda0_individual_grouped_20260921/logs/all_extra_%A_%a.err

set -eo pipefail

SOURCE_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
RUN_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/fixed_encoder_legacy_heads_lambda0_individual_grouped_20260921
CONFIG_PATH=${RUN_ROOT}/config.yaml

# Six heads per encoder: MLP seeds 42--44, then Attention seeds 42--44.
task=${SLURM_ARRAY_TASK_ID}
encoder_index=$((task / 6))
head_slot=$((task % 6))
encoder_seed=$((43 + encoder_index))

if (( head_slot < 3 )); then
  fusion_mode=mlp
  head_seed=$((42 + head_slot))
else
  fusion_mode=attention
  head_seed=$((42 + head_slot - 3))
fi

if (( encoder_seed == 43 )); then
  ENCODER_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/individual_grouped_unimodal_lambda0_20260920_retry1
else
  ENCODER_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/individual_grouped_unimodal_lambda0_20260921_retry2
fi

ENCODER_CHECKPOINT=${ENCODER_ROOT}/checkpoints/all_subjects/exhaustive/seed_${encoder_seed}/encoder_best.pt
ENCODER_RESULT=${ENCODER_ROOT}/results/all_subjects/exhaustive/seed_${encoder_seed}/result.json
OUTPUT_KEY=all_subjects/fixed_encoder_seed_${encoder_seed}/${fusion_mode}/head_seed_${head_seed}

cd "${SOURCE_ROOT}"
source ~/.bashrc
conda activate biomag_0530

echo "[resource] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
echo "[task] protocol=all_subjects encoder_seed=${encoder_seed} fusion_mode=${fusion_mode} head_seed=${head_seed}"
echo "[encoder] checkpoint=${ENCODER_CHECKPOINT}"

PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}" python "${RUN_ROOT}/train_legacy_head_only.py" \
  --source-root "${SOURCE_ROOT}" \
  --config "${CONFIG_PATH}" \
  --protocol all_subjects \
  --encoder-seed "${encoder_seed}" \
  --encoder-checkpoint "${ENCODER_CHECKPOINT}" \
  --encoder-result "${ENCODER_RESULT}" \
  --fusion-mode "${fusion_mode}" \
  --head-seed "${head_seed}" \
  --checkpoint-dir "${RUN_ROOT}/checkpoints/${OUTPUT_KEY}" \
  --result-dir "${RUN_ROOT}/results/${OUTPUT_KEY}"
