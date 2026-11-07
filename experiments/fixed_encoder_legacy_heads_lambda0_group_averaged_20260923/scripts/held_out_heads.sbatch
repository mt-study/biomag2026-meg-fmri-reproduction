#!/bin/bash
# Held-out participants: five fold-specific frozen mean-input encoders.
#SBATCH --job-name=bc_l0_mean_held
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-29
#SBATCH --output=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/fixed_encoder_legacy_heads_lambda0_group_averaged_20260923/logs/held_%A_%a.out
#SBATCH --error=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/fixed_encoder_legacy_heads_lambda0_group_averaged_20260923/logs/held_%A_%a.err

set -eo pipefail

SOURCE_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
RUN_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/fixed_encoder_legacy_heads_lambda0_group_averaged_20260923
ENCODER_ROOT=/gpfs/userdata/ldeng/Speech-BCI/biomag2026/paper_tables_lambda0_20260917
ENCODER_EXPERIMENT=unified_benchmark_task1529545_matched

task=${SLURM_ARRAY_TASK_ID}
fold=$((task / 6))
head_slot=$((task % 6))
encoder_seed=$((42 + fold))
if (( head_slot < 3 )); then
  fusion_mode=mlp
  head_seed=$((42 + head_slot))
else
  fusion_mode=attention
  head_seed=$((42 + head_slot - 3))
fi

ENCODER_KEY=unseen_subject/mean/seed_${encoder_seed}_fold_${fold}
ENCODER_CHECKPOINT=${ENCODER_ROOT}/checkpoints/${ENCODER_EXPERIMENT}/${ENCODER_KEY}/encoder_best.pt
ENCODER_RESULT=${ENCODER_ROOT}/results/${ENCODER_EXPERIMENT}/${ENCODER_KEY}/result.json
OUTPUT_KEY=unseen_subject/fold_${fold}_encoder_seed_${encoder_seed}/${fusion_mode}/head_seed_${head_seed}

cd "${SOURCE_ROOT}"
source ~/.bashrc
conda activate biomag_0530

echo "[resource] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
echo "[task] protocol=unseen_subject data_mode=mean fold=${fold} encoder_seed=${encoder_seed} fusion_mode=${fusion_mode} head_seed=${head_seed}"

PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}" python "${RUN_ROOT}/train_legacy_head_only.py" \
  --source-root "${SOURCE_ROOT}" \
  --config "${RUN_ROOT}/config.yaml" \
  --protocol unseen_subject \
  --fold "${fold}" \
  --encoder-seed "${encoder_seed}" \
  --encoder-checkpoint "${ENCODER_CHECKPOINT}" \
  --encoder-result "${ENCODER_RESULT}" \
  --fusion-mode "${fusion_mode}" \
  --head-seed "${head_seed}" \
  --checkpoint-dir "${RUN_ROOT}/checkpoints/${OUTPUT_KEY}" \
  --result-dir "${RUN_ROOT}/results/${OUTPUT_KEY}"
