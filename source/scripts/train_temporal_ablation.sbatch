#!/bin/bash
#SBATCH --job-name=bc_time_abl
#SBATCH --partition=g3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-17%2
#SBATCH --output=logs/temporal_ablation_%A_%a.out
#SBATCH --error=logs/temporal_ablation_%A_%a.err

set -eo pipefail
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

conditions=(w1_dall w2_dall w4_dall w2_d4 w2_d6 w2_d46)
caches=(w1_dall w2_dall w4_dall w2_dall w2_dall w2_dall)
condition_index=$((SLURM_ARRAY_TASK_ID / 3))
seed=$((42 + SLURM_ARRAY_TASK_ID % 3))
condition="${conditions[$condition_index]}"
cache_label="${caches[$condition_index]}"
delay_args=()

case "${condition}" in
    w2_d4) delay_args=(--fmri-delay-subset-seconds 4) ;;
    w2_d6) delay_args=(--fmri-delay-subset-seconds 6) ;;
    w2_d46) delay_args=(--fmri-delay-subset-seconds 4 6) ;;
esac

echo "[temporal ablation] condition=${condition} seed=${seed} cache=${cache_label}"
python unified_benchmark.py     --config configs/temporal_ablation.yaml     --protocol all_subjects     --data-mode exhaustive     --seed "${seed}"     --window-cache "cache/temporal_ablation/${cache_label}"     --checkpoint-root "checkpoints/temporal_ablation/${condition}"     --result-root "results/temporal_ablation/${condition}"     "${delay_args[@]}"