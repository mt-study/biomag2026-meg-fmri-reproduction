#!/bin/bash
#SBATCH --job-name=bc_sweep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530
mapfile -t checkpoints < <(find checkpoints/seeds -maxdepth 2 -type f -path "*/nojoint_s*/best.pt" | sort -V)
printf "Sweeping %d checkpoint(s)\n" "${#checkpoints[@]}"
for checkpoint in "${checkpoints[@]}"; do
  python sweep_fusion_weight.py --config configs/default.yaml --checkpoint "$checkpoint"
  echo
done
