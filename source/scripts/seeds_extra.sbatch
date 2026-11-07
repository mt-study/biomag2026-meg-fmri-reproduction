#!/bin/bash
#SBATCH --job-name=bc_seeds12
#SBATCH --partition=ga10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/seeds12_%j.out
#SBATCH --error=logs/seeds12_%j.err
set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530
for s in 4 5 6 7 8 9 10 11 12 13 14 15; do
  echo "########## nojoint_s$s ##########"
  python train.py --config configs/seeds/nojoint_s$s.yaml
done
