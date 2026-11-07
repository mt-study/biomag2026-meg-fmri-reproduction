#!/bin/bash
#SBATCH --job-name=bc_frozen
#SBATCH --partition=ga10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/frozen_%j.out
#SBATCH --error=logs/frozen_%j.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

for cfg in configs/frozen/*.yaml; do
  echo "########## $cfg ##########"
  python train.py --config $cfg
done
