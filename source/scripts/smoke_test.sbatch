#!/bin/bash
#SBATCH --job-name=brainclip_cc_test
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
# Note: no --mem here. This cluster's Slurm has RealMemory=1 (MB) configured
# on every node (a cluster misconfiguration, not something this project can
# fix), so any explicit --mem request is rejected at submission time.
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907

source ~/.bashrc
conda activate biomag_0530

python test_model.py
