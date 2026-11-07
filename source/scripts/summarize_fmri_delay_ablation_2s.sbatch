#!/bin/bash
#SBATCH --job-name=bc_fmri_tr2s_sum
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00
#SBATCH --output=logs/fmri_delay_ablation_2s_summary_%j.out
#SBATCH --error=logs/fmri_delay_ablation_2s_summary_%j.err

set -eo pipefail
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

python summarize_fmri_delay_ablation_2s.py
