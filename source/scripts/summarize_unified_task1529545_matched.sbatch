#!/bin/bash
#SBATCH --job-name=bc_1529545_summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/unified_1529545_summary_%j.out
#SBATCH --error=logs/unified_1529545_summary_%j.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907
source ~/.bashrc
conda activate biomag_0530

python summarize_unified_benchmark.py \
    --result-root results/unified_benchmark_task1529545_matched \
    --output-json results/unified_benchmark_task1529545_matched/summary.json \
    --output-md results/unified_benchmark_task1529545_matched/summary.md
