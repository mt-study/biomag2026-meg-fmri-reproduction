#!/bin/bash
#SBATCH --job-name=fmri_cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --array=1-35
#SBATCH --output=logs/fmri_cache_%A_%a.out
#SBATCH --error=logs/fmri_cache_%A_%a.err

set -e
cd /gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907

source ~/.bashrc
conda activate biomag_0530

mapfile -t SUBJECTS < <(ls -d /gpfs/userdata/ldeng/dataset/speech/fMRI/ds003643-download/derivatives/sub-CN* | xargs -n1 basename | sed 's/sub-//' | sort)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID-1))]}
echo "Processing subject $SUBJECT"

python scripts/build_fmri_cache.py --config configs/default.yaml --subject "$SUBJECT"
