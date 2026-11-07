# A shared temporal–semantic space for cross-subject MEG–fMRI language retrieval

This repository archives the **experiment code and configuration** underlying Table 1, Table 2, and Figures 2–4 of the Biomag 2026 manuscript. The `source/` training and preprocessing files were extracted without model-code edits from the server Git commit recorded in [SOURCE_COMMIT.txt](SOURCE_COMMIT.txt). The `experiments/` folders are the corresponding run configurations and launch scripts. See [EXPERIMENTS.md](EXPERIMENTS.md) for a figure-to-code map and the exact reported checks.

## What is included

- Encoder training, MEG/fMRI/text preprocessing, retrieval evaluation, and fusion models in `source/`.
- The exact historical `lambda_mf=0` YAML configurations and experiment launchers for the reported runs in `experiments/`.
- A path-configuration helper and [reproduction runbook](RUNBOOK.md).

The repository contains **no raw MEG/fMRI recordings, text embedding CSV, cached features, trained checkpoints, or participant-level output**. Those inputs must be obtained separately under the dataset providers' terms. The historical launch scripts retain the original Slurm paths as provenance; use `scripts/prepare_configs.py` to generate site-specific YAMLs before running. The code requires a CUDA-capable PyTorch environment for the full experiment matrix.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r source/requirements.txt
python scripts/prepare_configs.py \
  --data-root /path/to/speech-data \
  --output-root /path/to/new-experiment-output
```

The data root must contain `MEG/lpp_data_challenge` and `fMRI/ds003643-download`, including the Mandarin story annotations and BERT embedding CSV documented in [source/README.md](source/README.md). The [BIOMAG 2026 challenge portal](https://compneurolinglab.github.io/biomag_challenge/index.html#data-access) describes registration-based access to the MEG challenge bundle; the fMRI corpus is [OpenNeuro ds003643](https://openneuro.org/datasets/ds003643). Generated YAMLs are written under `.generated/configs/` and the archived YAMLs remain unchanged.

The methods and candidate-set definitions are in [RUNBOOK.md](RUNBOOK.md). Full reruns require substantial GPU time and the specified datasets; the publication package has been checked for code syntax and configuration consistency, not retrained outside the original server environment.

## Scope notes

Figure 2 in the supplied PDF uses the small 64/96 fusion heads for the individual-grouped side and the larger 512/767 heads for the group-averaged side. The figure reproduces the reported values, but its learned-fusion comparison is not a single-factor input-mode ablation. Figure 3 uses matched 512/767 heads for its frozen/unfrozen comparison. Figure 4 varies **fMRI delay count with fixed 2-second MEG and text windows**; it is distinct from the MEG/text window-duration experiment.

Figure 5 is not included: its execution source was not verified in the experiment-to-paper audit. The manuscript PDF and raw datasets are not redistributed here.
