# Paper-to-code map

All numbers below are read from `STSF_1015.pdf` and checked against the corresponding archived result summaries or original server JSON. This map distinguishes the paper's labels from older manuscript table numbers used in historical folder names.

| Paper item | Experiment code and configuration | Reported check |
|---|---|---|
| Table 1 | `source/unified_benchmark.py`; `experiments/paper_tables_lambda0_20260917/configs/unified_benchmark_task1529545_matched.yaml`; the three `individual_grouped_unimodal_lambda0_*` folders record the successful All seeds 42–44 and Held-out folds 0–4 | Held-out MLP global Top-10 `0.0322 ± 0.0048`, MedR `240.10 ± 16.82`; input mode `exhaustive`, `lambda_mf=0` |
| Figure 2, individual side | Same Table 1 individual-grouped runs | All fMRI Top-10 `0.0325` |
| Figure 2, group side | `experiments/fixed_encoder_legacy_heads_lambda0_group_averaged_20260923/` and its `train_legacy_head_only.py`; mean-input encoder from `paper_tables_lambda0_20260917` | All fMRI Top-10 `0.0721`, MLP `0.0643`; frozen heads 512/767 |
| Figure 3 | `experiments/fixed_encoder_legacy_heads_lambda0_individual_grouped_20260921/` and `experiments/unfrozen_legacy_heads_lambda0_individual_grouped_20260922/` | Held-out frozen MLP Top-10 `0.0339`; unfrozen MLP `0.0311`; paired 512/767 heads |
| Figure 4 | `source/unified_benchmark.py`; `experiments/paper_tables_lambda0_20260917/configs/fmri_delay_ablation_2s.yaml`; `experiments/paper_tables_lambda0_20260917/train_fmri_delay_ablation_2s.sbatch` | MLP global Top-10 rises `0.0234 ± 0.0007` → `0.0343 ± 0.0045` from delay `[2]` to `[2,4,6,8]`; fixed 2-s MEG/text windows, 2,247/684 candidates |
| Table 2, sections 8–9 | `source/train_subject_pair_exhaustive.py`; `experiments/paper_tables_lambda0_20260917/configs/unified_fixed_encoder_legacy_{mlp,attention}.yaml` | MLP global Top-10 `0.03570 ± 0.00037` |
| Table 2, sections 4–5 and 1–2 | `source/section_matched_0908/train_encoder.py`, `source/section_matched_0908/export_result.py`, `source/train_subject_pair_exhaustive.py`; `experiments/paper_tables_lambda0_20260917/configs/holdout_{45,12}_*.yaml` | MLP global Top-10 `0.05468 ± 0.00031` and `0.05182 ± 0.00048`; 589 and 595 candidates |

The later `individual_grouped_lambda0_sections_20261013` rerun is a different protocol and does **not** supply Table 2 values. The original `lambda_mf=1` delay-count results also do **not** supply Figure 4 in the supplied PDF.

## Provenance

The shared source commit is `88bd1c438e17e8f9f90e5fa98692c20441dfe64b`. Figure 4's 12 delay-run JSON files and Table 2's 12 section-head JSON files were checked on the server under `paper_tables_lambda0_20260917/results/`. Their full raw data are not distributed here. The historical `experiments/*/README.md` and manifests record seed, fold, Slurm job, checkpoint, and selection rules.
