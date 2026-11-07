# Reproduction runbook

## 1. Environment and inputs

Install the versions in `source/requirements.txt` in a Python environment with the corresponding CUDA runtime. Obtain the MEG challenge recordings and the fMRI `ds003643` derivative and annotation files independently. The expected directory layout and the exact preprocessing choices are documented in `source/README.md`. Keep the released YAMLs unchanged; generate location-specific copies:

```bash
python scripts/prepare_configs.py \
  --data-root /data/speech \
  --output-root /scratch/biomag2026-runs
```

The helper rewrites `project_dir`, source paths, data paths, output paths, and downstream checkpoint references in YAML only. It writes `.generated/configs/<original-path>`. Examine the generated files before scheduling a run. Historical `.sbatch` files are retained for task-array structure; edit their partition, account, time, Python environment, and paths for your cluster, or invoke the Python entry points directly.

## 2. Caches

From `source/`, prepare MEG, fMRI, and text caches using `scripts/preprocess_meg_unified.py`, `scripts/build_fmri_cache.py`, and `scripts/build_text_cache.py`. The archived `source/scripts/*.sbatch` files show the subject arrays. Then use `scripts/build_subject_pair_window_cache.py` for the default 2-s cache. For Figure 4 also build the fixed-onset, non-overlapping cache described in `source/scripts/build_meg_text_window_nonoverlap_cache.sbatch`. Confirm that the default split has 2,247 train and 684 evaluation windows before comparing to the paper.

## 3. Training order

1. **Table 1:** run `source/unified_benchmark.py` with the generated `unified_benchmark_task1529545_matched.yaml`, `--data-mode exhaustive`, All seeds 42–44 and unseen-subject folds 0–4 (encoder seeds 42–46). Use the per-fold CLI shown in the archived `individual_grouped_unimodal_lambda0_*/*.sbatch` files.
2. **Figure 2:** retain the Table 1 individual outputs. Train `--data-mode mean` encoders with the same benchmark YAML, then run `experiments/fixed_encoder_legacy_heads_lambda0_group_averaged_20260923/train_legacy_head_only.py` with the generated `config.yaml` for the group side. Each of three All encoders and five held-out encoders has MLP and Attention head seeds 42–44.
3. **Figure 3:** run the frozen individual head trainer in `experiments/fixed_encoder_legacy_heads_lambda0_individual_grouped_20260921/`, then the unfrozen trainer in `experiments/unfrozen_legacy_heads_lambda0_individual_grouped_20260922/` from the same encoder checkpoints. Keep protocol, fold, encoder seed, head seed, and fusion mode paired.
4. **Figure 4:** run `source/unified_benchmark.py` with generated `fmri_delay_ablation_2s.yaml`, `--meg-text-window-seconds 2`, and `--fmri-delay-subset-seconds` set successively to `2`, `2 4`, `2 4 6`, and `2 4 6 8`; repeat each condition for seeds 42–44. The archived delay `.sbatch` encodes the full array.
5. **Table 2:** the 8–9 baseline uses the frozen legacy MLP and matched attention configs. For 4–5 and 1–2, run `source/section_matched_0908/train_encoder.py` using generated `holdout_45_encoder.yaml` and `holdout_12_encoder.yaml` respectively, then train each MLP and attention head with `source/train_subject_pair_exhaustive.py` using the matching generated split config. Head seeds are 42–44. `source/section_matched_0908/export_result.py` records lineage and metrics.

Run all entry points from `source/` or set `PYTHONPATH` to that directory. Checkpoints produced in an earlier step are required by the subsequent head-training steps. The historical YAMLs have `lambda_mf=0`, `lambda_jt=1`; the path helper does not change these or any other scientific parameter. The `experiments/*/README.md` files give the original seeds, head dimensions, splits, and selection rules.

## 4. Expected results and limits

Use [EXPERIMENTS.md](EXPERIMENTS.md) to compare the output means and sample standard deviations with the supplied PDF. Table 1 uses three All seeds or five held-out folds as aggregation units. Figure 3 averages head seeds within an encoder or fold before comparing across three encoders or five folds. Table 2's learnable-head results average three head seeds; its 8–9 row reuses a fixed-encoder baseline. Different Table 2 partitions have different candidate counts, so compare methods within each partition.

The output depends on dataset availability, CUDA/PyTorch behavior, and the original checkpoint-selection procedure. The archived code is supplied for independent reruns; no pretrained checkpoint or raw participant data is included.
