# Paper-table rerun with direct-alignment weight zero

This execution package reruns every model-performance experiment represented in
the current manuscript tables with `training.lambda_mf: 0.0`. The source code,
data splits, seeds, model sizes, schedules, optimizer settings, augmentation,
checkpoint-selection rules, and `lambda_jt: 1.0` are unchanged.

The remote source tree is read-only for this run:

`/gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907`

The isolated remote run root is:

`/gpfs/userdata/ldeng/Speech-BCI/biomag2026/paper_tables_lambda0_20260917`

Operational output paths and downstream checkpoint references necessarily point
to the isolated run root so that no prior checkpoint, result, or log can be
overwritten. No model or training source file is modified.

## Scope

1. Historical all-participant and held-out mean/exhaustive benchmarks.
2. Fixed-encoder legacy MLP and parameter-matched attention.
3. Strict subject-disjoint five-fold MLP and attention.
4. Frozen-encoder versus joint-fine-tuning MLP.
5. Joint 2/4/6/8-s MEG-and-text window sensitivity.
6. Section 4--5 and section 1--2 sensitivity with MLP and attention heads.
7. Fixed-2-s MEG/text fMRI delay-count ablation over `[2]`, `[2,4]`,
   `[2,4,6]` and `[2,4,6,8]` seconds (Table 9; 2,247/684 candidates).

Dataset-size and partition-statistics tables are descriptive and do not require
model reruns.

## Submitted run

- Submission date: 2026-09-17 (Asia/Shanghai)
- Source commit: `88bd1c438e17e8f9f90e5fa98692c20441dfe64b`
- Source tracked worktree at submission: clean
- Parameter audit: passed for all 13 configs
- `main_all`: Slurm `1695036` (6 tasks)
- `main_unseen`: Slurm `1695037` (10 tasks)
- `strict_mlp`: Slurm `1695038` (5 tasks)
- `strict_attention`: Slurm `1695039` (5 tasks; after `1695038`)
- `fixed_heads`: Slurm `1695040` (6 tasks; after `1695036`)
- `finetune`: Slurm `1695041` (3 tasks; after `1695036`)
- `windows`: Slurm `1695042` (12 tasks)
- `section_enc`: Slurm `1695043` (2 tasks)
- `section_heads`: Slurm `1695044` (12 tasks; after `1695043`)

The initial process-alive check passed: eight independent array elements entered
`RUNNING`, downstream groups remained dependency-blocked as intended, and the
only initial stderr output was the cluster locale warning.

## Section-partition retry

The original section encoder array `1695043` failed before writing any
checkpoint or result because the submission wrapper omitted the source tree
from `PYTHONPATH`. Consequently, the original downstream array `1695044` never
started and was cancelled after replacement.

No code or experiment configuration was changed. The original runtime export
was restored and the two stages were resubmitted on 2026-09-18:

- `section_enc_retry`: Slurm `1699601` (2 tasks)
- `section_heads_retry`: Slurm `1699602` (12 tasks; `afterok:1699601`)

Both encoder retry elements passed source-hash and split preflight checks and
entered `RUNNING`. The head array will start automatically only if both encoder
elements finish successfully.

The encoder retry completed successfully. The first head retry `1699602`
stopped before training because the unchanged exporter resolves its encoder
lineage JSON relative to the working directory and therefore found the original
run's record. No head checkpoint or result was written by that attempt.

Two read-only lineage links were added under the isolated run root, pointing to
the new lambda-zero encoder records. All four split/head lineage preflights then
passed without changing code or experiment configuration. The 12 head tasks
were resubmitted as Slurm array `1699693`; its first two elements entered
training successfully with the new lambda-zero holdout-45 encoder.

## Table 9 delay-count correction

The initial scope audit missed the manuscript's fixed-2-s fMRI delay-count
ablation. The already completed window-duration experiment cannot substitute
for it because that experiment has 2,226/678 candidates, whereas Table 9 fixes
2,247/684 candidates. The parameter-matched primary benchmark also cannot
substitute because its fusion-head dimensions differ.

The correction uses the original `meg_text_window_nonoverlap.yaml` protocol and
source code unchanged, with only `training.lambda_mf: 1.0 -> 0.0` plus isolated
output paths. It explicitly runs all four delay sets and three seeds (12 array
tasks), so the full-delay row is also generated under the exact Table 9 setup.

- Submission date: 2026-09-18 (Asia/Shanghai)
- Slurm array: `1700607` (`0-11%2`)
- Source commit: `88bd1c438e17e8f9f90e5fa98692c20441dfe64b`
- Source tracked changes at submission: `0`
- Initial scheduler state: pending for priority
