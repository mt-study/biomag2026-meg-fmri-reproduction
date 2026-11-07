# Individual-grouped unimodal retrieval, lambda=0, retry 3

- Date: 2026-09-21
- Purpose: rerun only Held-out fold 4 / seed 46 after retry 2 was assigned to an externally occupied RTX 3090 GPU and failed with CUDA OOM.
- Source commit: `88bd1c438e17e8f9f90e5fa98692c20441dfe64b`
- Source root: `/gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907`
- Config: `/gpfs/userdata/ldeng/Speech-BCI/biomag2026/paper_tables_lambda0_20260917/configs/unified_benchmark_task1529545_matched.yaml`
- Config SHA-256: `feb3ef893cebfe4084be4919de3d5000b0015cea4617d56f80a212fa26bd9f7a`
- Remote run root: `/gpfs/userdata/ldeng/Speech-BCI/biomag2026/individual_grouped_unimodal_lambda0_20260921_retry3`
- Experimental settings remain `data_mode=exhaustive`, `lambda_mf=0.0`, and `lambda_jt=1.0`.
- Resource-only change: submit with one GPU. No experiment code, configuration, seed, fold, or output path was changed.

- Initial submitted job: `1705222`.
- The initial job never started and produced no result. Slurm reported `PENDING (BadConstraints)` because it combined partition `ga10` with requested node `n05`, while `n05` belongs to `g3090`.
- At 2026-09-21 09:59 CST, job `1705222` was cancelled before execution.
- Replacement job: `1705310`, submitted with command-line resource overrides `--partition=g3090 --nodelist=n05` and the unchanged batch script.
- Verified scheduler request for `1705310`: partition `g3090`, requested node `n05`, `gpu:1`.
- Initial replacement state: `PENDING (Priority)`.
- The batch file below is retained exactly as submitted; its embedded `ga10` directive was overridden by the recorded `sbatch` command-line options. Slurm command-line options take precedence over embedded directives.
- Replacement job `1705310` completed successfully on 2026-09-21 at 11:26 CST and produced the Held-out fold 4 / seed 46 `result.json`.
- Cross-directory validation found all 8 expected results: 3 All-participant seeds and 5 Held-out folds. See `RESULTS.md` for the verified metrics and replacement LaTeX.
