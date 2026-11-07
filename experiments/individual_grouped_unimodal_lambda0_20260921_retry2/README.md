# Individual-grouped unimodal retrieval, lambda=0, retry 2

- Date: 2026-09-21
- Purpose: supplement the six failed retry-1 runs without rerunning or overwriting successful All-participants seeds 42 and 43.
- Runs: All-participants seed 44; Held-out folds 0--4 with seeds 42--46.
- Source commit: `88bd1c438e17e8f9f90e5fa98692c20441dfe64b`
- Source root: `/gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907`
- Config: `/gpfs/userdata/ldeng/Speech-BCI/biomag2026/paper_tables_lambda0_20260917/configs/unified_benchmark_task1529545_matched.yaml`
- Config SHA-256: `feb3ef893cebfe4084be4919de3d5000b0015cea4617d56f80a212fa26bd9f7a`
- Remote run root: `/gpfs/userdata/ldeng/Speech-BCI/biomag2026/individual_grouped_unimodal_lambda0_20260921_retry2`
- Experimental settings remain `data_mode=exhaustive`, `lambda_mf=0.0`, and `lambda_jt=1.0`.
- Resource-only change: use partition `g3090` with one allocated GPU per task. Before resubmission, the node had six Slurm-available GPUs and the previously saturated physical GPU 0 had recovered to approximately 19 GiB free.
- Held-out array has no concurrency throttle (`0-4`).

The first retry-2 submission (`1705181`, `1705182`) targeted `ga10`, remained pending without creating any output, and was cancelled after the user noted that `g3090` had become available. The jobs were then resubmitted to `g3090`.

- Initial pending `ga10` jobs: All `1705181`, Held-out `1705182`; cancelled before start with no output.
- Completed `g3090` jobs: All seed 44 `1705184`; Held-out folds 0--3 from array `1705185`.
- Runtime array throttle: `0` (no concurrency limit).
- Initial `g3090` allocation: All seed 44 and Held-out folds 0--3 started successfully; Held-out fold 4/seed 46 was assigned physical GPU 7, which had only about 526 MiB free because of an external process, and failed at encoder startup with CUDA OOM.
- Held-out fold 4 was subsequently completed in the isolated retry-3 run; the failed retry-2 output was retained.
- Completion results in this directory: All seed 44 at 2026-09-21 10:26 CST; Held-out folds 0--3 between 09:57 and 10:10 CST.
