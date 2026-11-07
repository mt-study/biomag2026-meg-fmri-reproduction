# Individual-grouped unimodal retrieval, lambda=0, retry 1

- Date: 2026-09-20
- Status: submitted and pending
- Previous failed jobs: `1704572` and `1704573`
- Failure reason: all eight tasks encountered CUDA out-of-memory at encoder startup because the visible GPU was already occupied.
- Source commit: `88bd1c438e17e8f9f90e5fa98692c20441dfe64b`
- Source root: `/gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907`
- Config: `/gpfs/userdata/ldeng/Speech-BCI/biomag2026/paper_tables_lambda0_20260917/configs/unified_benchmark_task1529545_matched.yaml`
- Config SHA-256: `feb3ef893cebfe4084be4919de3d5000b0015cea4617d56f80a212fa26bd9f7a`
- Remote run root: `/gpfs/userdata/ldeng/Speech-BCI/biomag2026/individual_grouped_unimodal_lambda0_20260920_retry1`
- Experimental change from the matched mean-input benchmark: `data_mode=mean` to `data_mode=exhaustive` only.
- Fixed objective: `lambda_mf=0.0`, `lambda_jt=1.0`.
- All-participants runs: seeds 42, 43, and 44.
- Held-out runs: folds 0--4 with seeds 42--46.
- Resource correction for retry 1: explicitly request `g3090` and `gpu:1`; log the assigned node, `CUDA_VISIBLE_DEVICES`, and GPU memory before training.

- Retry all-participants job: `1704603` (submitted as `0-2%2`; runtime throttle removed to `0-2`)
- Retry held-out job: `1704604` (submitted as `0-4%2`; runtime throttle removed to `0-4`)
- Post-submission verification: both arrays are on partition `g3090` with `TresPerNode=gpu:1`.
- Runtime update on 2026-09-20: `scontrol update JobId=<job> ArrayTaskThrottle=0` was applied to jobs `1704603` and `1704604`, removing the original `%2` concurrency cap and allowing every array element to run whenever GPUs are available.
