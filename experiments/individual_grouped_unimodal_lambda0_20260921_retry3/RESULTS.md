# Individual-grouped unimodal retrieval at lambda=0: completed results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-09-21
- Verification Status: VERIFIED
- Version Label: individual_grouped_unimodal_lambda0_complete_v1
- Source commit: `88bd1c438e17e8f9f90e5fa98692c20441dfe64b`
- Config SHA-256: `feb3ef893cebfe4084be4919de3d5000b0015cea4617d56f80a212fa26bd9f7a`

## Completion and integrity checks

The final result set consists of 8 successful runs collected without overwriting prior output:

- All participants: seeds 42 and 43 from retry 1; seed 44 from retry 2.
- Held-out participants: folds 0--3 / seeds 42--45 from retry 2; fold 4 / seed 46 from retry 3.
- Last result completed: job `1705310`, Held-out fold 4 / seed 46, at 2026-09-21 11:26 CST.

Every `result.json` passed all of the following checks:

- `split.data_mode == "exhaustive"`;
- `hyperparameters.lambda_mf == 0.0` and `lambda_jt == 1.0`;
- seed and fold match the intended run;
- `git_commit == 88bd1c438e17e8f9f90e5fa98692c20441dfe64b`;
- 60 encoder epochs and 60 fusion epochs;
- checkpoint selection is `validation_best`;
- training sections are 1--7 and evaluation sections are 8--9;
- 2247 training candidates and 684 evaluation candidates.

The retry-2 fold-4 CUDA OOM log remains preserved as a failed attempt. It is not included in the statistics. Its replacement in retry 3 completed successfully.

## Per-run unimodal metrics

Top-10 is a proportion. MedR is the mean participant-level median rank stored in each result. The four columns for each modality are Global Top-10, Global MedR, Within-section Top-10 and Within-section MedR.

| Participants | Seed | Fold | Slurm job | MEG G-Top10 | MEG G-MedR | MEG W-Top10 | MEG W-MedR | fMRI G-Top10 | fMRI G-MedR | fMRI W-Top10 | fMRI W-MedR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 42 | -- | 1704620 | 0.019883 | 305.15 | 0.038012 | 154.08 | 0.032999 | 243.33 | 0.061153 | 122.29 |
| All | 43 | -- | 1704621 | 0.020395 | 310.95 | 0.038231 | 155.45 | 0.036633 | 239.47 | 0.065873 | 118.91 |
| All | 44 | -- | 1705184 | 0.017105 | 307.75 | 0.034137 | 153.85 | 0.027945 | 252.24 | 0.053718 | 125.97 |
| Held-out | 42 | 0 | 1705186 | 0.019737 | 308.62 | 0.038377 | 155.00 | 0.032790 | 237.29 | 0.063074 | 118.79 |
| Held-out | 43 | 1 | 1705187 | 0.013523 | 324.25 | 0.029971 | 160.25 | 0.031746 | 260.57 | 0.054511 | 131.86 |
| Held-out | 44 | 2 | 1705188 | 0.020102 | 310.62 | 0.038012 | 157.50 | 0.036132 | 228.93 | 0.067251 | 115.64 |
| Held-out | 45 | 3 | 1705189 | 0.019737 | 300.75 | 0.036550 | 149.50 | 0.030075 | 230.64 | 0.059524 | 116.64 |
| Held-out | 46 | 4 | 1705310 | 0.019737 | 317.50 | 0.039108 | 161.12 | 0.027778 | 258.29 | 0.050752 | 132.93 |

## Aggregate results

All-participant values are mean plus or minus sample standard deviation across 3 seeds. Held-out values are mean plus or minus sample standard deviation across 5 folds.

| Participants | Modality | Global Top-10 | Global MedR | Within-section Top-10 | Within-section MedR |
| --- | --- | ---: | ---: | ---: | ---: |
| All | MEG | 0.0191 ± 0.0018 | 307.95 ± 2.91 | 0.0368 ± 0.0023 | 154.46 ± 0.87 |
| All | fMRI | **0.0325 ± 0.0044** | **245.01 ± 6.55** | **0.0602 ± 0.0061** | **122.39 ± 3.53** |
| Held-out | MEG | 0.0186 ± 0.0028 | 312.35 ± 8.94 | 0.0364 ± 0.0037 | 156.68 ± 4.68 |
| Held-out | fMRI | **0.0317 ± 0.0031** | **243.14 ± 15.21** | **0.0590 ± 0.0066** | **123.17 ± 8.50** |

## LaTeX replacement table

```latex
\begin{table*}[t]
\centering
\caption{\textbf{Unimodal retrieval with individual-grouped inputs at $\lambda_{\mathrm{MF}}=0$.}
All-participant values are mean $\pm$ sample standard deviation across three complete training seeds. Held-out values are mean $\pm$ sample standard deviation across five participant folds. Each participant is encoded separately and metrics are computed per participant before averaging. Training uses sections 1--7; sections 8--9 provide checkpoint selection and the reported 684-candidate evaluation. Bold identifies the better modality within each participant setting.}
\label{tab:individual_grouped_unimodal_lambda0}
\small\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}llcccc@{}}
\toprule
Participants & Modality & \shortstack{Global\\Top-10} & \shortstack{Global\\MedR} & \shortstack{Within-section\\Top-10} & \shortstack{Within-section\\MedR} \\
\midrule
\multirow{2}{*}{All}
& MEG  & $0.0191\pm0.0018$ & $307.95\pm2.91$ & $0.0368\pm0.0023$ & $154.46\pm0.87$ \\
& fMRI & $\mathbf{0.0325\pm0.0044}$ & $\mathbf{245.01\pm6.55}$ & $\mathbf{0.0602\pm0.0061}$ & $\mathbf{122.39\pm3.53}$ \\
\midrule
\multirow{2}{*}{Held-out}
& MEG  & $0.0186\pm0.0028$ & $312.35\pm8.94$ & $0.0364\pm0.0037$ & $156.68\pm4.68$ \\
& fMRI & $\mathbf{0.0317\pm0.0031}$ & $\mathbf{243.14\pm15.21}$ & $\mathbf{0.0590\pm0.0066}$ & $\mathbf{123.17\pm8.50}$ \\
\bottomrule
\end{tabular}
\end{table*}
```

## Raw result locations

- Retry 1: `/gpfs/userdata/ldeng/Speech-BCI/biomag2026/individual_grouped_unimodal_lambda0_20260920_retry1/results`
- Retry 2: `/gpfs/userdata/ldeng/Speech-BCI/biomag2026/individual_grouped_unimodal_lambda0_20260921_retry2/results`
- Retry 3: `/gpfs/userdata/ldeng/Speech-BCI/biomag2026/individual_grouped_unimodal_lambda0_20260921_retry3/results`
