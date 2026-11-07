# Frozen legacy fusion heads on group-averaged inputs, lambda=0

- Prepared: 2026-09-23 CST.
- Purpose: make the group-averaged fusion comparison match the completed individual-grouped frozen-head experiment, changing the input mode to `mean` and using the corresponding previously trained mean-input encoders.
- Source encoder experiment: `paper_tables_lambda0_20260917/unified_benchmark_task1529545_matched`, `data_mode=mean`, `lambda_mf=0`, `lambda_jt=1`.
- Encoder policy: freeze the existing MEG, fMRI, and text encoders; train only the fusion head and temperature.
- Fusion: MLP hidden 512 (395,009 trainable parameters including temperature); parameter-matched attention hidden 767 (395,008 including temperature).
- Repetitions: All uses encoder seeds 42, 43, 44. Held-out uses folds 0--4 and their corresponding encoder seeds 42--46. Every encoder trains MLP and attention with head seeds 42, 43, 44. Total 18 All + 30 Held-out = 48 head runs.
- Schedule: 60 epochs, batch size 64, AdamW learning rate 0.0003, weight decay 0.01, input noise standard deviation 0.1. Selection is the minimum mean Global MedR on sections 8--9 using training participants. Evaluation uses sections 8--9 and, for Held-out, held-out participants.
- New output root: `/gpfs/userdata/ldeng/Speech-BCI/biomag2026/fixed_encoder_legacy_heads_lambda0_group_averaged_20260923/`.
- The independent output root preserves all existing encoder, small-head, and individual-grouped results.
- Submitted: 2026-09-23 13:49 CST. All array `1709117` (`0-17`) and Held-out array `1709119` (`0-29`), both on `g3090` with one GPU per task and no array throttle. At the first post-submission check, All task 0 was running on `n05`; the remaining tasks were queued.

## Comparison scope

The model code, head dimensions, head seeds, optimization schedule, checkpoint-selection rule, and All/Held-out fold structure match `fixed_encoder_legacy_heads_lambda0_individual_grouped_20260921`. Both sets use their own previously trained input-specific encoders. `mean` averages training participants before encoding; `exhaustive` keeps their individual inputs. This comparison changes the input representation at both encoder training and head training, as intended by the average-versus-individual experiment.

For each All encoder, first average its three head-seed results, then report mean and sample SD across three encoder seeds. For Held-out, first average the three head seeds per fold, then report mean and sample SD across five folds. The head seeds are nested repetitions, not independent encoder replicates.
