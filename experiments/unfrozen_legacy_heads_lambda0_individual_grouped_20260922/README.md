# Unfrozen legacy-scale fusion, individual-grouped input, lambda=0

This is an isolated follow-up to
`fixed_encoder_legacy_heads_lambda0_individual_grouped_20260921`.
The preceding 36 runs were verified to use `encoders_frozen=true`.  This
follow-up uses the same completed encoder checkpoints as initialization, but
updates the MEG, fMRI, and text encoders together with the fusion head and
temperature.

## Configuration held constant

- individual-grouped input (`data_mode=exhaustive`);
- `lambda_mf=0.0`, `lambda_jt=1.0`;
- MLP hidden dimension 512;
- parameter-matched attention hidden dimension 767;
- head seeds 42, 43, and 44;
- 60 epochs, batch size 64, AdamW learning rate 0.0003, weight decay 0.01;
- noise standard deviation 0.1;
- checkpoint selection on sections 8--9 by mean global MedR over all subject pairs;
- the same All and Held-out encoder checkpoint provenance as the frozen run.

## Changed setting

- `encoders_frozen=false`: all model parameters are trainable after loading the
  corresponding encoder checkpoint.

## Task layout

- All: 3 MLP + 3 attention runs (`array=0-5`).
- Held-out: 5 folds × (3 MLP + 3 attention) (`array=0-29`).
- Total: 36 independent runs.

Submitted on 2026-09-22 to `g3090`:

- All array: Slurm job `1707874` (`0-5`).
- Held-out array: Slurm job `1707875` (`0-29`).

## All encoder-seed supplement

- Purpose: add All encoder initialization seeds 43 and 44 to the unfrozen
  benchmark while preserving the completed seed-42 and Held-out runs.
- Each encoder initialization runs 3 MLP and 3 Attention heads using head seeds
  42--44, for 12 additional runs.
- Encoder checkpoints are initialization only; every model parameter remains
  trainable (`encoders_frozen=false`).
- Output keys: `all_subjects/fixed_encoder_seed_43/...` and
  `all_subjects/fixed_encoder_seed_44/...`.
- Submitted on 2026-09-22 as Slurm array `1708150` (`0-11`).

The frozen experiment directory and all of its results are preserved.
