# Frozen legacy heads, individual-grouped input, lambda=0

- Date: 2026-09-21
- Status: submitted at 2026-09-21 17:31 CST
- Purpose: correct the fusion-head portion of the individual-grouped lambda-zero benchmark without retraining any encoder.
- Data mode: `exhaustive` (individual-grouped).
- Objective record: `lambda_mf=0.0`, `lambda_jt=1.0`.
- Encoder policy: load existing completed encoder checkpoints and freeze MEG, fMRI, and text encoders.
- MLP: legacy `hidden=512`, 395,008 head parameters plus one trainable temperature (395,009 total trainable parameters).
- Attention: legacy parameter-matched `hidden=767`, 395,007 head parameters plus one trainable temperature (395,008 total trainable parameters).
- Fusion repetitions: head seeds 42, 43, and 44.
- Training schedule: 60 epochs, batch size 64, AdamW learning rate 0.0003, weight decay 0.01, training noise standard deviation 0.1.
- Selection/report sections: sections 8--9. As in the original fixed-encoder legacy-head code, checkpoint selection minimizes mean global MedR over every selection-set MEG--fMRI subject pair.
- Source commit: `88bd1c438e17e8f9f90e5fa98692c20441dfe64b`.
- Existing small-head results and checkpoints are preserved; this experiment uses a new remote root.

## Task layout

- All participants: one fixed seed-42 encoder, 3 MLP head seeds and 3 attention head seeds; Slurm array `0-5`.
- Held-out participants: five fold-specific fixed encoders, 3 MLP and 3 attention head seeds per fold; Slurm array `0-29`.
- Total head-training runs: 36.
- Slurm All array: `1705939` (`0-5`, six tasks, no array throttle).
- Slurm Held-out array: `1705940` (`0-29`, thirty tasks, no array throttle).

## All-participant encoder-seed supplement

- Submitted: 2026-09-22 CST.
- Purpose: add All-participant fixed encoders 43 and 44 so the All results can
  be summarized across three encoder seeds (42--44).
- Existing encoder seed 42 results remain unchanged.
- Each added encoder trains 3 MLP heads and 3 attention heads with head seeds
  42--44, for 12 additional runs.
- Output keys: `all_subjects/fixed_encoder_seed_43/...` and
  `all_subjects/fixed_encoder_seed_44/...`.
- Slurm array: `1707882` (`0-11`, twelve tasks, no array throttle).

## Encoder provenance

- All seed 42: `individual_grouped_unimodal_lambda0_20260920_retry1`, encoder epoch 7.
- Held-out folds 0--3: `individual_grouped_unimodal_lambda0_20260921_retry2`, encoder seeds 42--45.
- Held-out fold 4: `individual_grouped_unimodal_lambda0_20260921_retry3`, encoder seed 46.

The runner verifies source hashes, lambda values, exhaustive data mode, protocol/fold/seed provenance, encoder checkpoint path, hidden dimensions, and exact trainable parameter counts before training starts.
