# BrainCLIP (meg_fmri_clip_cc)

CLIP-style contrastive baseline mapping MEG, fMRI, and text into one shared
256-d embedding space. Pure PyTorch, no dependency on the original CLIP repo.

Project path (fixed):

```text
/gpfs/userdata/ldeng/Speech-BCI/biomag_ldeng/meg_fmri_clip_cc
```

A related project (`../meg_fmri_clip`, no `_cc` suffix) already exists on
this server and solves the same problem; some of its verified facts (data
paths, TR, run↔section mapping) informed this build, but the code here is
independent.

## Data actually used

**MEG** — `/gpfs/userdata/ldeng/dataset/speech/MEG/lpp_data_challenge`
- Only 10 of 20 subjects (`derivatives/sub-01..sub-10`) ship author
  `*_desc-preproc_meg.fif` derivatives. Per explicit instruction, MEG is
  unified across **all 20 subjects** with one identical, documented
  pipeline instead of only using the 10 with derivatives — see "MEG
  unification" below. fMRI, by contrast, is **not** re-preprocessed: only
  the author-provided `*_desc-preproc_bold.nii.gz` derivatives are read.
- `participants.tsv` lists sub-01..sub-20.

**fMRI** — `/gpfs/userdata/ldeng/dataset/speech/fMRI/ds003643-download`
- Preprocessed files: `derivatives/sub-CNxxx/func/sub-CNxxx_task-lppCN_run-{06..14}_space-MNIColin27_desc-preproc_bold.nii.gz`
  (35 CN subjects have derivatives). Read-only; never filtered/registered
  by this project.
- Volume shape `(73, 90, 74, T)` float32, TR = 2 s (from the raw BIDS
  sidecar `sub-CNxxx_task-lppCN_run-XX_echo-*_bold.json`,
  `RepetitionTime: 2`). No brain mask ships with the derivatives, so
  pooling runs over the full volume (a known simplification).
- Each subject has exactly 9 `task-lppCN` runs, but **the BIDS run number
  where section 1 starts is subject-specific** (other scanner tasks precede
  the story in some sessions) — see "fMRI run-to-section mapping" below.

**Text / annotation** — `.../ds003643-download/annotation/CN/`
- `lppCN_word_information.csv`: 16,009 rows, columns
  `word, lemma, onset, offset, logfreq, pos, section, top_down, bottom_up, left_corner`.
  `section` ∈ {1..9}; onsets/offsets are section-relative seconds.
- `lppCN_word_embeddings_BERT.csv`: same 16,009 rows in the same order,
  column `BERT` is a **stringified numpy vector** (`"[ 0.97  0.16 ... ]"`,
  whitespace-separated, wraps across lines, no commas) — parsed once with
  a regex float extractor (`features.py::parse_bert_vector`), not
  `ast.literal_eval`/`json`. Vector dimension: 768.
- Because this cached embedding file already exists, **no Hugging Face
  model is downloaded or run** — text samples are the mean of the BERT
  rows whose `[onset, offset)` overlaps the sample's stimulus window.
- 284/16,009 rows (1.8%) have a differing surface word between the two
  CSVs at the same row index (e.g. `猎获物` vs `猎物`, `事儿` vs `事`,
  `B612` vs `612`) — two independently-tokenized files disagreeing on
  erhua suffixes / compound splitting, not a row-shift (row **counts**
  match exactly, so positional alignment holds). `scripts/build_text_cache.py`
  logs this and proceeds; it is immaterial once averaged into a multi-word
  window embedding.

## fMRI run-to-section mapping

Initial assumption (run − 5 = section, based on one sampled subject,
sub-CN028) turned out **wrong** once more subjects were cached: run
numbers are session-global, so where the lppCN story starts shifts per
subject, e.g.

| subject | lppCN run numbers found |
|---------|--------------------------|
| sub-CN001 | 04–12 |
| sub-CN002 | 04–12 |
| sub-CN003 | 05–13 |
| sub-CN028 | 06–14 |

What **is** constant: every subject has exactly 9 `task-lppCN` runs, and
sorting a subject's own runs by run number reproduces the canonical
section-duration sequence (563.9, 642.7, 643.7, 612.2, 583.5, 782.9,
726.6, 585.2, 801.1 s) almost exactly, e.g. sub-CN001 runs 05→12 give
644, 644, 614, 586, 784, 728, 586, 802 s against sections 2→9. So
`alignment.py::_discover_fmri_cache` groups cached runs by subject, sorts
by run number, and assigns section = rank (1-indexed) — **not** a fixed
offset. All 35 cached CN subjects have exactly 9 runs, so none are
dropped by this check (a subject would be skipped, and counted in
`AlignmentStats.skipped_fmri_subjects_bad_run_count`, only if it didn't).

## MEG unification

`scripts/preprocess_meg_unified.py` applies one identical pipeline to
every one of the 20 subjects' raw per-run FIF files
(`sub-XX/meg/sub-XX_task-lppCN_run-Y_meg.fif`, 329 channels, 1000 Hz):

1. Pick the 306 MEG sensors (102 magnetometers + 204 gradiometers),
   dropping EOG/ECG/STI/etc — exactly the channel composition of the
   10 author-provided derivatives.
2. Band-pass filter 0.1–40 Hz — a standard MEG analysis band, well inside
   the acquisition's own 0.1–330 Hz hardware filter.
3. Resample to 100 Hz — exactly the sample rate already used by the
   10-subject author derivatives, so all 20 subjects share one array
   convention.

**Stimulus-onset alignment, verified empirically**: for sub-01 runs 1, 6
and 9 (the only runs directly checkable against a trusted author
derivative), the raw file's own duration matches the derivative epoch's
time span (`tmax - tmin`) to within ~10 ms:

| run | raw duration (s) | derivative span (s) |
|-----|------------------:|---------------------:|
| 1   | 566.083 | 566.08 |
| 6   | 783.979 | 783.97 |
| 9   | 802.201 | 802.20 |

This means raw sample 0 already corresponds to the derivative's
`tmin = -0.1 s` (the story begins ~0.1 s into the recording). No cropping
is applied — each run is kept in full and tagged `tmin = -0.1 s` in its
cache metadata, so window indexing (`round((t - tmin) * sfreq)`) is
identical to the derivative convention for all 20 subjects. (A single
stim-channel trigger found via `mne.find_events` on sub-01 run-1 sits at
t=27.4 s, inconsistent with this duration cross-check, so it is **not**
used for cropping — noted here so the choice isn't silently mysterious.)

This pipeline does not claim scientific equivalence to whatever
proprietary cleaning (bad-channel interpolation, ICA, Maxwell filtering,
source localization, etc.) produced the 10 author derivatives — it is a
consistent, documented sensor-space baseline applied identically to all
20 subjects.

## Caching (must be built before training)

Loading/decompressing ~550 MB fMRI volumes and re-filtering ~800 MB raw
MEG files on every training launch is too slow to repeat per run, so
everything is pre-extracted once into a project-owned cache (excluded
from git, see `.gitignore`):

```text
cache/
├── meg_unified/sub-XX/sub-XX_task-lppCN_run-Y_desc-unifiedpreproc_meg.{npy,json}
├── fmri_pooled/sub-CNxxx/sub-CNxxx_task-lppCN_run-Y_desc-pooled_bold.{npy,json}
└── text/lppCN_bert_embeddings.npy
```

- `meg_unified/*.npy`: float32 `[306, T]`, output of MEG unification above.
- `fmri_pooled/*.npy`: float32 `[n_TR, *fmri_pool_size]` (default pool
  grid `8x9x8`) — **every** TR of the run is average-pooled once
  (`features.py::pool_volume`, deterministic for any input size), so a
  window's 4-delay fMRI feature at train time is just 4 array look-ups + a
  concatenate, not a NIfTI decompression.
- `text/lppCN_bert_embeddings.npy`: the whole 16,009×768 BERT matrix,
  parsed once instead of on every launch.

`train.py` / `test_model.py` never touch MNE or NIfTI files — `alignment.py`
and `dataset.py` only read these caches (plus the small annotation CSV).
If a cache is missing, `alignment.py` raises a clear error naming the
`sbatch` script to run instead of silently falling back to anything.

Build order:

```bash
sbatch scripts/preprocess_meg_unified.sbatch   # array 1-20, one task per subject
sbatch scripts/build_fmri_cache.sbatch         # array 1-35, one task per CN subject
sbatch scripts/build_text_cache.sbatch         # single task
```

## Alignment (`alignment.py`)

- Stimulus windows of `window_seconds=2` (config), non-overlapping,
  starting at `t = 0, 2, 4, ...` within each section.
- A window's upper bound per section is
  `min(text section max offset, min MEG run duration, min fMRI run duration) - max(fmri_delay_seconds) - window_seconds`,
  so every window has valid data for all four fMRI delays.
- **MEG sample**: `meg_run[:, t : t+2s]` (via the cached array + tmin/sfreq)
  → `[306, 200]`.
- **fMRI sample**: for each delay `d` in `[2, 4, 6, 8]` s, the pooled
  volume at `TR = round((t+d)/2)` is looked up directly from the per-run
  cache and the 4 delayed features are concatenated → `4 * 8*9*8 = 2304`-d.
- **Text sample**: mean-pooled 768-d BERT vector of overlapping words.
- **Positives are defined by `(section, t)` identity only, never by array
  index or subject.** MEG and fMRI do not need to come from the same
  participant — at each `__getitem__` call one MEG-subject realization and
  one fMRI-subject realization sharing that `(section, t)` are randomly
  drawn.
- **Train/val split**: sections 1–7 train, sections 8–9 validation (whole,
  contiguous sections — no adjacent-window leakage across the split).
- Alignment statistics (files found, candidate windows per modality,
  successfully aligned windows, and windows skipped for missing
  annotation overlap or out-of-bounds sections) are printed by `train.py`
  at startup and reproduced in "Verified Runs" below.

## Model (`models/`)

- `MEGEncoder`: 1D CNN (3 conv blocks + global average pool) over
  `[batch, 306, 200]` → 256-d.
- `FMRIEncoder`: 2-layer MLP over the 2304-d concatenated multi-delay
  feature → 256-d.
- `TextEncoder`: 2-layer MLP projection head over the cached 768-d BERT
  vector → 256-d.
- All three outputs are L2-normalized; a learnable log-temperature scales
  the cosine similarities (`models/brain_clip.py`).
- All three encoders' input dimensions are discovered at runtime from the
  cache (`dataset.py::build_datasets`) — nothing is hardcoded except the
  target fMRI pooling grid, which is a config value.

## Loss (`losses/contrastive.py`)

Symmetric InfoNCE per pair, combined as

```text
loss = loss_meg_text + loss_fmri_text + lambda_mf * loss_meg_fmri
```

Batches are built from shared `(section, t)` windows, so every batch has
aligned meg/fmri/text triplets with in-batch negatives.

## Environment

The spec's placeholder conda env name (`meg_fmri_clip`) does not exist on
this server. `speech_biomag2026` (used by the sibling project) is
currently **broken** on this login node (`scipy==1.16.3` vs
`numpy==1.26.4` ABI mismatch breaks `mne` import). All scripts here
instead activate **`biomag_0530`**
(`/share/home/ldeng/.conda/envs/biomag_0530`), verified to import
`torch`, `mne`, `nibabel`, `pandas`, and read real `.fif`/`.nii.gz` files.
No new packages were installed.

This cluster's Slurm has `RealMemory=1` (MB) configured on every node —
a pre-existing cluster misconfiguration, not something fixed here — so
none of the `sbatch` scripts pass `--mem`; requesting any explicit memory
amount is rejected at submission time regardless of actual free memory.

## Run

All real runs go through Slurm, never the login node directly:

```bash
# one-time cache build (see Caching above)
sbatch scripts/preprocess_meg_unified.sbatch
sbatch scripts/build_fmri_cache.sbatch
sbatch scripts/build_text_cache.sbatch

# then
sbatch scripts/smoke_test.sbatch
sbatch scripts/train.sbatch

squeue -u ldeng
tail -f logs/test_<jobid>.out
tail -f logs/train_<jobid>.out
```

Best checkpoint is written to:

```text
checkpoints/best.pt
```

## Verified Runs

All on 2026-07-26, cluster user `ldeng`, env `biomag_0530`:

- Smoke test job `1529410` (cpu): `BrainCLIP smoke test passed`, no errors.
- Cache builds, all completed with zero errors:
  - `1529412` (`preprocess_meg_unified.sbatch`, array 1-20): 180/180 MEG
    `.npy` files written (20 subjects × 9 runs), ~14 GB.
  - `1529432` (`build_fmri_cache.sbatch`, array 1-35): 315/315 fMRI `.npy`
    files written (35 subjects × 9 runs, all with exactly 9 runs — no
    subjects skipped), ~231 MB.
  - `1529471` (`build_text_cache.sbatch`): 16,009×768 embedding matrix
    cached, ~47 MB; logged the 284-row word-mismatch warning described
    above and proceeded.
- Training job `1529472` (g3090, GPU): alignment produced
  **2,931 aligned windows** (2,247 train / 684 val) from 180 MEG cached
  runs and 315 fMRI cached runs, 0 windows skipped for missing annotation
  or out-of-bounds sections, 0 fMRI subjects skipped. Dims discovered at
  runtime: `meg_channels=306, meg_time=200, fmri_dim=2304, text_dim=768`
  (all match the design above). First 6 epochs (~110s/epoch on the GPU
  node):

  ```text
  [epoch 1/20] total=10.4233 meg_text=3.4783 fmri_text=3.4790 meg_fmri=3.4660 val_retrieval_acc=0.0015
  [epoch 2/20] total=10.4009 meg_text=3.4673 fmri_text=3.4678 meg_fmri=3.4658 val_retrieval_acc=0.0015
  [epoch 6/20] total=10.3986 meg_text=3.4660 fmri_text=3.4668 meg_fmri=3.4657 val_retrieval_acc=0.0015
  ```

  This run **did not learn anything** — every loss sat at exactly
  `ln(batch_size=32)=3.4657` for 16 epochs and retrieval never left chance.
  That is representational collapse, not slow convergence; the diagnosis
  and fix are below.

## Why v1 collapsed, and what fixed it

Three independent bugs, each found by measuring the data rather than by
tuning hyperparameters. The decisive measurement was mean pairwise cosine
similarity between windows: if any two windows of a modality look
identical, no contrastive loss can ever separate them.

| # | Bug | Evidence | Fix |
|---|-----|----------|-----|
| 1 | fMRI never standardized | cosine between *any* two fMRI windows = **0.999995**; per-voxel coefficient of variation 0.0016, i.e. 99.8% of the signal is constant BOLD baseline | z-score each voxel over time, per run → cosine **0.0032** |
| 2 | Text embeddings anisotropic | mean pairwise cosine of window embeddings **0.514** (BERT spaces share a large common component) | subtract the corpus mean vector → **0.0010** |
| 3 | MEG standardization silently a no-op | cached MEG windows had std **1e-4** instead of ~0.22 | see below → std **0.2350** |

Bug 3 is the subtle one. The standardization used `std + 1e-8`, but MEG is
stored in Tesla, so a channel's std is ~`4e-12` — the epsilon was **2500x
larger than the value it was guarding**, turning z-scoring into a ~2500x
shrink. The first conv layer's bias (~3e-2) then swamped the signal
(~1e-4), so every MEG embedding collapsed to the same vector (mean
off-diagonal cosine **1.000** from epoch 1, `meg_text` loss pinned at
`ln(64)`). Cosine similarity is scale-invariant, which is exactly why the
cache's own discriminability check could not see it. `scripts/build_window_cache.py`
now uses `_safe_std` (replace zero stds with 1.0, never add an absolute
epsilon) and asserts the resulting std is in a plausible range.

A fourth, architectural bug surfaced once the data was correct: the MEG
encoder ended in `AdaptiveAvgPool1d(1)`. MEG windows are ~zero-mean over
time and the stimulus information is in the temporal *pattern*, so global
average pooling discards precisely the signal. It was replaced by strided
convolutions plus a flatten, with a width-1 spatial filter over the 306
sensors as the front end.

Two supporting checks that validated the pipeline rather than breaking it:

- MEG inter-subject correlation is **+0.0104** for a matched stimulus
  window vs **+0.0002** for mismatched windows — 50x, which independently
  confirms the section/time alignment is correct. It is also small, which
  is why every window is now averaged across all subjects that have it
  (20 MEG, 35 fMRI) in `scripts/build_window_cache.py`.
- Slicing a 2 s window out of the `[306, n_samples]` per-run layout costs
  ~306 scattered reads; with cross-subject averaging an epoch went from
  110 s to over 10 minutes. The window cache stores windows contiguously,
  and an epoch is now **~1.2 s** with zero file I/O.

## Verified results (v2)

Job `1529545` (g3090, 60 epochs, ~1.2 s/epoch), same 2,931 aligned windows,
train sections 1-7 (2,247 windows) / val sections 8-9 (684 windows).
Chance median rank is 342, chance top-10 is 10/684 = 0.0146.

All numbers below come from the **single saved checkpoint** (epoch 15)
evaluated by `eval_alignment.py` (job `1529549`), never from a per-epoch
training line. Quoting each metric from whichever epoch happens to favour
it inflates the result: at epochs 7-9 this run showed fMRI→text 144/0.067
and MEG→text 271/0.040, both better than the model actually saved.

| direction | global medR (chance 341.5) | global top-10 (chance 0.0146) | in-section medR (chance 174.8) | in-section top-10 (chance 0.0285) |
|---|---|---|---|---|
| fMRI → text | **138.5** | **0.0789** (5.4x) | **69.0** | **0.1287** (4.5x) |
| MEG → fMRI  | **249.0** | **0.0234** (1.6x) | **126.5** | **0.0482** (1.7x) |
| MEG → text  | **277.0** | **0.0278** (1.9x) | **142.5** | **0.0497** (1.7x) |

The in-section columns restrict candidates to the query's own section and
are the protocol that rules out section identity as an explanation; see
RESULTS.md §4.3. All three directions beat chance under both protocols.
Embedding spread ended at meg=0.036, fmri=0.006, text=0.011, i.e. no
collapse anywhere.

The model peaks around epoch 7-15 and then overfits: training loss falls to
0.14 while validation metrics flatten. With 2,247 training windows that is
expected; see "Next steps".

## Next steps

Ordered by expected value, none of them implemented here:

1. **More windows.** 2,247 training windows is the binding constraint. A
   1 s hop with 2 s windows (50% overlap) roughly doubles the count; the
   section-level split keeps it leakage-free.
2. **Per-subject samples instead of one cohort average.** Averaging buys
   SNR but throws away 20x the sample count. Storing per-subject windows
   contiguously would let each subject be its own training example.
3. **Regularization sweep.** Dropout is 0.3 and weight decay 0.01; neither
   was tuned.
4. **Better fMRI features.** Pooling is a fixed 8x9x8 grid over the whole
   volume with no brain mask — 148 of 576 bins are background.
5. **Sentence-level text embeddings.** Windows currently average ~6.8 word
   vectors; a sentence encoder over the window's text would be sharper.

## Known limitations

- fMRI subject coverage is capped at the 35 CN subjects with author
  derivatives; no brain mask is applied before spatial pooling.
- The MEG unification pipeline is a documented sensor-space baseline, not
  a claim of equivalence to the author's (unknown) cleaning pipeline for
  the 10 derivative subjects.
- Because positives are keyed by `(section, t)` and not by subject, the
  same window can appear multiple times across an epoch with different
  randomly-chosen subject realizations (no explicit subject-repeat
  negative filtering in this first version).
