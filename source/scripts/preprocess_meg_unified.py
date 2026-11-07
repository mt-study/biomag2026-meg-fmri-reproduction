"""Build one uniform MEG preprocessing pass over ALL 20 subjects' raw FIF
files (only 10/20 ship author-preprocessed derivatives).

Method and justification (see README.md "MEG unification" for the full
evidence): each raw per-run FIF file (329 channels, 1000 Hz, exactly one
run's duration) is

  1. restricted to the 306 MEG sensors (102 magnetometers + 204
     gradiometers), dropping EOG/ECG/STI/etc, matching the channel
     composition of the 10 author-provided derivatives exactly;
  2. band-pass filtered 0.1-40 Hz -- a standard MEG analysis band that
     removes slow drift and high-frequency/muscle noise while preserving
     cortical dynamics, safely below the acquisition's own 0.1-330 Hz
     hardware filter;
  3. resampled to 100 Hz -- exactly the sample rate already used by the
     10-subject author derivatives, so all 20 subjects share one array
     convention.

Stimulus-onset alignment: for every raw run checked (sub-01 runs 1, 6, 9),
the raw file's own duration matches the corresponding author-derivative
epoch's time span (tmax - tmin) to within ~10 ms. That means raw sample 0
already corresponds to the derivative's tmin = -0.1 s (i.e. the story
begins ~0.1 s into the recording). No cropping is applied; each run is
kept in full and tagged with tmin = -0.1 s so window indexing
(``round((t - tmin) * sfreq)``) is identical to the derivative convention
for every one of the 20 subjects. (The stim-channel trigger found via
``mne.find_events`` on sub-01 run-1 sits at t=27.4s, which is inconsistent
with the duration cross-check and is therefore NOT used for cropping.)

Output: float32 ``[306, T]`` arrays plus a JSON provenance sidecar under
``cache/meg_unified/`` (a project-owned cache -- the original dataset is
never modified).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import mne
import numpy as np
import yaml

mne.set_log_level("ERROR")


def process_subject(subject: str, config: dict) -> None:
    meg_root = Path(config["data"]["meg_root"])
    cache_root = Path(config["project_dir"]) / config["data"]["meg_unified_cache"]
    prep = config["meg_preprocessing"]

    raw_dir = meg_root / f"sub-{subject}" / "meg"
    run_files = sorted(raw_dir.glob(f"sub-{subject}_task-lppCN_run-*_meg.fif"))
    if not run_files:
        raise FileNotFoundError(f"No raw MEG files found for sub-{subject} under {raw_dir}")

    out_dir = cache_root / f"sub-{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in run_files:
        run = raw_path.name.split("run-")[1].split("_")[0]
        out_npy = out_dir / f"sub-{subject}_task-lppCN_run-{run}_desc-unifiedpreproc_meg.npy"
        out_json = out_dir / f"sub-{subject}_task-lppCN_run-{run}_desc-unifiedpreproc_meg.json"
        if out_npy.exists() and out_json.exists():
            print(f"[skip] {out_npy} already exists")
            continue

        print(f"[run] sub-{subject} run-{run}: reading {raw_path}")
        raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose=False)
        raw.pick(prep["pick_types"])
        raw.filter(l_freq=prep["highpass_hz"], h_freq=prep["lowpass_hz"], fir_design="firwin", verbose=False)
        raw.resample(prep["resample_hz"], verbose=False)

        data = raw.get_data().astype(np.float32)
        n_channels, n_times = data.shape
        sfreq = float(raw.info["sfreq"])

        np.save(out_npy, data)
        meta = {
            "subject": subject,
            "run": int(run),
            "n_channels": n_channels,
            "n_times": n_times,
            "sfreq": sfreq,
            "tmin": float(prep["epoch_tmin_seconds"]),
            "duration_seconds": n_times / sfreq,
            "source_raw": str(raw_path),
            "highpass_hz": prep["highpass_hz"],
            "lowpass_hz": prep["lowpass_hz"],
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[done] {out_npy} shape={data.shape} sfreq={sfreq}")

        del raw, data
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--subject", type=str, required=True, help="two-digit subject id, e.g. 01")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    process_subject(args.subject, config)


if __name__ == "__main__":
    sys.exit(main())
