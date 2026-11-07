"""Pool every TR of each preprocessed CN fMRI run down to a small spatial
grid and cache the result.

This does not re-preprocess fMRI in any scientific sense -- the input is
still exactly the author-provided ``*_desc-preproc_bold.nii.gz`` volumes,
untouched. It only extracts a compact per-TR feature (deterministic
average pooling), which:

  * shrinks each run from ~550 MB (float32 4D volume) to well under 1 MB,
  * removes the need to decompress the .nii.gz at every training step, and
  * lets ``dataset.py`` build any of the 4 delayed features for a window
    with a cheap array index instead of a full-volume load + pool.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import pool_volume  # noqa: E402

_FMRI_RE = re.compile(
    r"sub-(?P<subject>[A-Za-z0-9]+)_task-lppCN_run-(?P<run>\d+)_space-MNIColin27_desc-preproc_bold\.nii\.gz$"
)


def process_subject(subject: str, config: dict) -> None:
    fmri_root = Path(config["data"]["fmri_root"]) / "derivatives" / f"sub-{subject}" / "func"
    cache_root = Path(config["project_dir"]) / config["data"]["fmri_pooled_cache"]
    pool_size = tuple(config["model"]["fmri_pool_size"])
    tr_seconds = float(config["alignment"]["fmri_tr_seconds"])

    if not fmri_root.exists():
        raise FileNotFoundError(f"No fMRI derivatives found for sub-{subject} under {fmri_root}")

    run_files = sorted(p for p in fmri_root.glob("*.nii.gz") if _FMRI_RE.search(p.name))
    if not run_files:
        raise FileNotFoundError(f"No task-lppCN preproc BOLD files matched under {fmri_root}")

    out_dir = cache_root / f"sub-{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in run_files:
        m = _FMRI_RE.search(path.name)
        run = m.group("run")
        out_npy = out_dir / f"sub-{subject}_task-lppCN_run-{run}_desc-pooled_bold.npy"
        out_json = out_dir / f"sub-{subject}_task-lppCN_run-{run}_desc-pooled_bold.json"
        if out_npy.exists() and out_json.exists():
            print(f"[skip] {out_npy} already exists")
            continue

        print(f"[run] sub-{subject} run-{run}: reading {path}")
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)  # [X, Y, Z, T]
        n_tr = data.shape[3]

        pooled = np.stack([pool_volume(data[..., t], pool_size) for t in range(n_tr)], axis=0)
        pooled = pooled.astype(np.float32)

        np.save(out_npy, pooled)
        meta = {
            "subject": subject,
            "run": int(run),
            "n_tr": n_tr,
            "tr_seconds": tr_seconds,
            "pool_size": list(pool_size),
            "source_bold": str(path),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[done] {out_npy} shape={pooled.shape}")

        del img, data, pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--subject", type=str, required=True, help="e.g. CN028")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    process_subject(args.subject, config)


if __name__ == "__main__":
    sys.exit(main())
