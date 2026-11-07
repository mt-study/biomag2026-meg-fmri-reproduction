"""PyTorch ``Dataset`` over the materialized window cache.

``scripts/build_window_cache.py`` has already done all the expensive work --
per-run standardization, cross-subject averaging, and the 4-delay fMRI
concatenation -- and written window-major arrays. This module just holds
those arrays in RAM (~750 MB) and hands out rows, so an epoch does no file
I/O at all.

Two properties of the cache matter for training and are asserted here:

* Standardization is baked in. Without it the model cannot learn: raw pooled
  BOLD is ~99.8% constant baseline, which makes any two fMRI windows 0.999995
  cosine-similar (i.e. indistinguishable).
* Each window is a cross-subject average. Single-subject 2 s MEG has an
  inter-subject correlation of only ~+0.010 for a matched window, so one
  subject's response is mostly noise.

The train/val split is by *stimulus section* (contiguous story segments), so
no window that shares a section with a training window can appear in
validation. Subjects are deliberately not a split axis -- MEG and fMRI
subjects are different people anyway.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowCacheDataset(Dataset):
    def __init__(
        self,
        meg: np.ndarray,
        fmri: np.ndarray,
        text: np.ndarray,
        sections: np.ndarray,
        times: np.ndarray,
        noise_std: float = 0.0,
        seed: int = 0,
    ):
        if len(meg) == 0:
            raise ValueError("WindowCacheDataset received zero windows")
        assert len(meg) == len(fmri) == len(text) == len(sections) == len(times)
        self.meg = meg
        self.fmri = fmri
        self.text = text
        self.sections = sections
        self.times = times
        self.noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.meg)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meg = self.meg[idx]
        fmri = self.fmri[idx]
        if self.noise_std > 0:
            # Cross-subject averaging removed the per-sample variability that
            # random subject draws used to provide, so add explicit jitter to
            # keep the small (2.2k window) training set from being memorized.
            meg = meg + self._rng.normal(0, self.noise_std, meg.shape).astype(np.float32)
            fmri = fmri + self._rng.normal(0, self.noise_std, fmri.shape).astype(np.float32)
        return {
            "meg": torch.from_numpy(np.ascontiguousarray(meg, dtype=np.float32)),
            "fmri": torch.from_numpy(np.ascontiguousarray(fmri, dtype=np.float32)),
            "text": torch.from_numpy(np.ascontiguousarray(self.text[idx], dtype=np.float32)),
            "section": int(self.sections[idx]),
            "t": float(self.times[idx]),
        }


def build_datasets(config: dict):
    """Load the window cache and split it back into train/val by section.

    Returns (train_dataset, val_dataset, info, dims). ``dims`` is read from
    the cached arrays, so MEG channel count, window length, fMRI feature dim
    and text dim are never hardcoded.
    """
    cache_mode = str(config.get("sampling", {}).get("window_cache_mode", "average"))
    if cache_mode == "subject_pair":
        from subject_pair_dataset import build_subject_pair_datasets

        return build_subject_pair_datasets(config)
    if cache_mode != "average":
        raise ValueError(f"unknown sampling.window_cache_mode {cache_mode!r}; expected 'average' or 'subject_pair'")

    cache_dir = Path(config["project_dir"]) / config["data"].get("window_cache", "cache/windows")
    required = ["meg_windows.npy", "fmri_windows.npy", "text_windows.npy", "windows_meta.npz"]
    missing = [f for f in required if not (cache_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Window cache incomplete at {cache_dir} (missing {', '.join(missing)}). "
            "Run `sbatch scripts/build_window_cache.sbatch` first."
        )

    meg = np.load(cache_dir / "meg_windows.npy")
    fmri = np.load(cache_dir / "fmri_windows.npy")
    text = np.load(cache_dir / "text_windows.npy")
    meta = np.load(cache_dir / "windows_meta.npz")
    sections, times = meta["section"], meta["t"]

    for name, arr in (("meg", meg), ("fmri", fmri), ("text", text)):
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite values in cached {name} windows; rebuild the window cache")

    align_cfg = config["alignment"]
    train_sections = set(align_cfg["train_sections"])
    val_sections = set(align_cfg["val_sections"])
    overlap = train_sections & val_sections
    if overlap:
        raise ValueError(f"train_sections and val_sections overlap on {sorted(overlap)}")

    train_mask = np.isin(sections, list(train_sections))
    val_mask = np.isin(sections, list(val_sections))
    if not train_mask.any():
        raise ValueError(f"No cached windows fall in train_sections {sorted(train_sections)}")
    if not val_mask.any():
        raise ValueError(f"No cached windows fall in val_sections {sorted(val_sections)}")

    noise_std = float(config.get("sampling", {}).get("noise_std", 0.0))
    seed = int(config["training"].get("seed", 0))

    def make(mask, ns, s):
        return WindowCacheDataset(
            meg[mask], fmri[mask], text[mask], sections[mask], times[mask], noise_std=ns, seed=s
        )

    train_ds = make(train_mask, noise_std, seed)
    val_ds = make(val_mask, 0.0, seed + 1)   # never perturb the eval set

    info = {
        "cache_mode": "average",
        "n_windows": len(meg),
        "train_sections": sorted(train_sections),
        "val_sections": sorted(val_sections),
        "meg_contributors": (int(meta["meg_n"].min()), int(meta["meg_n"].max())),
        "fmri_contributors": (int(meta["fmri_n"].min()), int(meta["fmri_n"].max())),
    }
    dims = {
        "meg_channels": meg.shape[1],
        "meg_time": meg.shape[2],
        "fmri_dim": fmri.shape[1],
        "text_dim": text.shape[1],
    }
    return train_ds, val_ds, info, dims
