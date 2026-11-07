"""Dataset over per-subject MEG/fMRI window caches.

This is a comparison path for the main averaged-window experiment. Each
sample keeps subject-level variation: one MEG subject window, one fMRI subject
window with the same stimulus ``(section, t)``, and the corresponding text
embedding. MEG and fMRI subjects are not paired people; they are independent
cohorts linked only by the shared stimulus time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class SubjectPairWindowCacheDataset(Dataset):
    def __init__(
        self,
        meg: np.ndarray,
        fmri: np.ndarray,
        text: np.ndarray,
        sections: np.ndarray,
        times: np.ndarray,
        indices: np.ndarray,
        train: bool,
        seed: int = 0,
        samples_per_window: int = 1,
    ):
        if len(indices) == 0:
            raise ValueError("SubjectPairWindowCacheDataset received zero windows")
        assert len(meg) == len(fmri) == len(text) == len(sections) == len(times)
        self.meg = meg
        self.fmri = fmri
        self.text = text
        self.sections = sections
        self.times = times
        self.train = bool(train)
        self.seed = int(seed)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.samples_per_window = max(1, int(samples_per_window)) if train else 1
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices) * self.samples_per_window

    def _fixed_pair(self, base_idx: int) -> tuple[int, int]:
        # Deterministic validation pair. This gives stable metrics without
        # averaging subjects back together or enumerating all 20*35 pairs.
        meg_idx = (base_idx * 1103515245 + self.seed) % self.meg.shape[1]
        fmri_idx = (base_idx * 1664525 + self.seed + 1013904223) % self.fmri.shape[1]
        return int(meg_idx), int(fmri_idx)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_pos = idx % len(self.indices)
        base_idx = int(self.indices[base_pos])
        if self.train:
            meg_idx = int(self._rng.integers(0, self.meg.shape[1]))
            fmri_idx = int(self._rng.integers(0, self.fmri.shape[1]))
        else:
            meg_idx, fmri_idx = self._fixed_pair(base_idx)

        meg = self.meg[base_idx, meg_idx]
        fmri = self.fmri[base_idx, fmri_idx]
        return {
            "meg": torch.from_numpy(np.ascontiguousarray(meg, dtype=np.float32)),
            "fmri": torch.from_numpy(np.ascontiguousarray(fmri, dtype=np.float32)),
            "text": torch.from_numpy(np.ascontiguousarray(self.text[base_idx], dtype=np.float32)),
            "section": int(self.sections[base_idx]),
            "t": float(self.times[base_idx]),
            "meg_subject_idx": int(meg_idx),
            "fmri_subject_idx": int(fmri_idx),
        }


def build_subject_pair_datasets(config: dict):
    cache_dir = Path(config["project_dir"]) / config["data"].get(
        "window_cache", "cache/windows_subject_pairs"
    )
    required = ["meg_windows.npy", "fmri_windows.npy", "text_windows.npy", "windows_meta.npz"]
    missing = [f for f in required if not (cache_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Subject-pair window cache incomplete at {cache_dir} (missing {', '.join(missing)}). "
            "Run `sbatch scripts/build_subject_pair_window_cache.sbatch` first."
        )

    meg = np.load(cache_dir / "meg_windows.npy", mmap_mode="r")
    fmri = np.load(cache_dir / "fmri_windows.npy", mmap_mode="r")
    text = np.load(cache_dir / "text_windows.npy")
    meta = np.load(cache_dir / "windows_meta.npz", allow_pickle=False)
    sections, times = meta["section"], meta["t"]

    if not np.isfinite(text).all():
        raise ValueError("non-finite values in cached text windows; rebuild the subject-pair cache")

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

    seed = int(config["training"].get("seed", 0))
    samples_per_window = int(config.get("sampling", {}).get("subject_pair_samples_per_window", 1))

    def make(mask, train, s):
        indices = np.flatnonzero(mask)
        return SubjectPairWindowCacheDataset(
            meg, fmri, text, sections, times, indices,
            train=train, seed=s, samples_per_window=samples_per_window,
        )

    train_ds = make(train_mask, True, seed)
    val_ds = make(val_mask, False, seed + 1)

    info = {
        "cache_mode": "subject_pair",
        "n_windows": len(text),
        "train_sections": sorted(train_sections),
        "val_sections": sorted(val_sections),
        "meg_subjects": int(meg.shape[1]),
        "fmri_subjects": int(fmri.shape[1]),
        "subject_pair_samples_per_window": samples_per_window,
    }
    dims = {
        "meg_channels": meg.shape[2],
        "meg_time": meg.shape[3],
        "fmri_dim": fmri.shape[2],
        "text_dim": text.shape[1],
    }
    return train_ds, val_ds, info, dims
