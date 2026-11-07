"""Subject-disjoint datasets for strict unseen-subject generalization."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from window_selection import meg_window_time_length, text_window_cache_file


class SubjectSubsetWindowDataset(Dataset):
    """Window dataset exposing only explicitly selected subject axes."""

    def __init__(
        self,
        meg: np.ndarray,
        fmri: np.ndarray,
        text: np.ndarray,
        sections: np.ndarray,
        times: np.ndarray,
        window_indices: Sequence[int],
        meg_subject_indices: Sequence[int],
        fmri_subject_indices: Sequence[int],
        meg_time_length: int | None = None,
    ):
        self.meg = meg
        self.fmri = fmri
        self.text = text
        self.sections = sections
        self.times = times
        self.window_indices = np.asarray(window_indices, dtype=np.int64)
        self.meg_subject_indices = np.asarray(meg_subject_indices, dtype=np.int64)
        self.fmri_subject_indices = np.asarray(fmri_subject_indices, dtype=np.int64)
        # A shared cache stores the longest MEG window; a shorter condition
        # reads only the leading samples of every row.
        if meg_time_length is not None:
            if meg_time_length <= 0 or meg_time_length > meg.shape[3]:
                raise ValueError(
                    f"meg_time_length {meg_time_length} is outside the cached "
                    f"length {meg.shape[3]}"
                )
        self.meg_time_length = (
            None if meg_time_length is None else int(meg_time_length)
        )
        if not len(self.window_indices):
            raise ValueError("subject subset dataset has no windows")
        if not len(self.meg_subject_indices) or not len(self.fmri_subject_indices):
            raise ValueError("subject subset dataset has an empty modality")

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window_idx = int(self.window_indices[idx])
        meg_view = (
            self.meg[window_idx, self.meg_subject_indices]
            if self.meg_time_length is None
            else self.meg[
                window_idx, self.meg_subject_indices, :, : self.meg_time_length
            ]
        )
        meg = np.array(meg_view, dtype=np.float32, copy=True)
        fmri = np.array(
            self.fmri[window_idx, self.fmri_subject_indices],
            dtype=np.float32,
            copy=True,
        )
        return {
            "meg": torch.from_numpy(meg),
            "fmri": torch.from_numpy(fmri),
            "text": torch.from_numpy(
                np.array(self.text[window_idx], dtype=np.float32, copy=True)
            ),
            "section": int(self.sections[window_idx]),
            "t": float(self.times[window_idx]),
        }


def subject_fold_indices(
    n_subjects: int, fold: int, n_folds: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return train/test indices, assigning every subject to one test fold."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_subjects < n_folds:
        raise ValueError(f"{n_subjects} subjects cannot be split into {n_folds} folds")
    if fold < 0 or fold >= n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
    all_indices = np.arange(n_subjects, dtype=np.int64)
    test_indices = np.array_split(all_indices, n_folds)[fold]
    train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
    if np.intersect1d(train_indices, test_indices).size:
        raise AssertionError("subject train/test folds overlap")
    return train_indices, test_indices


def _subject_names(meta, key: str, count: int) -> list[str]:
    if key in meta.files:
        names = [str(value) for value in meta[key].tolist()]
    else:
        names = [f"{key}_{idx:03d}" for idx in range(count)]
    if len(names) != count:
        raise ValueError(f"{key} has {len(names)} labels for {count} cached subjects")
    return names


def build_cross_subject_datasets(config: dict, fold: int):
    """Build train-subject train/val datasets and an unseen-subject test set."""
    cache_dir = Path(config["project_dir"]) / config["data"].get(
        "window_cache", "cache/windows_subject_pairs"
    )
    required = ["meg_windows.npy", "fmri_windows.npy", "text_windows.npy", "windows_meta.npz"]
    missing = [name for name in required if not (cache_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"subject cache incomplete at {cache_dir}: missing {', '.join(missing)}"
        )

    meg = np.load(cache_dir / "meg_windows.npy", mmap_mode="r")
    fmri = np.load(cache_dir / "fmri_windows.npy", mmap_mode="r")
    meta = np.load(cache_dir / "windows_meta.npz", allow_pickle=False)
    sections = meta["section"]
    times = meta["t"]

    # Optional MEG+text window-length selection. Without it the cache is read
    # exactly as before, so every earlier cross-subject run is unaffected.
    requested_window = config.get("data", {}).get("window_subset_seconds")
    meg_time_length, cached_window, meg_sfreq = meg_window_time_length(
        meta, int(meg.shape[3]), requested_window
    )
    text_filename, selected_text_window = text_window_cache_file(
        meta, requested_window
    )
    text_path = cache_dir / text_filename
    if not text_path.exists():
        raise FileNotFoundError(
            f"subject cache at {cache_dir} is missing text file {text_filename}"
        )
    text = np.load(text_path, mmap_mode="r")
    if not (len(meg) == len(fmri) == len(text) == len(sections) == len(times)):
        raise ValueError("window cache arrays have inconsistent lengths")

    split_cfg = config["cross_subject"]
    n_folds = int(split_cfg.get("n_folds", 5))
    meg_train, meg_test = subject_fold_indices(meg.shape[1], fold, n_folds)
    fmri_train, fmri_test = subject_fold_indices(fmri.shape[1], fold, n_folds)

    train_sections = set(split_cfg["train_sections"])
    val_sections = set(split_cfg["val_sections"])
    test_sections = set(split_cfg["test_sections"])
    if train_sections & val_sections or train_sections & test_sections or val_sections & test_sections:
        raise ValueError("cross-subject train/val/test sections must be disjoint")

    window_splits = {
        "train": np.flatnonzero(np.isin(sections, sorted(train_sections))),
        "val": np.flatnonzero(np.isin(sections, sorted(val_sections))),
        "test": np.flatnonzero(np.isin(sections, sorted(test_sections))),
    }
    if any(not len(indices) for indices in window_splits.values()):
        raise ValueError("at least one cross-subject window split is empty")

    datasets = {
        name: SubjectSubsetWindowDataset(
            meg,
            fmri,
            text,
            sections,
            times,
            window_splits[name],
            meg_indices,
            fmri_indices,
            meg_time_length=meg_time_length,
        )
        for name, meg_indices, fmri_indices in (
            ("train", meg_train, fmri_train),
            ("val", meg_train, fmri_train),
            ("test", meg_test, fmri_test),
        )
    }

    meg_names = _subject_names(meta, "meg_subjects", meg.shape[1])
    fmri_names = _subject_names(meta, "fmri_subjects", fmri.shape[1])
    info = {
        "fold": int(fold),
        "n_folds": n_folds,
        "train_sections": sorted(train_sections),
        "val_sections": sorted(val_sections),
        "test_sections": sorted(test_sections),
        "train_windows": int(len(datasets["train"])),
        "val_windows": int(len(datasets["val"])),
        "test_windows": int(len(datasets["test"])),
        "meg_train_indices": meg_train.tolist(),
        "meg_test_indices": meg_test.tolist(),
        "fmri_train_indices": fmri_train.tolist(),
        "fmri_test_indices": fmri_test.tolist(),
        "meg_train_subjects": [meg_names[idx] for idx in meg_train],
        "meg_test_subjects": [meg_names[idx] for idx in meg_test],
        "fmri_train_subjects": [fmri_names[idx] for idx in fmri_train],
        "fmri_test_subjects": [fmri_names[idx] for idx in fmri_test],
        "temporal_sampling": {
            "cached_meg_window_seconds": cached_window,
            "selected_meg_window_seconds": (
                cached_window if requested_window is None else float(requested_window)
            ),
            "selected_meg_time_samples": (
                int(meg.shape[3]) if meg_time_length is None else int(meg_time_length)
            ),
            "meg_sfreq": meg_sfreq,
            "selected_text_window_seconds": selected_text_window,
            "selected_text_window_file": text_filename,
            "window_cache": str(cache_dir),
        },
    }
    if set(info["meg_train_subjects"]) & set(info["meg_test_subjects"]):
        raise AssertionError("MEG subject leakage")
    if set(info["fmri_train_subjects"]) & set(info["fmri_test_subjects"]):
        raise AssertionError("fMRI subject leakage")

    dims = {
        "meg_channels": int(meg.shape[2]),
        "meg_time": (
            int(meg.shape[3]) if meg_time_length is None else int(meg_time_length)
        ),
        "fmri_dim": int(fmri.shape[2]),
        "text_dim": int(text.shape[1]),
    }
    return datasets, info, dims
