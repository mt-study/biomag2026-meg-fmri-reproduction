"""Stimulus-window alignment between cached MEG runs, cached fMRI runs, and
cached word-level text embeddings.

This module only reads the project-owned caches under ``cache/`` (built by
``scripts/preprocess_meg_unified.py``, ``scripts/build_fmri_cache.py`` and
``scripts/build_text_cache.py``) plus the original annotation CSV for
onset/offset/section -- it never opens a raw FIF or NIfTI file itself, and
never writes into the original dataset roots.

Alignment rule: a MEG window and an fMRI window are a positive pair only if
they share the same stimulus ``section`` and the same window start time
``t`` (never by array index). Subject identity is metadata only -- MEG and
fMRI recordings do not need to come from the same participant.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def resolve_temporal_grid(align_cfg: dict) -> dict:
    """Resolve window sampling parameters without changing legacy defaults."""
    window_seconds = float(align_cfg["window_seconds"])
    delays = [float(value) for value in align_cfg["fmri_delay_seconds"]]
    stride_seconds = float(align_cfg.get("stride_seconds", window_seconds))
    grid_window_bound_seconds = float(
        align_cfg.get("grid_window_bound_seconds", window_seconds)
    )
    grid_delay_bound_seconds = float(
        align_cfg.get("grid_delay_bound_seconds", max(delays) if delays else 0.0)
    )
    grid_text_presence_seconds = float(
        align_cfg.get("grid_text_presence_seconds", window_seconds)
    )
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window_seconds and stride_seconds must be positive")
    if not delays or any(delay < 0 for delay in delays):
        raise ValueError("fmri_delay_seconds must contain non-negative values")
    if grid_text_presence_seconds <= 0 or grid_text_presence_seconds > window_seconds:
        raise ValueError(
            "grid_text_presence_seconds must be positive and no longer than window_seconds"
        )
    if grid_window_bound_seconds < window_seconds:
        raise ValueError(
            "grid_window_bound_seconds cannot be shorter than window_seconds"
        )
    if grid_delay_bound_seconds < max(delays):
        raise ValueError(
            "grid_delay_bound_seconds cannot be shorter than the largest delay"
        )
    return {
        "window_seconds": window_seconds,
        "fmri_delay_seconds": delays,
        "stride_seconds": stride_seconds,
        "grid_window_bound_seconds": grid_window_bound_seconds,
        "grid_delay_bound_seconds": grid_delay_bound_seconds,
        "grid_text_presence_seconds": grid_text_presence_seconds,
    }


def temporal_window_starts(
    max_available_seconds: float,
    stride_seconds: float,
    grid_window_bound_seconds: float,
    grid_delay_bound_seconds: float,
) -> List[float]:
    """Return starts on a fixed grid after reserving common end margins."""
    t_max = (
        float(max_available_seconds)
        - float(grid_delay_bound_seconds)
        - float(grid_window_bound_seconds)
    )
    if t_max < 0:
        return []
    return [
        i * float(stride_seconds)
        for i in range(int(t_max // float(stride_seconds)) + 1)
    ]


@dataclass
class WindowSample:
    section: int
    t: float
    meg_options: List[Tuple[str, Path]]
    fmri_options: List[Tuple[str, Path]]
    text_embedding: np.ndarray


@dataclass
class AlignmentStats:
    meg_files_found: int = 0
    fmri_files_found: int = 0
    meg_sample_windows: int = 0
    fmri_sample_windows: int = 0
    aligned_windows: int = 0
    skipped_no_annotation: int = 0
    skipped_missing_modality: int = 0
    skipped_out_of_bounds_sections: List[int] = field(default_factory=list)
    skipped_fmri_subjects_bad_run_count: List[str] = field(default_factory=list)

    def as_lines(self) -> List[str]:
        return [
            f"MEG cached runs found: {self.meg_files_found}",
            f"fMRI cached runs found: {self.fmri_files_found}",
            f"MEG candidate windows: {self.meg_sample_windows}",
            f"fMRI candidate windows: {self.fmri_sample_windows}",
            f"Successfully aligned windows (MEG+fMRI+text): {self.aligned_windows}",
            f"Skipped windows (no overlapping annotation): {self.skipped_no_annotation}",
            f"Skipped windows (modality missing for section): {self.skipped_missing_modality}",
            f"Sections with no usable windows: {self.skipped_out_of_bounds_sections}",
            f"fMRI subjects skipped (run count != 9): {self.skipped_fmri_subjects_bad_run_count}",
        ]


def _discover_meg_cache(cache_root: Path) -> Tuple[Dict[Tuple[str, int], Path], Dict[Tuple[str, int], dict]]:
    files: Dict[Tuple[str, int], Path] = {}
    meta: Dict[Tuple[str, int], dict] = {}
    for json_path in cache_root.rglob("*_desc-unifiedpreproc_meg.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        key = (m["subject"], int(m["run"]))
        npy_path = json_path.with_suffix(".npy")
        if not npy_path.exists():
            continue
        files[key] = npy_path
        meta[key] = m
    return files, meta


def _discover_fmri_cache(
    cache_root: Path, expected_sections_per_subject: int
) -> Tuple[Dict[Tuple[str, int], Path], Dict[Tuple[str, int], dict], List[str]]:
    """Group cached fMRI runs by subject and assign sections by chronological
    run order within that subject, NOT by a fixed run-number offset.

    Evidence this is necessary: the BIDS run numbers are session-global (the
    lppCN story runs are interleaved with other scanner tasks), so the run
    number where "section 1" starts varies per subject -- e.g. sub-CN001's
    9 lppCN runs are numbered 04-12, sub-CN003's are numbered 05-13. Within
    one subject, however, the 9 lppCN runs are always consecutive and in
    chronological (== section) order, and their durations line up with the
    9 known section durations from the annotation CSV in that order. See
    README.md "fMRI run-to-section mapping" for the full cross-check.
    """
    by_subject: Dict[str, List[dict]] = {}
    for json_path in cache_root.rglob("*_desc-pooled_bold.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        npy_path = json_path.with_suffix(".npy")
        if not npy_path.exists():
            continue
        m["_npy_path"] = npy_path
        by_subject.setdefault(m["subject"], []).append(m)

    files: Dict[Tuple[str, int], Path] = {}
    meta: Dict[Tuple[str, int], dict] = {}
    skipped_subjects: List[str] = []
    for subject, runs in by_subject.items():
        runs = sorted(runs, key=lambda m: int(m["run"]))
        if len(runs) != expected_sections_per_subject:
            skipped_subjects.append(subject)
            continue
        for section, m in enumerate(runs, start=1):
            key = (subject, section)
            files[key] = m["_npy_path"]
            meta[key] = m
    return files, meta, skipped_subjects


class AlignmentIndex:
    """Builds the full train/val window index from the on-disk caches."""

    def __init__(self, config: dict):
        data_cfg = config["data"]
        align_cfg = config["alignment"]
        project_dir = Path(config["project_dir"])

        self.meg_cache_root = project_dir / data_cfg["meg_unified_cache"]
        self.fmri_cache_root = project_dir / data_cfg["fmri_pooled_cache"]
        self.text_cache_path = project_dir / data_cfg["text_embedding_cache"]

        temporal = resolve_temporal_grid(align_cfg)
        self.window_seconds = temporal["window_seconds"]
        self.fmri_delay_seconds = temporal["fmri_delay_seconds"]
        self.stride_seconds = temporal["stride_seconds"]
        self.grid_window_bound_seconds = temporal["grid_window_bound_seconds"]
        self.grid_delay_bound_seconds = temporal["grid_delay_bound_seconds"]
        self.grid_text_presence_seconds = temporal["grid_text_presence_seconds"]
        self.fmri_tr_seconds = float(align_cfg.get("fmri_tr_seconds", 2.0))
        self.train_sections = set(align_cfg["train_sections"])
        self.val_sections = set(align_cfg["val_sections"])

        if not self.meg_cache_root.exists():
            raise FileNotFoundError(
                f"MEG unified cache does not exist: {self.meg_cache_root}. "
                "Run `sbatch scripts/preprocess_meg_unified.sbatch` first."
            )
        if not self.fmri_cache_root.exists():
            raise FileNotFoundError(
                f"fMRI pooled cache does not exist: {self.fmri_cache_root}. "
                "Run `sbatch scripts/build_fmri_cache.sbatch` first."
            )
        if not self.text_cache_path.exists():
            raise FileNotFoundError(
                f"Text embedding cache does not exist: {self.text_cache_path}. "
                "Run `sbatch scripts/build_text_cache.sbatch` first."
            )

        self.stats = AlignmentStats()

        n_sections = len(self.train_sections | self.val_sections)
        meg_files, meg_meta = _discover_meg_cache(self.meg_cache_root)
        fmri_files, fmri_meta, skipped_fmri_subjects = _discover_fmri_cache(
            self.fmri_cache_root, n_sections
        )
        self.stats.meg_files_found = len(meg_files)
        self.stats.fmri_files_found = len(fmri_files)
        self.stats.skipped_fmri_subjects_bad_run_count = skipped_fmri_subjects
        if not meg_files:
            raise FileNotFoundError(f"No cached MEG runs found under {self.meg_cache_root}")
        if not fmri_files:
            raise FileNotFoundError(f"No cached fMRI runs found under {self.fmri_cache_root}")

        center_text = bool(config.get("normalization", {}).get("text_mean_center", True))
        self.text_df = self._load_text(data_cfg, center_text=center_text)
        self.text_dim = self._text_embeddings.shape[1]

        self.meg_channels, self.meg_sfreq, meg_bounds, self.meg_tmin = self._meg_bounds(meg_meta)
        self.fmri_pool_size, fmri_bounds = self._fmri_bounds(fmri_meta)

        self.meg_files = meg_files
        self.fmri_files = fmri_files

        self.train_windows, self.val_windows = self._build_windows(
            meg_files, fmri_files, meg_bounds, fmri_bounds
        )

        self.meg_window_length = int(round(self.window_seconds * self.meg_sfreq))

    def _load_text(self, data_cfg: dict, center_text: bool = True) -> pd.DataFrame:
        info_df = pd.read_csv(data_cfg["text_information_csv"])
        embeddings = np.load(self.text_cache_path)
        if len(info_df) != embeddings.shape[0]:
            raise ValueError(
                f"Text annotation rows ({len(info_df)}) != cached embedding rows "
                f"({embeddings.shape[0]}); rebuild the text cache."
            )
        if center_text:
            # BERT embedding spaces are strongly anisotropic: every vector
            # shares a large common component, so averaging the ~7 words in a
            # 2 s window gives windows a mean pairwise cosine of ~0.51 -- the
            # contrastive target is then mostly "the same direction" for every
            # window. Removing the corpus mean vector decorrelates them.
            embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)
        self._text_embeddings = embeddings.astype(np.float32)
        info_df = info_df.copy()
        info_df["row"] = np.arange(len(info_df))
        return info_df

    def _meg_bounds(self, meg_meta: Dict[Tuple[str, int], dict]):
        channels: Optional[int] = None
        sfreq: Optional[float] = None
        tmin: Optional[float] = None
        bounds: Dict[Tuple[str, int], float] = {}
        for key, m in meg_meta.items():
            if channels is None:
                channels, sfreq, tmin = m["n_channels"], m["sfreq"], m["tmin"]
            elif m["n_channels"] != channels or abs(m["sfreq"] - sfreq) > 1e-6:
                raise ValueError(f"Inconsistent MEG cache metadata for {key}: {m}")
            # window t (stimulus-relative) is valid while t - tmin <= duration
            bounds[key] = m["duration_seconds"] + tmin
        return channels, sfreq, bounds, tmin

    def _fmri_bounds(self, fmri_meta: Dict[Tuple[str, int], dict]):
        pool_size: Optional[Tuple[int, int, int]] = None
        bounds: Dict[Tuple[str, int], float] = {}
        for key, m in fmri_meta.items():
            this_pool = tuple(m["pool_size"])
            if pool_size is None:
                pool_size = this_pool
            elif this_pool != pool_size:
                raise ValueError(f"Inconsistent fMRI pool size for {key}: {this_pool} != {pool_size}")
            bounds[key] = m["n_tr"] * m["tr_seconds"]
        return pool_size, bounds

    def _text_window_embedding(
        self, section: int, t: float, window_seconds: Optional[float] = None
    ) -> Optional[np.ndarray]:
        sub = self.text_df[self.text_df["section"] == section]
        duration = self.window_seconds if window_seconds is None else window_seconds
        mask = (sub["onset"].to_numpy() < t + duration) & (sub["offset"].to_numpy() > t)
        if not mask.any():
            return None
        rows = sub.loc[mask, "row"].to_numpy()
        return self._text_embeddings[rows].mean(axis=0).astype(np.float32)

    def _build_windows(self, meg_files, fmri_files, meg_bounds, fmri_bounds):
        train_windows: List[WindowSample] = []
        val_windows: List[WindowSample] = []

        all_sections = sorted(self.train_sections | self.val_sections)
        for section in all_sections:
            section_meg = [(sub, p) for (sub, sec), p in meg_files.items() if sec == section]
            section_fmri = [(sub, p) for (sub, sec), p in fmri_files.items() if sec == section]
            if not section_meg or not section_fmri:
                self.stats.skipped_out_of_bounds_sections.append(section)
                continue

            text_max_offset = self.text_df.loc[self.text_df["section"] == section, "offset"].max()
            meg_tmax = min(meg_bounds[(sub, section)] for sub, _ in section_meg)
            fmri_dur = min(fmri_bounds[(sub, section)] for sub, _ in section_fmri)
            starts = temporal_window_starts(
                min(text_max_offset, meg_tmax, fmri_dur),
                self.stride_seconds,
                self.grid_window_bound_seconds,
                self.grid_delay_bound_seconds,
            )
            if not starts:
                self.stats.skipped_out_of_bounds_sections.append(section)
                continue

            target = train_windows if section in self.train_sections else val_windows
            for t in starts:
                self.stats.meg_sample_windows += len(section_meg)
                self.stats.fmri_sample_windows += len(section_fmri)
                presence = self._text_window_embedding(
                    section, t, self.grid_text_presence_seconds
                )
                if presence is None:
                    self.stats.skipped_no_annotation += 1
                    continue
                text_emb = self._text_window_embedding(section, t)
                if text_emb is None:
                    self.stats.skipped_no_annotation += 1
                    continue
                target.append(
                    WindowSample(
                        section=section,
                        t=t,
                        meg_options=section_meg,
                        fmri_options=section_fmri,
                        text_embedding=text_emb,
                    )
                )
                self.stats.aligned_windows += 1
        return train_windows, val_windows
