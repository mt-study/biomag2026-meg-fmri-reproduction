"""Materialize the aligned windows into contiguous, window-major arrays.

Why this exists
---------------
The per-run caches store MEG as ``[306 channels, n_samples]``. Slicing a 2 s
window out of that touches 306 *non-contiguous* 800-byte spans, so one
training sample costs ~306 scattered reads per subject. With cross-subject
averaging that became ~5.5M tiny GPFS reads per epoch and an epoch went from
~110 s to over 10 minutes.

This script does the expensive part exactly once, subject-major (each run
file is read sequentially, once), and writes window-major arrays:

    meg_windows.npy    [n_windows, n_channels, n_times]   float32
    fmri_windows.npy   [n_windows, fmri_dim]              float32
    text_windows.npy   [n_windows, text_dim]              float32
    windows_meta.npz   section[], t[], meg_n[], fmri_n[]

Each entry is already standardized (MEG per channel, fMRI per voxel, both
per run) and averaged across subjects, which is also the SNR fix: measured
MEG inter-subject correlation for a matched window is only ~+0.010, so a
single subject's 2 s response is mostly noise.

Training then reads whole rows out of RAM, and an epoch is I/O-free.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alignment import AlignmentIndex  # noqa: E402

_EPS = 1e-8


def _safe_std(std: np.ndarray) -> np.ndarray:
    """Guard a standard deviation without distorting its scale.

    Adding an absolute epsilon here is a trap: MEG is stored in Tesla, so a
    channel's std is ~4e-12 and ``std + 1e-8`` is 2500x the real value. That
    silently turns z-scoring into a ~2500x shrink -- the cached windows came
    out with std 1e-4 instead of ~1, the first conv layer's bias (~3e-2) then
    swamped the signal, and every MEG embedding collapsed to the same vector.
    Cosine similarity is scale-invariant, so the cache's own discriminability
    check could not see it. Only genuinely dead channels are replaced.
    """
    return np.where(std > 0, std, 1.0)


def _index_meg_stats(blob) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Index MEG channel stats by exact path and by unique filename."""
    by_path = {}
    by_name = {}
    duplicate_names = set()
    for key, mean, std in zip(blob["keys"], blob["mean"], blob["std"]):
        value = (
            mean[:, None].astype(np.float32),
            _safe_std(std[:, None].astype(np.float64)),
        )
        key_str = str(key)
        by_path[key_str] = value
        name = Path(key_str).name
        if name in by_name:
            duplicate_names.add(name)
        by_name[name] = value
    for name in duplicate_names:
        by_name.pop(name, None)
    return by_path, by_name


def _lookup_meg_stats(
    path: Path,
    by_path: dict[str, tuple[np.ndarray, np.ndarray]],
    by_name: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[tuple[np.ndarray, np.ndarray], bool]:
    key = str(path)
    if key in by_path:
        return by_path[key], False
    name = Path(path).name
    if name in by_name:
        return by_name[name], True
    raise KeyError(
        f"no MEG channel stats for {path}; rebuild channel_stats.npz or "
        "ensure cached MEG filenames are unique after project relocation"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    index = AlignmentIndex(config)
    windows = list(index.train_windows) + list(index.val_windows)
    n_windows = len(windows)
    n_ch = index.meg_channels
    n_time = index.meg_window_length
    fmri_dim = len(index.fmri_delay_seconds) * int(np.prod(index.fmri_pool_size))
    print(f"[windows] {n_windows} (train={len(index.train_windows)} val={len(index.val_windows)})")
    print(f"[dims] meg=[{n_ch},{n_time}] fmri={fmri_dim} text={index.text_dim}")

    meg_sum = np.zeros((n_windows, n_ch, n_time), dtype=np.float32)
    meg_cnt = np.zeros(n_windows, dtype=np.int32)
    fmri_sum = np.zeros((n_windows, fmri_dim), dtype=np.float32)
    fmri_cnt = np.zeros(n_windows, dtype=np.int32)
    text = np.zeros((n_windows, index.text_dim), dtype=np.float32)
    sections = np.zeros(n_windows, dtype=np.int32)
    times = np.zeros(n_windows, dtype=np.float32)

    # Invert the per-window option lists into per-file work lists so each
    # cached run file is opened exactly once.
    meg_work = defaultdict(list)   # path -> [(window_idx, t), ...]
    fmri_work = defaultdict(list)
    for i, w in enumerate(windows):
        sections[i] = w.section
        times[i] = w.t
        text[i] = w.text_embedding
        for _, path in w.meg_options:
            meg_work[path].append((i, w.t))
        for _, path in w.fmri_options:
            fmri_work[path].append((i, w.t))

    # ---- MEG ----------------------------------------------------------
    stats_path = Path(config["project_dir"]) / config["data"]["meg_unified_cache"] / "channel_stats.npz"
    blob = np.load(stats_path, allow_pickle=False)
    meg_stats, meg_stats_by_name = _index_meg_stats(blob)
    relocated_stats = 0

    for n, (path, items) in enumerate(sorted(meg_work.items()), 1):
        arr = np.load(path, mmap_mode="r")
        # float64 for the divide: Tesla-scale values (~4e-12) divided by an
        # equally tiny std lose precision badly in float32.
        data = np.asarray(arr, dtype=np.float64)          # one sequential read
        (mean, std), used_relocated_stats = _lookup_meg_stats(
            path, meg_stats, meg_stats_by_name
        )
        relocated_stats += int(used_relocated_stats)
        data = ((data - mean) / std).astype(np.float32)
        for idx, t in items:
            start = int(round((t - index.meg_tmin) * index.meg_sfreq))
            end = start + n_time
            if start < 0 or end > data.shape[1]:
                continue
            meg_sum[idx] += data[:, start:end]
            meg_cnt[idx] += 1
        if n % 20 == 0 or n == len(meg_work):
            print(f"[meg {n}/{len(meg_work)}] {Path(path).name}", flush=True)
    if relocated_stats:
        print(
            f"[meg stats] matched {relocated_stats} files by unique filename "
            "because channel_stats.npz stores relocated absolute paths",
            flush=True,
        )

    # ---- fMRI ---------------------------------------------------------
    for n, (path, items) in enumerate(sorted(fmri_work.items()), 1):
        data = np.load(path).astype(np.float64)
        data = data.reshape(data.shape[0], -1)
        data = ((data - data.mean(0, keepdims=True)) / _safe_std(data.std(0, keepdims=True))).astype(np.float32)
        n_tr = data.shape[0]
        for idx, t in items:
            feats = []
            for delay in index.fmri_delay_seconds:
                tr = int(round((t + delay) / index.fmri_tr_seconds))
                feats.append(data[min(max(tr, 0), n_tr - 1)])
            fmri_sum[idx] += np.concatenate(feats)
            fmri_cnt[idx] += 1
        if n % 40 == 0 or n == len(fmri_work):
            print(f"[fmri {n}/{len(fmri_work)}] {Path(path).name}", flush=True)

    if (meg_cnt == 0).any() or (fmri_cnt == 0).any():
        raise RuntimeError(
            f"{int((meg_cnt == 0).sum())} windows had no MEG contributor and "
            f"{int((fmri_cnt == 0).sum())} had no fMRI contributor"
        )

    # dividing float32 by an int array promotes to float64; keep float32 so
    # the cache stays ~720 MB and needs no per-batch dtype conversion.
    meg = (meg_sum / meg_cnt[:, None, None]).astype(np.float32)
    fmri = (fmri_sum / fmri_cnt[:, None]).astype(np.float32)

    if not np.isfinite(meg).all() or not np.isfinite(fmri).all():
        raise RuntimeError("non-finite values in the window cache")

    # Standardized features averaged over n subjects should have std ~1/sqrt(n)
    # given the weak inter-subject correlation, i.e. roughly 0.1-1.0. Anything
    # far outside that means the standardization silently did not take effect
    # (the 1e-8-epsilon-vs-Tesla-scale bug), which cosine-based checks cannot
    # detect because they are scale-invariant.
    for name, arr in (("meg", meg), ("fmri", fmri)):
        s = float(arr.std())
        if not (0.01 < s < 100.0):
            raise RuntimeError(
                f"{name} window cache has std={s:.3e}, far from the expected ~1/sqrt(n_subjects). "
                "Standardization almost certainly did not apply; check the per-run mean/std."
            )
        print(f"  {name} std after standardization + averaging: {s:.4f}")

    out = Path(config["project_dir"]) / config["data"].get("window_cache", "cache/windows")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "meg_windows.npy", meg)
    np.save(out / "fmri_windows.npy", fmri)
    np.save(out / "text_windows.npy", text)
    np.savez(
        out / "windows_meta.npz",
        section=sections, t=times, meg_n=meg_cnt, fmri_n=fmri_cnt,
        n_train=len(index.train_windows),
    )

    # Discriminability report: this is the number that told us v1 was broken
    # (raw fMRI windows had mean pairwise cosine 0.999995 == all identical).
    def cos(x):
        v = x.reshape(len(x), -1)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + _EPS)
        s = v @ v.T
        return s[~np.eye(len(s), dtype=bool)].mean()

    sel = np.random.default_rng(0).choice(n_windows, size=min(400, n_windows), replace=False)
    print(f"[done] {out}")
    print(f"  contributors per window: meg {meg_cnt.min()}-{meg_cnt.max()}, fmri {fmri_cnt.min()}-{fmri_cnt.max()}")
    print(f"  mean pairwise cosine (want << 1.0):  meg={cos(meg[sel]):+.4f}  "
          f"fmri={cos(fmri[sel]):+.4f}  text={cos(text[sel]):+.4f}")


if __name__ == "__main__":
    main()
