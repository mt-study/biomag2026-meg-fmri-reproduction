"""Materialize per-subject MEG/fMRI windows for a non-averaged comparison.

The main cache averages all MEG subjects and all fMRI subjects for each
``(section, t)``. This script keeps subject axes instead:

    meg_windows.npy    [n_windows, n_meg_subjects, 306, n_times]
    fmri_windows.npy   [n_windows, n_fmri_subjects, fmri_dim]
    text_windows.npy   [n_windows, text_dim]
    windows_meta.npz   section[], t[], meg_subjects[], fmri_subjects[],
                       window/stride/delay sampling metadata

Training samples one MEG subject and one fMRI subject from the same stimulus
window. It does not claim those subjects are paired people; the cohorts are
disjoint and only the stimulus time is shared.
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


def _safe_std(std: np.ndarray) -> np.ndarray:
    return np.where(std > 0, std, 1.0)


def _index_meg_stats(blob) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Index MEG channel stats by exact path and by unique filename.

    Older caches can contain absolute paths from a previous project location.
    The filename fallback lets a relocated project reuse those stats without
    recomputing them, while still failing if names are ambiguous.
    """
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


def _seconds_label(seconds: float) -> str:
    value = float(seconds)
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/subject_pair.yaml")
    parser.add_argument("--window-seconds", type=float)
    parser.add_argument("--stride-seconds", type=float)
    parser.add_argument("--grid-window-bound-seconds", type=float)
    parser.add_argument("--grid-delay-bound-seconds", type=float)
    parser.add_argument("--grid-text-presence-seconds", type=float)
    parser.add_argument("--window-cache")
    parser.add_argument("--text-window-variants", nargs="+", type=float)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.window_seconds is not None:
        config["alignment"]["window_seconds"] = args.window_seconds
    if args.stride_seconds is not None:
        config["alignment"]["stride_seconds"] = args.stride_seconds
    if args.grid_window_bound_seconds is not None:
        config["alignment"]["grid_window_bound_seconds"] = (
            args.grid_window_bound_seconds
        )
    if args.grid_delay_bound_seconds is not None:
        config["alignment"]["grid_delay_bound_seconds"] = (
            args.grid_delay_bound_seconds
        )
    if args.grid_text_presence_seconds is not None:
        config["alignment"]["grid_text_presence_seconds"] = (
            args.grid_text_presence_seconds
        )
    if args.window_cache:
        config["data"]["window_cache"] = args.window_cache
    text_window_variants = (
        args.text_window_variants
        if args.text_window_variants is not None
        else config["alignment"].get("text_window_variants_seconds", [])
    )

    index = AlignmentIndex(config)
    text_window_variants = sorted(
        {float(value) for value in text_window_variants}
    )
    for value in text_window_variants:
        if (
            value < index.grid_text_presence_seconds
            or value > index.grid_window_bound_seconds
        ):
            raise ValueError(
                f"text window variant {value:g}s must be within "
                f"[{index.grid_text_presence_seconds:g}, "
                f"{index.grid_window_bound_seconds:g}]"
            )
    windows = list(index.train_windows) + list(index.val_windows)
    n_windows = len(windows)
    n_ch = index.meg_channels
    n_time = index.meg_window_length
    fmri_dim = len(index.fmri_delay_seconds) * int(np.prod(index.fmri_pool_size))

    meg_subjects = sorted({sub for w in windows for sub, _ in w.meg_options})
    fmri_subjects = sorted({sub for w in windows for sub, _ in w.fmri_options})
    meg_col = {s: i for i, s in enumerate(meg_subjects)}
    fmri_col = {s: i for i, s in enumerate(fmri_subjects)}

    print(f"[windows] {n_windows} (train={len(index.train_windows)} val={len(index.val_windows)})")
    print(f"[subjects] meg={len(meg_subjects)} fmri={len(fmri_subjects)}")
    print(f"[dims] meg=[{n_ch},{n_time}] fmri={fmri_dim} text={index.text_dim}")
    if text_window_variants:
        print(f"[text window variants] {text_window_variants}")

    out = Path(config["project_dir"]) / config["data"].get(
        "window_cache", "cache/windows_subject_pairs"
    )
    out.mkdir(parents=True, exist_ok=True)

    meg = np.lib.format.open_memmap(
        out / "meg_windows.npy", mode="w+", dtype=np.float32,
        shape=(n_windows, len(meg_subjects), n_ch, n_time),
    )
    fmri = np.lib.format.open_memmap(
        out / "fmri_windows.npy", mode="w+", dtype=np.float32,
        shape=(n_windows, len(fmri_subjects), fmri_dim),
    )
    text = np.zeros((n_windows, index.text_dim), dtype=np.float32)
    text_variants = {
        seconds: np.zeros((n_windows, index.text_dim), dtype=np.float32)
        for seconds in text_window_variants
    }
    sections = np.zeros(n_windows, dtype=np.int32)
    times = np.zeros(n_windows, dtype=np.float32)
    meg_cnt = np.zeros((n_windows, len(meg_subjects)), dtype=np.int16)
    fmri_cnt = np.zeros((n_windows, len(fmri_subjects)), dtype=np.int16)

    meg_work = defaultdict(list)
    fmri_work = defaultdict(list)
    for i, w in enumerate(windows):
        sections[i] = w.section
        times[i] = w.t
        text[i] = w.text_embedding
        for seconds, variant in text_variants.items():
            embedding = index._text_window_embedding(
                w.section, w.t, window_seconds=seconds
            )
            if embedding is None:
                raise RuntimeError(
                    f"no text for section={w.section} t={w.t:g} "
                    f"window={seconds:g}s"
                )
            variant[i] = embedding
        for sub, path in w.meg_options:
            meg_work[path].append((i, meg_col[sub], w.t))
        for sub, path in w.fmri_options:
            fmri_work[path].append((i, fmri_col[sub], w.t))

    stats_path = Path(config["project_dir"]) / config["data"]["meg_unified_cache"] / "channel_stats.npz"
    blob = np.load(stats_path, allow_pickle=False)
    meg_stats, meg_stats_by_name = _index_meg_stats(blob)
    relocated_stats = 0

    for n, (path, items) in enumerate(sorted(meg_work.items()), 1):
        arr = np.load(path, mmap_mode="r")
        data = np.asarray(arr, dtype=np.float64)
        (mean, std), used_relocated_stats = _lookup_meg_stats(
            path, meg_stats, meg_stats_by_name
        )
        relocated_stats += int(used_relocated_stats)
        data = ((data - mean) / std).astype(np.float32)
        for idx, subj_idx, t in items:
            start = int(round((t - index.meg_tmin) * index.meg_sfreq))
            end = start + n_time
            if start < 0 or end > data.shape[1]:
                continue
            meg[idx, subj_idx] = data[:, start:end]
            meg_cnt[idx, subj_idx] += 1
        if n % 20 == 0 or n == len(meg_work):
            print(f"[meg {n}/{len(meg_work)}] {Path(path).name}", flush=True)
    if relocated_stats:
        print(
            f"[meg stats] matched {relocated_stats} files by unique filename "
            "because channel_stats.npz stores relocated absolute paths",
            flush=True,
        )

    for n, (path, items) in enumerate(sorted(fmri_work.items()), 1):
        data = np.load(path).astype(np.float64)
        data = data.reshape(data.shape[0], -1)
        data = ((data - data.mean(0, keepdims=True)) / _safe_std(data.std(0, keepdims=True))).astype(np.float32)
        n_tr = data.shape[0]
        for idx, subj_idx, t in items:
            feats = []
            for delay in index.fmri_delay_seconds:
                tr = int(round((t + delay) / index.fmri_tr_seconds))
                feats.append(data[min(max(tr, 0), n_tr - 1)])
            fmri[idx, subj_idx] = np.concatenate(feats)
            fmri_cnt[idx, subj_idx] += 1
        if n % 40 == 0 or n == len(fmri_work):
            print(f"[fmri {n}/{len(fmri_work)}] {Path(path).name}", flush=True)

    if (meg_cnt != 1).any() or (fmri_cnt != 1).any():
        raise RuntimeError(
            f"subject-pair cache incomplete: MEG missing/duplicate cells={(meg_cnt != 1).sum()}, "
            f"fMRI missing/duplicate cells={(fmri_cnt != 1).sum()}"
        )
    if not np.isfinite(text).all() or any(
        not np.isfinite(value).all() for value in text_variants.values()
    ):
        raise RuntimeError("non-finite values in text windows")

    rng = np.random.default_rng(0)
    win_sel = rng.choice(n_windows, size=min(400, n_windows), replace=False)
    meg_sel = rng.integers(0, len(meg_subjects), size=len(win_sel))
    fmri_sel = rng.integers(0, len(fmri_subjects), size=len(win_sel))
    print(f"  sampled subject-window std: meg={float(meg[win_sel, meg_sel].std()):.4f} "
          f"fmri={float(fmri[win_sel, fmri_sel].std()):.4f}")

    np.save(out / "text_windows.npy", text)
    text_variant_files = []
    for seconds, variant in text_variants.items():
        filename = f"text_windows_{_seconds_label(seconds)}s.npy"
        np.save(out / filename, variant)
        text_variant_files.append(filename)
    np.savez(
        out / "windows_meta.npz",
        section=sections,
        t=times,
        meg_subjects=np.array(meg_subjects),
        fmri_subjects=np.array(fmri_subjects),
        n_train=len(index.train_windows),
        window_seconds=np.float32(index.window_seconds),
        meg_sfreq=np.float32(index.meg_sfreq),
        text_window_variants_seconds=np.asarray(
            text_window_variants, dtype=np.float32
        ),
        text_window_variant_files=np.asarray(text_variant_files),
        stride_seconds=np.float32(index.stride_seconds),
        fmri_delay_seconds=np.asarray(index.fmri_delay_seconds, dtype=np.float32),
        grid_window_bound_seconds=np.float32(index.grid_window_bound_seconds),
        grid_delay_bound_seconds=np.float32(index.grid_delay_bound_seconds),
        grid_text_presence_seconds=np.float32(index.grid_text_presence_seconds),
    )
    meg.flush()
    fmri.flush()
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
