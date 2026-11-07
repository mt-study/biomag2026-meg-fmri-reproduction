"""Temporal window selection helpers shared by every runner that reads a
multi-window cache.

The 2/4/6/8 s conditions all read one cache that stores the longest MEG
window plus one text feature file per window length, so both the MEG prefix
length and the text file are resolved from the cache metadata rather than
from separate caches.
"""
from __future__ import annotations

import numpy as np


def meg_window_time_length(
    meta,
    cached_time_length: int,
    requested_seconds: float | None,
) -> tuple[int | None, float | None, float | None]:
    """Resolve a prefix length from temporal metadata in a shared MEG cache."""
    cached_seconds = (
        float(np.asarray(meta["window_seconds"]).item())
        if "window_seconds" in meta.files
        else None
    )
    sfreq = (
        float(np.asarray(meta["meg_sfreq"]).item())
        if "meg_sfreq" in meta.files
        else None
    )
    if requested_seconds is None:
        return None, cached_seconds, sfreq
    requested = float(requested_seconds)
    if requested <= 0:
        raise ValueError("window_subset_seconds must be positive")
    if cached_seconds is None or sfreq is None:
        raise ValueError(
            "MEG window selection requires window_seconds and meg_sfreq "
            "in windows_meta.npz"
        )
    if requested > cached_seconds + 1e-6:
        raise ValueError(
            f"requested {requested:g}s MEG window exceeds "
            f"cached {cached_seconds:g}s"
        )
    length = int(round(requested * sfreq))
    if length <= 0 or length > cached_time_length:
        raise ValueError(
            f"requested {requested:g}s maps to invalid MEG length {length}"
        )
    return length, cached_seconds, sfreq


def text_window_cache_file(
    meta,
    requested_seconds: float | None,
) -> tuple[str, float | None]:
    """Select the text feature file matching a requested temporal window."""
    if requested_seconds is None:
        cached = (
            float(np.asarray(meta["window_seconds"]).item())
            if "window_seconds" in meta.files
            else None
        )
        return "text_windows.npy", cached
    requested = float(requested_seconds)
    if (
        "text_window_variants_seconds" not in meta.files
        or "text_window_variant_files" not in meta.files
    ):
        raise ValueError(
            "text window selection requires variant metadata in windows_meta.npz"
        )
    variants = [
        float(value) for value in meta["text_window_variants_seconds"].tolist()
    ]
    files = [str(value) for value in meta["text_window_variant_files"].tolist()]
    if len(variants) != len(files):
        raise ValueError("text window variant metadata lengths differ")
    matches = [
        idx
        for idx, value in enumerate(variants)
        if np.isclose(requested, value, atol=1e-6, rtol=0.0)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"requested text window {requested:g}s is absent or ambiguous "
            f"in {variants}"
        )
    return files[matches[0]], variants[matches[0]]
