"""Small numeric helpers shared by the offline cache-building scripts
(``scripts/build_fmri_cache.py``, ``scripts/build_text_cache.py``).

Not used at train time: by the time ``train.py`` runs, everything these
functions produce has already been written to ``cache/`` as plain
``.npy`` arrays.
"""
from __future__ import annotations

import re
from typing import Tuple

import numpy as np

_FLOAT_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")


def parse_bert_vector(cell: str) -> np.ndarray:
    """Parse a stringified numpy vector such as ``"[ 0.97 0.16 ... ]"``.

    The source CSV stores ``str(np.ndarray)`` output: whitespace-separated,
    wraps across lines, no commas -- a plain CSV/JSON parser cannot read it.
    """
    values = _FLOAT_RE.findall(cell)
    return np.asarray(values, dtype=np.float32)


def pool_volume(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Deterministic average-pool a 3D volume down to ``target_shape``.

    Uses ``np.linspace``-derived bin edges per axis so it works for any
    input size, not just sizes evenly divisible by the target.
    """
    out = volume
    for axis, target in enumerate(target_shape):
        n = out.shape[axis]
        edges = np.round(np.linspace(0, n, target + 1)).astype(int)
        pooled = []
        for i in range(target):
            lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
            hi = min(hi, n)
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(lo, hi)
            pooled.append(out[tuple(sl)].mean(axis=axis, keepdims=True))
        out = np.concatenate(pooled, axis=axis)
    return out
