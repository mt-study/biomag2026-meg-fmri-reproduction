"""Precompute per-run, per-channel MEG mean/std for load-time standardization.

Without this, ``dataset.py`` would have to read every 69 MB cached MEG run in
full just to standardize a 2 s slice of it -- and because DataLoader workers
are respawned each epoch, that ~12 GB read would repeat every epoch. Here it
is done once into a single small ``.npz``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml


def main():
    config_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "configs/default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cache_root = Path(config["project_dir"]) / config["data"]["meg_unified_cache"]
    paths = sorted(cache_root.rglob("*_desc-unifiedpreproc_meg.npy"))
    if not paths:
        raise FileNotFoundError(f"No cached MEG runs under {cache_root}")

    means, stds, keys = [], [], []
    for i, path in enumerate(paths, 1):
        data = np.load(path, mmap_mode="r")
        arr = np.asarray(data, dtype=np.float64)
        means.append(arr.mean(axis=1).astype(np.float32))
        stds.append(arr.std(axis=1).astype(np.float32))
        keys.append(str(path))
        if i % 20 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] {path.name}")

    out = cache_root / "channel_stats.npz"
    np.savez(out, keys=np.array(keys), mean=np.stack(means), std=np.stack(stds))
    print(f"[done] {out}  runs={len(keys)}  channels={means[0].shape[0]}")


if __name__ == "__main__":
    main()
