"""Preflight the real-data candidate grid before materializing ablation caches."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alignment import AlignmentIndex  # noqa: E402


def signature(index: AlignmentIndex) -> list[tuple[int, float]]:
    windows = list(index.train_windows) + list(index.val_windows)
    return [(window.section, float(window.t)) for window in windows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/temporal_ablation_cache.yaml"
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    signatures = {}
    counts = {}
    for window_seconds in (1, 2, 4):
        config = copy.deepcopy(base)
        config["alignment"]["window_seconds"] = window_seconds
        index = AlignmentIndex(config)
        key = f"w{window_seconds}_dall"
        signatures[key] = signature(index)
        counts[key] = {
            "train": len(index.train_windows),
            "test": len(index.val_windows),
            "total": len(signatures[key]),
            "meg_time": index.meg_window_length,
        }
        print(f"[{key}] {counts[key]}")

    reference = signatures["w1_dall"]
    for condition, observed in signatures.items():
        if observed != reference:
            mismatch = next(
                (
                    idx
                    for idx, pair in enumerate(zip(reference, observed))
                    if pair[0] != pair[1]
                ),
                min(len(reference), len(observed)),
            )
            raise RuntimeError(
                f"{condition} candidate grid differs at index {mismatch}: "
                f"reference={reference[mismatch:mismatch + 1]} "
                f"observed={observed[mismatch:mismatch + 1]}"
            )
    print(f"[pass] identical candidate grid across all windows: {len(reference)}")


if __name__ == "__main__":
    main()