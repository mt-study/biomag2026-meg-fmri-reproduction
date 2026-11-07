"""Preflight the real-data non-overlapping MEG+text window experiment."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alignment import AlignmentIndex  # noqa: E402
from unified_benchmark import build_model  # noqa: E402


WINDOWS = (2, 4, 6, 8)


def signature(index: AlignmentIndex) -> list[tuple[int, float]]:
    windows = list(index.train_windows) + list(index.val_windows)
    return [(window.section, float(window.t)) for window in windows]


def assert_non_overlapping(items: list[tuple[int, float]], length: float) -> None:
    by_section: dict[int, list[float]] = {}
    for section, start in items:
        by_section.setdefault(section, []).append(start)
    for section, starts in by_section.items():
        starts.sort()
        gaps = np.diff(starts)
        if len(gaps) and float(gaps.min()) + 1e-6 < length:
            raise RuntimeError(
                f"w{length:g}: section {section} overlaps; min start gap="
                f"{float(gaps.min()):g}s"
            )


def load_cache_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "alignment" in config:
        return config

    cache_path = path.with_name(f"{path.stem}_cache{path.suffix}")
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as handle:
            cache_config = yaml.safe_load(handle)
        if "alignment" in cache_config:
            print(f"[config] using cache config {cache_path}")
            return cache_config

    raise KeyError(
        f"{path} has no alignment section and no matching *_cache config"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/meg_text_window_nonoverlap_cache.yaml"
    )
    parser.add_argument(
        "--training-config", default="configs/meg_text_window_nonoverlap.yaml"
    )
    args = parser.parse_args()
    base = load_cache_config(Path(args.config))
    with open(args.training_config, "r", encoding="utf-8") as handle:
        training_config = yaml.safe_load(handle)

    alignment = base["alignment"]
    signatures = {}
    text_by_condition = {}
    counts = {}
    for window_seconds in WINDOWS:
        config = copy.deepcopy(base)
        config["alignment"]["window_seconds"] = window_seconds
        config["alignment"]["stride_seconds"] = window_seconds
        config["alignment"]["grid_window_bound_seconds"] = window_seconds
        config["alignment"]["grid_delay_bound_seconds"] = 8
        config["alignment"]["grid_text_presence_seconds"] = window_seconds
        config["alignment"]["text_window_variants_seconds"] = [window_seconds]
        config["data"]["window_cache"] = (
            f"cache/meg_text_window_nonoverlap/w{window_seconds}_stride{window_seconds}"
        )
        index = AlignmentIndex(config)
        key = f"w{window_seconds}"
        windows = list(index.train_windows) + list(index.val_windows)
        signatures[key] = signature(index)
        assert_non_overlapping(signatures[key], float(window_seconds))
        text_by_condition[key] = np.stack(
            [window.text_embedding for window in windows]
        )
        counts[key] = {
            "train": len(index.train_windows),
            "test": len(index.val_windows),
            "total": len(windows),
            "meg_time": index.meg_window_length,
            "text_dim": index.text_dim,
        }
        print(f"[{key}] {counts[key]}")

    total_counts = [counts[f"w{window_seconds}"]["total"] for window_seconds in WINDOWS]
    if not all(left > right for left, right in zip(total_counts, total_counts[1:])):
        raise RuntimeError(f"candidate counts are not 2s>4s>6s>8s: {total_counts}")
    expected_meg_times = {"w2": 200, "w4": 400, "w6": 600, "w8": 800}
    observed_meg_times = {
        condition: values["meg_time"] for condition, values in counts.items()
    }
    if observed_meg_times != expected_meg_times:
        raise RuntimeError(f"unexpected MEG lengths: {observed_meg_times}")
    if any(
        np.array_equal(text_by_condition["w2"], text_by_condition[condition])
        for condition in ("w4", "w6", "w8")
    ):
        raise RuntimeError("text window variants unexpectedly have identical features")

    parameter_counts = {}
    for condition, meg_time in expected_meg_times.items():
        dims = {
            "meg_channels": 306,
            "meg_time": meg_time,
            "fmri_dim": 2304,
            "text_dim": 768,
        }
        model = build_model(
            training_config, dims, torch.device("cpu"), fusion_mode=None
        )
        parameter_counts[condition] = sum(
            parameter.numel() for parameter in model.parameters()
        )
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"encoder parameter counts differ: {parameter_counts}")

    print(f"[encoder parameters] {parameter_counts}")
    print(
        "[pass] condition-specific non-overlapping grids, decreasing candidate "
        f"counts={total_counts}, distinct text targets"
    )


if __name__ == "__main__":
    main()
