"""Preflight real-data MEG+text windows before building the shared cache."""
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


def signature(index: AlignmentIndex) -> list[tuple[int, float]]:
    windows = list(index.train_windows) + list(index.val_windows)
    return [(window.section, float(window.t)) for window in windows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/meg_text_window_ablation_cache.yaml"
    )
    parser.add_argument(
        "--training-config", default="configs/meg_text_window_ablation.yaml"
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    with open(args.training_config, "r", encoding="utf-8") as handle:
        training_config = yaml.safe_load(handle)

    signatures = {}
    text_by_condition = {}
    counts = {}
    for window_seconds in (2, 4, 6, 8):
        config = copy.deepcopy(base)
        config["alignment"]["window_seconds"] = window_seconds
        index = AlignmentIndex(config)
        key = f"w{window_seconds}"
        windows = list(index.train_windows) + list(index.val_windows)
        signatures[key] = [
            (window.section, float(window.t)) for window in windows
        ]
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

    reference = signatures["w2"]
    for condition, observed in signatures.items():
        if observed != reference:
            raise RuntimeError(f"{condition} candidate grid differs from w2")
    expected_meg_times = {"w2": 200, "w4": 400, "w6": 600, "w8": 800}
    if {
        condition: values["meg_time"]
        for condition, values in counts.items()
    } != expected_meg_times:
        raise RuntimeError(f"unexpected MEG lengths: {counts}")
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
        f"[pass] identical candidates, distinct text targets: {len(reference)}"
    )


if __name__ == "__main__":
    main()