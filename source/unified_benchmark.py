"""Unified benchmark for MEG, fMRI, and multimodal fusion.

Every condition uses sections 1-7 for training. Checkpoint selection is an
explicit protocol choice: the original fixed-epoch behavior remains available,
while task-1529545-matched runs select on sections 8-9. Averaged and exhaustive
subject representations share the same base-window batches.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from cross_subject_generalization import subject_fold_indices
from exhaustive_subject_pair import (
    evaluate_all_subject_combinations,
    exhaustive_brain_clip_loss,
    format_metric,
)
from models.brain_clip import BrainCLIP
from window_selection import meg_window_time_length, text_window_cache_file


FUSION_MODES = ("mlp", "attention")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class UnifiedSubjectWindowDataset(Dataset):
    """Expose selected subjects either individually or averaged per window."""

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
        data_mode: str,
        noise_std: float = 0.0,
        seed: int = 42,
        fmri_feature_indices: Sequence[int] | None = None,
        meg_time_length: int | None = None,
    ):
        if data_mode not in {"mean", "exhaustive"}:
            raise ValueError(f"unknown data_mode {data_mode!r}")
        self.meg = meg
        self.fmri = fmri
        self.text = text
        self.sections = sections
        self.times = times
        self.window_indices = np.asarray(window_indices, dtype=np.int64)
        self.meg_subject_indices = np.asarray(meg_subject_indices, dtype=np.int64)
        self.fmri_subject_indices = np.asarray(fmri_subject_indices, dtype=np.int64)
        self.data_mode = data_mode
        self.noise_std = float(noise_std)
        self.fmri_feature_indices = (
            None
            if fmri_feature_indices is None
            else np.asarray(fmri_feature_indices, dtype=np.int64)
        )
        self.meg_time_length = (
            None if meg_time_length is None else int(meg_time_length)
        )
        if self.meg_time_length is not None and not (
            0 < self.meg_time_length <= meg.shape[3]
        ):
            raise ValueError(
                f"invalid MEG time length {self.meg_time_length} "
                f"for cached length {meg.shape[3]}"
            )
        self._rng = np.random.default_rng(seed)
        if not len(self.window_indices):
            raise ValueError("unified dataset has no windows")
        if not len(self.meg_subject_indices) or not len(self.fmri_subject_indices):
            raise ValueError("unified dataset has an empty subject axis")

    def __len__(self) -> int:
        return len(self.window_indices)

    def reset_noise(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> dict:
        window_idx = int(self.window_indices[idx])
        meg = np.asarray(
            self.meg[window_idx, self.meg_subject_indices], dtype=np.float32
        )
        if self.meg_time_length is not None:
            meg = meg[..., : self.meg_time_length]
        fmri = np.asarray(
            self.fmri[window_idx, self.fmri_subject_indices], dtype=np.float32
        )
        if self.fmri_feature_indices is not None:
            fmri = fmri[:, self.fmri_feature_indices]
        if self.data_mode == "mean":
            meg = meg.mean(axis=0, keepdims=True, dtype=np.float32)
            fmri = fmri.mean(axis=0, keepdims=True, dtype=np.float32)
        else:
            meg = np.array(meg, dtype=np.float32, copy=True)
            fmri = np.array(fmri, dtype=np.float32, copy=True)
        if self.noise_std > 0:
            meg = np.array(meg, dtype=np.float32, copy=True)
            fmri = np.array(fmri, dtype=np.float32, copy=True)
            meg += self._rng.normal(0, self.noise_std, meg.shape).astype(np.float32)
            fmri += self._rng.normal(0, self.noise_std, fmri.shape).astype(np.float32)
        return {
            "meg": torch.from_numpy(meg),
            "fmri": torch.from_numpy(fmri),
            "text": torch.from_numpy(
                np.array(self.text[window_idx], dtype=np.float32, copy=True)
            ),
            "section": int(self.sections[window_idx]),
            "t": float(self.times[window_idx]),
            "window_index": window_idx,
        }


def _subject_names(meta, key: str, count: int) -> list[str]:
    if key in meta.files:
        names = [str(value) for value in meta[key].tolist()]
    else:
        names = [f"{key}_{idx:03d}" for idx in range(count)]
    if len(names) != count:
        raise ValueError(f"{key} has {len(names)} names for {count} subjects")
    return names


def fmri_delay_feature_indices(
    meta,
    fmri_dim: int,
    requested_delays: Sequence[float] | None,
) -> tuple[np.ndarray | None, list[float] | None]:
    """Map requested delay values to complete cached fMRI feature blocks."""
    cached_delays = (
        [float(value) for value in meta["fmri_delay_seconds"].tolist()]
        if "fmri_delay_seconds" in meta.files
        else None
    )
    if requested_delays is None:
        return None, cached_delays
    requested = [float(value) for value in requested_delays]
    if not requested:
        raise ValueError("fmri_delay_subset_seconds cannot be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("fmri_delay_subset_seconds contains duplicate delays")
    if cached_delays is None:
        raise ValueError(
            "delay subset selection requires fmri_delay_seconds in windows_meta.npz"
        )
    if fmri_dim % len(cached_delays):
        raise ValueError(
            f"fMRI dimension {fmri_dim} is not divisible by "
            f"{len(cached_delays)} cached delays"
        )
    block_size = fmri_dim // len(cached_delays)
    blocks = []
    for requested_delay in requested:
        matches = [
            idx
            for idx, cached_delay in enumerate(cached_delays)
            if np.isclose(requested_delay, cached_delay, atol=1e-6, rtol=0.0)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"requested delay {requested_delay:g}s is absent or ambiguous in "
                f"cached delays {cached_delays}"
            )
        start = matches[0] * block_size
        blocks.append(np.arange(start, start + block_size, dtype=np.int64))
    return np.concatenate(blocks), cached_delays


def build_unified_datasets(
    config: dict,
    protocol: str,
    data_mode: str,
    fold: int | None,
    seed: int = 42,
):
    cache_dir = Path(config["project_dir"]) / config["data"]["window_cache"]
    required = [
        "meg_windows.npy",
        "fmri_windows.npy",
        "windows_meta.npz",
    ]
    missing = [name for name in required if not (cache_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"cache at {cache_dir} is missing {missing}")

    meg = np.load(cache_dir / "meg_windows.npy", mmap_mode="r")
    fmri = np.load(cache_dir / "fmri_windows.npy", mmap_mode="r")
    meta = np.load(cache_dir / "windows_meta.npz", allow_pickle=False)
    sections, times = meta["section"], meta["t"]

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
            f"cache at {cache_dir} is missing selected text file {text_filename}"
        )
    text = np.load(text_path, mmap_mode="r")
    if not (len(meg) == len(fmri) == len(text) == len(sections) == len(times)):
        raise ValueError("window cache arrays have inconsistent lengths")

    requested_delays = config.get("data", {}).get("fmri_delay_subset_seconds")
    fmri_feature_indices, cached_delays = fmri_delay_feature_indices(
        meta, int(fmri.shape[2]), requested_delays
    )
    temporal_sampling = {
        key: (
            float(np.asarray(meta[key]).item())
            if key in meta.files
            else None
        )
        for key in (
            "window_seconds",
            "meg_sfreq",
            "stride_seconds",
            "grid_window_bound_seconds",
            "grid_delay_bound_seconds",
            "grid_text_presence_seconds",
        )
    }
    temporal_sampling["cached_meg_window_seconds"] = cached_window
    temporal_sampling["selected_meg_window_seconds"] = (
        cached_window if requested_window is None else float(requested_window)
    )
    temporal_sampling["selected_text_window_seconds"] = selected_text_window
    temporal_sampling["selected_text_window_file"] = text_filename
    temporal_sampling["cached_fmri_delay_seconds"] = cached_delays
    temporal_sampling["selected_fmri_delay_seconds"] = (
        cached_delays
        if requested_delays is None
        else [float(value) for value in requested_delays]
    )
    protocol_cfg = config["protocol"]
    train_sections = sorted(set(protocol_cfg["train_sections"]))
    test_sections = sorted(set(protocol_cfg["test_sections"]))
    if set(train_sections) & set(test_sections):
        raise ValueError("train and test sections overlap")
    if train_sections != list(range(1, 8)) or test_sections != [8, 9]:
        raise ValueError(
            "unified benchmark is locked to train sections 1-7 and test sections 8-9"
        )

    all_meg = np.arange(meg.shape[1], dtype=np.int64)
    all_fmri = np.arange(fmri.shape[1], dtype=np.int64)
    if protocol == "all_subjects":
        if fold is not None:
            raise ValueError("all_subjects protocol does not take a fold")
        meg_train = meg_test = all_meg
        fmri_train = fmri_test = all_fmri
        n_folds = None
    elif protocol == "unseen_subject":
        if fold is None:
            raise ValueError("unseen_subject protocol requires --fold")
        n_folds = int(protocol_cfg.get("n_subject_folds", 5))
        meg_train, meg_test = subject_fold_indices(len(all_meg), fold, n_folds)
        fmri_train, fmri_test = subject_fold_indices(len(all_fmri), fold, n_folds)
    else:
        raise ValueError(f"unknown protocol {protocol!r}")

    train_windows = np.flatnonzero(np.isin(sections, train_sections))
    test_windows = np.flatnonzero(np.isin(sections, test_sections))
    noise_std = float(config.get("sampling", {}).get("noise_std", 0.0))
    datasets = {
        "train": UnifiedSubjectWindowDataset(
            meg,
            fmri,
            text,
            sections,
            times,
            train_windows,
            meg_train,
            fmri_train,
            data_mode,
            noise_std=noise_std,
            seed=seed,
            fmri_feature_indices=fmri_feature_indices,
            meg_time_length=meg_time_length,
        ),
        # For unseen-subject runs, epoch selection may inspect the held-out
        # sections but only through training subjects. Test subjects remain
        # untouched until the final evaluation.
        "selection": UnifiedSubjectWindowDataset(
            meg,
            fmri,
            text,
            sections,
            times,
            test_windows,
            meg_train,
            fmri_train,
            data_mode,
            fmri_feature_indices=fmri_feature_indices,
            meg_time_length=meg_time_length,
        ),
        "test": UnifiedSubjectWindowDataset(
            meg,
            fmri,
            text,
            sections,
            times,
            test_windows,
            meg_test,
            fmri_test,
            data_mode,
            fmri_feature_indices=fmri_feature_indices,
            meg_time_length=meg_time_length,
        ),
    }

    meg_names = _subject_names(meta, "meg_subjects", meg.shape[1])
    fmri_names = _subject_names(meta, "fmri_subjects", fmri.shape[1])
    info = {
        "protocol": protocol,
        "data_mode": data_mode,
        "window_cache": str(config["data"]["window_cache"]),
        "temporal_sampling": temporal_sampling,
        "fold": fold,
        "n_subject_folds": n_folds,
        "train_sections": train_sections,
        "test_sections": test_sections,
        "train_windows": int(len(train_windows)),
        "selection_windows": int(len(test_windows)),
        "test_windows": int(len(test_windows)),
        "cached_meg_window_seconds": cached_window,
        "selected_meg_window_seconds": (
            cached_window if requested_window is None else float(requested_window)
        ),
        "selected_text_window_seconds": selected_text_window,
        "selected_text_window_file": text_filename,
        "cached_fmri_delay_seconds": cached_delays,
        "selected_fmri_delay_seconds": (
            cached_delays
            if requested_delays is None
            else [float(value) for value in requested_delays]
        ),
        "selection_meg_indices": meg_train.tolist(),
        "selection_fmri_indices": fmri_train.tolist(),
        "selection_subject_policy": "training_subjects",
        "meg_train_indices": meg_train.tolist(),
        "meg_test_indices": meg_test.tolist(),
        "fmri_train_indices": fmri_train.tolist(),
        "fmri_test_indices": fmri_test.tolist(),
        "meg_train_subjects": [meg_names[idx] for idx in meg_train],
        "meg_test_subjects": [meg_names[idx] for idx in meg_test],
        "fmri_train_subjects": [fmri_names[idx] for idx in fmri_train],
        "fmri_test_subjects": [fmri_names[idx] for idx in fmri_test],
        "train_subject_pairs": int(len(meg_train) * len(fmri_train)),
        "test_subject_pairs": int(len(meg_test) * len(fmri_test)),
        "effective_train_meg_groups": 1 if data_mode == "mean" else int(len(meg_train)),
        "effective_train_fmri_groups": 1 if data_mode == "mean" else int(len(fmri_train)),
    }
    if protocol == "unseen_subject":
        if set(info["meg_train_subjects"]) & set(info["meg_test_subjects"]):
            raise AssertionError("MEG subject leakage")
        if set(info["fmri_train_subjects"]) & set(info["fmri_test_subjects"]):
            raise AssertionError("fMRI subject leakage")

    dims = {
        "meg_channels": int(meg.shape[2]),
        "meg_time": int(
            meg.shape[3] if meg_time_length is None else meg_time_length
        ),
        "fmri_dim": int(
            fmri.shape[2]
            if fmri_feature_indices is None
            else len(fmri_feature_indices)
        ),
        "text_dim": int(text.shape[1]),
    }
    return datasets, info, dims


def build_model(
    config: dict,
    dims: dict,
    device: torch.device,
    fusion_mode: str | None,
) -> BrainCLIP:
    model_cfg = config["model"]
    hidden_dims = model_cfg.get("fusion_hidden_dims", {})
    hidden_dim = int(hidden_dims.get(fusion_mode, model_cfg.get("joint_hidden_dim", 64)))
    return BrainCLIP(
        meg_channels=dims["meg_channels"],
        meg_time=dims["meg_time"],
        fmri_dim=dims["fmri_dim"],
        text_dim=dims["text_dim"],
        embedding_dim=int(model_cfg["embedding_dim"]),
        meg_hidden_channels=int(model_cfg["meg_hidden_channels"]),
        fmri_hidden_dim=int(model_cfg["fmri_hidden_dim"]),
        text_hidden_dim=int(model_cfg["text_hidden_dim"]),
        joint_hidden_dim=hidden_dim,
        use_joint=fusion_mode is not None,
        fusion_mode=fusion_mode or "mlp",
        dropout=float(model_cfg.get("dropout", 0.3)),
        meg_temporal_bins=(
            None
            if model_cfg.get("meg_temporal_bins") is None
            else int(model_cfg["meg_temporal_bins"])
        ),
    ).to(device)


def make_train_loader(dataset, config: dict, sampler_seed: int) -> DataLoader:
    train_cfg = config["training"]
    workers = int(train_cfg.get("num_workers", 0))
    noise_dataset = dataset
    while isinstance(noise_dataset, Subset):
        noise_dataset = noise_dataset.dataset
    if hasattr(noise_dataset, "reset_noise"):
        noise_dataset.reset_noise(sampler_seed)
    generator = torch.Generator()
    generator.manual_seed(int(sampler_seed))
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["base_window_batch_size"]),
        shuffle=True,
        generator=generator,
        drop_last=bool(train_cfg.get("drop_last", True)),
        num_workers=workers,
        persistent_workers=workers > 0,
    )


def _freeze_encoders(model: BrainCLIP) -> list[torch.nn.Module]:
    modules = [model.meg_encoder, model.fmri_encoder, model.text_encoder]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = False
    return modules


@torch.no_grad()
def evaluate_selection_mean(
    model: BrainCLIP,
    dataset,
    config: dict,
    device: torch.device,
    stage: str,
) -> dict:
    """Select epochs from one cross-subject raw mean per stimulus window."""
    model.eval()
    train_cfg = config["training"]
    workers = int(train_cfg.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("eval_base_window_batch_size", 64)),
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    meg_embeddings, fmri_embeddings, text_embeddings, joint_embeddings = [], [], [], []
    for batch_data in loader:
        meg = batch_data["meg"].to(device).mean(dim=1)
        fmri = batch_data["fmri"].to(device).mean(dim=1)
        text = batch_data["text"].to(device)
        meg_emb, fmri_emb, text_emb, _, joint_emb = model(meg, fmri, text)
        meg_embeddings.append(meg_emb.cpu())
        fmri_embeddings.append(fmri_emb.cpu())
        text_embeddings.append(text_emb.cpu())
        if joint_emb is not None:
            joint_embeddings.append(joint_emb.cpu())

    meg_emb = torch.cat(meg_embeddings)
    fmri_emb = torch.cat(fmri_embeddings)
    text_emb = torch.cat(text_embeddings)

    def metric(query: torch.Tensor, candidate: torch.Tensor) -> dict:
        similarity = query @ candidate.t()
        diagonal = torch.arange(len(query))
        correct = similarity[diagonal, diagonal].unsqueeze(1)
        ranks = (similarity > correct).sum(dim=1)
        return {
            "top10": float((ranks < 10).float().mean()),
            # torch.median uses the lower middle rank for an even candidate
            # count, exactly matching task 1529545 checkpoint selection.
            "median_rank": float(ranks.float().median()),
        }

    metrics = {
        "representation": "cross_subject_raw_mean",
        "meg_text": metric(meg_emb, text_emb),
        "fmri_text": metric(fmri_emb, text_emb),
    }
    if stage == "encoder":
        metrics["score"] = (
            metrics["meg_text"]["median_rank"]
            + metrics["fmri_text"]["median_rank"]
        ) / 2
    else:
        if not joint_embeddings:
            raise ValueError(f"{stage} selection requires a fusion encoder")
        metrics["joint_text"] = metric(torch.cat(joint_embeddings), text_emb)
        metrics["score"] = metrics["joint_text"]["median_rank"]
    return metrics


def train_fixed_epochs(
    stage: str,
    model: BrainCLIP,
    train_dataset,
    config: dict,
    device: torch.device,
    epochs: int,
    sampler_seed: int,
    checkpoint_path: Path,
    frozen_modules: list[torch.nn.Module],
    selection_dataset=None,
) -> dict:
    """Train using either the fixed-final or validation-best protocol."""
    loader = make_train_loader(train_dataset, config, sampler_seed)
    train_cfg = config["training"]
    selection_mode = str(train_cfg.get("checkpoint_selection", "fixed_final_epoch"))
    if selection_mode not in {"fixed_final_epoch", "validation_best"}:
        raise ValueError(f"unknown checkpoint_selection {selection_mode!r}")
    if selection_mode == "validation_best" and selection_dataset is None:
        raise ValueError("validation_best requires a selection dataset")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    history = []
    total_steps = 0
    best_score = float("inf")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for module in frozen_modules:
            module.eval()
        started = time.time()
        totals: dict[str, float] = {}
        batches = 0
        for batch_data in loader:
            meg = batch_data["meg"].to(device)
            fmri = batch_data["fmri"].to(device)
            text = batch_data["text"].to(device)
            losses = exhaustive_brain_clip_loss(
                model,
                meg,
                fmri,
                text,
                lambda_mf=float(train_cfg["lambda_mf"]),
                lambda_jt=float(train_cfg["lambda_jt"]),
            )
            if stage != "encoder":
                losses["total"] = float(train_cfg["lambda_jt"]) * losses["joint_text"]
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(f"non-finite {stage} loss at epoch {epoch}")
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            batches += 1
            total_steps += 1
        if not batches:
            raise RuntimeError("training produced zero batches")
        totals = {key: value / batches for key, value in totals.items()}
        record = {"epoch": epoch, "batches": batches, "losses": totals}
        print(
            f"[{stage} epoch {epoch}/{epochs}] "
            + " ".join(f"{key}={value:.4f}" for key, value in totals.items())
            + f" batches={batches} [{time.time() - started:.1f}s]"
        )

        if selection_mode == "validation_best":
            selection_metrics = evaluate_selection_mean(
                model, selection_dataset, config, device, stage
            )
            record["selection_metrics"] = selection_metrics
            if stage == "encoder":
                print(
                    f"    selection mean: meg top10={selection_metrics['meg_text']['top10']:.4f} "
                    f"medR={selection_metrics['meg_text']['median_rank']:.0f} | "
                    f"fmri top10={selection_metrics['fmri_text']['top10']:.4f} "
                    f"medR={selection_metrics['fmri_text']['median_rank']:.0f} | "
                    f"score={selection_metrics['score']:.1f}"
                )
            else:
                joint = selection_metrics["joint_text"]
                print(
                    f"    selection mean: {stage} top10={joint['top10']:.4f} "
                    f"medR={joint['median_rank']:.0f}"
                )
            if selection_metrics["score"] < best_score:
                best_score = float(selection_metrics["score"])
                torch.save(
                    {
                        "stage": stage,
                        "epoch": epoch,
                        "selection": selection_mode,
                        "selection_metrics": selection_metrics,
                        "model_state_dict": model.state_dict(),
                        "optimizer_steps": total_steps,
                        "sampler_seed": int(sampler_seed),
                    },
                    checkpoint_path,
                )
                print(
                    f"[checkpoint] {stage} new best epoch {epoch} "
                    f"score={best_score:.1f} -> {checkpoint_path}"
                )
        history.append(record)

    if selection_mode == "fixed_final_epoch":
        checkpoint = {
            "stage": stage,
            "epoch": epochs,
            "selection": selection_mode,
            "model_state_dict": model.state_dict(),
            "optimizer_steps": total_steps,
            "sampler_seed": int(sampler_seed),
            "history": history,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"[checkpoint] fixed final epoch -> {checkpoint_path}")
        return checkpoint

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["history"] = history
    checkpoint["total_optimizer_steps"] = total_steps
    torch.save(checkpoint, checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"[checkpoint] restored {stage} best epoch {checkpoint['epoch']} "
        f"score={checkpoint['selection_metrics']['score']:.1f}"
    )
    return checkpoint


def load_frozen_fusion(
    config: dict,
    dims: dict,
    device: torch.device,
    fusion_mode: str,
    encoder_state: dict,
) -> tuple[BrainCLIP, list[torch.nn.Module]]:
    model = build_model(config, dims, device, fusion_mode)
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    if unexpected or not missing or any(
        not key.startswith("joint_encoder.") for key in missing
    ):
        raise ValueError(
            f"invalid encoder load for {fusion_mode}: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    return model, _freeze_encoders(model)


def random_retrieval_baseline(
    dataset: UnifiedSubjectWindowDataset,
    repeats: int,
    seed: int,
) -> dict:
    """Monte Carlo null for a uniformly random retrieval ranking."""
    sections = np.asarray(dataset.sections[dataset.window_indices], dtype=np.int64)
    n = len(sections)
    rng = np.random.default_rng(seed)
    global_ranks = rng.integers(0, n, size=(repeats, n), dtype=np.int32)
    pool_sizes = np.asarray([(sections == section).sum() for section in sections])
    section_ranks = np.floor(rng.random((repeats, n)) * pool_sizes).astype(np.int32)

    def summarize(ranks: np.ndarray) -> dict:
        values = {
            "top1": (ranks < 1).mean(axis=1),
            "top5": (ranks < 5).mean(axis=1),
            "top10": (ranks < 10).mean(axis=1),
            "median_rank": np.median(ranks, axis=1),
            "mean_rank": ranks.mean(axis=1),
            "mrr": (1.0 / (ranks + 1)).mean(axis=1),
        }
        return {
            key: {
                "mean": float(value.mean()),
                "std": float(value.std(ddof=1)),
                "ci95": [
                    float(np.quantile(value, 0.025)),
                    float(np.quantile(value, 0.975)),
                ],
            }
            for key, value in values.items()
        }

    return {
        "definition": "uniformly random candidate rank per query",
        "repeats": int(repeats),
        "seed": int(seed),
        "global": summarize(global_ranks),
        "in_section": summarize(section_ranks),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _print_test_metrics(method: str, metric: dict) -> None:
    print("    " + format_metric(method, metric))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unified_benchmark.yaml")
    parser.add_argument("--protocol", choices=["all_subjects", "unseen_subject"], required=True)
    parser.add_argument("--data-mode", choices=["mean", "exhaustive"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--encoder-epochs", type=int)
    parser.add_argument("--fusion-epochs", type=int)
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--result-root")
    parser.add_argument("--window-cache")
    parser.add_argument("--meg-text-window-seconds", type=float)
    parser.add_argument(
        "--fmri-delay-subset-seconds", nargs="+", type=float
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.window_cache:
        config["data"]["window_cache"] = args.window_cache
    if args.meg_text_window_seconds is not None:
        config["data"]["window_subset_seconds"] = args.meg_text_window_seconds
    if args.fmri_delay_subset_seconds is not None:
        config["data"]["fmri_delay_subset_seconds"] = (
            args.fmri_delay_subset_seconds
        )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, split_info, dims = build_unified_datasets(
        config, args.protocol, args.data_mode, args.fold, seed=args.seed
    )
    full_train_windows = len(datasets["train"])
    train_dataset = datasets["train"]
    if args.max_train_windows and args.max_train_windows < len(train_dataset):
        train_dataset = Subset(train_dataset, range(args.max_train_windows))

    train_cfg = config["training"]
    fusion_modes = tuple(train_cfg.get("fusion_modes", FUSION_MODES))
    if (
        not fusion_modes
        or len(set(fusion_modes)) != len(fusion_modes)
        or any(mode not in FUSION_MODES for mode in fusion_modes)
    ):
        raise ValueError(
            f"training.fusion_modes must be a non-empty unique subset of {FUSION_MODES}"
        )
    selection_mode = str(train_cfg.get("checkpoint_selection", "fixed_final_epoch"))
    if selection_mode not in {"fixed_final_epoch", "validation_best"}:
        raise ValueError(f"unknown checkpoint_selection {selection_mode!r}")
    checkpoint_label = "best" if selection_mode == "validation_best" else "final"
    protocol_version = (
        "unified_task1529545_matched_v1"
        if selection_mode == "validation_best"
        else "unified_fixed_epoch_v1"
    )
    encoder_epochs = int(args.encoder_epochs or train_cfg["encoder_epochs"])
    fusion_epochs = int(args.fusion_epochs or train_cfg["fusion_epochs"])
    run_name = f"{args.protocol}/{args.data_mode}/seed_{args.seed}"
    if args.fold is not None:
        run_name += f"_fold_{args.fold}"
    checkpoint_root = Path(
        args.checkpoint_root or config["output"]["checkpoint_root"]
    ) / run_name
    result_root = Path(args.result_root or config["output"]["result_root"]) / run_name

    print(f"[run] {run_name}")
    print(f"[device] {device}")
    print(
        f"[split] train sections={split_info['train_sections']} windows={full_train_windows}; "
        f"selection/test sections={split_info['test_sections']} "
        f"windows={len(datasets['test'])}"
    )
    print(
        f"[subjects] train={len(split_info['meg_train_subjects'])} MEG x "
        f"{len(split_info['fmri_train_subjects'])} fMRI; "
        f"test={len(split_info['meg_test_subjects'])} MEG x "
        f"{len(split_info['fmri_test_subjects'])} fMRI"
    )
    if selection_mode == "fixed_final_epoch":
        print("[selection] fixed final epoch; test sections never select a checkpoint")
        selection_dataset = None
    else:
        selection_dataset = datasets["selection"]
        print(
            "[selection] best epoch on sections 8-9 using cross-subject raw means "
            "from training subjects only"
        )
        if args.protocol == "all_subjects":
            print("[guard] all-subject metrics use the same validation/report set as task 1529545")
        else:
            print("[guard] held-out test subjects never participate in checkpoint selection")

    encoder_model = build_model(config, dims, device, fusion_mode=None)
    encoder_trainable_parameters = int(
        sum(parameter.numel() for parameter in encoder_model.parameters())
    )
    print(
        f"[encoder] trainable={encoder_trainable_parameters} "
        f"meg_temporal_bins={config['model'].get('meg_temporal_bins')}"
    )
    encoder_checkpoint = train_fixed_epochs(
        "encoder",
        encoder_model,
        train_dataset,
        config,
        device,
        encoder_epochs,
        args.seed,
        checkpoint_root / f"encoder_{checkpoint_label}.pt",
        frozen_modules=[],
        selection_dataset=selection_dataset,
    )
    encoder_state = encoder_checkpoint["model_state_dict"]

    fusion_models = {}
    fusion_checkpoints = {}
    fusion_params = {}
    fusion_seed = args.seed + 1000
    for fusion_mode in fusion_modes:
        set_seed(fusion_seed)
        model, frozen_modules = load_frozen_fusion(
            config, dims, device, fusion_mode, encoder_state
        )
        fusion_params[fusion_mode] = int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )
        print(
            f"[fusion {fusion_mode}] trainable={fusion_params[fusion_mode]} "
            f"sampler_seed={fusion_seed}"
        )
        fusion_checkpoints[fusion_mode] = train_fixed_epochs(
            fusion_mode,
            model,
            train_dataset,
            config,
            device,
            fusion_epochs,
            fusion_seed,
            checkpoint_root / f"{fusion_mode}_{checkpoint_label}.pt",
            frozen_modules,
            selection_dataset=selection_dataset,
        )
        fusion_models[fusion_mode] = model

    eval_kwargs = {
        "batch_size": int(train_cfg["eval_base_window_batch_size"]),
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "pair_chunk_size": int(train_cfg["eval_pair_chunk_size"]),
    }
    if args.protocol == "unseen_subject":
        print("[final test on held-out subjects]")
    elif selection_mode == "validation_best":
        print("[validation/report evaluation, matching task 1529545]")
    else:
        print("[untouched final test]")
    encoder_metrics = evaluate_all_subject_combinations(
        encoder_model, datasets["test"], device, **eval_kwargs
    )
    learned_metrics = {
        mode: evaluate_all_subject_combinations(
            model, datasets["test"], device, **eval_kwargs
        )
        for mode, model in fusion_models.items()
    }
    methods = {
        "meg": encoder_metrics["meg_text"],
        "fmri": encoder_metrics["fmri_text"],
        "late": encoder_metrics["late_text"],
    }
    methods.update(
        {
            mode: learned_metrics[mode]["joint_text"]
            for mode in fusion_modes
        }
    )
    for name, metric in methods.items():
        _print_test_metrics(name, metric)

    random_baseline = random_retrieval_baseline(
        datasets["test"],
        int(config["evaluation"].get("random_repeats", 2000)),
        args.seed + 2000,
    )
    parameter_ratio = None
    if set(FUSION_MODES).issubset(fusion_params):
        parameter_gap = abs(fusion_params["mlp"] - fusion_params["attention"])
        parameter_ratio = parameter_gap / max(fusion_params.values())
    result = {
        "protocol_version": protocol_version,
        "run_name": run_name,
        "git_commit": _git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "seed": int(args.seed),
        "fusion_seed": int(fusion_seed),
        "split": split_info,
        "hyperparameters": {
            "base_window_batch_size": int(train_cfg["base_window_batch_size"]),
            "drop_last": bool(train_cfg.get("drop_last", True)),
            "encoder_epochs": encoder_epochs,
            "fusion_epochs": fusion_epochs,
            "learning_rate": float(train_cfg["learning_rate"]),
            "weight_decay": float(train_cfg["weight_decay"]),
            "lambda_mf": float(train_cfg["lambda_mf"]),
            "lambda_jt": float(train_cfg["lambda_jt"]),
            "noise_std": float(config.get("sampling", {}).get("noise_std", 0.0)),
            "meg_text_window_seconds": split_info[
                "selected_meg_window_seconds"
            ],
            "meg_temporal_bins": config["model"].get("meg_temporal_bins"),
            "encoder_trainable_parameters": encoder_trainable_parameters,
            "encoder_sampler_seed": int(args.seed),
            "fusion_sampler_seed": int(fusion_seed),
            "fusion_modes": list(fusion_modes),
            "fusion_hidden_dims": {
                mode: int(
                    config["model"].get("fusion_hidden_dims", {}).get(
                        mode, config["model"].get("joint_hidden_dim", 64)
                    )
                )
                for mode in fusion_modes
            },
            "encoders_frozen_during_fusion": True,
            "checkpoint_selection": selection_mode,
            "selection_representation": (
                "cross_subject_raw_mean"
                if selection_mode == "validation_best"
                else None
            ),
        },
        "checkpoints": {
            "encoder": str(checkpoint_root / f"encoder_{checkpoint_label}.pt"),
            **{
                mode: str(checkpoint_root / f"{mode}_{checkpoint_label}.pt")
                for mode in fusion_modes
            },
        },
        "selected_epochs": {
            "encoder": int(encoder_checkpoint["epoch"]),
            **{
                mode: int(fusion_checkpoints[mode]["epoch"])
                for mode in fusion_modes
            },
        },
        "optimizer_steps": {
            "encoder": int(encoder_checkpoint["optimizer_steps"]),
            **{
                mode: int(fusion_checkpoints[mode]["optimizer_steps"])
                for mode in fusion_modes
            },
        },
        "trainable_fusion_parameters": fusion_params,
        "fusion_parameter_gap_ratio": parameter_ratio,
        "random_baseline": random_baseline,
        "test_metrics": methods,
        "test_evaluation": (
            "performed on held-out subjects only after checkpoint selection"
            if args.protocol == "unseen_subject"
            else (
                "performed on the validation/report set used for checkpoint selection"
                if selection_mode == "validation_best"
                else "performed only after every fixed-epoch model finished training"
            )
        ),
    }
    if "attention" in learned_metrics:
        result["attention_weights"] = learned_metrics["attention"]["attention_weights"]
    result_root.mkdir(parents=True, exist_ok=True)
    result_path = result_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[result] {result_path}")


if __name__ == "__main__":
    main()
