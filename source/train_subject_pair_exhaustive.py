"""Train BrainCLIP with every MEG x fMRI subject combination per window."""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from exhaustive_subject_pair import (
    build_all_subject_datasets,
    evaluate_all_subject_combinations,
    exhaustive_brain_clip_loss,
    format_metric,
)
from models.brain_clip import BrainCLIP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_training_objective(configured: str, encoders_frozen: bool) -> str:
    """Resolve the loss policy while preserving the historical default."""
    if configured == "auto":
        return "joint_text_only" if encoders_frozen else "full_alignment"
    if configured not in {"joint_text_only", "full_alignment"}:
        raise ValueError(
            "training.objective must be 'auto', 'joint_text_only', or "
            f"'full_alignment', got {configured!r}"
        )
    return configured


def build_model(config: dict, dims: dict, device: torch.device) -> BrainCLIP:
    model_cfg = config["model"]
    return BrainCLIP(
        meg_channels=dims["meg_channels"],
        meg_time=dims["meg_time"],
        fmri_dim=dims["fmri_dim"],
        text_dim=dims["text_dim"],
        embedding_dim=int(model_cfg["embedding_dim"]),
        meg_hidden_channels=int(model_cfg["meg_hidden_channels"]),
        fmri_hidden_dim=int(model_cfg["fmri_hidden_dim"]),
        text_hidden_dim=int(model_cfg["text_hidden_dim"]),
        joint_hidden_dim=int(model_cfg.get("joint_hidden_dim", 512)),
        use_joint=bool(model_cfg.get("use_joint", False)),
        fusion_mode=str(model_cfg.get("fusion_mode", "mlp")),
        dropout=float(model_cfg.get("dropout", 0.3)),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    train_cfg = config["training"]
    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, info, dims = build_all_subject_datasets(config)
    train_ds.reset_noise(int(train_cfg.get("seed", 42)))
    if args.max_train_windows and args.max_train_windows < len(train_ds):
        train_ds = Subset(train_ds, range(args.max_train_windows))
    base_batch_size = int(train_cfg.get("base_window_batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 0))
    loader = DataLoader(
        train_ds,
        batch_size=base_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    print(f"[config] {args.config}")
    print(f"[device] {device}")
    print(f"[seed] {train_cfg.get('seed', 42)}")
    print(
        f"[data] {len(train_ds)} base training windows; {len(val_ds)} validation windows; "
        f"{info['meg_subjects']} MEG x {info['fmri_subjects']} fMRI = "
        f"{info['subject_pairs']} pairs per window"
    )
    print("[contrastive] pair axis is grouped; only distinct stimulus windows are negatives")

    model = build_model(config, dims, device)
    init_from = config["model"].get("init_from")
    freeze = bool(config["model"].get("freeze_encoders", False))
    frozen_modules = []
    if init_from:
        checkpoint = torch.load(init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if unexpected:
            raise ValueError(f"checkpoint has unexpected keys: {unexpected[:5]}")
        print(f"[init] loaded {init_from} epoch={checkpoint.get('epoch')}; new tensors={len(missing)}")
    if freeze:
        if not init_from or model.joint_encoder is None:
            raise ValueError("frozen exhaustive fusion requires init_from and use_joint=true")
        frozen_modules = [model.meg_encoder, model.fmri_encoder, model.text_encoder]
        for module in frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False
        print("[freeze] MEG/fMRI/text encoders frozen; only fusion head and temperature train")

    training_objective = resolve_training_objective(
        str(train_cfg.get("objective", "auto")), bool(frozen_modules)
    )
    print(f"[objective] {training_objective}")

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"[model] {total_params/1e6:.2f}M total; {trainable_params/1e6:.2f}M trainable")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    epochs = int(args.epochs or train_cfg["epochs"])
    checkpoint_dir = Path(args.checkpoint_dir or train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    best_score = -float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        for module in frozen_modules:
            module.eval()
        started = time.time()
        totals = {}
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
                lambda_mf=float(train_cfg.get("lambda_mf", 1.0)),
                lambda_jt=float(train_cfg.get("lambda_jt", 1.0)),
            )
            if training_objective == "joint_text_only":
                losses["total"] = float(train_cfg.get("lambda_jt", 1.0)) * losses["joint_text"]
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, batch {batches}")
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            batches += 1

        for key in totals:
            totals[key] /= max(batches, 1)
        metrics = evaluate_all_subject_combinations(
            model,
            val_ds,
            device,
            batch_size=int(train_cfg.get("eval_base_window_batch_size", base_batch_size)),
            num_workers=num_workers,
            pair_chunk_size=int(train_cfg.get("eval_pair_chunk_size", 8)),
        )
        elapsed = time.time() - started
        print(
            f"[epoch {epoch}/{epochs}] "
            + " ".join(f"{key}={value:.4f}" for key, value in totals.items())
            + f" [{elapsed:.1f}s]"
        )
        print("    " + format_metric("MEG->text", metrics["meg_text"]))
        print("    " + format_metric("fMRI->text", metrics["fmri_text"]))
        print("    " + format_metric("late->text", metrics["late_text"]))
        if metrics["joint_text"] is not None:
            print("    " + format_metric("learned->text", metrics["joint_text"]))
        if metrics["attention_weights"] is not None:
            weights = metrics["attention_weights"]
            print(
                "    attention weights: "
                f"MEG mean={weights['meg_mean']:.4f} std={weights['meg_std']:.4f} "
                f"p10/median/p90={weights['meg_p10']:.4f}/"
                f"{weights['meg_median']:.4f}/{weights['meg_p90']:.4f}; "
                f"fMRI mean={weights['fmri_mean']:.4f}"
            )

        selected = metrics["joint_text"]
        if selected is None:
            mean_medr = (
                metrics["meg_text"]["global"]["median_rank_mean"]
                + metrics["fmri_text"]["global"]["median_rank_mean"]
            ) / 2
        else:
            mean_medr = selected["global"]["median_rank_mean"]
        score = -mean_medr
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dims": dims,
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": metrics,
                },
                best_path,
            )
            print(f"[checkpoint] {best_path} mean group medR={mean_medr:.2f}")

    print(f"[done] best mean group medR={-best_score:.2f} checkpoint={best_path}")


if __name__ == "__main__":
    main()
