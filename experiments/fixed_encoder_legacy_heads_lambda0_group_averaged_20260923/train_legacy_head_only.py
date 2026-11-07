"""Train one frozen legacy-scale fusion head from an existing encoder checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


EXPECTED_SOURCE_HASHES = {
    "unified_benchmark.py": "4667fa26e40a2fc8adcd1cd27a3882f94db30f354157b3fbb58b7d5389e01ce9",
    "exhaustive_subject_pair.py": "0665336fb0f4cd9acec367693c3102a3a0cacfb1d6f94cb4dc57fa926ec01a3a",
    "models/brain_clip.py": "ddf89ce99c7839f865858165bf4d514953208faf6e11ccd9e01499695f79dc66",
}
EXPECTED_HIDDEN_DIMS = {"mlp": 512, "attention": 767}
EXPECTED_HEAD_PARAMETERS = {"mlp": 395008, "attention": 395007}
EXPECTED_TRAINABLE_PARAMETERS = {"mlp": 395009, "attention": 395008}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(source_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_source(source_root: Path) -> dict[str, str]:
    observed = {}
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source file is missing: {path}")
        observed[relative] = sha256(path)
        if observed[relative] != expected:
            raise RuntimeError(
                f"source hash mismatch for {relative}: "
                f"expected {expected}, observed {observed[relative]}"
            )
    return observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--protocol", choices=("all_subjects", "unseen_subject"), required=True
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--encoder-seed", type=int, required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--encoder-result", required=True)
    parser.add_argument("--fusion-mode", choices=("mlp", "attention"), required=True)
    parser.add_argument("--head-seed", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="verify provenance, load the frozen encoder, and check head size only",
    )
    return parser.parse_args()


def validate_request(args: argparse.Namespace, config: dict, encoder_result: dict) -> None:
    if args.protocol == "all_subjects" and args.fold is not None:
        raise ValueError("all_subjects must not specify --fold")
    if args.protocol == "unseen_subject" and args.fold not in range(5):
        raise ValueError("unseen_subject requires --fold in 0..4")
    if args.head_seed not in {42, 43, 44}:
        raise ValueError("head seed must be one of 42, 43, 44")

    hidden_dims = config["model"]["fusion_hidden_dims"]
    if hidden_dims != EXPECTED_HIDDEN_DIMS:
        raise ValueError(
            f"legacy hidden dimensions changed: expected {EXPECTED_HIDDEN_DIMS}, "
            f"observed {hidden_dims}"
        )
    training = config["training"]
    required_training = {
        "fusion_epochs": 60,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "lambda_mf": 0.0,
        "lambda_jt": 1.0,
        "checkpoint_selection": "exhaustive_group_medr_validation_best",
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            raise ValueError(
                f"unexpected training.{key}: expected {expected!r}, "
                f"observed {training.get(key)!r}"
            )

    split = encoder_result["split"]
    if split["protocol"] != args.protocol or split["data_mode"] != "mean":
        raise ValueError(
            "encoder result is not from the requested protocol with "
            "group-averaged/mean input"
        )
    if split.get("fold") != args.fold:
        raise ValueError(
            f"encoder fold mismatch: expected {args.fold}, observed {split.get('fold')}"
        )
    if int(encoder_result["seed"]) != args.encoder_seed:
        raise ValueError("encoder seed does not match the requested encoder seed")
    hyperparameters = encoder_result["hyperparameters"]
    if float(hyperparameters["lambda_mf"]) != 0.0:
        raise ValueError("encoder was not trained with lambda_mf=0.0")
    if float(hyperparameters["lambda_jt"]) != 1.0:
        raise ValueError("encoder was not trained with lambda_jt=1.0")

    checkpoint = Path(args.encoder_checkpoint).resolve()
    recorded = Path(encoder_result["checkpoints"]["encoder"]).resolve()
    if checkpoint != recorded:
        raise ValueError(
            f"encoder checkpoint path differs from result provenance: "
            f"{checkpoint} != {recorded}"
        )


def train_legacy_head(
    stage,
    model,
    train_dataset,
    selection_dataset,
    config,
    device,
    sampler_seed,
    checkpoint_path,
    frozen_modules,
    make_train_loader,
    exhaustive_brain_clip_loss,
    evaluate_all_subject_combinations,
):
    """Match the legacy frozen-head training and group-MedR selection policy."""
    import torch

    loader = make_train_loader(train_dataset, config, sampler_seed)
    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["fusion_epochs"])
    history = []
    total_steps = 0
    best_medr = float("inf")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

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
                lambda_mf=float(train_cfg["lambda_mf"]),
                lambda_jt=float(train_cfg["lambda_jt"]),
            )
            # The legacy frozen-head runs optimize joint-to-text only.
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

        selection_metrics = evaluate_all_subject_combinations(
            model,
            selection_dataset,
            device,
            batch_size=int(train_cfg["eval_base_window_batch_size"]),
            num_workers=int(train_cfg["num_workers"]),
            pair_chunk_size=int(train_cfg["eval_pair_chunk_size"]),
        )
        joint = selection_metrics["joint_text"]
        mean_global_medr = float(joint["global"]["median_rank_mean"])
        record = {
            "epoch": epoch,
            "batches": batches,
            "losses": totals,
            "selection_mean_global_medr": mean_global_medr,
        }
        history.append(record)
        print(
            f"[{stage} epoch {epoch}/{epochs}] "
            + " ".join(f"{key}={value:.4f}" for key, value in totals.items())
            + f" batches={batches} [{time.time() - started:.1f}s]"
        )
        print(
            f"    selection all-pair global: top10="
            f"{joint['global']['top10_mean']:.4f} "
            f"medR={mean_global_medr:.2f} groups={joint['global']['groups']}"
        )
        if mean_global_medr < best_medr:
            best_medr = mean_global_medr
            torch.save(
                {
                    "stage": stage,
                    "epoch": epoch,
                    "selection": "exhaustive_group_medr_validation_best",
                    "selection_mean_global_medr": mean_global_medr,
                    "selection_metrics": selection_metrics,
                    "model_state_dict": model.state_dict(),
                    "optimizer_steps": total_steps,
                    "sampler_seed": int(sampler_seed),
                },
                checkpoint_path,
            )
            print(
                f"[checkpoint] {stage} new best epoch {epoch} "
                f"mean group medR={mean_global_medr:.2f} -> {checkpoint_path}"
            )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["history"] = history
    checkpoint["total_optimizer_steps"] = total_steps
    torch.save(checkpoint, checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"[checkpoint] restored {stage} best epoch {checkpoint['epoch']} "
        f"mean group medR={checkpoint['selection_mean_global_medr']:.2f}"
    )
    return checkpoint


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    source_hashes = verify_source(source_root)
    sys.path.insert(0, str(source_root))

    import torch
    from exhaustive_subject_pair import (
        evaluate_all_subject_combinations,
        exhaustive_brain_clip_loss,
    )
    from unified_benchmark import (
        build_unified_datasets,
        load_frozen_fusion,
        make_train_loader,
        set_seed,
    )

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    encoder_checkpoint_path = Path(args.encoder_checkpoint).resolve()
    encoder_result_path = Path(args.encoder_result).resolve()
    if not encoder_checkpoint_path.is_file():
        raise FileNotFoundError(encoder_checkpoint_path)
    if not encoder_result_path.is_file():
        raise FileNotFoundError(encoder_result_path)
    encoder_result = json.loads(encoder_result_path.read_text(encoding="utf-8"))
    validate_request(args, config, encoder_result)

    set_seed(args.head_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, split_info, dims = build_unified_datasets(
        config,
        args.protocol,
        "mean",
        args.fold,
        seed=args.encoder_seed,
    )
    encoder_checkpoint = torch.load(
        encoder_checkpoint_path, map_location="cpu", weights_only=False
    )
    if int(encoder_checkpoint["epoch"]) != int(
        encoder_result["selected_epochs"]["encoder"]
    ):
        raise ValueError("encoder checkpoint epoch differs from recorded result")

    model, frozen_modules = load_frozen_fusion(
        config,
        dims,
        device,
        args.fusion_mode,
        encoder_checkpoint["model_state_dict"],
    )
    trainable_parameters = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    head_parameters = int(
        sum(parameter.numel() for parameter in model.joint_encoder.parameters())
    )
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    unexpected_trainable = {
        name
        for name in trainable_names
        if name != "log_logit_scale" and not name.startswith("joint_encoder.")
    }
    if unexpected_trainable or "log_logit_scale" not in trainable_names:
        raise ValueError(
            "frozen model has unexpected trainable parameters: "
            f"names={sorted(trainable_names)}"
        )
    expected_head_parameters = EXPECTED_HEAD_PARAMETERS[args.fusion_mode]
    if head_parameters != expected_head_parameters:
        raise ValueError(
            f"head parameter mismatch for {args.fusion_mode}: "
            f"expected {expected_head_parameters}, observed {head_parameters}"
        )
    expected_parameters = EXPECTED_TRAINABLE_PARAMETERS[args.fusion_mode]
    if trainable_parameters != expected_parameters:
        raise ValueError(
            f"trainable parameter mismatch for {args.fusion_mode}: "
            f"expected {expected_parameters}, observed {trainable_parameters}"
        )

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    checkpoint_path = checkpoint_dir / "head_best.pt"
    result_path = result_dir / "result.json"

    print(
        f"[run] protocol={args.protocol} fold={args.fold} "
        f"encoder_seed={args.encoder_seed} mode={args.fusion_mode} "
        f"head_seed={args.head_seed}"
    )
    print(f"[encoder] {encoder_checkpoint_path} epoch={encoder_checkpoint['epoch']}")
    print(
        f"[fusion] hidden={EXPECTED_HIDDEN_DIMS[args.fusion_mode]} "
        f"head={head_parameters} temperature=1 total_trainable={trainable_parameters}; "
        "encoders frozen"
    )
    if args.validate_only:
        print("[validate-only] all checks passed; training was not started")
        return
    if checkpoint_path.exists() or result_path.exists():
        raise FileExistsError(
            "refusing to overwrite an existing run: "
            f"checkpoint={checkpoint_path.exists()} result={result_path.exists()}"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    head_checkpoint = train_legacy_head(
        args.fusion_mode,
        model,
        datasets["train"],
        datasets["selection"],
        config,
        device,
        args.head_seed,
        checkpoint_path,
        frozen_modules,
        make_train_loader,
        exhaustive_brain_clip_loss,
        evaluate_all_subject_combinations,
    )
    metrics = evaluate_all_subject_combinations(
        model,
        datasets["test"],
        device,
        batch_size=int(config["training"]["eval_base_window_batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        pair_chunk_size=int(config["training"]["eval_pair_chunk_size"]),
    )

    result = {
        "experiment": "fixed_encoder_legacy_heads_lambda0_group_averaged",
        "date": "2026-09-23",
        "source_root": str(source_root),
        "source_commit": git_commit(source_root),
        "source_hashes": source_hashes,
        "config": str(config_path),
        "protocol": args.protocol,
        "fold": args.fold,
        "data_mode": "mean",
        "encoder_seed": args.encoder_seed,
        "encoder_checkpoint": str(encoder_checkpoint_path),
        "encoder_result": str(encoder_result_path),
        "encoder_epoch": int(encoder_checkpoint["epoch"]),
        "encoders_frozen": True,
        "fusion_mode": args.fusion_mode,
        "fusion_hidden_dim": EXPECTED_HIDDEN_DIMS[args.fusion_mode],
        "head_seed": args.head_seed,
        "fusion_head_parameters": head_parameters,
        "trainable_temperature_parameters": 1,
        "trainable_fusion_parameters": trainable_parameters,
        "selected_head_epoch": int(head_checkpoint["epoch"]),
        "optimizer_steps": int(head_checkpoint["optimizer_steps"]),
        "hyperparameters": {
            "fusion_epochs": int(config["training"]["fusion_epochs"]),
            "learning_rate": float(config["training"]["learning_rate"]),
            "weight_decay": float(config["training"]["weight_decay"]),
            "lambda_mf": float(config["training"]["lambda_mf"]),
            "lambda_jt": float(config["training"]["lambda_jt"]),
            "noise_std": float(config["sampling"]["noise_std"]),
            "checkpoint_selection": config["training"]["checkpoint_selection"],
        },
        "split": split_info,
        "checkpoint": str(checkpoint_path),
        "test_metrics": {
            "meg": metrics["meg_text"],
            "fmri": metrics["fmri_text"],
            "late": metrics["late_text"],
            args.fusion_mode: metrics["joint_text"],
        },
        "attention_weights": (
            metrics["attention_weights"]
            if args.fusion_mode == "attention"
            else None
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[result] {result_path}")


if __name__ == "__main__":
    main()
