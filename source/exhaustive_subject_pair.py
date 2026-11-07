"""Exact all-subject MEG/fMRI training and evaluation utilities.

Subject pairs are a group dimension, not duplicate samples, so repeated
measurements of one text window never become contrastive negatives.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.brain_clip import BrainCLIP


class AllSubjectWindowCacheDataset(Dataset):
    """One item per stimulus window with complete MEG/fMRI subject axes."""

    def __init__(
        self,
        meg,
        fmri,
        text,
        sections,
        times,
        indices,
        noise_std: float = 0.0,
        seed: int = 42,
    ):
        if len(indices) == 0:
            raise ValueError("AllSubjectWindowCacheDataset received zero windows")
        assert len(meg) == len(fmri) == len(text) == len(sections) == len(times)
        self.meg = meg
        self.fmri = fmri
        self.text = text
        self.sections = sections
        self.times = times
        self.indices = np.asarray(indices, dtype=np.int64)
        self.noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed)

    def reset_noise(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_idx = int(self.indices[idx])
        meg = np.array(self.meg[base_idx], dtype=np.float32, copy=True)
        fmri = np.array(self.fmri[base_idx], dtype=np.float32, copy=True)
        if self.noise_std > 0:
            meg += self._rng.normal(0, self.noise_std, meg.shape).astype(np.float32)
            fmri += self._rng.normal(0, self.noise_std, fmri.shape).astype(np.float32)
        return {
            "meg": torch.from_numpy(meg),
            "fmri": torch.from_numpy(fmri),
            "text": torch.from_numpy(np.array(self.text[base_idx], dtype=np.float32, copy=True)),
            "section": int(self.sections[base_idx]),
            "t": float(self.times[base_idx]),
        }


def build_all_subject_datasets(config: dict):
    cache_dir = Path(config["project_dir"]) / config["data"].get(
        "window_cache", "cache/windows_subject_pairs"
    )
    required = ["meg_windows.npy", "fmri_windows.npy", "text_windows.npy", "windows_meta.npz"]
    missing = [name for name in required if not (cache_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Subject-pair cache incomplete at {cache_dir} (missing {', '.join(missing)})"
        )

    meg = np.load(cache_dir / "meg_windows.npy", mmap_mode="r")
    fmri = np.load(cache_dir / "fmri_windows.npy", mmap_mode="r")
    text = np.load(cache_dir / "text_windows.npy", mmap_mode="r")
    meta = np.load(cache_dir / "windows_meta.npz", allow_pickle=False)
    sections, times = meta["section"], meta["t"]

    align_cfg = config["alignment"]
    train_sections = set(align_cfg["train_sections"])
    val_sections = set(align_cfg["val_sections"])
    overlap = train_sections & val_sections
    if overlap:
        raise ValueError(f"train_sections and val_sections overlap on {sorted(overlap)}")

    train_idx = np.flatnonzero(np.isin(sections, list(train_sections)))
    val_idx = np.flatnonzero(np.isin(sections, list(val_sections)))
    noise_std = float(config.get("sampling", {}).get("noise_std", 0.0))
    seed = int(config.get("training", {}).get("seed", 42))
    train_ds = AllSubjectWindowCacheDataset(
        meg,
        fmri,
        text,
        sections,
        times,
        train_idx,
        noise_std=noise_std,
        seed=seed,
    )
    val_ds = AllSubjectWindowCacheDataset(meg, fmri, text, sections, times, val_idx)
    info = {
        "cache_mode": "subject_pair_exhaustive",
        "n_windows": len(text),
        "train_sections": sorted(train_sections),
        "val_sections": sorted(val_sections),
        "meg_subjects": int(meg.shape[1]),
        "fmri_subjects": int(fmri.shape[1]),
        "subject_pairs": int(meg.shape[1] * fmri.shape[1]),
    }
    dims = {
        "meg_channels": int(meg.shape[2]),
        "meg_time": int(meg.shape[3]),
        "fmri_dim": int(fmri.shape[2]),
        "text_dim": int(text.shape[1]),
    }
    return train_ds, val_ds, info, dims


def _symmetric_group_info_nce(logits: torch.Tensor) -> torch.Tensor:
    """Average symmetric InfoNCE over ``[groups, windows, windows]``."""
    groups, batch, candidate_count = logits.shape
    if batch != candidate_count:
        raise ValueError(f"expected square grouped logits, got {tuple(logits.shape)}")
    targets = torch.arange(batch, device=logits.device).repeat(groups)
    forward = F.cross_entropy(logits.reshape(groups * batch, batch), targets)
    backward = F.cross_entropy(logits.transpose(-1, -2).reshape(groups * batch, batch), targets)
    return (forward + backward) / 2


def exhaustive_brain_clip_loss(
    model: BrainCLIP,
    meg: torch.Tensor,
    fmri: torch.Tensor,
    text: torch.Tensor,
    lambda_mf: float = 1.0,
    lambda_jt: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Average losses over all subjects and all MEG x fMRI combinations."""
    if meg.ndim != 4 or fmri.ndim != 3 or text.ndim != 2:
        raise ValueError(f"unexpected shapes: {tuple(meg.shape)}, {tuple(fmri.shape)}, {tuple(text.shape)}")
    batch, n_meg, channels, samples = meg.shape
    fmri_batch, n_fmri, fmri_dim = fmri.shape
    if batch != fmri_batch or batch != len(text):
        raise ValueError("MEG, fMRI and text batch dimensions differ")

    meg_proj = model.meg_encoder(meg.reshape(batch * n_meg, channels, samples))
    embedding_dim = meg_proj.shape[-1]
    meg_proj = meg_proj.reshape(batch, n_meg, embedding_dim).permute(1, 0, 2)
    fmri_proj = model.fmri_encoder(fmri.reshape(batch * n_fmri, fmri_dim))
    fmri_proj = fmri_proj.reshape(batch, n_fmri, embedding_dim).permute(1, 0, 2)
    text_emb = model.encode_text(text)
    meg_emb = F.normalize(meg_proj, dim=-1)
    fmri_emb = F.normalize(fmri_proj, dim=-1)
    scale = model.log_logit_scale.exp().clamp(max=100.0)

    mt_logits = scale * torch.einsum("mbd,cd->mbc", meg_emb, text_emb)
    ft_logits = scale * torch.einsum("fbd,cd->fbc", fmri_emb, text_emb)
    mf_logits = scale * torch.einsum("mbd,fcd->mfbc", meg_emb, fmri_emb)
    mf_logits = mf_logits.reshape(n_meg * n_fmri, batch, batch)
    losses = {
        "meg_text": _symmetric_group_info_nce(mt_logits),
        "fmri_text": _symmetric_group_info_nce(ft_logits),
        "meg_fmri": _symmetric_group_info_nce(mf_logits),
    }
    losses["total"] = losses["meg_text"] + losses["fmri_text"] + lambda_mf * losses["meg_fmri"]

    if model.joint_encoder is not None:
        meg_input = meg_emb if model.joint_encoder.uses_normalized else meg_proj
        fmri_input = fmri_emb if model.joint_encoder.uses_normalized else fmri_proj
        meg_pairs = meg_input[:, None].expand(n_meg, n_fmri, batch, embedding_dim)
        fmri_pairs = fmri_input[None].expand(n_meg, n_fmri, batch, embedding_dim)
        joint = model.joint_encoder(
            meg_pairs.reshape(-1, embedding_dim), fmri_pairs.reshape(-1, embedding_dim)
        )
        joint = F.normalize(joint, dim=-1).reshape(n_meg * n_fmri, batch, embedding_dim)
        jt_logits = scale * torch.einsum("pbd,cd->pbc", joint, text_emb)
        losses["joint_text"] = _symmetric_group_info_nce(jt_logits)
        losses["total"] = losses["total"] + lambda_jt * losses["joint_text"]
    return losses

@torch.no_grad()
def _encode_validation(model, dataset, device, batch_size: int, num_workers: int):
    model.eval()
    meg_raw, fmri_raw, text_emb, sections, times = [], [], [], [], []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    for batch_data in loader:
        meg = batch_data["meg"].to(device)
        fmri = batch_data["fmri"].to(device)
        text = batch_data["text"].to(device)
        batch, n_meg, channels, samples = meg.shape
        _, n_fmri, fmri_dim = fmri.shape
        m = model.meg_encoder(meg.reshape(batch * n_meg, channels, samples))
        f = model.fmri_encoder(fmri.reshape(batch * n_fmri, fmri_dim))
        dim = m.shape[-1]
        meg_raw.append(m.reshape(batch, n_meg, dim).permute(1, 0, 2).cpu())
        fmri_raw.append(f.reshape(batch, n_fmri, dim).permute(1, 0, 2).cpu())
        text_emb.append(model.encode_text(text).cpu())
        sections.append(batch_data["section"])
        times.append(batch_data["t"])
    return (
        torch.cat(meg_raw, dim=1),
        torch.cat(fmri_raw, dim=1),
        torch.cat(text_emb, dim=0),
        torch.cat(sections).numpy(),
        torch.cat(times).numpy(),
    )


def _rank_arrays(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    sections: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return global and same-section ranks for grouped query sets."""
    if candidates.ndim == 2:
        candidates = candidates.unsqueeze(0).expand(len(queries), -1, -1)
    sim = torch.bmm(queries, candidates.transpose(1, 2))
    n = sim.shape[1]
    diagonal = torch.arange(n, device=sim.device)
    correct = sim[:, diagonal, diagonal].unsqueeze(-1)
    higher = sim > correct
    global_ranks = higher.sum(dim=-1)
    same_section = sections[:, None].eq(sections[None, :])
    section_ranks = (higher & same_section.unsqueeze(0)).sum(dim=-1)
    return global_ranks.cpu().numpy(), section_ranks.cpu().numpy()


def _summarize_ranks(global_ranks: np.ndarray, section_ranks: np.ndarray) -> dict:
    def summarize(ranks: np.ndarray) -> dict:
        medr = np.median(ranks, axis=1)
        mean_rank = ranks.mean(axis=1)
        mrr = (1.0 / (ranks + 1)).mean(axis=1)
        top1 = (ranks == 0).mean(axis=1)
        top5 = (ranks < 5).mean(axis=1)
        top10 = (ranks < 10).mean(axis=1)
        return {
            "groups": int(len(ranks)),
            "median_rank_mean": float(medr.mean()),
            "median_rank_std": float(medr.std(ddof=1)) if len(medr) > 1 else 0.0,
            "median_rank_min": float(medr.min()),
            "median_rank_max": float(medr.max()),
            "mean_rank_mean": float(mean_rank.mean()),
            "mean_rank_std": float(mean_rank.std(ddof=1)) if len(mean_rank) > 1 else 0.0,
            "mrr_mean": float(mrr.mean()),
            "mrr_std": float(mrr.std(ddof=1)) if len(mrr) > 1 else 0.0,
            "top1_mean": float(top1.mean()),
            "top1_std": float(top1.std(ddof=1)) if len(top1) > 1 else 0.0,
            "top5_mean": float(top5.mean()),
            "top5_std": float(top5.std(ddof=1)) if len(top5) > 1 else 0.0,
            "top10_mean": float(top10.mean()),
            "top10_std": float(top10.std(ddof=1)) if len(top10) > 1 else 0.0,
        }

    return {"global": summarize(global_ranks), "in_section": summarize(section_ranks)}


def _evaluate_group_chunks(
    chunks: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    sections: torch.Tensor,
) -> dict:
    global_parts, section_parts = [], []
    for queries, candidates in chunks:
        global_ranks, section_ranks = _rank_arrays(queries, candidates, sections)
        global_parts.append(global_ranks)
        section_parts.append(section_ranks)
    return _summarize_ranks(np.concatenate(global_parts), np.concatenate(section_parts))


@torch.no_grad()
def evaluate_all_subject_combinations(
    model: BrainCLIP,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 0,
    pair_chunk_size: int = 8,
) -> dict:
    """Evaluate individual subjects and all M x F fusion combinations."""
    meg_raw, fmri_raw, text, sections_np, _ = _encode_validation(
        model, dataset, device, batch_size, num_workers
    )
    meg_raw = meg_raw.to(device)
    fmri_raw = fmri_raw.to(device)
    text = text.to(device)
    sections = torch.as_tensor(sections_np, device=device)
    meg_emb = F.normalize(meg_raw, dim=-1)
    fmri_emb = F.normalize(fmri_raw, dim=-1)

    def subject_chunks(values):
        for start in range(0, len(values), pair_chunk_size):
            yield values[start : start + pair_chunk_size], text

    metrics = {
        "meg_text": _evaluate_group_chunks(subject_chunks(meg_emb), sections),
        "fmri_text": _evaluate_group_chunks(subject_chunks(fmri_emb), sections),
    }
    n_meg, n_fmri = len(meg_emb), len(fmri_emb)
    pair_ids = [(m, f) for m in range(n_meg) for f in range(n_fmri)]
    attention_parts = []

    def pair_chunks(kind: str):
        for start in range(0, len(pair_ids), pair_chunk_size):
            ids = pair_ids[start : start + pair_chunk_size]
            mi = torch.as_tensor([pair[0] for pair in ids], device=device)
            fi = torch.as_tensor([pair[1] for pair in ids], device=device)
            if kind == "meg_fmri":
                yield meg_emb[mi], fmri_emb[fi]
            elif kind == "late":
                yield BrainCLIP.late_fusion(meg_emb[mi], fmri_emb[fi]), text
            elif kind == "joint":
                meg_input = meg_emb[mi] if model.joint_encoder.uses_normalized else meg_raw[mi]
                fmri_input = fmri_emb[fi] if model.joint_encoder.uses_normalized else fmri_raw[fi]
                groups, windows, dim = meg_input.shape
                flat_meg = meg_input.reshape(groups * windows, dim)
                flat_fmri = fmri_input.reshape(groups * windows, dim)
                if hasattr(model.joint_encoder, "attention_weights"):
                    attention_parts.append(
                        model.joint_encoder.attention_weights(flat_meg, flat_fmri).cpu()
                    )
                joint = model.joint_encoder(
                    flat_meg,
                    flat_fmri,
                )
                yield F.normalize(joint, dim=-1).reshape(groups, windows, dim), text
            else:
                raise ValueError(kind)

    metrics["meg_fmri"] = _evaluate_group_chunks(pair_chunks("meg_fmri"), sections)
    metrics["late_text"] = _evaluate_group_chunks(pair_chunks("late"), sections)
    metrics["joint_text"] = (
        _evaluate_group_chunks(pair_chunks("joint"), sections)
        if model.joint_encoder is not None
        else None
    )
    if attention_parts:
        weights = torch.cat(attention_parts).numpy()
        meg_weights = weights[:, 0]
        metrics["attention_weights"] = {
            "meg_mean": float(meg_weights.mean()),
            "meg_std": float(meg_weights.std()),
            "meg_p10": float(np.quantile(meg_weights, 0.10)),
            "meg_median": float(np.median(meg_weights)),
            "meg_p90": float(np.quantile(meg_weights, 0.90)),
            "fmri_mean": float(weights[:, 1].mean()),
        }
    else:
        metrics["attention_weights"] = None
    metrics["n_windows"] = int(len(text))
    metrics["n_meg_subjects"] = int(n_meg)
    metrics["n_fmri_subjects"] = int(n_fmri)
    metrics["n_pairs"] = int(len(pair_ids))
    return metrics


def format_metric(name: str, metric: dict) -> str:
    global_stats = metric["global"]
    section_stats = metric["in_section"]
    return (
        f"{name}: global top10={global_stats['top10_mean']:.4f}+/-{global_stats['top10_std']:.4f} "
        f"medR={global_stats['median_rank_mean']:.1f}+/-{global_stats['median_rank_std']:.1f} | "
        f"in-section top10={section_stats['top10_mean']:.4f}+/-{section_stats['top10_std']:.4f} "
        f"medR={section_stats['median_rank_mean']:.1f}+/-{section_stats['median_rank_std']:.1f} "
        f"(groups={global_stats['groups']})"
    )
