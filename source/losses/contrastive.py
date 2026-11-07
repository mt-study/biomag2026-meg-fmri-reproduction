"""Symmetric CLIP-style InfoNCE loss."""
from typing import Dict

import torch
import torch.nn.functional as F


def info_nce(a: torch.Tensor, b: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE between two batches of L2-normalized embeddings.

    Positives are the diagonal (index i in ``a`` corresponds to index i in
    ``b``); everything else in the batch is a negative.
    """
    logits = logit_scale * a @ b.t()
    targets = torch.arange(a.shape[0], device=a.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.t(), targets)
    return (loss_a + loss_b) / 2


def brain_clip_loss(
    meg_emb: torch.Tensor,
    fmri_emb: torch.Tensor,
    text_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    lambda_mf: float = 1.0,
    joint_emb: torch.Tensor = None,
    lambda_jt: float = 1.0,
) -> Dict[str, torch.Tensor]:
    loss_meg_text = info_nce(meg_emb, text_emb, logit_scale)
    loss_fmri_text = info_nce(fmri_emb, text_emb, logit_scale)
    loss_meg_fmri = info_nce(meg_emb, fmri_emb, logit_scale)
    total = loss_meg_text + loss_fmri_text + lambda_mf * loss_meg_fmri
    out = {
        "total": total,
        "meg_text": loss_meg_text,
        "fmri_text": loss_fmri_text,
        "meg_fmri": loss_meg_fmri,
    }
    if joint_emb is not None:
        # The single-modality terms are kept so the MEG and fMRI branches stay
        # trained and directly comparable against the fused branch within one
        # run; the joint term is added rather than replacing them.
        loss_joint_text = info_nce(joint_emb, text_emb, logit_scale)
        out["joint_text"] = loss_joint_text
        out["total"] = total + lambda_jt * loss_joint_text
    return out
