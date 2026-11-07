"""BrainCLIP: three encoders projected into one shared, L2-normalized space,
plus an optional MEG+fMRI fusion branch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import (
    FMRIEncoder,
    GatedAttentionFusion,
    JointEncoder,
    MEGEncoder,
    ScalarFusion,
    TextEncoder,
)


class BrainCLIP(nn.Module):
    def __init__(
        self,
        meg_channels: int,
        meg_time: int,
        fmri_dim: int,
        text_dim: int,
        embedding_dim: int = 256,
        meg_hidden_channels: int = 64,
        fmri_hidden_dim: int = 512,
        text_hidden_dim: int = 512,
        joint_hidden_dim: int = 512,
        use_joint: bool = False,
        fusion_mode: str = "mlp",
        dropout: float = 0.3,
        init_temperature: float = 0.07,
        meg_temporal_bins: int | None = None,
    ):
        super().__init__()
        self.meg_encoder = MEGEncoder(
            meg_channels,
            meg_hidden_channels,
            embedding_dim,
            time_steps=meg_time,
            dropout=dropout,
            temporal_bins=meg_temporal_bins,
        )
        self.fmri_encoder = FMRIEncoder(fmri_dim, fmri_hidden_dim, embedding_dim, dropout=dropout)
        self.text_encoder = TextEncoder(text_dim, text_hidden_dim, embedding_dim)
        if not use_joint:
            self.joint_encoder = None
        elif fusion_mode == "scalar":
            self.joint_encoder = ScalarFusion()
        elif fusion_mode == "mlp":
            self.joint_encoder = JointEncoder(embedding_dim, joint_hidden_dim, dropout=dropout)
        elif fusion_mode == "attention":
            self.joint_encoder = GatedAttentionFusion(
                embedding_dim, joint_hidden_dim, dropout=dropout
            )
        else:
            raise ValueError(
                f"unknown fusion_mode {fusion_mode!r}; expected 'mlp', 'scalar' or 'attention'"
            )
        self.log_logit_scale = nn.Parameter(torch.tensor(float(np.log(1.0 / init_temperature))))

    def encode_meg(self, meg: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.meg_encoder(meg), dim=-1)

    def encode_fmri(self, fmri: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fmri_encoder(fmri), dim=-1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_encoder(text), dim=-1)

    @staticmethod
    def late_fusion(meg_emb: torch.Tensor, fmri_emb: torch.Tensor) -> torch.Tensor:
        """Parameter-free fusion: mean of the two normalized embeddings.

        This is the control the learned joint head has to beat.
        """
        return F.normalize(meg_emb + fmri_emb, dim=-1)

    def forward(self, meg: torch.Tensor, fmri: torch.Tensor, text: torch.Tensor):
        meg_proj = self.meg_encoder(meg)
        fmri_proj = self.fmri_encoder(fmri)
        meg_emb = F.normalize(meg_proj, dim=-1)
        fmri_emb = F.normalize(fmri_proj, dim=-1)
        text_emb = self.encode_text(text)
        logit_scale = self.log_logit_scale.exp().clamp(max=100.0)
        joint_emb = None
        if self.joint_encoder is not None:
            # Scalar and attention fusion consume normalized embeddings so
            # their weights are not confounded by projection norms; the MLP
            # head consumes raw projections.
            a, b = (
                (meg_emb, fmri_emb)
                if self.joint_encoder.uses_normalized
                else (meg_proj, fmri_proj)
            )
            joint_emb = F.normalize(self.joint_encoder(a, b), dim=-1)
        return meg_emb, fmri_emb, text_emb, logit_scale, joint_emb
