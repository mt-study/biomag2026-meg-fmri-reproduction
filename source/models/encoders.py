"""Per-modality encoders. Deliberately simple: a 1D CNN for MEG, and small
MLPs for fMRI and text. No cross-attention, no generative components.
"""
import torch
import torch.nn as nn


class MEGEncoder(nn.Module):
    """1D CNN over MEG sensors x time -> fixed-size embedding.

    Input: [batch, channels, time]

    Two things here are load-bearing:

    * **No global average pooling.** The first version ended in
      ``AdaptiveAvgPool1d(1)`` and collapsed completely -- every window
      produced the same embedding (mean off-diagonal cosine 1.000) and the
      meg-text loss sat at exactly ln(batch_size). MEG windows are z-scored
      per channel, so each window is ~zero-mean over time and the stimulus
      information lives in the temporal *pattern*; averaging over time is
      precisely the operation that discards it. The time axis is instead
      reduced by strided convolutions and then flattened.
    * **A width-1 spatial filter first.** Mixing the 306 sensors before any
      temporal convolution is the standard MEG decoding front end: the
      network learns sensor combinations (roughly, spatial sources)
      separately from the temporal filters that follow.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        time_steps: int,
        dropout: float = 0.3,
        temporal_bins: int | None = None,
    ):
        super().__init__()
        c = hidden_channels
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=1),          # spatial mixing
            nn.BatchNorm1d(c),
            nn.GELU(),
            nn.Conv1d(c, c, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c, c, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(c),
            nn.GELU(),
        )
        t = time_steps
        for _ in range(3):          # three stride-2 stages
            t = (t + 1) // 2
        if temporal_bins is not None and int(temporal_bins) <= 0:
            raise ValueError("temporal_bins must be positive")
        self.temporal_bins = (
            None if temporal_bins is None else int(temporal_bins)
        )
        self.temporal_pool = (
            nn.Identity()
            if self.temporal_bins is None
            else nn.AdaptiveAvgPool1d(self.temporal_bins)
        )
        pooled_t = t if self.temporal_bins is None else self.temporal_bins
        self.flat_dim = c * pooled_t
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.flat_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.temporal_pool(self.features(x))
        return self.proj(features)


class FMRIEncoder(nn.Module):
    """Two-layer MLP over the concatenated multi-delay fMRI feature vector.

    Input: [batch, fmri_dim]

    Dropout is not optional here: with 2.2k training windows and a 2304-dim
    input, the undropped version drove its training loss to ~0.03 (i.e.
    memorized the training set) while validation retrieval barely moved.
    """

    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarFusion(nn.Module):
    """Two learnable scalars: ``normalize(a * meg_emb + b * fmri_emb)``.

    Sits between the two extremes that bracket it -- parameter-free late
    fusion (fixed 50:50) and the 0.40M-parameter MLP head. Late fusion is
    dragged down by weighting a low-precision MEG embedding equally with a
    high-precision fMRI one; the MLP can correct that but visibly overfits
    2.2k training windows. Two scalars can correct the weighting and
    essentially cannot overfit.

    Initialized at a = b = 1, i.e. *exactly* late fusion, so training starts
    at the control and any movement away from it is a measured preference.
    Because the output is L2-normalized, only the ratio a/b is identified --
    which makes it directly readable as how much the model thinks MEG is
    worth relative to fMRI.

    Consumes the already-normalized embeddings, so the ratio is not confounded
    by the two branches having different projection norms.
    """

    uses_normalized = True

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2))

    def forward(self, meg_emb: torch.Tensor, fmri_emb: torch.Tensor) -> torch.Tensor:
        return self.weights[0] * meg_emb + self.weights[1] * fmri_emb

    def ratio(self) -> float:
        """MEG weight relative to fMRI; 1.0 == late fusion."""
        w = self.weights.detach()
        return float(w[0] / w[1])


class GatedAttentionFusion(nn.Module):
    """Dynamic per-window attention over normalized MEG and fMRI embeddings.

    The final scoring layer is zero-initialized, so every sample starts at
    exactly 0.5/0.5 (the parameter-free late-fusion control). Training can
    then move each window and subject pair toward the more reliable modality.
    """

    uses_normalized = True

    def __init__(self, embedding_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def attention_weights(self, meg_emb: torch.Tensor, fmri_emb: torch.Tensor) -> torch.Tensor:
        logits = self.score(torch.cat([meg_emb, fmri_emb], dim=-1))
        return torch.softmax(logits, dim=-1)

    def forward(self, meg_emb: torch.Tensor, fmri_emb: torch.Tensor) -> torch.Tensor:
        weights = self.attention_weights(meg_emb, fmri_emb)
        return weights[..., :1] * meg_emb + weights[..., 1:] * fmri_emb


class JointEncoder(nn.Module):
    """Fuse the MEG and fMRI projections into one brain embedding.

    Input: the two encoders' *pre-normalization* 256-d outputs.

    Fusing after each modality's own encoder (rather than concatenating raw
    features) keeps the single-modality branches trainable and directly
    comparable, so one run yields MEG-only, fMRI-only and joint numbers on
    the same checkpoint. It also keeps the added parameter count small,
    which matters at 2.2k training windows.

    The control this must beat is late fusion -- simply averaging the two
    L2-normalized embeddings, which costs no parameters at all. If the
    learned head does not beat that, it is not earning its capacity.
    """

    uses_normalized = False

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, meg_proj: torch.Tensor, fmri_proj: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([meg_proj, fmri_proj], dim=-1))


class TextEncoder(nn.Module):
    """Two-layer MLP projection head over precomputed/cached text embeddings.

    Input: [batch, text_dim]
    """

    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
