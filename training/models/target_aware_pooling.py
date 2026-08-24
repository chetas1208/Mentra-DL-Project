"""P0 experiment: enrollment-aware, TRAINED attention pooling over
SpeakerNet's raw per-frame representation, plus a tiny fusion head.

Context (see docs/SEPARATION_RESEARCH.md / the recent V2 autopsy): frame-level
wearer-vs-environment identity signal is real and measured (10-vs-01 AUROC
0.78-0.87 across TIR), but clip-level EER is bad (18.75%/37.50%/45-50% clean/
TIR0/TIR-10) because the ONLY clip aggregation ever used is a naive mean over
frame-level wearer probability, which was never trained -- just assumed. A
sweep of 15 hand-designed pooling heuristics topped out around clip AUROC
0.70-0.72. This module tests whether a small TRAINED, enrollment-conditioned
attention pooling head can recover more of that signal.

Both modules below hold ONLY newly-trained parameters. Every frozen
component they consume (the SpeakerNet backbone's frame features, the
enrollment embedding, the frozen MentraWearNet's frame-level logits, and the
frozen backbone.embedding_projection module) is passed in from the outside
as a plain tensor / module reference -- never stored as a submodule here --
so `sum(p.numel() for p in TargetAwarePooler().parameters())` can never
accidentally include a single frozen parameter.

TargetAwarePooler param budget (proj_dim=32, attn_hidden=64): ~65k.
FusionHead param budget (4 -> 16 -> 1): ~100.
Total: well under the 200k target, comfortably under the 100k ideal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetAwarePooler(nn.Module):
    """Computes attention weights alpha_t over frames from a small MLP over
    enrollment-conditioned interaction features, then does weighted
    mean+std statistics pooling over the ORIGINAL (unprojected) 1500-D
    frame representation -- kept at 1500-D specifically so the pooled
    [mu, sigma] concatenation (3000-D) can be fed through the EXISTING,
    already-fixed, frozen `backbone.embedding_projection` (which expects
    exactly that StatsPoolLayer-shaped input), landing the result in the
    SAME embedding space as `encode_speaker()`'s enrollment embeddings --
    directly comparable via cosine similarity, no new projection needed for
    that final step.
    """

    def __init__(self, frame_channels: int = 1500, embedding_dim: int = 256,
                 proj_dim: int = 32, attn_hidden: int = 64, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.proj_dim = proj_dim

        # Project the (expensive, 1500-D) frame features and the 256-D
        # enrollment embedding down to a small shared interaction space.
        # This projection is ONLY used to compute attention weights -- the
        # actual pooling below runs over the original 1500-D h_t, not this
        # projected version (see class docstring).
        self.h_proj = nn.Linear(frame_channels, proj_dim)
        self.e_proj = nn.Linear(embedding_dim, proj_dim)

        # Interaction features: [h_proj, e_proj(expanded), h_proj*e_proj,
        # |h_proj-e_proj|] (4*proj_dim) + 3 scalar features (D_t,
        # wearer_probability, environment_probability).
        interaction_dim = proj_dim * 4 + 3
        self.attn_mlp = nn.Sequential(
            nn.Linear(interaction_dim, attn_hidden),
            nn.SiLU(),
            nn.Linear(attn_hidden, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        assert n_params < 100_000, f"TargetAwarePooler has {n_params:,} params, over the 100k budget"
        self.n_params = n_params

    def forward(self, h_t: torch.Tensor, frame_lengths: torch.Tensor,
                enrollment_embedding: torch.Tensor,
                wearer_logit: torch.Tensor, environment_logit: torch.Tensor,
                wearer_probability: torch.Tensor, environment_probability: torch.Tensor,
                embedding_projection: nn.Module) -> torch.Tensor:
        """
        h_t: [B, 1500, T] raw frame features (backbone.encode_frames() output,
             treated as a fixed input here -- caller is responsible for not
             requiring gradients back into the frozen backbone).
        frame_lengths: [B] valid frame counts (for masking padded frames).
        enrollment_embedding: [B, 256] (backbone.encode_speaker() output,
             already L2-normalized, fixed input).
        wearer_logit/environment_logit/wearer_probability/environment_probability:
             [B, T] frame-level outputs of the frozen MentraWearNet forward pass.
        embedding_projection: the FROZEN backbone.embedding_projection module
             (Linear(3000,256)+BatchNorm(affine=False), ReLU-free, already the
             fixed embedding-extraction path) -- passed in and called directly,
             never copied, so the target-aware embedding lands in exactly the
             same space as encode_speaker()'s output.

        Returns: target_aware_embedding [B, 256], L2-normalized.
        """
        B, C, T = h_t.shape
        device = h_t.device

        h_bt = h_t.transpose(1, 2)  # [B, T, 1500]
        h_p = self.h_proj(h_bt)  # [B, T, proj_dim]
        e_p = self.e_proj(enrollment_embedding)  # [B, proj_dim]
        e_p_exp = e_p.unsqueeze(1).expand(-1, T, -1)  # [B, T, proj_dim]

        prod = h_p * e_p_exp
        diff = (h_p - e_p_exp).abs()
        d_t = (wearer_logit - environment_logit).unsqueeze(-1)  # [B, T, 1]
        wp = wearer_probability.unsqueeze(-1)
        ep = environment_probability.unsqueeze(-1)

        feats = torch.cat([h_p, e_p_exp, prod, diff, d_t, wp, ep], dim=-1)  # [B, T, 4*proj_dim+3]
        attn_logit = self.attn_mlp(feats).squeeze(-1)  # [B, T]

        frame_idx = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        valid_mask = frame_idx < frame_lengths.unsqueeze(1)  # [B, T] bool
        attn_logit = attn_logit.masked_fill(~valid_mask, float("-inf"))
        alpha = torch.softmax(attn_logit, dim=1)  # [B, T], sums to 1 over valid frames

        mu = torch.einsum("bt,bct->bc", alpha, h_t)  # [B, 1500]
        var = torch.einsum("bt,bct->bc", alpha, (h_t - mu.unsqueeze(-1)) ** 2)  # [B, 1500]
        sigma = torch.sqrt(var + self.eps)  # [B, 1500]

        pooled = torch.cat([mu, sigma], dim=-1)  # [B, 3000]
        target_embedding = embedding_projection(pooled)  # [B, 256], frozen weights but gradients flow through
        return F.normalize(target_embedding, p=2, dim=-1)


class FusionHead(nn.Module):
    """Tiny MLP over 4 scalar per-window scores -> one window-level
    wearer-present logit. Lets training decide the tradeoff between
    S_original (reliable clean/low-overlap) and S_target_pool (meant to help
    under overlap) rather than hand-tuning it.
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        n_params = sum(p.numel() for p in self.parameters())
        assert n_params < 1_000, f"FusionHead has {n_params:,} params, unexpectedly large"
        self.n_params = n_params

    def forward(self, s_original: torch.Tensor, s_target_pool: torch.Tensor,
                s_frame: torch.Tensor, overlap_fraction: torch.Tensor) -> torch.Tensor:
        # all inputs [B] -> logit [B]
        feats = torch.stack([s_original, s_target_pool, s_frame, overlap_fraction], dim=-1)  # [B, 4]
        return self.mlp(feats).squeeze(-1)
