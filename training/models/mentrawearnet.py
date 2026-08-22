"""MentraWearNet-v1 (sprint spec sections 2, 6-25, "GO" phase).

Architecture:

    ENROLLMENT (once, cached at deployment)
    reference audio -> SpeakerNetBackbone.encode_speaker() -> e_w [B,256]

    LIVE (every rolling window)
    waveform -> SpeakerNetBackbone.encode_frames() -> [B,1500,T]
             -> 1x1 projection -> [B,D,T]
             -> FiLM(e_w) + explicit similarity branch
             -> causal depthwise-separable TCN
             -> wearer_head / environment_head (independent sigmoid logits)

Two independent binary outputs, not a 4-way softmax, because wearer and
environment activity are not mutually exclusive (section 15/19 -- SILENCE/
WEARER/ENVIRONMENT/OVERLAP is derived from the pair (P_wearer, P_environment)
outside this module, not predicted as one of 4 classes).

Hard parameter budget: <10,000,000 for the DEPLOYMENT graph (backbone minus
the 7205-way training classifier, plus everything below). See
count_params.py for the enforced assertion.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from training.models.speakernet_backbone import SpeakerNetBackbone


class FiLM(nn.Module):
    """gamma/beta close to identity at init (section 10) so the model starts
    close to the unconditioned backbone rather than destroying pretrained
    features on step 0."""

    def __init__(self, embedding_dim: int, feature_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(embedding_dim, feature_dim)
        self.to_beta = nn.Linear(embedding_dim, feature_dim)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, e_w: torch.Tensor) -> torch.Tensor:
        # h: [B, D, T], e_w: [B, E]
        gamma = self.to_gamma(e_w).unsqueeze(-1)  # [B, D, 1]
        beta = self.to_beta(e_w).unsqueeze(-1)
        return (1.0 + gamma) * h + beta


class CausalDepthwiseSeparableBlock(nn.Module):
    """One TCN block: causal (left-pad only, never sees future samples).
    depthwise Conv1d -> pointwise Conv1d -> GroupNorm -> SiLU -> dropout ->
    residual. GroupNorm normalizes over channels only (not time), so it
    doesn't leak future-frame statistics into current predictions -- unlike
    e.g. a naive LayerNorm-over-time or BatchNorm computed across the batch's
    full temporal extent during eval mode with running stats disabled."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation  # causal: pad only on the left
        self.depthwise = nn.Conv1d(channels, channels, kernel_size,
                                    dilation=dilation, groups=channels, bias=False)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x
        h = nn.functional.pad(x, (self.left_pad, 0))  # left-pad only -- causal
        h = self.depthwise(h)
        h = self.pointwise(h)
        h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        return residual + h


class MentraWearNet(nn.Module):
    def __init__(
        self,
        projection_dim: int = 192,
        similarity_dim: int = 128,
        tcn_blocks: int = 5,
        tcn_kernel_size: int = 3,
        tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(tcn_dilations) == tcn_blocks, "one dilation per TCN block"

        self.backbone = SpeakerNetBackbone()  # the ONE shared backbone instance (section 6)
        D = projection_dim

        # Section 8: project 1500 -> D immediately, expensive to carry 1500 channels through the TCN.
        self.frame_projection = nn.Conv1d(self.backbone.frame_channels, D, kernel_size=1)

        # Section 9: project the 256-D wearer embedding into the live feature space.
        self.enrollment_projection = nn.Linear(self.backbone.embedding_dim, D)

        # Section 10: FiLM conditioning.
        self.film = FiLM(embedding_dim=D, feature_dim=D)

        # Section 11: explicit cosine-similarity branch -- don't make the TCN
        # rediscover what plain verification already does well.
        self.frame_speaker_projection = nn.Conv1d(D, similarity_dim, kernel_size=1)
        self.enrollment_similarity_projection = nn.Linear(D, similarity_dim)
        self.similarity_fusion = nn.Conv1d(D + 1, D, kernel_size=1)  # concat(h_conditioned, s_t) -> D

        # Section 12: causal depthwise-separable TCN.
        self.tcn = nn.Sequential(*[
            CausalDepthwiseSeparableBlock(D, tcn_kernel_size, dilation, dropout)
            for dilation in tcn_dilations
        ])

        # Section 14: independent logit heads, no sigmoid inside the model (loss takes logits).
        self.wearer_head = nn.Conv1d(D, 1, kernel_size=1)
        self.environment_head = nn.Conv1d(D, 1, kernel_size=1)

        self.projection_dim = D

    def encode_enrollment(self, enrollment_waveform: torch.Tensor, enrollment_lengths: torch.Tensor) -> torch.Tensor:
        """Section 22/23: call this ONCE at enrollment time, cache the
        result, never re-run enrollment audio through the graph per
        live-audio call."""
        return self.backbone.encode_speaker(enrollment_waveform, enrollment_lengths)  # [B, 256]

    def process_with_embedding(self, live_waveform: torch.Tensor, live_lengths: torch.Tensor,
                                wearer_embedding: torch.Tensor) -> dict:
        """Deployment-path forward: live audio + a precomputed 256-D wearer
        embedding (no enrollment audio in this call -- section 23)."""
        frame_features, frame_lengths = self.backbone.encode_frames(live_waveform, live_lengths)  # [B,1500,T]
        h = self.frame_projection(frame_features)  # [B, D, T]

        e_proj = self.enrollment_projection(wearer_embedding)  # [B, D]
        h_conditioned = self.film(h, e_proj)  # [B, D, T]

        frame_speaker = self.frame_speaker_projection(h_conditioned)  # [B, E, T]
        enrollment_sim_space = self.enrollment_similarity_projection(e_proj)  # [B, E]
        enrollment_sim_space = nn.functional.normalize(enrollment_sim_space, p=2, dim=-1)
        frame_speaker_norm = nn.functional.normalize(frame_speaker, p=2, dim=1)  # normalize over E
        similarity = torch.einsum("bet,be->bt", frame_speaker_norm, enrollment_sim_space)  # [B, T]
        similarity = similarity.unsqueeze(1)  # [B, 1, T]

        fused = self.similarity_fusion(torch.cat([h_conditioned, similarity], dim=1))  # [B, D, T]
        temporal = self.tcn(fused)  # [B, D, T]

        wearer_logits = self.wearer_head(temporal).squeeze(1)  # [B, T]
        environment_logits = self.environment_head(temporal).squeeze(1)  # [B, T]

        return {
            "wearer_logits": wearer_logits,
            "environment_logits": environment_logits,
            "frame_lengths": frame_lengths,
            "similarity": similarity.squeeze(1),
        }

    def forward(self, mixture_waveform: torch.Tensor, mixture_lengths: torch.Tensor,
                enrollment_waveform: torch.Tensor, enrollment_lengths: torch.Tensor) -> dict:
        """Training-path forward (section 21): computes enrollment embedding
        every call (needed since enrollment speaker varies per training
        example), unlike the deployment path above."""
        wearer_embedding = self.encode_enrollment(enrollment_waveform, enrollment_lengths)
        return self.process_with_embedding(mixture_waveform, mixture_lengths, wearer_embedding)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_upper_backbone(self, num_upper_blocks: int = 1):
        """Unfreezes the last `num_upper_blocks` Jasper blocks of the
        ConvASREncoder (section 27, training stage 2), leaving earlier
        blocks + preprocessor frozen."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        encoder_blocks = list(self.backbone.encoder.encoder)  # nn.Sequential of JasperBlocks
        for block in encoder_blocks[-num_upper_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.backbone.pooling.parameters():
            p.requires_grad = True
        for p in self.backbone.embedding_projection.parameters():
            p.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def deployment_param_count(self) -> int:
        """Total params actually needed at inference time: the full module
        tree here (backbone already excludes the 7205-way classifier since
        SpeakerNetBackbone never loads model.decoder.final -- see
        speakernet_backbone.py)."""
        return sum(p.numel() for p in self.parameters())
