"""GeoWearNet-E1 architecture (Phase 11-14).

Small, causal (bounded-lookahead), CPU-deployable, enrollment-free
wearer/environment activity detector. Target 0.3M-1.5M params (hard cap
<10M per this project's existing constraint, but E1 should target far
less per Phase 11).

Pipeline (Phase 13): log-mel -> small depthwise-separable Conv frontend ->
small causal TCN -> shared temporal representation (optionally fused with a
physical-scalar projection, Phase 12 Path B) -> two INDEPENDENT sigmoid
heads: P(wearer_active), P(environment_active) (Phase 14 -- independent
heads, not a forced-mutually-exclusive softmax, so overlap/silence fall out
of the 2x2 naturally). An optional small auxiliary 4-state head may also be
attached (Phase 14).

No Transformer/Mamba/raw-waveform network/pretrained speaker embedding/
FiLM-from-identity/enrollment vector anywhere in this module, per Phase 13.
This is intentionally NOT reusing MentraWearNet's architecture code
(`training/models/mentrawearnet.py`) -- new, isolated module (Phase 44).
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn


@dataclasses.dataclass
class GeoWearNetE1Config:
    n_mels: int = 64
    n_physical_features: int = 14  # len(features.FEATURE_NAMES), Path B scalars
    use_physical_features: bool = True
    conv_channels: int = 48
    tcn_channels: int = 64
    tcn_kernel_size: int = 3
    tcn_num_layers: int = 4          # dilations 1,2,4,8 -> receptive field growth
    tcn_dropout: float = 0.1
    aux_four_state_head: bool = True
    frame_hop_ms: float = 10.0


class CausalDepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1D conv, causally padded (left-pad only, no
    lookahead beyond the kernel's own dilation*[K-1] receptive field into
    the past)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size, dilation=dilation, groups=in_ch, padding=0
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T). Left-pad only -> causal.
        x = nn.functional.pad(x, (self.pad, 0))
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class CausalTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=0)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=0)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = nn.functional.pad(x, (self.pad, 0))
        y = self.act(self.bn1(self.conv1(y)))
        y = self.dropout(y)
        y = nn.functional.pad(y, (self.pad, 0))
        y = self.act(self.bn2(self.conv2(y)))
        y = self.dropout(y)
        return self.act(y + residual)

    def receptive_field_frames(self) -> int:
        return 2 * self.pad


class GeoWearNetE1(nn.Module):
    def __init__(self, config: Optional[GeoWearNetE1Config] = None):
        super().__init__()
        self.config = config or GeoWearNetE1Config()
        c = self.config

        self.frontend = nn.Sequential(
            CausalDepthwiseSeparableConv1d(c.n_mels, c.conv_channels, kernel_size=5, dilation=1),
            CausalDepthwiseSeparableConv1d(c.conv_channels, c.tcn_channels, kernel_size=5, dilation=1),
        )

        dilations = [2 ** i for i in range(c.tcn_num_layers)]
        self.tcn_blocks = nn.ModuleList(
            [CausalTCNBlock(c.tcn_channels, c.tcn_kernel_size, d, c.tcn_dropout) for d in dilations]
        )

        fused_dim = c.tcn_channels
        if c.use_physical_features:
            self.physical_proj = nn.Sequential(
                nn.Linear(c.n_physical_features, 16),
                nn.ReLU(inplace=True),
            )
            fused_dim += 16
        else:
            self.physical_proj = None

        self.wearer_head = nn.Linear(fused_dim, 1)
        self.environment_head = nn.Linear(fused_dim, 1)
        self.four_state_head = nn.Linear(fused_dim, 4) if c.aux_four_state_head else None

    def receptive_field_frames(self) -> int:
        """Total causal lookback in frames (10ms hop by default), i.e. how
        far into the past the model's decision at frame t can see. Purely
        causal -- 0 lookahead into the future."""
        rf = 5 - 1 + 5 - 1  # frontend two conv layers, kernel 5 each, dilation 1
        for block in self.tcn_blocks:
            rf += block.receptive_field_frames()
        return rf

    def forward(self, log_mel: torch.Tensor, physical_features: Optional[torch.Tensor] = None):
        """
        log_mel: (B, T, n_mels) -- calibrated log-mel, NOT per-utterance
            normalized (Phase 12: fixed physical scaling, retain calibrated
            reference to PCM magnitude upstream in the feature pipeline).
        physical_features: (B, T, n_physical_features) optional, Path B
            scalars, normalized only via TRAINING-SET global statistics
            upstream (this module does not normalize them itself).
        Returns dict with wearer_logits, environment_logits (both (B,T)),
        and four_state_logits (B,T,4) if enabled.
        """
        x = log_mel.transpose(1, 2)  # (B, n_mels, T)
        x = self.frontend(x)
        for block in self.tcn_blocks:
            x = block(x)
        x = x.transpose(1, 2)  # (B, T, tcn_channels)

        if self.physical_proj is not None:
            assert physical_features is not None, "use_physical_features=True requires physical_features input"
            p = self.physical_proj(physical_features)
            fused = torch.cat([x, p], dim=-1)
        else:
            fused = x

        wearer_logits = self.wearer_head(fused).squeeze(-1)
        environment_logits = self.environment_head(fused).squeeze(-1)
        out = {"wearer_logits": wearer_logits, "environment_logits": environment_logits}
        if self.four_state_head is not None:
            out["four_state_logits"] = self.four_state_head(fused)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
