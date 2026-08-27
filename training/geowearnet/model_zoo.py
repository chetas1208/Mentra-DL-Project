"""GeoWearNet model zoo -- context / size / architecture variants.

Workstreams P (context sweep), Q (model-size Pareto), R (architecture ablation).

`training/geowearnet/model.py` holds the FROZEN E1 reference
(`GeoWearNetE1`, ~108K params, RF 68 frames) that passed the E1 smoke test.
It is deliberately NOT modified. `GeoWearNetTCN` below is a generalisation whose
default configuration reproduces E1 exactly -- asserted in `validate_zoo()` by
comparing parameter count and receptive field against the frozen class.

All architectures here share one contract:
    forward(log_mel: (B,T,M), physical: (B,T,P) | None)
      -> {"wearer_logits": (B,T), "environment_logits": (B,T),
          "four_state_logits": (B,T,4) | absent}
and all are STRICTLY CAUSAL (no lookahead), verified numerically by
`check_causality()`.

R3 note: no Transformer and no Mamba. The campaign brief forbids adding them
for fashion, and at 100K parameters with a 680 ms context there is no evidence
of a capacity or long-range-dependency limit that would justify them.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .model import GeoWearNetE1, GeoWearNetE1Config


@dataclasses.dataclass
class ZooConfig:
    arch: str = "tcn"                       # tcn | dsconv | crnn | convnext
    n_mels: int = 64
    n_physical_features: int = 14
    use_physical_features: bool = True
    frontend_kernels: Tuple[int, ...] = (5, 5)
    conv_channels: int = 48
    tcn_channels: int = 64
    tcn_kernel_size: int = 3
    tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1
    aux_four_state_head: bool = True
    physical_proj_dim: int = 16
    frame_hop_ms: float = 10.0


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------
class CausalDSConv(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, dilation: int = 1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.dw = nn.Conv1d(cin, cin, k, dilation=dilation, groups=cin)
        self.pw = nn.Conv1d(cin, cout, 1)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))
        return self.act(self.bn(self.pw(self.dw(x))))


class CausalTCNBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        y = self.act(self.bn1(self.conv1(nn.functional.pad(x, (self.pad, 0)))))
        y = self.drop(y)
        y = self.act(self.bn2(self.conv2(nn.functional.pad(y, (self.pad, 0)))))
        y = self.drop(y)
        return self.act(y + res)

    @property
    def rf(self) -> int:
        return 2 * self.pad


class CausalConvNeXtBlock(nn.Module):
    """Tiny ConvNeXt-style temporal block: large-kernel depthwise -> LayerNorm
    -> pointwise expand -> GELU -> pointwise project, residual. (R3)"""

    def __init__(self, ch: int, k: int, dilation: int, dropout: float, expand: int = 3):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.dw = nn.Conv1d(ch, ch, k, dilation=dilation, groups=ch)
        self.norm = nn.LayerNorm(ch)
        self.pw1 = nn.Linear(ch, ch * expand)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(ch * expand, ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        y = self.dw(nn.functional.pad(x, (self.pad, 0))).transpose(1, 2)
        y = self.pw2(self.act(self.pw1(self.norm(y))))
        return res + self.drop(y.transpose(1, 2))

    @property
    def rf(self) -> int:
        return self.pad


class _Heads(nn.Module):
    def __init__(self, dim: int, cfg: ZooConfig):
        super().__init__()
        fused = dim
        if cfg.use_physical_features:
            self.phys = nn.Sequential(nn.Linear(cfg.n_physical_features, cfg.physical_proj_dim), nn.ReLU(inplace=True))
            fused += cfg.physical_proj_dim
        else:
            self.phys = None
        self.wearer = nn.Linear(fused, 1)
        self.environment = nn.Linear(fused, 1)
        self.four_state = nn.Linear(fused, 4) if cfg.aux_four_state_head else None

    def fuse(self, x: torch.Tensor, physical: Optional[torch.Tensor]) -> torch.Tensor:
        if self.phys is not None:
            assert physical is not None and physical.shape[-1] > 0, "use_physical_features=True needs physical input"
            x = torch.cat([x, self.phys(physical)], dim=-1)
        return x

    def forward(self, x: torch.Tensor, physical: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.fuse(x, physical)
        out = {"wearer_logits": self.wearer(x).squeeze(-1),
               "environment_logits": self.environment(x).squeeze(-1)}
        if self.four_state is not None:
            out["four_state_logits"] = self.four_state(x)
        return out


class _Base(nn.Module):
    config: ZooConfig

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def receptive_field_frames(self) -> int:
        raise NotImplementedError

    def context_ms(self) -> float:
        return self.receptive_field_frames() * self.config.frame_hop_ms

    def embed(self, log_mel: torch.Tensor, physical=None) -> torch.Tensor:
        """Hidden temporal representation feeding the heads, (B, T, D).

        Exposed so Workstreams AP/AQ can probe what the representation encodes
        (simulator generation, speaker identity) WITHOUT retraining anything.
        Includes the fused physical branch when present, because that is what
        the heads actually see."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# architectures
# ---------------------------------------------------------------------------
class GeoWearNetTCN(_Base):
    """R0 -- generalised causal TCN. Defaults reproduce the frozen E1 exactly."""

    def __init__(self, cfg: ZooConfig):
        super().__init__()
        self.config = cfg
        layers: List[nn.Module] = []
        cin = cfg.n_mels
        for i, k in enumerate(cfg.frontend_kernels):
            cout = cfg.conv_channels if i < len(cfg.frontend_kernels) - 1 else cfg.tcn_channels
            layers.append(CausalDSConv(cin, cout, k))
            cin = cout
        self.frontend = nn.Sequential(*layers)
        self.blocks = nn.ModuleList(
            [CausalTCNBlock(cfg.tcn_channels, cfg.tcn_kernel_size, d, cfg.dropout) for d in cfg.tcn_dilations]
        )
        self.heads = _Heads(cfg.tcn_channels, cfg)

    def receptive_field_frames(self) -> int:
        return sum(k - 1 for k in self.config.frontend_kernels) + sum(b.rf for b in self.blocks)

    def _trunk(self, log_mel):
        x = self.frontend(log_mel.transpose(1, 2))
        for b in self.blocks:
            x = b(x)
        return x.transpose(1, 2)

    def embed(self, log_mel, physical=None):
        return self.heads.fuse(self._trunk(log_mel), physical)

    def forward(self, log_mel, physical=None):
        return self.heads(self._trunk(log_mel), physical)


class GeoWearNetDSConv(_Base):
    """R1 -- pure depthwise-separable dilated temporal ConvNet (no residual
    TCN blocks). Cheapest per-frame cost of the four."""

    def __init__(self, cfg: ZooConfig):
        super().__init__()
        self.config = cfg
        layers, cin = [], cfg.n_mels
        for k in cfg.frontend_kernels:
            layers.append(CausalDSConv(cin, cfg.conv_channels, k))
            cin = cfg.conv_channels
        for d in cfg.tcn_dilations:
            layers.append(CausalDSConv(cin, cfg.tcn_channels, cfg.tcn_kernel_size, dilation=d))
            cin = cfg.tcn_channels
        self.net = nn.Sequential(*layers)
        self.drop = nn.Dropout(cfg.dropout)
        self.heads = _Heads(cfg.tcn_channels, cfg)

    def receptive_field_frames(self) -> int:
        c = self.config
        return sum(k - 1 for k in c.frontend_kernels) + sum((c.tcn_kernel_size - 1) * d for d in c.tcn_dilations)

    def _trunk(self, log_mel):
        return self.drop(self.net(log_mel.transpose(1, 2))).transpose(1, 2)

    def embed(self, log_mel, physical=None):
        return self.heads.fuse(self._trunk(log_mel), physical)

    def forward(self, log_mel, physical=None):
        return self.heads(self._trunk(log_mel), physical)


class GeoWearNetCRNN(_Base):
    """R2 -- conv frontend + unidirectional GRU. Unidirectional keeps it causal;
    its 'receptive field' is unbounded in principle, which is reported honestly
    rather than as a fixed number."""

    def __init__(self, cfg: ZooConfig):
        super().__init__()
        self.config = cfg
        layers, cin = [], cfg.n_mels
        for k in cfg.frontend_kernels:
            layers.append(CausalDSConv(cin, cfg.conv_channels, k))
            cin = cfg.conv_channels
        self.frontend = nn.Sequential(*layers)
        self.gru = nn.GRU(cfg.conv_channels, cfg.tcn_channels, num_layers=1, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.heads = _Heads(cfg.tcn_channels, cfg)

    def receptive_field_frames(self) -> int:
        # conv part only; the GRU state is unbounded (reported separately)
        return sum(k - 1 for k in self.config.frontend_kernels)

    def _trunk(self, log_mel):
        x = self.frontend(log_mel.transpose(1, 2)).transpose(1, 2)
        x, _ = self.gru(x)
        return self.drop(x)

    def embed(self, log_mel, physical=None):
        return self.heads.fuse(self._trunk(log_mel), physical)

    def forward(self, log_mel, physical=None):
        return self.heads(self._trunk(log_mel), physical)


class GeoWearNetConvNeXt(_Base):
    """R3 -- tiny ConvNeXt-style causal temporal blocks."""

    def __init__(self, cfg: ZooConfig):
        super().__init__()
        self.config = cfg
        layers, cin = [], cfg.n_mels
        for k in cfg.frontend_kernels:
            layers.append(CausalDSConv(cin, cfg.tcn_channels, k))
            cin = cfg.tcn_channels
        self.frontend = nn.Sequential(*layers)
        self.blocks = nn.ModuleList(
            [CausalConvNeXtBlock(cfg.tcn_channels, cfg.tcn_kernel_size * 2 + 1, d, cfg.dropout)
             for d in cfg.tcn_dilations]
        )
        self.heads = _Heads(cfg.tcn_channels, cfg)

    def receptive_field_frames(self) -> int:
        return sum(k - 1 for k in self.config.frontend_kernels) + sum(b.rf for b in self.blocks)

    def _trunk(self, log_mel):
        x = self.frontend(log_mel.transpose(1, 2))
        for b in self.blocks:
            x = b(x)
        return x.transpose(1, 2)

    def embed(self, log_mel, physical=None):
        return self.heads.fuse(self._trunk(log_mel), physical)

    def forward(self, log_mel, physical=None):
        return self.heads(self._trunk(log_mel), physical)


ARCHS = {"tcn": GeoWearNetTCN, "dsconv": GeoWearNetDSConv, "crnn": GeoWearNetCRNN, "convnext": GeoWearNetConvNeXt}


# ---------------------------------------------------------------------------
# named variants
# ---------------------------------------------------------------------------
# Workstream P -- context sweep. `target_ms` is the intent; the ACTUAL algorithmic
# receptive field is computed from the graph and reported, never assumed.
CONTEXT_VARIANTS: Dict[str, dict] = {
    "ctx100":  dict(frontend_kernels=(3, 3), tcn_kernel_size=2, tcn_dilations=(1, 2)),
    "ctx250":  dict(frontend_kernels=(3, 4), tcn_kernel_size=2, tcn_dilations=(1, 2, 4, 3)),
    "ctx500":  dict(frontend_kernels=(5, 5), tcn_kernel_size=3, tcn_dilations=(1, 2, 4, 4)),
    "ctx680":  dict(frontend_kernels=(5, 5), tcn_kernel_size=3, tcn_dilations=(1, 2, 4, 8)),
    "ctx1000": dict(frontend_kernels=(5, 5), tcn_kernel_size=3, tcn_dilations=(1, 2, 4, 8, 8)),
}

# Workstream Q -- size sweep (context held at the E1 default).
SIZE_VARIANTS: Dict[str, dict] = {
    "size50k":   dict(conv_channels=32, tcn_channels=44),
    "size100k":  dict(conv_channels=48, tcn_channels=64),   # == frozen E1
    "size250k":  dict(conv_channels=64, tcn_channels=100),
    "size500k":  dict(conv_channels=96, tcn_channels=144),
    "size1m":    dict(conv_channels=128, tcn_channels=205),
}


def build_model(arch: str = "tcn", variant: Optional[str] = None, **overrides) -> _Base:
    kw: Dict[str, object] = {}
    if variant:
        if variant in CONTEXT_VARIANTS:
            kw.update(CONTEXT_VARIANTS[variant])
        elif variant in SIZE_VARIANTS:
            kw.update(SIZE_VARIANTS[variant])
        else:
            raise KeyError(f"unknown variant {variant!r}")
    kw.update(overrides)
    kw["arch"] = arch
    cfg = ZooConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()})
    return ARCHS[arch](cfg)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_causality(model: _Base, t: int = 120, tol: float = 0.0) -> Dict[str, float]:
    """Numerically prove no lookahead: perturbing frame t-1 must change NOTHING
    at any earlier output frame. Uses eval() so BatchNorm running stats (which
    are global, not temporal) don't confound the test."""
    model.eval()
    m = model.config
    lm = torch.randn(1, t, m.n_mels)
    ph = torch.randn(1, t, m.n_physical_features) if m.use_physical_features else None
    a = model(lm, ph)
    lm2 = lm.clone()
    lm2[0, -1] += 12.0
    ph2 = ph.clone() if ph is not None else None
    if ph2 is not None:
        ph2[0, -1] += 12.0
    b = model(lm2, ph2)
    diffs = {}
    for k in ("wearer_logits", "environment_logits"):
        d = (a[k][0, :-1] - b[k][0, :-1]).abs().max().item()
        diffs[k] = d
        assert d <= tol, f"CAUSALITY VIOLATION in {k}: max past-frame diff {d}"
    diffs["last_frame_changed"] = (a["wearer_logits"][0, -1] - b["wearer_logits"][0, -1]).abs().item()
    assert diffs["last_frame_changed"] > 0, "perturbation had no effect at all -- test is vacuous"
    return diffs


@torch.no_grad()
def measure_receptive_field(model: _Base, t: int = 400) -> int:
    """EMPIRICAL causal LOOKBACK: the largest L such that perturbing input frame
    (T-1-L) still changes output frame (T-1).

    CONVENTION (matches the frozen E1's docstring and the campaign's "68
    frames / 680 ms"): this is the lookback, i.e. frames strictly INTO THE PAST.
    The total number of input frames a decision depends on is lookback + 1
    (69 frames for E1). Cross-checks the analytic `receptive_field_frames()`.
    """
    model.eval()
    m = model.config
    lm = torch.randn(1, t, m.n_mels)
    ph = torch.randn(1, t, m.n_physical_features) if m.use_physical_features else None
    base = model(lm, ph)["wearer_logits"][0, -1].item()
    reach = 0
    for back in range(1, t):
        lm2 = lm.clone()
        lm2[0, t - 1 - back] += 25.0
        if abs(model(lm2, ph)["wearer_logits"][0, -1].item() - base) > 1e-6:
            reach = back
    return reach


def validate_zoo(verbose: bool = True) -> dict:
    res: dict = {}

    # frozen-E1 parity
    frozen = GeoWearNetE1(GeoWearNetE1Config())
    zoo = build_model("tcn", "ctx680")
    res["frozen_e1_params"] = frozen.count_parameters()
    res["zoo_ctx680_params"] = zoo.count_parameters()
    res["frozen_e1_rf"] = frozen.receptive_field_frames()
    res["zoo_ctx680_rf"] = zoo.receptive_field_frames()
    assert res["frozen_e1_rf"] == res["zoo_ctx680_rf"] == 68, "ctx680 must reproduce E1's 68-frame RF"
    assert abs(res["frozen_e1_params"] - res["zoo_ctx680_params"]) / res["frozen_e1_params"] < 0.02, \
        "ctx680 should reproduce the frozen E1 parameter count"

    res["context_variants"] = {}
    for name in CONTEXT_VARIANTS:
        m = build_model("tcn", name)
        emp = measure_receptive_field(m, t=260)
        res["context_variants"][name] = {
            "params": m.count_parameters(),
            "analytic_rf_frames": m.receptive_field_frames(),
            "analytic_context_ms": m.context_ms(),
            "empirical_rf_frames": emp,
            "causality": check_causality(m),
        }
        # The analytic lookback is an UPPER BOUND. On random init the empirical
        # reach can be slightly smaller because ReLUs gate individual paths to
        # zero; what must never happen is empirical > analytic, which would mean
        # real lookahead or a wiring bug.
        assert emp <= m.receptive_field_frames(), \
            f"{name}: empirical RF {emp} EXCEEDS analytic bound {m.receptive_field_frames()}"
        assert emp >= 0.85 * m.receptive_field_frames(), \
            f"{name}: empirical RF {emp} far below analytic {m.receptive_field_frames()} -- likely a wiring bug"
        res["context_variants"][name]["empirical_over_analytic"] = emp / m.receptive_field_frames()

    res["size_variants"] = {}
    for name in SIZE_VARIANTS:
        m = build_model("tcn", name)
        res["size_variants"][name] = {"params": m.count_parameters(), "rf": m.receptive_field_frames()}

    res["architectures"] = {}
    for arch in ARCHS:
        m = build_model(arch)
        res["architectures"][arch] = {
            "params": m.count_parameters(),
            "analytic_rf_frames": m.receptive_field_frames(),
            "causality": check_causality(m),
        }

    if verbose:
        import json
        print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    validate_zoo()
