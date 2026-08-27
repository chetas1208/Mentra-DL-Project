"""P1.5 -- the six-way pipeline matrix.

    RAW                     mic straight to ASR (the "do nothing" control)
    RNNOISE                 machinery denoise only
    GEOWEAR_GATE            predicted GeoWearNet probs -> GeoWear Gate
    RNNOISE_GEOWEAR_GATE    denoise, then predicted gate
    ORACLE_GATE             ground-truth activity -> the SAME gate (P1.3)
    RNNOISE_ORACLE_GATE     denoise, then oracle gate

Every pipeline is evaluated on IDENTICAL windows with IDENTICAL metric code
and an IDENTICAL ASR backend, so differences between rows are attributable
to the processing and nothing else. That is the entire reason this is one
module rather than six scripts.

TWO DELIBERATE, DOCUMENTED CHOICES
----------------------------------
1. In `RNNOISE_GEOWEAR_GATE`, GeoWearNet's probabilities are computed on the
   **denoised** signal, because that is the actual product pipeline order
   (denoise -> detect -> route). GeoWearNet was trained on raw,
   amplitude-preserving audio, so this is a real domain shift and it may
   *hurt* the detector. That is a genuine finding about pipeline ordering,
   not a bug, and it is reported rather than engineered around by secretly
   feeding the detector clean audio. `detector_input` is recorded in every
   row so this is never ambiguous.

2. Oracle rows use ground-truth activity as probabilities but the SAME gate,
   SAME policy, SAME smoothing, SAME crossfade. The oracle-vs-predicted gap
   is therefore purely detector error, and the oracle's own residual error is
   purely the routing architecture's ceiling (i.e. overlap).
"""
from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .denoise import Denoiser
from .gate import GatePolicy, GeoWearGate, algorithmic_latency_ms, oracle_probabilities

PIPELINES = ("RAW", "RNNOISE", "GEOWEAR_GATE", "RNNOISE_GEOWEAR_GATE",
             "ORACLE_GATE", "RNNOISE_ORACLE_GATE")

SR = 16000


@dataclasses.dataclass
class EvalItem:
    """One evaluation window. The unit every metric is computed over."""
    item_id: str
    audio: np.ndarray                 # float32 mono 16k
    wearer_active: np.ndarray         # (T,) 0/1 ground truth, 10 ms frames
    env_active: np.ndarray
    self_text: str = ""
    other_text: str = ""
    wearer_commands: Tuple[str, ...] = ()
    bystander_commands: Tuple[str, ...] = ()
    scenario: str = "unspecified"
    source: str = "unspecified"       # "mmcsg_internal_val" | "stress_bench"
    recording_id: str = ""            # base MMCSG id, for anomaly bookkeeping
    clean_wearer: Optional[np.ndarray] = None   # only the stress bench has this

    @property
    def duration_s(self) -> float:
        return len(self.audio) / float(SR)


@dataclasses.dataclass
class ProcessedAudio:
    audio: np.ndarray
    pipeline: str
    gate_gain: Optional[np.ndarray]
    gate_states: Optional[np.ndarray]
    detector_input: str               # "raw" | "denoised" | "oracle"
    p_wearer: Optional[np.ndarray]
    p_env: Optional[np.ndarray]
    latency: Dict[str, float]
    timings: Dict[str, float]


class PipelineContext:
    """Shared, reusable heavy objects (model, denoiser) so a matrix run does
    not reload a checkpoint 6x per item."""

    def __init__(self, predictor=None, policy: GatePolicy = None,
                 denoiser: Optional[Denoiser] = None, sr: int = SR):
        self.predictor = predictor
        self.policy = policy or GatePolicy()
        self.denoiser = denoiser if denoiser is not None else Denoiser(sr=sr)
        self.sr = sr
        self._pred_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def probs(self, key: str, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GeoWearNet inference is the expensive CPU step; cache per
        (item, detector-input) so GEOWEAR_GATE and any policy sweep over the
        same item reuse one forward pass."""
        if key not in self._pred_cache:
            if self.predictor is None:
                n = len(audio) // int(round(self.sr * 0.01))
                self._pred_cache[key] = (np.zeros(n, np.float32), np.zeros(n, np.float32))
            else:
                self._pred_cache[key] = self.predictor.predict(audio)
        return self._pred_cache[key]


def run_pipeline(name: str, item: EvalItem, ctx: PipelineContext,
                 policy: Optional[GatePolicy] = None) -> ProcessedAudio:
    policy = policy or ctx.policy
    timings: Dict[str, float] = {}
    lat = {"latency_algorithmic_ms": 0.0}
    x = np.ascontiguousarray(item.audio, dtype=np.float32)
    denoised = False

    if name.startswith("RNNOISE"):
        # EvalItems are independent recordings/windows.  Preserve the live
        # Denoiser's recurrent state inside one transport session, but reset
        # here so one matrix item cannot affect another or a comparison row.
        ctx.denoiser.reset()
        t0 = time.perf_counter()
        d = ctx.denoiser.process(x)
        timings["denoise_s"] = time.perf_counter() - t0
        timings["denoise_rtf"] = d.rtf
        if d.available:
            x = d.audio
            denoised = True
            dl = ctx.denoiser.algorithmic_latency_ms()
            lat["latency_algorithmic_ms"] += float(dl.get("latency_algorithmic_ms") or 0.0)
            lat["denoise_latency_ms"] = dl.get("latency_algorithmic_ms")

    if name in ("RAW", "RNNOISE"):
        return ProcessedAudio(audio=x, pipeline=name, gate_gain=None, gate_states=None,
                              detector_input="none", p_wearer=None, p_env=None,
                              latency=lat, timings=timings)

    # --- gated pipelines -----------------------------------------------------
    if "ORACLE" in name:
        pw, pe = oracle_probabilities(item.wearer_active, item.env_active)
        det_in = "oracle"
    else:
        det_in = "denoised" if denoised else "raw"
        t0 = time.perf_counter()
        pw, pe = ctx.probs(f"{item.item_id}|{det_in}", x)
        timings["detect_s"] = time.perf_counter() - t0
        timings["detect_rtf"] = timings["detect_s"] / max(item.duration_s, 1e-9)
        g = algorithmic_latency_ms(policy)
        lat["latency_algorithmic_ms"] += g["latency_algorithmic_ms"]
        lat.update({k: v for k, v in g.items() if k != "latency_algorithmic_ms"})

    if "ORACLE" in name:
        g = algorithmic_latency_ms(policy, model_lookahead_ms=0.0)
        lat["latency_algorithmic_ms"] += g["latency_algorithmic_ms"]
        lat.update({k: v for k, v in g.items() if k != "latency_algorithmic_ms"})

    t0 = time.perf_counter()
    gate = GeoWearGate(policy, sr=ctx.sr)
    y, gain = gate.apply(x, pw, pe)
    timings["gate_s"] = time.perf_counter() - t0
    timings["gate_rtf"] = timings["gate_s"] / max(item.duration_s, 1e-9)

    return ProcessedAudio(audio=y, pipeline=name, gate_gain=gain,
                          gate_states=getattr(gate, "frame_states", None),
                          detector_input=det_in, p_wearer=np.asarray(pw), p_env=np.asarray(pe),
                          latency=lat, timings=timings)
