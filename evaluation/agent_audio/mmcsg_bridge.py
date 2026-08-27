"""Read-only bridge from the MMCSG corpus to the agent-audio harness.

Provides, per recording:
  * mono 16 kHz audio on the deployment channel (channel 2 -- the channel the
    G2 campaign measured as best, not a guess);
  * ground-truth per-frame SELF/OTHER activity from the official RTTM,
    via `training/geowearnet/mmcsg/labels.py` (reused, not reimplemented);
  * the SELF reference transcript and the OTHER reference transcript,
    reconstructed from the official word-level TSV using the SAME 0/1
    speaker convention (`labels.SELF_LABEL` / `labels.OTHER_LABEL`);
  * GeoWearNet frame probabilities from a checkpoint, via
    `training/geowearnet/mmcsg_transfer.py`'s existing inference path.

HARD SAFETY RULES ENFORCED HERE
  * `assert_not_dev()` refuses the official MMCSG **dev** split outright.
    P1's brief reserves dev for the single guarded final-selection
    evaluation; this package must never spend a dev look. `eval` is
    likewise refused by default (it is the second official held-out set).
  * every file is opened read-only; nothing under the corpus root is
    written, moved, or normalised in place.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from training.geowearnet.mmcsg import audio_io as A
from training.geowearnet.mmcsg.config import REPO_ROOT, resolve_root
from training.geowearnet.mmcsg.labels import (OTHER_LABEL, SELF_LABEL, FRAME_HOP_S,
                                              frame_labels_from_rttm, read_rttm,
                                              read_words, state_from_labels)
from training.geowearnet.mmcsg.splits import load_split

DEPLOYMENT_CHANNEL = 2      # measured winner in the G2 channel study
SR = 16000

FORBIDDEN_SPLITS = {"dev", "eval"}


class DevSplitAccessError(RuntimeError):
    pass


def assert_not_dev(split: str) -> None:
    """The agent-audio lane is explicitly forbidden from touching the official
    held-out splits. Only `final_selection.py` (guarded by `dev_guard.py`)
    may evaluate dev, exactly once, after a checkpoint is frozen."""
    if str(split).lower() in FORBIDDEN_SPLITS:
        raise DevSplitAccessError(
            f"split {split!r} is an official MMCSG held-out split and is OFF LIMITS to "
            "evaluation.agent_audio. Use the GeoWearNet-internal 'val' split "
            "(mmcsg_geowearnet_val.json). Final dev evaluation belongs to "
            "training/geowearnet/mmcsg/final_selection.py under dev_guard.")


@dataclasses.dataclass
class Recording:
    recording_id: str
    source_split: str
    audio: np.ndarray            # float32 mono 16k, absolute amplitude preserved
    wearer_active: np.ndarray    # (T,) float32 0/1, 10 ms frames
    env_active: np.ndarray
    self_text: str
    other_text: str
    self_words: List[Tuple[float, float, str]]
    other_words: List[Tuple[float, float, str]]

    @property
    def n_frames(self) -> int:
        return int(min(len(self.wearer_active), len(self.env_active)))

    @property
    def duration_s(self) -> float:
        return len(self.audio) / float(SR)

    def state(self) -> np.ndarray:
        return state_from_labels(self.wearer_active, self.env_active)


def internal_split(name: str = "val") -> List[Dict[str, object]]:
    """The GeoWearNet-internal wearer-disjoint split (derived from official
    TRAIN only). Safe to look at repeatedly."""
    assert_not_dev(name)
    return load_split(name)


def load_recording(rec: Dict[str, object], channel: int = DEPLOYMENT_CHANNEL,
                   root: Optional[Path] = None) -> Recording:
    root = root or resolve_root()
    split = str(rec["source_split"])
    rid = str(rec["recording_id"])
    assert_not_dev(split)

    sl = A.read_full(root, split, rid, channel=channel)
    audio = sl.audio
    n_frames = int(len(audio) // int(round(SR * FRAME_HOP_S)))

    segs = read_rttm(root / "rttm" / split / f"{rid}.rttm")
    w, e = frame_labels_from_rttm(segs, n_frames)

    self_words, other_words = [], []
    tsv = root / "transcriptions" / split / f"{rid}.tsv"
    if tsv.exists():
        for st, en, word, spk in read_words(tsv):
            (self_words if spk == SELF_LABEL else
             other_words if spk == OTHER_LABEL else []).append((st, en, word))

    return Recording(
        recording_id=rid, source_split=split, audio=audio,
        wearer_active=w, env_active=e,
        self_text=" ".join(t[2] for t in self_words),
        other_text=" ".join(t[2] for t in other_words),
        self_words=self_words, other_words=other_words,
    )


def slice_recording(r: Recording, start_s: float, dur_s: float) -> Recording:
    """Cut a window out of a recording, keeping labels and per-speaker word
    references consistent. Used to keep ASR windows short enough that a
    streaming transducer stays accurate and the harness stays cheap."""
    hop = int(round(SR * FRAME_HOP_S))
    a, b = int(start_s * SR), int((start_s + dur_s) * SR)
    fa, fb = int(start_s / FRAME_HOP_S), int((start_s + dur_s) / FRAME_HOP_S)

    def _w(words):
        return [(s - start_s, e - start_s, t) for s, e, t in words
                if e > start_s and s < start_s + dur_s]

    sw, ow = _w(r.self_words), _w(r.other_words)
    return Recording(
        recording_id=f"{r.recording_id}@{start_s:.1f}+{dur_s:.1f}",
        source_split=r.source_split,
        audio=r.audio[a:b],
        wearer_active=r.wearer_active[fa:fb], env_active=r.env_active[fa:fb],
        self_text=" ".join(t[2] for t in sw), other_text=" ".join(t[2] for t in ow),
        self_words=sw, other_words=ow,
    )


# ---------------------------------------------------------------------------
# GeoWearNet predictions (read-only checkpoint use)
# ---------------------------------------------------------------------------
class GeoWearNetPredictor:
    """Loads a GeoWearNet checkpoint and produces per-frame P(wearer)/P(env).

    Reuses `mmcsg_transfer.load_checkpoint` / `run_recording` so a number
    produced here cannot be blamed on a reimplemented feature pipeline. The
    checkpoint file is opened read-only and never written back.

    Normalisation stats: REAL-trained checkpoints (from `train_real.py`) use
    the real MMCSG stats cache (`training/geowearnet/mmcsg/cache/
    norm_stats_real_ch*.npz`); SIM-trained G1 checkpoints use the simulated
    cache. We pick based on the checkpoint's own `domain`/`config`, never by
    guessing.
    """

    def __init__(self, checkpoint: Path, device: str = "cpu"):
        import torch
        from training.geowearnet.mmcsg_transfer import load_checkpoint
        self.checkpoint = Path(checkpoint)
        self.model, self.cfg, self.ck = load_checkpoint(self.checkpoint)
        self.model.eval()
        self.train_cfg = self.ck.get("config", {}) or {}
        self.mode = self.train_cfg.get("normalization", "global")
        self.stats = self._resolve_stats()
        self.is_real_trained = bool(self.ck.get("domain")) or "minutes_budget" in self.train_cfg

    def _resolve_stats(self):
        from training.geowearnet.mmcsg.norm_stats import get_or_compute, path_for
        from training.geowearnet.mmcsg_transfer import load_norm_stats
        if self.mode != "global":
            return None
        tc = self.train_cfg
        # real-trained checkpoints carry MMCSG-specific training knobs
        if "minutes_budget" in tc or self.ck.get("domain"):
            p = path_for(tc.get("channel"), bool(tc.get("drop_amplitude_features", False)),
                         bool(tc.get("level_normalize", False)))
            if p.exists():
                with np.load(p) as z:
                    return {k: z[k] for k in z.files}
            return get_or_compute(tc.get("channel"),
                                  bool(tc.get("drop_amplitude_features", False)),
                                  level_normalize=bool(tc.get("level_normalize", False)))
        return load_norm_stats(self.ck)

    def predict(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """-> (P(wearer), P(environment)) per 10 ms frame, sigmoid-calibrated
        only in the sense of being the model's own sigmoid output; no
        post-hoc temperature is applied (and none is silently assumed)."""
        from training.geowearnet.mmcsg_transfer import run_recording
        wl, el, t = run_recording(self.model, self.cfg, self.mode, self.stats, audio)
        return _sigmoid(wl), _sigmoid(el)

    def describe(self) -> Dict[str, object]:
        return {
            "checkpoint": str(self.checkpoint),
            "arch": self.cfg.arch,
            "params": self.ck.get("params"),
            "context_ms": self.ck.get("context_ms"),
            "train_run": self.train_cfg.get("name"),
            "normalization": self.mode,
            "has_norm_stats": self.stats is not None,
            "best_val": (self.ck.get("best") or {}).get("metrics"),
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))).astype(np.float32)


def find_checkpoint(pattern: str = "g2_sim_ft_60m") -> Optional[Path]:
    """Pick a completed real-trained checkpoint by run-name prefix, newest
    first. Deliberately prefers a run that finished BEFORE the live ablation
    campaign started, so this lane never reads a checkpoint a training job is
    concurrently writing."""
    runs = REPO_ROOT / "training/geowearnet/mmcsg/runs"
    cands = sorted((p for p in runs.glob(f"{pattern}*") if (p / "checkpoints/best.pt").exists()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return (cands[0] / "checkpoints/best.pt") if cands else None
