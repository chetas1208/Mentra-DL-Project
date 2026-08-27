"""GeoWearNet torch data pipeline (Workstreams D, G, J, O).

An ONLINE simulator-backed dataset: scenes are synthesised in the DataLoader
workers rather than pre-rendered to disk (Workstream BG -- no petabytes of
waveforms). Determinism comes from seeding each item's RNG from
(base_seed, epoch_free_index), so:

  * the validation and test sets are byte-for-byte reproducible from a seed
    alone -- a "deterministic validation manifest" without storing audio;
  * training never repeats a scene, because the index space is huge.

NORMALISATION (Workstream O) is a first-class switch:
  O0 `none`     -- raw calibrated log-mel + raw physical scalars
  O1 `global`   -- one fixed mean/std from the training distribution
  O2 `cmvn`     -- causal per-window running mean/var normalisation
O2 is expected to destroy absolute-level cues; that is the point of measuring it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import manifests, scenes
from .acoustics import log_mel
from .features import FEATURE_NAMES, extract_features

CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Physical scalars that encode ABSOLUTE amplitude. Workstream N3 drops exactly
# these to test "combined minus absolute-amplitude scalars".
ABSOLUTE_AMPLITUDE_FEATURES = ["log_rms", "peak", "clipping_fraction"]


@dataclasses.dataclass
class DataConfig:
    split: str = "train"
    generation: str = "S1"                    # S0 | S1 | S2 | mixed
    condition: str = "train_mix"              # a CONDITIONS key, or "train_mix"
    duration_s: float = 8.0
    n_mels: int = 64
    normalization: str = "global"             # none | global | cmvn
    use_physical: bool = True
    drop_amplitude_features: bool = False     # Workstream N3
    virtual_size: int = 100_000
    seed: int = 1234
    with_noise: bool = True

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(dataclasses.asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def physical_feature_indices(drop_amplitude: bool) -> List[int]:
    if not drop_amplitude:
        return list(range(len(FEATURE_NAMES)))
    return [i for i, n in enumerate(FEATURE_NAMES) if n not in ABSOLUTE_AMPLITUDE_FEATURES]


class GeoWearNetDataset(Dataset):
    def __init__(self, cfg: DataConfig, norm_stats: Optional[Dict[str, np.ndarray]] = None):
        self.cfg = cfg
        self.speaker_index = manifests.load_speaker_index(cfg.split)
        self.pool = scenes.SpeechPool(self.speaker_index)
        self._noise: Optional[scenes.NoisePool] = None
        self.feat_idx = physical_feature_indices(cfg.drop_amplitude_features)
        self.norm_stats = norm_stats

    # NoisePool caches decoded wavs; build it lazily so it lives per worker.
    @property
    def noise(self) -> Optional[scenes.NoisePool]:
        if not self.cfg.with_noise:
            return None
        if self._noise is None:
            self._noise = scenes.NoisePool()
        return self._noise

    def __len__(self) -> int:
        return self.cfg.virtual_size

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng([self.cfg.seed, index])

    def _generation(self, rng: np.random.Generator) -> str:
        if self.cfg.generation == "mixed":
            return str(rng.choice(["S1", "S2"]))
        return self.cfg.generation

    def scene_for(self, index: int) -> scenes.Scene:
        rng = self._rng(index)
        cond = scenes.sample_train_condition(rng) if self.cfg.condition == "train_mix" else self.cfg.condition
        return scenes.generate_scene(
            self.pool, rng, cond, self._generation(rng), self.cfg.duration_s, self.noise
        )

    def featurize(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lm = log_mel(audio, n_mels=self.cfg.n_mels)
        ph = extract_features(audio)[:, self.feat_idx] if self.cfg.use_physical else np.zeros((len(lm), 0), np.float32)
        t = min(len(lm), len(ph)) if self.cfg.use_physical else len(lm)
        return lm[:t], ph[:t]

    def _normalize(self, lm: np.ndarray, ph: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mode = self.cfg.normalization
        if mode == "none":
            return lm, ph
        if mode == "global":
            s = self.norm_stats
            assert s is not None, "normalization='global' requires norm_stats"
            lm = (lm - s["mel_mean"]) / s["mel_std"]
            if ph.shape[1]:
                ph = (ph - s["phys_mean"]) / s["phys_std"]
            return lm, ph
        if mode == "cmvn":
            # Causal running CMVN -- deliberately destroys absolute level.
            def causal(x: np.ndarray) -> np.ndarray:
                c = np.cumsum(x, axis=0)
                c2 = np.cumsum(x**2, axis=0)
                k = np.arange(1, len(x) + 1)[:, None]
                m = c / k
                v = np.maximum(c2 / k - m**2, 1e-6)
                return (x - m) / np.sqrt(v)

            return causal(lm), (causal(ph) if ph.shape[1] else ph)
        raise ValueError(mode)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sc = self.scene_for(index)
        lm, ph = self.featurize(sc.audio)
        t = len(lm)
        w = sc.wearer[:t]
        e = sc.environment[:t]
        st = sc.state[:t]
        if len(w) < t:  # pad labels defensively
            pad = t - len(w)
            w = np.pad(w, (0, pad)); e = np.pad(e, (0, pad)); st = np.pad(st, (0, pad))
        lm, ph = self._normalize(lm, ph)
        return {
            "log_mel": torch.from_numpy(np.ascontiguousarray(lm, dtype=np.float32)),
            "physical": torch.from_numpy(np.ascontiguousarray(ph, dtype=np.float32)),
            "wearer": torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)),
            "environment": torch.from_numpy(np.ascontiguousarray(e, dtype=np.float32)),
            "state": torch.from_numpy(np.ascontiguousarray(st, dtype=np.int64)),
            "tir_db": torch.tensor(float(sc.meta["tir_db"]), dtype=torch.float32),
        }


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Right-pad to the longest sequence and carry a validity mask, so loss and
    metrics never count padded frames."""
    t = max(int(b["log_mel"].shape[0]) for b in batch)
    out: Dict[str, torch.Tensor] = {}
    mask = torch.zeros(len(batch), t)
    for i, b in enumerate(batch):
        mask[i, : b["log_mel"].shape[0]] = 1.0
    for k in ("log_mel", "physical"):
        d = batch[0][k].shape[1]
        x = torch.zeros(len(batch), t, d)
        for i, b in enumerate(batch):
            x[i, : b[k].shape[0]] = b[k]
        out[k] = x
    for k, dt in (("wearer", torch.float32), ("environment", torch.float32), ("state", torch.int64)):
        x = torch.zeros(len(batch), t, dtype=dt)
        for i, b in enumerate(batch):
            x[i, : b[k].shape[0]] = b[k]
        out[k] = x
    out["mask"] = mask
    out["tir_db"] = torch.stack([b["tir_db"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# Global normalisation statistics (Workstream O1)
# ---------------------------------------------------------------------------
def norm_stats_path(cfg: DataConfig) -> Path:
    key = f"{cfg.generation}_{cfg.n_mels}_{int(cfg.drop_amplitude_features)}_{cfg.condition}"
    return CACHE_DIR / f"norm_stats_{key}.npz"


def compute_norm_stats(cfg: DataConfig, n_scenes: int = 120, force: bool = False) -> Dict[str, np.ndarray]:
    """One fixed mean/std from the TRAINING split only. Never recomputed from
    val/test (that would be leakage) and never per-utterance."""
    p = norm_stats_path(cfg)
    if p.exists() and not force:
        with np.load(p) as z:
            return {k: z[k] for k in z.files}
    train_cfg = dataclasses.replace(cfg, split="train", normalization="none", seed=999_001)
    ds = GeoWearNetDataset(train_cfg)
    mels, phs = [], []
    for i in range(n_scenes):
        sc = ds.scene_for(i)
        lm, ph = ds.featurize(sc.audio)
        mels.append(lm)
        if ph.shape[1]:
            phs.append(ph)
    M = np.concatenate(mels, 0)
    stats = {
        "mel_mean": M.mean(0).astype(np.float32),
        "mel_std": np.maximum(M.std(0), 1e-3).astype(np.float32),
    }
    if phs:
        P = np.concatenate(phs, 0)
        stats["phys_mean"] = P.mean(0).astype(np.float32)
        stats["phys_std"] = np.maximum(P.std(0), 1e-3).astype(np.float32)
    else:
        stats["phys_mean"] = np.zeros(0, np.float32)
        stats["phys_std"] = np.ones(0, np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(p, **stats)
    return stats


def build_dataset(cfg: DataConfig) -> GeoWearNetDataset:
    stats = compute_norm_stats(cfg) if cfg.normalization == "global" else None
    return GeoWearNetDataset(cfg, norm_stats=stats)
