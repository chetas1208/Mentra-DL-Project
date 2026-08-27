"""Runtime adapter for the selected GeoWearNet G2 checkpoint.

The training and evaluation code operates on feature tensors, while the live
receiver operates on PCM16 frames. This adapter reproduces the G2 feature
extraction and normalization exactly, then returns a scalar wearer
probability compatible with ``MentraInferenceConsumer``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from training.geowearnet.acoustics import log_mel
from training.geowearnet.data import physical_feature_indices
from training.geowearnet.features import extract_features
from training.geowearnet.mmcsg.norm_stats import path_for
from training.geowearnet.mmcsg_transfer import load_checkpoint


SAMPLE_RATE = 16_000


class GeoWearNetDetector:
    """PCM-to-score adapter for an enrollment-free GeoWearNet checkpoint."""

    def __init__(self, checkpoint_path: str | Path, num_threads: int = 1):
        if num_threads < 1:
            raise ValueError("num_threads must be >= 1")

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"GeoWearNet checkpoint not found: {self.checkpoint_path}")

        torch.set_num_threads(num_threads)
        self.model, self.config, self.checkpoint = load_checkpoint(self.checkpoint_path)
        self.model.eval()

        train_config = self.checkpoint.get("config", {})
        self.normalization = train_config.get("normalization", "global")
        self.drop_amplitude_features = bool(train_config.get("drop_amplitude_features", False))
        self.feature_indices = physical_feature_indices(self.drop_amplitude_features)
        self.stats: Optional[Dict[str, np.ndarray]] = None

        if self.normalization == "global":
            channel = train_config.get("channel", 2)
            stats_path = path_for(channel, self.drop_amplitude_features)
            if not stats_path.is_file():
                raise FileNotFoundError(
                    "GeoWearNet global-normalization stats not found: "
                    f"{stats_path}. Rebuild the train-derived stats before starting G2."
                )
            with np.load(stats_path) as values:
                self.stats = {key: values[key].copy() for key in values.files}
        elif self.normalization != "none":
            raise ValueError(
                f"unsupported live GeoWearNet normalization={self.normalization!r}; "
                "only 'global' and 'none' are supported"
            )

        if self.config.use_physical_features:
            expected = len(self.feature_indices)
            if self.config.n_physical_features != expected:
                raise ValueError(
                    "checkpoint feature contract mismatch: "
                    f"model expects {self.config.n_physical_features} physical features, "
                    f"but its training config selects {expected}"
                )

    def enroll(self, segments: list[tuple[np.ndarray, int]]) -> None:
        """Reject enrollment explicitly; G2 is enrollment-free."""
        del segments
        raise RuntimeError("GeoWearNet G2 is enrollment-free and does not accept enrollment")

    def _features(self, samples: np.ndarray, sample_rate: int) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"GeoWearNet expects {SAMPLE_RATE} Hz PCM, got {sample_rate} Hz")

        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0:
            raise ValueError("GeoWearNet expects a non-empty mono PCM array")
        if not np.isfinite(audio).all():
            raise ValueError("GeoWearNet input contains NaN or Inf")

        mel = log_mel(audio, n_mels=self.config.n_mels, sr=SAMPLE_RATE)
        if self.config.use_physical_features:
            physical = extract_features(audio)[:, self.feature_indices]
        else:
            physical = np.zeros((len(mel), 0), dtype=np.float32)

        n_frames = min(len(mel), len(physical)) if self.config.use_physical_features else len(mel)
        mel = np.ascontiguousarray(mel[:n_frames], dtype=np.float32)
        physical = np.ascontiguousarray(physical[:n_frames], dtype=np.float32)

        if self.normalization == "global":
            assert self.stats is not None
            mel = (mel - self.stats["mel_mean"]) / self.stats["mel_std"]
            if physical.shape[1]:
                physical = (physical - self.stats["phys_mean"]) / self.stats["phys_std"]

        mel_tensor = torch.from_numpy(mel).unsqueeze(0)
        physical_tensor = (
            torch.from_numpy(physical).unsqueeze(0)
            if physical.shape[1]
            else None
        )
        return mel_tensor, physical_tensor

    @torch.inference_mode()
    def process(self, samples: np.ndarray, sample_rate: int) -> dict:
        """Return final-frame G2 scores for a causal rolling PCM window."""
        mel, physical = self._features(samples, sample_rate)
        if mel.shape[1] == 0:
            raise ValueError("GeoWearNet produced no feature frames")

        output = self.model(mel, physical)
        wearer_logit = float(output["wearer_logits"][0, -1].item())
        environment_logit = float(output["environment_logits"][0, -1].item())
        return {
            "wearer_score": float(torch.sigmoid(torch.tensor(wearer_logit)).item()),
            "environment_score": float(torch.sigmoid(torch.tensor(environment_logit)).item()),
            "wearer_logit": wearer_logit,
            "environment_logit": environment_logit,
        }
