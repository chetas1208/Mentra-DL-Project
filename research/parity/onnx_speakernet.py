"""ONNX SpeakerNet wrapper for parity testing (sprint spec section 11).
Thin adapter over the already-validated research/sherpa_onnx/detector.py --
reuses the working embedding path rather than reimplementing it.
"""
from __future__ import annotations

import numpy as np

from research.sherpa_onnx.detector import SherpaOnnxWearerDetector


class OnnxSpeakerNetWrapper:
    def __init__(self, model_path: str = "models/sherpa-speaker/nemo_en_speakerverification_speakernet.onnx"):
        self._detector = SherpaOnnxWearerDetector(model_path)

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return self._detector._embed(samples, sample_rate)

    @property
    def dim(self) -> int:
        return self._detector.dim
