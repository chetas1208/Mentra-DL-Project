"""Source-agnostic PCM abstraction (sprint spec section 2).

WearerDetector implementations must never know whether audio came from a WAV
file, the Mentra Bluetooth callback, or a live mic. Today: WavPcmSource.
Later: MentraPcmSource (Kotlin, wraps onMicPcm) — same shape, downstream
model code unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import soundfile as sf


class PcmSource(ABC):
    """Yields mono PCM chunks at a fixed sample rate, in temporal order."""

    sample_rate: int

    @abstractmethod
    def chunks(self, chunk_ms: int):
        """Yields float32 mono arrays of length chunk_ms/1000 * sample_rate,
        in order, simulating a live streaming callback. Last chunk may be
        shorter."""
        ...

    @abstractmethod
    def read_all(self) -> np.ndarray:
        """Returns the full signal at once (for non-streaming baseline use)."""
        ...


class WavPcmSource(PcmSource):
    def __init__(self, wav_path: str):
        data, sr = sf.read(wav_path, always_2d=True, dtype="float32")
        self._samples = np.ascontiguousarray(data[:, 0])  # first channel only
        self.sample_rate = sr
        self.path = wav_path

    def read_all(self) -> np.ndarray:
        return self._samples

    def chunks(self, chunk_ms: int):
        chunk_len = int(self.sample_rate * chunk_ms / 1000)
        for start in range(0, len(self._samples), chunk_len):
            yield self._samples[start:start + chunk_len]
