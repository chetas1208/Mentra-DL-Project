"""Real sherpa-onnx speaker-embedding wrapper (Day 1 continuation).

Mirrors the eventual Android WearerDetector contract
(android/.../inference/WearerDetector.kt) but runs on the workstation against
WAV files today. Source-agnostic: enroll()/process() take raw float32/int16
samples, never a filename, so swapping WavPcmSource for a live PCM stream
later doesn't touch this class.

Verified against: sherpa-onnx python-api-examples/speaker-identification.py
(https://github.com/k2-fsa/sherpa-onnx, master, fetched 2026-08-21).
"""
from __future__ import annotations

import numpy as np
import sherpa_onnx


class SherpaOnnxWearerDetector:
    def __init__(self, model_path: str, num_threads: int = 1, provider: str = "cpu"):
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path,
            num_threads=num_threads,
            debug=False,
            provider=provider,
        )
        if not config.validate():
            raise ValueError(f"invalid sherpa-onnx config for model={model_path}")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.dim = self.extractor.dim
        self._wearer_embedding: np.ndarray | None = None

    def _embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """samples: float32 mono in [-1, 1]. Returns L2-normalized embedding."""
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()
        assert self.extractor.is_ready(stream)
        emb = np.array(self.extractor.compute(stream), dtype=np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def enroll(self, segments: list[tuple[np.ndarray, int]]) -> np.ndarray:
        """segments: list of (samples, sample_rate). Computes mean of
        L2-normalized per-utterance embeddings, per section 7/12 of the
        sprint spec. Stores and returns the wearer embedding (computed once,
        cached — never recomputed per inference call)."""
        embs = [self._embed(s, sr) for s, sr in segments]
        mean = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean)
        wearer_emb = mean / norm if norm > 0 else mean
        self._wearer_embedding = wearer_emb
        return wearer_emb

    def process(self, samples: np.ndarray, sample_rate: int) -> dict:
        if self._wearer_embedding is None:
            raise RuntimeError("call enroll() before process()")
        emb = self._embed(samples, sample_rate)
        score = float(np.dot(emb, self._wearer_embedding))  # both L2-normalized -> cosine sim
        return {"embedding": emb, "wearer_score": score}
