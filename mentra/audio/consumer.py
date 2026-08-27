"""Receiver -> model consumer (sprint spec section 15-19: "wire it now,
not blocked on hardware"). Takes the AudioFrame stream released by
MentraRemoteReceiver's jitter buffer and turns it into live
WEARER/ENVIRONMENT decisions -- same abstraction regardless of whether the
frames originated from real Mentra PCM or a WavReplay source, per the
architecture note "the eventual real source is literally an adapter swap."

No WAV intermediary, no extra serialization: PCM16 bytes -> float32 ->
rolling buffer -> detector, once, per section 18/19.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from mentra.audio.frame import AudioFrame
from research.sherpa_onnx.detector import SherpaOnnxWearerDetector


def pcm16_bytes_to_float32(payload: bytes) -> np.ndarray:
    """int16 LE -> float32 in [-1, 1]. Explicit, tested against known
    extreme values (section 19): -32768, 0, 32767."""
    samples = np.frombuffer(payload, dtype="<i2")  # little-endian int16
    return (samples.astype(np.float32)) / 32768.0


@dataclass
class LiveDetectionResult:
    wearer_score: float
    environment_score: float | None
    state: str
    context_ms: float
    inference_ms: float
    capture_to_prediction_ms: float
    frame_capture_timestamp_ns: int


@dataclass
class ConsumerStats:
    frames_consumed: int = 0
    predictions_made: int = 0
    dropped_wrong_format: int = 0


class MentraInferenceConsumer:
    """Rolling-window consumer: accumulates float32 PCM up to `context_s`
    seconds and re-scores every `hop_s` seconds of newly-arrived audio.

    Legacy detectors expose only ``wearer_score`` and retain the original
    binary hysteresis behavior. GeoWearNet additionally exposes
    ``environment_score``; in that case the consumer independently latches
    both classes and publishes WEARER, ENVIRONMENT, OVERLAP, or SILENCE.
    """

    def __init__(self, detector: SherpaOnnxWearerDetector, sample_rate: int = 16000,
                 context_s: float = 2.0, hop_s: float = 0.2,
                 wearer_high_threshold: float = 0.5, wearer_low_threshold: float = 0.35,
                 environment_high_threshold: float | None = None,
                 environment_low_threshold: float | None = None):
        if not 0.0 <= wearer_low_threshold <= wearer_high_threshold <= 1.0:
            raise ValueError("wearer thresholds must satisfy 0 <= low <= high <= 1")
        if environment_high_threshold is None:
            environment_high_threshold = wearer_high_threshold
        if environment_low_threshold is None:
            environment_low_threshold = wearer_low_threshold
        if not 0.0 <= environment_low_threshold <= environment_high_threshold <= 1.0:
            raise ValueError("environment thresholds must satisfy 0 <= low <= high <= 1")
        self.detector = detector
        self.sample_rate = sample_rate
        self.context_samples = int(context_s * sample_rate)
        self.hop_samples = int(hop_s * sample_rate)
        self.wearer_high = wearer_high_threshold
        self.wearer_low = wearer_low_threshold
        self.environment_high = environment_high_threshold
        self.environment_low = environment_low_threshold

        self._buffer = np.zeros(0, dtype=np.float32)
        self._samples_since_last_hop = 0
        self._current_state = "SILENCE"
        self._wearer_active = False
        self._environment_active = False
        self.stats = ConsumerStats()
        self.last_result: LiveDetectionResult | None = None

    def enroll(self, enrollment_segments: list[tuple[np.ndarray, int]]) -> None:
        self.detector.enroll(enrollment_segments)

    def reset(self) -> None:
        """Forget stream-local state at a transport/session boundary.

        This deliberately does not call ``detector.enroll`` or otherwise
        modify a detector's long-lived model parameters.  In the GeoWearNet
        path there is no enrolled identity to retain; in the legacy
        SpeakerNet path enrollment is owned by that detector and remains its
        explicit, separate lifecycle.
        """
        self._buffer = np.zeros(0, dtype=np.float32)
        self._samples_since_last_hop = 0
        self._current_state = "SILENCE"
        self._wearer_active = False
        self._environment_active = False
        self.stats = ConsumerStats()
        self.last_result = None

    @property
    def current_state(self) -> str:
        """Current classification state, updated on hop boundaries but
        readable on every raw frame -- needed for gating/ASR-routing
        decisions that should apply continuously, not just at hop time."""
        return self._current_state

    def consume_frame(self, frame: AudioFrame) -> LiveDetectionResult | None:
        if frame.sample_rate != self.sample_rate or frame.bits_per_sample != 16 or frame.channels != 1:
            self.stats.dropped_wrong_format += 1
            return None

        samples = pcm16_bytes_to_float32(frame.payload)
        self._buffer = np.concatenate([self._buffer, samples])
        if len(self._buffer) > self.context_samples:
            self._buffer = self._buffer[-self.context_samples:]

        self.stats.frames_consumed += 1
        self._samples_since_last_hop += len(samples)

        if self._samples_since_last_hop < self.hop_samples:
            return None  # not time for a new prediction yet
        if len(self._buffer) < self.hop_samples:
            return None  # not enough context accumulated yet (startup)

        self._samples_since_last_hop = 0
        t0 = time.perf_counter()
        result = self.detector.process(self._buffer, self.sample_rate)
        inference_ms = (time.perf_counter() - t0) * 1000

        score = float(result["wearer_score"])
        environment_score = result.get("environment_score")
        if environment_score is None:
            # Preserve the legacy SpeakerNet contract exactly: only a
            # wearer/non-wearer score exists, and the hysteresis band holds
            # the previously emitted binary state.
            if score >= self.wearer_high:
                self._current_state = "WEARER"
            elif score < self.wearer_low:
                self._current_state = "ENVIRONMENT"
        else:
            environment_score = float(environment_score)
            self._wearer_active = (
                score > self.wearer_low if self._wearer_active
                else score >= self.wearer_high
            )
            self._environment_active = (
                environment_score > self.environment_low if self._environment_active
                else environment_score >= self.environment_high
            )
            if self._wearer_active and self._environment_active:
                self._current_state = "OVERLAP"
            elif self._wearer_active:
                self._current_state = "WEARER"
            elif self._environment_active:
                self._current_state = "ENVIRONMENT"
            else:
                self._current_state = "SILENCE"

        now_ns = time.monotonic_ns()
        capture_to_prediction_ms = (now_ns - frame.capture_timestamp_ns) / 1e6

        live_result = LiveDetectionResult(
            wearer_score=score,
            environment_score=environment_score,
            state=self._current_state,
            context_ms=(len(self._buffer) / self.sample_rate) * 1000,
            inference_ms=inference_ms,
            capture_to_prediction_ms=capture_to_prediction_ms,
            frame_capture_timestamp_ns=frame.capture_timestamp_ns,
        )
        self.stats.predictions_made += 1
        self.last_result = live_result
        return live_result
