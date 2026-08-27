from __future__ import annotations

import time

import numpy as np
import pytest

from mentra.audio.consumer import MentraInferenceConsumer
from mentra.audio.frame import AudioFrame, MessageType
from server.audio.remote_receiver import MentraRemoteReceiver
from server.models.capabilities import active_capabilities, with_audio_policy


class _Detector:
    def __init__(self, scores: list[float]):
        self.scores = iter(scores)
        self.inputs: list[np.ndarray] = []

    def process(self, samples: np.ndarray, sample_rate: int):
        assert sample_rate == 16_000
        self.inputs.append(samples.copy())
        return {"wearer_score": next(self.scores)}


class _FourStateDetector:
    def __init__(self, scores: list[tuple[float, float]]):
        self.scores = iter(scores)

    def process(self, samples: np.ndarray, sample_rate: int):
        wearer, environment = next(self.scores)
        return {"wearer_score": wearer, "environment_score": environment}


def _audio_frame(value: int) -> AudioFrame:
    samples = np.full(3_200, value, dtype="<i2")
    return AudioFrame(
        0, time.monotonic_ns(), 16_000, 1, 16, samples.tobytes(),
        message_type=MessageType.AUDIO_FRAME,
    )


def test_consumer_reset_prevents_audio_and_state_from_crossing_wearer_sessions():
    detector = _Detector([0.9, 0.1])
    consumer = MentraInferenceConsumer(detector, context_s=2.0, hop_s=0.2)

    first = consumer.consume_frame(_audio_frame(16_384))
    assert first is not None and consumer.current_state == "WEARER"
    assert consumer.stats.frames_consumed == 1

    consumer.reset()
    assert consumer.current_state == "SILENCE"
    assert consumer.last_result is None
    assert consumer.stats.frames_consumed == 0

    second = consumer.consume_frame(_audio_frame(-16_384))
    assert second is not None and consumer.current_state == "ENVIRONMENT"
    assert len(detector.inputs) == 2
    # The second person starts with exactly their own frame, not residual PCM
    # from the previous wearer session.
    assert np.all(detector.inputs[-1] < 0.0)


def test_geowearnet_consumer_preserves_four_detector_states():
    detector = _FourStateDetector([(0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.2, 0.2)])
    consumer = MentraInferenceConsumer(
        detector, context_s=2.0, hop_s=0.2,
        wearer_high_threshold=0.6, wearer_low_threshold=0.4,
        environment_high_threshold=0.6, environment_low_threshold=0.4,
    )

    states = []
    for _ in range(4):
        result = consumer.consume_frame(_audio_frame(0))
        assert result is not None
        states.append(result.state)
    assert states == ["WEARER", "ENVIRONMENT", "OVERLAP", "SILENCE"]
    assert consumer.last_result is not None
    assert consumer.last_result.environment_score == 0.2


class _WebSocket:
    def __init__(self, start: AudioFrame):
        self.start = start.encode()
        self.sent: list[bytes] = []

    async def recv(self):
        return self.start

    async def send(self, payload: bytes):
        self.sent.append(payload)

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_receiver_emits_session_boundaries_for_shared_glasses_reset():
    starts, ends = [], []
    receiver = MentraRemoteReceiver("127.0.0.1", 19_283)
    start = AudioFrame(
        0, time.monotonic_ns(), 16_000, 1, 16, b"wearer-A|metadata",
        message_type=MessageType.STREAM_START,
    )

    await receiver._handle_connection(
        _WebSocket(start), lambda _: None,
        on_session_start=starts.append, on_session_end=ends.append,
    )

    assert starts == ["wearer-A"]
    assert ends == ["wearer-A"]


def test_capabilities_report_policy_without_claiming_source_separation():
    base = active_capabilities("geowearnet_g2")
    configured = with_audio_policy(base, "rnnoise_geowear_gate")

    assert configured is not base
    assert configured.audioPolicy == "rnnoise_geowear_gate"
    assert configured.experimental is True
    assert configured.supportsSourceSeparation is False


def test_opt_in_audio_policy_is_experimental_for_legacy_model_too():
    configured = with_audio_policy(active_capabilities("speakernet"), "geowear_gate")

    assert configured.experimental is True
