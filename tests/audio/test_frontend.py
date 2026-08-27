from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from server.audio.frontend import AudioFrontend


@dataclass
class _DenoiseResult:
    audio: np.ndarray
    available: bool = True


class _FakeDenoiser:
    def __init__(self, available: bool = True):
        self.available = available
        self.calls = 0
        self.reset_calls = 0

    def process(self, audio: np.ndarray) -> _DenoiseResult:
        self.calls += 1
        return _DenoiseResult(audio + 0.25, self.available)

    def reset(self) -> None:
        self.reset_calls += 1


def test_passthrough_preserves_audio_for_every_detector_state():
    audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    frontend = AudioFrontend("passthrough")

    for state in ("WEARER", "ENVIRONMENT", "SILENCE", "OVERLAP"):
        result = frontend.process(audio, state)
        assert np.array_equal(result.audio, audio)
        assert result.gated is False
        assert result.denoised is False


@pytest.mark.parametrize("state", ("ENVIRONMENT", "SILENCE"))
def test_geowear_gate_mutes_inactive_states(state):
    audio = np.ones(160, dtype=np.float32)
    result = AudioFrontend("geowear_gate").process(audio, state)

    assert np.array_equal(result.audio, np.zeros_like(audio))
    assert result.gated is True


@pytest.mark.parametrize("state", ("WEARER", "OVERLAP"))
def test_geowear_gate_passes_wearer_dominant_states(state):
    audio = np.linspace(-0.5, 0.5, 160, dtype=np.float32)
    result = AudioFrontend("geowear_gate").process(audio, state)

    assert np.array_equal(result.audio, audio)


def test_rnnoise_geowear_gate_uses_injected_denoiser_and_reset():
    denoiser = _FakeDenoiser()
    audio = np.zeros(160, dtype=np.float32)
    frontend = AudioFrontend("rnnoise_geowear_gate", denoiser)

    result = frontend.process(audio, "WEARER")
    frontend.reset()

    assert np.allclose(result.audio, 0.25)
    assert result.denoised is True
    assert denoiser.calls == 1
    assert denoiser.reset_calls == 1


def test_rnnoise_policy_requires_available_denoiser():
    with pytest.raises(ValueError, match="requires an injected denoiser"):
        AudioFrontend("rnnoise_geowear_gate")

    with pytest.raises(RuntimeError, match="denoiser is unavailable"):
        AudioFrontend("rnnoise_geowear_gate", _FakeDenoiser(available=False)).process(
            np.zeros(160, dtype=np.float32), "WEARER"
        )


def test_frontend_rejects_invalid_inputs():
    frontend = AudioFrontend("passthrough")
    with pytest.raises(ValueError, match="unknown detector state"):
        frontend.process(np.zeros(160, dtype=np.float32), "UNKNOWN")
    with pytest.raises(ValueError, match="mono 1-D"):
        frontend.process(np.zeros((2, 160), dtype=np.float32), "WEARER")
    with pytest.raises(ValueError, match="NaN"):
        frontend.process(np.array([np.nan], dtype=np.float32), "WEARER")
