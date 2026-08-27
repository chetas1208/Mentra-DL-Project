"""G4 WS1/WS41 -- the envelope audio policies on the live AudioFrontend.

Also pins the invariants that must survive glasses being handed between
people: no enrollment is required, no identity is retained, and a session
reset genuinely returns the frontend to its initial state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from evaluation.agent_audio.gate import GatePolicy, GeoWearGate
from server.audio.frontend import (ENVELOPE_POLICIES, SUPPORTED_POLICIES, AudioFrontend)
from server.audio.streaming_gate import StreamingGateRouter

HOP = 160


@dataclass
class _DenoiseResult:
    audio: np.ndarray
    available: bool = True


class _FakeDenoiser:
    def __init__(self):
        self.calls = 0
        self.reset_calls = 0

    def process(self, audio: np.ndarray) -> _DenoiseResult:
        self.calls += 1
        return _DenoiseResult(audio * 0.5)

    def reset(self) -> None:
        self.reset_calls += 1


def test_envelope_policies_are_registered_but_not_the_default():
    assert set(ENVELOPE_POLICIES) <= set(SUPPORTED_POLICIES)
    # the binary G3 policies must still exist unchanged, so previously
    # reported live behaviour stays reproducible
    assert "geowear_gate" in SUPPORTED_POLICIES
    assert AudioFrontend().policy == "passthrough"


def test_capability_policy_list_does_not_drift_from_the_frontend():
    """The capability module duplicates the policy list to stay import-light.
    This is the anti-drift guard that makes that safe."""
    from server.models.capabilities import AUDIO_POLICIES

    assert tuple(AUDIO_POLICIES) == tuple(SUPPORTED_POLICIES)


@pytest.mark.parametrize("policy", ENVELOPE_POLICIES)
def test_envelope_policies_stay_experimental_enrollment_free_and_non_separating(policy):
    from server.models.capabilities import active_capabilities, with_audio_policy

    capabilities = with_audio_policy(active_capabilities("geowearnet_g2"), policy)
    assert capabilities.audioPolicy == policy
    assert capabilities.experimental is True
    assert capabilities.requiresEnrollment is False
    assert capabilities.supportsSourceSeparation is False
    # the registry itself must not have been mutated by a per-receiver choice
    assert active_capabilities("geowearnet_g2").audioPolicy == "passthrough"


def test_no_geowearnet_g4_model_is_advertised():
    """G4 produced no new checkpoint. Registering a `geowearnet_g4` id would
    imply a Mentra-adapted model that does not exist."""
    from server.models.capabilities import REGISTRY

    assert "geowearnet_g4" not in REGISTRY


def test_envelope_policy_reproduces_the_offline_gate_through_the_frontend():
    rng = np.random.default_rng(11)
    n_frames = 200
    audio = (rng.standard_normal(n_frames * HOP) * 0.1).astype(np.float32)
    p_w = rng.random(n_frames).astype(np.float32)
    p_e = rng.random(n_frames).astype(np.float32)
    policy = GatePolicy(name="A_balanced")

    offline, _ = GeoWearGate(policy, sr=16000).apply(audio, p_w, p_e)

    frontend = AudioFrontend("geowear_envelope",
                             router=StreamingGateRouter(policy, sr=16000))
    frontend.reset()
    out = []
    for i in range(n_frames):
        frontend.update_probabilities(p_w[i], p_e[i])
        result = frontend.process(audio[i * HOP:(i + 1) * HOP], "WEARER")
        assert result.gated is True
        assert result.gain is not None and result.gate_state is not None
        out.append(result.audio)

    assert np.array_equal(np.concatenate(out), offline)


def test_binary_gate_policy_behaviour_is_unchanged():
    """Regression guard: G3's coarse policy must not silently acquire an
    envelope, or its published numbers would stop describing it."""
    audio = np.ones(HOP, dtype=np.float32)
    assert np.array_equal(AudioFrontend("geowear_gate").process(audio, "WEARER").audio, audio)
    assert np.array_equal(AudioFrontend("geowear_gate").process(audio, "ENVIRONMENT").audio,
                          np.zeros_like(audio))
    assert AudioFrontend("geowear_gate").process(audio, "WEARER").gain is None


def test_envelope_policy_falls_back_to_state_before_any_detector_result():
    """A live stream produces audio frames before the first 200 ms detector
    hop. Those frames must still be routed, from the detector's state."""
    frontend = AudioFrontend("geowear_envelope")
    frontend.reset()
    result = frontend.process(np.ones(HOP, dtype=np.float32), "WEARER")
    assert result.gated is True
    assert frontend.router.stats()["have_probabilities"] is True


def test_rnnoise_envelope_requires_a_denoiser_and_uses_it():
    with pytest.raises(ValueError, match="requires an injected denoiser"):
        AudioFrontend("rnnoise_geowear_envelope")

    denoiser = _FakeDenoiser()
    frontend = AudioFrontend("rnnoise_geowear_envelope", denoiser=denoiser)
    frontend.reset()
    frontend.update_probabilities(1.0, 0.0)
    frontend.process(np.ones(HOP, dtype=np.float32), "WEARER")
    assert denoiser.calls == 1
    assert denoiser.reset_calls == 1


def test_router_is_rejected_for_non_envelope_policies():
    with pytest.raises(ValueError, match="does not use a streaming gate router"):
        AudioFrontend("geowear_gate", router=StreamingGateRouter())


def test_envelope_policy_requires_frame_aligned_blocks():
    frontend = AudioFrontend("geowear_envelope")
    frontend.reset()
    with pytest.raises(ValueError, match="whole number"):
        frontend.process(np.zeros(101, dtype=np.float32), "WEARER")


def test_session_reset_clears_the_envelope_and_retains_no_wearer():
    """The shared-glasses software contract: after reset, wearer B gets the
    same treatment wearer A got at the start of their session."""
    rng = np.random.default_rng(12)
    audio = (rng.standard_normal(60 * HOP) * 0.1).astype(np.float32)
    frontend = AudioFrontend("geowear_envelope",
                             router=StreamingGateRouter(preroll_ms=100.0))

    def session() -> np.ndarray:
        frontend.reset()
        out = []
        for i in range(60):
            frontend.update_probabilities(1.0 if i > 20 else 0.0, 0.0)
            out.append(frontend.process(audio[i * HOP:(i + 1) * HOP], "WEARER").audio)
        return np.concatenate(out)

    assert np.array_equal(session(), session())


def test_frontend_never_claims_source_separation():
    """OVERLAP passes the ORIGINAL mixed PCM. A router cannot separate two
    voices and must never imply that it did."""
    rng = np.random.default_rng(13)
    audio = (rng.standard_normal(40 * HOP) * 0.1).astype(np.float32)
    frontend = AudioFrontend("geowear_envelope")
    frontend.reset()
    out = []
    for i in range(40):
        frontend.update_probabilities(1.0, 1.0)  # both speaking
        out.append(frontend.process(audio[i * HOP:(i + 1) * HOP], "OVERLAP").audio)
    routed = np.concatenate(out)
    # once the gate is fully open the overlap frames are the untouched mixture
    assert np.allclose(routed[-10 * HOP:], audio[-10 * HOP:], atol=1e-6)
