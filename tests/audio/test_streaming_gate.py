"""G4 WS1/WS2 -- the live streaming gate envelope.

The central claim these tests defend is the one G4 exists to establish: the
live receiver now routes audio with the SAME envelope the offline product
evaluator measured. If that stops being true, the product numbers stop
describing the shipped path, so it is asserted bit-for-bit rather than
approximately.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.agent_audio.gate import GatePolicy, GeoWearGate, apply_gain_envelope
from server.audio.streaming_gate import (MAX_PREROLL_MS, StreamingGateRouter,
                                         named_gate_policy)

SR = 16000
HOP = 160


def _signal(n_frames: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(n_frames * HOP) * 0.1).astype(np.float32)
    return (audio,
            rng.random(n_frames).astype(np.float32),
            rng.random(n_frames).astype(np.float32))


def _drive(router: StreamingGateRouter, audio: np.ndarray, p_w, p_e,
           block_frames: int = 1) -> np.ndarray:
    out = []
    step = block_frames * HOP
    for i in range(0, len(audio), step):
        frame_index = i // HOP
        for k in range(block_frames):
            if frame_index + k < len(p_w):
                router.update_probabilities(p_w[frame_index + k], p_e[frame_index + k])
        routed, _ = router.process(audio[i:i + step])
        out.append(routed)
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def test_router_is_bit_identical_to_the_offline_evaluator():
    """The whole point of WS1. Same probabilities, zero pre-roll -> the live
    router must produce exactly what the product matrix measured."""
    audio, p_w, p_e = _signal(400, seed=1)
    policy = GatePolicy(name="A_balanced")

    offline, _ = GeoWearGate(policy, sr=SR).apply(audio, p_w, p_e)
    live = _drive(StreamingGateRouter(policy, sr=SR), audio, p_w, p_e)

    assert live.shape == offline.shape
    assert np.array_equal(live, offline)


@pytest.mark.parametrize("policy", [p for p in __import__(
    "evaluation.agent_audio.gate", fromlist=["policy_variants"]).policy_variants()])
def test_router_matches_offline_for_every_predeclared_policy(policy):
    audio, p_w, p_e = _signal(250, seed=2)
    offline, _ = GeoWearGate(policy, sr=SR).apply(audio, p_w, p_e)
    live = _drive(StreamingGateRouter(policy, sr=SR), audio, p_w, p_e)
    assert np.array_equal(live, offline)


@pytest.mark.parametrize("block_frames", [2, 5, 20])
def test_router_output_is_independent_of_block_size(block_frames):
    """Block size is a transport detail; for the SAME interleaving of
    probability updates and audio it must not change a single sample.

    The probability is published once per block in both arms, because that is
    the real live pattern: the detector updates far more slowly than the 10 ms
    audio cadence, and the router holds the last value in between."""
    audio, p_w, p_e = _signal(200, seed=3)
    policy = GatePolicy()

    def drive(step_frames: int) -> np.ndarray:
        router = StreamingGateRouter(policy, sr=SR)
        out = []
        for start in range(0, len(p_w), block_frames):
            router.update_probabilities(p_w[start], p_e[start])
            block = audio[start * HOP:(start + block_frames) * HOP]
            for offset in range(0, len(block), step_frames * HOP):
                routed, _ = router.process(block[offset:offset + step_frames * HOP])
                out.append(routed)
        return np.concatenate(out)

    assert np.array_equal(drive(1), drive(block_frames))


def test_preroll_delays_audio_relative_to_the_gain_and_keeps_length():
    """Pre-roll must be a real delay line, not lookahead: output sample j is
    input sample j-d scaled by the gain decided at j."""
    audio, p_w, p_e = _signal(300, seed=4)
    policy = GatePolicy()
    preroll_ms = 100.0
    d = int(preroll_ms / policy.frame_hop_ms) * HOP

    envelope = GeoWearGate(policy, sr=SR).run(p_w, p_e)
    per_sample = apply_gain_envelope(np.ones_like(audio), envelope, HOP)
    delayed = np.concatenate([np.zeros(d, np.float32), audio[:-d]])
    expected = (delayed * per_sample).astype(np.float32)

    live = _drive(StreamingGateRouter(policy, sr=SR, preroll_ms=preroll_ms),
                  audio, p_w, p_e)

    assert live.shape == audio.shape
    assert np.array_equal(live, expected)


def test_preroll_offline_shift_matches_the_live_delay_line():
    """`shift_gain_for_preroll` is used offline to evaluate pre-roll cheaply.
    It must be exactly what the router does, or the WS2 grid is fiction."""
    from training.geowearnet.g4.probsource import shift_gain_for_preroll

    audio, p_w, p_e = _signal(300, seed=5)
    policy = GatePolicy()
    envelope = GeoWearGate(policy, sr=SR).run(p_w, p_e)

    for preroll_ms in (50.0, 100.0, 150.0, 200.0):
        shifted = shift_gain_for_preroll(envelope, preroll_ms)
        live = _drive(StreamingGateRouter(policy, sr=SR, preroll_ms=preroll_ms),
                      audio, p_w, p_e)
        d = int(preroll_ms / policy.frame_hop_ms) * HOP
        # compare where both definitions are in steady state (past the primed
        # silence and before the shifted envelope runs off the end)
        offline_equiv = apply_gain_envelope(audio, shifted, HOP)
        assert np.allclose(live[d:], offline_equiv[:len(live) - d], atol=1e-6)


def test_preroll_reports_its_real_delay_and_is_bounded():
    router = StreamingGateRouter(GatePolicy(), sr=SR, preroll_ms=150.0)
    report = router.latency_report()
    assert report["preroll_ms"] == 150.0
    assert report["gate_lookahead_ms"] == 0.0
    assert report["algorithmic_delay_ms"] == pytest.approx(25.0 + 10.0 + 150.0)

    # a requested value that is not a whole frame is quantised, and the
    # EFFECTIVE value is what gets reported
    quantised = StreamingGateRouter(GatePolicy(), sr=SR, preroll_ms=97.0)
    assert quantised.preroll_ms == 100.0

    with pytest.raises(ValueError, match="bounded maximum"):
        StreamingGateRouter(GatePolicy(), sr=SR, preroll_ms=MAX_PREROLL_MS + 1.0)
    with pytest.raises(ValueError):
        StreamingGateRouter(GatePolicy(), sr=SR, preroll_ms=-1.0)


def test_offline_tail_convention_differs_and_only_at_the_tail():
    """A real, documented, bounded difference found while measuring WS3.

    ``apply_gain_envelope`` holds the last gain over any audio beyond the
    supplied envelope; the router instead keeps advancing its state machine.
    That divergence is confined to the frames with no probability -- the
    feature extractor emits slightly fewer frames than the audio contains --
    and must never spread earlier. Pinned here so parity comparisons keep
    excluding that tail deliberately rather than by accident."""
    audio, p_w, p_e = _signal(100, seed=9)
    short_w, short_e = p_w[:98], p_e[:98]
    policy = GatePolicy()

    offline, _ = GeoWearGate(policy, sr=SR).apply(audio, short_w, short_e)
    live = _drive(StreamingGateRouter(policy, sr=SR), audio, short_w, short_e)

    covered = 98 * HOP
    assert np.array_equal(live[:covered], offline[:covered])
    # beyond it the two conventions may disagree; that region is 2 frames long
    assert len(offline) - covered == 2 * HOP


def test_router_rejects_misaligned_blocks_instead_of_drifting():
    router = StreamingGateRouter(GatePolicy(), sr=SR)
    with pytest.raises(ValueError, match="whole number"):
        router.process(np.zeros(150, dtype=np.float32))


def test_router_rejects_non_finite_and_multichannel_input():
    router = StreamingGateRouter(GatePolicy(), sr=SR)
    with pytest.raises(ValueError, match="NaN"):
        router.process(np.full(HOP, np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="mono"):
        router.process(np.zeros((2, HOP), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        router.update_probabilities(float("nan"), 0.0)


def test_reset_restores_a_pristine_router_and_keeps_no_identity():
    """Shared-glasses invariant at the software level: after reset the router
    must behave as if it had never seen the previous wearer."""
    audio, p_w, p_e = _signal(120, seed=6)
    router = StreamingGateRouter(GatePolicy(), sr=SR, preroll_ms=100.0)

    first = _drive(router, audio, p_w, p_e)
    router.reset()
    second = _drive(router, audio, p_w, p_e)

    assert np.array_equal(first, second)
    stats = router.stats()
    assert stats["frames_scored"] == len(p_w)
    # no attribute anywhere on the router holds a wearer/user identity
    assert not any("wearer_id" in name or "speaker" in name or "enroll" in name
                   for name in vars(router))


def test_state_fallback_maps_detector_states_to_probabilities():
    router = StreamingGateRouter(GatePolicy(), sr=SR)
    router.update_from_state("OVERLAP")
    # the policy's min_state_ms debounce means a state needs several frames
    # to commit -- exactly the anti-chop behaviour we want preserved here
    router.process(np.zeros(10 * HOP, dtype=np.float32))
    assert router.state_name == "OVERLAP"
    with pytest.raises(ValueError, match="unknown detector state"):
        router.update_from_state("NOT_A_STATE")


def test_named_gate_policy_is_restricted_to_measured_variants():
    assert named_gate_policy("A_balanced").name == "A_balanced"
    with pytest.raises(ValueError, match="unknown gate policy"):
        named_gate_policy("something_nobody_measured")


def test_silence_is_still_silence_through_the_envelope():
    """A closed gate must actually produce zeros, not a small residual."""
    audio = np.ones(100 * HOP, dtype=np.float32)
    router = StreamingGateRouter(GatePolicy(), sr=SR)
    out = _drive(router, audio, np.zeros(100, np.float32), np.zeros(100, np.float32))
    # after the release ramp completes the output must be exactly zero
    assert np.max(np.abs(out[-50 * HOP:])) == 0.0
