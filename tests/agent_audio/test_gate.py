"""P1.19 -- deterministic tests for the GeoWear Gate state machine.

Every test here is closed-form: synthetic probability sequences, exact
expected gains. No model, no corpus, no randomness.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.agent_audio.gate import (GatePolicy, GeoWearGate, SILENCE, WEARER_ONLY,
                                         ENVIRONMENT_ONLY, OVERLAP,
                                         algorithmic_latency_ms, apply_gain_envelope,
                                         oracle_gate, policy_variants)

SR = 16000


def _instant() -> GatePolicy:
    """No smoothing at all -- isolates the state machine from the ramps."""
    return GatePolicy(name="t", attack_ms=0.0, release_ms=0.0,
                      min_state_ms=0.0, hangover_ms=0.0)


# ---------------------------------------------------------------------------
# basic routing
# ---------------------------------------------------------------------------
def test_wearer_only_passes():
    g = GeoWearGate(_instant())
    gain = g.run([0.9] * 20, [0.0] * 20)
    assert np.allclose(gain, 1.0)
    assert g.state == WEARER_ONLY


def test_environment_only_is_muted():
    g = GeoWearGate(_instant())
    gain = g.run([0.0] * 20, [0.9] * 20)
    assert np.allclose(gain, 0.0)
    assert g.state == ENVIRONMENT_ONLY


def test_silence_is_muted():
    g = GeoWearGate(_instant())
    gain = g.run([0.0] * 20, [0.0] * 20)
    assert np.allclose(gain, 0.0)
    assert g.state == SILENCE


def test_environment_attenuation_policy():
    p = _instant()
    p.env_gain = 0.25
    g = GeoWearGate(p)
    gain = g.run([0.0] * 10, [0.9] * 10)
    assert np.allclose(gain, 0.25)


# ---------------------------------------------------------------------------
# overlap policy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy,expected", [
    ("pass_raw", 1.0),
    ("mute", 0.0),
    ("attenuate", 0.4),
])
def test_overlap_policies(policy, expected):
    p = _instant()
    p.overlap_policy = policy
    p.overlap_gain = 0.4
    g = GeoWearGate(p)
    gain = g.run([0.9] * 10, [0.9] * 10)
    assert g.state == OVERLAP
    assert np.allclose(gain, expected)


def test_unknown_overlap_policy_raises():
    p = _instant()
    p.overlap_policy = "nonsense"
    g = GeoWearGate(p)
    with pytest.raises(ValueError):
        g.run([0.9] * 3, [0.9] * 3)


# ---------------------------------------------------------------------------
# hysteresis
# ---------------------------------------------------------------------------
def test_hysteresis_prevents_flapping():
    """A probability oscillating strictly BETWEEN off and on thresholds must
    not change the latched decision even once."""
    p = _instant()
    p.wearer_on, p.wearer_off = 0.6, 0.4
    g = GeoWearGate(p)
    g.run([0.9], [0.0])                      # latch ON
    assert g.state == WEARER_ONLY
    osc = [0.45, 0.55, 0.41, 0.59] * 10      # all inside the dead zone
    gain = g.run(osc, [0.0] * len(osc))
    assert np.allclose(gain, 1.0), "gate flapped inside the hysteresis dead zone"


def test_hysteresis_releases_below_off_threshold():
    p = _instant()
    p.wearer_on, p.wearer_off = 0.6, 0.4
    g = GeoWearGate(p)
    g.run([0.9], [0.0])
    gain = g.run([0.2] * 5, [0.0] * 5)
    assert gain[-1] == 0.0


def test_hysteresis_requires_on_threshold_to_open():
    p = _instant()
    p.wearer_on, p.wearer_off = 0.6, 0.4
    g = GeoWearGate(p)
    gain = g.run([0.5] * 10, [0.0] * 10)     # above OFF but below ON, never latched
    assert np.allclose(gain, 0.0)


# ---------------------------------------------------------------------------
# attack / release
# ---------------------------------------------------------------------------
def test_attack_ramps_over_expected_frames():
    p = GatePolicy(name="t", attack_ms=50.0, release_ms=0.0, min_state_ms=0.0,
                   hangover_ms=0.0, frame_hop_ms=10.0)
    g = GeoWearGate(p)
    gain = g.run([0.9] * 10, [0.0] * 10)
    # 50 ms attack / 10 ms hop = 5 frames to go 0 -> 1
    assert gain[0] == pytest.approx(0.2)
    assert gain[4] == pytest.approx(1.0)
    assert np.all(np.diff(gain[:5]) > 0), "attack must be monotonically rising"


def test_release_ramps_over_expected_frames():
    p = GatePolicy(name="t", attack_ms=0.0, release_ms=50.0, min_state_ms=0.0,
                   hangover_ms=0.0, frame_hop_ms=10.0)
    g = GeoWearGate(p)
    g.run([0.9] * 3, [0.0] * 3)
    assert g.gain == pytest.approx(1.0)
    gain = g.run([0.0] * 10, [0.0] * 10)
    assert gain[0] == pytest.approx(0.8)
    assert gain[4] == pytest.approx(0.0)
    assert np.all(np.diff(gain[:5]) < 0), "release must be monotonically falling"


def test_attack_is_faster_than_release_by_default():
    p = GatePolicy()
    assert p.attack_ms < p.release_ms, "opening must be faster than closing"


# ---------------------------------------------------------------------------
# hangover / min-state
# ---------------------------------------------------------------------------
def test_hangover_holds_gate_open_through_silence():
    p = GatePolicy(name="t", attack_ms=0.0, release_ms=0.0, min_state_ms=0.0,
                   hangover_ms=100.0, frame_hop_ms=10.0)
    g = GeoWearGate(p)
    g.run([0.9] * 5, [0.0] * 5)
    gain = g.run([0.0] * 20, [0.0] * 20)     # silence after wearer speech
    assert gain[0] == 1.0, "hangover must keep the tail of a word"
    assert gain[-1] == 0.0, "hangover must eventually expire"
    assert 1 <= int((gain > 0.5).sum()) <= 12


def test_hangover_does_not_hold_open_into_bystander_speech():
    """The critical anti-leakage property: hangover must never keep the gate
    open when the BYSTANDER starts talking."""
    p = GatePolicy(name="t", attack_ms=0.0, release_ms=0.0, min_state_ms=0.0,
                   hangover_ms=300.0, frame_hop_ms=10.0)
    g = GeoWearGate(p)
    g.run([0.9] * 5, [0.0] * 5)
    gain = g.run([0.0] * 20, [0.9] * 20)     # bystander takes the floor
    assert gain[-1] == 0.0
    assert gain.mean() < 0.5, "hangover leaked the gate open into bystander speech"


def test_min_state_duration_debounces_single_frame_glitch():
    p = GatePolicy(name="t", attack_ms=0.0, release_ms=0.0, min_state_ms=50.0,
                   hangover_ms=0.0, frame_hop_ms=10.0)
    g = GeoWearGate(p)
    g.run([0.9] * 10, [0.0] * 10)            # settle in WEARER_ONLY
    pw = [0.0, 0.9, 0.9, 0.9, 0.9]           # one-frame dropout
    gain = g.run(pw, [0.0] * 5)
    assert np.allclose(gain, 1.0), "a 1-frame glitch escaped the debounce"


# ---------------------------------------------------------------------------
# crossfade / click-freedom
# ---------------------------------------------------------------------------
def test_crossfade_has_no_sample_level_step():
    """A gain envelope that jumps 0->1 between frames must still produce a
    continuous output: that continuity IS the absence of a click."""
    hop = 160
    audio = np.ones(hop * 4, dtype=np.float32)
    env = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    out = apply_gain_envelope(audio, env, hop)
    d = np.abs(np.diff(out))
    assert d.max() <= 1.0 / hop + 1e-6, f"gain step of {d.max():.4f} would click"


def test_crossfade_reaches_frame_gain_at_frame_end():
    hop = 100
    audio = np.ones(hop * 3, dtype=np.float32)
    env = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    out = apply_gain_envelope(audio, env, hop)
    assert out[2 * hop - 1] == pytest.approx(1.0, abs=1e-2)


def test_no_smoothing_policy_does_click():
    """Control: the E_no_smoothing policy is SUPPOSED to be abrupt at frame
    level -- but crossfade still removes the sample-level step. This test
    documents that crossfade is independent of attack/release."""
    p = _instant()
    g = GeoWearGate(p, sr=SR)
    pw = [0.0] * 10 + [0.9] * 10
    audio = np.ones(len(pw) * g.hop, dtype=np.float32)
    y, gain = g.apply(audio, pw, [0.0] * len(pw))
    assert np.abs(np.diff(y)).max() <= 1.0 / g.hop + 1e-6


# ---------------------------------------------------------------------------
# streaming == offline
# ---------------------------------------------------------------------------
def test_streaming_matches_offline_exactly():
    rng = np.random.default_rng(0)
    pw = rng.random(200).astype(np.float32)
    pe = rng.random(200).astype(np.float32)
    a = GeoWearGate(GatePolicy()).run(pw, pe)
    g = GeoWearGate(GatePolicy())
    b = np.array([g.process_frame(x, y) for x, y in zip(pw, pe)], dtype=np.float32)
    assert np.array_equal(a, b), "offline helper diverged from the streaming call"


def test_gate_is_causal():
    """Changing a FUTURE probability must never change a PAST gain."""
    rng = np.random.default_rng(1)
    pw = rng.random(100).astype(np.float32)
    pe = rng.random(100).astype(np.float32)
    a = GeoWearGate(GatePolicy()).run(pw, pe)
    pw2 = pw.copy()
    pw2[60:] = 1.0 - pw2[60:]
    b = GeoWearGate(GatePolicy()).run(pw2, pe)
    assert np.array_equal(a[:60], b[:60]), "gate used future information"


def test_reset_restores_initial_state():
    g = GeoWearGate(GatePolicy())
    g.run([0.9] * 20, [0.0] * 20)
    g.reset()
    assert g.state == SILENCE and g.gain == 0.0


# ---------------------------------------------------------------------------
# oracle gate + latency accounting
# ---------------------------------------------------------------------------
def test_oracle_gate_passes_wearer_and_mutes_bystander():
    hop = 160
    n = 50
    w = np.zeros(n, np.float32); w[:25] = 1.0
    e = np.zeros(n, np.float32); e[25:] = 1.0
    audio = np.ones(n * hop, dtype=np.float32)
    y, gain = oracle_gate(audio, w, e, GatePolicy(name="o", attack_ms=0.0, release_ms=0.0,
                                                  min_state_ms=0.0, hangover_ms=0.0))
    assert gain[:25].mean() > 0.95
    assert gain[30:].mean() < 0.05


def test_latency_accounting_excludes_gate_lookahead():
    lat = algorithmic_latency_ms(GatePolicy())
    assert lat["gate_lookahead_ms"] == 0.0, "the gate must never claim lookahead"
    assert lat["latency_algorithmic_ms"] == pytest.approx(25.0 + 10.0)
    # attack/release are trajectories, not delays -- reported separately
    assert "gate_full_open_ms" in lat and "gate_full_close_ms" in lat


def test_latency_includes_model_lookahead_when_present():
    lat = algorithmic_latency_ms(GatePolicy(), model_lookahead_ms=20.0)
    assert lat["latency_algorithmic_ms"] == pytest.approx(55.0)


def test_policy_variants_are_distinct_and_named():
    ps = policy_variants()
    names = [p.name for p in ps]
    assert len(names) == len(set(names)) and len(ps) >= 5
    assert all(p.overlap_policy in ("pass_raw", "mute", "attenuate", "wearer_dominant")
               for p in ps)
