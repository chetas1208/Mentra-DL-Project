"""P1.2 -- GeoWear Gate v0 ("Wearer-Routed Audio").  P1.3 -- Oracle gate.

WHAT THIS IS NOT
----------------
This is **not source separation**. It never estimates a mask over a mixture
and never tries to pull two overlapping voices apart. It is a *router*: a
streaming, per-frame gain applied to the ORIGINAL waveform, driven by
GeoWearNet's P(wearer) / P(environment). The name is deliberate -- do not
call it WearerSepNet. WearerSepNet is the (unbuilt, spec-only) separator
described in `docs/geowearnet_wearersepnet_spec.md`.

Consequence, stated up front because it bounds every number this produces:
in state 11 (wearer AND bystander speaking simultaneously) a gain-only
router has exactly two options -- pass the mixture (bystander leaks) or mute
it (wearer words are deleted). It cannot do better. Quantifying how much
product error is trapped in that state is P1.6's entire job, and it is what
decides whether WearerSepNet is worth building.

STATE MACHINE
-------------
Four decision states derived from the two probabilities per 10 ms frame:
    SILENCE            neither -> mute
    WEARER_ONLY        wearer only -> pass
    ENVIRONMENT_ONLY   environment only -> attenuate (mute by default)
    OVERLAP            both -> policy-configurable; PASS_RAW by default

Anti-chop machinery (all real, all tested in tests/agent_audio/):
    * hysteresis     separate on/off thresholds, so a probability hovering at
                     the threshold does not flap
    * attack/release exponential-ish ramp times for opening and closing the
                     gain, expressed in ms and converted to frames
    * min-state-ms   a state must persist this long before the gate is
                     allowed to leave it (debounce)
    * hangover-ms    keep passing for this long after wearer speech ends, so
                     word-final consonants are not clipped
    * crossfade      the per-frame gain is linearly interpolated across each
                     frame at sample rate, so no gain step ever lands inside
                     a sample block (that step IS the click)

Everything is causal and streaming: `GeoWearGate.process_frame` consumes one
frame's probabilities and returns that frame's gain, with no lookahead. The
offline helper `apply()` is a loop over exactly that call, so offline and
streaming results are identical by construction.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SILENCE, WEARER_ONLY, ENVIRONMENT_ONLY, OVERLAP = 0, 1, 2, 3
STATE_NAMES = {0: "SILENCE", 1: "WEARER_ONLY", 2: "ENVIRONMENT_ONLY", 3: "OVERLAP"}

OVERLAP_POLICIES = ("pass_raw", "mute", "attenuate", "wearer_dominant")


@dataclasses.dataclass
class GatePolicy:
    """One gate configuration. Every field is a real knob exercised by the
    policy sweep in `pipelines.py`."""
    name: str = "default"

    # hysteresis thresholds on P(wearer) / P(environment)
    wearer_on: float = 0.60
    wearer_off: float = 0.40
    env_on: float = 0.60
    env_off: float = 0.40

    # gains per state (linear, 0..1). env_gain<1 attenuates rather than mutes.
    wearer_gain: float = 1.0
    env_gain: float = 0.0
    silence_gain: float = 0.0

    # overlap handling -- the interesting policy axis
    overlap_policy: str = "pass_raw"
    overlap_gain: float = 1.0          # used by "attenuate"/"wearer_dominant"

    # temporal smoothing (ms)
    attack_ms: float = 10.0            # gain rising  (open fast: don't clip onsets)
    release_ms: float = 60.0           # gain falling (close slow: don't chop tails)
    min_state_ms: float = 40.0         # debounce: minimum dwell before switching
    hangover_ms: float = 200.0         # keep passing after wearer speech ends

    frame_hop_ms: float = 10.0

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


# A predeclared sweep. Chosen to span the real product trade-off (aggressive
# suppression vs wearer retention), not to hill-climb a single number.
def policy_variants() -> List[GatePolicy]:
    return [
        GatePolicy(name="A_balanced"),
        GatePolicy(name="B_aggressive", wearer_on=0.70, wearer_off=0.50,
                   env_gain=0.0, overlap_policy="mute", hangover_ms=100.0, release_ms=40.0),
        GatePolicy(name="C_conservative", wearer_on=0.45, wearer_off=0.30,
                   env_gain=0.15, overlap_policy="pass_raw", hangover_ms=350.0, release_ms=120.0),
        GatePolicy(name="D_overlap_attenuate", overlap_policy="attenuate", overlap_gain=0.35),
        GatePolicy(name="E_no_smoothing", attack_ms=0.0, release_ms=0.0,
                   min_state_ms=0.0, hangover_ms=0.0),
        GatePolicy(name="F_long_hangover", hangover_ms=400.0, release_ms=150.0),
    ]


class GeoWearGate:
    """Causal, streaming, per-frame gain router. No lookahead, no separation."""

    def __init__(self, policy: GatePolicy = None, sr: int = 16000):
        self.p = policy or GatePolicy()
        self.sr = sr
        self.hop = int(round(sr * self.p.frame_hop_ms / 1000.0))
        self.reset()

    # -- helpers -------------------------------------------------------------
    def _ms_to_frames(self, ms: float) -> int:
        return int(round(ms / max(self.p.frame_hop_ms, 1e-9)))

    def reset(self) -> None:
        self.state = SILENCE
        self.gain = float(self.p.silence_gain)
        self._wearer_latched = False
        self._env_latched = False
        self._frames_in_state = 0
        self._pending_state = None        # candidate state awaiting debounce
        self._pending_count = 0
        self._hangover_left = 0
        self.history: List[Dict[str, float]] = []

    # -- the state machine ---------------------------------------------------
    def _latch(self, p_w: float, p_e: float) -> Tuple[bool, bool]:
        """Schmitt-trigger latching. A latched signal stays on until it drops
        below the OFF threshold; an unlatched one stays off until it exceeds
        the ON threshold."""
        if self._wearer_latched:
            self._wearer_latched = p_w > self.p.wearer_off
        else:
            self._wearer_latched = p_w >= self.p.wearer_on
        if self._env_latched:
            self._env_latched = p_e > self.p.env_off
        else:
            self._env_latched = p_e >= self.p.env_on
        return self._wearer_latched, self._env_latched

    def _target_state(self, w: bool, e: bool) -> int:
        if w and e:
            return OVERLAP
        if w:
            return WEARER_ONLY
        if e:
            return ENVIRONMENT_ONLY
        return SILENCE

    def _state_gain(self, state: int) -> float:
        if state == WEARER_ONLY:
            return self.p.wearer_gain
        if state == ENVIRONMENT_ONLY:
            return self.p.env_gain
        if state == SILENCE:
            return self.p.silence_gain
        # OVERLAP
        pol = self.p.overlap_policy
        if pol == "pass_raw":
            return self.p.wearer_gain
        if pol == "mute":
            return self.p.silence_gain
        if pol in ("attenuate", "wearer_dominant"):
            return self.p.overlap_gain
        raise ValueError(f"unknown overlap_policy {pol!r}")

    def _ramp(self, target: float) -> float:
        """One-pole-ish linear ramp toward `target`, with separate attack and
        release rates. Rate is expressed as full-scale-per-N-frames so
        attack_ms=0 gives an instantaneous jump (used by the E_no_smoothing
        control policy)."""
        if target > self.gain:
            n = self._ms_to_frames(self.p.attack_ms)
        else:
            n = self._ms_to_frames(self.p.release_ms)
        if n <= 0:
            return float(target)
        step = 1.0 / n
        if target > self.gain:
            return float(min(target, self.gain + step))
        return float(max(target, self.gain - step))

    def process_frame(self, p_wearer: float, p_env: float) -> float:
        """Consume one frame's probabilities, return that frame's END gain.
        Strictly causal."""
        w, e = self._latch(float(p_wearer), float(p_env))
        desired = self._target_state(w, e)

        # hangover: after wearer speech stops, keep behaving as WEARER_ONLY
        if desired in (WEARER_ONLY, OVERLAP):
            self._hangover_left = self._ms_to_frames(self.p.hangover_ms)
        elif self._hangover_left > 0:
            self._hangover_left -= 1
            if desired == SILENCE:
                desired = WEARER_ONLY   # hold the gate open through the tail
            # NOTE: hangover deliberately does NOT override ENVIRONMENT_ONLY --
            # holding the gate open into a bystander's turn is exactly the
            # leakage we are trying to prevent.

        # min-state debounce: a CANDIDATE state must persist for min_state_ms
        # before it is committed. (Debouncing the *new* state, not the old
        # one -- gating on how long we had been in the previous state lets a
        # single-frame glitch flip the gate and then traps it there, which is
        # exactly the chopping this knob exists to prevent.)
        min_frames = self._ms_to_frames(self.p.min_state_ms)
        if desired != self.state:
            if desired == self._pending_state:
                self._pending_count += 1
            else:
                self._pending_state, self._pending_count = desired, 1
            if self._pending_count >= max(min_frames, 1):
                self.state = desired
                self._frames_in_state = 0
                self._pending_state, self._pending_count = None, 0
            else:
                self._frames_in_state += 1
        else:
            self._pending_state, self._pending_count = None, 0
            self._frames_in_state += 1

        self.gain = self._ramp(self._state_gain(self.state))
        return self.gain

    # -- offline driver (identical maths, just a loop) -----------------------
    def run(self, p_wearer: Sequence[float], p_env: Sequence[float]) -> np.ndarray:
        """Returns the per-frame gain envelope. Same call path as streaming."""
        n = min(len(p_wearer), len(p_env))
        g = np.empty(n, dtype=np.float32)
        st = np.empty(n, dtype=np.int64)
        for i in range(n):
            g[i] = self.process_frame(p_wearer[i], p_env[i])
            st[i] = self.state
        self.frame_states = st
        return g

    def apply(self, audio: np.ndarray, p_wearer: Sequence[float],
              p_env: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Gate `audio` and return (processed_audio, per_frame_gain).

        The gain is CROSSFADED at sample rate across each frame (linear
        interpolation from the previous frame's gain to this frame's), so the
        output never contains a gain discontinuity inside a block. That is the
        difference between "gated" and "clicky"."""
        g = self.run(p_wearer, p_env)
        return apply_gain_envelope(audio, g, self.hop), g


def apply_gain_envelope(audio: np.ndarray, frame_gain: np.ndarray, hop: int,
                        start_gain: Optional[float] = None) -> np.ndarray:
    """Sample-rate crossfade of a per-frame gain envelope onto a waveform.

    Frame i's gain is reached at the END of frame i; within frame i the gain
    ramps linearly from frame i-1's value. No zero-order hold, hence no step
    discontinuity, hence no click.
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    n_frames = len(frame_gain)
    n = min(len(audio), n_frames * hop)
    out = np.zeros(len(audio), dtype=np.float32)
    prev = float(frame_gain[0] if start_gain is None else start_gain)
    ramp = np.arange(hop, dtype=np.float32) / max(hop, 1)
    for i in range(n_frames):
        a, b = i * hop, min((i + 1) * hop, n)
        if a >= n:
            break
        cur = float(frame_gain[i])
        seg = prev + (cur - prev) * ramp[: b - a]
        out[a:b] = audio[a:b] * seg
        prev = cur
    if n < len(audio):           # trailing partial frame: hold last gain
        out[n:] = audio[n:] * prev
    return out


# ---------------------------------------------------------------------------
# P1.3 -- ORACLE GATE
# ---------------------------------------------------------------------------
def oracle_probabilities(wearer_active: np.ndarray, env_active: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Ground-truth activity as if it were a perfect detector's output.

    This is the whole point of P1.3: feed the SAME gate the SAME policy with
    perfect probabilities, so the ONLY difference between the oracle result
    and the predicted result is detector error. Everything else -- gate
    smoothing, hangover, crossfade, ASR, metric definitions -- is held
    identical. The oracle number is therefore a genuine upper bound on what
    detection-plus-routing can deliver, and the oracle-vs-predicted gap is a
    genuine measurement of how much is lost to detector error rather than to
    the routing architecture.
    """
    return (np.asarray(wearer_active, dtype=np.float32),
            np.asarray(env_active, dtype=np.float32))


def oracle_gate(audio: np.ndarray, wearer_active: np.ndarray, env_active: np.ndarray,
                policy: GatePolicy = None, sr: int = 16000
                ) -> Tuple[np.ndarray, np.ndarray]:
    pw, pe = oracle_probabilities(wearer_active, env_active)
    g = GeoWearGate(policy or GatePolicy(name="oracle"), sr=sr)
    return g.apply(audio, pw, pe)


# ---------------------------------------------------------------------------
# latency accounting
# ---------------------------------------------------------------------------
def algorithmic_latency_ms(policy: GatePolicy, feature_window_ms: float = 25.0,
                           model_lookahead_ms: float = 0.0) -> Dict[str, float]:
    """Honest latency accounting for the gate path.

    The gate itself adds ZERO lookahead -- it is strictly causal. The real
    algorithmic latency of the routed path is the feature window (25 ms, the
    log-mel analysis window GeoWearNet uses) plus one frame hop, plus any
    model lookahead (0 for the causal GeoWearNet architectures). Attack and
    release are NOT latency -- they are gain trajectories, not delays -- but
    `attack_ms` does delay the moment the gate is fully open, so it is
    reported separately as `gate_full_open_ms` rather than hidden or
    double-counted as latency.
    """
    base = feature_window_ms + policy.frame_hop_ms + model_lookahead_ms
    return {
        "latency_algorithmic_ms": base,
        "feature_window_ms": feature_window_ms,
        "frame_hop_ms": policy.frame_hop_ms,
        "model_lookahead_ms": model_lookahead_ms,
        "gate_lookahead_ms": 0.0,
        "gate_full_open_ms": policy.attack_ms,
        "gate_full_close_ms": policy.release_ms,
    }
