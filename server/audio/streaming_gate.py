"""G4 WS1 -- the real streaming gate envelope for the LIVE receiver.

WHY THIS FILE EXISTS
--------------------
G3 measured and honestly reported a discrepancy: the offline/product
evaluator routes audio with ``evaluation.agent_audio.gate.GeoWearGate`` --
10 ms frames, hysteresis, attack/release ramps, min-state debounce,
hangover, and a sample-rate crossfade -- while the live receiver applied a
*binary* mute/pass decision derived from ``MentraInferenceConsumer``'s
200 ms state, with no ramp, no hangover and no crossfade. Every product
number in the G3 report therefore described a router the live path did not
actually implement.

This module closes that gap by **reusing the evaluator's own state machine**
rather than reimplementing it. ``StreamingGateRouter`` wraps the exact same
``GeoWearGate`` object and the exact same ``apply_gain_envelope`` crossfade
that produced the offline numbers, and drives it from a live PCM stream that
arrives in blocks and whose probabilities update on a slower cadence than
the gate's 10 ms frame rate.

Consequence, stated precisely because it is the whole point of WS1: when the
router is fed the same probability sequence as the offline evaluator and
``preroll_ms=0``, its output is **bit-identical** to
``GeoWearGate.apply()``. That is asserted by tests, not assumed.

THE TWO REAL, MEASURED DIFFERENCES THAT REMAIN
----------------------------------------------
1. **Probability cadence.** Offline, GeoWearNet is run over a whole
   recording and yields a dense probability per 10 ms frame. Live, the
   detector re-scores a rolling window every ``hop_s`` (200 ms by default)
   and only the final frame's logit is available. Between updates this
   router holds the last probabilities (zero-order hold). That is a genuine
   information difference, not an implementation bug, and it is quantified
   in ``training/geowearnet/g4/parity.py`` rather than hand-waved.

2. **Pre-roll.** Optional, bounded, and off by default. See below.

PRE-ROLL (WS2)
--------------
A gate can only open *after* the detector has heard enough of an onset to
be confident, so the first phoneme of the first word is systematically the
part most likely to be attenuated. The fix is a bounded delay line: the
router emits audio ``preroll_ms`` behind real time, so the gain decided at
time *t* is applied to audio captured at *t - preroll_ms*.

This is honest added latency and is reported as such
(``algorithmic_delay_ms``). It is NOT lookahead into future audio: no sample
is ever consulted before it would have been available. The delay line is
primed with silence at reset so output length always equals input length.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np

from evaluation.agent_audio.gate import (STATE_NAMES, GatePolicy, GeoWearGate,
                                         apply_gain_envelope, policy_variants)


def named_gate_policy(name: str) -> GatePolicy:
    """Look up one of the evaluator's predeclared gate policies by name.

    Deliberately restricted to the sweep the offline product matrix already
    measured, so the live receiver cannot be started with an envelope nobody
    has numbers for."""
    variants = {p.name: p for p in policy_variants()}
    if name not in variants:
        raise ValueError(
            f"unknown gate policy {name!r}; choose one of {sorted(variants)}"
        )
    return variants[name]

# Bounded by design: a "pre-roll" long enough to matter for a first word is
# short. Anything beyond this is a buffering decision that belongs to the
# product, not a gate parameter, and would silently inflate agent latency.
MAX_PREROLL_MS = 400.0

#: Live detector states (MentraInferenceConsumer vocabulary) mapped onto the
#: hard probabilities the gate would have seen. Used only when a detector
#: exposes a state but no environment probability (legacy SpeakerNet path).
STATE_TO_PROBABILITIES = {
    "WEARER": (1.0, 0.0),
    "ENVIRONMENT": (0.0, 1.0),
    "OVERLAP": (1.0, 1.0),
    "SILENCE": (0.0, 0.0),
}


@dataclasses.dataclass(frozen=True)
class RouterFrameInfo:
    """What the router did to one call's worth of audio."""
    frames_emitted: int
    gains: np.ndarray
    states: np.ndarray
    gain_first: float
    gain_last: float
    held_probability_frames: int


class StreamingGateRouter:
    """Causal, streaming, sample-accurate gate envelope for the live path.

    Contract
    --------
    * ``update_probabilities`` is called whenever the detector produces a new
      score (every ``hop_s`` in the live receiver). Between calls the last
      values are held.
    * ``process`` is called with each arriving PCM block. The block length
      MUST be a whole number of gate frames (10 ms = 160 samples at 16 kHz);
      the live transport already frames at exactly 10 ms. A non-aligned block
      raises rather than silently drifting the envelope.
    * Output length always equals input length.
    """

    def __init__(self, policy: Optional[GatePolicy] = None, sr: int = 16000,
                 preroll_ms: float = 0.0):
        if sr <= 0:
            raise ValueError("sample rate must be positive")
        if preroll_ms < 0.0:
            raise ValueError("preroll_ms must be >= 0")
        if preroll_ms > MAX_PREROLL_MS:
            raise ValueError(
                f"preroll_ms={preroll_ms} exceeds the bounded maximum {MAX_PREROLL_MS} ms; "
                "a longer buffer is a product latency decision, not a gate parameter"
            )
        self.policy = policy or GatePolicy()
        self.sr = int(sr)
        self.gate = GeoWearGate(self.policy, sr=self.sr)
        self.hop = self.gate.hop
        if self.hop <= 0:
            raise ValueError("gate hop resolved to zero samples")

        # Pre-roll is quantised to whole gate frames so the delay line and the
        # gain envelope stay index-aligned. The *effective* value is reported,
        # never the requested one, so a caller cannot believe in a delay the
        # router did not actually apply.
        self.preroll_frames = int(round(preroll_ms / self.policy.frame_hop_ms))
        self.preroll_ms = self.preroll_frames * self.policy.frame_hop_ms
        self.preroll_samples = self.preroll_frames * self.hop

        self.reset()

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> None:
        """Forget all stream-local state. No wearer identity exists to keep."""
        self.gate.reset()
        # Delay line primed with silence: keeps output length == input length
        # from the very first block, at the cost of preroll_ms of leading
        # silence, which is exactly the delay we are declaring.
        self._buffer = np.zeros(self.preroll_samples, dtype=np.float32)
        self._prev_gain: Optional[float] = None
        self._p_wearer = 0.0
        self._p_env = 0.0
        self._have_probabilities = False
        self._samples_in = 0
        self._frames_scored = 0
        self._frames_emitted = 0
        self._frames_since_update = 0
        self._held_frames_total = 0
        self._gains_pending: List[float] = []
        self._states_pending: List[int] = []

    # -- probability input ---------------------------------------------------
    def update_probabilities(self, p_wearer: float, p_env: float) -> None:
        """Publish a new detector observation. Held until the next update."""
        pw, pe = float(p_wearer), float(p_env)
        if not (np.isfinite(pw) and np.isfinite(pe)):
            raise ValueError("gate probabilities must be finite")
        self._p_wearer = float(np.clip(pw, 0.0, 1.0))
        self._p_env = float(np.clip(pe, 0.0, 1.0))
        self._have_probabilities = True
        self._frames_since_update = 0

    def update_from_state(self, state: str) -> None:
        """Fallback for detectors that expose a state but no environment
        probability (the legacy SpeakerNet path). Maps the state onto the
        hard probabilities the gate would have latched. Documented as a
        fallback so a state-only detector is never mistaken for a real
        two-probability GeoWearNet observation."""
        try:
            pw, pe = STATE_TO_PROBABILITIES[state]
        except KeyError:
            raise ValueError(f"unknown detector state {state!r}") from None
        self.update_probabilities(pw, pe)

    # -- audio ---------------------------------------------------------------
    def process(self, audio: np.ndarray) -> tuple[np.ndarray, RouterFrameInfo]:
        x = np.ascontiguousarray(audio, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("StreamingGateRouter expects a mono 1-D frame")
        if not np.isfinite(x).all():
            raise ValueError("StreamingGateRouter input contains NaN or Inf")
        if x.size % self.hop != 0:
            raise ValueError(
                f"block of {x.size} samples is not a whole number of {self.hop}-sample "
                f"({self.policy.frame_hop_ms} ms) gate frames; the live transport frames "
                "at 10 ms and the envelope must stay frame-aligned"
            )

        self._buffer = np.concatenate([self._buffer, x])
        self._samples_in += x.size

        # 1. score every gate frame whose audio has now fully arrived
        self._gains_pending.clear()
        self._states_pending.clear()
        held = 0
        while (self._frames_scored + 1) * self.hop <= self._samples_in:
            gain = self.gate.process_frame(self._p_wearer, self._p_env)
            self._gains_pending.append(gain)
            self._states_pending.append(self.gate.state)
            self._frames_scored += 1
            if self._frames_since_update > 0:
                held += 1
            self._frames_since_update += 1
        self._held_frames_total += held

        n_new = len(self._gains_pending)
        if n_new == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, RouterFrameInfo(0, empty, np.zeros(0, dtype=np.int64),
                                          float(self.gate.gain), float(self.gate.gain), held)

        # 2. emit exactly those frames' worth of DELAYED audio. Frame k's gain
        # is applied to the audio sitting preroll_samples behind it, which is
        # the front of the buffer by construction.
        gains = np.asarray(self._gains_pending, dtype=np.float32)
        n_out = n_new * self.hop
        block = self._buffer[:n_out]
        self._buffer = self._buffer[n_out:]
        out = apply_gain_envelope(block, gains, self.hop, start_gain=self._prev_gain)
        self._prev_gain = float(gains[-1])
        self._frames_emitted += n_new

        return out, RouterFrameInfo(
            frames_emitted=n_new,
            gains=gains,
            states=np.asarray(self._states_pending, dtype=np.int64),
            gain_first=float(gains[0]),
            gain_last=float(gains[-1]),
            held_probability_frames=held,
        )

    # -- introspection -------------------------------------------------------
    @property
    def state_name(self) -> str:
        return STATE_NAMES[self.gate.state]

    @property
    def gain(self) -> float:
        return float(self.gate.gain)

    def latency_report(self, feature_window_ms: float = 25.0) -> dict:
        """Honest, non-double-counted delay accounting for the live route.

        ``preroll_ms`` IS added algorithmic delay and is reported as such.
        ``attack_ms``/``release_ms`` are gain trajectories, not delays, and are
        reported separately (same convention as
        ``evaluation.agent_audio.gate.algorithmic_latency_ms``)."""
        base = feature_window_ms + self.policy.frame_hop_ms
        return {
            "algorithmic_delay_ms": base + self.preroll_ms,
            "feature_window_ms": feature_window_ms,
            "frame_hop_ms": self.policy.frame_hop_ms,
            "preroll_ms": self.preroll_ms,
            "preroll_frames": self.preroll_frames,
            "gate_lookahead_ms": 0.0,
            "gate_full_open_ms": self.policy.attack_ms,
            "gate_full_close_ms": self.policy.release_ms,
            "hangover_ms": self.policy.hangover_ms,
        }

    def stats(self) -> dict:
        return {
            "samples_in": self._samples_in,
            "frames_scored": self._frames_scored,
            "frames_emitted": self._frames_emitted,
            "held_probability_frames": self._held_frames_total,
            "buffered_samples": int(self._buffer.size),
            "have_probabilities": self._have_probabilities,
        }
