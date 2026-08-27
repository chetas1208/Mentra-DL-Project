"""Streaming audio-policy boundary for the Mentra receiver.

This module deliberately implements routing, not source separation. The
GeoWear gate can preserve or mute a frame, but it cannot disentangle two
voices in an overlap frame. Denoising is injected so the server remains
independent of a particular RNNoise binding and is easy to test.
"""
from __future__ import annotations

import dataclasses
from typing import Literal, Optional, Protocol

import numpy as np


AudioPolicy = Literal["passthrough", "geowear_gate", "rnnoise_geowear_gate",
                      "geowear_envelope", "rnnoise_geowear_envelope"]
SUPPORTED_POLICIES = ("passthrough", "geowear_gate", "rnnoise_geowear_gate",
                      "geowear_envelope", "rnnoise_geowear_envelope")

#: Policies that route audio through the full offline-equivalent gate
#: envelope (G4 WS1) instead of the coarse binary state mute that G3 shipped.
ENVELOPE_POLICIES = ("geowear_envelope", "rnnoise_geowear_envelope")
_DENOISE_POLICIES = ("rnnoise_geowear_gate", "rnnoise_geowear_envelope")
_ACTIVE_STATES = frozenset(("WEARER", "OVERLAP"))


class FrameDenoiser(Protocol):
    def process(self, audio: np.ndarray):
        """Return an object with an audio ndarray and optional availability."""

    def reset(self) -> None:
        ...


@dataclasses.dataclass(frozen=True)
class FrontendResult:
    audio: np.ndarray
    policy: AudioPolicy
    gated: bool
    denoised: bool
    #: Envelope policies only: mean gate gain actually applied to this frame,
    #: and the gate's state at the end of it. ``None`` for the binary policies
    #: so a caller can never mistake a coarse mute for an envelope decision.
    gain: Optional[float] = None
    gate_state: Optional[str] = None


class AudioFrontend:
    """Apply one explicit, causal policy to one mono float32 frame.

    Two families of gate policy exist, and the difference is deliberate:

    ``geowear_gate`` / ``rnnoise_geowear_gate`` (G3 behaviour, preserved)
        A binary decision driven by the detector's latest state: WEARER and
        OVERLAP pass, ENVIRONMENT and SILENCE mute. No ramp, no hangover, no
        crossfade. Kept unchanged so previously reported live behaviour stays
        reproducible.

    ``geowear_envelope`` / ``rnnoise_geowear_envelope`` (G4 WS1)
        The offline/product evaluator's own gate envelope -- hysteresis,
        attack/release, min-state debounce, hangover, sample-rate crossfade
        and an optional bounded pre-roll -- via
        ``server.audio.streaming_gate.StreamingGateRouter``. This is the
        policy that actually matches the numbers in the product matrix.

    Passing OVERLAP preserves the product policy's default no-separation
    behavior: the original mixed PCM remains mixed. This is classified/gated
    audio, not source separation.
    """

    def __init__(self, policy: AudioPolicy = "passthrough",
                 denoiser: FrameDenoiser | None = None,
                 router: "object | None" = None):
        if policy not in SUPPORTED_POLICIES:
            raise ValueError(
                f"unsupported audio policy {policy!r}; choose one of {SUPPORTED_POLICIES}"
            )
        if policy in _DENOISE_POLICIES and denoiser is None:
            raise ValueError(f"{policy} requires an injected denoiser")
        if policy in ENVELOPE_POLICIES:
            if router is None:
                from server.audio.streaming_gate import StreamingGateRouter
                router = StreamingGateRouter()
        elif router is not None:
            raise ValueError(
                f"policy {policy!r} does not use a streaming gate router; "
                f"choose one of {ENVELOPE_POLICIES}"
            )
        self.policy = policy
        self.denoiser = denoiser
        self.router = router

    def reset(self) -> None:
        if self.denoiser is not None:
            reset = getattr(self.denoiser, "reset", None)
            if reset is not None:
                reset()
        if self.router is not None:
            self.router.reset()

    def update_probabilities(self, p_wearer: float, p_env: float) -> None:
        """Publish a new detector observation to the envelope router.

        A no-op for the binary policies, which read the detector state on
        every frame instead."""
        if self.router is not None:
            self.router.update_probabilities(p_wearer, p_env)

    def process(self, audio: np.ndarray, state: str) -> FrontendResult:
        x = np.ascontiguousarray(audio, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("AudioFrontend expects a mono 1-D frame")
        if not np.isfinite(x).all():
            raise ValueError("AudioFrontend input contains NaN or Inf")
        if state not in {"WEARER", "ENVIRONMENT", "SILENCE", "OVERLAP"}:
            raise ValueError(f"unknown detector state {state!r}")

        denoised = False
        if self.policy in _DENOISE_POLICIES:
            assert self.denoiser is not None
            result = self.denoiser.process(x)
            if getattr(result, "available", True) is False:
                raise RuntimeError("RNNoise policy selected but the denoiser is unavailable")
            x = np.ascontiguousarray(result.audio, dtype=np.float32)
            if x.shape != audio.shape:
                raise ValueError("denoiser changed the frame shape")
            denoised = True

        if self.policy in ENVELOPE_POLICIES:
            assert self.router is not None
            if not self.router.stats()["have_probabilities"]:
                # No detector observation has arrived yet on this stream. Fall
                # back to the state-derived probabilities rather than silently
                # holding SILENCE through a stream's opening frames.
                self.router.update_from_state(state)
            routed, info = self.router.process(x)
            if routed.size != x.size:
                raise RuntimeError(
                    "streaming gate router returned a different frame length; the live "
                    "transport must deliver whole 10 ms frames"
                )
            return FrontendResult(
                audio=routed, policy=self.policy, gated=True, denoised=denoised,
                gain=float(np.mean(info.gains)) if info.frames_emitted else float(self.router.gain),
                gate_state=self.router.state_name,
            )

        gated = self.policy != "passthrough"
        if gated and state not in _ACTIVE_STATES:
            x = np.zeros_like(x)
        return FrontendResult(audio=x, policy=self.policy, gated=gated, denoised=denoised)
