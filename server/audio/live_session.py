"""Per-connection live session: one bound model, one set of stream-local state.

This is the seam that makes the receiver multi-model WITHOUT duplicating the
audio pipeline. There is exactly one transport, one jitter buffer, one
AudioFrame contract, one audio frontend implementation and one ASR runtime;
what varies per session is which ``SessionRuntime`` (and therefore which
model) the frames are scored by.

Everything with stream lifetime lives on the session object -- rolling PCM and
hysteresis latches (inside the SessionRuntime's consumer), the gate/RNNoise
envelope, the ASR stream, the last transcript, the playback-mode preference,
and for SpeakerNet the session's own enrolled embedding. Two concurrent
sessions therefore cannot see each other's model choice, scores, latches or
identity, and a session reset cannot leave a previous wearer's state behind.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from mentra.audio.consumer import pcm16_bytes_to_float32
from mentra.audio.frame import AudioFrame, FrameFlags, MessageType
from server.audio.frontend import AudioFrontend, ENVELOPE_POLICIES
from server.models.registry import ModelRegistry, SessionRuntime

SAMPLE_RATE = 16_000
PLAYBACK_MODES = ("both", "wearer", "environment", "muted")


def float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    """Encode mono float32 audio without changing frame length."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("expected a finite mono float32 array")
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


@dataclass
class LiveSessionConfig:
    """Everything a session needs that is NOT the model."""

    audio_policy: str = "passthrough"
    gate_policy_name: str = "A_balanced"
    gate_preroll_ms: float = 0.0
    #: Called once per session that needs denoising. A factory (not a shared
    #: instance) because RNNoise carries per-stream filter state.
    denoiser_factory: Optional[Callable[[], object]] = None
    #: Shared, immutable ASR model; each session still creates its own stream.
    asr_recognizer: object | None = None
    capture_sink: object | None = None
    log: Optional[Callable[[str], None]] = None
    enable_transcript: bool = True


class LiveSession:
    def __init__(self, registry: ModelRegistry, config: LiveSessionConfig,
                 session_id: str = "", requested_model: Optional[str] = None):
        self.registry = registry
        self.config = config
        self.session_id = session_id
        self.runtime: Optional[SessionRuntime] = None
        self.bind_error: Optional[str] = None
        self.model_id: str = registry.default_model
        self.frames_scored = 0
        self.inference_ms_samples: List[float] = []

        self.frontend: AudioFrontend | None = None
        self.router = None
        self._denoiser = None
        self.asr_stream = None
        self.last_transcript = ""
        self.playback_mode = "both"
        self._sequence = 0

        self._build_frontend()
        self._bind(requested_model or registry.default_model, initial=True)

    # -- logging -----------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.config.log is not None:
            self.config.log(message)

    # -- construction ------------------------------------------------------
    def _build_frontend(self) -> None:
        policy = self.config.audio_policy
        if policy in ("rnnoise_geowear_gate", "rnnoise_geowear_envelope"):
            if self.config.denoiser_factory is None:
                raise ValueError(f"audio policy {policy!r} requires a denoiser factory")
            self._denoiser = self.config.denoiser_factory()
            if getattr(self._denoiser, "available", True) is False:
                raise RuntimeError(
                    f"audio policy {policy!r} requested, but the RNNoise binding is unavailable")
        if policy in ENVELOPE_POLICIES:
            from server.audio.streaming_gate import StreamingGateRouter, named_gate_policy

            self.router = StreamingGateRouter(
                named_gate_policy(self.config.gate_policy_name), sr=SAMPLE_RATE,
                preroll_ms=self.config.gate_preroll_ms)
        self.frontend = AudioFrontend(policy, denoiser=self._denoiser, router=self.router)
        self.frontend.reset()

    def _new_asr_stream(self):
        recognizer = self.config.asr_recognizer
        return recognizer.create_stream() if recognizer is not None else None

    # -- model binding -----------------------------------------------------
    def _bind(self, model_id: str, initial: bool = False) -> None:
        """Bind ONE model to this session, dropping all stream-local state.

        Never hot-swaps into a running model's state: a rebind always builds a
        brand new SessionRuntime (fresh rolling PCM, fresh latches, fresh
        SpeakerNet embedding view) and resets the gate/ASR/transcript, so no
        TCN context, enrollment or envelope state can cross a model change.
        """
        resolved, error = self.registry.resolve_requested_model(model_id)
        if error is not None or resolved is None:
            self.bind_error = error
            if initial:
                self.model_id = model_id
                self.runtime = None
            return
        self.runtime = self.registry.create_session_runtime(resolved)
        self.model_id = resolved
        self.bind_error = None
        self.reset(f"model bound to {resolved}")

    def select_model(self, model_id: str) -> Dict[str, object]:
        """Handle a client's session_config. Returns the ack dictionary."""
        previous = self.model_id
        self._bind(model_id)
        if self.bind_error is not None:
            return {
                "type": "session_config_ack",
                "model": previous,
                "requestedModel": model_id,
                "ready": self.registry.is_ready(previous),
                "error": self.bind_error,
                "capabilities": json.loads(
                    self.registry.capabilities_for(previous).to_payload())
                if previous in self.registry.models else None,
            }
        assert self.runtime is not None
        self._log(f"session {self.session_id or '-'}: model bound to {self.model_id} "
                  f"(requiresEnrollment={self.runtime.capabilities.requiresEnrollment})")
        return {
            "type": "session_config_ack",
            "model": self.model_id,
            "requestedModel": model_id,
            "ready": True,
            "error": None,
            "capabilities": json.loads(self.runtime.capabilities.to_payload()),
        }

    def on_session_config(self, payload: bytes) -> bytes:
        try:
            request = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return json.dumps({
                "type": "session_config_ack", "model": self.model_id,
                "ready": self.registry.is_ready(self.model_id),
                "error": f"malformed session_config: {exc}", "capabilities": None,
            }).encode("utf-8")
        requested = request.get("model")
        if requested is None:
            # A config that carries no model (e.g. audio-policy-only) leaves the
            # binding untouched rather than silently resetting to the default.
            ack = {
                "type": "session_config_ack", "model": self.model_id,
                "requestedModel": None,
                "ready": self.registry.is_ready(self.model_id), "error": None,
                "capabilities": json.loads(
                    self.registry.capabilities_for(self.model_id).to_payload())
                if self.model_id in self.registry.models else None,
            }
        else:
            ack = self.select_model(str(requested))
        return json.dumps(ack).encode("utf-8")

    def capabilities_payload(self) -> bytes:
        """STREAM_ACCEPTED payload: the capabilities of the model this session
        is ACTUALLY bound to right now (the default, before any explicit
        selection) -- never a guess and never another session's model."""
        if self.runtime is not None:
            return self.runtime.capabilities.to_payload()
        if self.model_id in self.registry.models:
            return self.registry.capabilities_for(self.model_id).to_payload()
        return b""

    # -- lifecycle ---------------------------------------------------------
    def reset(self, reason: str) -> None:
        """Shared-glasses boundary: drop every stream-local value.

        Rolling PCM, detector latches, RNNoise state, gate envelope state, ASR
        stream, transcript deduplication and playback-mode preference are all
        fresh afterwards. No user identifier or learned identity is stored.
        """
        if self.runtime is not None:
            self.runtime.reset()
        if self.frontend is not None:
            self.frontend.reset()
        self.asr_stream = self._new_asr_stream()
        self.last_transcript = ""
        self.playback_mode = "both"
        self.frames_scored = 0
        self.inference_ms_samples = []
        self._log(f"stream state reset ({reason}); no wearer identity retained")

    def close(self, reason: str) -> None:
        sink = self.config.capture_sink
        if sink is not None:
            sink.close(reason)
        self.reset(reason)

    # -- transport callbacks ----------------------------------------------
    def on_playback_mode(self, mode: str) -> None:
        if mode not in PLAYBACK_MODES:
            self._log(f"ignoring unknown playback mode {mode!r}")
            return
        self.playback_mode = mode
        self._log(f"playback mode set to {mode!r} -- server will stop sending "
                  f"the unselected stream(s) entirely")

    def on_enroll(self, frame: AudioFrame) -> None:
        if self.runtime is None:
            self._log("enrollment ignored: no model bound to this session")
            return
        if not self.runtime.capabilities.supportsEnrollment:
            # GeoWearNet is enrollment-free. Never fabricate an enrollment
            # lifecycle for it; a client that sends one is simply told nothing
            # changed rather than silently corrupting the session.
            self._log(f"enrollment ignored: {self.runtime.display_name} is enrollment-free")
            return
        samples = pcm16_bytes_to_float32(frame.payload)
        self.runtime.enroll([(samples, SAMPLE_RATE)])
        self._log(f"REAL ENROLLMENT received: {len(samples) / SAMPLE_RATE:.1f}s of audio -- "
                  f"wearer embedding replaced for THIS session only")

    def on_frame(self, frame: AudioFrame) -> Optional[List[AudioFrame]]:
        if frame.sample_rate != SAMPLE_RATE or frame.bits_per_sample != 16 or frame.channels != 1:
            return None
        if self.runtime is None:
            return None
        if frame.flags & FrameFlags.DISCONTINUITY:
            self.reset("transport discontinuity")

        responses: List[AudioFrame] = []
        samples_f32 = pcm16_bytes_to_float32(frame.payload)

        if self.config.capture_sink is not None:
            self.config.capture_sink.on_derived_frame(frame.payload)

        # 1. Detection -- one model, chosen by THIS session.
        result = self.runtime.infer(frame)
        if result:
            self.frames_scored += 1
            self.inference_ms_samples.append(result.inference_ms)
            if self.router is not None:
                if result.environment_score is not None:
                    self.router.update_probabilities(result.wearer_score, result.environment_score)
                else:
                    # Legacy single-score detector: no environment evidence
                    # exists, so the envelope is driven from the state the
                    # detector's own hysteresis produced.
                    self.router.update_from_state(result.state)
            self._sequence += 1
            responses.append(AudioFrame(
                sequence_number=0, capture_timestamp_ns=frame.capture_timestamp_ns,
                sample_rate=SAMPLE_RATE, channels=1, bits_per_sample=16,
                payload=self._detection_payload(result),
                message_type=MessageType.DETECTION,
            ))

        # 2. Gated audio -- every frame. NOT source separation: this is the
        # original mixed PCM, passed through when the current state matches and
        # silence otherwise (see docs/MENTRA_REMOTE_AUDIO.md).
        state = self.runtime.current_state
        frontend_result = self.frontend.process(samples_f32, state)
        frontend_payload = float32_to_pcm16_bytes(frontend_result.audio)
        silence_payload = b"\x00" * len(frame.payload)
        if self.playback_mode in ("both", "wearer"):
            wearer_payload = (
                frontend_payload if frontend_result.gated
                else frame.payload if state == "WEARER"
                else silence_payload
            )
            responses.append(AudioFrame(
                sequence_number=frame.sequence_number,
                capture_timestamp_ns=frame.capture_timestamp_ns,
                sample_rate=SAMPLE_RATE, channels=1, bits_per_sample=16,
                payload=wearer_payload, message_type=MessageType.WEARER_PCM,
            ))
        if self.playback_mode in ("both", "environment"):
            env_payload = frame.payload if state == "ENVIRONMENT" else silence_payload
            responses.append(AudioFrame(
                sequence_number=frame.sequence_number,
                capture_timestamp_ns=frame.capture_timestamp_ns,
                sample_rate=SAMPLE_RATE, channels=1, bits_per_sample=16,
                payload=env_payload, message_type=MessageType.ENVIRONMENT_PCM,
            ))

        # 3. ASR -- default passthrough remains continuous for backward
        # compatibility; a gated policy feeds the same processed frame.
        if self.asr_stream is not None:
            asr_samples = frontend_result.audio if frontend_result.gated else samples_f32
            recognizer = self.config.asr_recognizer
            self.asr_stream.accept_waveform(SAMPLE_RATE, asr_samples)
            while recognizer.is_ready(self.asr_stream):
                recognizer.decode_stream(self.asr_stream)
            text = recognizer.get_result(self.asr_stream)
            if text != self.last_transcript:
                self.last_transcript = text
                responses.append(AudioFrame(
                    sequence_number=0, capture_timestamp_ns=frame.capture_timestamp_ns,
                    sample_rate=SAMPLE_RATE, channels=1, bits_per_sample=16,
                    payload=json.dumps({"text": text}).encode("utf-8"),
                    message_type=MessageType.TRANSCRIPT,
                ))

        return responses if responses else None

    def _detection_payload(self, result) -> bytes:
        """Common outer envelope for every model's result.

        Model-specific fields are present only when the model really produces
        them: ``environment_score`` stays null for SpeakerNet rather than being
        fabricated, and the gate fields stay null unless an envelope policy is
        actually running.
        """
        assert self.runtime is not None
        return json.dumps({
            # --- common envelope -------------------------------------------
            "type": "inference",
            "modelId": self.runtime.model_id,
            "modelVersion": self.runtime.model_version,
            "sequence": self._sequence,
            "timestamp": time.time(),
            # --- existing field names, preserved byte-for-byte so older
            # clients keep parsing exactly what they parsed before ----------
            "wearer_score": result.wearer_score,
            "environment_score": result.environment_score,
            "state": result.state,
            "context_ms": result.context_ms,
            "inference_ms": result.inference_ms,
            "echoed_capture_timestamp_ns": str(result.frame_capture_timestamp_ns),
            "gate_gain": (round(self.router.gain, 4) if self.router is not None else None),
            "gate_state": (self.router.state_name if self.router is not None else None),
            "gate_preroll_ms": (self.router.preroll_ms if self.router is not None else None),
        }).encode("utf-8")

    # -- telemetry ---------------------------------------------------------
    def latency_percentiles(self) -> Dict[str, float | None]:
        if not self.inference_ms_samples:
            return {"p50": None, "p95": None, "n": 0}
        values = np.asarray(self.inference_ms_samples, dtype=np.float64)
        return {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "n": int(values.size),
        }


class LiveSessionFactory:
    """Callable handed to ``MentraRemoteReceiver.serve(session_factory=...)``."""

    def __init__(self, registry: ModelRegistry, config: LiveSessionConfig):
        self.registry = registry
        self.config = config
        self.sessions: Dict[str, LiveSession] = {}

    def __call__(self, session_id: str, client_info: str = "") -> LiveSession:
        session = LiveSession(self.registry, self.config, session_id=session_id)
        self.sessions[session_id] = session
        return session

    def catalog_payload(self) -> bytes:
        return self.registry.catalog_payload()
