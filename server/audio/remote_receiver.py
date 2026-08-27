"""Server-side WebSocket audio receiver (sprint spec section 11). Binds to
one interface (Tailscale IP, not 0.0.0.0, by default -- section 5), accepts
sessions, validates frames, tracks sequence/jitter, and calls `on_frame`
for each frame released from the jitter buffer -- in order, with
discontinuities marked, ready to feed straight into the existing
WavPcmSource-shaped consumer used by the rest of this repo's inference
code (research/common/pcm_source.py) without a second parallel pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import websockets

from mentra.audio.frame import AudioFrame, MessageType, ProtocolError
from mentra.audio.jitter_buffer import JitterBuffer

logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    session_id: str
    connected_at_ns: int
    frames_received: int = 0
    bytes_received: int = 0
    protocol_errors: int = 0
    capture_to_receive_ms: list = field(default_factory=list)  # rolling, capped


class MentraRemoteReceiver:
    #: Token a client puts in its STREAM_START payload to say it understands
    #: the multi-model control plane (MODEL_CATALOG / SESSION_CONFIG /
    #: SESSION_CONFIG_ACK). Feature detection, not a version bump: the binary
    #: frame PROTOCOL_VERSION is unchanged, and a client that never sends this
    #: token is served exactly the message types it was served before.
    MODEL_SELECTION_FEATURE = "model_selection"

    def __init__(self, host: str, port: int, jitter_target_ms: float = 30,
                 jitter_max_ms: float = 100, sample_rate: int = 16000,
                 allow_public_bind: bool = False, capabilities: "object | None" = None,
                 session_factory=None):
        self.host = host
        self.port = port
        self.jitter_target_ms = jitter_target_ms
        self.jitter_max_ms = jitter_max_ms
        self.sample_rate = sample_rate
        # Model-capability handshake (Workstream AW). `capabilities` is a
        # server.models.capabilities.ModelCapabilities, or None to preserve the
        # previous behaviour of sending an empty payload -- which the frontend
        # already renders as "NOT AVAILABLE" rather than guessing a default.
        self.capabilities = capabilities
        # Multi-model mode. When supplied, ONE session object is created per
        # connection and owns that connection's model binding and stream-local
        # state; the legacy flat-callback form below stays byte-for-byte
        # compatible for every existing caller and test.
        self.session_factory = session_factory

        if host in ("0.0.0.0", "::") and not allow_public_bind:
            raise ValueError(
                f"refusing to bind {host} (public) without allow_public_bind=True -- "
                f"section 5: bind to the Tailscale interface IP, not 0.0.0.0")

        self.sessions: dict[str, SessionStats] = {}

    async def _handle_connection(self, websocket, on_frame, on_enroll=None, on_playback_mode=None,
                                 on_session_start=None, on_session_end=None,
                                 on_capture_meta=None, on_capture_raw=None):
        session_id = None
        session = None
        jitter_buffer: JitterBuffer | None = None
        try:
            raw = await websocket.recv()
            start_frame = AudioFrame.decode(raw)
            if start_frame.message_type != MessageType.STREAM_START:
                reject = AudioFrame(0, time.monotonic_ns(), self.sample_rate, 1, 16,
                                     payload=b"expected STREAM_START",
                                     message_type=MessageType.STREAM_REJECTED)
                await websocket.send(reject.encode())
                return

            payload_str = start_frame.payload.decode("utf-8", errors="replace")
            tokens = payload_str.split("|")
            session_id = tokens[0]
            client_info = "|".join(tokens[1:])
            wants_model_selection = any(
                self.MODEL_SELECTION_FEATURE in token for token in tokens[1:])
            self.sessions[session_id] = SessionStats(session_id, time.monotonic_ns())
            jitter_buffer = JitterBuffer(self.jitter_target_ms, self.jitter_max_ms, self.sample_rate)

            if self.session_factory is not None:
                session = self.session_factory(session_id, client_info)
                on_frame = session.on_frame
                on_enroll = session.on_enroll
                on_playback_mode = session.on_playback_mode
                on_session_start = None  # the session constructor already reset itself
                on_session_end = (
                    lambda sid, s=session: s.close(f"transport session {sid} ended"))
                accept_payload = session.capabilities_payload()
                advertised = session.model_id
            else:
                accept_payload = (self.capabilities.to_payload()
                                  if self.capabilities is not None else b"")
                advertised = getattr(self.capabilities, "modelId", "none-advertised")

            accept = AudioFrame(0, time.monotonic_ns(), self.sample_rate, 1, 16,
                                 payload=accept_payload, message_type=MessageType.STREAM_ACCEPTED)
            await websocket.send(accept.encode())
            logger.info("session %s accepted (model=%s)", session_id, advertised)

            # Model catalog: sent ONLY to a client that advertised the feature,
            # so a client built before this message type existed never receives
            # an unknown message type.
            if session is not None and wants_model_selection:
                await websocket.send(AudioFrame(
                    0, time.monotonic_ns(), self.sample_rate, 1, 16,
                    payload=session.registry.catalog_payload(),
                    message_type=MessageType.MODEL_CATALOG).encode())

            if on_session_start is not None:
                on_session_start(session_id)

            async for raw in websocket:
                stats = self.sessions[session_id]
                try:
                    frame = AudioFrame.decode(raw)
                except ProtocolError as e:
                    stats.protocol_errors += 1
                    logger.warning("session %s: protocol error: %s", session_id, e)
                    continue

                if frame.message_type == MessageType.PING:
                    pong = AudioFrame(0, time.monotonic_ns(), self.sample_rate, 1, 16,
                                       payload=b"", message_type=MessageType.PONG)
                    await websocket.send(pong.encode())
                    continue

                if frame.message_type == MessageType.ENROLL_AUDIO:
                    if on_enroll is not None:
                        on_enroll(frame)
                        done = AudioFrame(0, time.monotonic_ns(), self.sample_rate, 1, 16,
                                           payload=b"", message_type=MessageType.ENROLL_DONE)
                        await websocket.send(done.encode())
                    continue

                # Per-session model selection. Control metadata only -- the
                # binary AUDIO_FRAME framing is untouched, and the binding
                # applies to THIS session, never to the process.
                if frame.message_type == MessageType.SESSION_CONFIG:
                    if session is not None:
                        ack = session.on_session_config(frame.payload)
                        await websocket.send(AudioFrame(
                            0, time.monotonic_ns(), self.sample_rate, 1, 16,
                            payload=ack,
                            message_type=MessageType.SESSION_CONFIG_ACK).encode())
                    else:
                        logger.info("session %s: session_config ignored (single-model receiver)",
                                    session_id)
                    continue

                if frame.message_type == MessageType.SET_PLAYBACK_MODE:
                    if on_playback_mode is not None:
                        on_playback_mode(frame.payload.decode("ascii", errors="replace"))
                    continue

                # G4 WS6 research capture. Ignored entirely unless the operator
                # started the receiver with a capture directory, so the normal
                # live path is byte-for-byte unchanged.
                if frame.message_type == MessageType.CAPTURE_META:
                    if on_capture_meta is not None:
                        ack = on_capture_meta(session_id, frame.payload)
                        if ack is not None:
                            await websocket.send(AudioFrame(
                                0, time.monotonic_ns(), self.sample_rate, 1, 16,
                                payload=ack, message_type=MessageType.CAPTURE_ACK).encode())
                    continue

                if frame.message_type == MessageType.CAPTURE_RAW_PCM:
                    if on_capture_raw is not None:
                        on_capture_raw(session_id, frame)
                    continue

                if frame.message_type != MessageType.AUDIO_FRAME:
                    continue

                stats.frames_received += 1
                stats.bytes_received += len(frame.payload)
                latency_ms = (frame.receive_timestamp_ns - frame.capture_timestamp_ns) / 1e6
                stats.capture_to_receive_ms.append(latency_ms)
                if len(stats.capture_to_receive_ms) > 1000:
                    stats.capture_to_receive_ms.pop(0)

                jitter_buffer.push(frame)
                for ready_frame in jitter_buffer.release_ready():
                    response = on_frame(ready_frame)
                    if response is None:
                        continue
                    # on_frame may return a single AudioFrame or a list of
                    # them (e.g. DETECTION + gated audio + TRANSCRIPT per
                    # input frame) -- normalize to a list.
                    responses = response if isinstance(response, list) else [response]
                    for r in responses:
                        await websocket.send(r.encode())

        except websockets.exceptions.ConnectionClosed:
            logger.info("session %s disconnected", session_id)
        finally:
            if session_id and jitter_buffer:
                if on_frame is not None:
                    for ready_frame in jitter_buffer.flush():
                        on_frame(ready_frame)
                logger.info("session %s final stats: %s, jitter: %s",
                             session_id, self.sessions[session_id], jitter_buffer.stats)
            if session_id and on_session_end is not None:
                on_session_end(session_id)

    async def serve(self, on_frame=None, on_enroll=None, on_playback_mode=None,
                    on_session_start=None, on_session_end=None,
                    on_capture_meta=None, on_capture_raw=None) -> None:
        if on_frame is None and self.session_factory is None:
            raise ValueError("serve() needs either on_frame or a session_factory")
        async with websockets.serve(
            lambda ws: self._handle_connection(
                ws, on_frame, on_enroll, on_playback_mode, on_session_start, on_session_end,
                on_capture_meta, on_capture_raw,
            ),
            self.host, self.port, max_size=None,
        ):
            logger.info("MentraRemoteReceiver listening on %s:%d", self.host, self.port)
            await asyncio.Future()  # run forever until cancelled

    async def close(self) -> None:
        pass  # server lifecycle managed by the `async with` in serve()


def is_bound_publicly(host: str) -> bool:
    """Section 5: validation helper to warn when accidentally listening
    publicly instead of on the Tailscale interface."""
    return host in ("0.0.0.0", "::", "", None)
