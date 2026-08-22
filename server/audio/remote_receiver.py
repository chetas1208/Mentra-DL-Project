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
    def __init__(self, host: str, port: int, jitter_target_ms: float = 30,
                 jitter_max_ms: float = 100, sample_rate: int = 16000,
                 allow_public_bind: bool = False):
        self.host = host
        self.port = port
        self.jitter_target_ms = jitter_target_ms
        self.jitter_max_ms = jitter_max_ms
        self.sample_rate = sample_rate

        if host in ("0.0.0.0", "::") and not allow_public_bind:
            raise ValueError(
                f"refusing to bind {host} (public) without allow_public_bind=True -- "
                f"section 5: bind to the Tailscale interface IP, not 0.0.0.0")

        self.sessions: dict[str, SessionStats] = {}

    async def _handle_connection(self, websocket, on_frame):
        session_id = None
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
            session_id = payload_str.split("|", 1)[0]
            self.sessions[session_id] = SessionStats(session_id, time.monotonic_ns())
            jitter_buffer = JitterBuffer(self.jitter_target_ms, self.jitter_max_ms, self.sample_rate)

            accept = AudioFrame(0, time.monotonic_ns(), self.sample_rate, 1, 16,
                                 payload=b"", message_type=MessageType.STREAM_ACCEPTED)
            await websocket.send(accept.encode())
            logger.info("session %s accepted", session_id)

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
                    if response is not None:
                        await websocket.send(response.encode())

        except websockets.exceptions.ConnectionClosed:
            logger.info("session %s disconnected", session_id)
        finally:
            if session_id and jitter_buffer:
                for ready_frame in jitter_buffer.flush():
                    on_frame(ready_frame)
                logger.info("session %s final stats: %s, jitter: %s",
                             session_id, self.sessions[session_id], jitter_buffer.stats)

    async def serve(self, on_frame) -> None:
        async with websockets.serve(
            lambda ws: self._handle_connection(ws, on_frame),
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
