"""WebSocket audio transport (sprint spec section 7). Binary frames only --
no JSON/base64 in the audio path. Session handshake, heartbeat, and
reconnect-with-backoff all live here so the sender interface (connect/
send_frame/close) stays trivial for callers.

Works identically over localhost, LAN, or a Tailscale tailnet -- Tailscale
supplies the IP layer underneath, nothing here is Tailscale-specific.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid

import websockets
from websockets.asyncio.client import connect as ws_connect

from mentra.audio.frame import AudioFrame, Codec, MessageType, ProtocolError
from mentra.audio.transport.base import AudioTransport, TransportStats

logger = logging.getLogger(__name__)


class WebSocketAudioTransport(AudioTransport):
    def __init__(
        self,
        uri: str,
        sample_rate: int = 16000,
        channels: int = 1,
        bits_per_sample: int = 16,
        codec: Codec = Codec.PCM16,
        reconnect_initial_ms: int = 250,
        reconnect_max_ms: int = 5000,
        heartbeat_interval_s: float = 5.0,
        dead_connection_s: float = 15.0,
        client_version: str = "mentra-edge-bridge-0.1",
    ):
        self.uri = uri
        self.sample_rate = sample_rate
        self.channels = channels
        self.bits_per_sample = bits_per_sample
        self.codec = codec
        self.reconnect_initial_ms = reconnect_initial_ms
        self.reconnect_max_ms = reconnect_max_ms
        self.heartbeat_interval_s = heartbeat_interval_s
        self.dead_connection_s = dead_connection_s
        self.client_version = client_version

        self.session_id = str(uuid.uuid4())
        self._ws = None
        self._sequence_number = 0
        self._last_pong_monotonic: float | None = None
        self._last_ping_sent_monotonic: float | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._receive_task: asyncio.Task | None = None
        self.ping_count = 0
        self.pong_count = 0
        self.false_dead_count = 0
        self.stats = TransportStats()

    async def connect(self) -> None:
        """Connects and performs the STREAM_START handshake. Raises if the
        server rejects the session -- caller decides retry policy."""
        self._ws = await ws_connect(self.uri, max_size=None)
        self.stats.state = "CONNECTING"

        start_frame = AudioFrame(
            sequence_number=0,
            capture_timestamp_ns=time.monotonic_ns(),
            sample_rate=self.sample_rate,
            channels=self.channels,
            bits_per_sample=self.bits_per_sample,
            payload=self.session_id.encode("utf-8") + b"|" + self.client_version.encode("utf-8"),
            message_type=MessageType.STREAM_START,
            codec=self.codec,
        )
        await self._ws.send(start_frame.encode())

        response_raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
        response = AudioFrame.decode(response_raw)
        if response.message_type != MessageType.STREAM_ACCEPTED:
            self.stats.state = "REJECTED"
            reason = response.payload.decode("utf-8", errors="replace")
            await self._ws.close()
            raise ConnectionError(f"server rejected session: {reason}")

        self.stats.state = "CONNECTED"
        self._sequence_number = 0
        self._last_pong_monotonic = time.monotonic()
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("session %s connected to %s", self.session_id, self.uri)

    async def _receive_loop(self) -> None:
        """Dedicated receive task -- the connection cannot be send-only if
        it needs to consume control messages like PONG (section 2). Runs
        concurrently with the caller's send_frame() calls and the
        heartbeat loop."""
        try:
            async for raw in self._ws:
                try:
                    msg = AudioFrame.decode(raw)
                except ProtocolError as e:
                    logger.warning("received malformed frame: %s", e)
                    continue

                if msg.message_type == MessageType.PONG:
                    self._last_pong_monotonic = time.monotonic()
                    self.pong_count += 1
                    if self._last_ping_sent_monotonic is not None:
                        self.stats.last_rtt_ms = (time.monotonic() - self._last_ping_sent_monotonic) * 1000
                elif msg.message_type == MessageType.STREAM_REJECTED:
                    logger.warning("server sent STREAM_REJECTED mid-session: %s",
                                    msg.payload.decode("utf-8", errors="replace"))
                # AUDIO_FRAME/STREAM_START/STREAM_ACCEPTED are not expected
                # from the server on this connection direction -- ignored,
                # not an error (deterministic parser, not a strict-mode reject).
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed:
            self.stats.state = "DISCONNECTED"

    async def send_frame(self, frame: AudioFrame) -> None:
        if self._ws is None:
            raise ConnectionError("not connected -- call connect() first")
        frame.sequence_number = self._sequence_number
        self._sequence_number += 1
        encoded = frame.encode()
        await self._ws.send(encoded)
        self.stats.frames_sent += 1
        self.stats.bytes_sent += len(encoded)

    async def _heartbeat_loop(self) -> None:
        """Sends PING on an interval; dead-connection detection now reads
        _last_pong_monotonic as updated by _receive_loop() (the bug this
        fixes: previously nothing ever consumed the server's PONG, so this
        check always measured time-since-connect instead of time-since-
        last-actual-PONG, and would eventually fire a false DEAD state on
        any long-lived connection)."""
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval_s)
                ping = AudioFrame(0, time.monotonic_ns(), self.sample_rate, self.channels,
                                   self.bits_per_sample, payload=b"", message_type=MessageType.PING)
                self._last_ping_sent_monotonic = time.monotonic()
                await self._ws.send(ping.encode())
                self.ping_count += 1

                elapsed_since_pong = time.monotonic() - (self._last_pong_monotonic or self._last_ping_sent_monotonic)
                if elapsed_since_pong > self.dead_connection_s:
                    logger.warning("connection dead: no PONG in %.1fs", elapsed_since_pong)
                    self.stats.state = "DEAD"
                    return
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed:
            self.stats.state = "DISCONNECTED"

    async def close(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._receive_task:
            self._receive_task.cancel()
        if self._ws is not None:
            await self._ws.close()
        self.stats.state = "DISCONNECTED"

    async def run_with_reconnect(self, frame_source, stop_event: asyncio.Event) -> None:
        """frame_source: async generator yielding AudioFrame. Reconnects
        with exponential backoff + jitter on failure (section 14). On
        reconnect: new session_id, sequence resets to 0, no replay of
        buffered audio -- resumes live."""
        backoff_ms = self.reconnect_initial_ms
        while not stop_event.is_set():
            try:
                await self.connect()
                backoff_ms = self.reconnect_initial_ms  # reset on success
                async for frame in frame_source:
                    if stop_event.is_set():
                        break
                    await self.send_frame(frame)
            except (ConnectionError, websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
                logger.warning("transport error: %s -- reconnecting in %dms", e, backoff_ms)
                self.stats.reconnect_count += 1
                self.session_id = str(uuid.uuid4())  # new session on reconnect
                await self.close()
                jitter = random.uniform(0, backoff_ms * 0.3)
                await asyncio.sleep((backoff_ms + jitter) / 1000)
                backoff_ms = min(backoff_ms * 2, self.reconnect_max_ms)
