"""Transport interface (sprint spec section 6/7). Model/pipeline code
depends on this abstraction, never on WebSocket specifics directly --
swapping in a future QuicAudioTransport should not touch anything upstream."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mentra.audio.frame import AudioFrame


@dataclass
class TransportStats:
    state: str = "DISCONNECTED"
    frames_sent: int = 0
    frames_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    reconnect_count: int = 0
    last_rtt_ms: float | None = None


class AudioTransport(ABC):
    """Sender-side interface (edge device)."""

    stats: TransportStats

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def send_frame(self, frame: AudioFrame) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class RemoteAudioTransport(ABC):
    """Receiver-side interface (server). `on_frame` is called for every
    audio frame released from the jitter buffer, in order."""

    stats: TransportStats

    @abstractmethod
    async def serve(self, on_frame) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...
