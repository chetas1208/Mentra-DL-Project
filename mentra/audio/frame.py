"""Binary audio frame protocol for the Mentra edge-to-server audio tunnel
(sprint spec section 8). Fixed 30-byte header, network byte order, no
padding -- deterministic parser, explicit version, bounded payload.

Header layout (struct format '!4sBBBBIQIBBI'):
    magic(4s) version(B) message_type(B) codec(B) flags(B)
    sequence_number(I) capture_timestamp_ns(Q) sample_rate(I)
    channels(B) bits_per_sample(B) payload_length(I)
followed by `payload_length` bytes of raw payload.
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum

MAGIC = b"MTRA"
PROTOCOL_VERSION = 1
MAX_PAYLOAD_BYTES = 1 << 20  # 1MB -- generous upper bound, real audio frames are ~KB

_HEADER_FORMAT = "!4sBBBBIQIBBI"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)
assert _HEADER_SIZE == 30, f"unexpected header size {_HEADER_SIZE}, format string changed?"


class MessageType(IntEnum):
    STREAM_START = 1
    STREAM_ACCEPTED = 2
    STREAM_REJECTED = 3
    AUDIO_FRAME = 4
    PING = 5
    PONG = 6
    DETECTION = 7  # server->client: live inference result, payload = UTF-8 JSON


class Codec(IntEnum):
    PCM16 = 1
    LC3 = 2  # scaffolded, not implemented -- see docs/MENTRA_REMOTE_AUDIO.md


class FrameFlags(IntEnum):
    NONE = 0
    DISCONTINUITY = 1 << 0  # set when this frame follows a detected gap (missing sequence numbers)


class ProtocolError(ValueError):
    """Malformed packet: bad magic, unsupported version, truncated header,
    truncated payload, or oversized payload. Never silently ignored."""


def monotonic_ns() -> int:
    return time.monotonic_ns()


@dataclass
class AudioFrame:
    sequence_number: int
    capture_timestamp_ns: int
    sample_rate: int
    channels: int
    bits_per_sample: int
    payload: bytes
    message_type: MessageType = MessageType.AUDIO_FRAME
    codec: Codec = Codec.PCM16
    flags: int = FrameFlags.NONE
    version: int = PROTOCOL_VERSION
    receive_timestamp_ns: int | None = field(default=None, compare=False)

    def encode(self) -> bytes:
        if len(self.payload) > MAX_PAYLOAD_BYTES:
            raise ProtocolError(
                f"payload {len(self.payload)} bytes exceeds MAX_PAYLOAD_BYTES={MAX_PAYLOAD_BYTES}")
        header = struct.pack(
            _HEADER_FORMAT,
            MAGIC,
            self.version,
            int(self.message_type),
            int(self.codec),
            self.flags,
            self.sequence_number & 0xFFFFFFFF,
            self.capture_timestamp_ns & 0xFFFFFFFFFFFFFFFF,
            self.sample_rate,
            self.channels,
            self.bits_per_sample,
            len(self.payload),
        )
        return header + self.payload

    @classmethod
    def decode(cls, raw: bytes) -> "AudioFrame":
        if len(raw) < _HEADER_SIZE:
            raise ProtocolError(f"truncated header: got {len(raw)} bytes, need >= {_HEADER_SIZE}")

        (magic, version, message_type, codec, flags, seq, cap_ts, sample_rate,
         channels, bits_per_sample, payload_length) = struct.unpack(_HEADER_FORMAT, raw[:_HEADER_SIZE])

        if magic != MAGIC:
            raise ProtocolError(f"bad magic: {magic!r} != {MAGIC!r}")
        if version != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol version: {version} (this build supports {PROTOCOL_VERSION})")
        if payload_length > MAX_PAYLOAD_BYTES:
            raise ProtocolError(f"declared payload_length {payload_length} exceeds MAX_PAYLOAD_BYTES={MAX_PAYLOAD_BYTES}")

        payload = raw[_HEADER_SIZE:_HEADER_SIZE + payload_length]
        if len(payload) != payload_length:
            raise ProtocolError(
                f"truncated payload: declared {payload_length} bytes, got {len(payload)}")

        try:
            message_type = MessageType(message_type)
        except ValueError:
            raise ProtocolError(f"unknown message_type: {message_type}")
        try:
            codec = Codec(codec)
        except ValueError:
            raise ProtocolError(f"unknown codec: {codec}")

        return cls(
            sequence_number=seq,
            capture_timestamp_ns=cap_ts,
            sample_rate=sample_rate,
            channels=channels,
            bits_per_sample=bits_per_sample,
            payload=payload,
            message_type=message_type,
            codec=codec,
            flags=flags,
            version=version,
            receive_timestamp_ns=monotonic_ns(),
        )

    @property
    def sample_count(self) -> int:
        bytes_per_sample = self.bits_per_sample // 8
        if bytes_per_sample == 0 or self.channels == 0:
            return 0
        return len(self.payload) // (bytes_per_sample * self.channels)
