"""MentraOS audio source adapter (G4 follow-up, product-native path).

Per `mentra/audio/consumer.py`'s own docstring: "the eventual real source is
literally an adapter swap." This module is that adapter for the OFFICIAL
MentraOS audio paths (as opposed to the browser-mic capture path already
built in `web/`), so GeoWearNet/the live receiver never has to care whether
a frame originated from a browser AudioWorklet or from MentraOS.

Two documented paths exist (see docs/geowearnet_g4_mentraos_audio_path.md
for full sourcing — S1-S5, fetched 2026-08-27, desk research only, nothing
here has been run against a real device):

  Path A -- cloud/app SDK: `session.mic.onAudioChunk()`. Payload is
  "base64-encoded audio, PCM or LC3 depending on the phone's mic mode."
  Sample rate is reported "when reported" (optional).

  Path B -- Bluetooth SDK on the phone: `onMicPcm` / `'mic_pcm'` raw
  ByteArray frames. Shorter chain, least processed, but sample rate / bit
  depth / channel count are UNSPECIFIED in the docs actually read.

Neither path's exact wire format has been confirmed against hardware. This
module deliberately does NOT invent numbers where the docs are silent --
every frame this module produces is stamped with `assumed_format=True`
metadata (see `MentraOSFrameMeta`) so any downstream consumer, capture
session, or report can tell "measured" from "assumed" at a glance, per this
project's standing rule against quietly-guessed values invalidating a later
domain-gap analysis.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

from mentra.audio.frame import AudioFrame, Codec, MessageType

# Unconfirmed hypothesis only (see docs/geowearnet_g4_mentraos_audio_path.md
# "Unverified claim, recorded as such") -- used as a fallback default, never
# silently trusted. Every frame built with this default carries
# assumed_format=True.
_UNCONFIRMED_DEFAULT_SAMPLE_RATE = 16000
_UNCONFIRMED_DEFAULT_BITS_PER_SAMPLE = 16
_UNCONFIRMED_DEFAULT_CHANNELS = 1


@dataclass
class MentraOSFrameMeta:
    """Provenance for one decoded frame -- what was actually reported by
    the API vs. what this adapter had to assume because the docs don't say."""
    path: str  # "session.mic.onAudioChunk" | "mic_pcm"
    sample_rate_reported: bool
    assumed_format: bool
    codec: str


class MentraOSDecodeError(Exception):
    """Raised when a payload can't be turned into PCM16 by this adapter --
    e.g. an LC3-encoded Path A chunk, which this environment cannot decode
    (no LC3 codec library available/verified here). Callers should surface
    this as a real, visible failure, not silently drop the frame."""


def decode_audio_chunk(
    base64_payload: str,
    *,
    sequence_number: int,
    capture_timestamp_ns: int,
    reported_sample_rate: int | None = None,
    codec_hint: str = "pcm",
) -> tuple[AudioFrame, MentraOSFrameMeta]:
    """Path A: `session.mic.onAudioChunk()`.

    `reported_sample_rate` should be the value from the SDK's
    `AudioChunkData` if present -- the docs say this field is optional
    ("when reported"), so callers must pass None when it's absent rather
    than fabricating a number.

    Raises MentraOSDecodeError for the LC3 branch -- not silently
    downgraded to garbage PCM.
    """
    if codec_hint.lower() == "lc3":
        raise MentraOSDecodeError(
            "audio_chunk reported LC3 encoding; this adapter has no verified "
            "LC3 decoder in this environment. Do not guess-decode as PCM16 -- "
            "that would silently corrupt every sample. Route this stream "
            "through a real LC3 decoder before calling this adapter, or "
            "request PCM mode on the phone side if the SDK exposes that."
        )

    try:
        raw = base64.b64decode(base64_payload, validate=True)
    except Exception as exc:  # noqa: BLE001 -- surfaced as a typed decode error
        raise MentraOSDecodeError(f"audio_chunk payload is not valid base64: {exc}") from exc

    assumed = reported_sample_rate is None
    sample_rate = reported_sample_rate or _UNCONFIRMED_DEFAULT_SAMPLE_RATE

    frame = AudioFrame(
        sequence_number=sequence_number,
        capture_timestamp_ns=capture_timestamp_ns,
        sample_rate=sample_rate,
        channels=_UNCONFIRMED_DEFAULT_CHANNELS,
        bits_per_sample=_UNCONFIRMED_DEFAULT_BITS_PER_SAMPLE,
        payload=raw,
        message_type=MessageType.AUDIO_FRAME,
        codec=Codec.PCM16,
    )
    meta = MentraOSFrameMeta(
        path="session.mic.onAudioChunk",
        sample_rate_reported=not assumed,
        assumed_format=assumed,
        codec="pcm",
    )
    return frame, meta


def decode_mic_pcm(
    raw_payload: bytes,
    *,
    sequence_number: int,
    capture_timestamp_ns: int,
    known_sample_rate: int | None = None,
) -> tuple[AudioFrame, MentraOSFrameMeta]:
    """Path B: Bluetooth SDK `onMicPcm` / `'mic_pcm'` raw frames.

    `known_sample_rate` should come from a real-device measurement (the
    docs don't specify it) -- pass None on the very first pilot connection
    before that measurement exists; this adapter will use the documented
    UNCONFIRMED default and mark the frame `assumed_format=True` so the
    capture session's own metadata (see mentra/capture/session.py) records
    that the rate is not yet verified, per docs/geowearnet_g4_mentraos_audio_path.md's
    'rawest available audio' table.
    """
    assumed = known_sample_rate is None
    sample_rate = known_sample_rate or _UNCONFIRMED_DEFAULT_SAMPLE_RATE

    frame = AudioFrame(
        sequence_number=sequence_number,
        capture_timestamp_ns=capture_timestamp_ns,
        sample_rate=sample_rate,
        channels=_UNCONFIRMED_DEFAULT_CHANNELS,
        bits_per_sample=_UNCONFIRMED_DEFAULT_BITS_PER_SAMPLE,
        payload=raw_payload,
        message_type=MessageType.AUDIO_FRAME,
        codec=Codec.PCM16,
    )
    meta = MentraOSFrameMeta(
        path="mic_pcm",
        sample_rate_reported=False,  # never reported on this path per the docs read
        assumed_format=assumed,
        codec="pcm",
    )
    return frame, meta
