"""Tests for mentra/audio/mentraos_source.py -- the MentraOS Path A/B
adapter. No hardware involved: these test the decode contract against
synthetic payloads shaped like what the docs describe, per
docs/geowearnet_g4_mentraos_audio_path.md."""
import base64
import struct

import pytest

from mentra.audio.mentraos_source import (
    MentraOSDecodeError,
    decode_audio_chunk,
    decode_mic_pcm,
)


def _pcm16(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


class TestDecodeAudioChunk:
    def test_valid_base64_pcm_roundtrips_payload_exactly(self):
        raw = _pcm16(100, -100, 32000, -32768)
        b64 = base64.b64encode(raw).decode()
        frame, meta = decode_audio_chunk(b64, sequence_number=1, capture_timestamp_ns=42)
        assert frame.payload == raw
        assert frame.sequence_number == 1
        assert frame.capture_timestamp_ns == 42

    def test_missing_reported_sample_rate_is_marked_assumed(self):
        raw = _pcm16(0, 0)
        b64 = base64.b64encode(raw).decode()
        frame, meta = decode_audio_chunk(b64, sequence_number=1, capture_timestamp_ns=0)
        assert meta.assumed_format is True
        assert meta.sample_rate_reported is False

    def test_reported_sample_rate_is_used_and_not_marked_assumed(self):
        raw = _pcm16(0, 0)
        b64 = base64.b64encode(raw).decode()
        frame, meta = decode_audio_chunk(
            b64, sequence_number=1, capture_timestamp_ns=0, reported_sample_rate=48000
        )
        assert frame.sample_rate == 48000
        assert meta.assumed_format is False
        assert meta.sample_rate_reported is True

    def test_lc3_codec_hint_refuses_rather_than_silently_corrupting(self):
        raw = _pcm16(1, 2, 3)
        b64 = base64.b64encode(raw).decode()
        with pytest.raises(MentraOSDecodeError, match="LC3"):
            decode_audio_chunk(b64, sequence_number=1, capture_timestamp_ns=0, codec_hint="lc3")

    def test_invalid_base64_raises_typed_error_not_generic_exception(self):
        with pytest.raises(MentraOSDecodeError):
            decode_audio_chunk("not valid base64 !!!", sequence_number=1, capture_timestamp_ns=0)

    def test_path_label_is_correct(self):
        raw = _pcm16(0)
        b64 = base64.b64encode(raw).decode()
        _, meta = decode_audio_chunk(b64, sequence_number=1, capture_timestamp_ns=0)
        assert meta.path == "session.mic.onAudioChunk"


class TestDecodeMicPcm:
    def test_raw_payload_passes_through_exactly(self):
        raw = _pcm16(500, -500, 0)
        frame, meta = decode_mic_pcm(raw, sequence_number=7, capture_timestamp_ns=99)
        assert frame.payload == raw
        assert frame.sequence_number == 7
        assert frame.capture_timestamp_ns == 99

    def test_no_known_sample_rate_is_marked_assumed(self):
        raw = _pcm16(0)
        frame, meta = decode_mic_pcm(raw, sequence_number=1, capture_timestamp_ns=0)
        assert meta.assumed_format is True
        assert meta.sample_rate_reported is False  # never reported on this path per docs

    def test_known_sample_rate_from_real_measurement_is_used(self):
        raw = _pcm16(0)
        frame, meta = decode_mic_pcm(
            raw, sequence_number=1, capture_timestamp_ns=0, known_sample_rate=44100
        )
        assert frame.sample_rate == 44100
        # caller supplied a real measured value, so this frame is no longer guessed
        assert meta.assumed_format is False
        # ...but the API itself still never *reports* a rate on this path
        assert meta.sample_rate_reported is False

    def test_path_label_is_correct(self):
        raw = _pcm16(0)
        _, meta = decode_mic_pcm(raw, sequence_number=1, capture_timestamp_ns=0)
        assert meta.path == "mic_pcm"


class TestSharedContract:
    """Both paths must feed the exact same downstream contract
    (mentra.audio.frame.AudioFrame) that the browser-capture path already
    uses -- this is the point of the adapter."""

    def test_both_paths_produce_pcm16_codec_audio_frames(self):
        from mentra.audio.frame import Codec, MessageType

        raw = _pcm16(1, 2, 3, 4)
        b64 = base64.b64encode(raw).decode()
        frame_a, _ = decode_audio_chunk(b64, sequence_number=1, capture_timestamp_ns=0)
        frame_b, _ = decode_mic_pcm(raw, sequence_number=1, capture_timestamp_ns=0)

        for frame in (frame_a, frame_b):
            assert frame.codec == Codec.PCM16
            assert frame.message_type == MessageType.AUDIO_FRAME
            assert frame.channels == 1
            assert frame.bits_per_sample == 16
