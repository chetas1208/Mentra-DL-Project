#!/usr/bin/env python3
"""Protocol tests (sprint spec section 22). Run directly -- no pytest
harness exists in this repo yet, matches the pattern of the other
diagnostic test scripts already in tests/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from mentra.audio.frame import AudioFrame, Codec, MessageType, ProtocolError, MAGIC, MAX_PAYLOAD_BYTES

PASS = []
FAIL = []


def check(name, condition):
    (PASS if condition else FAIL).append(name)
    print(f"{'PASS' if condition else 'FAIL'}: {name}")


def test_round_trip():
    payload = bytes(range(256)) * 4  # 1024 bytes of deterministic content
    frame = AudioFrame(
        sequence_number=42,
        capture_timestamp_ns=123456789,
        sample_rate=16000,
        channels=1,
        bits_per_sample=16,
        payload=payload,
    )
    encoded = frame.encode()
    decoded = AudioFrame.decode(encoded)
    check("round_trip: sequence_number", decoded.sequence_number == 42)
    check("round_trip: capture_timestamp_ns", decoded.capture_timestamp_ns == 123456789)
    check("round_trip: sample_rate", decoded.sample_rate == 16000)
    check("round_trip: channels", decoded.channels == 1)
    check("round_trip: bits_per_sample", decoded.bits_per_sample == 16)
    check("round_trip: payload bit-identical", decoded.payload == payload)
    check("round_trip: message_type", decoded.message_type == MessageType.AUDIO_FRAME)
    check("round_trip: codec", decoded.codec == Codec.PCM16)
    check("round_trip: receive_timestamp_ns populated", decoded.receive_timestamp_ns is not None)


def test_empty_payload():
    frame = AudioFrame(0, 0, 16000, 1, 16, payload=b"")
    decoded = AudioFrame.decode(frame.encode())
    check("empty_payload: round trips", decoded.payload == b"")


def test_malformed_magic():
    bad = b"XXXX" + b"\x00" * 26
    try:
        AudioFrame.decode(bad)
        check("malformed_magic: rejected", False)
    except ProtocolError:
        check("malformed_magic: rejected", True)


def test_unsupported_version():
    frame = AudioFrame(0, 0, 16000, 1, 16, payload=b"abc")
    encoded = bytearray(frame.encode())
    encoded[4] = 99  # version byte, right after 4-byte magic
    try:
        AudioFrame.decode(bytes(encoded))
        check("unsupported_version: rejected", False)
    except ProtocolError:
        check("unsupported_version: rejected", True)


def test_truncated_header():
    try:
        AudioFrame.decode(MAGIC + b"\x01\x02")  # way too short
        check("truncated_header: rejected", False)
    except ProtocolError:
        check("truncated_header: rejected", True)


def test_truncated_payload():
    frame = AudioFrame(0, 0, 16000, 1, 16, payload=b"0123456789")
    encoded = frame.encode()
    truncated = encoded[:-5]  # cut off part of the declared payload
    try:
        AudioFrame.decode(truncated)
        check("truncated_payload: rejected", False)
    except ProtocolError:
        check("truncated_payload: rejected", True)


def test_oversized_payload_rejected_on_encode():
    huge = b"\x00" * (MAX_PAYLOAD_BYTES + 1)
    frame = AudioFrame(0, 0, 16000, 1, 16, payload=huge)
    try:
        frame.encode()
        check("oversized_payload: rejected on encode", False)
    except ProtocolError:
        check("oversized_payload: rejected on encode", True)


def test_invalid_codec():
    frame = AudioFrame(0, 0, 16000, 1, 16, payload=b"abc")
    encoded = bytearray(frame.encode())
    encoded[6] = 255  # codec byte position (magic4+version1+message_type1)
    try:
        AudioFrame.decode(bytes(encoded))
        check("invalid_codec: rejected", False)
    except ProtocolError:
        check("invalid_codec: rejected", True)


def test_sample_count():
    payload = b"\x00\x01" * 320  # 320 samples of 16-bit mono
    frame = AudioFrame(0, 0, 16000, 1, 16, payload=payload)
    check("sample_count: correct for 16-bit mono", frame.sample_count == 320)


if __name__ == "__main__":
    test_round_trip()
    test_empty_payload()
    test_malformed_magic()
    test_unsupported_version()
    test_truncated_header()
    test_truncated_payload()
    test_oversized_payload_rejected_on_encode()
    test_invalid_codec()
    test_sample_count()

    print()
    print(f"PASS: {len(PASS)}  FAIL: {len(FAIL)}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
