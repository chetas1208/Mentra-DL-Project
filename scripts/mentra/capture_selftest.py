#!/usr/bin/env python3
"""G4 WS6 -- end-to-end self-test of the research capture path.

WHAT THIS PROVES
    The capture chain works: a client opens a real WebSocket to a real
    ``MentraRemoteReceiver``, sends real CAPTURE_META, real native-rate
    CAPTURE_RAW_PCM and real 16 kHz AUDIO_FRAMEs over the real MTRA binary
    protocol, and the real ``CaptureSink`` -- the same object
    ``scripts/mentra/run_receiver.py`` uses -- writes raw.wav,
    derived_16k.wav, metadata.json and validation.json, which are then read
    back and checked.

WHAT THIS DOES NOT PROVE
    Anything whatsoever about Mentra Live. The audio is LibriSpeech from this
    repo's own public evaluation data, resampled up to a plausible browser
    native rate. Every take this script writes is stamped
    ``source_kind=SYNTHETIC_TEST_INPUT_NOT_MENTRA_AUDIO``, and
    ``mentra.capture.session.assert_mentra_hardware`` will refuse it. That
    refusal is itself asserted below, so the guard against mislabelling
    synthetic audio as Mentra evidence is tested, not just documented.

Usage:
    python3 scripts/mentra/capture_selftest.py [--port 18790] [--seconds 6]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf
import websockets

from mentra.audio.frame import AudioFrame, Codec, MessageType
from mentra.capture.session import CaptureMetadataError, assert_mentra_hardware
from mentra.capture.sink import CaptureSink
from server.audio.remote_receiver import MentraRemoteReceiver

REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_RATE = 48000          # the rate a browser AudioContext typically reports
NATIVE_BLOCK = 480           # 10 ms at 48 kHz
TRANSPORT_RATE = 16000
TRANSPORT_BLOCK = 160        # 10 ms at 16 kHz


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    x = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


def load_test_audio(seconds: float) -> np.ndarray:
    """Real speech from this repo's public LibriSpeech evaluation data.

    Explicitly NOT Mentra audio, and never labelled as such downstream."""
    manifest = json.loads(
        (REPO_ROOT / "evaluation/manifests/day1_public_speakers.json").read_text())
    paths = []
    for speaker in sorted(manifest["speakers"]):
        paths.extend(manifest["speakers"][speaker]["test"])
    chunks, total = [], 0
    for relative in paths:
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        data, rate = sf.read(str(path), always_2d=True, dtype="float32")
        mono = np.ascontiguousarray(data[:, 0])
        if rate != TRANSPORT_RATE:
            raise RuntimeError(f"expected {TRANSPORT_RATE} Hz test audio, got {rate}")
        chunks.append(mono)
        total += len(mono)
        if total >= seconds * TRANSPORT_RATE:
            break
    if not chunks:
        raise FileNotFoundError("no LibriSpeech test audio found for the capture self-test")
    audio = np.concatenate(chunks)[: int(seconds * TRANSPORT_RATE)]
    return audio


def upsample(audio16k: np.ndarray, factor: int = 3) -> np.ndarray:
    """16 kHz -> 48 kHz by linear interpolation.

    A real browser delivers native-rate audio and resamples DOWN to 16 kHz;
    this test has only 16 kHz source material, so it interpolates up to
    produce a plausible native-rate stream. That direction means the two
    streams are related by resampling in the opposite order from a real
    capture -- fine for exercising the transport, files and validation, and
    another reason this is not evidence about any device."""
    n = len(audio16k)
    src = np.arange(n, dtype=np.float64)
    dst = np.linspace(0, n - 1, n * factor)
    return np.interp(dst, src, audio16k).astype(np.float32)


async def _client(port: int, audio16k: np.ndarray, session_id: str, wearer_id: str) -> dict:
    uri = f"ws://127.0.0.1:{port}"
    native = upsample(audio16k)
    ack: dict = {}
    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(AudioFrame(0, 0, TRANSPORT_RATE, 1, 16,
                                 payload=f"{session_id}|capture-selftest".encode(),
                                 message_type=MessageType.STREAM_START).encode())
        accepted = AudioFrame.decode(await ws.recv())
        if accepted.message_type != MessageType.STREAM_ACCEPTED:
            raise RuntimeError(f"receiver did not accept the stream: {accepted.message_type}")

        metadata = {
            "session_id": session_id,
            "wearer_id": wearer_id,
            "condition": "selftest_synthetic_librispeech",
            "source_kind": "SYNTHETIC_TEST_INPUT_NOT_MENTRA_AUDIO",
            "raw_sample_rate": NATIVE_RATE,
            "raw_channels": 1,
            "client_version": "capture-selftest-0.1",
            "notes": "LibriSpeech replayed through the real capture chain. NOT Mentra audio.",
        }
        await ws.send(AudioFrame(0, 0, TRANSPORT_RATE, 1, 16,
                                 payload=json.dumps(metadata).encode("utf-8"),
                                 message_type=MessageType.CAPTURE_META).encode())
        response = AudioFrame.decode(await ws.recv())
        if response.message_type != MessageType.CAPTURE_ACK:
            raise RuntimeError(f"expected CAPTURE_ACK, got {response.message_type}")
        ack = json.loads(response.payload.decode("utf-8"))
        if ack.get("status") != "OPEN":
            raise RuntimeError(f"capture session not opened: {ack}")

        n_blocks = len(audio16k) // TRANSPORT_BLOCK
        for i in range(n_blocks):
            await ws.send(AudioFrame(
                i, i * 10_000_000, NATIVE_RATE, 1, 16,
                payload=float32_to_pcm16(native[i * NATIVE_BLOCK:(i + 1) * NATIVE_BLOCK]),
                message_type=MessageType.CAPTURE_RAW_PCM, codec=Codec.PCM16).encode())
            await ws.send(AudioFrame(
                i, i * 10_000_000, TRANSPORT_RATE, 1, 16,
                payload=float32_to_pcm16(audio16k[i * TRANSPORT_BLOCK:(i + 1) * TRANSPORT_BLOCK]),
                message_type=MessageType.AUDIO_FRAME, codec=Codec.PCM16).encode())
            if i % 25 == 0:
                await asyncio.sleep(0)  # let the server drain
        await asyncio.sleep(0.3)
    return ack


async def run(port: int, seconds: float, capture_root: Path) -> dict:
    audio16k = load_test_audio(seconds)
    sink = CaptureSink(capture_root)
    received_derived = {"frames": 0}

    def on_frame(frame: AudioFrame):
        # Mirrors the receiver's derived-stream archiving without loading a
        # model: this self-test is about the capture path, not inference.
        sink.on_derived_frame(frame.payload)
        received_derived["frames"] += 1
        return None

    receiver = MentraRemoteReceiver("127.0.0.1", port, jitter_target_ms=20, jitter_max_ms=80)
    server_task = asyncio.create_task(receiver.serve(
        on_frame, None, None,
        lambda sid: None,
        lambda sid: sink.close(f"transport session {sid} ended"),
        sink.on_capture_meta, sink.on_capture_raw,
    ))
    await asyncio.sleep(0.4)
    try:
        session_id = "S-SELFTEST-SYNTHETIC"
        ack = await _client(port, audio16k, session_id, "W-selftest")
        await asyncio.sleep(0.5)
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    result = sink.last_result
    if result is None:
        raise RuntimeError("capture session was never finalised")

    session_dir = Path(result["session_dir"])
    validation = result["validation"]
    metadata = result["metadata"]

    # The whole point: this must NOT be usable as Mentra evidence.
    guard_refused = False
    try:
        assert_mentra_hardware(session_dir)
    except CaptureMetadataError:
        guard_refused = True

    raw_info = sf.info(str(session_dir / "raw.wav"))
    derived_info = sf.info(str(session_dir / "derived_16k.wav"))

    return {
        "ack": ack,
        "session_dir": str(session_dir),
        "derived_frames_received": received_derived["frames"],
        "validation_status": validation["status"],
        "n_fail": validation["n_fail"],
        "n_warn": validation["n_warn"],
        "failures": validation["failures"],
        "raw": {"samplerate": raw_info.samplerate, "channels": raw_info.channels,
                "duration_s": round(raw_info.duration, 3)},
        "derived": {"samplerate": derived_info.samplerate, "channels": derived_info.channels,
                    "duration_s": round(derived_info.duration, 3)},
        "is_mentra_hardware_audio": metadata["is_mentra_hardware_audio"],
        "source_kind": metadata["source_kind"],
        "mentra_guard_refused_synthetic_capture": guard_refused,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18790,
                        help="unused localhost port; never the production receiver's 8765")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--capture-root", type=Path,
                        default=REPO_ROOT / "captures/selftest")
    args = parser.parse_args()
    if args.port == 8765:
        raise SystemExit("refusing to bind the production receiver port 8765")

    result = asyncio.run(run(args.port, args.seconds, args.capture_root))
    print(json.dumps(result, indent=2))

    ok = (
        result["validation_status"] in ("PASS", "WARN")
        and result["n_fail"] == 0
        and result["raw"]["samplerate"] == NATIVE_RATE
        and result["derived"]["samplerate"] == TRANSPORT_RATE
        and result["is_mentra_hardware_audio"] is False
        and result["mentra_guard_refused_synthetic_capture"] is True
    )
    print("CAPTURE_SELFTEST:", "PASS" if ok else "FAIL")
    print("NOTE: SYNTHETIC_TEST_INPUT_NOT_MENTRA_AUDIO -- this proves the capture "
          "tool works, and says nothing about Mentra Live.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
