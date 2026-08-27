"""G4 WS6/WS41 -- the capture path over the real transport.

Runs the real ``MentraRemoteReceiver`` and the real ``CaptureSink`` on an
isolated localhost port and pushes real MTRA frames through them. The audio is
synthetic and is labelled as such at every layer; what is under test is the
plumbing and the honesty guards, never a claim about any device.
"""
from __future__ import annotations

import json
import socket

import numpy as np
import pytest

from scripts.mentra import capture_selftest as S


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.asyncio
async def test_capture_selftest_round_trip_writes_a_valid_session(tmp_path):
    result = await S.run(_free_port(), seconds=3.0, capture_root=tmp_path)

    assert result["ack"]["status"] == "OPEN"
    assert result["validation_status"] in ("PASS", "WARN")
    assert result["n_fail"] == 0, result["failures"]
    # both streams survived at their own rates -- the raw stream is NOT
    # silently resampled to the transport rate
    assert result["raw"]["samplerate"] == S.NATIVE_RATE
    assert result["derived"]["samplerate"] == S.TRANSPORT_RATE
    assert result["raw"]["duration_s"] == pytest.approx(result["derived"]["duration_s"], abs=0.1)
    assert result["derived_frames_received"] > 0


@pytest.mark.asyncio
async def test_synthetic_capture_can_never_be_promoted_to_mentra_evidence(tmp_path):
    result = await S.run(_free_port(), seconds=2.0, capture_root=tmp_path)

    assert result["source_kind"] == "SYNTHETIC_TEST_INPUT_NOT_MENTRA_AUDIO"
    assert result["is_mentra_hardware_audio"] is False
    # the guard downstream evaluation code must call actually refuses it
    assert result["mentra_guard_refused_synthetic_capture"] is True

    metadata = json.loads((tmp_path / "S-SELFTEST-SYNTHETIC" / "metadata.json").read_text())
    assert metadata["is_mentra_hardware_audio"] is False


def test_selftest_refuses_the_production_receiver_port(monkeypatch):
    monkeypatch.setattr("sys.argv", ["capture_selftest.py", "--port", "8765"])
    with pytest.raises(SystemExit, match="8765"):
        S.main()


def test_upsampled_test_audio_keeps_the_same_duration():
    audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    native = S.upsample(audio, factor=3)
    assert len(native) == len(audio) * 3
    assert np.max(np.abs(native)) <= 0.5 + 1e-6
