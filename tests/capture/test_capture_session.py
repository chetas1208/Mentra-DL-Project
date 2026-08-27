"""G4 WS6/WS41 -- research capture session and validation.

These tests exercise the SOFTWARE CONTRACT of the capture tool using clearly
synthetic inputs. They deliberately do not, and cannot, say anything about
real Mentra hardware: what they prove is that when a bad take arrives, the
tool catches it, and that synthetic or browser-mic audio can never be
promoted to "Mentra evidence" by accident.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from mentra.capture import validation as V
from mentra.capture.session import (CaptureMetadata, CaptureMetadataError, CaptureSession,
                                    anonymous_wearer_id, assert_mentra_hardware)


def _metadata(**overrides) -> CaptureMetadata:
    base = dict(
        session_id="S-TEST", wearer_id="W-test", condition="unit_test",
        source_kind="SYNTHETIC_TEST_INPUT_NOT_MENTRA_AUDIO",
        started_utc="2026-08-27T00:00:00+00:00",
        raw_sample_rate=48000, raw_channels=1,
    )
    base.update(overrides)
    return CaptureMetadata(**base)


def _speech_like(seconds: float, sample_rate: int, seed: int = 0) -> np.ndarray:
    """Amplitude-modulated noise: bursts above a quiet floor, which is what
    the activity check looks for. Not speech, and never labelled as such."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    envelope = 0.02 + 0.4 * (np.sin(2 * np.pi * 0.7 * t) > 0.2)
    return (rng.standard_normal(n) * envelope).astype(np.float32)


def _pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


# --- metadata contract ------------------------------------------------------
def test_condition_and_source_are_mandatory():
    with pytest.raises(CaptureMetadataError, match="condition is required"):
        _metadata(condition="")
    with pytest.raises(CaptureMetadataError, match="unknown source_kind"):
        _metadata(source_kind="MENTRA_PROBABLY")
    with pytest.raises(CaptureMetadataError, match="missing"):
        CaptureMetadata.from_json({"session_id": "S", "wearer_id": "W"})


def test_anonymous_wearer_ids_are_opaque_and_unique():
    ids = {anonymous_wearer_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(i.startswith("W-") and len(i) == 10 for i in ids)


@pytest.mark.parametrize("source,expected", [
    ("MENTRA_LIVE_GLASSES_MIC", True),
    ("MENTRA_COMPATIBLE_GLASSES_MIC", True),
    ("BROWSER_DEFAULT_MIC", False),
    ("SYNTHETIC_TEST_INPUT_NOT_MENTRA_AUDIO", False),
])
def test_mentra_hardware_flag_follows_the_declared_source(source, expected):
    assert _metadata(source_kind=source).is_mentra_hardware_audio is expected


# --- end-to-end session writing --------------------------------------------
def test_session_writes_both_streams_metadata_and_validation(tmp_path):
    metadata = _metadata()
    session = CaptureSession(metadata, root=tmp_path)
    raw = _speech_like(4.0, 48000, seed=1)
    derived = _speech_like(4.0, 16000, seed=1)
    session.add_raw_pcm16(_pcm16(raw), 48000, 1)
    session.add_derived_pcm16(_pcm16(derived))
    result = session.close()

    session_dir = tmp_path / "S-TEST"
    assert (session_dir / "raw.wav").is_file()
    assert (session_dir / "derived_16k.wav").is_file()
    assert sf.info(str(session_dir / "raw.wav")).samplerate == 48000
    assert sf.info(str(session_dir / "derived_16k.wav")).samplerate == 16000

    written = json.loads((session_dir / "metadata.json").read_text())
    assert written["is_mentra_hardware_audio"] is False
    assert written["condition"] == "unit_test"
    assert result["validation"]["status"] in ("PASS", "WARN")
    assert result["validation"]["n_fail"] == 0


def test_session_rejects_a_raw_chunk_whose_format_changed(tmp_path):
    session = CaptureSession(_metadata(), root=tmp_path)
    with pytest.raises(ValueError, match="does not match the declared"):
        session.add_raw_pcm16(_pcm16(np.zeros(160, np.float32)), 44100, 1)
    with pytest.raises(ValueError, match="channel count"):
        session.add_raw_pcm16(_pcm16(np.zeros(160, np.float32)), 48000, 2)


def test_mentra_guard_refuses_a_synthetic_capture(tmp_path):
    session = CaptureSession(_metadata(), root=tmp_path)
    session.add_raw_pcm16(_pcm16(_speech_like(2.0, 48000)), 48000, 1)
    session.add_derived_pcm16(_pcm16(_speech_like(2.0, 16000)))
    session.close()
    with pytest.raises(CaptureMetadataError, match="NOT Mentra hardware audio"):
        assert_mentra_hardware(tmp_path / "S-TEST")


def test_mentra_guard_accepts_a_declared_mentra_capture(tmp_path):
    session = CaptureSession(
        _metadata(session_id="S-MENTRA", source_kind="MENTRA_LIVE_GLASSES_MIC"),
        root=tmp_path)
    session.add_raw_pcm16(_pcm16(_speech_like(2.0, 48000)), 48000, 1)
    session.add_derived_pcm16(_pcm16(_speech_like(2.0, 16000)))
    session.close()
    metadata = assert_mentra_hardware(tmp_path / "S-MENTRA")
    assert metadata["is_mentra_hardware_audio"] is True


# --- validation catches the failures a pilot session actually suffers -------
def test_validation_flags_a_silent_take():
    checks = V.validate_audio(np.zeros(16000 * 5, np.float32), 16000, 1)
    statuses = {c.name: c.status for c in checks}
    assert statuses["audio.rms"] == "FAIL"
    assert V.overall_status(checks) == "FAIL"


def test_validation_flags_a_clipped_take():
    x = np.ones(16000 * 5, dtype=np.float32)
    checks = V.validate_audio(x, 16000, 1)
    assert {c.name: c.status for c in checks}["audio.clipping"] == "FAIL"


def test_validation_flags_a_too_short_take():
    checks = V.validate_audio(_speech_like(0.4, 16000), 16000, 1)
    assert {c.name: c.status for c in checks}["audio.duration_s"] == "FAIL"


def test_validation_flags_the_wrong_sample_rate_and_channel_count():
    checks = V.validate_audio(_speech_like(3.0, 16000), 8000, 2,
                              expected_sample_rate=16000, expected_channels=1)
    statuses = {c.name: c.status for c in checks}
    assert statuses["audio.sample_rate"] == "FAIL"
    assert statuses["audio.channels"] == "FAIL"


def test_validation_flags_non_finite_samples():
    x = _speech_like(3.0, 16000).copy()
    x[100] = np.nan
    assert {c.name: c.status for c in V.validate_audio(x, 16000, 1)}["audio.finite"] == "FAIL"


def test_validation_warns_when_nothing_rises_above_the_noise_floor():
    rng = np.random.default_rng(4)
    flat = (rng.standard_normal(16000 * 5) * 0.05).astype(np.float32)
    checks = V.validate_audio(flat, 16000, 1)
    assert {c.name: c.status for c in checks}["audio.speech_activity"] == "WARN"


def test_validation_flags_an_unreadable_or_corrupt_file(tmp_path):
    missing = V.validate_file(tmp_path / "nope.wav")
    assert missing[0].status == "FAIL"

    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"RIFF____not really a wav file at all")
    checks = V.validate_file(corrupt)
    assert checks[0].status == "FAIL"
    assert V.overall_status(checks) == "FAIL"


def test_duration_mismatch_between_streams_is_caught():
    """The single most likely silent failure in a browser capture chain:
    the raw and derived streams drifting apart. Invisible in either file
    alone, so it is checked across them."""
    assert V.check_rate_consistency(10.0, 10.05).status == "PASS"
    assert V.check_rate_consistency(10.0, 7.0).status == "FAIL"


def test_a_failed_take_is_still_written_to_disk(tmp_path):
    """A FAIL that exists on disk is worth far more than a take that
    silently vanished while the wearer was still in the room."""
    session = CaptureSession(_metadata(session_id="S-BAD"), root=tmp_path)
    session.add_raw_pcm16(_pcm16(np.zeros(48000 * 3, np.float32)), 48000, 1)
    session.add_derived_pcm16(_pcm16(np.zeros(16000 * 3, np.float32)))
    result = session.close()
    assert result["validation"]["status"] == "FAIL"
    assert (tmp_path / "S-BAD" / "validation.json").is_file()
    assert (tmp_path / "S-BAD" / "raw.wav").is_file()
