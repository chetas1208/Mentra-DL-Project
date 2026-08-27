"""P1.19 -- RNNoise baseline (P1.4) and ASR adapter contract (P1.9) tests.

These skip cleanly when a backend is unavailable rather than failing, because
both are declared OPTIONAL in the brief -- but when they ARE available the
contract is tested for real.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.agent_audio.asr import (ASRResult, ASRWord, NullASR, audio_key,
                                        get_backend, normalize_text)
from evaluation.agent_audio.denoise import Denoiser, roundtrip_null_cost

SR = 16000


# ---------------------------------------------------------------------------
# RNNoise
# ---------------------------------------------------------------------------
def _dn():
    d = Denoiser(sr=SR)
    if not d.available:
        pytest.skip("RNNoise binding unavailable")
    return d


def test_rnnoise_suppresses_stationary_noise():
    d = _dn()
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.05, SR * 2).astype(np.float32)
    r = d.process(x)
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    rms_out = float(np.sqrt(np.mean(r.audio.astype(np.float64) ** 2)))
    assert rms_out < rms_in * 0.5, "RNNoise failed to suppress white noise"


def test_rnnoise_preserves_output_length():
    d = _dn()
    x = np.zeros(SR * 3 + 137, dtype=np.float32)
    assert len(d.process(x).audio) == len(x)


def test_rnnoise_handles_amplitude_above_full_scale():
    """Regression: real MMCSG audio preserves absolute amplitude and polyphase
    resampling overshoots slightly, so some samples exceed +-1.0. pyrnnoise's
    float->int16 heuristic silently skips conversion in that case and then
    asserts. We must convert explicitly and keep going."""
    d = _dn()
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.3, SR).astype(np.float32)
    x[100] = 1.05
    x[200] = -1.08
    r = d.process(x)                      # must not raise
    assert len(r.audio) == len(x)
    assert r.n_samples_clipped >= 2, "clipping must be counted and reported, not hidden"


def test_rnnoise_reports_latency_and_is_causal_framed():
    d = _dn()
    lat = d.algorithmic_latency_ms()
    assert lat["rnnoise_lookahead_ms"] == 0.0
    assert lat["rnnoise_frame_ms"] == pytest.approx(10.0)


def test_rnnoise_realtime_capable_on_cpu():
    d = _dn()
    x = np.random.default_rng(2).normal(0, 0.05, SR * 4).astype(np.float32)
    r = d.process(x)
    assert r.rtf < 1.0, f"not real-time capable: RTF={r.rtf:.3f}"


def test_resample_roundtrip_is_near_transparent_on_speech_band_content():
    """Any RNNoise 'improvement' smaller than the 16k->48k->16k round-trip
    error is not a real effect, so the round-trip must be transparent for the
    content we care about.

    Measured on this machine: 56 dB SNR on band-limited (<3.8 kHz) content and
    36.5 dB on real speech, but only 18.7 dB on full-band white noise -- the
    anti-alias filter legitimately removes the top octave, which white noise
    (uniquely) fills. So the assertion is made on speech-band content, which
    is what the pipeline actually carries, and the white-noise figure is
    documented here rather than asserted."""
    from scipy.signal import butter, lfilter
    x = np.random.default_rng(3).normal(0, 0.1, SR).astype(np.float32)
    b, a = butter(6, 3800 / (SR / 2))
    speechband = lfilter(b, a, x).astype(np.float32)
    r = roundtrip_null_cost(speechband, SR)
    assert r["roundtrip_snr_db"] > 30.0, f"round-trip not transparent: {r}"
    assert 0.95 < r["roundtrip_rms_ratio"] < 1.05


def test_denoiser_degrades_gracefully_when_unavailable():
    d = Denoiser(sr=SR)
    d.available = False
    x = np.ones(1000, dtype=np.float32)
    r = d.process(x)
    assert r.available is False
    assert np.array_equal(r.audio, x), "unavailable denoiser must pass audio through"


# ---------------------------------------------------------------------------
# ASR adapter contract
# ---------------------------------------------------------------------------
def test_null_backend_is_explicitly_unavailable():
    b = NullASR()
    r = b.transcribe(np.zeros(SR, dtype=np.float32), SR)
    assert r.available is False and r.text == "" and not r.has_timestamps


def test_null_backend_never_fakes_a_perfect_score():
    """A missing ASR must not read as 0% WER."""
    r = NullASR().transcribe(np.zeros(SR, np.float32), SR)
    assert r.available is False


def test_audio_key_is_content_addressed():
    a = np.zeros(100, dtype=np.float32)
    b = a.copy(); b[0] = 1e-6
    assert audio_key(a, SR, "x") == audio_key(a.copy(), SR, "x")
    assert audio_key(a, SR, "x") != audio_key(b, SR, "x")
    assert audio_key(a, SR, "x") != audio_key(a, SR, "y"), "backend id must be in the key"
    assert audio_key(a, SR, "x") != audio_key(a, 8000, "x"), "sample rate must be in the key"


def test_asr_result_roundtrips_through_json():
    r = ASRResult("hello world", [ASRWord("hello", 0.0, 0.4), ASRWord("world", 0.5, 0.9)],
                  0.1, 1.0, "test/backend")
    r2 = ASRResult.from_json(r.to_json())
    assert r2.text == r.text and len(r2.words) == 2
    assert r2.words[1].word == "world" and r2.cached is True


def test_rtf_is_computed_not_stored():
    r = ASRResult("x", [], 0.5, 10.0, "b")
    assert r.rtf == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# real local backend (skipped if the model is absent)
# ---------------------------------------------------------------------------
def _asr():
    b = get_backend("auto")
    if not b.available:
        pytest.skip("no local ASR backend available")
    return b


def test_local_asr_transcribes_and_timestamps():
    b = _asr()
    import soundfile as sf
    from evaluation.agent_audio.asr import DEFAULT_MODEL_DIR
    wav = DEFAULT_MODEL_DIR / "test_wavs/0.wav"
    if not wav.exists():
        pytest.skip("no test wav shipped with the model")
    x, sr = sf.read(str(wav), dtype="float32")
    r = b.transcribe(x, sr)
    assert r.available and len(r.text) > 0
    assert r.has_timestamps
    assert all(w.end_s >= w.start_s for w in r.words)
    assert "country" in normalize_text(r.text)


def test_local_asr_is_faster_than_realtime():
    b = _asr()
    x = np.random.default_rng(4).normal(0, 0.01, SR * 3).astype(np.float32)
    r = b.transcribe(x, SR)
    assert r.rtf < 1.0, f"ASR not real-time capable: RTF={r.rtf:.2f}"


def test_asr_cache_returns_identical_transcript(tmp_path):
    b = _asr()
    x = np.random.default_rng(5).normal(0, 0.02, SR).astype(np.float32)
    r1 = b.transcribe_cached(x, SR, cache_dir=tmp_path)
    r2 = b.transcribe_cached(x, SR, cache_dir=tmp_path)
    assert r1.text == r2.text
    assert r2.cached is True and r1.cached is False


def test_asr_cache_distinguishes_different_audio(tmp_path):
    b = _asr()
    rng = np.random.default_rng(6)
    x = rng.normal(0, 0.02, SR).astype(np.float32)
    y = rng.normal(0, 0.02, SR).astype(np.float32)
    b.transcribe_cached(x, SR, cache_dir=tmp_path)
    r = b.transcribe_cached(y, SR, cache_dir=tmp_path)
    assert r.cached is False, "different audio must not hit the same cache entry"
