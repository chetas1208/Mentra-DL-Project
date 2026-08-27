"""P1.4 -- machinery-noise baseline (RNNoise), optional and isolated.

STATUS: **AVAILABLE**. `pyrnnoise` 0.4.3 ships a manylinux wheel containing a
prebuilt `librnnoise.so`, installed into this repo's `.venv` with `--no-deps`
(its declared `audiolab`/`tqdm` deps are only used by its file-I/O helper,
which we do not use -- and installing them would have risked upgrading the
numpy/torch the live GPU campaign is running on). We bind the low-level
ctypes module `pyrnnoise/rnnoise.py` directly, which imports nothing but
`ctypes` + `numpy`.

RNNoise itself is Xiph's BSD-3-Clause real-time noise suppressor (GRU-based,
~85k params, 10 ms frames, 48 kHz native). It is used here strictly as an
EXTERNAL, FROZEN, OPTIONAL baseline:
  * it is NOT retrained, NOT fine-tuned, NOT part of GeoWearNet;
  * it is NOT in GeoWearNet's training or inference path;
  * it lives behind `Denoiser.available` so the entire evaluation still runs
    if it is missing.

WHY IT IS THE RIGHT BASELINE AND ITS KNOWN LIMIT
RNNoise suppresses non-speech noise (impact wrenches, compressors, HVAC).
It does NOT and cannot suppress a *coworker's valid speech* -- to RNNoise a
bystander is speech and speech is what it preserves. That limitation is the
entire reason GeoWearNet exists, and the P1.5 matrix measures it directly:
compare RNNOISE's bystander leakage against GEOWEAR_GATE's.

Sample rate: RNNoise is 48 kHz native. Our pipeline is 16 kHz. We resample
16k->48k, denoise, resample back with the SAME polyphase method the MMCSG
audio layer uses (`scipy.signal.resample_poly`), and report the round-trip
cost so it is never mistaken for a denoising effect.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import site
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

RNNOISE_SR = 48000


def _load_ll():
    """Import pyrnnoise's low-level ctypes module WITHOUT executing the
    package `__init__`, which pulls in the unneeded `audiolab` dependency."""
    for base in list(site.getsitepackages()) + [p for p in sys.path if p]:
        p = Path(base) / "pyrnnoise" / "rnnoise.py"
        if p.exists():
            spec = importlib.util.spec_from_file_location("_agent_audio_rnnoise_ll", str(p))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    return None


_LL = None
_LL_TRIED = False


def _ll():
    global _LL, _LL_TRIED
    if not _LL_TRIED:
        _LL_TRIED = True
        try:
            _LL = _load_ll()
        except Exception:
            _LL = None
    return _LL


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.ascontiguousarray(x, dtype=np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


@dataclasses.dataclass
class DenoiseResult:
    audio: np.ndarray
    speech_prob: np.ndarray        # RNNoise's own per-10ms VAD-ish probability
    latency_s: float
    rtf: float
    backend: str
    available: bool = True
    n_samples_clipped: int = 0     # samples past full scale at the int16 boundary


class Denoiser:
    """Frozen RNNoise wrapper. Streaming-shaped (10 ms frames, persistent
    state) so measured RTF/latency reflect real-time use, not a batch mode."""

    backend = "rnnoise/xiph-bsd3/pyrnnoise-0.4.3"

    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.ll = _ll()
        self.available = self.ll is not None
        self._state = None

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> None:
        if self._state is not None and self.ll is not None:
            self.ll.destroy(self._state)
        self._state = None

    def __del__(self):
        try:
            self.reset()
        except Exception:
            pass

    # -- processing ----------------------------------------------------------
    def process(self, audio: np.ndarray) -> DenoiseResult:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if not self.available:
            return DenoiseResult(audio.copy(), np.zeros(0, np.float32), 0.0, 0.0,
                                 "unavailable", available=False)
        dur = len(audio) / float(self.sr)
        t0 = time.perf_counter()
        x48 = _resample(audio, self.sr, RNNOISE_SR)
        F = self.ll.FRAME_SIZE
        n_pad = (-len(x48)) % F
        if n_pad:
            x48 = np.concatenate([x48, np.zeros(n_pad, np.float32)])
        # Convert to int16 EXPLICITLY rather than relying on pyrnnoise's
        # float->int16 heuristic, which only fires when the frame already lies
        # inside [-1, 1] and otherwise falls through to an assert. Real MMCSG
        # audio preserves absolute PCM amplitude and polyphase resampling adds
        # a little overshoot, so a handful of samples per recording sit just
        # past full scale (measured: ~100-600 samples in a 3-minute file).
        # Those samples are clipped -- unavoidable when handing 16-bit audio to
        # a 16-bit denoiser -- and the clipped fraction is reported rather than
        # hidden, so it can never be mistaken for a denoising effect.
        n_clipped = int(np.count_nonzero(np.abs(x48) > 1.0))
        x48_i16 = np.clip(x48 * 32767.0, -32768.0, 32767.0).astype(np.int16)

        # Keep RNNoise's recurrent state for the life of this Denoiser.  The
        # live receiver calls process once per transport frame; creating and
        # destroying state there would silently turn a streaming GRU into a
        # sequence of unrelated single-frame denoisers.  Callers reset at an
        # actual session boundary through AudioFrontend.reset().
        if self._state is None:
            self._state = self.ll.create()
        out, probs = [], []
        for i in range(0, len(x48_i16), F):
            f, p = self.ll.process_mono_frame(self._state, x48_i16[i:i + F])
            out.append(f)
            probs.append(p)
        y48 = np.concatenate(out).astype(np.float32) / 32768.0
        y = _resample(y48, RNNOISE_SR, self.sr)[: len(audio)]
        if len(y) < len(audio):
            y = np.pad(y, (0, len(audio) - len(y)))
        lat = time.perf_counter() - t0
        return DenoiseResult(audio=y.astype(np.float32),
                             speech_prob=np.asarray(probs, dtype=np.float32),
                             latency_s=lat, rtf=lat / max(dur, 1e-9),
                             backend=self.backend, n_samples_clipped=n_clipped)

    # -- honest cost accounting ---------------------------------------------
    def algorithmic_latency_ms(self) -> Dict[str, float]:
        """RNNoise is a 10 ms-frame causal GRU: one frame of algorithmic
        latency. The resampler adds its own filter group delay; polyphase
        FIR group delay at 16<->48 kHz is well under a millisecond and is
        reported rather than ignored."""
        return {
            "rnnoise_frame_ms": 1000.0 * self.ll.FRAME_SIZE / RNNOISE_SR if self.available else None,
            "rnnoise_lookahead_ms": 0.0,
            "resampler_group_delay_ms_est": 0.5,
            "latency_algorithmic_ms": (1000.0 * self.ll.FRAME_SIZE / RNNOISE_SR + 0.5)
                                       if self.available else None,
        }


def roundtrip_null_cost(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """Measures what the 16k->48k->16k resample ALONE costs, with no
    denoising. Any RNNoise 'improvement' smaller than this is not real."""
    y = _resample(_resample(audio, sr, RNNOISE_SR), RNNOISE_SR, sr)[: len(audio)]
    if len(y) < len(audio):
        y = np.pad(y, (0, len(audio) - len(y)))
    err = audio.astype(np.float64) - y.astype(np.float64)
    return {
        "roundtrip_snr_db": float(10 * np.log10(
            (audio.astype(np.float64) ** 2).sum() / max((err ** 2).sum(), 1e-20))),
        "roundtrip_rms_ratio": float(np.sqrt(np.mean(y ** 2)) /
                                     max(np.sqrt(np.mean(audio ** 2)), 1e-12)),
    }


def availability_report() -> Dict[str, object]:
    d = Denoiser()
    r: Dict[str, object] = {
        "backend": d.backend,
        "available": bool(d.available),
        "status": "AVAILABLE" if d.available else "BLOCKED_NO_RNNOISE_BINDING",
        "license": "BSD-3-Clause (RNNoise/Xiph); python wrapper Apache-2.0",
        "note": ("bound via pyrnnoise's low-level ctypes module directly; the "
                 "package __init__ is bypassed because its audiolab dependency "
                 "is only needed for file I/O and installing it would risk "
                 "perturbing the numpy/torch used by the live GPU campaign"),
    }
    if d.available:
        r.update(d.algorithmic_latency_ms())
        r["native_sample_rate"] = RNNOISE_SR
        r["frame_size"] = int(d.ll.FRAME_SIZE)
    return r


if __name__ == "__main__":
    import json
    print(json.dumps(availability_report(), indent=2))
