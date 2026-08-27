"""GeoWearNet E0 physical-heuristic feature extraction.

Phase 5/6 of the GeoWearNet research brief: short-time, interpretable,
physically-meaningful features computed directly on PCM audio with NO
per-utterance amplitude normalization, no peak-normalization, no forced
equal-RMS, and no CMVN. Absolute level is the whole point of this track's
hypothesis (wearer-vs-environment geometry), so destroying it before feature
extraction would make the entire experiment meaningless.

PCM SCALE CONVENTION (recorded explicitly per Phase 6):
  Audio is loaded via `soundfile.read(..., dtype='float32')`, which for
  16-bit PCM WAV/FLAC divides int16 samples by 32768.0, giving float32
  samples in [-1.0, 1.0]. All RMS/peak/clipping features below are computed
  directly on that scale. "Clipping" is therefore defined relative to
  full-scale (|x| >= CLIP_THRESHOLD, default 0.99), matching how a real
  16-bit ADC would clip. No other rescaling happens anywhere in this module.

This module is standalone and does not import anything from
`training/train.py` or the MentraWearNet-specific code paths (per the
brief's Phase 44 instruction to keep GeoWearNet isolated).
"""
from __future__ import annotations

import dataclasses
from typing import List

import numpy as np

SAMPLE_RATE = 16000
FRAME_HOP_MS = 10.0
FRAME_WIN_MS = 25.0
FRAME_HOP = int(round(SAMPLE_RATE * FRAME_HOP_MS / 1000.0))   # 160
FRAME_WIN = int(round(SAMPLE_RATE * FRAME_WIN_MS / 1000.0))   # 400
CLIP_THRESHOLD = 0.99
EPS = 1e-10

FEATURE_NAMES = [
    "log_rms",
    "peak",
    "crest_factor",
    "clipping_fraction",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_rolloff85",
    "spectral_tilt",
    "lf_energy_ratio",
    "mf_energy_ratio",
    "hf_energy_ratio",
    "spectral_flux",
    "energy_derivative",
    "harmonicity",
]

# Band edges (Hz) for the coarse LF/MF/HF energy-ratio features. Chosen to
# roughly separate "chest/near-field body resonance" (<500Hz), "core speech
# formant band" (500-2500Hz) and "fricative/consonant/high-frequency
# near-field detail" (>2500Hz) -- kept few and interpretable per Phase 5's
# "don't add hundreds of arbitrary features" instruction.
LF_HZ = 500.0
HF_HZ = 2500.0


def num_frames(n_samples: int, win: int = FRAME_WIN, hop: int = FRAME_HOP) -> int:
    if n_samples < win:
        return 1 if n_samples > 0 else 0
    return 1 + (n_samples - win) // hop


def frame_signal(x: np.ndarray, win: int = FRAME_WIN, hop: int = FRAME_HOP) -> np.ndarray:
    """Causal-friendly framing: frame i covers samples
    [i*hop, i*hop+win) (right-padded with zeros for the final frame),
    i.e. each frame only looks at audio up to `win` samples ahead of its
    start -- no whole-utterance lookahead of any kind happens here."""
    n = len(x)
    nf = num_frames(n, win, hop)
    # Vectorized (Workstream D): the original per-frame Python loop was ~25 ms
    # per 6-second clip and ran inside every DataLoader worker. This produces a
    # BIT-IDENTICAL array -- it is a gather, not a recomputation -- which the
    # test suite asserts against the original loop.
    need = hop * (nf - 1) + win
    xd = np.asarray(x, dtype=np.float64)
    if len(xd) < need:
        xd = np.concatenate([xd, np.zeros(need - len(xd), dtype=np.float64)])
    return np.ascontiguousarray(
        np.lib.stride_tricks.sliding_window_view(xd[:need], win)[::hop]
    )


def _spectral_features(frames: np.ndarray, sr: int = SAMPLE_RATE):
    """Vectorized rFFT-based spectral features for a (n_frames, win) array."""
    nf, win = frames.shape
    window = np.hanning(win)
    windowed = frames * window[None, :]
    spec = np.fft.rfft(windowed, axis=1)
    mag = np.abs(spec) + EPS
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)

    power = mag ** 2
    total_power = power.sum(axis=1) + EPS

    centroid = (power * freqs[None, :]).sum(axis=1) / total_power

    cumpower = np.cumsum(power, axis=1)
    rolloff_idx = np.argmax(cumpower >= 0.85 * total_power[:, None], axis=1)
    rolloff = freqs[rolloff_idx]

    # Spectral tilt: slope of log-magnitude vs log-frequency (skip DC bin).
    log_f = np.log(freqs[1:] + 1.0)
    log_m = np.log(mag[:, 1:])
    log_f_c = log_f - log_f.mean()
    denom = (log_f_c ** 2).sum() + EPS
    tilt = (log_m * log_f_c[None, :]).sum(axis=1) / denom

    lf_mask = freqs < LF_HZ
    mf_mask = (freqs >= LF_HZ) & (freqs < HF_HZ)
    hf_mask = freqs >= HF_HZ
    lf_ratio = power[:, lf_mask].sum(axis=1) / total_power
    mf_ratio = power[:, mf_mask].sum(axis=1) / total_power
    hf_ratio = power[:, hf_mask].sum(axis=1) / total_power

    norm_mag = mag / (np.linalg.norm(mag, axis=1, keepdims=True) + EPS)
    flux = np.zeros(nf)
    if nf > 1:
        diff = norm_mag[1:] - norm_mag[:-1]
        flux[1:] = np.sqrt((diff ** 2).sum(axis=1))

    return centroid, rolloff, tilt, lf_ratio, mf_ratio, hf_ratio, flux


def _harmonicity(frames: np.ndarray) -> np.ndarray:
    """Cheap harmonicity proxy: normalized-autocorrelation peak (excluding
    lag 0) over a plausible pitch-period lag range for speech (80-400 Hz).
    Not a full pitch tracker -- deliberately cheap per Phase 5. Vectorized
    across all frames at once via FFT-based autocorrelation (batched),
    rather than a per-frame Python loop over lags, purely for speed --
    the math (normalized autocorrelation peak in a bounded lag band) is
    unchanged."""
    sr = SAMPLE_RATE
    min_lag = int(sr / 400.0)
    max_lag = int(sr / 80.0)
    nf, win = frames.shape
    max_lag = min(max_lag, win - 1)
    if max_lag <= min_lag or nf == 0:
        return np.zeros(nf)

    x = frames - frames.mean(axis=1, keepdims=True)
    energy0 = (x ** 2).sum(axis=1) + EPS

    nfft = 1
    while nfft < 2 * win:
        nfft *= 2
    X = np.fft.rfft(x, n=nfft, axis=1)
    ac = np.fft.irfft(X * np.conj(X), n=nfft, axis=1)[:, : win]  # (nf, win) autocorr, lag 0..win-1

    band = ac[:, min_lag : max_lag + 1]
    peak = band.max(axis=1)
    r = peak / energy0
    return np.clip(r, 0.0, 1.0)


def extract_features(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Returns (n_frames, len(FEATURE_NAMES)) float32 array of physical
    features. `x` must already be at `sr` and in the float32 [-1,1] PCM
    convention documented at the top of this module -- NO normalization is
    applied here."""
    assert sr == SAMPLE_RATE, "resample before calling extract_features"
    x = np.asarray(x, dtype=np.float64)
    frames = frame_signal(x)
    nf = frames.shape[0]
    if nf == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

    rms = np.sqrt((frames ** 2).mean(axis=1) + EPS)
    log_rms = np.log(rms + EPS)
    peak = np.abs(frames).max(axis=1)
    crest = peak / (rms + EPS)
    clip_frac = (np.abs(frames) >= CLIP_THRESHOLD).mean(axis=1)

    signs = np.sign(frames)
    signs[signs == 0] = 1
    zcr = (np.abs(np.diff(signs, axis=1)) > 0).mean(axis=1)

    centroid, rolloff, tilt, lf_r, mf_r, hf_r, flux = _spectral_features(frames)
    harmonicity = _harmonicity(frames)

    energy_deriv = np.zeros(nf)
    if nf > 1:
        energy_deriv[1:] = log_rms[1:] - log_rms[:-1]

    feats = np.stack(
        [
            log_rms, peak, crest, clip_frac, zcr,
            centroid, rolloff, tilt, lf_r, mf_r, hf_r, flux,
            energy_deriv, harmonicity,
        ],
        axis=1,
    ).astype(np.float32)
    assert feats.shape[1] == len(FEATURE_NAMES)
    return feats


@dataclasses.dataclass
class FeatureGroup:
    name: str
    indices: List[int]


def feature_group_indices(names: List[str]) -> List[int]:
    return [FEATURE_NAMES.index(n) for n in names]


FEATURE_GROUPS = {
    "rms_only": FeatureGroup("rms_only", feature_group_indices(["log_rms"])),
    "peak_only": FeatureGroup("peak_only", feature_group_indices(["peak"])),
    "clipping_only": FeatureGroup("clipping_only", feature_group_indices(["clipping_fraction"])),
    "amplitude_only": FeatureGroup(
        "amplitude_only",
        feature_group_indices(["log_rms", "peak", "crest_factor", "clipping_fraction"]),
    ),
    "spectral_only": FeatureGroup(
        "spectral_only",
        feature_group_indices(
            [
                "spectral_centroid", "spectral_rolloff85", "spectral_tilt",
                "lf_energy_ratio", "mf_energy_ratio", "hf_energy_ratio",
                "spectral_flux", "harmonicity", "zero_crossing_rate",
            ]
        ),
    ),
    "amplitude_plus_spectral": FeatureGroup(
        "amplitude_plus_spectral",
        feature_group_indices(
            [
                "log_rms", "peak", "crest_factor", "clipping_fraction",
                "spectral_centroid", "spectral_rolloff85", "spectral_tilt",
                "lf_energy_ratio", "mf_energy_ratio", "hf_energy_ratio",
                "spectral_flux", "harmonicity", "zero_crossing_rate",
                "energy_derivative",
            ]
        ),
    ),
    "all": FeatureGroup("all", list(range(len(FEATURE_NAMES)))),
}
