"""GeoWearNet simulator generations S1 and S2 (Workstream E).

S0 (`training/geowearnet/simulate.py`) is preserved untouched as the historical
baseline. This module adds:

  S1 -- PHYSICALLY PARAMETRIC. Real geometry (distance, azimuth, direct-path
        delay, spherical spreading, ISO 9613-1 air absorption), a parametric
        frequency-dependent head-shadow/near-field colouration, physically
        derived direct-to-reverberant ratio from room volume + RT60, discrete
        early reflections and a band-split late tail.

  S2 -- RIGID-SPHERE HEAD. Identical to S1 except the parametric colouration is
        replaced by the exact analytic rigid-sphere response
        (`acoustics.rigid_sphere_response`, via LUT). This is the published
        analytical->rigid-sphere->HATS progression (arXiv:2603.02724) stage 2.

  S3 -- MEASURED WEARABLE/HATS RESPONSES. **BLOCKED**: no open measured
        wearable or head-and-torso response corpus of verified, permissive
        license was located. Recorded, not faked.

THREE THINGS S0 GOT WRONG THAT S1/S2 FIX
----------------------------------------
1. **Device/mic EQ was applied PER SOURCE.** That is physically impossible --
   one microphone has one response -- and it hands the classifier a free label.
   Here the entire device chain (mic EQ, resonances, AGC, clipping, noise,
   recording gain) is applied to the MIXTURE, after summing, exactly once.
2. **`simple_eq` ran a Python loop over every sample.** Replaced by
   frequency-domain construction / `scipy.signal.lfilter` (Workstream D).
3. **Wearer geometry was effectively fixed.** Here head radius, mouth-to-mic
   range and angle, and device response are all resampled per scene, so
   "wearer" is a *distribution* of geometries, not one memorisable EQ curve.

ANTI-SHORTCUT (Workstream I/J)
------------------------------
The wearer's advantage is meant to be *geometric*, not loud:
  * DRR is the primary intended cue and is a WITHIN-SIGNAL RATIO, so it
    survives arbitrary gain randomisation.
  * Near-field sphere colouration (rho = r/a small) is likewise level-free.
  * Distance/level ranges deliberately OVERLAP: a bystander may be at 0.15 m
    (whispering in your ear) and a wearer may be whispering at -25 dB.
  * `gain_mode` supports normal / level_balanced / gain_randomized.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sps

from .acoustics import C_SOUND, SAMPLE_RATE, air_absorption_db_per_m, get_sphere_lut

EPS = 1e-12

# Acoustic-pressure -> digital-full-scale calibration.
#
# The simulation works in relative pressure units (source amplitude x 1/r x
# diffraction). One FIXED constant converts that to the ADC's [-1, 1] scale.
# It MUST be a constant, never a per-scene normalisation: normalising per scene
# would destroy exactly the absolute-level cue this campaign is measuring
# (Workstream J) and would silently turn every condition into `level_balanced`.
#
# Measured calibration: with this factor, a wearer speaking at nominal 0 dB from
# 0.09-0.175 m lands at a median of about -18 dBFS active-region RMS, leaving
# ~18 dB of peak headroom -- a realistic wearable record level. Everything else
# then follows from physics: a talker at 2 m arrives ~24 dB lower, a shouting
# bystander at 0.3 m arrives hot, and clipping happens only when it physically
# should.
MIC_CALIBRATION = 0.2  # = -14.0 dB


def db_to_lin(db: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (np.asarray(db) / 20.0)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + EPS))


# ---------------------------------------------------------------------------
# Geometry / room parameter sampling
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Rig:
    """The person + device. Resampled per scene so 'wearer' is a distribution."""

    head_radius_m: float           # sphere radius a
    mouth_range_m: float           # mouth -> mic distance
    mouth_angle_rad: float         # angle at head centre between mouth and mic


@dataclasses.dataclass
class Room:
    volume_m3: float
    rt60_s: float

    @property
    def critical_distance_m(self) -> float:
        # Classical d_c = 0.057 * sqrt(V / RT60) for a diffuse field.
        return 0.057 * math.sqrt(max(self.volume_m3, 1.0) / max(self.rt60_s, 0.05))


@dataclasses.dataclass
class SourceGeometry:
    range_m: float
    angle_rad: float               # angle at head centre from mic to source
    role: str                      # "wearer" | "environment"

    def drr_db(self, room: Room) -> float:
        """Direct-to-reverberant ratio implied by range and room.

        DRR_dB = 20 log10(d_c / r). This is the physically honest, level-free
        wearer cue: a mouth at 0.13 m is far inside the critical distance; a
        talker at 2 m is well outside it.
        """
        return float(np.clip(20.0 * math.log10(room.critical_distance_m / max(self.range_m, 0.02)), -18.0, 40.0))


# Held-out OOD geometry bands (Workstream U). Training samplers EXCLUDE these;
# the OOD evaluation sampler uses ONLY these.
OOD_BANDS = {
    "wearer_range_m": (0.155, 0.185),
    "env_range_m": (2.6, 3.6),
    "env_angle_deg": (100.0, 130.0),
    "rt60_s": (0.62, 0.80),
    "head_radius_m": (0.094, 0.100),
}


def _sample_avoiding(rng: np.random.Generator, lo: float, hi: float,
                     band: Optional[Tuple[float, float]], log: bool = False) -> float:
    """Uniform (or log-uniform) draw from [lo,hi] that avoids `band`."""
    for _ in range(64):
        if log:
            v = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
        else:
            v = float(rng.uniform(lo, hi))
        if band is None or not (band[0] <= v <= band[1]):
            return v
    return v


def sample_rig(rng: np.random.Generator, ood: bool = False, hold_out: bool = True) -> Rig:
    """Adult head radii ~7.2-9.8 cm; glasses-temple mic sits 9-17 cm from the
    mouth and roughly 40-105 deg around the head from it."""
    band = OOD_BANDS["head_radius_m"] if hold_out else None
    if ood:
        a = float(rng.uniform(*OOD_BANDS["head_radius_m"]))
    else:
        a = _sample_avoiding(rng, 0.072, 0.100, band)
    r = float(rng.uniform(0.090, 0.175))
    if hold_out and not ood:
        r = _sample_avoiding(rng, 0.090, 0.175, OOD_BANDS["wearer_range_m"])
    elif ood:
        r = float(rng.uniform(*OOD_BANDS["wearer_range_m"]))
    r = max(r, a * 1.06)  # source must be outside the sphere
    theta = float(np.deg2rad(rng.uniform(40.0, 105.0)))
    return Rig(head_radius_m=a, mouth_range_m=r, mouth_angle_rad=theta)


def sample_room(rng: np.random.Generator, ood: bool = False, hold_out: bool = True) -> Room:
    v = float(np.exp(rng.uniform(math.log(18.0), math.log(400.0))))
    if ood:
        rt = float(rng.uniform(*OOD_BANDS["rt60_s"]))
    else:
        rt = _sample_avoiding(rng, 0.12, 1.10, OOD_BANDS["rt60_s"] if hold_out else None, log=True)
    return Room(volume_m3=v, rt60_s=rt)


def sample_environment_geometry(
    rng: np.random.Generator,
    close: bool = False,
    ood: bool = False,
    hold_out: bool = True,
) -> SourceGeometry:
    """Bystander geometry. `close` forces the hard, overlapping regime where a
    bystander is nearer than a typical mouth-to-temple path (0.15-0.45 m) --
    i.e. it deliberately DOES overlap the wearer's range distribution."""
    if ood:
        r = float(rng.uniform(*OOD_BANDS["env_range_m"]))
        ang = float(np.deg2rad(rng.uniform(*OOD_BANDS["env_angle_deg"])))
        return SourceGeometry(r, ang, "environment")
    if close:
        r = float(rng.uniform(0.15, 0.45))
    else:
        r = float(np.exp(rng.uniform(math.log(0.30), math.log(7.0))))
        if hold_out:
            r = _sample_avoiding(rng, 0.30, 7.0, OOD_BANDS["env_range_m"], log=True)
    deg = _sample_avoiding(rng, 0.0, 180.0, OOD_BANDS["env_angle_deg"] if hold_out else None)
    return SourceGeometry(r, float(np.deg2rad(deg)), "environment")


def wearer_geometry(rig: Rig) -> SourceGeometry:
    return SourceGeometry(rig.mouth_range_m, rig.mouth_angle_rad, "wearer")


# ---------------------------------------------------------------------------
# Transfer function / impulse response
# ---------------------------------------------------------------------------
def _parametric_head_response(freqs: np.ndarray, a: float, r: float, theta: float) -> np.ndarray:
    """S1's cheap parametric stand-in for head diffraction + near-field.

    Two physically-motivated terms, both dimensionally sane, neither claiming
    to be exact:
      * HF shadow: first-order shelf whose corner is c/(2*pi*a) and whose HF
        gain interpolates between +6 dB (source in view, theta=0) and about
        -14 dB (fully shadowed, theta=pi).
      * Near-field: spherical-wave curvature gives a low-frequency lift that
        grows as the source approaches the surface, controlled by rho=r/a.
    S2 replaces BOTH with the exact rigid-sphere solution.
    """
    rho = max(r / a, 1.02)
    f0 = C_SOUND / (2.0 * math.pi * a)
    w = freqs / f0
    hf_gain = db_to_lin(6.0 - 20.0 * (1.0 - math.cos(theta)) / 2.0 * 2.0)
    shelf = (1.0 + 1j * w * float(hf_gain)) / (1.0 + 1j * w)
    nf_strength = float(np.clip(1.0 / (rho - 0.9), 0.0, 1.6))
    f_nf = 380.0
    nearfield = 1.0 + nf_strength * (f_nf / (freqs + f_nf))
    return shelf * nearfield


def source_transfer(
    freqs: np.ndarray,
    geom: SourceGeometry,
    rig: Rig,
    generation: str,
) -> np.ndarray:
    """Complex direct-path transfer function: spreading, delay, air absorption,
    head diffraction. NOT including room reverberation (added in `build_ir`)."""
    r = geom.range_m
    spreading = 1.0 / max(r, 0.02)
    delay = np.exp(-2j * np.pi * freqs * (r / C_SOUND))
    air = db_to_lin(-air_absorption_db_per_m(freqs) * r)
    if generation == "S2":
        head = get_sphere_lut().query(freqs, rig.head_radius_m, r, geom.angle_rad)
    elif generation == "S1":
        head = _parametric_head_response(freqs, rig.head_radius_m, r, geom.angle_rad)
    else:
        raise ValueError(f"unknown generation {generation!r}")
    return spreading * delay * air * head


def build_ir(
    geom: SourceGeometry,
    rig: Rig,
    room: Room,
    rng: np.random.Generator,
    generation: str,
    sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Full room impulse response for one source: direct path + discrete early
    reflections + band-split exponential late tail, with the late tail scaled to
    hit the room-and-range-implied DRR."""
    rt60 = room.rt60_s
    tail_len = int(np.clip(rt60 * 1.2, 0.08, 0.55) * sr)
    n_direct = 1024
    total_len = max(n_direct, tail_len + n_direct // 2)
    nfft = 1 << int(np.ceil(np.log2(total_len)))

    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    H = source_transfer(freqs, geom, rig, generation)

    # --- discrete early reflections (first ~30 ms) ---
    n_refl = int(rng.integers(3, 9))
    for _ in range(n_refl):
        dt = float(rng.uniform(0.002, 0.030))
        amp = float(rng.uniform(0.12, 0.55)) * (1.0 / max(geom.range_m + C_SOUND * dt, 0.05))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        # reflections are duller than the direct path (absorptive surfaces)
        dull = 1.0 / (1.0 + 1j * freqs / float(rng.uniform(1500.0, 6000.0)))
        H = H + sign * amp * dull * np.exp(-2j * np.pi * freqs * (geom.range_m / C_SOUND + dt))

    ir_early = np.fft.irfft(H, n=nfft)[:total_len]

    # --- band-split late tail: HF decays faster than LF (real rooms) ---
    t = np.arange(tail_len) / sr
    noise = rng.standard_normal(tail_len)
    rt_lf, rt_hf = rt60, rt60 * float(rng.uniform(0.45, 0.75))
    lf = sps.lfilter(*sps.butter(2, 1200.0 / (sr / 2), btype="low"), noise)
    hf = sps.lfilter(*sps.butter(2, 1200.0 / (sr / 2), btype="high"), noise)
    tail = lf * np.exp(-6.907 * t / rt_lf) + hf * np.exp(-6.907 * t / rt_hf)

    ir = np.zeros(total_len)
    ir[: len(ir_early)] += ir_early
    onset = int((geom.range_m / C_SOUND + 0.012) * sr)
    seg = min(tail_len, total_len - onset)
    if seg > 0:
        # Scale the tail to the target DRR (direct+early energy vs late energy)
        drr_db = geom.drr_db(room)
        e_direct = float(np.sum(ir_early**2)) + EPS
        e_tail = float(np.sum(tail[:seg] ** 2)) + EPS
        target = e_direct / (10.0 ** (drr_db / 10.0))
        ir[onset : onset + seg] += tail[:seg] * math.sqrt(target / e_tail)
    else:
        drr_db = geom.drr_db(room)

    meta = {
        "range_m": geom.range_m,
        "angle_deg": float(np.rad2deg(geom.angle_rad)),
        "rho": geom.range_m / rig.head_radius_m,
        "drr_db": drr_db,
        "rt60_s": rt60,
        "room_volume_m3": room.volume_m3,
        "critical_distance_m": room.critical_distance_m,
        "head_radius_m": rig.head_radius_m,
        "generation": generation,
    }
    return ir.astype(np.float64), meta


def convolve(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """FFT convolution truncated to the input length (Workstream D: this is the
    hot path; `np.convolve` here would be ~100x slower)."""
    y = sps.fftconvolve(x.astype(np.float64), ir, mode="full")[: len(x)]
    return y


# ---------------------------------------------------------------------------
# Device chain -- applied ONCE, to the MIXTURE (never per source)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class DeviceConfig:
    tilt_db_per_oct: float
    resonances: List[Tuple[float, float, float]]   # (freq, Q, gain_db)
    hp_cutoff_hz: float
    recording_gain_db: float
    compressor_ratio: float
    clip_drive: Optional[float]
    wind_lf_db: Optional[float]
    noise_snr_db: Optional[float]


def sample_device(rng: np.random.Generator, harsh: bool = False,
                  ood_eq: bool = False) -> DeviceConfig:
    n_res = int(rng.integers(1, 4))
    res = []
    for _ in range(n_res):
        f = float(np.exp(rng.uniform(math.log(400.0), math.log(6500.0))))
        q = float(rng.uniform(1.0, 6.0))
        g = float(rng.uniform(-11.0, 9.0) if not ood_eq else rng.uniform(-20.0, 16.0))
        res.append((f, q, g))
    return DeviceConfig(
        tilt_db_per_oct=float(rng.uniform(-2.0, 1.5) if not ood_eq else rng.uniform(-4.5, 3.5)),
        resonances=res,
        hp_cutoff_hz=float(rng.uniform(40.0, 140.0)),
        recording_gain_db=float(rng.uniform(-12.0, 8.0)),
        compressor_ratio=float(rng.uniform(1.0, 4.0)) if (harsh or rng.random() < 0.35) else 1.0,
        clip_drive=float(rng.uniform(1.2, 5.0)) if (harsh or rng.random() < 0.15) else None,
        wind_lf_db=float(rng.uniform(-40.0, -18.0)) if (harsh or rng.random() < 0.20) else None,
        noise_snr_db=float(rng.uniform(0.0, 32.0)) if rng.random() < 0.75 else None,
    )


def _device_eq_response(freqs: np.ndarray, cfg: DeviceConfig) -> np.ndarray:
    f = np.maximum(freqs, 1.0)
    db = cfg.tilt_db_per_oct * np.log2(f / 1000.0)
    for fc, q, g in cfg.resonances:
        db = db + g * np.exp(-0.5 * (np.log(f / fc) * q * 1.6) ** 2)
    hp = 1.0 / (1.0 + (cfg.hp_cutoff_hz / f) ** 2)
    return db_to_lin(db) * hp


def apply_device_chain(
    x: np.ndarray,
    cfg: DeviceConfig,
    rng: np.random.Generator,
    noise_source: Optional[np.ndarray] = None,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Mic response -> wind -> additive noise -> gain -> compression -> clipping.

    Applied to the SUMMED mixture only. Every stage here is common-mode with
    respect to wearer/environment and therefore cannot be used as a label
    shortcut; that is the entire point.
    """
    n = len(x)
    x = np.asarray(x, dtype=np.float64) * MIC_CALIBRATION
    nfft = 1 << int(np.ceil(np.log2(max(n, 2))))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    X = np.fft.rfft(x, n=nfft)
    y = np.fft.irfft(X * _device_eq_response(freqs, cfg), n=nfft)[:n]

    if cfg.wind_lf_db is not None:
        w = rng.standard_normal(n)
        w = sps.lfilter(*sps.butter(2, 120.0 / (sr / 2), btype="low"), w)
        y = y + w / (rms(w) + EPS) * rms(y) * float(db_to_lin(cfg.wind_lf_db))

    if cfg.noise_snr_db is not None and noise_source is not None and len(noise_source) >= n:
        st = int(rng.integers(0, len(noise_source) - n + 1))
        nz = noise_source[st : st + n].astype(np.float64)
        if rms(nz) > EPS:
            y = y + nz / rms(nz) * rms(y) * float(db_to_lin(-cfg.noise_snr_db))

    y = y * float(db_to_lin(cfg.recording_gain_db))

    if cfg.compressor_ratio > 1.01:
        env = np.abs(sps.lfilter([0.002], [1.0, -0.998], np.abs(y))) + EPS
        thr = np.percentile(env, 70) + EPS
        over = np.maximum(env / thr, 1.0)
        y = y * over ** (1.0 / cfg.compressor_ratio - 1.0)

    if cfg.clip_drive is not None:
        y = np.tanh(y * cfg.clip_drive) / math.tanh(cfg.clip_drive)

    # Hard-clip at digital full scale, like a real 16-bit ADC. Deliberately NOT
    # peak-normalised: clipping is a real covariate, not something to hide.
    return np.clip(y, -1.0, 1.0)


SIMULATOR_VERSION = "geowearnet-sim-v1.0.0 (S1/S2)"


def simulator_config_summary(generation: str) -> Dict[str, object]:
    return {
        "simulator_version": SIMULATOR_VERSION,
        "generation": generation,
        "sample_rate": SAMPLE_RATE,
        "speed_of_sound_m_s": C_SOUND,
        "head_radius_m": [0.072, 0.100],
        "wearer_mouth_range_m": [0.090, 0.175],
        "wearer_mouth_angle_deg": [40.0, 105.0],
        "environment_range_m": [0.15, 7.0],
        "environment_angle_deg": [0.0, 180.0],
        "rt60_s": [0.12, 1.10],
        "room_volume_m3": [18.0, 400.0],
        "held_out_ood_bands": OOD_BANDS,
        "device_chain": "applied once to the summed mixture (never per source)",
        "air_absorption": "ISO 9613-1",
        "head_model": {"S1": "parametric shelf + near-field lift", "S2": "exact rigid sphere (Duda-Martens series)"}[generation],
    }
