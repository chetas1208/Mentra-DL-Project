"""GeoWearNet acoustic physics primitives (Workstreams E/S2).

Everything here is implemented from OPEN, PUBLISHED equations and is
numerically self-validating (`validate()` checks each model against limits
that are known analytically). No unpublished code is cloned, and no fitted
constants are taken on faith from a source this campaign could not actually
read -- where a well-known approximation could not be verified verbatim, the
exact analytic solution is implemented instead.

Contents
--------
1. `air_absorption_db_per_m` -- ISO 9613-1 atmospheric absorption.
2. `rigid_sphere_response` -- the exact analytic acoustic response of a rigid
   sphere to a point source, evaluated on the sphere surface (the "spherical
   head model"). This is the S2 head model.
3. `SphereLUT` -- a precomputed (mu, rho, theta) lookup table making (2) cheap
   enough for an online data loader (Workstream D).
4. `mel_filterbank` / `log_mel` -- feature front end (no torchaudio in this env).

SOURCES / JUSTIFICATION
-----------------------
* Rigid-sphere scattering: the series solution for a point source and a rigid
  sphere is textbook acoustics (Morse & Ingard, *Theoretical Acoustics*); its
  use as a head model with the (mu = omega*a/c, rho = r/a) normalization is
  Duda & Martens, "Range dependence of the response of a spherical head model",
  JASA 104(5), 1998. This module implements the series directly rather than the
  Brown & Duda 1-pole/1-zero *approximation*, because this session could not
  retrieve the approximation's fitted constants (alpha_min / theta_min) from a
  primary source -- the PDFs would not extract. Implementing the exact series
  removes the need to trust unverified constants and is validated below against
  the known low-frequency (H -> 1) and high-frequency (front -> +6 dB pressure
  doubling, rear -> deep shadow) limits.
* The simulation *progression* used here (analytical parametric ATFs -> rigid
  sphere -> measured head-and-torso) mirrors the published own-voice-detection
  result this campaign is building on: Jensen et al., "Single Microphone Own
  Voice Detection based on Simulated Transfer Functions for Hearing Aids",
  arXiv:2603.02724, whose abstract states the classifier "is first trained on
  analytically generated ATFs and then progressively fine-tuned using
  numerically simulated ATFs, transitioning from a rigid-sphere model to a
  detailed head-and-torso representation" (verified by fetch, 2026-08-25).
  Our S1/S2/S3 map onto those three stages; S3 is BLOCKED (no open measured
  wearable/HATS response corpus of known license was found).
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.special import lpmv, spherical_jn, spherical_yn

SAMPLE_RATE = 16000
C_SOUND = 343.0  # m/s at 20 C
CACHE_DIR = Path(__file__).resolve().parent / "cache"


# --------------------------------------------------------------------------
# 1. Atmospheric absorption (ISO 9613-1)
# --------------------------------------------------------------------------
def air_absorption_db_per_m(
    freqs_hz: np.ndarray,
    temp_c: float = 20.0,
    humidity_pct: float = 50.0,
    pressure_kpa: float = 101.325,
) -> np.ndarray:
    """ISO 9613-1 pure-tone atmospheric attenuation coefficient, dB/m.

    Matters mainly above ~4 kHz and beyond ~2 m, which is exactly the
    wearer-vs-environment regime, so it is modelled rather than ignored.
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    T = temp_c + 273.15
    T0 = 293.15
    T01 = 273.16
    pa = pressure_kpa
    pr = 101.325

    psat_ratio = 10.0 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    h = humidity_pct * psat_ratio * (pr / pa)

    frO = (pa / pr) * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (pa / pr) * (T / T0) ** -0.5 * (
        9.0 + 280.0 * h * np.exp(-4.170 * ((T / T0) ** (-1.0 / 3.0) - 1.0))
    )

    alpha = (
        8.686
        * f**2
        * (
            1.84e-11 * (pr / pa) * (T / T0) ** 0.5
            + (T / T0) ** -2.5
            * (
                0.01275 * np.exp(-2239.1 / T) / (frO + f**2 / frO)
                + 0.1068 * np.exp(-3352.0 / T) / (frN + f**2 / frN)
            )
        )
    )
    return alpha


# --------------------------------------------------------------------------
# 2. Rigid-sphere (spherical head) response -- exact series
# --------------------------------------------------------------------------
def _spherical_hankel2(n: int, x: np.ndarray) -> np.ndarray:
    """h_n^(2)(x) = j_n(x) - i*y_n(x)  (outgoing wave, e^{+iwt} convention).

    Overflows to inf/nan for small x at high order; callers substitute the
    small-argument asymptote of the RATIO, which stays finite.
    """
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        return spherical_jn(n, x) - 1j * spherical_yn(n, x)


def _spherical_hankel2_prime(n: int, x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        return spherical_jn(n, x, derivative=True) - 1j * spherical_yn(n, x, derivative=True)


def rigid_sphere_response(
    mu: np.ndarray,
    rho: float,
    theta_rad: float,
    max_order: int | None = None,
) -> np.ndarray:
    """Pressure at a point on a rigid sphere of radius `a`, due to a point
    source at range `r`, normalised by the free-field pressure the source
    would produce at the sphere's centre.

        mu    = omega * a / c   (normalised frequency), array
        rho   = r / a           (normalised source range), scalar >= 1
        theta = angle between the source direction and the receiver point,
                measured from the sphere centre (0 = source directly over the
                receiver, pi = antipodal / fully shadowed)

    Series (derived from the rigid Neumann boundary condition, matching Duda &
    Martens 1998, JASA 104(5)):

        H = -(rho / mu) * exp(+i*mu*rho) *
             sum_m (2m+1) * P_m(cos theta) * h_m(mu*rho) / h'_m(mu)

    with h_m the spherical Hankel function of the SECOND kind (e^{+i w t} time
    convention, so an outgoing wave is e^{-ikr} -- which is also numpy's irfft
    convention, i.e. a delay tau maps to exp(-i w tau)). The leading exp(+i mu
    rho) cancels the propagation phase of the reference point, so H carries
    ONLY diffraction/near-field colouration; callers add 1/r and the r/c delay
    themselves (see `simulate_s1.build_ir`).

    Returns complex array with the same shape as `mu`.

    Limits (all asserted in `validate()`):
      * mu -> 0, rho -> inf (plane wave, DC):  H -> 1
      * mu -> 0, finite rho:  H -> sum_m (2m+1) P_m(cos th) / ((m+1) rho^m).
        NOTE this is NOT 1 and NOT the free-field ratio rho/(rho-1): a rigid
        sphere in the *near field* of a point source retains an O(1/rho)
        induced-dipole response even at DC. An earlier version of this file
        asserted the naive "sphere is transparent at DC" limit and was wrong.
      * mu -> inf, theta = 0, rho -> inf: |H| -> 2 (+6 dB pressure doubling)
      * mu -> inf, theta = pi: deep shadow
    """
    mu = np.asarray(mu, dtype=np.float64)
    assert rho >= 1.0, "source must be outside the sphere"

    mu_safe = np.where(mu < 1e-6, 1e-6, mu)
    if max_order is None:
        # Two convergence drivers: (a) diffraction order grows with mu;
        # (b) the near-field series converges like rho^-m, so small rho needs
        # many more terms (rho=1.5 needs ~35, not the ~12 that mu alone implies).
        by_mu = int(np.ceil(mu_safe.max() * 1.5)) + 12
        by_rho = int(np.ceil(np.log(1e-8) / np.log(1.0 / rho))) if rho > 1.0001 else 80
        max_order = min(max(by_mu, by_rho, 8), 120)

    x = np.cos(theta_rad)
    total = np.zeros_like(mu_safe, dtype=np.complex128)
    for m in range(max_order + 1):
        pm = float(lpmv(0, m, x))
        if pm == 0.0 and m > 0:
            continue
        num = _spherical_hankel2(m, mu_safe * rho)
        den = _spherical_hankel2_prime(m, mu_safe)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            ratio = num / den
        # For small arguments both h_m and h'_m overflow float64 at high order
        # while their RATIO stays perfectly well-conditioned. Substitute the
        # exact small-argument asymptote there:
        #   h_m(x) ~ i(2m-1)!!/x^(m+1),  h'_m(x) ~ -i(m+1)(2m-1)!!/x^(m+2)
        #   => h_m(mu*rho)/h'_m(mu) -> -mu / ((m+1) * rho^(m+1))
        bad = ~np.isfinite(ratio)
        if bad.any():
            ratio = np.where(bad, -mu_safe / ((m + 1) * rho ** (m + 1)), ratio)
        total += (2 * m + 1) * pm * ratio

    h = -(rho / mu_safe) * np.exp(1j * mu_safe * rho) * total
    # At mu -> 0 the 0/0 is resolved by the analytic DC limit below.
    if np.any(mu < 1e-6):
        h = np.where(mu < 1e-6, rigid_sphere_dc_limit(rho, theta_rad), h)
    return h


def rigid_sphere_dc_limit(rho: float, theta_rad: float, max_order: int = 64) -> complex:
    """Analytic mu->0 limit of `rigid_sphere_response`:

        sum_m (2m+1) P_m(cos theta) / ((m+1) rho^m)

    Derived from the small-argument forms h_m(x) ~ i(2m-1)!!/x^(m+1) and
    h'_m(x) ~ -i(m+1)(2m-1)!!/x^(m+2). Used both to resolve the 0/0 at DC and,
    more importantly, as an INDEPENDENT check on the series implementation in
    `validate()` -- it is a different derivation reaching the same number.
    """
    x = np.cos(theta_rad)
    u = 1.0 / rho
    s = 0.0
    for m in range(max_order + 1):
        s += (2 * m + 1) * float(lpmv(0, m, x)) / (m + 1) * u**m
    return complex(s)


class SphereLUT:
    """Precomputed rigid-sphere magnitude/phase table.

    `rigid_sphere_response` costs a few ms per call (a scipy Bessel evaluation
    per order). An online data loader needs thousands per second, so the table
    is built once over a normalised (mu, rho, theta) grid and interpolated.
    Because the response depends on head radius `a` only through mu = w*a/c and
    rho = r/a, one normalised table covers ALL head sizes and ranges.
    """

    MU_GRID = np.concatenate([np.linspace(0.0, 2.0, 41), np.linspace(2.1, 40.0, 190)])
    RHO_GRID = np.array(
        [1.05, 1.15, 1.3, 1.5, 1.8, 2.2, 2.8, 3.5, 4.5, 6.0, 8.0, 11.0, 15.0, 22.0, 32.0, 50.0, 80.0, 200.0]
    )
    THETA_GRID = np.deg2rad(np.arange(0, 181, 5, dtype=np.float64))

    def __init__(self, table: np.ndarray):
        # table: (n_rho, n_theta, n_mu) complex
        self.table = table

    @classmethod
    def cache_path(cls) -> Path:
        return CACHE_DIR / "sphere_lut_v1.npz"

    @classmethod
    def build(cls) -> "SphereLUT":
        nr, nt, nm = len(cls.RHO_GRID), len(cls.THETA_GRID), len(cls.MU_GRID)
        table = np.zeros((nr, nt, nm), dtype=np.complex128)
        for i, rho in enumerate(cls.RHO_GRID):
            for j, th in enumerate(cls.THETA_GRID):
                table[i, j] = rigid_sphere_response(cls.MU_GRID, float(rho), float(th))
        return cls(table)

    @classmethod
    def load(cls) -> "SphereLUT":
        p = cls.cache_path()
        if p.exists():
            with np.load(p) as z:
                return cls(z["table"])
        lut = cls.build()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, table=lut.table)
        return lut

    def query(self, freqs_hz: np.ndarray, a_m: float, r_m: float, theta_rad: float) -> np.ndarray:
        """Interpolated complex response for a real head radius/range/angle."""
        rho = float(np.clip(r_m / a_m, self.RHO_GRID[0], self.RHO_GRID[-1]))
        theta = float(np.clip(theta_rad, 0.0, np.pi))
        mu = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64) * a_m / C_SOUND

        # bilinear over (rho, theta), then linear over mu
        ri = np.searchsorted(self.RHO_GRID, rho).clip(1, len(self.RHO_GRID) - 1)
        r0, r1 = self.RHO_GRID[ri - 1], self.RHO_GRID[ri]
        wr = (np.log(rho) - np.log(r0)) / (np.log(r1) - np.log(r0))
        ti = np.searchsorted(self.THETA_GRID, theta).clip(1, len(self.THETA_GRID) - 1)
        t0, t1 = self.THETA_GRID[ti - 1], self.THETA_GRID[ti]
        wt = (theta - t0) / (t1 - t0) if t1 > t0 else 0.0

        def slab(i, j):
            return self.table[i, j]

        c00, c01 = slab(ri - 1, ti - 1), slab(ri - 1, ti)
        c10, c11 = slab(ri, ti - 1), slab(ri, ti)
        curve = (1 - wr) * ((1 - wt) * c00 + wt * c01) + wr * ((1 - wt) * c10 + wt * c11)

        mu_c = np.clip(mu, self.MU_GRID[0], self.MU_GRID[-1])
        mag = np.interp(mu_c, self.MU_GRID, np.abs(curve))
        # interpolate unwrapped phase to avoid wraparound artefacts
        ph = np.interp(mu_c, self.MU_GRID, np.unwrap(np.angle(curve)))
        return mag * np.exp(1j * ph)


@functools.lru_cache(maxsize=1)
def get_sphere_lut() -> SphereLUT:
    return SphereLUT.load()


# --------------------------------------------------------------------------
# 3. Mel front end (no torchaudio in this environment)
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=8)
def _hann(win: int) -> np.ndarray:
    return np.hanning(win).astype(np.float32)


@functools.lru_cache(maxsize=8)
def mel_filterbank(n_mels: int = 64, n_fft: int = 512, sr: int = SAMPLE_RATE,
                   fmin: float = 20.0, fmax: float = 7800.0) -> np.ndarray:
    """Slaney-style triangular mel filterbank, (n_mels, n_fft//2+1) float32.

    Built with librosa (available in this env) at import/first-use time only;
    the hot path is a single matmul, so librosa is never in the inner loop.
    """
    import librosa

    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, norm="slaney")
    return np.ascontiguousarray(fb.astype(np.float32))


def log_mel(
    x: np.ndarray,
    n_mels: int = 64,
    n_fft: int = 512,
    hop: int = 160,
    win: int = 400,
    sr: int = SAMPLE_RATE,
    floor_db: float = -100.0,
) -> np.ndarray:
    """Calibrated log-mel, (T, n_mels) float32.

    CRITICAL (Workstream J / AY): NO per-utterance normalisation, no CMVN, no
    peak normalisation. The absolute PCM scale is preserved all the way into
    the feature, because absolute level is one of the physical cues under test.
    The only fixed transform is `10*log10(power)` with a floor, which is a
    fixed monotone map, not a data-dependent normalisation.
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n < win:
        x = np.pad(x, (0, win - n))
        n = win
    n_frames = 1 + (n - win) // hop
    # Strided view instead of a fancy-index gather (Workstream D). Same
    # elements, same order, bit-identical output.
    frames = np.lib.stride_tricks.sliding_window_view(x[: hop * (n_frames - 1) + win], win)[::hop]
    frames = frames * _hann(win)[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spec.real**2 + spec.imag**2).astype(np.float32)
    mel = power @ mel_filterbank(n_mels, n_fft, sr).T
    out = 10.0 * np.log10(np.maximum(mel, 1e-10))
    return np.maximum(out, floor_db).astype(np.float32)


# --------------------------------------------------------------------------
# 4. Self-validation
# --------------------------------------------------------------------------
def validate(verbose: bool = True) -> dict:
    """Check every physical model against limits known in closed form.

    Failing any of these means the physics is wrong, and every downstream
    number in the campaign would be meaningless -- so this runs in the test
    suite (Workstream BH) and at simulator start-up.
    """
    res: dict = {}

    # --- air absorption: known reference magnitudes ---
    f = np.array([125.0, 1000.0, 4000.0, 8000.0])
    a = air_absorption_db_per_m(f, 20.0, 50.0)
    res["air_absorption_db_per_m_at_20C_50pct"] = {str(int(k)): float(v) for k, v in zip(f, a)}
    # ISO 9613-1 @20C/50%RH: ~0.0004, ~0.005, ~0.03, ~0.08 dB/m order of magnitude
    assert a[0] < a[1] < a[2] < a[3], "absorption must increase with frequency"
    assert 0.0001 < a[1] < 0.02, f"1 kHz absorption out of range: {a[1]}"
    assert 0.02 < a[3] < 0.30, f"8 kHz absorption out of range: {a[3]}"
    res["air_absorption_monotone"] = True

    # --- rigid sphere: DC + plane-wave limit -> 1 ---
    mu_lo = np.array([1e-3, 3e-3])
    h_pw = rigid_sphere_response(mu_lo, 2000.0, 0.0)
    res["sphere_dc_planewave"] = float(np.abs(h_pw).mean())
    assert abs(np.abs(h_pw).mean() - 1.0) < 2e-2, f"DC plane-wave limit != 1: {np.abs(h_pw)}"

    # --- rigid sphere: series vs INDEPENDENT analytic DC limit, near field ---
    dc_err = []
    for rho in (1.5, 4.0, 20.0):
        for th in (0.0, np.pi / 3, np.pi / 2, np.pi):
            series = rigid_sphere_response(np.array([1e-4]), rho, th)[0]
            closed = rigid_sphere_dc_limit(rho, th)
            dc_err.append(abs(series - closed))
    res["sphere_dc_series_vs_closed_form_max_abs_err"] = float(max(dc_err))
    assert max(dc_err) < 1e-3, f"series disagrees with analytic DC limit: {max(dc_err)}"

    # --- rigid sphere: high-frequency front = pressure doubling (+6 dB) ---
    mu_hi = np.array([25.0, 30.0, 35.0])
    h_front = np.abs(rigid_sphere_response(mu_hi, 2000.0, 0.0))
    front_db = float(20 * np.log10(h_front.mean()))
    res["sphere_hf_front_db"] = front_db
    assert 4.0 < front_db < 7.0, f"HF front should approach +6 dB, got {front_db:.2f}"

    # --- rigid sphere: high-frequency rear = deep shadow ---
    h_rear = np.abs(rigid_sphere_response(mu_hi, 2000.0, np.pi))
    rear_db = float(20 * np.log10(h_rear.mean()))
    res["sphere_hf_rear_db"] = rear_db
    assert rear_db < front_db - 3.0, f"rear must be shadowed vs front: {rear_db:.2f} vs {front_db:.2f}"

    # --- rigid sphere: near source is louder at the surface than a far one ---
    mu = np.array([2.0, 5.0, 10.0])
    near = np.abs(rigid_sphere_response(mu, 1.2, 0.0)).mean()
    far = np.abs(rigid_sphere_response(mu, 50.0, 0.0)).mean()
    res["sphere_near_over_far_db"] = float(20 * np.log10(near / far))
    assert near > far, "near-field surface pressure must exceed far-field"

    # --- LUT fidelity vs exact ---
    lut = get_sphere_lut()
    freqs = np.fft.rfftfreq(512, 1.0 / SAMPLE_RATE)
    errs = []
    for a_m, r_m, th in [(0.0875, 0.12, 0.6), (0.09, 1.5, 2.0), (0.08, 3.0, 1.2)]:
        approx = lut.query(freqs, a_m, r_m, th)
        mu = 2 * np.pi * freqs * a_m / C_SOUND
        exact = rigid_sphere_response(mu, r_m / a_m, th)
        db_err = 20 * np.log10(np.abs(approx) + 1e-12) - 20 * np.log10(np.abs(exact) + 1e-12)
        errs.append(float(np.abs(db_err).max()))
    res["sphere_lut_max_abs_db_error"] = max(errs)
    assert max(errs) < 3.0, f"LUT error too large: {max(errs)} dB"

    # --- log-mel preserves absolute level (Workstream J) ---
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(16000).astype(np.float32) * 0.01
    m1 = log_mel(sig)
    m2 = log_mel(sig * 10.0)
    delta = float(np.median(m2 - m1))
    res["log_mel_gain_10x_db_shift"] = delta
    assert 19.0 < delta < 21.0, f"log-mel must shift ~20 dB for 10x gain, got {delta}"

    if verbose:
        import json

        print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    validate()
