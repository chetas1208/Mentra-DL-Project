"""GeoWearNet E1-SIM S0 simulated transfer-function track (Phase 19-20).

Everything produced by this module is SIMULATED_GEOMETRY: gain + EQ +
a crude synthetically-generated (not measured) decaying-noise impulse
response standing in for "RIR". Per Phase 19, this repo has no RIR corpus
and no pyroomacoustics/room-acoustics simulator installed, so this
deliberately stops at S0 rather than faking a more realistic S1/S2 -- see
`docs/geowearnet_research_plan.md` for that decision recorded explicitly.

Nothing here may be presented as "Mentra wearer detection validated" (Phase
2/47). It exists to (a) give the E0 heuristic classifier and E1 smoke test
something to run on, and (b) exercise the anti-shortcut controls (Phase 9,
18, 20) in code, not to claim real device-relative signal exists.

ROLE RANDOMIZATION (Phase 18): the same underlying LibriSpeech utterance/
speaker can be sent through the "wearer" transfer function in one example
and the "environment" transfer function in another -- role is a property of
the simulated example, not of the speaker. This is enforced by
`make_example`, which accepts pre-selected utterances and independently
rolls which one plays "wearer" and which plays "environment" (see
`role_assignment=` in `SimConfig`).

ANTI-SHORTCUT (Phase 9/20): wearer and environment gain are drawn from
OVERLAPPING log-normal-ish ranges, not "wearer always louder". Dedicated
condition generators below (level_matched, random_gain, wearer_whisper,
close_bystander, shouting_bystander) explicitly construct the adversarial
distributions Phase 9 requires.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def rms_of(x: np.ndarray) -> float:
    return float(np.sqrt((x.astype(np.float64) ** 2).mean() + 1e-12))


def synthetic_ir(rng: np.random.Generator, decay_ms: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Crude synthetic room-response stand-in: exponentially-decaying
    filtered noise. NOT a measured RIR. `decay_ms` controls the RT-ish
    decay time; short (~15-40ms) approximates a near-field/close-mic
    "wearer" path, longer (~80-250ms) approximates a more reverberant
    "environment" path. Documented explicitly as synthetic per Phase 19."""
    n = max(8, int(sr * decay_ms / 1000.0 * 3))
    t = np.arange(n) / sr
    env = np.exp(-t / (decay_ms / 1000.0 + 1e-6))
    noise = rng.standard_normal(n)
    ir = noise * env
    ir[0] = 1.0  # direct path dominates the first tap
    ir = ir / (np.abs(ir).sum() + 1e-9) * 1.0
    return ir.astype(np.float64)


def apply_ir(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    y = np.convolve(x, ir, mode="full")[: len(x)]
    return y


def simple_eq(x: np.ndarray, lf_gain_db: float, hf_gain_db: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Very simple two-band shelving EQ via one-pole filters, used to
    approximate near-field bass boost (wearer, proximity effect) or
    distance-related high-frequency air absorption (environment)."""
    x = x.astype(np.float64)
    # low-shelf via leaky integrator, high part = residual
    alpha = 0.05
    low = np.zeros_like(x)
    prev = 0.0
    for i in range(len(x)):
        prev = alpha * x[i] + (1 - alpha) * prev
        low[i] = prev
    high = x - low
    return low * db_to_lin(lf_gain_db) + high * db_to_lin(hf_gain_db)


def soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    return np.tanh(x * drive) / max(np.tanh(drive), 1e-6)


@dataclasses.dataclass
class TransferParams:
    gain_db: float
    ir_decay_ms: float
    lf_gain_db: float
    hf_gain_db: float
    clip_drive: Optional[float] = None


def apply_transfer(x: np.ndarray, params: TransferParams, rng: np.random.Generator) -> np.ndarray:
    y = apply_ir(x, synthetic_ir(rng, params.ir_decay_ms))
    y = simple_eq(y, params.lf_gain_db, params.hf_gain_db)
    y = y * db_to_lin(params.gain_db)
    if params.clip_drive is not None:
        y = soft_clip(y, params.clip_drive)
    return y


def sample_wearer_transfer(rng: np.random.Generator, gain_range=(-6.0, 10.0)) -> TransferParams:
    """Near-field: shorter synthetic decay, mild LF boost (proximity
    effect), gain range OVERLAPS environment's range on purpose (Phase 20)."""
    return TransferParams(
        gain_db=rng.uniform(*gain_range),
        ir_decay_ms=rng.uniform(8.0, 35.0),
        lf_gain_db=rng.uniform(0.0, 3.0),
        hf_gain_db=rng.uniform(-1.0, 1.0),
    )


def sample_environment_transfer(rng: np.random.Generator, gain_range=(-18.0, 6.0)) -> TransferParams:
    """Far-field: longer synthetic decay, mild HF rolloff (air absorption
    proxy), gain range overlaps wearer's on purpose (Phase 20)."""
    return TransferParams(
        gain_db=rng.uniform(*gain_range),
        ir_decay_ms=rng.uniform(60.0, 220.0),
        lf_gain_db=rng.uniform(-1.0, 0.5),
        hf_gain_db=rng.uniform(-3.0, 0.0),
    )


# ---- Anti-shortcut condition generators (Phase 9) ----

def condition_normal():
    return dict(wearer_gain_range=(-6.0, 10.0), env_gain_range=(-18.0, 6.0))


def condition_level_matched():
    # force both roles into the SAME gain distribution -- if a classifier
    # still separates them, it isn't just using level.
    r = (-6.0, 6.0)
    return dict(wearer_gain_range=r, env_gain_range=r)


def condition_random_gain(rng: np.random.Generator):
    # additional independent random gain -18..+12dB applied on TOP of the
    # transfer function's own gain, per Phase 9(B).
    return rng.uniform(-18.0, 12.0), rng.uniform(-18.0, 12.0)


def condition_wearer_whisper():
    return dict(wearer_gain_range=(-24.0, -12.0), env_gain_range=(-18.0, 6.0))


def condition_close_bystander():
    # environment >= wearer in level
    return dict(wearer_gain_range=(-12.0, -2.0), env_gain_range=(0.0, 12.0))


def condition_shouting_bystander():
    return dict(wearer_gain_range=(-6.0, 4.0), env_gain_range=(10.0, 20.0))


def condition_wearer_loud():
    return dict(wearer_gain_range=(6.0, 16.0), env_gain_range=(-18.0, -4.0))


@dataclasses.dataclass
class SimExample:
    audio: np.ndarray          # mono float32 [-1,1], SAMPLE_RATE
    role: str                  # "wearer" or "environment" (single-source clip, for E0)
    speaker_id: str
    condition: str


def make_single_source_clip(
    utterance: np.ndarray,
    role: str,
    rng: np.random.Generator,
    condition: str = "normal",
    extra_gain_db: Optional[float] = None,
) -> np.ndarray:
    """Applies the role's transfer function (+ optional extra gain, for the
    random-gain anti-shortcut condition) to a clean utterance. `role` is
    chosen by the CALLER, independent of speaker identity -- role
    randomization (Phase 18) happens one level up in the manifest builder."""
    cond_map = {
        "normal": condition_normal,
        "level_matched": condition_level_matched,
        "wearer_whisper": condition_wearer_whisper,
        "close_bystander": condition_close_bystander,
        "shouting_bystander": condition_shouting_bystander,
        "wearer_loud": condition_wearer_loud,
    }
    ranges = cond_map.get(condition, condition_normal)()
    if role == "wearer":
        params = sample_wearer_transfer(rng, ranges["wearer_gain_range"])
    else:
        params = sample_environment_transfer(rng, ranges["env_gain_range"])
    y = apply_transfer(utterance, params, rng)
    if extra_gain_db is not None:
        y = y * db_to_lin(extra_gain_db)
    peak = np.abs(y).max()
    if peak > 4.0:  # guard against pathological synthetic-IR blowups
        y = y / peak * 4.0
    return y.astype(np.float32)


def make_overlap_mixture(
    wearer_utt: np.ndarray,
    env_utt: np.ndarray,
    rng: np.random.Generator,
    condition: str = "normal",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds a fixed-length mixture with independently-placed wearer and
    environment segments (so silence/10/01/11 all occur), returns
    (mixture, wearer_activity_samples, environment_activity_samples)."""
    total_len = max(len(wearer_utt), len(env_utt)) + rng.integers(0, SAMPLE_RATE)
    total_len = max(total_len, SAMPLE_RATE)  # at least 1s

    w = make_single_source_clip(wearer_utt, "wearer", rng, condition)
    e = make_single_source_clip(env_utt, "environment", rng, condition)

    mix = np.zeros(total_len, dtype=np.float64)
    w_act = np.zeros(total_len, dtype=np.float32)
    e_act = np.zeros(total_len, dtype=np.float32)

    w_start = int(rng.integers(0, max(1, total_len - len(w) + 1)))
    e_start = int(rng.integers(0, max(1, total_len - len(e) + 1)))

    mix[w_start : w_start + len(w)] += w
    w_act[w_start : w_start + len(w)] = 1.0
    mix[e_start : e_start + len(e)] += e
    e_act[e_start : e_start + len(e)] = 1.0

    peak = np.abs(mix).max()
    if peak > 0.999:
        mix = mix / peak * 0.999
    return mix.astype(np.float32), w_act, e_act
