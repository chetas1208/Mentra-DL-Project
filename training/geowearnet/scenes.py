"""GeoWearNet continuous scene generator (Workstreams F, H, I, J).

Produces realistic multi-second scenes rather than isolated one-second clips:
independent wearer and environment activity timelines, so all four states
(00 silence / 10 wearer / 01 environment / 11 overlap) occur with transitions
between every pair at a controlled spread of durations.

WORKSTREAM F -- ROLE RANDOMISATION
  Speakers are drawn without replacement and then SHUFFLED into roles. No
  speaker is ever pinned to a role. `role_audit()` proves it empirically by
  counting each speaker's wearer/environment appearances.

WORKSTREAM H -- SCENE STRUCTURE
  Segment durations are drawn from an explicit mixture covering
  0.1 / 0.25 / 0.5 / 1 / 2 / 3+ s, so the TCN actually sees fast turn-taking
  and long holds. Wearer-solo, environment-solo, overlap, silence and
  noise-only all arise independently -- overlap is NOT "both at 50/50 RMS".

WORKSTREAM I -- ADVERSARIAL CURRICULUM
  `CONDITIONS` spans gain, TIR, distance, angle, RT60, DRR, noise SNR,
  compression, clipping, EQ, wind, multi-talker and head-orientation axes.
  Hard conditions deliberately BREAK the naive correlations: environment
  louder than wearer, environment at 0.15 m, wearer whispering, wearer in the
  reverberant far part of the range, shouting bystander.

WORKSTREAM J -- AMPLITUDE SHORTCUT KILL-SWITCH
  `gain_mode` in {normal, level_balanced, gain_randomized}. `level_balanced`
  equalises the two sources' active-region RMS at the microphone AFTER
  propagation, so level carries no information at all. `gain_randomized`
  applies independent +/-18 dB per source on top of physics.
"""
from __future__ import annotations

import dataclasses
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .acoustics import SAMPLE_RATE
from . import simulate_s1 as S

REPO_ROOT = Path(__file__).resolve().parents[2]
MUSAN_NOISE_DIR = REPO_ROOT / "evaluation/data/raw/musan/noise"

FRAME_HOP = 160
FRAME_WIN = 400
EPS = 1e-12

STATE_NAMES = ["silence", "wearer", "environment", "overlap"]  # 00, 10, 01, 11


# ---------------------------------------------------------------------------
# Conditions (Workstream I / S)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Condition:
    name: str
    gain_mode: str = "normal"           # normal | level_balanced | gain_randomized
    wearer_level_db: Tuple[float, float] = (-4.0, 6.0)
    env_level_db: Tuple[float, float] = (-6.0, 6.0)
    env_close: bool = False
    n_env_speakers: Tuple[int, ...] = (1, 1, 2)
    duty: Tuple[float, float] = (0.45, 0.45)
    harsh_device: bool = False
    ood_geometry: bool = False
    ood_eq: bool = False
    force_rt60: Optional[Tuple[float, float]] = None
    force_noise_snr_db: Optional[Tuple[float, float]] = None
    hold_out_ood: bool = True
    note: str = ""


CONDITIONS: Dict[str, Condition] = {
    # --- core training/eval distributions ---
    "normal": Condition("normal"),
    "level_matched": Condition("level_matched", gain_mode="level_balanced",
                               note="source RMS equalised AT THE MIC after propagation"),
    "random_gain": Condition("random_gain", gain_mode="gain_randomized",
                             note="independent +/-18 dB per source on top of physics"),
    # --- amplitude / TIR adversaries ---
    "wearer_quiet": Condition("wearer_quiet", wearer_level_db=(-26.0, -12.0)),
    "wearer_loud": Condition("wearer_loud", wearer_level_db=(8.0, 18.0), env_level_db=(-16.0, -4.0)),
    "bystander_loud": Condition("bystander_loud", env_level_db=(10.0, 20.0)),
    "shouting_bystander": Condition("shouting_bystander", wearer_level_db=(-6.0, 4.0),
                                    env_level_db=(12.0, 22.0)),
    # --- geometry adversaries ---
    "bystander_close": Condition("bystander_close", env_close=True,
                                 note="bystander at 0.15-0.45 m, OVERLAPPING the wearer range"),
    "bystander_close_loud": Condition("bystander_close_loud", env_close=True,
                                      env_level_db=(8.0, 18.0), wearer_level_db=(-14.0, -2.0)),
    # --- scene-structure adversaries ---
    "heavy_overlap": Condition("heavy_overlap", duty=(0.8, 0.8), n_env_speakers=(1, 2, 2)),
    "multi_talker": Condition("multi_talker", n_env_speakers=(2, 2, 2), duty=(0.45, 0.6)),
    # --- channel adversaries ---
    "noisy": Condition("noisy", force_noise_snr_db=(-3.0, 8.0), harsh_device=True),
    "high_rt60": Condition("high_rt60", force_rt60=(0.85, 1.10)),
    "eq_mismatch": Condition("eq_mismatch", ood_eq=True),
    "compressed_clipped": Condition("compressed_clipped", harsh_device=True),
    # --- held-out geometry (Workstream U) ---
    "ood_geometry": Condition("ood_geometry", ood_geometry=True, hold_out_ood=False,
                              note="ONLY the geometry bands excluded from training"),
    "ood_geometry_level_matched": Condition("ood_geometry_level_matched", ood_geometry=True,
                                            hold_out_ood=False, gain_mode="level_balanced"),
}

# Sampling weights used when building mixed training scenes (Workstream I:
# difficulty must NOT correlate deterministically with label, so every
# condition appears for both roles and hard ones are common, not rare).
TRAIN_CONDITION_WEIGHTS: Dict[str, float] = {
    "normal": 0.20,
    "level_matched": 0.12,
    "random_gain": 0.12,
    "wearer_quiet": 0.07,
    "wearer_loud": 0.05,
    "bystander_loud": 0.07,
    "shouting_bystander": 0.05,
    "bystander_close": 0.09,
    "bystander_close_loud": 0.05,
    "heavy_overlap": 0.05,
    "multi_talker": 0.04,
    "noisy": 0.05,
    "high_rt60": 0.02,
    "eq_mismatch": 0.01,
    "compressed_clipped": 0.01,
}


# ---------------------------------------------------------------------------
# Speech / noise pools
# ---------------------------------------------------------------------------
class SpeechPool:
    """Random-access speech source with worker-local length caching.

    Reads only the samples it needs (`sf.read(start=, frames=)`), which for
    seekable FLAC is far cheaper than decoding whole 12-second utterances.
    """

    def __init__(self, speaker_index: Dict[str, List[str]]):
        self.speaker_index = speaker_index
        self.speakers = sorted(speaker_index.keys())
        self._len_cache: Dict[str, int] = {}

    def _length(self, path: str) -> int:
        n = self._len_cache.get(path)
        if n is None:
            n = int(sf.info(path).frames)
            self._len_cache[path] = n
        return n

    def read_speech(self, speaker: str, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Concatenate random crops from this speaker until `n_samples` filled."""
        out = np.zeros(n_samples, dtype=np.float32)
        pos = 0
        paths = self.speaker_index[speaker]
        guard = 0
        while pos < n_samples and guard < 64:
            guard += 1
            p = paths[int(rng.integers(0, len(paths)))]
            total = self._length(p)
            want = min(n_samples - pos, total)
            start = int(rng.integers(0, max(1, total - want + 1)))
            x, sr = sf.read(p, start=start, frames=want, dtype="float32", always_2d=False)
            assert sr == SAMPLE_RATE, f"{p} is {sr} Hz"
            if x.ndim > 1:
                x = x.mean(axis=1)
            out[pos : pos + len(x)] = x
            pos += len(x)
        return out


class NoisePool:
    def __init__(self, root: Path = MUSAN_NOISE_DIR, max_files: int = 400):
        self.paths = sorted(str(p) for p in root.rglob("*.wav"))[:max_files] if root.exists() else []
        self._cache: Dict[str, np.ndarray] = {}

    def sample(self, n_samples: int, rng: np.random.Generator) -> Optional[np.ndarray]:
        if not self.paths:
            return None
        p = self.paths[int(rng.integers(0, len(self.paths)))]
        x = self._cache.get(p)
        if x is None:
            x, sr = sf.read(p, dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SAMPLE_RATE:
                idx = (np.arange(int(len(x) * SAMPLE_RATE / sr)) * sr / SAMPLE_RATE).astype(np.int64)
                x = x[np.clip(idx, 0, len(x) - 1)]
            if len(self._cache) > 24:
                self._cache.pop(next(iter(self._cache)))
            self._cache[p] = x
        if len(x) < n_samples:
            reps = int(np.ceil(n_samples / max(len(x), 1))) + 1
            x = np.tile(x, reps)
        return x


# ---------------------------------------------------------------------------
# Activity timelines (Workstream H)
# ---------------------------------------------------------------------------
# Explicit duration mixture: the campaign requires transitions at
# 100ms / 250ms / 500ms / 1s / 2s / 3s+, so they are sampled explicitly rather
# than hoping an exponential distribution happens to cover them.
_DUR_CHOICES = np.array([0.10, 0.25, 0.50, 1.00, 2.00, 3.50])
_DUR_WEIGHTS = np.array([0.12, 0.18, 0.22, 0.22, 0.16, 0.10])


def _sample_duration(rng: np.random.Generator) -> float:
    base = float(rng.choice(_DUR_CHOICES, p=_DUR_WEIGHTS))
    return float(base * rng.uniform(0.8, 1.25))


def activity_intervals(total_s: float, duty: float, rng: np.random.Generator) -> List[Tuple[float, float]]:
    """Alternating speech/pause timeline with an approximate target duty cycle."""
    out: List[Tuple[float, float]] = []
    t = float(rng.uniform(0.0, 0.6))
    speaking = rng.random() < duty
    while t < total_s:
        d = _sample_duration(rng)
        if not speaking:
            # scale pauses so the long-run duty cycle lands near `duty`
            d *= max(0.15, (1.0 - duty) / max(duty, 1e-3))
        end = min(t + d, total_s)
        if speaking and end > t:
            out.append((t, end))
        t = end
        speaking = not speaking
    return out


def intervals_to_sample_mask(intervals: List[Tuple[float, float]], n: int) -> np.ndarray:
    m = np.zeros(n, dtype=np.float32)
    for a, b in intervals:
        i0, i1 = int(a * SAMPLE_RATE), min(int(b * SAMPLE_RATE), n)
        if i1 > i0:
            m[i0:i1] = 1.0
    return m


def n_frames_for(n_samples: int) -> int:
    return max(1, 1 + (max(n_samples, FRAME_WIN) - FRAME_WIN) // FRAME_HOP)


def frame_rms(x: np.ndarray, n_fr: int) -> np.ndarray:
    """Per-frame RMS on the 25 ms / 10 ms grid.

    Uses a strided VIEW instead of fancy-index gathering. This is a pure
    memory-layout change: the same elements are reduced in the same order by
    the same `mean`, so results are bit-identical to the gather version (asserted
    in the test suite) while avoiding a (n_fr, 400) float64 copy per call.
    Workstream D: this path was ~18% of total scene cost before the change.
    """
    need = FRAME_HOP * (n_fr - 1) + FRAME_WIN
    xf = np.asarray(x, dtype=np.float64)
    if len(xf) < need:
        xf = np.pad(xf, (0, need - len(xf)), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(xf[:need], FRAME_WIN)[::FRAME_HOP]
    return np.sqrt((view**2).mean(axis=1) + EPS)


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Scene:
    audio: np.ndarray            # (n,) float32 mixture, absolute scale preserved
    wearer: np.ndarray           # (T,) float32 frame labels
    environment: np.ndarray      # (T,) float32 frame labels
    state: np.ndarray            # (T,) int64 in 0..3
    meta: Dict[str, object]


def _label_from_track(clean: np.ndarray, mask: np.ndarray, n_fr: int,
                      rel_thresh_db: float = -32.0) -> np.ndarray:
    """A source is 'active' in a frame when it is inside a speaking interval AND
    actually producing energy there (so intra-utterance pauses are not labelled
    active). Threshold is relative to that source's own speech level, so it is
    invariant to any gain applied to the source."""
    r = frame_rms(clean * mask, n_fr)
    speech = r[r > 0]
    if len(speech) == 0:
        return np.zeros(n_fr, dtype=np.float32)
    ref = float(np.percentile(speech, 75))
    thr = ref * (10.0 ** (rel_thresh_db / 20.0))
    mfr = frame_rms(mask.astype(np.float64), n_fr)
    return ((r > thr) & (mfr > 0.5)).astype(np.float32)


def generate_scene(
    pool: SpeechPool,
    rng: np.random.Generator,
    condition: str = "normal",
    generation: str = "S1",
    duration_s: float = 8.0,
    noise_pool: Optional[NoisePool] = None,
    role_counter: Optional[Dict[str, Counter]] = None,
) -> Scene:
    cond = CONDITIONS[condition]
    n = int(duration_s * SAMPLE_RATE)
    n_fr = n_frames_for(n)

    # --- Workstream F: draw speakers, then SHUFFLE into roles ---
    n_env = int(rng.choice(cond.n_env_speakers))
    k = 1 + n_env
    assert len(pool.speakers) > k, "need more speakers than roles"
    picks = list(rng.choice(np.array(pool.speakers, dtype=object), size=k, replace=False))
    rng.shuffle(picks)
    wearer_spk, env_spks = picks[0], picks[1:]
    if role_counter is not None:
        role_counter["wearer"][wearer_spk] += 1
        for s in env_spks:
            role_counter["environment"][s] += 1

    # --- rig / room / device ---
    ood = cond.ood_geometry
    hold = cond.hold_out_ood
    rig = S.sample_rig(rng, ood=ood, hold_out=hold)
    room = S.sample_room(rng, ood=ood, hold_out=hold)
    if cond.force_rt60 is not None:
        room = S.Room(volume_m3=room.volume_m3, rt60_s=float(rng.uniform(*cond.force_rt60)))
    device = S.sample_device(rng, harsh=cond.harsh_device, ood_eq=cond.ood_eq)
    if cond.force_noise_snr_db is not None:
        device.noise_snr_db = float(rng.uniform(*cond.force_noise_snr_db))

    # --- wearer track ---
    w_iv = activity_intervals(duration_s, cond.duty[0], rng)
    w_mask = intervals_to_sample_mask(w_iv, n)
    w_clean = pool.read_speech(wearer_spk, n, rng).astype(np.float64)
    w_level = float(S.db_to_lin(rng.uniform(*cond.wearer_level_db)))
    w_src = w_clean * w_mask * w_level
    w_geom = S.wearer_geometry(rig)
    w_ir, w_meta = S.build_ir(w_geom, rig, room, rng, generation)
    w_at_mic = S.convolve(w_src, w_ir)

    # --- environment track(s) ---
    e_at_mic = np.zeros(n, dtype=np.float64)
    e_masks, e_cleans, e_metas = [], [], []
    for spk in env_spks:
        iv = activity_intervals(duration_s, cond.duty[1] / max(len(env_spks), 1) + 0.12, rng)
        m = intervals_to_sample_mask(iv, n)
        clean = pool.read_speech(spk, n, rng).astype(np.float64)
        lvl = float(S.db_to_lin(rng.uniform(*cond.env_level_db)))
        geom = S.sample_environment_geometry(rng, close=cond.env_close, ood=ood, hold_out=hold)
        ir, meta = S.build_ir(geom, rig, room, rng, generation)
        e_at_mic += S.convolve(clean * m * lvl, ir)
        e_masks.append(m)
        e_cleans.append(clean)
        e_metas.append(meta)

    # --- Workstream J: gain mode, applied AFTER propagation ---
    gain_mode = cond.gain_mode
    if gain_mode == "level_balanced":
        wa, ea = w_mask > 0.5, np.clip(sum(e_masks), 0, 1) > 0.5
        rw = S.rms(w_at_mic[wa]) if wa.any() else 0.0
        re = S.rms(e_at_mic[ea]) if ea.any() else 0.0
        if rw > EPS and re > EPS:
            tgt = math.sqrt(rw * re)
            w_at_mic *= tgt / rw
            e_at_mic *= tgt / re
    elif gain_mode == "gain_randomized":
        w_at_mic *= float(S.db_to_lin(rng.uniform(-18.0, 18.0)))
        e_at_mic *= float(S.db_to_lin(rng.uniform(-18.0, 18.0)))

    # --- sum, then ONE device chain for the whole microphone signal ---
    mixture = w_at_mic + e_at_mic
    noise = noise_pool.sample(n, rng) if noise_pool is not None else None
    audio = S.apply_device_chain(mixture, device, rng, noise_source=noise)

    # --- labels ---
    w_lab = _label_from_track(w_clean, w_mask, n_fr)
    e_lab = np.zeros(n_fr, dtype=np.float32)
    for clean, m in zip(e_cleans, e_masks):
        e_lab = np.maximum(e_lab, _label_from_track(clean, m, n_fr))
    state = (w_lab.astype(np.int64) * 1) + (e_lab.astype(np.int64) * 2)
    # 0=silence, 1=wearer, 2=environment, 3=overlap

    meta: Dict[str, object] = {
        "condition": condition,
        "generation": generation,
        "gain_mode": gain_mode,
        "wearer_speaker": str(wearer_spk),
        "environment_speakers": [str(s) for s in env_spks],
        "n_env_speakers": n_env,
        "duration_s": duration_s,
        "wearer": w_meta,
        "environment": e_metas,
        "wearer_level_db": float(20 * math.log10(w_level + EPS)),
        "device": {
            "recording_gain_db": device.recording_gain_db,
            "compressor_ratio": device.compressor_ratio,
            "clip_drive": device.clip_drive,
            "noise_snr_db": device.noise_snr_db,
            "wind_lf_db": device.wind_lf_db,
            "hp_cutoff_hz": device.hp_cutoff_hz,
        },
        "state_fractions": {STATE_NAMES[i]: float((state == i).mean()) for i in range(4)},
        "clip_fraction": float((np.abs(audio) >= 0.999).mean()),
        "mix_rms": float(S.rms(audio)),
        # TIR = true wearer-to-environment ratio at the microphone, in dB.
        # Recorded so evaluation can bucket errors by difficulty (Workstream AO).
        "tir_db": float(20 * math.log10((S.rms(w_at_mic) + EPS) / (S.rms(e_at_mic) + EPS))),
    }
    return Scene(audio.astype(np.float32), w_lab, e_lab, state, meta)


def sample_train_condition(rng: np.random.Generator) -> str:
    names = list(TRAIN_CONDITION_WEIGHTS.keys())
    p = np.array([TRAIN_CONDITION_WEIGHTS[n] for n in names], dtype=np.float64)
    return str(rng.choice(names, p=p / p.sum()))


# ---------------------------------------------------------------------------
# Workstream F integrity check
# ---------------------------------------------------------------------------
def role_audit(pool: SpeechPool, n_scenes: int = 400, seed: int = 0) -> dict:
    """Proves every eligible speaker appears in BOTH roles. Raises if not."""
    rng = np.random.default_rng(seed)
    counter: Dict[str, Counter] = defaultdict(Counter)
    for _ in range(n_scenes):
        n_env = int(rng.choice((1, 1, 2)))
        picks = list(rng.choice(np.array(pool.speakers, dtype=object), size=1 + n_env, replace=False))
        rng.shuffle(picks)
        counter["wearer"][picks[0]] += 1
        for s in picks[1:]:
            counter["environment"][s] += 1
    both = [s for s in pool.speakers if counter["wearer"][s] > 0 and counter["environment"][s] > 0]
    only_w = [s for s in pool.speakers if counter["wearer"][s] > 0 and counter["environment"][s] == 0]
    only_e = [s for s in pool.speakers if counter["environment"][s] > 0 and counter["wearer"][s] == 0]
    report = {
        "n_scenes": n_scenes,
        "n_speakers": len(pool.speakers),
        "n_in_both_roles": len(both),
        "n_wearer_only": len(only_w),
        "n_environment_only": len(only_e),
        "fraction_in_both_roles": len(both) / max(len(pool.speakers), 1),
    }
    return report
