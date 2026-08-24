"""Synthetic mixture generator with frame-level temporal activity labels
(sprint spec Track C, sections 20-24). Builds SELF/OTHER/OVERLAP/SILENCE
sequences from real speaker audio, not just single-segment mixtures --
the network needs to see transitions, not only static full-clip mixtures.

Uses the existing day1_public_speakers.json manifest (8 real LibriSpeech
speakers, already on disk) for the tiny-overfit correctness gate. A larger
corpus (train-clean-100) is a drop-in replacement via the same manifest
shape -- this generator doesn't care which JSON it's pointed at, only that
each speaker has enroll/test file lists.

Frame rate matches the measured SpeakerNet encoder stride: ~10ms/frame
(docs/SPEAKERNET_CONTEXT_REPORT.md). Labels are generated at that same
rate directly, not sample-rate labels downsampled later, to avoid a
silent alignment bug.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME_MS = 10  # matches measured encoder stride
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_MS / 1000)  # 160


def load_wav(path: str) -> np.ndarray:
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE}, got {sr}: {path}"
    return np.ascontiguousarray(data[:, 0])


@dataclass
class MixtureExample:
    mixture: np.ndarray            # [N] float32
    enrollment: np.ndarray         # [M] float32, different utterance than mixture content
    # Sample-resolution labels (length == len(mixture)), NOT pre-binned into
    # model frames. The model's actual encoder frame count for N samples is
    # NOT N // SAMPLES_PER_FRAME (measured: T ~= 0.00601*N + 16, a fixed
    # window offset, not a pure hop-based mapping -- see
    # docs/SPEAKERNET_CONTEXT_REPORT.md). Binning here at an assumed stride
    # would silently misalign against the model's real frame timing.
    # training/train.py's align_labels_to_frames() does the real alignment
    # once the model's actual frame_lengths is known from a forward pass.
    wearer_activity: np.ndarray        # [N] float32 in {0,1}
    environment_activity: np.ndarray   # [N] float32 in {0,1}
    metadata: dict


class NoisePool:
    """Lazy-loading pool of MUSAN noise/music clips for additive background
    augmentation. Optional -- mixture_generator works without one (no noise
    added), so this doesn't block anything already built while MUSAN
    extraction is in progress."""

    def __init__(self, noise_dir: str | None):
        self.clips: list[str] = []
        if noise_dir and Path(noise_dir).exists():
            self.clips = [str(p) for p in Path(noise_dir).rglob("*.wav")]
        self._cache: dict[str, np.ndarray] = {}

    @property
    def available(self) -> bool:
        return len(self.clips) > 0

    def random_clip(self, rng: random.Random) -> np.ndarray:
        path = rng.choice(self.clips)
        if path not in self._cache:
            self._cache[path] = load_wav(path)
        return self._cache[path]


def sample_snr_db(rng: random.Random) -> float:
    """Weighted SNR sampler for additive noise. Not uniform: mostly
    moderate-to-clean, occasional hard cases -- avoids training primarily
    on unrealistically noisy audio (section 26 -- curriculum, not
    maximally-hard-everywhere)."""
    levels = [20, 15, 10, 5, 0, -5]
    weights = [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]
    return rng.choices(levels, weights=weights, k=1)[0]


def _add_noise(signal: np.ndarray, noise_pool: NoisePool, rng: random.Random) -> np.ndarray:
    if not noise_pool.available:
        return signal
    noise_clip = noise_pool.random_clip(rng)
    noise_seg = _fit_or_loop(noise_clip, len(signal), rng)
    snr_db = sample_snr_db(rng)
    signal_rms = _rms(signal)
    if signal_rms < 1e-6:
        # SILENCE segment (no speech) -- add noise at a fixed modest level
        # rather than trying to hit an SNR target against ~zero signal.
        target_noise_rms = 0.02
    else:
        target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise_rms = _rms(noise_seg)
    scale = target_noise_rms / noise_rms if noise_rms > 1e-9 else 0.0
    return signal + noise_seg * scale


class SpeakerPool:
    """Wraps a manifest of {speaker_id: {enroll: [...], test: [...]}} and
    caches loaded audio so repeated sampling (tiny-overfit deliberately
    reuses the same handful of examples) doesn't re-read from disk."""

    def __init__(self, manifest_path: str):
        manifest = json.loads(Path(manifest_path).read_text())
        self.speakers: dict[str, dict] = manifest["speakers"]
        self.speaker_ids = sorted(self.speakers.keys())
        self._cache: dict[str, np.ndarray] = {}

    def _get(self, path: str) -> np.ndarray:
        if path not in self._cache:
            self._cache[path] = load_wav(path)
        return self._cache[path]

    def random_clip(self, speaker_id: str, split: str, rng: random.Random) -> np.ndarray:
        path = rng.choice(self.speakers[speaker_id][split])
        return self._get(path)

    def enrollment_clip(self, speaker_id: str, rng: random.Random) -> np.ndarray:
        # ALWAYS from the enroll list, never the same utterance used as
        # mixture content -- section 24's explicit leakage-prevention rule.
        return self._get(rng.choice(self.speakers[speaker_id]["enroll"]))


def _fit_or_loop(clip: np.ndarray, n_samples: int, rng: random.Random) -> np.ndarray:
    """Returns exactly n_samples from clip: random crop if longer, loop
    (not zero-pad) if shorter, so short utterances don't inject artificial
    silence into what's supposed to be an active-speech segment."""
    if len(clip) >= n_samples:
        start = rng.randint(0, len(clip) - n_samples)
        return clip[start:start + n_samples]
    reps = int(np.ceil(n_samples / len(clip)))
    return np.tile(clip, reps)[:n_samples]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def sample_tir_db(rng: random.Random) -> float:
    """Weighted TIR sampler, concentrated where SpeakerNet actually failed
    (measured: 13.3/20.8/29.2% EER at TIR 0/-5/-10dB -- see
    docs/MODEL_SELECTION.md). Not uniform: wearer-dominant mixtures are
    already known to be easy, don't spend training budget there."""
    levels = [10, 5, 0, -5, -10, -15]
    weights = [0.10, 0.10, 0.25, 0.25, 0.20, 0.10]
    return rng.choices(levels, weights=weights, k=1)[0]


def sample_duration_s(rng: random.Random) -> float:
    """Weighted duration sampler, concentrated 0.75-1.5s -- the product-
    relevant region where SpeakerNet is weakest (measured: 6.25% EER @1.0s
    vs 2.08% @2.0s). Training exclusively on 2-3s clips would never teach
    the model to be good at the duration that actually matters for latency."""
    durations = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    weights = [0.08, 0.20, 0.22, 0.18, 0.15, 0.12, 0.05]
    return rng.choices(durations, weights=weights, k=1)[0]


def generate_example(pool: SpeakerPool, rng: random.Random, total_s: float = 2.0,
                      noise_pool: "NoisePool | None" = None) -> MixtureExample:
    """One temporally-structured example: a random sequence of SILENCE /
    WEARER / ENVIRONMENT / OVERLAP segments (section 21) summing to
    total_s, built from real speaker audio with role randomization
    (section 23 -- the same person is sometimes wearer, sometimes not,
    across different calls to this function with different wearer_id)."""
    wearer_id = rng.choice(pool.speaker_ids)
    other_candidates = [s for s in pool.speaker_ids if s != wearer_id]
    other_id = rng.choice(other_candidates)

    n_samples = int(total_s * SAMPLE_RATE)
    # Planning grid: 10ms chunks used only to decide WHERE segment
    # boundaries fall in time -- an arbitrary-but-reasonable planning
    # resolution, independent of the model's actual encoder stride (labels
    # below are sample-resolution and get properly resampled to the
    # model's real frame count later, see the class docstring/comment above).
    n_plan_chunks = n_samples // SAMPLES_PER_FRAME
    n_samples = n_plan_chunks * SAMPLES_PER_FRAME  # exact chunk boundary, no remainder

    # Random segment plan: 2-4 segments, each one of the 4 states,
    # weighted so SILENCE/OVERLAP are less common than single-speaker
    # segments (section 22 -- verify empirical balance, not perfectly
    # uniform, but no state should be structurally absent).
    states = ["SILENCE", "WEARER", "ENVIRONMENT", "OVERLAP"]
    weights = [0.15, 0.35, 0.35, 0.15]
    n_segments = rng.randint(2, 4)
    segment_states = rng.choices(states, weights=weights, k=n_segments)

    # split total planning chunks into n_segments roughly-equal-but-randomized pieces
    cut_points = sorted(rng.sample(range(1, n_plan_chunks), n_segments - 1)) if n_segments > 1 else []
    boundaries = [0] + cut_points + [n_plan_chunks]

    mixture = np.zeros(n_samples, dtype=np.float32)
    wearer_activity = np.zeros(n_samples, dtype=np.float32)
    environment_activity = np.zeros(n_samples, dtype=np.float32)

    wearer_clip = pool.random_clip(wearer_id, "test", rng)
    other_clip = pool.random_clip(other_id, "test", rng)

    for i, state in enumerate(segment_states):
        c0, c1 = boundaries[i], boundaries[i + 1]
        if c1 <= c0:
            continue
        s0 = c0 * SAMPLES_PER_FRAME
        seg_samples = (c1 - c0) * SAMPLES_PER_FRAME

        if state == "SILENCE":
            pass  # mixture segment stays zero; low-level noise floor added below globally
        elif state == "WEARER":
            mixture[s0:s0 + seg_samples] = _fit_or_loop(wearer_clip, seg_samples, rng)
            wearer_activity[s0:s0 + seg_samples] = 1.0
        elif state == "ENVIRONMENT":
            mixture[s0:s0 + seg_samples] = _fit_or_loop(other_clip, seg_samples, rng)
            environment_activity[s0:s0 + seg_samples] = 1.0
        elif state == "OVERLAP":
            w = _fit_or_loop(wearer_clip, seg_samples, rng)
            o = _fit_or_loop(other_clip, seg_samples, rng)
            # TIR-controlled overlap (weighted toward the measured failure
            # region, sample_tir_db() above), not a flat unscaled sum --
            # an unscaled sum is an uncontrolled, usually-easy TIR.
            tir_db = sample_tir_db(rng)
            o_scaled = o * ((_rms(w) / (10 ** (tir_db / 20))) / _rms(o))
            mixture[s0:s0 + seg_samples] = w + o_scaled
            wearer_activity[s0:s0 + seg_samples] = 1.0
            environment_activity[s0:s0 + seg_samples] = 1.0

    # Real additive background noise (MUSAN, CC BY 4.0) when a noise_pool
    # is supplied -- applied to the whole mixture, not per-segment, since
    # real background noise is continuous across speech/silence boundaries,
    # not something that switches on/off with the speaker. Falls back to a
    # tiny noise floor (not real augmentation, just avoids an unrealistic
    # all-zero SILENCE input) when no noise_pool is available yet.
    if noise_pool is not None and noise_pool.available:
        mixture = _add_noise(mixture, noise_pool, rng)
    else:
        mixture = mixture + np.random.RandomState(rng.randint(0, 2**31)).normal(0, 1e-4, size=mixture.shape).astype(np.float32)

    peak = np.max(np.abs(mixture))
    if peak > 1.0:
        mixture = mixture / peak

    enrollment = pool.enrollment_clip(wearer_id, rng)

    return MixtureExample(
        mixture=mixture,
        enrollment=enrollment,
        wearer_activity=wearer_activity,
        environment_activity=environment_activity,
        metadata={
            "wearer_id": wearer_id, "other_id": other_id,
            "segment_states": segment_states, "n_plan_chunks": n_plan_chunks,
        },
    )
