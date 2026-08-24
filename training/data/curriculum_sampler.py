#!/usr/bin/env python3
"""V2 infrastructure 2d: curriculum-stage mixture sampler.

New sibling module to training/data/mixture_generator.py -- does NOT modify
that file's generate_example() or any other existing function, so the
currently-training V1 job (which imports generate_example directly and
never imports this module) is completely unaffected.

Wraps the same segment-based construction generate_example() uses (silence
/ wearer / environment / overlap segments over a real speaker pool), but
exposes a `phase` parameter (A through E, per the engineering review's
curriculum table) that controls:
  - the state mix (how much of the timeline is silence/solo/overlap)
  - the TIR distribution for overlap segments
  - (phase E only) whether the interferer is drawn preferentially from a
    hard-negative-pairs table (training/diagnostics/hard_negative_mining.py,
    2c) instead of uniformly at random from the speaker pool

Phase progression (solo -> easy-overlap -> moderate-TIR -> hard-TIR ->
hard-negative-speakers), matching the review's stated curriculum shape:

  A: solo only            -- no overlap segments at all, easiest possible
  B: easy overlap         -- some overlap, TIR concentrated at +5/+10dB
  C: moderate TIR         -- more overlap, TIR concentrated around 0dB
  D: hard TIR             -- mostly overlap, TIR concentrated at -5/-10dB
  E: hard-negative spkrs  -- like D, but interferers preferentially drawn
                              from each wearer's hardest-to-distinguish
                              speakers (acoustically-similar, e.g. same
                              gender/pitch range -- see hard_negative_mining.py)

This module is standalone tooling: nothing in training/train.py imports it
yet, so it has zero effect on the currently-running V1 job. A future V2
training script would opt in explicitly.

Usage example:

    from training.data.mixture_generator import SpeakerPool, NoisePool
    from training.data.curriculum_sampler import CurriculumSampler
    import random, json

    pool = SpeakerPool("evaluation/manifests/librispeech_train_clean_100_train.json")
    noise_pool = NoisePool("evaluation/data/raw/musan/noise")
    hard_pairs = json.loads(open("evaluation/manifests/hard_negative_pairs.json").read())
    sampler = CurriculumSampler(pool, noise_pool, hard_pairs=hard_pairs)

    rng = random.Random(0)
    # early in training:
    example = sampler.generate(rng, phase="A", total_s=2.0)
    # later, once phase-A/B/C loss has plateaued:
    example = sampler.generate(rng, phase="E", total_s=2.0)
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from training.data.mixture_generator import (
    SAMPLE_RATE, SAMPLES_PER_FRAME, MixtureExample, NoisePool, SpeakerPool,
    _add_noise, _fit_or_loop, _rms,
)

STATES = ["SILENCE", "WEARER", "ENVIRONMENT", "OVERLAP"]


@dataclass(frozen=True)
class PhaseConfig:
    state_weights: tuple[float, float, float, float]  # SILENCE, WEARER, ENVIRONMENT, OVERLAP
    tir_levels: tuple[int, ...]
    tir_weights: tuple[float, ...]
    hard_negative_prob: float  # probability of drawing the interferer from the hard-pairs table (0 = always random)


# Reasonable defaults implementing the review's stated progression. Tunable
# per-call by passing a custom `phases` dict to CurriculumSampler.__init__
# instead of relying on these -- these are a sensible starting point, not
# claimed to be independently re-derived/tuned numbers.
DEFAULT_PHASES: dict[str, PhaseConfig] = {
    "A": PhaseConfig(state_weights=(0.30, 0.35, 0.35, 0.00),
                      tir_levels=(10,), tir_weights=(1.0,),
                      hard_negative_prob=0.0),
    "B": PhaseConfig(state_weights=(0.20, 0.30, 0.30, 0.20),
                      tir_levels=(10, 5, 0), tir_weights=(0.4, 0.4, 0.2),
                      hard_negative_prob=0.0),
    "C": PhaseConfig(state_weights=(0.15, 0.25, 0.25, 0.35),
                      tir_levels=(5, 0, -5), tir_weights=(0.2, 0.4, 0.4),
                      hard_negative_prob=0.0),
    "D": PhaseConfig(state_weights=(0.10, 0.20, 0.20, 0.50),
                      tir_levels=(0, -5, -10), tir_weights=(0.2, 0.4, 0.4),
                      hard_negative_prob=0.0),
    "E": PhaseConfig(state_weights=(0.10, 0.20, 0.20, 0.50),
                      tir_levels=(0, -5, -10), tir_weights=(0.15, 0.35, 0.5),
                      hard_negative_prob=0.8),
}


class CurriculumSampler:
    """Wraps generate_example()'s underlying segment-construction logic with
    a phase-controlled state mix / TIR distribution / interferer-selection
    strategy. See module docstring for a usage example."""

    def __init__(self, pool: SpeakerPool, noise_pool: NoisePool | None = None,
                 hard_pairs: dict | None = None, phases: dict[str, PhaseConfig] | None = None):
        self.pool = pool
        self.noise_pool = noise_pool
        self.hard_pairs = hard_pairs or {}
        self.phases = phases or DEFAULT_PHASES

    def _pick_other_id(self, rng: random.Random, wearer_id: str, cfg: PhaseConfig) -> str:
        use_hard = cfg.hard_negative_prob > 0 and rng.random() < cfg.hard_negative_prob
        if use_hard and wearer_id in self.hard_pairs and self.hard_pairs[wearer_id]:
            return rng.choice(self.hard_pairs[wearer_id])
        return rng.choice([s for s in self.pool.speaker_ids if s != wearer_id])

    def generate(self, rng: random.Random, phase: str, total_s: float = 2.0) -> MixtureExample:
        if phase not in self.phases:
            raise ValueError(f"unknown curriculum phase {phase!r}, expected one of {sorted(self.phases)}")
        cfg = self.phases[phase]

        wearer_id = rng.choice(self.pool.speaker_ids)
        other_id = self._pick_other_id(rng, wearer_id, cfg)

        n_samples = int(total_s * SAMPLE_RATE)
        n_plan_chunks = n_samples // SAMPLES_PER_FRAME
        n_samples = n_plan_chunks * SAMPLES_PER_FRAME

        n_segments = rng.randint(2, 4)
        segment_states = rng.choices(STATES, weights=list(cfg.state_weights), k=n_segments)
        cut_points = sorted(rng.sample(range(1, n_plan_chunks), n_segments - 1)) if n_segments > 1 and n_plan_chunks > 1 else []
        boundaries = [0] + cut_points + [n_plan_chunks]

        mixture = np.zeros(n_samples, dtype=np.float32)
        wearer_activity = np.zeros(n_samples, dtype=np.float32)
        environment_activity = np.zeros(n_samples, dtype=np.float32)

        wearer_clip = self.pool.random_clip(wearer_id, "test", rng)
        other_clip = self.pool.random_clip(other_id, "test", rng)

        for i, state in enumerate(segment_states):
            c0, c1 = boundaries[i], boundaries[i + 1]
            if c1 <= c0:
                continue
            s0 = c0 * SAMPLES_PER_FRAME
            seg_samples = (c1 - c0) * SAMPLES_PER_FRAME

            if state == "SILENCE":
                pass
            elif state == "WEARER":
                mixture[s0:s0 + seg_samples] = _fit_or_loop(wearer_clip, seg_samples, rng)
                wearer_activity[s0:s0 + seg_samples] = 1.0
            elif state == "ENVIRONMENT":
                mixture[s0:s0 + seg_samples] = _fit_or_loop(other_clip, seg_samples, rng)
                environment_activity[s0:s0 + seg_samples] = 1.0
            elif state == "OVERLAP":
                w = _fit_or_loop(wearer_clip, seg_samples, rng)
                o = _fit_or_loop(other_clip, seg_samples, rng)
                tir_db = rng.choices(list(cfg.tir_levels), weights=list(cfg.tir_weights), k=1)[0]
                o_scaled = o * ((_rms(w) / (10 ** (tir_db / 20))) / _rms(o))
                mixture[s0:s0 + seg_samples] = w + o_scaled
                wearer_activity[s0:s0 + seg_samples] = 1.0
                environment_activity[s0:s0 + seg_samples] = 1.0

        if self.noise_pool is not None and self.noise_pool.available:
            mixture = _add_noise(mixture, self.noise_pool, rng)
        else:
            mixture = mixture + np.random.RandomState(rng.randint(0, 2**31)).normal(0, 1e-4, size=mixture.shape).astype(np.float32)

        peak = np.max(np.abs(mixture))
        if peak > 1.0:
            mixture = mixture / peak

        enrollment = self.pool.enrollment_clip(wearer_id, rng)

        return MixtureExample(
            mixture=mixture,
            enrollment=enrollment,
            wearer_activity=wearer_activity,
            environment_activity=environment_activity,
            metadata={
                "wearer_id": wearer_id, "other_id": other_id,
                "segment_states": segment_states, "n_plan_chunks": n_plan_chunks,
                "phase": phase,
            },
        )
