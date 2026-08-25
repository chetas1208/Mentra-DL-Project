#!/usr/bin/env python3
"""V3B Phase 2: hard-impostor evaluation suite builder.

Builds a NEW, fixed, deterministic, speaker-disjoint evaluation suite on
the HELD-OUT validation manifest (evaluation/manifests/
librispeech_train_clean_100_val.json -- 20 speakers, ZERO overlap with the
211 training speakers, verified in this run). This suite is separate from
validation_suite_v2.npz (V2/V3A's suite) -- it exists specifically to give
V3B's hard-negative-mining experiment a suite with an EXPLICIT, deliberate
hard-vs-random impostor split at every difficulty level, rather than V2/
V3A's frame-dataset scripts' opportunistic "did the randomly-drawn
interferer happen to land in the top-K list" post-hoc tagging (which, with
only 20 held-out speakers and a top-50 table, would degenerate to "nearly
every trial", since top-50 saturates at top-19 = everyone).

Trial design (clip-level speaker-verification framing, matching this
project's existing day1_public historical-protocol convention of scoring
one enrollment vs one test mixture per trial):

  For each of 6 conditions -- "clean", "tir+10", "tir+5", "tir+0",
  "tir-5", "tir-10" -- three trial TYPES are built:

  - positive: enrolled wearer W's own speech (solo at "clean", or
    W-dominant overlap with a RANDOMLY drawn interferer at the given TIR).
    Ground truth: wearer present (label=1).
  - random_negative: W is NOT in the audio at all. A random impostor I1
    speaks (solo at "clean", or I1-dominant overlap with a second random
    speaker I2 at the given TIR). I1 is drawn uniformly at random from the
    other 19 validation speakers. Ground truth: wearer absent (label=0).
  - hard_negative: same construction as random_negative, except I1 is
    drawn from hard_negative_pairs_val.json[W] -- W's own nearest
    neighbors by frozen-SpeakerNet cosine similarity, computed FRESH among
    the 20 held-out validation speakers only (training/diagnostics/
    hard_negative_mining.py's importable functions, run against
    VAL_MANIFEST, NOT reusing the 211-training-speaker table). k=5 is used
    for this table (not k=50): with only 20 validation speakers, top-50
    caps out at top-19 = literally everyone, which would make "hard" and
    "random" the same population. k=5 (~26% of the pool) keeps a
    meaningful minority "acoustically closest" definition, matching the
    ALREADY-VERIFIED hard_negative_pairs_val.json this repo already ships
    (see training/diagnostics/hard_negative_mining.py's k=50 run report:
    the val-manifest k=50 output is identical in content to k=19/all).
    This directly measures the failure mode hard-mining targets: an
    impostor who sounds like the wearer, mistaken for the wearer.

Each trial's enrollment is drawn from W's `enroll` list (never `test`,
matching every other generator in this codebase's leakage-prevention
convention).

CALIBRATION SPLIT: the first CALIB_FRACTION (default 20%) of each
(condition, trial_type) bucket, in deterministic generation order, is
tagged split="calibration"; the rest split="eval". Threshold selection for
FAR/FRR must use ONLY the calibration split; the eval split's numbers are
the ones reported. This split is frozen at build time, before any
threshold is ever chosen.

Output:
  evaluation/manifests/hard_eval_suite_v1.npz          (mixtures/enrollments/activities, object-dtype)
  evaluation/manifests/hard_eval_suite_v1_metadata.json (one record per trial)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from training.data.mixture_generator import (
    SAMPLE_RATE, SAMPLES_PER_FRAME, NoisePool, SpeakerPool, _fit_or_loop, _rms,
)

VAL_MANIFEST = "evaluation/manifests/librispeech_train_clean_100_val.json"
HARD_PAIRS_VAL = "evaluation/manifests/hard_negative_pairs_val.json"  # k=5, pre-verified: no self-pairs, no train-speaker leakage, deterministic
GLOBAL_SEED = 20260824  # frozen at build time, distinct from every other suite's seed in this repo
CONDITIONS = [("clean", None), ("tir+10", 10), ("tir+5", 5), ("tir+0", 0), ("tir-5", -5), ("tir-10", -10)]
N_TARGET_PER_BUCKET = 200  # brief's target: >=200 trials per (condition, trial_type)
CALIB_FRACTION = 0.20
TOTAL_S = 2.0


def cond_rng(condition: str, trial_type: str) -> random.Random:
    return random.Random(f"{GLOBAL_SEED}:{condition}:{trial_type}")


def _solo_clip(pool: SpeakerPool, rng: random.Random, speaker_id: str, total_s: float) -> np.ndarray:
    n_samples = int(total_s * SAMPLE_RATE)
    n_samples = (n_samples // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    clip = pool.random_clip(speaker_id, "test", rng)
    mixture = _fit_or_loop(clip, n_samples, rng).astype(np.float32)
    peak = np.max(np.abs(mixture))
    if peak > 1.0:
        mixture = mixture / peak
    return mixture, n_samples


def _overlap_clip(pool: SpeakerPool, rng: random.Random, dominant_id: str, interferer_id: str,
                   tir_db: float, total_s: float) -> np.ndarray:
    n_samples = int(total_s * SAMPLE_RATE)
    n_samples = (n_samples // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    d = _fit_or_loop(pool.random_clip(dominant_id, "test", rng), n_samples, rng)
    i = _fit_or_loop(pool.random_clip(interferer_id, "test", rng), n_samples, rng)
    i_scaled = i * ((_rms(d) / (10 ** (tir_db / 20))) / _rms(i))
    mixture = (d + i_scaled).astype(np.float32)
    peak = np.max(np.abs(mixture))
    if peak > 1.0:
        mixture = mixture / peak
    return mixture, n_samples


def build_positive_trial(pool: SpeakerPool, rng: random.Random, tir_db, total_s=TOTAL_S):
    wearer_id = rng.choice(pool.speaker_ids)
    interferer_id = rng.choice([s for s in pool.speaker_ids if s != wearer_id])
    if tir_db is None:
        mixture, n = _solo_clip(pool, rng, wearer_id, total_s)
        w_act = np.ones(n, dtype=np.float32)
        e_act = np.zeros(n, dtype=np.float32)
        interferer_id = None
    else:
        mixture, n = _overlap_clip(pool, rng, wearer_id, interferer_id, tir_db, total_s)
        w_act = np.ones(n, dtype=np.float32)
        e_act = np.ones(n, dtype=np.float32)
    enrollment = pool.enrollment_clip(wearer_id, rng)
    return dict(mixture=mixture, enrollment=enrollment, wearer_activity=w_act, environment_activity=e_act,
                meta=dict(wearer_id=wearer_id, dominant_id=wearer_id, interferer_id=interferer_id, label=1))


def build_negative_trial(pool: SpeakerPool, rng: random.Random, tir_db, hard: bool, hard_pairs: dict, total_s=TOTAL_S):
    wearer_id = rng.choice(pool.speaker_ids)  # the ENROLLED (but acoustically absent) identity
    if hard:
        candidates = [s for s in hard_pairs.get(wearer_id, []) if s != wearer_id]
        if not candidates:
            candidates = [s for s in pool.speaker_ids if s != wearer_id]
    else:
        candidates = [s for s in pool.speaker_ids if s != wearer_id]
    impostor_id = rng.choice(candidates)
    if tir_db is None:
        mixture, n = _solo_clip(pool, rng, impostor_id, total_s)
        w_act = np.zeros(n, dtype=np.float32)
        e_act = np.ones(n, dtype=np.float32)
        secondary_id = None
    else:
        secondary_id = rng.choice([s for s in pool.speaker_ids if s not in (wearer_id, impostor_id)])
        mixture, n = _overlap_clip(pool, rng, impostor_id, secondary_id, tir_db, total_s)
        w_act = np.zeros(n, dtype=np.float32)
        e_act = np.ones(n, dtype=np.float32)
    enrollment = pool.enrollment_clip(wearer_id, rng)  # enroll against the TRUE (absent) wearer
    return dict(mixture=mixture, enrollment=enrollment, wearer_activity=w_act, environment_activity=e_act,
                meta=dict(wearer_id=wearer_id, dominant_id=impostor_id, interferer_id=secondary_id, label=0,
                          is_hard_pair_by_construction=hard))


def build_suite(pool: SpeakerPool, hard_pairs: dict):
    examples = []
    achieved = {}
    for condition, tir_db in CONDITIONS:
        for trial_type in ("positive", "random_negative", "hard_negative"):
            rng = cond_rng(condition, trial_type)
            n_calib = int(round(N_TARGET_PER_BUCKET * CALIB_FRACTION))
            bucket_examples = []
            for i in range(N_TARGET_PER_BUCKET):
                if trial_type == "positive":
                    ex = build_positive_trial(pool, rng, tir_db)
                else:
                    ex = build_negative_trial(pool, rng, tir_db, hard=(trial_type == "hard_negative"),
                                               hard_pairs=hard_pairs)
                ex["meta"]["condition"] = condition
                ex["meta"]["tir_db"] = tir_db
                ex["meta"]["trial_type"] = trial_type
                ex["meta"]["split"] = "calibration" if i < n_calib else "eval"
                bucket_examples.append(ex)
            examples.extend(bucket_examples)
            achieved[f"{condition}/{trial_type}"] = len(bucket_examples)
    return examples, achieved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=VAL_MANIFEST)
    ap.add_argument("--hard-pairs", default=HARD_PAIRS_VAL)
    ap.add_argument("--out", default="evaluation/manifests/hard_eval_suite_v1.npz")
    ap.add_argument("--meta-out", default="evaluation/manifests/hard_eval_suite_v1_metadata.json")
    args = ap.parse_args()

    print(f"loading held-out validation manifest: {args.manifest}")
    pool = SpeakerPool(args.manifest)
    print(f"  {len(pool.speaker_ids)} speakers: {pool.speaker_ids}")

    hard_pairs = json.loads(Path(args.hard_pairs).read_text())
    print(f"loaded hard-negative pairs (val-only, k={len(next(iter(hard_pairs.values())))}) from {args.hard_pairs}")

    # Leakage/self-pair sanity checks (Phase 1-style audit, applied here too
    # since this table drives negative-trial construction directly).
    train_manifest = json.loads(Path("evaluation/manifests/librispeech_train_clean_100_train.json").read_text())
    train_speakers = set(train_manifest["speakers"].keys())
    for sid, lst in hard_pairs.items():
        assert sid not in lst, f"self-pair found for {sid}"
        assert sid not in train_speakers, f"train-speaker leakage: {sid} used as a validation speaker key"
        for other in lst:
            assert other not in train_speakers, f"train-speaker leakage: {other} appears in val hard-pairs list for {sid}"
    print("  audit OK: no self-pairs, no train-speaker leakage in the hard-pairs table")

    print(f"building hard-eval suite: {len(CONDITIONS)} conditions x 3 trial types x {N_TARGET_PER_BUCKET} trials ...")
    examples, achieved = build_suite(pool, hard_pairs)
    print(f"  built {len(examples)} total trials")
    for k, v in achieved.items():
        print(f"    {k}: {v}")

    np.savez(
        args.out,
        mixtures=np.array([e["mixture"] for e in examples], dtype=object),
        enrollments=np.array([e["enrollment"] for e in examples], dtype=object),
        wearer_activities=np.array([e["wearer_activity"] for e in examples], dtype=object),
        environment_activities=np.array([e["environment_activity"] for e in examples], dtype=object),
    )
    metadata = [e["meta"] for e in examples]
    Path(args.meta_out).write_text(json.dumps(metadata, indent=2))
    print(f"saved {args.out} and {args.meta_out}")


if __name__ == "__main__":
    main()
