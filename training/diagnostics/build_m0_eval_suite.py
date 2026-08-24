#!/usr/bin/env python3
"""M0 Phase 1: deterministic, speaker-disjoint (VAL manifest), multi-segment
evaluation suite for the M0 mask-predictor project. Bigger and more
statistically stable than R0's 240-trial supplement (whose own docstring
flags n_neg as low as 1 in some sub-buckets).

Reuses the same multi-segment generator already used by R0/v2_frame_dataset
(build_clean_example for "clean" -- SILENCE/WEARER/ENVIRONMENT only, no
overlap ever -- and build_overlap_bucket_example for the TIR buckets --
SILENCE/WEARER/ENVIRONMENT/OVERLAP with real within-clip transitions), same
VAL_MANIFEST speakers as every other held-out eval in this project (disjoint
from TRAIN_MANIFEST speakers used for M0 training).

Each generated clip becomes ONE verification trial pair: the clip's mixture
audio is scored (elsewhere, at eval time) against BOTH the true wearer's
enrollment (positive) and a random impostor's enrollment (negative) -- same
protocol as R0's run_multisegment_oracle_supplement, so 1 clip = 1 pos + 1
neg trial, giving n_pos == n_neg == n_clips per bucket.

NO model is loaded here -- purely audio + sample-level activity labels + a
fixed calibration/eval split (by trial index parity within each bucket, same
convention as R0's predicted-mask calibration split), saved up front so
threshold selection downstream can never leak into the reported eval numbers.

Output:
  evaluation/manifests/m0_eval_suite_v1.npz       (object arrays: mixture,
    enrollment (true wearer), impostor_enrollment, wearer_activity,
    environment_activity -- one row per clip)
  evaluation/manifests/m0_eval_suite_v1_metadata.json (per-clip: tir_tag,
    split, wearer_id, other_id, impostor_id, state_occupancy fractions
    computed at SAMPLE resolution from the activity tracks)

Run: .venv/bin/python3 training/diagnostics/build_m0_eval_suite.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from training.data.mixture_generator import SpeakerPool
from training.diagnostics.v2_frame_dataset import build_clean_example, build_overlap_bucket_example

VAL_MANIFEST = "evaluation/manifests/librispeech_train_clean_100_val.json"
TIR_LEVELS = (5, 0, -5, -10)
OUT_NPZ = "evaluation/manifests/m0_eval_suite_v1.npz"
OUT_META = "evaluation/manifests/m0_eval_suite_v1_metadata.json"
SEED = 20260827  # fresh seed, distinct from R0's supplement (20260826) and its train/calib seeds


def state_occupancy(w_act: np.ndarray, e_act: np.ndarray) -> dict:
    w = w_act > 0.5
    e = e_act > 0.5
    n = len(w_act)
    return {
        "00": float(np.mean(~w & ~e)),
        "10": float(np.mean(w & ~e)),
        "01": float(np.mean(~w & e)),
        "11": float(np.mean(w & e)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples-per-tir", type=int, default=300)
    ap.add_argument("--examples-clean", type=int, default=300)
    ap.add_argument("--total-s", type=float, default=2.0)
    ap.add_argument("--calib-frac", type=float, default=0.2,
                     help="fraction of each bucket reserved for threshold calibration "
                          "(remainder is the reported held-out eval set)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    t0 = time.time()

    pool = SpeakerPool(VAL_MANIFEST)
    print(f"loaded VAL manifest pool: {len(pool.speaker_ids)} speakers")
    rng = random.Random(args.seed)

    mixtures, enrollments, impostor_enrollments = [], [], []
    wearer_activities, environment_activities = [], []
    metadata = []

    def add_clip(mixture, w_act, e_act, wearer_id, other_id, tir_tag, bucket_pos):
        impostor_candidates = [s for s in pool.speaker_ids if s not in (wearer_id, other_id)]
        impostor_id = rng.choice(impostor_candidates)
        enrollment = pool.enrollment_clip(wearer_id, rng)
        impostor_enrollment = pool.enrollment_clip(impostor_id, rng)
        split = "calib" if bucket_pos < args.calib_frac * bucket_size else "eval"
        mixtures.append(mixture.astype(np.float32))
        enrollments.append(np.asarray(enrollment, dtype=np.float32))
        impostor_enrollments.append(np.asarray(impostor_enrollment, dtype=np.float32))
        wearer_activities.append(w_act.astype(np.float32))
        environment_activities.append(e_act.astype(np.float32))
        occ = state_occupancy(w_act, e_act)
        metadata.append({
            "tir_tag": tir_tag, "split": split, "wearer_id": wearer_id, "other_id": other_id,
            "impostor_id": impostor_id, "n_samples": int(len(mixture)),
            "state_occupancy": occ,
        })

    bucket_size = args.examples_clean
    print(f"generating clean bucket: {args.examples_clean} clips ...")
    for i in range(args.examples_clean):
        mixture, enrollment, w_act, e_act, wearer_id, other_id = build_clean_example(pool, rng, args.total_s)
        add_clip(mixture, w_act, e_act, wearer_id, other_id, "clean", i)

    for tir in TIR_LEVELS:
        bucket_size = args.examples_per_tir
        print(f"generating TIR={tir:+d}dB bucket: {args.examples_per_tir} clips ...")
        for i in range(args.examples_per_tir):
            mixture, enrollment, w_act, e_act, wearer_id, other_id = build_overlap_bucket_example(pool, rng, tir, args.total_s)
            add_clip(mixture, w_act, e_act, wearer_id, other_id, f"{tir:+d}", i)

    n = len(mixtures)
    print(f"total clips: {n}")

    tir_tags = [m["tir_tag"] for m in metadata]
    splits = [m["split"] for m in metadata]
    from collections import Counter
    print("per-bucket split counts:", {tag: dict(Counter(s for t, s in zip(tir_tags, splits) if t == tag))
                                         for tag in ["clean", "+5", "+0", "-5", "-10"]})

    Path(OUT_NPZ).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_NPZ,
        mixture=np.array(mixtures, dtype=object),
        enrollment=np.array(enrollments, dtype=object),
        impostor_enrollment=np.array(impostor_enrollments, dtype=object),
        wearer_activity=np.array(wearer_activities, dtype=object),
        environment_activity=np.array(environment_activities, dtype=object),
    )
    Path(OUT_META).write_text(json.dumps({
        "seed": args.seed, "n_clips": n, "total_s": args.total_s,
        "calib_frac": args.calib_frac, "val_manifest": VAL_MANIFEST,
        "clips": metadata,
    }, indent=2))
    print(f"saved {OUT_NPZ} and {OUT_META}")
    print(f"total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
