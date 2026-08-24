#!/usr/bin/env python3
"""V2 evaluation-autopsy infra: TIR-conditioned NEGATIVE verification trials
for the held-out validation manifest.

validation_suite_v2.npz (built by build_validation_suite.py) has, for each
overlap_random_tir{X} / overlap_hardpair_tir{X} slice, only POSITIVE
verification trials: the enrolled wearer's own voice mixed with an
interferer at TIR X. There is no NEGATIVE trial at those same TIR levels
(two speakers, NEITHER of which is the enrolled wearer, mixed at TIR X) --
so clip-level EER cannot be computed per-TIR from validation_suite_v2.npz
alone; only "clean" EER can (using clean_solo_wearer vs clean_solo_environment).

This script builds that missing negative leg, using the exact same
construction the historical scripts/model/evaluate_mentrawearnet.py TIR
section uses for its negative trials (`a_id, b_id = rng.sample(others, 2)`,
neither is the enrolled/query wearer_id), but on the HELD-OUT VALIDATION
manifest (evaluation/manifests/librispeech_train_clean_100_val.json), fully
deterministic (fixed seed), so it is validation-only data -- never the
day1_public final test set -- safe to use for checkpoint/threshold/
aggregation selection without touching the final protocol.

Output: evaluation/manifests/validation_suite_v2_tir_negatives.npz (same
schema as validation_suite_v2.npz: mixtures/enrollments/wearer_activities/
environment_activities, object arrays) + a companion _metadata.json.

Run: .venv/bin/python3 training/diagnostics/build_tir_negative_trials.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from training.data.mixture_generator import SAMPLE_RATE, SAMPLES_PER_FRAME, SpeakerPool, _fit_or_loop, _rms

VAL_MANIFEST = "evaluation/manifests/librispeech_train_clean_100_val.json"
GLOBAL_SEED = 20260822  # same fixed seed convention as build_validation_suite.py
TIR_LEVELS = (5, 0, -5, -10)
EXAMPLES_PER_TIR = 8  # matches build_validation_suite.py's EXAMPLES_PER_SLICE
DURATION_S = 2.0  # matches overlap_random_tir{X} / overlap_hardpair_tir{X} slices


def slice_rng(slice_name: str) -> random.Random:
    return random.Random(f"{GLOBAL_SEED}:tir_negative:{slice_name}")


def make_negative_trial(pool: SpeakerPool, rng: random.Random, wearer_id: str,
                         a_id: str, b_id: str, tir_db: float):
    """Mixture of a_id + b_id (NEITHER is wearer_id) at tir_db, enrolled
    against wearer_id -- a true impostor trial: the enrolled wearer's voice
    is absent from the mixture entirely."""
    n_samples = int(DURATION_S * SAMPLE_RATE)
    n_samples = (n_samples // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    a = _fit_or_loop(pool.random_clip(a_id, "test", rng), n_samples, rng)
    b = _fit_or_loop(pool.random_clip(b_id, "test", rng), n_samples, rng)
    b_scaled = b * ((_rms(a) / (10 ** (tir_db / 20))) / _rms(b))
    mixture = (a + b_scaled).astype(np.float32)
    peak = np.max(np.abs(mixture))
    if peak > 1.0:
        mixture = mixture / peak
    # wearer absent from the mixture -- both activity arrays are all-zero
    # w.r.t. the ENROLLED wearer_id (a_id/b_id's own activity is irrelevant
    # to the verification question this trial asks: "is the enrolled wearer
    # in this clip?" -- answer is no, by construction).
    wearer_activity = np.zeros(n_samples, dtype=np.float32)
    environment_activity = np.ones(n_samples, dtype=np.float32)
    return mixture, wearer_activity, environment_activity


def main():
    print(f"loading held-out validation manifest: {VAL_MANIFEST}")
    pool = SpeakerPool(VAL_MANIFEST)
    print(f"  {len(pool.speaker_ids)} speakers")

    examples = []
    for tir in TIR_LEVELS:
        slice_name = f"tir_negative_tir{tir:+d}"
        rng = slice_rng(slice_name)
        for _ in range(EXAMPLES_PER_TIR):
            wearer_id = rng.choice(pool.speaker_ids)
            others = [s for s in pool.speaker_ids if s != wearer_id]
            a_id, b_id = rng.sample(others, 2)
            mixture, w_act, e_act = make_negative_trial(pool, rng, wearer_id, a_id, b_id, tir)
            enrollment = pool.enrollment_clip(wearer_id, rng)
            examples.append({
                "mixture": mixture,
                "enrollment": enrollment,
                "wearer_activity": w_act,
                "environment_activity": e_act,
                "meta": {
                    "slice": slice_name, "wearer_id": wearer_id, "other_id": None,
                    "impostor_a_id": a_id, "impostor_b_id": b_id,
                    "duration_s": DURATION_S, "tir_db": tir, "pair_type": "tir_negative",
                    "noisy": False, "clip_is_target": False,
                },
            })
        print(f"  {slice_name}: {EXAMPLES_PER_TIR} examples")

    out = "evaluation/manifests/validation_suite_v2_tir_negatives.npz"
    meta_out = "evaluation/manifests/validation_suite_v2_tir_negatives_metadata.json"
    np.savez(
        out,
        mixtures=np.array([e["mixture"] for e in examples], dtype=object),
        enrollments=np.array([e["enrollment"] for e in examples], dtype=object),
        wearer_activities=np.array([e["wearer_activity"] for e in examples], dtype=object),
        environment_activities=np.array([e["environment_activity"] for e in examples], dtype=object),
    )
    Path(meta_out).write_text(json.dumps([e["meta"] for e in examples], indent=2))
    print(f"saved {out} and {meta_out} ({len(examples)} negative trials)")


if __name__ == "__main__":
    main()
