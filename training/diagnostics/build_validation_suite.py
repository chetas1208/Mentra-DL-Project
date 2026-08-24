#!/usr/bin/env python3
"""V2 infrastructure 2b: deterministic validation manifest / slices builder.

Generates and SAVES to disk a fixed validation set covering the slices the
engineering review specified:
  - clean solo (wearer-only, environment-only)
  - random-speaker overlap
  - hard-speaker-pair overlap (via training/diagnostics/hard_negative_mining.py, 2c)
  - TIR sweep: +5 / 0 / -5 / -10 dB
  - noisy (real MUSAN background noise)
  - short durations: 0.5s / 1.0s / 2.0s

Uses a FIXED SEED and the HELD-OUT VALIDATION speaker manifest
(evaluation/manifests/librispeech_train_clean_100_val.json -- 20 speakers,
ZERO overlap with the 211-speaker training manifest, verified), so
re-running this script produces the IDENTICAL set every time and the set
was never seen during V1 (or any future) training.

Output: a single .npz (object-dtype arrays: mixture, enrollment,
wearer_activity, environment_activity waveforms, one entry per example) at
evaluation/manifests/validation_suite_v2.npz, plus a companion
evaluation/manifests/validation_suite_v2_metadata.json (slice name, TIR,
duration, wearer/other speaker ids, pair type) for easy inspection without
loading the full npz.

Hard-negative pairs for the held-out validation speakers are computed
FRESH here (not reused from the training-manifest table in
evaluation/manifests/hard_negative_pairs.json, which only covers training
speakers) via training/diagnostics/hard_negative_mining.py's importable
functions, and also saved to evaluation/manifests/hard_negative_pairs_val.json.

Run: .venv/bin/python3 training/diagnostics/build_validation_suite.py --device cpu
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

from training.data.mixture_generator import SAMPLE_RATE, SAMPLES_PER_FRAME, NoisePool, SpeakerPool, _add_noise, _fit_or_loop, _rms
from training.diagnostics.hard_negative_mining import embed_all_speakers, compute_hard_negative_pairs
from training.models.mentrawearnet import MentraWearNet

VAL_MANIFEST = "evaluation/manifests/librispeech_train_clean_100_val.json"
NOISE_DIR = "evaluation/data/raw/musan/noise"
GLOBAL_SEED = 20260822  # fixed -- today's date at the time this suite was built, arbitrary but frozen
DURATIONS = (0.5, 1.0, 2.0)
TIR_LEVELS = (5, 0, -5, -10)
EXAMPLES_PER_SLICE = 8


def slice_rng(slice_name: str) -> random.Random:
    """Deterministic per-slice RNG, seeded from (GLOBAL_SEED, slice_name) so
    slices are independent and adding/removing a slice doesn't perturb the
    others' random draws."""
    return random.Random(f"{GLOBAL_SEED}:{slice_name}")


def make_solo_example(pool: SpeakerPool, rng: random.Random, speaker_id_for_content: str,
                       role: str, duration_s: float, add_noise: bool, noise_pool: NoisePool):
    """role: 'wearer' or 'environment' -- which activity array gets marked
    active for the FULL duration. speaker_id_for_content provides the audio;
    a SEPARATE wearer_id is chosen for enrollment (below) so 'environment'
    solo examples still have a meaningful (different) enrolled wearer to
    verify against, matching how the real deployment scenario works."""
    n_samples = int(duration_s * SAMPLE_RATE)
    n_samples = (n_samples // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    clip = pool.random_clip(speaker_id_for_content, "test", rng)
    mixture = _fit_or_loop(clip, n_samples, rng).astype(np.float32)
    if add_noise and noise_pool.available:
        mixture = _add_noise(mixture, noise_pool, rng)
    peak = np.max(np.abs(mixture))
    if peak > 1.0:
        mixture = mixture / peak
    wearer_activity = np.zeros(n_samples, dtype=np.float32)
    environment_activity = np.zeros(n_samples, dtype=np.float32)
    if role == "wearer":
        wearer_activity[:] = 1.0
    else:
        environment_activity[:] = 1.0
    return mixture, wearer_activity, environment_activity


def make_overlap_example(pool: SpeakerPool, rng: random.Random, wearer_id: str, other_id: str,
                          duration_s: float, tir_db: float, add_noise: bool, noise_pool: NoisePool):
    n_samples = int(duration_s * SAMPLE_RATE)
    n_samples = (n_samples // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME
    w = _fit_or_loop(pool.random_clip(wearer_id, "test", rng), n_samples, rng)
    o = _fit_or_loop(pool.random_clip(other_id, "test", rng), n_samples, rng)
    o_scaled = o * ((_rms(w) / (10 ** (tir_db / 20))) / _rms(o))
    mixture = (w + o_scaled).astype(np.float32)
    if add_noise and noise_pool.available:
        mixture = _add_noise(mixture, noise_pool, rng)
    peak = np.max(np.abs(mixture))
    if peak > 1.0:
        mixture = mixture / peak
    wearer_activity = np.ones(n_samples, dtype=np.float32)
    environment_activity = np.ones(n_samples, dtype=np.float32)
    return mixture, wearer_activity, environment_activity


def build_suite(pool: SpeakerPool, noise_pool: NoisePool, hard_pairs: dict):
    examples = []  # list of dicts: mixture, enrollment, wearer_activity, environment_activity, meta

    def add_example(slice_name, mixture, wearer_activity, environment_activity, wearer_id, other_id, meta_extra, rng):
        enrollment = pool.enrollment_clip(wearer_id, rng)
        examples.append({
            "mixture": mixture,
            "enrollment": enrollment,
            "wearer_activity": wearer_activity,
            "environment_activity": environment_activity,
            "meta": {"slice": slice_name, "wearer_id": wearer_id, "other_id": other_id, **meta_extra},
        })

    # --- 1. clean solo (wearer-only, environment-only), across durations ---
    for role in ("wearer", "environment"):
        for dur in DURATIONS:
            slice_name = f"clean_solo_{role}_{dur}s"
            rng = slice_rng(slice_name)
            for _ in range(EXAMPLES_PER_SLICE):
                wearer_id = rng.choice(pool.speaker_ids)
                other_id = rng.choice([s for s in pool.speaker_ids if s != wearer_id])
                content_id = wearer_id if role == "wearer" else other_id
                mixture, w_act, e_act = make_solo_example(pool, rng, content_id, role, dur,
                                                            add_noise=False, noise_pool=noise_pool)
                add_example(slice_name, mixture, w_act, e_act, wearer_id, other_id,
                            {"duration_s": dur, "noisy": False}, rng)

    # --- 2. random-speaker overlap, TIR sweep, duration=2.0s, no noise ---
    for tir in TIR_LEVELS:
        slice_name = f"overlap_random_tir{tir:+d}"
        rng = slice_rng(slice_name)
        for _ in range(EXAMPLES_PER_SLICE):
            wearer_id = rng.choice(pool.speaker_ids)
            other_id = rng.choice([s for s in pool.speaker_ids if s != wearer_id])
            mixture, w_act, e_act = make_overlap_example(pool, rng, wearer_id, other_id, 2.0, tir,
                                                           add_noise=False, noise_pool=noise_pool)
            add_example(slice_name, mixture, w_act, e_act, wearer_id, other_id,
                        {"duration_s": 2.0, "tir_db": tir, "pair_type": "random", "noisy": False}, rng)

    # --- 3. hard-speaker-pair overlap, TIR sweep, duration=2.0s, no noise ---
    for tir in TIR_LEVELS:
        slice_name = f"overlap_hardpair_tir{tir:+d}"
        rng = slice_rng(slice_name)
        for _ in range(EXAMPLES_PER_SLICE):
            wearer_id = rng.choice(pool.speaker_ids)
            candidates = hard_pairs.get(wearer_id, [s for s in pool.speaker_ids if s != wearer_id])
            other_id = rng.choice(candidates)
            mixture, w_act, e_act = make_overlap_example(pool, rng, wearer_id, other_id, 2.0, tir,
                                                           add_noise=False, noise_pool=noise_pool)
            add_example(slice_name, mixture, w_act, e_act, wearer_id, other_id,
                        {"duration_s": 2.0, "tir_db": tir, "pair_type": "hard", "noisy": False}, rng)

    # --- 4. noisy (real MUSAN noise), random-pair overlap at TIR 0dB ---
    slice_name = "overlap_random_tir0_noisy"
    rng = slice_rng(slice_name)
    for _ in range(EXAMPLES_PER_SLICE):
        wearer_id = rng.choice(pool.speaker_ids)
        other_id = rng.choice([s for s in pool.speaker_ids if s != wearer_id])
        mixture, w_act, e_act = make_overlap_example(pool, rng, wearer_id, other_id, 2.0, 0,
                                                       add_noise=True, noise_pool=noise_pool)
        add_example(slice_name, mixture, w_act, e_act, wearer_id, other_id,
                    {"duration_s": 2.0, "tir_db": 0, "pair_type": "random", "noisy": True}, rng)

    # --- 5. short durations, random-pair overlap at TIR 0dB, WITH noise ---
    for dur in DURATIONS:
        slice_name = f"short_duration_overlap_{dur}s"
        rng = slice_rng(slice_name)
        for _ in range(EXAMPLES_PER_SLICE):
            wearer_id = rng.choice(pool.speaker_ids)
            other_id = rng.choice([s for s in pool.speaker_ids if s != wearer_id])
            mixture, w_act, e_act = make_overlap_example(pool, rng, wearer_id, other_id, dur, 0,
                                                           add_noise=True, noise_pool=noise_pool)
            add_example(slice_name, mixture, w_act, e_act, wearer_id, other_id,
                        {"duration_s": dur, "tir_db": 0, "pair_type": "random", "noisy": True}, rng)

    return examples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=VAL_MANIFEST)
    ap.add_argument("--noise-dir", default=NOISE_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="evaluation/manifests/validation_suite_v2.npz")
    ap.add_argument("--meta-out", default="evaluation/manifests/validation_suite_v2_metadata.json")
    ap.add_argument("--hard-pairs-out", default="evaluation/manifests/hard_negative_pairs_val.json")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    print(f"loading held-out validation manifest: {args.manifest}")
    pool = SpeakerPool(args.manifest)
    print(f"  {len(pool.speaker_ids)} speakers")
    noise_pool = NoisePool(args.noise_dir)
    print(f"noise pool: {len(noise_pool.clips)} clips" if noise_pool.available else "noise pool: none")

    print("loading MentraWearNet backbone (for hard-negative-pair mining on val speakers) ...")
    model = MentraWearNet().to(args.device)
    model.eval()
    print(f"embedding {len(pool.speaker_ids)} validation speakers ...")
    embeddings = embed_all_speakers(pool, model, args.device)
    hard_pairs, sim, speaker_ids = compute_hard_negative_pairs(embeddings, k=args.k)
    Path(args.hard_pairs_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.hard_pairs_out).write_text(json.dumps(hard_pairs, indent=2))
    print(f"saved val-manifest hard-negative pairs to {args.hard_pairs_out}")

    print("building validation suite ...")
    examples = build_suite(pool, noise_pool, hard_pairs)
    print(f"  built {len(examples)} examples across "
          f"{len(set(e['meta']['slice'] for e in examples))} slices")

    # save waveforms as an object-dtype npz (variable-length arrays). Saving
    # object arrays always pickles them; loading them back requires passing
    # allow_pickle=True to np.load explicitly (numpy's load-side default is
    # False for security -- documented in the module docstring / see any
    # loader script that reads this file back).
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

    # per-slice counts, for a quick real sanity print
    from collections import Counter
    counts = Counter(e["meta"]["slice"] for e in examples)
    for slice_name, count in sorted(counts.items()):
        print(f"  {slice_name}: {count} examples")


if __name__ == "__main__":
    main()
