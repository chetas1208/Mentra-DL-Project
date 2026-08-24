#!/usr/bin/env python3
"""V2 evaluation autopsy: canonical joint-logit frame dataset (Phase 3).

Extends scripts/model/evaluate_mentrawearnet_metrics.py's structured
SILENCE/WEARER/ENVIRONMENT/OVERLAP example generator (build_frame_example)
to dump RAW per-frame records (not just aggregated AUROC/F1 numbers) --
the aggregated-only script doesn't save enough to answer phases 4-11
(state-conditional distributions, 10-vs-01 discrimination, joint-logit
geometry, calibration, aggregation sweep all need the raw logits).

Also adds a "clean" bucket: SILENCE/WEARER/ENVIRONMENT segments only, NO
overlap ever mixed in -- this is what makes genuine, TIR-free 10-only and
01-only frames possible (validation_suite_v2.npz's overlap slices are
100% state=11 for their full duration by construction, so it alone cannot
answer "10 vs 01 discrimination at TIR X" -- see report for why this
script exists as well as validation_suite_v2.npz).

Uses the HELD-OUT validation speaker manifest (20 speakers, zero overlap
with the 211 training speakers, zero overlap with day1_public), same as
evaluate_mentrawearnet_metrics.py, fixed seed -> fully reproducible.

Output: evaluation/results/mentrawearnet_v2_<tag>_joint_frame_dataset.npz
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
from training.models.mentrawearnet import MentraWearNet
from training.train import align_labels_to_frames

VAL_MANIFEST = "evaluation/manifests/librispeech_train_clean_100_val.json"
NOISE_DIR = "evaluation/data/raw/musan/noise"
HARD_PAIRS_VAL = "evaluation/manifests/hard_negative_pairs_val.json"
GLOBAL_SEED = 20260823  # fixed, distinct from build_validation_suite.py's seed -- independent draw
TIR_LEVELS = [5, 0, -5, -10]
STATE_NAMES = ["00", "10", "01", "11"]


def build_overlap_bucket_example(pool: SpeakerPool, rng: random.Random, tir_db: float, total_s: float):
    """Same construction as evaluate_mentrawearnet_metrics.py's
    build_frame_example: SILENCE/WEARER/ENVIRONMENT/OVERLAP segments, OVERLAP
    segments pinned to tir_db. No noise added (kept clean/controlled so TIR
    is the only manipulated variable) -- matches this repo's existing
    convention of separate noisy vs TIR-sweep slices."""
    wearer_id = rng.choice(pool.speaker_ids)
    other_id = rng.choice([s for s in pool.speaker_ids if s != wearer_id])

    n_samples = int(total_s * SAMPLE_RATE)
    n_plan_chunks = n_samples // SAMPLES_PER_FRAME
    n_samples = n_plan_chunks * SAMPLES_PER_FRAME

    states = ["SILENCE", "WEARER", "ENVIRONMENT", "OVERLAP"]
    weights = [0.15, 0.35, 0.35, 0.15]
    n_segments = rng.randint(2, 4)
    segment_states = rng.choices(states, weights=weights, k=n_segments)
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
            o_scaled = o * ((_rms(w) / (10 ** (tir_db / 20))) / _rms(o))
            mixture[s0:s0 + seg_samples] = w + o_scaled
            wearer_activity[s0:s0 + seg_samples] = 1.0
            environment_activity[s0:s0 + seg_samples] = 1.0

    mixture = mixture + np.random.RandomState(rng.randint(0, 2**31)).normal(0, 1e-4, size=mixture.shape).astype(np.float32)
    peak = np.max(np.abs(mixture))
    if peak > 1.0:
        mixture = mixture / peak

    enrollment = pool.enrollment_clip(wearer_id, rng)
    return mixture, enrollment, wearer_activity, environment_activity, wearer_id, other_id


def build_clean_example(pool: SpeakerPool, rng: random.Random, total_s: float):
    """SILENCE/WEARER/ENVIRONMENT segments only -- NEVER OVERLAP. Gives
    genuinely TIR-free 00/10/01 frames (validation_suite_v2.npz's overlap
    slices are pure state=11 with no silence segments at all)."""
    wearer_id = rng.choice(pool.speaker_ids)
    other_id = rng.choice([s for s in pool.speaker_ids if s != wearer_id])

    n_samples = int(total_s * SAMPLE_RATE)
    n_plan_chunks = n_samples // SAMPLES_PER_FRAME
    n_samples = n_plan_chunks * SAMPLES_PER_FRAME

    states = ["SILENCE", "WEARER", "ENVIRONMENT"]
    weights = [0.2, 0.4, 0.4]
    n_segments = rng.randint(2, 4)
    segment_states = rng.choices(states, weights=weights, k=n_segments)
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
            pass
        elif state == "WEARER":
            mixture[s0:s0 + seg_samples] = _fit_or_loop(wearer_clip, seg_samples, rng)
            wearer_activity[s0:s0 + seg_samples] = 1.0
        elif state == "ENVIRONMENT":
            mixture[s0:s0 + seg_samples] = _fit_or_loop(other_clip, seg_samples, rng)
            environment_activity[s0:s0 + seg_samples] = 1.0

    mixture = mixture + np.random.RandomState(rng.randint(0, 2**31)).normal(0, 1e-4, size=mixture.shape).astype(np.float32)
    peak = np.max(np.abs(mixture))
    if peak > 1.0:
        mixture = mixture / peak

    enrollment = pool.enrollment_clip(wearer_id, rng)
    return mixture, enrollment, wearer_activity, environment_activity, wearer_id, other_id


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def state_code(wearer, env):
    w = (wearer > 0.5).astype(np.int64)
    e = (env > 0.5).astype(np.int64)
    return w * 1 + e * 2  # 0='00',1='10',2='01',3='11'


@torch.no_grad()
def score(model, mixture, enrollment, device):
    mix_t = torch.from_numpy(np.asarray(mixture, dtype=np.float32)).unsqueeze(0).to(device)
    mix_len = torch.tensor([len(mixture)]).to(device)
    enr_t = torch.from_numpy(np.asarray(enrollment, dtype=np.float32)).unsqueeze(0).to(device)
    enr_len = torch.tensor([len(enrollment)]).to(device)
    wearer_embedding = model.encode_enrollment(enr_t, enr_len)
    out = model.process_with_embedding(mix_t, mix_len, wearer_embedding)
    return out["wearer_logits"][0].cpu().numpy(), out["environment_logits"][0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--examples-per-tir", type=int, default=40)
    ap.add_argument("--examples-clean", type=int, default=80)
    ap.add_argument("--total-s", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--out-dir", default="evaluation/results")
    args = ap.parse_args()

    model = MentraWearNet().to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded {args.checkpoint} (step={ckpt.get('step')})")

    pool = SpeakerPool(VAL_MANIFEST)
    hard_pairs = json.loads(Path(HARD_PAIRS_VAL).read_text())
    rng = random.Random(args.seed)

    rows = []
    clip_id = 0

    # clean bucket (tir = NaN)
    for _ in range(args.examples_clean):
        mixture, enrollment, w_act, e_act, wearer_id, other_id = build_clean_example(pool, rng, args.total_s)
        wl, el = score(model, mixture, enrollment, args.device)
        n = len(wl)
        wt = align_labels_to_frames(w_act, n)
        et = align_labels_to_frames(e_act, n)
        st = state_code(wt, et)
        hn = other_id in hard_pairs.get(wearer_id, [])
        wp, ep = sigmoid(wl), sigmoid(el)
        for fi in range(n):
            rows.append((clip_id, fi, wt[fi], et[fi], st[fi], wl[fi], el[fi], wp[fi], ep[fi],
                         np.nan, wearer_id, other_id, hn, "clean", False))
        clip_id += 1

    # TIR buckets
    for tir in TIR_LEVELS:
        for _ in range(args.examples_per_tir):
            mixture, enrollment, w_act, e_act, wearer_id, other_id = build_overlap_bucket_example(pool, rng, tir, args.total_s)
            wl, el = score(model, mixture, enrollment, args.device)
            n = len(wl)
            wt = align_labels_to_frames(w_act, n)
            et = align_labels_to_frames(e_act, n)
            st = state_code(wt, et)
            hn = other_id in hard_pairs.get(wearer_id, [])
            wp, ep = sigmoid(wl), sigmoid(el)
            for fi in range(n):
                rows.append((clip_id, fi, wt[fi], et[fi], st[fi], wl[fi], el[fi], wp[fi], ep[fi],
                             float(tir), wearer_id, other_id, hn, f"tir{tir:+d}", False))
            clip_id += 1
        print(f"  tir={tir}: done ({clip_id} clips so far)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mentrawearnet_v2_{args.tag}_joint_frame_dataset.npz"
    np.savez(
        out_path,
        clip_id=np.array([r[0] for r in rows], dtype=np.int32),
        frame_index=np.array([r[1] for r in rows], dtype=np.int32),
        true_wearer=np.array([r[2] for r in rows], dtype=np.float32),
        true_environment=np.array([r[3] for r in rows], dtype=np.float32),
        true_state=np.array([r[4] for r in rows], dtype=np.int32),
        wearer_logit=np.array([r[5] for r in rows], dtype=np.float32),
        environment_logit=np.array([r[6] for r in rows], dtype=np.float32),
        wearer_probability=np.array([r[7] for r in rows], dtype=np.float32),
        environment_probability=np.array([r[8] for r in rows], dtype=np.float32),
        tir=np.array([r[9] for r in rows], dtype=np.float32),
        target_speaker=np.array([r[10] for r in rows], dtype=object),
        interferer_speaker=np.array([r[11] for r in rows], dtype=object),
        hard_negative=np.array([r[12] for r in rows], dtype=bool),
        bucket=np.array([r[13] for r in rows], dtype=object),
        is_tir_negative_trial=np.array([r[14] for r in rows], dtype=bool),
        state_names=np.array(STATE_NAMES, dtype=object),
        n_clips=np.array([clip_id]),
    )
    print(f"saved {out_path} ({len(rows)} frames across {clip_id} clips)")


if __name__ == "__main__":
    main()
