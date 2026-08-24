#!/usr/bin/env python3
"""V2 diagnostic 2a: richer frame-level evaluation metrics for a trained
MentraWearNet checkpoint, sibling to scripts/model/evaluate_mentrawearnet.py
(which reports CLIP-level rotation EER only, matching the original
SpeakerNet baseline protocol). This script adds, per TIR level and overall:

  - AUROC, AUPRC (wearer head, environment head)
  - F1 (wearer, environment, and "overlap" -- overlap meaning: a frame
    where BOTH heads simultaneously predict active, scored against frames
    where BOTH true labels are active)
  - FAR (false accept rate = FP / (FP + TN), i.e. false positive rate)
  - FRR (false reject rate = FN / (FN + TP), i.e. false negative /miss rate)

Operates at FRAME resolution (via the same align_labels_to_frames() used
by training/train.py), not clip resolution -- these are complementary to
the existing clip-level rotation-EER script, not a replacement for it.

Uses the HELD-OUT validation speaker manifest
(evaluation/manifests/librispeech_train_clean_100_val.json) by default --
zero speaker overlap with the training manifest (verified: 211 train
speakers / 20 val speakers, 0 in common).

Read-only w.r.t. any checkpoint: loading a .pt file for eval does not
touch the training process. Safe to run standalone, CPU-only.

Usage:
    .venv/bin/python3 scripts/model/evaluate_mentrawearnet_metrics.py \\
        --checkpoint training/checkpoints/mentrawearnet_v1.pt \\
        --device cpu
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from training.data.mixture_generator import (
    SAMPLE_RATE, SAMPLES_PER_FRAME, NoisePool, SpeakerPool,
    _add_noise, _fit_or_loop, _rms,
)
from training.models.mentrawearnet import MentraWearNet
from training.train import align_labels_to_frames

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

TIR_LEVELS = [10, 5, 0, -5, -10]


def build_frame_example(pool: SpeakerPool, rng: random.Random, tir_db: float,
                         total_s: float, noise_pool: NoisePool):
    """Same structured SILENCE/WEARER/ENVIRONMENT/OVERLAP segment generator
    as training/data/mixture_generator.py's generate_example(), except the
    OVERLAP segment's TIR is PINNED to tir_db rather than sampled from
    sample_tir_db() -- lets this script bucket frame-level metrics by a
    controlled TIR level while still exercising realistic segment
    transitions and temporal variety, matching train.py's own segment
    construction logic (kept in sync deliberately -- both use the same
    private helpers from mixture_generator.py)."""
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

    if noise_pool is not None and noise_pool.available:
        mixture = _add_noise(mixture, noise_pool, rng)
    else:
        mixture = mixture + np.random.RandomState(rng.randint(0, 2**31)).normal(0, 1e-4, size=mixture.shape).astype(np.float32)

    peak = np.max(np.abs(mixture))
    if peak > 1.0:
        mixture = mixture / peak

    enrollment = pool.enrollment_clip(wearer_id, rng)
    return mixture, enrollment, wearer_activity, environment_activity


def safe_auroc(y_true, y_score):
    if not HAVE_SKLEARN:
        return float("nan")
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def safe_auprc(y_true, y_score):
    if not HAVE_SKLEARN:
        return float("nan")
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return average_precision_score(y_true, y_score)


def f1_far_frr(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Hand-rollable without sklearn -- plain numpy confusion-matrix counts."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision == precision and recall == recall and (precision + recall) > 0) else float("nan"))
    far = fp / (fp + tn) if (fp + tn) > 0 else float("nan")  # false accept rate = FPR
    frr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")  # false reject rate = miss rate
    return {"f1": f1, "far": far, "frr": frr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


@torch.no_grad()
def score_example(model: MentraWearNet, mixture: np.ndarray, enrollment: np.ndarray, device: str):
    mix_t = torch.from_numpy(mixture).unsqueeze(0).float().to(device)
    mix_len = torch.tensor([len(mixture)]).to(device)
    enr_t = torch.from_numpy(enrollment).unsqueeze(0).float().to(device)
    enr_len = torch.tensor([len(enrollment)]).to(device)
    wearer_embedding = model.encode_enrollment(enr_t, enr_len)
    out = model.process_with_embedding(mix_t, mix_len, wearer_embedding)
    target_frames = out["wearer_logits"].shape[1]
    wearer_prob = torch.sigmoid(out["wearer_logits"])[0].cpu().numpy()
    env_prob = torch.sigmoid(out["environment_logits"])[0].cpu().numpy()
    return wearer_prob, env_prob, target_frames


def evaluate_bucket(model, pool, noise_pool, rng, tir_db, n_examples, total_s, device):
    wearer_probs, wearer_targets = [], []
    env_probs, env_targets = [], []
    for _ in range(n_examples):
        mixture, enrollment, w_act, e_act = build_frame_example(pool, rng, tir_db, total_s, noise_pool)
        wearer_prob, env_prob, target_frames = score_example(model, mixture, enrollment, device)
        w_tgt = align_labels_to_frames(w_act, target_frames)
        e_tgt = align_labels_to_frames(e_act, target_frames)
        wearer_probs.append(wearer_prob)
        wearer_targets.append(w_tgt)
        env_probs.append(env_prob)
        env_targets.append(e_tgt)
    wp = np.concatenate(wearer_probs)
    wt = np.concatenate(wearer_targets)
    ep = np.concatenate(env_probs)
    et = np.concatenate(env_targets)
    return wp, wt, ep, et


def report_bucket(label: str, wp, wt, ep, et, threshold: float = 0.5):
    wearer_pred = (wp > threshold).astype(np.float32)
    env_pred = (ep > threshold).astype(np.float32)
    overlap_true = ((wt > 0.5) & (et > 0.5)).astype(np.float32)
    overlap_pred = ((wearer_pred > 0.5) & (env_pred > 0.5)).astype(np.float32)

    w_auroc, w_auprc = safe_auroc(wt, wp), safe_auprc(wt, wp)
    e_auroc, e_auprc = safe_auroc(et, ep), safe_auprc(et, ep)
    w_stats = f1_far_frr(wt, wearer_pred)
    e_stats = f1_far_frr(et, env_pred)
    o_stats = f1_far_frr(overlap_true, overlap_pred)

    print(f"\n--- {label} (n_frames={len(wt)}) ---")
    print(f"  wearer: AUROC={w_auroc:.4f}  AUPRC={w_auprc:.4f}  F1={w_stats['f1']:.4f}  "
          f"FAR={w_stats['far']:.4f}  FRR={w_stats['frr']:.4f}")
    print(f"  env:    AUROC={e_auroc:.4f}  AUPRC={e_auprc:.4f}  F1={e_stats['f1']:.4f}  "
          f"FAR={e_stats['far']:.4f}  FRR={e_stats['frr']:.4f}")
    print(f"  overlap (both heads jointly active): F1={o_stats['f1']:.4f}  "
          f"FAR={o_stats['far']:.4f}  FRR={o_stats['frr']:.4f}  "
          f"(true_overlap_frac={overlap_true.mean():.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="training/checkpoints/mentrawearnet_v1.pt")
    ap.add_argument("--manifest", default="evaluation/manifests/librispeech_train_clean_100_val.json",
                     help="held-out validation speaker manifest by default -- NOT the training manifest")
    ap.add_argument("--noise-dir", default="evaluation/data/raw/musan/noise")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--examples-per-tir", type=int, default=40)
    ap.add_argument("--total-s", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    if not HAVE_SKLEARN:
        print("WARNING: sklearn not importable -- AUROC/AUPRC will report NaN "
              "(F1/FAR/FRR are hand-rolled with numpy and unaffected).")

    model = MentraWearNet().to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint from {args.checkpoint} (step={ckpt.get('step', '?')})")

    pool = SpeakerPool(args.manifest)
    noise_pool = NoisePool(args.noise_dir)
    print(f"manifest: {args.manifest}  ({len(pool.speaker_ids)} speakers)")
    print(f"noise pool: {len(noise_pool.clips)} clips" if noise_pool.available else "noise pool: none")

    rng = random.Random(args.seed)

    all_wp, all_wt, all_ep, all_et = [], [], [], []
    for tir in TIR_LEVELS:
        wp, wt, ep, et = evaluate_bucket(model, pool, noise_pool, rng, tir,
                                          args.examples_per_tir, args.total_s, args.device)
        report_bucket(f"TIR {tir:+d}dB", wp, wt, ep, et, args.threshold)
        all_wp.append(wp); all_wt.append(wt); all_ep.append(ep); all_et.append(et)

    report_bucket("OVERALL (all TIR levels pooled)",
                   np.concatenate(all_wp), np.concatenate(all_wt),
                   np.concatenate(all_ep), np.concatenate(all_et), args.threshold)


if __name__ == "__main__":
    main()
