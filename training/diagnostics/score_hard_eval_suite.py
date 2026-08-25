#!/usr/bin/env python3
"""V3B Phase 8/9: score a checkpoint against the frozen hard-eval suite
(training/diagnostics/build_hard_eval_suite.py's output,
evaluation/manifests/hard_eval_suite_v1.npz).

For each trial (one enrollment + one test mixture, label 1=wearer-present/
0=wearer-absent -- see build_hard_eval_suite.py's docstring for the exact
positive/random_negative/hard_negative construction), computes a clip-level
score by mean-pooling the model's per-frame wearer probability (sigmoid of
wearer_logits) and, separately, the raw speaker-metric cosine
(out["similarity"]) over every frame of the clip (every trial's activity
label is constant across the whole clip by construction, so no frame
masking is needed here).

For each condition (clean/tir+10/.../tir-10), builds two verification
"arms":
  - random arm: positive vs random_negative trials
  - hard arm:   positive vs hard_negative trials
and reports, using ONLY the calibration split to pick an operating
threshold (never the eval split):
  - AUROC / AUPRC (threshold-free, computed on the EVAL split)
  - FAR / FRR / F1 at the calibration-selected threshold (applied to eval)
  - FRR at fixed FAR targets (20/10/5/1%), threshold chosen on calibration,
    applied to eval
  - positive vs negative score distributions (mean/std)

Also dumps raw per-trial scores to an npz so bootstrap CIs can be computed
without re-running the model.

Usage:
  .venv/bin/python3 training/diagnostics/score_hard_eval_suite.py \\
      --checkpoint training/checkpoints/mentrawearnet_v3b_hard_latest.pt \\
      --tag v3b_hard_step2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from training.models.mentrawearnet import MentraWearNet

SUITE_NPZ = "evaluation/manifests/hard_eval_suite_v1.npz"
SUITE_META = "evaluation/manifests/hard_eval_suite_v1_metadata.json"
CONDITIONS = ["clean", "tir+10", "tir+5", "tir+0", "tir-5", "tir-10"]
FAR_TARGETS = [0.20, 0.10, 0.05, 0.01]


@torch.no_grad()
def score_clip(model, mixture, enrollment, device):
    """Single-trial reference path (unbatched) -- kept for correctness
    cross-checks; score_batch() below is the one actually used for the full
    3600-trial suite (a per-trial Python loop with a GPU sync on every
    .item() call measured at >1.2s/trial -- 3600 trials would take over an
    hour per checkpoint; batching gives the same numbers in seconds)."""
    mix_t = torch.from_numpy(np.asarray(mixture, dtype=np.float32)).unsqueeze(0).to(device)
    mix_len = torch.tensor([len(mixture)]).to(device)
    enr_t = torch.from_numpy(np.asarray(enrollment, dtype=np.float32)).unsqueeze(0).to(device)
    enr_len = torch.tensor([len(enrollment)]).to(device)
    wearer_embedding = model.encode_enrollment(enr_t, enr_len)
    out = model.process_with_embedding(mix_t, mix_len, wearer_embedding)
    wp = torch.sigmoid(out["wearer_logits"][0]).mean().item()
    sim = out["similarity"][0].mean().item() if "similarity" in out else float("nan")
    return wp, sim


@torch.no_grad()
def score_batch(model, mixtures, enrollments, device):
    """Batched version of score_clip, same math, same per-clip mean-pooled
    wearer-probability / speaker-metric outputs -- just B trials through the
    model in one forward pass instead of B separate ones. Pads variable-
    length enrollments (mixtures in this suite are all fixed 2.0s, but
    padding is applied generally in case that ever changes) and masks the
    per-frame outputs to each clip's true frame_lengths before pooling, so
    padding never contaminates the mean."""
    B = len(mixtures)
    mix_len_max = max(len(m) for m in mixtures)
    enr_len_max = max(len(e) for e in enrollments)

    mix_t = torch.zeros(B, mix_len_max, dtype=torch.float32)
    mix_lengths = torch.zeros(B, dtype=torch.long)
    enr_t = torch.zeros(B, enr_len_max, dtype=torch.float32)
    enr_lengths = torch.zeros(B, dtype=torch.long)
    for i in range(B):
        m, e = mixtures[i], enrollments[i]
        mix_t[i, :len(m)] = torch.from_numpy(np.asarray(m, dtype=np.float32))
        mix_lengths[i] = len(m)
        enr_t[i, :len(e)] = torch.from_numpy(np.asarray(e, dtype=np.float32))
        enr_lengths[i] = len(e)

    mix_t, mix_lengths = mix_t.to(device), mix_lengths.to(device)
    enr_t, enr_lengths = enr_t.to(device), enr_lengths.to(device)

    wearer_embedding = model.encode_enrollment(enr_t, enr_lengths)
    out = model.process_with_embedding(mix_t, mix_lengths, wearer_embedding)
    frame_lengths = out["frame_lengths"]  # [B]
    T = out["wearer_logits"].shape[1]
    frame_mask = (torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1)).float()  # [B,T]
    denom = frame_mask.sum(dim=1).clamp(min=1)

    wp = (torch.sigmoid(out["wearer_logits"]) * frame_mask).sum(dim=1) / denom
    if "similarity" in out:
        sim = (out["similarity"] * frame_mask).sum(dim=1) / denom
    else:
        sim = torch.full((B,), float("nan"), device=device)
    return wp.cpu().numpy(), sim.cpu().numpy()


def far_frr_at_threshold(pos_scores, neg_scores, thresh):
    far = float((neg_scores >= thresh).mean()) if len(neg_scores) else float("nan")
    frr = float((pos_scores < thresh).mean()) if len(pos_scores) else float("nan")
    return far, frr


def threshold_for_far(calib_pos, calib_neg, far_target):
    """Smallest threshold achieving neg-accept-rate <= far_target on the
    CALIBRATION split (higher score = more wearer-like)."""
    if len(calib_neg) == 0:
        return float("nan")
    sorted_neg = np.sort(calib_neg)[::-1]
    idx = int(np.floor(far_target * len(sorted_neg)))
    idx = min(max(idx, 0), len(sorted_neg) - 1)
    return float(sorted_neg[idx])


def eer(pos_scores, neg_scores):
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    thresholds = np.unique(np.concatenate([pos_scores, neg_scores]))
    best_gap, best_eer = None, float("nan")
    for t in thresholds:
        far = float((neg_scores >= t).mean())
        frr = float((pos_scores < t).mean())
        gap = abs(far - frr)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_eer = (far + frr) / 2
    return float(best_eer)


def bootstrap_ci(eval_pos, eval_neg, n_boot=1000, seed=0):
    """Bootstrap 95% CI for AUROC and margin-of-means, resampling trial
    indices WITH replacement independently within pos/neg (standard
    two-sample bootstrap), on the EVAL split only. Returns None if either
    side is empty or degenerate."""
    if len(eval_pos) == 0 or len(eval_neg) == 0:
        return None
    rng = np.random.RandomState(seed)
    aurocs, margins = [], []
    for _ in range(n_boot):
        bp = eval_pos[rng.randint(0, len(eval_pos), len(eval_pos))]
        bn = eval_neg[rng.randint(0, len(eval_neg), len(eval_neg))]
        margins.append(float(bp.mean() - bn.mean()))
        labels = np.concatenate([np.ones(len(bp)), np.zeros(len(bn))])
        scores = np.concatenate([bp, bn])
        try:
            aurocs.append(float(roc_auc_score(labels, scores)))
        except ValueError:
            pass
    result = {"margin_ci95": [float(np.percentile(margins, 2.5)), float(np.percentile(margins, 97.5))]}
    if aurocs:
        result["auroc_ci95"] = [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))]
    else:
        result["auroc_ci95"] = None
    result["n_boot"] = n_boot
    return result


def arm_report(pos_all, neg_all, split_pos, split_neg, n_boot=1000):
    calib_pos = pos_all[split_pos == "calibration"]
    calib_neg = neg_all[split_neg == "calibration"]
    eval_pos = pos_all[split_pos == "eval"]
    eval_neg = neg_all[split_neg == "eval"]

    report = {
        "n_pos_calib": int(len(calib_pos)), "n_neg_calib": int(len(calib_neg)),
        "n_pos_eval": int(len(eval_pos)), "n_neg_eval": int(len(eval_neg)),
        "pos_score_mean": float(eval_pos.mean()) if len(eval_pos) else float("nan"),
        "pos_score_std": float(eval_pos.std()) if len(eval_pos) else float("nan"),
        "neg_score_mean": float(eval_neg.mean()) if len(eval_neg) else float("nan"),
        "neg_score_std": float(eval_neg.std()) if len(eval_neg) else float("nan"),
        "margin_of_means": float(eval_pos.mean() - eval_neg.mean()) if len(eval_pos) and len(eval_neg) else float("nan"),
    }
    labels = np.concatenate([np.ones(len(eval_pos)), np.zeros(len(eval_neg))])
    scores = np.concatenate([eval_pos, eval_neg])
    if len(np.unique(labels)) == 2:
        report["auroc"] = float(roc_auc_score(labels, scores))
        report["auprc"] = float(average_precision_score(labels, scores))
    else:
        report["auroc"] = float("nan")
        report["auprc"] = float("nan")
    report["eer_eval"] = eer(eval_pos, eval_neg)
    report["bootstrap_95ci"] = bootstrap_ci(eval_pos, eval_neg, n_boot=n_boot)

    report["frr_at_far"] = {}
    for far_t in FAR_TARGETS:
        thresh = threshold_for_far(calib_pos, calib_neg, far_t)
        far_realized, frr_realized = far_frr_at_threshold(eval_pos, eval_neg, thresh)
        report["frr_at_far"][f"far_{int(far_t*100)}pct"] = {
            "threshold": thresh, "far_realized_on_eval": far_realized, "frr_realized_on_eval": frr_realized,
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="evaluation/results")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--adapter-mode", default=None, choices=["static", "mixture"],
                     help="V3C-MA0: build the model with a MixtureAwareEnrollmentAdapter "
                          "(training/models/mentrawearnet.py). Omit (default) for exact "
                          "prior behavior -- no adapter, byte-identical to before this flag "
                          "existed.")
    ap.add_argument("--adapter-hidden-dim", type=int, default=32)
    args = ap.parse_args()

    model = MentraWearNet(aux_losses=True, adapter_mode=args.adapter_mode,
                           adapter_hidden_dim=args.adapter_hidden_dim).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"loaded {args.checkpoint} (step={ckpt.get('step')})  missing={missing}  unexpected={unexpected}")
    model.eval()

    data = np.load(SUITE_NPZ, allow_pickle=True)
    metadata = json.loads(Path(SUITE_META).read_text())
    n = len(metadata)
    print(f"scoring {n} trials from {SUITE_NPZ} ...")

    wearer_prob_scores = np.zeros(n, dtype=np.float32)
    metric_scores = np.zeros(n, dtype=np.float32)
    bs = args.batch_size
    for start in range(0, n, bs):
        end = min(start + bs, n)
        wp, sim = score_batch(model, data["mixtures"][start:end], data["enrollments"][start:end], args.device)
        wearer_prob_scores[start:end] = wp
        metric_scores[start:end] = sim
        if end % 500 == 0 or end == n:
            print(f"  scored {end}/{n}")

    conditions = np.array([m["condition"] for m in metadata])
    trial_types = np.array([m["trial_type"] for m in metadata])
    splits = np.array([m["split"] for m in metadata])
    labels = np.array([m["label"] for m in metadata])

    result = {"tag": args.tag, "checkpoint": args.checkpoint, "source_step": ckpt.get("step"),
              "by_condition": {}}
    overall = {"random_arm": {"pos": [], "neg": [], "split_pos": [], "split_neg": []},
               "hard_arm": {"pos": [], "neg": [], "split_pos": [], "split_neg": []}}

    for score_name, scores in (("wearer_prob", wearer_prob_scores), ("speaker_metric", metric_scores)):
        result_key = f"by_condition_{score_name}"
        result[result_key] = {}
        for cond in CONDITIONS:
            cmask = conditions == cond
            pos_mask = cmask & (trial_types == "positive")
            rand_neg_mask = cmask & (trial_types == "random_negative")
            hard_neg_mask = cmask & (trial_types == "hard_negative")

            pos_scores, pos_split = scores[pos_mask], splits[pos_mask]
            rand_neg_scores, rand_neg_split = scores[rand_neg_mask], splits[rand_neg_mask]
            hard_neg_scores, hard_neg_split = scores[hard_neg_mask], splits[hard_neg_mask]

            random_arm = arm_report(pos_scores, rand_neg_scores, pos_split, rand_neg_split)
            hard_arm = arm_report(pos_scores, hard_neg_scores, pos_split, hard_neg_split)
            result[result_key][cond] = {"random_arm": random_arm, "hard_arm": hard_arm}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}_hard_eval_suite_scores.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved {out_path}")

    raw_path = out_dir / f"{args.tag}_hard_eval_suite_raw_scores.npz"
    np.savez(raw_path, wearer_prob_scores=wearer_prob_scores, metric_scores=metric_scores,
             condition=conditions, trial_type=trial_types, split=splits, label=labels)
    print(f"saved {raw_path}")


if __name__ == "__main__":
    main()
