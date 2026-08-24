#!/usr/bin/env python3
"""V2 evaluation autopsy: analysis driver for phases 2, 4, 5, 6, 8, 9.

Reads the two artifacts produced by v2_score_checkpoint.py (clip-level
verification/aggregation table, from validation_suite_v2.npz +
validation_suite_v2_tir_negatives.npz) and v2_frame_dataset.py (the
canonical joint-logit frame dataset, from the structured SILENCE/WEARER/
ENVIRONMENT/OVERLAP generator, which is the only one of the two that
contains genuine state=00 silence frames and genuine per-TIR 10-only /
01-only frames -- validation_suite_v2.npz's overlap slices are 100%
state=11 for their full duration by construction).

Prints one big JSON blob per checkpoint tag to stdout AND saves it to
evaluation/results/mentrawearnet_v2_<tag>_analysis.json.

Usage:
    .venv/bin/python3 training/diagnostics/v2_analyze.py --tag step5000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

RESULTS_DIR = Path("evaluation/results")


def safe_auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_auprc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def compute_eer(pos, neg):
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    allscores = np.unique(np.concatenate([pos, neg]))
    diffs = [(abs((neg >= t).mean() - (pos < t).mean()), (neg >= t).mean(), (pos < t).mean()) for t in allscores]
    diffs.sort(key=lambda x: x[0])
    return float((diffs[0][1] + diffs[0][2]) / 2)


def f1_far_frr(y_true, y_pred):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred)); fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred)); tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision == precision and recall == recall and (precision + recall) > 0) else float("nan"))
    far = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    frr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    return {"f1": f1, "far": far, "frr": frr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def frr_at_far_targets(y_true, y_score, far_targets=(0.20, 0.10, 0.05, 0.01)):
    """FRR (miss rate) at given FAR targets, from the ROC curve. Threshold
    chosen ON THIS DATA (validation-only use throughout this script -- never
    the final day1_public test protocol)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return {f"far_{int(f*100)}pct": float("nan") for f in far_targets}
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    frr_curve = 1 - tpr
    out = {}
    for f in far_targets:
        idx = np.searchsorted(fpr, f, side="left")
        idx = min(idx, len(fpr) - 1)
        out[f"far_{int(f*100)}pct"] = float(frr_curve[idx])
    return out


def describe(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {k: float("nan") for k in ("mean", "std", "median", "p10", "p25", "p75", "p90", "n")}
    return {
        "mean": float(np.mean(x)), "std": float(np.std(x)), "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)), "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)), "p90": float(np.percentile(x, 90)), "n": int(len(x)),
    }


def load_frame_dataset(tag):
    d = np.load(RESULTS_DIR / f"mentrawearnet_v2_{tag}_joint_frame_dataset.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_clip_scores(tag):
    d = np.load(RESULTS_DIR / f"mentrawearnet_v2_{tag}_clip_scores.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


STATE_NAMES = ["00", "10", "01", "11"]


def phase4_state_conditional(fd):
    state = fd["true_state"]
    out = {}
    for code, name in enumerate(STATE_NAMES):
        mask = state == code
        out[name] = {
            "wearer_logit": describe(fd["wearer_logit"][mask]),
            "environment_logit": describe(fd["environment_logit"][mask]),
            "wearer_probability": describe(fd["wearer_probability"][mask]),
            "environment_probability": describe(fd["environment_probability"][mask]),
        }
    return out


def phase5_10v01(fd, bucket=None):
    state = fd["true_state"]
    mask = (state == 1) | (state == 2)
    if bucket is not None:
        mask = mask & (fd["bucket"] == bucket)
    if mask.sum() == 0:
        return {"n": 0}
    label = (state[mask] == 1).astype(int)  # 1 = wearer-only(10), 0 = environment-only(01)
    wl, el = fd["wearer_logit"][mask], fd["environment_logit"][mask]
    wp, ep = fd["wearer_probability"][mask], fd["environment_probability"][mask]
    dom_logit = wl - el
    dom_prob = wp - ep
    return {
        "n": int(mask.sum()),
        "n_wearer_only": int(label.sum()), "n_env_only": int((1 - label).sum()),
        "wearer_logit_auroc": safe_auroc(label, wl), "wearer_logit_auprc": safe_auprc(label, wl),
        "wearer_prob_auroc": safe_auroc(label, wp), "wearer_prob_auprc": safe_auprc(label, wp),
        "dominance_logit_auroc": safe_auroc(label, dom_logit), "dominance_logit_auprc": safe_auprc(label, dom_logit),
        "dominance_prob_auroc": safe_auroc(label, dom_prob), "dominance_prob_auprc": safe_auprc(label, dom_prob),
    }


def phase6_common_activity(fd):
    state = fd["true_state"]
    is_speech = (state != 0).astype(int)
    common = (fd["wearer_logit"] + fd["environment_logit"]) / 2.0
    return {
        "common_activity_logit_auroc_speech_vs_silence": safe_auroc(is_speech, common),
        "common_activity_logit_auprc_speech_vs_silence": safe_auprc(is_speech, common),
        "n_speech": int(is_speech.sum()), "n_silence": int((1 - is_speech).sum()),
    }


def phase8_prediction_bias(fd, bucket):
    mask = fd["bucket"] == bucket
    if mask.sum() == 0:
        return {"n": 0}
    wt, wp = fd["true_wearer"][mask], fd["wearer_probability"][mask]
    et, ep = fd["true_environment"][mask], fd["environment_probability"][mask]
    return {
        "n": int(mask.sum()),
        "true_wearer_positive_rate": float(wt.mean()), "pred_wearer_positive_rate_at_0.5": float((wp > 0.5).mean()),
        "mean_wearer_prob": float(wp.mean()),
        "true_env_positive_rate": float(et.mean()), "pred_env_positive_rate_at_0.5": float((ep > 0.5).mean()),
        "mean_env_prob": float(ep.mean()),
    }


def phase9_far_operating_points(fd, bucket):
    mask = fd["bucket"] == bucket
    if mask.sum() == 0:
        return {"n": 0}
    wt, wl = fd["true_wearer"][mask], fd["wearer_logit"][mask]
    return {"n": int(mask.sum()), **frr_at_far_targets(wt, wl)}


def phase2_frame_metrics(fd):
    wt, wl, wp = fd["true_wearer"], fd["wearer_logit"], fd["wearer_probability"]
    et, el, ep = fd["true_environment"], fd["environment_logit"], fd["environment_probability"]
    state = fd["true_state"]
    wearer_pred = (wp > 0.5).astype(np.float32)
    env_pred = (ep > 0.5).astype(np.float32)
    overlap_true = (state == 3).astype(np.float32)
    overlap_pred = ((wearer_pred > 0.5) & (env_pred > 0.5)).astype(np.float32)
    w_stats = f1_far_frr(wt, wearer_pred)
    e_stats = f1_far_frr(et, env_pred)
    o_stats = f1_far_frr(overlap_true, overlap_pred)
    return {
        "wearer_auroc": safe_auroc(wt, wp), "wearer_auprc": safe_auprc(wt, wp),
        "wearer_f1": w_stats["f1"], "wearer_far": w_stats["far"], "wearer_frr": w_stats["frr"],
        "environment_auroc": safe_auroc(et, ep), "environment_auprc": safe_auprc(et, ep),
        "environment_f1": e_stats["f1"], "environment_far": e_stats["far"], "environment_frr": e_stats["frr"],
        "overlap_f1": o_stats["f1"], "overlap_far": o_stats["far"], "overlap_frr": o_stats["frr"],
        "wearer_frr_at_far": frr_at_far_targets(wt, wl),
        "n_frames": int(len(wt)),
    }


def phase2_clip_eer(cs):
    """Clip-level verification EER using validation_suite_v2 + tir_negatives
    clip-score table (mean_wearer_prob aggregation -- matches the historical
    scripts/model/evaluate_mentrawearnet.py protocol's scoring function).
    NOT the final test protocol (that uses day1_public, run once in Phase 13)
    -- this is validation-only, used for checkpoint/threshold selection."""
    slice_ = cs["slice"]; target = cs["clip_is_target"].astype(bool); score = cs["mean_wearer_prob"]

    def eer_for(pos_mask, neg_mask):
        pos = score[pos_mask]; neg = score[neg_mask]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        return compute_eer(pos, neg) * 100

    clean_pos = np.array([s.startswith("clean_solo_wearer") for s in slice_]) & target
    clean_neg = np.array([s.startswith("clean_solo_environment") for s in slice_]) & ~target
    out = {"clean_eer_pct": eer_for(clean_pos, clean_neg)}
    for tir in (5, 0, -5, -10):
        pos = np.array([s in (f"overlap_random_tir{tir:+d}", f"overlap_hardpair_tir{tir:+d}") for s in slice_]) & target
        neg = np.array([s == f"tir_negative_tir{tir:+d}" for s in slice_]) & ~target
        out[f"tir{tir:+d}_eer_pct"] = eer_for(pos, neg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    fd = load_frame_dataset(args.tag)
    cs = load_clip_scores(args.tag)

    buckets = ["clean", "tir+5", "tir+0", "tir-5", "tir-10"]
    result = {
        "tag": args.tag,
        "phase2_frame_metrics_overall": phase2_frame_metrics(fd),
        "phase2_clip_eer": phase2_clip_eer(cs),
        "phase4_state_conditional": phase4_state_conditional(fd),
        "phase5_10v01_overall": phase5_10v01(fd),
        "phase5_10v01_by_bucket": {b: phase5_10v01(fd, b) for b in buckets},
        "phase6_common_activity": phase6_common_activity(fd),
        "phase8_prediction_bias_by_bucket": {b: phase8_prediction_bias(fd, b) for b in buckets},
        "phase9_far_operating_points_by_bucket": {b: phase9_far_operating_points(fd, b) for b in buckets},
    }

    out_path = RESULTS_DIR / f"mentrawearnet_v2_{args.tag}_analysis.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
