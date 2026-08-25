#!/usr/bin/env python3
"""V3A evaluation: analysis driver, extending v2_analyze.py's phases 2/4/5/6/8/9
with the two diagnostics V3A actually exists to answer:

  - phase5_metric_10v01: AUROC of the RAW speaker-metric branch
    (out["similarity"], the frame_speaker/enrollment cosine fusion) for
    10-vs-01 classification -- the direct, primary success criterion. R0
    already measured this at ~0.45-0.51 (chance) for V2/M0; this is the
    "after" number.
  - phase10_solo_target: precision/recall/F1 of the four-state head's class-1
    (wearer-only == "solo-target", see Step-1 finding in the run's own
    report) prediction, both via argmax and via a 0.5-probability threshold.

Reads mentrawearnet_v3a_<tag>_joint_frame_dataset.npz (from
v3a_frame_dataset.py). Saves evaluation/results/mentrawearnet_v3a_<tag>_analysis.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from training.diagnostics.v2_analyze import (
    safe_auroc, safe_auprc, f1_far_frr, frr_at_far_targets, describe,
    phase4_state_conditional, phase5_10v01, phase6_common_activity,
    phase8_prediction_bias, phase9_far_operating_points, phase2_frame_metrics,
)

RESULTS_DIR = Path("evaluation/results")
STATE_NAMES = ["00", "10", "01", "11"]


def load_frame_dataset(tag):
    d = np.load(RESULTS_DIR / f"mentrawearnet_v3a_{tag}_joint_frame_dataset.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def phase5_metric_10v01(fd, bucket=None):
    """The actual point of V3A: does the raw cosine-similarity metric branch
    (not wearer_logit, not dominance -- the SAME `similarity` tensor
    --speaker-disc-loss trains) separate state-10 from state-01 frames?"""
    state = fd["true_state"]
    mask = (state == 1) | (state == 2)
    if bucket is not None:
        mask = mask & (fd["bucket"] == bucket)
    if mask.sum() == 0:
        return {"n": 0}
    label = (state[mask] == 1).astype(int)
    sim = fd["similarity"][mask]
    pos_sim = sim[label == 1]
    neg_sim = sim[label == 0]
    return {
        "n": int(mask.sum()),
        "n_wearer_only": int(label.sum()), "n_env_only": int((1 - label).sum()),
        "metric_auroc": safe_auroc(label, sim), "metric_auprc": safe_auprc(label, sim),
        "pos_10_cosine": describe(pos_sim),
        "neg_01_cosine": describe(neg_sim),
        "realized_margin_of_means": float(pos_sim.mean() - neg_sim.mean()) if len(pos_sim) and len(neg_sim) else float("nan"),
    }


def phase10_solo_target(fd, bucket=None):
    """Solo-target (four-state class 1 = wearer-only) precision/recall/F1 --
    this head IS the "solo-target decoder" per the run's Step-1 finding, so
    its own quality doubles as that diagnostic; no separate decoder exists."""
    state = fd["true_state"]
    mask = np.ones(len(state), dtype=bool) if bucket is None else (fd["bucket"] == bucket)
    if mask.sum() == 0 or (fd["four_state_argmax"] == -1).all():
        return {"n": 0, "note": "four_state head not present in this checkpoint"}
    true_solo = (state[mask] == 1).astype(int)
    pred_argmax_solo = (fd["four_state_argmax"][mask] == 1).astype(int)
    pred_thresh_solo = (fd["four_state_solo_prob"][mask] > 0.5).astype(int)
    stats_argmax = f1_far_frr(true_solo, pred_argmax_solo)
    stats_thresh = f1_far_frr(true_solo, pred_thresh_solo)
    return {
        "n": int(mask.sum()), "n_true_solo": int(true_solo.sum()),
        "solo_prob_auroc": safe_auroc(true_solo, fd["four_state_solo_prob"][mask]),
        "solo_prob_auprc": safe_auprc(true_solo, fd["four_state_solo_prob"][mask]),
        "argmax": {"precision": stats_argmax["tp"] / (stats_argmax["tp"] + stats_argmax["fp"]) if (stats_argmax["tp"] + stats_argmax["fp"]) > 0 else float("nan"),
                   "recall": stats_argmax["tp"] / (stats_argmax["tp"] + stats_argmax["fn"]) if (stats_argmax["tp"] + stats_argmax["fn"]) > 0 else float("nan"),
                   "f1": stats_argmax["f1"], "far": stats_argmax["far"], "frr": stats_argmax["frr"]},
        "thresh_0.5": {"precision": stats_thresh["tp"] / (stats_thresh["tp"] + stats_thresh["fp"]) if (stats_thresh["tp"] + stats_thresh["fp"]) > 0 else float("nan"),
                       "recall": stats_thresh["tp"] / (stats_thresh["tp"] + stats_thresh["fn"]) if (stats_thresh["tp"] + stats_thresh["fn"]) > 0 else float("nan"),
                       "f1": stats_thresh["f1"], "far": stats_thresh["far"], "frr": stats_thresh["frr"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    fd = load_frame_dataset(args.tag)
    buckets = ["clean", "tir+5", "tir+0", "tir-5", "tir-10"]
    result = {
        "tag": args.tag,
        "phase2_frame_metrics_overall": phase2_frame_metrics(fd),
        "phase4_state_conditional": phase4_state_conditional(fd),
        "phase5_10v01_overall": phase5_10v01(fd),
        "phase5_10v01_by_bucket": {b: phase5_10v01(fd, b) for b in buckets},
        "phase5_metric_10v01_overall": phase5_metric_10v01(fd),
        "phase5_metric_10v01_by_bucket": {b: phase5_metric_10v01(fd, b) for b in buckets},
        "phase6_common_activity": phase6_common_activity(fd),
        "phase8_prediction_bias_by_bucket": {b: phase8_prediction_bias(fd, b) for b in buckets},
        "phase9_far_operating_points_by_bucket": {b: phase9_far_operating_points(fd, b) for b in buckets},
        "phase10_solo_target_overall": phase10_solo_target(fd),
        "phase10_solo_target_by_bucket": {b: phase10_solo_target(fd, b) for b in buckets},
    }

    out_path = RESULTS_DIR / f"mentrawearnet_v3a_{args.tag}_analysis.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
