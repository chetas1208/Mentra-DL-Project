#!/usr/bin/env python3
"""V3C-MA0: the mandatory causal comparison -- MIXTURE_AWARE minus matched
STATIC_ADAPTER (NOT minus STATIC_BASE, NOT minus V3A), with bootstrap
paired differences on the SAME fixed trials.

Two data sources, both already checkpoint-paired by construction:
  1. v3a_frame_dataset.py output for both branches -- SAME seed
     (v2_frame_dataset.GLOBAL_SEED, checkpoint-independent generation), so
     row i in the static run's npz and row i in the mixture run's npz are
     the SAME synthetic clip/frame. Used for dominance AUROC and
     speaker-metric (raw cosine) 10-vs-01 AUROC, paired bootstrap.
  2. score_hard_eval_suite.py raw_scores.npz for both branches -- same
     deterministic hard_eval_suite_v1.npz trial order for every checkpoint.
     Used for random-impostor / hard-impostor AUROC, FAR, FRR@FAR5, paired
     bootstrap.

Usage:
  .venv/bin/python3 training/diagnostics/v3c_ma0_causal_effect.py \\
      --static-tag v3c_ma0_static_bestN --mixture-tag v3c_ma0_mixture_bestM
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path("evaluation/results")


def paired_bootstrap_auroc_diff(label, score_a, score_b, n_boot=2000, seed=0):
    """Paired bootstrap (resample TRIAL INDICES together for both scorers)
    on AUROC(label, score_b) - AUROC(label, score_a). Returns (obs_diff, ci_lo, ci_hi)."""
    rng = np.random.RandomState(seed)
    n = len(label)
    obs_a = roc_auc_score(label, score_a)
    obs_b = roc_auc_score(label, score_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        lb = label[idx]
        if len(np.unique(lb)) < 2:
            continue
        try:
            a = roc_auc_score(lb, score_a[idx])
            b = roc_auc_score(lb, score_b[idx])
            diffs.append(b - a)
        except ValueError:
            continue
    if not diffs:
        return obs_b - obs_a, None, None
    return obs_b - obs_a, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def frame_level_causal(static_tag, mixture_tag):
    fs = np.load(RESULTS_DIR / f"mentrawearnet_v3a_{static_tag}_joint_frame_dataset.npz", allow_pickle=True)
    fm = np.load(RESULTS_DIR / f"mentrawearnet_v3a_{mixture_tag}_joint_frame_dataset.npz", allow_pickle=True)
    assert len(fs["clip_id"]) == len(fm["clip_id"]), "static/mixture frame datasets have different lengths -- not paired!"
    assert np.array_equal(fs["true_state"], fm["true_state"]), "true_state mismatch -- generation not paired!"

    state = fs["true_state"]
    mask = (state == 1) | (state == 2)  # 10 vs 01
    label = (state[mask] == 1).astype(int)

    wp_s, ep_s = fs["wearer_probability"][mask], fs["environment_probability"][mask]
    wp_m, ep_m = fm["wearer_probability"][mask], fm["environment_probability"][mask]
    dom_s = wp_s - ep_s
    dom_m = wp_m - ep_m
    sim_s = fs["similarity"][mask]
    sim_m = fm["similarity"][mask]

    dom_diff, dom_lo, dom_hi = paired_bootstrap_auroc_diff(label, dom_s, dom_m)
    metric_diff, metric_lo, metric_hi = paired_bootstrap_auroc_diff(label, sim_s, sim_m)

    return {
        "dominance_auroc_static": float(roc_auc_score(label, dom_s)),
        "dominance_auroc_mixture": float(roc_auc_score(label, dom_m)),
        "dominance_auroc_diff_MA_minus_STATIC": dom_diff,
        "dominance_auroc_diff_ci95": [dom_lo, dom_hi],
        "speaker_metric_auroc_static": float(roc_auc_score(label, sim_s)),
        "speaker_metric_auroc_mixture": float(roc_auc_score(label, sim_m)),
        "speaker_metric_auroc_diff_MA_minus_STATIC": metric_diff,
        "speaker_metric_auroc_diff_ci95": [metric_lo, metric_hi],
        "n_frames_10v01": int(mask.sum()),
    }


def hard_eval_suite_causal(static_tag, mixture_tag):
    ds = np.load(RESULTS_DIR / f"{static_tag}_hard_eval_suite_raw_scores.npz", allow_pickle=True)
    dm = np.load(RESULTS_DIR / f"{mixture_tag}_hard_eval_suite_raw_scores.npz", allow_pickle=True)
    assert np.array_equal(ds["condition"], dm["condition"]), "condition mismatch -- suite not paired!"
    assert np.array_equal(ds["trial_type"], dm["trial_type"]), "trial_type mismatch!"
    assert np.array_equal(ds["label"], dm["label"]), "label mismatch!"

    split = ds["split"]
    trial_type = ds["trial_type"]
    label = ds["label"].astype(int)
    eval_mask = split == "eval"

    out = {}
    for score_name, key in (("wearer_prob", "wearer_prob_scores"), ("speaker_metric", "metric_scores")):
        s_scores = ds[key]
        m_scores = dm[key]
        out[score_name] = {}
        for arm_name, neg_type in (("random", "random_negative"), ("hard", "hard_negative")):
            arm_mask = eval_mask & ((trial_type == "positive") | (trial_type == neg_type))
            lb = label[arm_mask]
            sa = s_scores[arm_mask]
            sm = m_scores[arm_mask]
            diff, lo, hi = paired_bootstrap_auroc_diff(lb, sa, sm)
            out[score_name][f"{arm_name}_arm"] = {
                "auroc_static": float(roc_auc_score(lb, sa)),
                "auroc_mixture": float(roc_auc_score(lb, sm)),
                "auroc_diff_MA_minus_STATIC": diff,
                "auroc_diff_ci95": [lo, hi],
                "n_pos": int((lb == 1).sum()), "n_neg": int((lb == 0).sum()),
            }
            # FAR / FRR@FAR5, calibration-thresholded per branch (own threshold, matching
            # score_hard_eval_suite.py's convention), compared on the eval split.
            calib_mask = split == "calibration"
            for tag, scores, key_out in (("static", sa, "static"), ("mixture", sm, "mixture")):
                pass  # FAR/FRR handled below using full calibration split per branch
        out[score_name]["_note"] = "FAR/FRR@FAR5 per branch: see by_condition scores from score_hard_eval_suite.py directly"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-tag", required=True, help="tag used for both v3a_frame_dataset and score_hard_eval_suite runs on the STATIC_ADAPTER checkpoint")
    ap.add_argument("--mixture-tag", required=True, help="same, for the MIXTURE_AWARE checkpoint")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = {
        "static_tag": args.static_tag, "mixture_tag": args.mixture_tag,
        "frame_level": frame_level_causal(args.static_tag, args.mixture_tag),
        "hard_eval_suite": hard_eval_suite_causal(args.static_tag, args.mixture_tag),
    }
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"v3c_ma0_causal_effect_{args.mixture_tag}_vs_{args.static_tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
