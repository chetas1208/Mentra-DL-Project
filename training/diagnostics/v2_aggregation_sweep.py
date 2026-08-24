#!/usr/bin/env python3
"""V2 evaluation autopsy Phase 12: clip-level aggregation-function sweep.

Uses the clip-level aggregation table produced by v2_score_checkpoint.py
(evaluation/results/mentrawearnet_v2_<tag>_clip_scores.npz, built from
validation_suite_v2.npz + validation_suite_v2_tir_negatives.npz -- VALIDATION
data only, never day1_public). Compares the 15 predefined candidate
aggregation functions (see clip_aggregations() in v2_score_checkpoint.py for
exact definitions) by validation clip-level AUROC (positive = enrolled
wearer actually present in the clip) and EER (clean + per-TIR, matching
phase2_clip_eer's pos/neg pairing).

No new aggregation methods are invented here -- exactly the 15 from the
task spec, selected purely by validation performance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path("evaluation/results")

METHODS = [
    "mean_wearer_prob", "mean_wearer_logit", "median_wearer_logit", "max_wearer_prob",
    "top10pct_mean_prob", "top25pct_mean_prob", "p75_wearer_prob", "p90_wearer_prob",
    "frac_above_calibrated_threshold", "longest_run_frac", "logsumexp_wearer_logit",
    "speech_gated_mean_wearer_logit", "mean_dominance_logit", "top25pct_dominance_logit",
    "frac_speech_active_dominant",
]


def compute_eer(pos, neg):
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    allscores = np.unique(np.concatenate([pos, neg]))
    diffs = [(abs((neg >= t).mean() - (pos < t).mean()), (neg >= t).mean(), (pos < t).mean()) for t in allscores]
    diffs.sort(key=lambda x: x[0])
    return float((diffs[0][1] + diffs[0][2]) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    cs = np.load(RESULTS_DIR / f"mentrawearnet_v2_{args.tag}_clip_scores.npz", allow_pickle=True)
    slice_ = cs["slice"]
    target = cs["clip_is_target"].astype(bool)

    def eer_for(score, pos_mask, neg_mask):
        pos, neg = score[pos_mask], score[neg_mask]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        return compute_eer(pos, neg) * 100

    results = {}
    for method in METHODS:
        score = cs[method].astype(np.float64)
        overall_auroc = float(roc_auc_score(target, score)) if len(np.unique(target)) > 1 else float("nan")

        clean_pos = np.array([s.startswith("clean_solo_wearer") for s in slice_]) & target
        clean_neg = np.array([s.startswith("clean_solo_environment") for s in slice_]) & ~target
        clean_eer = eer_for(score, clean_pos, clean_neg)

        tir_eers = {}
        for tir in (5, 0, -5, -10):
            pos = np.array([s in (f"overlap_random_tir{tir:+d}", f"overlap_hardpair_tir{tir:+d}") for s in slice_]) & target
            neg = np.array([s == f"tir_negative_tir{tir:+d}" for s in slice_]) & ~target
            tir_eers[f"tir{tir:+d}_eer_pct"] = eer_for(score, pos, neg)

        overall_pos = target
        overall_neg = ~target
        overall_eer = eer_for(score, overall_pos, overall_neg)

        results[method] = {
            "overall_auroc": overall_auroc, "overall_eer_pct": overall_eer,
            "clean_eer_pct": clean_eer, **tir_eers,
        }

    ranked = sorted(results.items(), key=lambda kv: (kv[1]["overall_eer_pct"] if kv[1]["overall_eer_pct"] == kv[1]["overall_eer_pct"] else 1e9))
    best_method = ranked[0][0]

    out = {"tag": args.tag, "results": results, "ranked_by_overall_eer": [r[0] for r in ranked], "best_method": best_method}
    out_path = RESULTS_DIR / f"mentrawearnet_v2_{args.tag}_aggregation_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved {out_path}")

    print(f"\n{'method':35s} {'AUROC':>8s} {'ovlEER':>8s} {'cleanEER':>9s} {'+5EER':>7s} {'0EER':>7s} {'-5EER':>7s} {'-10EER':>7s}")
    for method, r in results.items():
        print(f"{method:35s} {r['overall_auroc']:8.4f} {r['overall_eer_pct']:8.2f} {r['clean_eer_pct']:9.2f} "
              f"{r['tir+5_eer_pct']:7.2f} {r['tir+0_eer_pct']:7.2f} {r['tir-5_eer_pct']:7.2f} {r['tir-10_eer_pct']:7.2f}")
    print(f"\nbest by overall EER: {best_method}")


if __name__ == "__main__":
    main()
