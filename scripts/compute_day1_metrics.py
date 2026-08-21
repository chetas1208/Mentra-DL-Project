#!/usr/bin/env python3
"""Real metrics (accuracy/FAR/FRR/EER/AUC/threshold) from a day1 rotation CSV.

No sklearn dependency — small enough to do the ROC/EER sweep by hand.
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def compute(csv_path: Path) -> dict:
    rows = list(csv.DictReader(open(csv_path)))
    y = np.array([1 if r["ground_truth"] == "WEARER" else 0 for r in rows])
    s = np.array([float(r["score"]) for r in rows])

    thresholds = np.unique(s)
    best_bal_acc, best_t = -1, None
    far_frr_curve = []
    for t in thresholds:
        pred = s >= t
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        tn = np.sum((pred == 0) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        n_pos, n_neg = y.sum(), (1 - y).sum()
        tpr = tp / n_pos if n_pos else 0.0
        far = fp / n_neg if n_neg else 0.0  # false accept rate
        frr = fn / n_pos if n_pos else 0.0  # false reject rate
        bal_acc = (tpr + tn / n_neg) / 2 if n_neg else tpr
        far_frr_curve.append((t, far, frr))
        if bal_acc > best_bal_acc:
            best_bal_acc, best_t = bal_acc, t

    # EER: threshold where FAR ~= FRR
    far_frr_curve.sort(key=lambda x: x[0])
    eer, eer_t = 1.0, None
    for t, far, frr in far_frr_curve:
        if abs(far - frr) < abs(eer - eer_t[1] if eer_t else 999):
            pass
    diffs = [(abs(far - frr), t, far, frr) for t, far, frr in far_frr_curve]
    diffs.sort(key=lambda x: x[0])
    _, eer_t, eer_far, eer_frr = diffs[0]
    eer = (eer_far + eer_frr) / 2

    # ROC-AUC via rank statistic (Mann-Whitney U)
    pos_scores = s[y == 1]
    neg_scores = s[y == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    auc = (np.sum(pos_scores[:, None] > neg_scores[None, :]) +
           0.5 * np.sum(pos_scores[:, None] == neg_scores[None, :])) / (n_pos * n_neg)

    pred_best = s >= best_t
    tp = np.sum((pred_best == 1) & (y == 1))
    fp = np.sum((pred_best == 1) & (y == 0))
    tn = np.sum((pred_best == 0) & (y == 0))
    fn = np.sum((pred_best == 0) & (y == 1))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    return {
        "n": len(rows), "n_wearer": int(n_pos), "n_environment": int(n_neg),
        "roc_auc": auc,
        "eer": eer, "eer_threshold": eer_t,
        "best_balanced_acc_threshold": best_t,
        "best_balanced_accuracy": best_bal_acc,
        "wearer_precision_at_best_t": precision,
        "wearer_recall_at_best_t": recall,
        "wearer_f1_at_best_t": f1,
        "note": "threshold selected on same data used for reporting — NOT calibration/test-split. See sprint spec section 9.",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", type=Path)
    args = ap.parse_args()
    for k, v in compute(args.csv_path).items():
        print(f"{k}: {v}")
