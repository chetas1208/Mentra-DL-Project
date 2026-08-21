#!/usr/bin/env python3
"""Compute classification + verification metrics from an evaluate.py output CSV.

Sprint spec section 19. Positive class = WEARER (False Accept = environment
misclassified as wearer; this is the metric to minimize hardest for this
product).
"""
import argparse
import csv
from pathlib import Path


def compute(csv_path: Path) -> dict:
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise ValueError("empty results csv")

    tp = fp = tn = fn = 0
    latencies = []
    for r in rows:
        gt, pred = r["ground_truth"], r["prediction"]
        if r.get("latency_ms"):
            latencies.append(float(r["latency_ms"]))
        if gt == "WEARER" and pred == "WEARER":
            tp += 1
        elif gt == "ENVIRONMENT" and pred == "WEARER":
            fp += 1  # false accept
        elif gt == "ENVIRONMENT" and pred == "ENVIRONMENT":
            tn += 1
        elif gt == "WEARER" and pred == "ENVIRONMENT":
            fn += 1  # false reject

    n_wearer = tp + fn
    n_env = tn + fp
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")  # = 1 - FRR
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else float("nan"))
    far = fp / n_env if n_env else float("nan")
    frr = fn / n_wearer if n_wearer else float("nan")
    balanced_acc = ((recall if n_wearer else 0) +
                     (tn / n_env if n_env else 0)) / 2

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else None
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else None

    return {
        "n": len(rows),
        "n_wearer": n_wearer,
        "n_environment": n_env,
        "precision_wearer": precision,
        "recall_wearer": recall,
        "f1_wearer": f1,
        "FAR": far,
        "FRR": frr,
        "balanced_accuracy": balanced_acc,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    args = parser.parse_args()
    for k, v in compute(args.results_csv).items():
        print(f"{k}: {v}")
