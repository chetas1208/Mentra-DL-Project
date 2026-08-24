#!/usr/bin/env python3
"""V2 evaluation autopsy Phase 7: joint 2D logit-space scatter plot.

wearer_logit (x) vs environment_logit (y), colored by true state
(00/10/01/11), from the canonical joint frame dataset (v2_frame_dataset.py).
Also prints numeric cluster-separation characterization (centroids, spreads,
a rough Fisher-style separation ratio) so the plot's visual read is backed
by real numbers, not eyeballing alone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("evaluation/results")
STATE_NAMES = ["00 (silence)", "10 (wearer-only)", "01 (environment-only)", "11 (overlap)"]
COLORS = ["#9e9e9e", "#1f77b4", "#d62728", "#2ca02c"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-points-per-state", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(RESULTS_DIR / f"mentrawearnet_v2_{args.tag}_joint_frame_dataset.npz", allow_pickle=True)
    wl, el, state = d["wearer_logit"], d["environment_logit"], d["true_state"]

    rng = np.random.RandomState(args.seed)
    fig, ax = plt.subplots(figsize=(8, 8))
    stats = {}
    for code in range(4):
        mask = state == code
        idx = np.where(mask)[0]
        if len(idx) > args.max_points_per_state:
            idx = rng.choice(idx, args.max_points_per_state, replace=False)
        ax.scatter(wl[idx], el[idx], s=4, alpha=0.35, color=COLORS[code], label=f"{STATE_NAMES[code]} (n={mask.sum()})")
        stats[code] = {
            "centroid_wl": float(wl[mask].mean()) if mask.any() else float("nan"),
            "centroid_el": float(el[mask].mean()) if mask.any() else float("nan"),
            "std_wl": float(wl[mask].std()) if mask.any() else float("nan"),
            "std_el": float(el[mask].std()) if mask.any() else float("nan"),
        }

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("wearer_logit")
    ax.set_ylabel("environment_logit")
    ax.set_title(f"MentraWearNet V2 ({args.tag}) joint logit space by true state\n"
                 "dashed lines = 0.5-probability decision boundary for each head")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path = RESULTS_DIR / f"mentrawearnet_v2_{args.tag}_joint_logits.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

    # Numeric separation characterization: 10 vs 01 centroid distance,
    # normalized by pooled std along the (wl - el) "dominance" axis
    # (matches Phase 5/6's dominance_logit metric).
    dom = wl - el
    dom10 = dom[state == 1]
    dom01 = dom[state == 2]
    pooled_std = np.sqrt((dom10.var() + dom01.var()) / 2) if len(dom10) and len(dom01) else float("nan")
    fisher_10v01 = abs(dom10.mean() - dom01.mean()) / pooled_std if pooled_std > 0 else float("nan")

    common = (wl + el) / 2.0
    common_speech = common[state != 0]
    common_silence = common[state == 0]
    pooled_std_c = np.sqrt((common_speech.var() + common_silence.var()) / 2) if len(common_silence) else float("nan")
    fisher_speech_silence = abs(common_speech.mean() - common_silence.mean()) / pooled_std_c if pooled_std_c > 0 else float("nan")

    print("\nper-state centroids/spreads (logit space):")
    for code in range(4):
        print(f"  {STATE_NAMES[code]}: centroid=({stats[code]['centroid_wl']:.3f}, {stats[code]['centroid_el']:.3f})  "
              f"std=({stats[code]['std_wl']:.3f}, {stats[code]['std_el']:.3f})")
    print(f"\nFisher-style separation, dominance axis (wl-el), 10 vs 01: {fisher_10v01:.3f}  "
          "(>1 = centroids separated by more than 1 pooled std; rule of thumb only)")
    print(f"Fisher-style separation, common-activity axis (wl+el)/2, speech vs silence: {fisher_speech_silence:.3f}")

    # Does the model's 0.5-threshold decision boundary (logit=0 on each axis)
    # cut cleanly through the 10/01 clusters, or is it offset (biased)?
    frac_10_wl_gt0 = float((wl[state == 1] > 0).mean())
    frac_01_wl_gt0 = float((wl[state == 2] > 0).mean())
    print(f"\nfraction of state=10 frames with wearer_logit>0 (correct side of 0.5 boundary): {frac_10_wl_gt0:.3f}")
    print(f"fraction of state=01 frames with wearer_logit>0 (WRONG side -- false wearer-positive): {frac_01_wl_gt0:.3f}")
    print("(both close to their ideal value with a clear centroid gap => 'shifted threshold' story; "
          "heavy overlap with similar fractions => 'representation itself insufficient' story)")


if __name__ == "__main__":
    main()
