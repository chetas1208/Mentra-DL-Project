"""E0 heuristic baseline on the S1/S2 simulators + amplitude-shortcut report.

Workstreams J (amp shortcut kill-switch), AH (E0 vs E1 must be comparable),
AU (ablation table rows for E0).

The original E0 (`geowearnet_e0.py`) was CLIP-level on S0. E1 is FRAME-level on
S1/S2, so a clip-level E0 could not be compared to it honestly. This rebuilds
E0 at the frame level on the same scenes, same splits, same conditions, so
every E0 row in the final table is measured on the identical trials E1 sees.

E0 is still a *heuristic*: 14 interpretable physical scalars plus running means
over 10/30/68 past frames (matching E1's 680 ms receptive field, so E0 is not
handicapped on context), fed to logistic regression and to a 2-layer MLP. No
temporal model is learned.

Run:
  OMP_NUM_THREADS=2 ... python3 -m training.diagnostics.geowearnet_e0_s1 \
      --generation S1 --n-train 600 --n-eval 200
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from training.geowearnet import scenes
from training.geowearnet.data import DataConfig, GeoWearNetDataset
from training.geowearnet.features import FEATURE_GROUPS, FEATURE_NAMES

RESULTS = Path(__file__).resolve().parents[2] / "evaluation/geowearnet/results"

CONTEXT_TAPS = (10, 30, 68)  # frames of causal running mean; 68 == E1's RF

EVAL_CONDITIONS = [
    "normal", "level_matched", "random_gain", "wearer_quiet", "wearer_loud",
    "bystander_close", "bystander_close_loud", "bystander_loud",
    "shouting_bystander", "heavy_overlap", "noisy", "high_rt60", "ood_geometry",
]


def causal_running_means(x: np.ndarray, taps: Tuple[int, ...]) -> np.ndarray:
    """Causal boxcar means over the past `t` frames, for each t in taps."""
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    outs = []
    n = len(x)
    for t in taps:
        idx = np.arange(1, n + 1)
        lo = np.maximum(idx - t, 0)
        outs.append((c[idx] - c[lo]) / (idx - lo)[:, None])
    return np.concatenate(outs, axis=1)


def expand(feats: np.ndarray, cols: List[int]) -> np.ndarray:
    base = feats[:, cols]
    return np.concatenate([base, causal_running_means(base, CONTEXT_TAPS)], axis=1)


def gather(cfg: DataConfig, n_scenes: int, offset: int = 0) -> Dict[str, np.ndarray]:
    ds = GeoWearNetDataset(dataclasses.replace(cfg, normalization="none"))
    F, W, E, R = [], [], [], []
    for i in range(n_scenes):
        sc = ds.scene_for(offset + i)
        _, ph = ds.featurize(sc.audio)
        t = min(len(ph), len(sc.wearer))
        F.append(ph[:t]); W.append(sc.wearer[:t]); E.append(sc.environment[:t])
        R.append(np.full(t, sc.meta["tir_db"], dtype=np.float32))
    return {
        "feats": np.concatenate(F, 0).astype(np.float64),
        "wearer": np.concatenate(W, 0).astype(np.int64),
        "environment": np.concatenate(E, 0).astype(np.int64),
        "tir_db": np.concatenate(R, 0),
    }


def metrics(y: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "n": int(len(y)), "positive_rate": float(y.mean())}
    fpr_sorted = np.argsort(-s)
    ys = y[fpr_sorted]
    neg = (1 - ys).cumsum()
    pos = ys.cumsum()
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    far = neg / max(n_neg, 1)
    frr = 1.0 - pos / max(n_pos, 1)
    out = {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
    }
    for target in (0.20, 0.10, 0.05, 0.01):
        j = int(np.searchsorted(far, target))
        out[f"frr_at_far{int(target*100)}"] = float(frr[min(j, len(frr) - 1)])
    i = int(np.argmin(np.abs(far - frr)))
    out["eer"] = float((far[i] + frr[i]) / 2)
    return out


def amplitude_shortcut(d: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Workstream J: how much does raw level alone tell you about the label?"""
    log_rms = d["feats"][:, FEATURE_NAMES.index("log_rms")]
    out = {}
    for name in ("wearer", "environment"):
        y = d[name].astype(np.float64)
        if len(np.unique(y)) < 2:
            continue
        out[f"pearson_logrms_{name}"] = float(np.corrcoef(log_rms, y)[0, 1])
        out[f"auroc_logrms_alone_{name}"] = float(roc_auc_score(y, log_rms))
    # wearer-vs-environment discrimination on exclusive frames only (10 vs 01)
    excl = (d["wearer"] == 1) ^ (d["environment"] == 1)
    solo = excl & ((d["wearer"] + d["environment"]) == 1)
    if solo.sum() > 50 and len(np.unique(d["wearer"][solo])) == 2:
        out["auroc_logrms_alone_10_vs_01"] = float(roc_auc_score(d["wearer"][solo], log_rms[solo]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", default="S1")
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-eval", type=int, default=180)
    ap.add_argument("--duration", type=float, default=8.0)
    a = ap.parse_args()

    t_start = time.time()
    base = DataConfig(generation=a.generation, duration_s=a.duration, normalization="none",
                      condition="train_mix", use_physical=True)

    print("gathering train...", flush=True)
    tr = gather(dataclasses.replace(base, split="train", seed=101), a.n_train)
    print("gathering val...", flush=True)
    va = gather(dataclasses.replace(base, split="val", seed=202), a.n_eval)
    print(f"train frames={len(tr['wearer'])} val frames={len(va['wearer'])}", flush=True)

    results: Dict[str, object] = {
        "generation": a.generation,
        "n_train_scenes": a.n_train,
        "n_eval_scenes": a.n_eval,
        "train_frames": int(len(tr["wearer"])),
        "context_taps_frames": list(CONTEXT_TAPS),
        "note": "FRAME-LEVEL E0 on the S1/S2 simulator. SIMULATED GEOMETRY, not Mentra.",
    }

    # ---- feature-group ablation, wearer head ----
    group_models: Dict[str, object] = {}
    ablation: Dict[str, dict] = {}
    for gname, group in FEATURE_GROUPS.items():
        Xtr = expand(tr["feats"], group.indices)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=400, C=1.0)
        clf.fit(sc.transform(Xtr), tr["wearer"])
        Xva = sc.transform(expand(va["feats"], group.indices))
        ablation[gname] = {
            "n_features": Xtr.shape[1],
            "val_wearer": metrics(va["wearer"], clf.predict_proba(Xva)[:, 1]),
        }
        group_models[gname] = (sc, clf)
        print(f"  {gname:24s} val wearer AUROC {ablation[gname]['val_wearer']['auroc']:.4f}", flush=True)
    results["feature_group_ablation_val"] = ablation

    # ---- tiny MLP on all features, both heads ----
    all_idx = FEATURE_GROUPS["all"].indices
    Xtr = expand(tr["feats"], all_idx)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    heads = {}
    for head in ("wearer", "environment"):
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=120, random_state=0,
                            early_stopping=True, n_iter_no_change=8)
        mlp.fit(Xtr_s, tr[head])
        heads[head] = mlp
        print(f"  MLP {head} fitted, train iters={mlp.n_iter_}", flush=True)

    # ---- per-condition evaluation (test split, unseen speakers) ----
    per_condition: Dict[str, dict] = {}
    shortcut: Dict[str, dict] = {}
    for cond in EVAL_CONDITIONS:
        d = gather(dataclasses.replace(base, split="test", condition=cond, seed=303), a.n_eval)
        X = scaler.transform(expand(d["feats"], all_idx))
        entry = {}
        for head in ("wearer", "environment"):
            entry[head] = metrics(d[head], heads[head].predict_proba(X)[:, 1])
        # 10-vs-01: wearer-only frames vs environment-only frames
        solo = (d["wearer"] + d["environment"]) == 1
        if solo.sum() > 50 and len(np.unique(d["wearer"][solo])) == 2:
            entry["wearer_vs_env_solo"] = metrics(d["wearer"][solo], heads["wearer"].predict_proba(X[solo])[:, 1])
        entry["state_fractions"] = {
            "silence": float(((d["wearer"] == 0) & (d["environment"] == 0)).mean()),
            "wearer": float(((d["wearer"] == 1) & (d["environment"] == 0)).mean()),
            "environment": float(((d["wearer"] == 0) & (d["environment"] == 1)).mean()),
            "overlap": float(((d["wearer"] == 1) & (d["environment"] == 1)).mean()),
        }
        per_condition[cond] = entry
        shortcut[cond] = amplitude_shortcut(d)
        print(f"  [{cond:26s}] wearer AUROC {entry['wearer']['auroc']:.4f}  "
              f"logRMS-alone {shortcut[cond].get('auroc_logrms_alone_wearer', float('nan')):.4f}", flush=True)

    results["e0_mlp_test_by_condition"] = per_condition
    results["amplitude_shortcut_report"] = shortcut

    # ---- Workstream J verdict ----
    normal_auroc = per_condition["normal"]["wearer"]["auroc"]
    lm_auroc = per_condition["level_matched"]["wearer"]["auroc"]
    rg_auroc = per_condition["random_gain"]["wearer"]["auroc"]
    rms_alone_normal = shortcut["normal"].get("auroc_logrms_alone_wearer", float("nan"))
    rms_alone_lm = shortcut["level_matched"].get("auroc_logrms_alone_wearer", float("nan"))
    results["workstream_j_summary"] = {
        "normal_auroc": normal_auroc,
        "level_matched_auroc": lm_auroc,
        "random_gain_auroc": rg_auroc,
        "level_match_delta": lm_auroc - normal_auroc,
        "gain_random_delta": rg_auroc - normal_auroc,
        "logrms_alone_auroc_normal": rms_alone_normal,
        "logrms_alone_auroc_level_matched": rms_alone_lm,
        "verdict": (
            "AMPLITUDE_SHORTCUT_DOMINANT" if rms_alone_normal > 0.90 and lm_auroc < 0.70
            else "AMPLITUDE_INFORMATIVE_BUT_NOT_SUFFICIENT" if rms_alone_normal > 0.70
            else "AMPLITUDE_WEAK"
        ),
    }
    results["runtime_seconds"] = time.time() - t_start

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"geowearnet_e0_{a.generation.lower()}_frame_results.json"
    out.write_text(json.dumps(results, indent=2))
    print("\nWORKSTREAM J:", json.dumps(results["workstream_j_summary"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
