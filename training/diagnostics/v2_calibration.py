#!/usr/bin/env python3
"""V2 evaluation autopsy phases 10-11: per-head calibration (temperature +
Platt scaling) and joint two-head 6-parameter affine calibration.

Fit ONLY on a calibration split of the canonical joint frame dataset
(v2_frame_dataset.py output, held-out validation speakers) -- split by
CLIP id (not frame id, to avoid frame-level leakage across the split from
the same clip) into a 50/50 calib/heldout partition, deterministic. All
metrics reported "after" are measured on the heldout half, never on the
clips used to fit the calibration parameters, and never on day1_public
(the final test set, untouched here).

Phase 11's fitted 2x2 affine is additionally applied to the
validation_suite_v2-based per-frame logits (mentrawearnet_v2_<tag>_frame_
logits.npz, which is clip-structured with known clip_is_target labels) to
test whether it improves validation clip-level EER -- still validation-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

RESULTS_DIR = Path("evaluation/results")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def ece(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    total = len(y_true)
    e = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        e += (mask.sum() / total) * abs(acc - conf)
    return float(e)


def f1_far_frr(y_true, y_pred):
    y_true = np.asarray(y_true).astype(bool); y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred)); fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred)); tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision == precision and recall == recall and (precision + recall) > 0) else float("nan"))
    far = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    frr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    return {"f1": f1, "far": far, "frr": frr}


def fit_temperature(logit, label, iters=200, lr=0.05):
    """1-parameter temperature T>0: p = sigmoid(logit / T). Fit by gradient
    descent on NLL (closed-form-free, tiny 1D problem)."""
    logT = 0.0  # T = exp(logT), start at T=1
    for _ in range(iters):
        T = np.exp(logT)
        p = sigmoid(logit / T)
        # dNLL/dT via chain rule (numerical-safe): grad wrt logT
        grad_logit_over_T = -(logit / T)  # d(logit/T)/dlogT = -logit/T
        dL_dp = (p - label)  # dNLL/dp * dp/dlogitscaled = (p-label) for BCE-sigmoid combo
        grad = np.mean(dL_dp * grad_logit_over_T)
        logT -= lr * grad
    return float(np.exp(logT))


def fit_platt(logit, label):
    """scale+bias: p = sigmoid(a*logit + b), via 1-feature logistic regression."""
    lr = LogisticRegression(C=1e6, max_iter=2000)
    lr.fit(logit.reshape(-1, 1), label)
    a = float(lr.coef_[0, 0]); b = float(lr.intercept_[0])
    return a, b


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

    d = np.load(RESULTS_DIR / f"mentrawearnet_v2_{args.tag}_joint_frame_dataset.npz", allow_pickle=True)
    clip_id = d["clip_id"]
    n_clips = int(clip_id.max()) + 1
    calib_clips = set(range(0, n_clips, 2))  # even clip ids -> calibration
    is_calib = np.isin(clip_id, list(calib_clips))

    wl, el = d["wearer_logit"], d["environment_logit"]
    wt, et = d["true_wearer"], d["true_environment"]
    state = d["true_state"]

    result = {"tag": args.tag, "n_calib_frames": int(is_calib.sum()), "n_heldout_frames": int((~is_calib).sum())}

    # ---------------- Phase 10: per-head temperature + Platt ----------------
    for head, logit, label in (("wearer", wl, wt), ("environment", el, et)):
        cal_logit, cal_label = logit[is_calib], label[is_calib]
        held_logit, held_label = logit[~is_calib], label[~is_calib]

        p_raw = sigmoid(held_logit)
        raw_metrics = {
            "nll": float(log_loss(held_label, np.clip(p_raw, 1e-7, 1 - 1e-7))),
            "brier": float(brier_score_loss(held_label, p_raw)),
            "ece": ece(held_label, p_raw),
            "auroc": float(roc_auc_score(held_label, p_raw)),
            **f1_far_frr(held_label, p_raw > 0.5),
        }

        T = fit_temperature(cal_logit, cal_label)
        p_temp = sigmoid(held_logit / T)
        temp_metrics = {
            "T": T,
            "nll": float(log_loss(held_label, np.clip(p_temp, 1e-7, 1 - 1e-7))),
            "brier": float(brier_score_loss(held_label, p_temp)),
            "ece": ece(held_label, p_temp),
            "auroc": float(roc_auc_score(held_label, p_temp)),  # must equal raw AUROC (monotonic transform)
            **f1_far_frr(held_label, p_temp > 0.5),  # must equal raw F1/FAR/FRR (T>0 doesn't move the logit=0 boundary)
        }

        a, b = fit_platt(cal_logit, cal_label)
        p_platt = sigmoid(a * held_logit + b)
        platt_metrics = {
            "a": a, "b": b,
            "nll": float(log_loss(held_label, np.clip(p_platt, 1e-7, 1 - 1e-7))),
            "brier": float(brier_score_loss(held_label, p_platt)),
            "ece": ece(held_label, p_platt),
            "auroc": float(roc_auc_score(held_label, p_platt)),  # also must equal raw AUROC (still monotonic in logit)
            **f1_far_frr(held_label, p_platt > 0.5),  # CAN differ from raw: bias term shifts the effective threshold
        }

        result[f"phase10_{head}"] = {"raw": raw_metrics, "temperature_scaled": temp_metrics, "platt_scaled": platt_metrics}

    # ---------------- Phase 11: joint 2x2 affine + bias (6 params) ----------------
    X_cal = np.stack([wl[is_calib], el[is_calib]], axis=1)
    X_held = np.stack([wl[~is_calib], el[~is_calib]], axis=1)

    lr_w = LogisticRegression(C=1e6, max_iter=2000).fit(X_cal, wt[is_calib])
    lr_e = LogisticRegression(C=1e6, max_iter=2000).fit(X_cal, et[is_calib])
    A = np.array([lr_w.coef_[0], lr_e.coef_[0]])  # 2x2
    bvec = np.array([lr_w.intercept_[0], lr_e.intercept_[0]])  # 2
    wl2_held = X_held @ A[0] + bvec[0]
    el2_held = X_held @ A[1] + bvec[1]

    mask_10v01_held = ((state[~is_calib] == 1) | (state[~is_calib] == 2))
    label_10v01 = (state[~is_calib][mask_10v01_held] == 1).astype(int)
    raw_wl_10v01 = wl[~is_calib][mask_10v01_held]
    raw_dom_10v01 = (wl - el)[~is_calib][mask_10v01_held]
    cal_wl_10v01 = wl2_held[mask_10v01_held]
    cal_dom_10v01 = (wl2_held - el2_held)[mask_10v01_held]

    def far_frr_f1_at_0(label, score):
        return f1_far_frr(label, score > np.median(score))  # threshold-free-ish sanity; AUROC is the primary number

    result["phase11_joint_affine"] = {
        "A_matrix": A.tolist(), "bias": bvec.tolist(),
        "note": "row0=wearer' coeffs on [wl,el], row1=environment' coeffs on [wl,el]",
        "10v01_heldout": {
            "raw_wearer_logit_auroc": float(roc_auc_score(label_10v01, raw_wl_10v01)),
            "raw_dominance_logit_auroc": float(roc_auc_score(label_10v01, raw_dom_10v01)),
            "calibrated_wearer_prime_auroc": float(roc_auc_score(label_10v01, cal_wl_10v01)),
            "calibrated_dominance_prime_auroc": float(roc_auc_score(label_10v01, cal_dom_10v01)),
        },
    }

    # apply the fitted affine to the validation_suite_v2-based frame_logits.npz
    # (clip-structured, has clip_is_target) to test clip-level EER impact.
    fl_path = RESULTS_DIR / f"mentrawearnet_v2_{args.tag}_frame_logits.npz"
    if fl_path.exists():
        fl = np.load(fl_path, allow_pickle=True)
        cwl, cel = fl["wearer_logit"], fl["environment_logit"]
        ccid = fl["clip_id"]
        cslice = fl["slice_name"]
        Xc = np.stack([cwl, cel], axis=1)
        cal_wl_all = Xc @ A[0] + bvec[0]
        cal_prob_all = sigmoid(cal_wl_all)
        raw_prob_all = sigmoid(cwl)

        cs_path = RESULTS_DIR / f"mentrawearnet_v2_{args.tag}_clip_scores.npz"
        cs = np.load(cs_path, allow_pickle=True)
        clip_ids_order = cs["clip_id"].astype(int)
        clip_slice = cs["slice"]
        clip_target = cs["clip_is_target"].astype(bool)

        def mean_per_clip(frame_score):
            out = np.zeros(len(clip_ids_order))
            for i, cid in enumerate(clip_ids_order):
                m = ccid == cid
                out[i] = frame_score[m].mean() if m.any() else np.nan
            return out

        raw_clip_mean = mean_per_clip(raw_prob_all)
        cal_clip_mean = mean_per_clip(cal_prob_all)

        def eer_table(score):
            def eer_for(pos_mask, neg_mask):
                pos, neg = score[pos_mask], score[neg_mask]
                if len(pos) == 0 or len(neg) == 0:
                    return float("nan")
                return compute_eer(pos, neg) * 100
            clean_pos = np.array([s.startswith("clean_solo_wearer") for s in clip_slice]) & clip_target
            clean_neg = np.array([s.startswith("clean_solo_environment") for s in clip_slice]) & ~clip_target
            out = {"clean_eer_pct": eer_for(clean_pos, clean_neg)}
            for tir in (5, 0, -5, -10):
                pos = np.array([s in (f"overlap_random_tir{tir:+d}", f"overlap_hardpair_tir{tir:+d}") for s in clip_slice]) & clip_target
                neg = np.array([s == f"tir_negative_tir{tir:+d}" for s in clip_slice]) & ~clip_target
                out[f"tir{tir:+d}_eer_pct"] = eer_for(pos, neg)
            return out

        result["phase11_clip_eer_impact"] = {
            "raw_mean_wearer_prob": eer_table(raw_clip_mean),
            "joint_affine_calibrated_mean_wearer_prob": eer_table(cal_clip_mean),
        }

    out_path = RESULTS_DIR / f"mentrawearnet_v2_{args.tag}_calibration.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
