"""GeoWearNet-E0: physical-heuristic baseline (Phase 5-10).

No neural network. Extracts the interpretable physical features from
`training/geowearnet/features.py` on SIMULATED (Phase 2/19: LibriSpeech +
synthetic transfer function, never real Mentra geometry) wearer/environment
clips from `training/geowearnet/dataset.py`, trains simple classifiers
(logistic regression / tiny MLP), runs the single-feature ablation (Phase
8), and runs the mandatory anti-cheating battery (Phase 9), reporting the
Phase 10 verdict and the Phase 33 amplitude-dependence score.

Run: .venv/bin/python3 -m training.diagnostics.geowearnet_e0
Output: evaluation/geowearnet/results/geowearnet_e0_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.preprocessing import StandardScaler

from training.geowearnet import dataset as gwd
from training.geowearnet import features as gwf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "evaluation/geowearnet/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRAIN = 500
N_VAL = 120
N_TEST = 180
N_ANTICHEAT = 150
SEED = 1234


def clip_to_pooled_features(waveform: np.ndarray) -> np.ndarray:
    """Per-clip feature vector: [mean, std] over frames for every physical
    feature -- a clip-level classifier needs a fixed-size vector, and
    mean+std keeps this an aggregate of genuinely frame-local, absolute-
    level-preserving features (no per-clip normalization is applied to the
    WAVEFORM at any point; only the final feature vector is z-scored by the
    classifier's fitted StandardScaler, which is fit on TRAIN ONLY and
    reused unchanged at eval time -- this does not touch the audio)."""
    f = gwf.extract_features(waveform)
    if f.shape[0] == 0:
        return np.zeros(2 * len(gwf.FEATURE_NAMES), dtype=np.float32)
    return np.concatenate([f.mean(axis=0), f.std(axis=0)]).astype(np.float32)


def featurize_all(waveforms):
    return np.stack([clip_to_pooled_features(w) for w in waveforms], axis=0)


def pooled_group_indices(names):
    """Maps a FEATURE_GROUPS name list to indices into the [mean(14) |
    std(14)] pooled vector (both mean and std columns of each named
    feature)."""
    base = gwf.feature_group_indices(names)
    n = len(gwf.FEATURE_NAMES)
    return base + [i + n for i in base]


POOLED_GROUPS = {
    "rms_only": pooled_group_indices(["log_rms"]),
    "peak_only": pooled_group_indices(["peak"]),
    "clipping_only": pooled_group_indices(["clipping_fraction"]),
    "amplitude_only": pooled_group_indices(["log_rms", "peak", "crest_factor", "clipping_fraction"]),
    "spectral_only": pooled_group_indices(
        ["spectral_centroid", "spectral_rolloff85", "spectral_tilt",
         "lf_energy_ratio", "mf_energy_ratio", "hf_energy_ratio",
         "spectral_flux", "harmonicity", "zero_crossing_rate"]
    ),
    "amplitude_plus_spectral": pooled_group_indices(
        ["log_rms", "peak", "crest_factor", "clipping_fraction",
         "spectral_centroid", "spectral_rolloff85", "spectral_tilt",
         "lf_energy_ratio", "mf_energy_ratio", "hf_energy_ratio",
         "spectral_flux", "harmonicity", "zero_crossing_rate", "energy_derivative"]
    ),
    "all": list(range(2 * len(gwf.FEATURE_NAMES))),
}


def eer_far_frr(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    # FAR/FRR at the 0.5-probability operating point is reported separately
    # by the caller (needs raw predictions, not just scores).
    return eer


def evaluate(y_true, probs, preds):
    auroc = float(roc_auc_score(y_true, probs)) if len(set(y_true)) > 1 else float("nan")
    auprc = float(average_precision_score(y_true, probs)) if len(set(y_true)) > 1 else float("nan")
    eer = eer_far_frr(y_true, probs) if len(set(y_true)) > 1 else float("nan")
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)
    neg = y_true == 0
    pos = y_true == 1
    far = float((preds[neg] == 1).mean()) if neg.sum() > 0 else float("nan")  # env misclassified as wearer
    frr = float((preds[pos] == 0).mean()) if pos.sum() > 0 else float("nan")  # wearer misclassified as env
    acc = float((preds == y_true).mean())
    return dict(auroc=auroc, auprc=auprc, eer=eer, far=far, frr=frr, accuracy=acc, n=int(len(y_true)))


def main():
    t0 = time.time()
    print("[geowearnet-e0] generating simulated train/val/test clip sets (condition=normal)...")
    train_wav, train_y, train_spk = gwd.build_e0_clip_examples("train", N_TRAIN, "normal", seed=SEED)
    val_wav, val_y, _ = gwd.build_e0_clip_examples("val", N_VAL, "normal", seed=SEED + 1)
    test_wav, test_y, _ = gwd.build_e0_clip_examples("test", N_TEST, "normal", seed=SEED + 2)

    print(f"[geowearnet-e0] extracting features ({len(train_wav)+len(val_wav)+len(test_wav)} clips)...")
    X_train_full = featurize_all(train_wav)
    X_val_full = featurize_all(val_wav)
    X_test_full = featurize_all(test_wav)
    print(f"[geowearnet-e0] feature extraction done in {time.time()-t0:.1f}s")

    results = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    results["data"] = {
        "type": "SIMULATED (LibriSpeech + synthetic S0 transfer function, see docs/geowearnet_dataset_licenses.md)",
        "n_train": len(train_wav), "n_val": len(val_wav), "n_test": len(test_wav),
        "person_disjoint_split": True,
        "role_randomization": True,
    }

    # ---- Phase 7/8: main classifier + single-feature ablation ----
    ablation_results = {}
    trained_models = {}
    for group_name, idx in POOLED_GROUPS.items():
        scaler = StandardScaler().fit(X_train_full[:, idx])
        Xtr = scaler.transform(X_train_full[:, idx])
        Xva = scaler.transform(X_val_full[:, idx])
        Xte = scaler.transform(X_test_full[:, idx])

        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, train_y)
        val_probs = clf.predict_proba(Xva)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        test_probs = clf.predict_proba(Xte)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)

        ablation_results[group_name] = {
            "n_features": len(idx),
            "val": evaluate(val_y, val_probs, val_preds),
            "test": evaluate(test_y, test_probs, test_preds),
        }
        if group_name == "all":
            coefs = dict(zip(
                [f"mean_{n}" for n in gwf.FEATURE_NAMES] + [f"std_{n}" for n in gwf.FEATURE_NAMES],
                clf.coef_[0].tolist(),
            ))
            ablation_results["all"]["logreg_coefficients"] = coefs
        trained_models[group_name] = (clf, scaler, idx)

    results["e0_ablation_results"] = ablation_results

    # Also a tiny MLP on the full feature set, for comparison (Phase 7 allows "tiny MLP")
    scaler_all = StandardScaler().fit(X_train_full)
    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=0)
    mlp.fit(scaler_all.transform(X_train_full), train_y)
    mlp_test_probs = mlp.predict_proba(scaler_all.transform(X_test_full))[:, 1]
    mlp_test_preds = (mlp_test_probs >= 0.5).astype(int)
    results["e0_tiny_mlp_all_features_test"] = evaluate(test_y, mlp_test_probs, mlp_test_preds)

    # ---- Phase 9: anti-cheating battery, using the "all"-feature LogisticRegression
    #      trained ONLY on the normal condition (train_wav above), evaluated on
    #      adversarially-generated test sets. This directly answers "does
    #      discrimination survive" perturbation, per Phase 9's own framing. ----
    clf_all, scaler_all_lr, idx_all = trained_models["all"]

    def eval_condition(name, waveforms, y):
        X = featurize_all(waveforms)
        Xs = scaler_all_lr.transform(X[:, idx_all])
        probs = clf_all.predict_proba(Xs)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return evaluate(y, probs, preds)

    anti_cheat = {}
    normal_metrics = ablation_results["all"]["test"]
    anti_cheat["normal"] = normal_metrics

    for cond in ["level_matched", "wearer_whisper", "close_bystander", "shouting_bystander", "wearer_loud"]:
        wav, y, _ = gwd.build_e0_clip_examples("test", N_ANTICHEAT, cond, seed=SEED + 100 + hash(cond) % 1000)
        anti_cheat[cond] = eval_condition(cond, wav, y)

    rg_wav, rg_y, _ = gwd.build_e0_random_gain_examples("test", N_ANTICHEAT, seed=SEED + 500)
    anti_cheat["random_gain"] = eval_condition("random_gain", rg_wav, rg_y)

    results["anti_cheating"] = anti_cheat

    # Phase 33: amplitude dependence score (deltas vs normal)
    def delta(cond):
        d = {}
        for k in ["auroc", "accuracy", "far", "frr"]:
            a = anti_cheat[cond][k]
            b = normal_metrics[k]
            d[f"delta_{k}"] = None if (a is None or b is None or np.isnan(a) or np.isnan(b)) else float(a - b)
        return d

    amplitude_dependence = {
        "level_matched": delta("level_matched"),
        "random_gain": delta("random_gain"),
    }
    results["amplitude_dependence_score"] = amplitude_dependence

    # ---- Phase 10 verdict ----
    all_auroc = ablation_results["all"]["test"]["auroc"]
    amp_auroc = ablation_results["amplitude_only"]["test"]["auroc"]
    spec_auroc = ablation_results["spectral_only"]["test"]["auroc"]
    level_matched_auroc = anti_cheat["level_matched"]["auroc"]
    random_gain_auroc = anti_cheat["random_gain"]["auroc"]

    if all_auroc < 0.60:
        verdict = "E0_NO_USEFUL_SIGNAL"
    elif level_matched_auroc < 0.58 and random_gain_auroc < 0.58 and spec_auroc < 0.58:
        verdict = "E0_MOSTLY_LOUDNESS_SHORTCUT"
    elif level_matched_auroc >= 0.60 or spec_auroc >= 0.60:
        verdict = "E0_PHYSICAL_SIGNAL_STRONG" if level_matched_auroc >= 0.70 else "E0_PHYSICAL_SIGNAL_PARTIAL"
    else:
        verdict = "E0_PHYSICAL_SIGNAL_PARTIAL"

    results["verdict"] = {
        "value": verdict,
        "all_features_test_auroc": all_auroc,
        "amplitude_only_test_auroc": amp_auroc,
        "spectral_only_test_auroc": spec_auroc,
        "level_matched_auroc": level_matched_auroc,
        "random_gain_auroc": random_gain_auroc,
        "note": "SIMULATED_GEOMETRY data only (Phase 2/47) -- this verdict is about "
                "whether the simulated transfer-function hypothesis has ANY nontrivial, "
                "non-loudness signal, not about real Mentra wearer detection.",
    }

    results["runtime_seconds"] = time.time() - t0

    out_path = OUT_DIR / "geowearnet_e0_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: None)
    print(f"[geowearnet-e0] wrote {out_path}")
    print(json.dumps(results["verdict"], indent=2))
    return results


if __name__ == "__main__":
    main()
