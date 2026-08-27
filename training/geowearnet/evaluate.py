"""Score a GeoWearNet checkpoint against the deterministic evaluation suite.

Workstreams S, T, U, AI, AJ, AO, AH, AS.

Usage
  # build the suites once (CPU, parallel-safe)
  python3 -m training.geowearnet.evaluate --build-suite --generation S1 --n-scenes 200
  # score a checkpoint
  python3 -m training.geowearnet.evaluate --checkpoint <path> --generation S1
  # cross-simulator (Workstream T): score an S1-trained model on S2 trials
  python3 -m training.geowearnet.evaluate --checkpoint <S1 ckpt> --generation S2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from . import evalsuite as ES
from .data import DataConfig, compute_norm_stats, physical_feature_indices
from .evalsuite import (SUITE_CONDITIONS, bootstrap_ci, calibration_report, core_metrics,
                        error_buckets, fit_platt, fit_temperature, load_condition)
from .model_zoo import ZooConfig, build_model

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "evaluation/geowearnet/results"


def load_checkpoint(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = ck["model_config"]
    cfg = ZooConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in mc.items()})
    model = {"tcn": None}  # placeholder to keep import list explicit
    from .model_zoo import ARCHS
    model = ARCHS[cfg.arch](cfg)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    return model, ck


def _normalize(lm: np.ndarray, ph: np.ndarray, mode: str, stats) -> tuple:
    if mode == "none":
        return lm, ph
    if mode == "global":
        lm = (lm - stats["mel_mean"]) / stats["mel_std"]
        if ph.shape[1]:
            ph = (ph - stats["phys_mean"]) / stats["phys_std"]
        return lm, ph
    if mode == "cmvn":
        def causal(x):
            c = np.cumsum(x, 0); c2 = np.cumsum(x**2, 0)
            k = np.arange(1, len(x) + 1)[:, None]
            m = c / k
            return (x - m) / np.sqrt(np.maximum(c2 / k - m**2, 1e-6))
        return causal(lm), (causal(ph) if ph.shape[1] else ph)
    raise ValueError(mode)


@torch.no_grad()
def run_model(model, ck, data: Dict[str, np.ndarray], device: str = "cpu",
              stats=None) -> tuple:
    """Runs the model scene-by-scene (so CMVN and TCN state respect scene
    boundaries exactly as they would at inference time)."""
    tcfg = ck["config"]
    mode = tcfg["normalization"]
    keep = physical_feature_indices(tcfg.get("drop_amplitude_features", False))
    use_phys = tcfg.get("use_physical", True) and model.config.use_physical_features
    si = data["scene_index"]
    W, E = [], []
    for s in np.unique(si):
        m = si == s
        lm = data["log_mel"][m].astype(np.float32)
        ph = data["physical"][m][:, keep].astype(np.float32) if use_phys else np.zeros((m.sum(), 0), np.float32)
        lm, ph = _normalize(lm, ph, mode, stats)
        # cmvn's causal running-stats division promotes float32 -> float64 (the
        # frame-count array is int64), which then mismatches the model's float32
        # weights. Cast explicitly, matching the training-side dataset's convention.
        t_lm = torch.from_numpy(np.ascontiguousarray(lm, dtype=np.float32))[None].to(device)
        t_ph = torch.from_numpy(np.ascontiguousarray(ph, dtype=np.float32))[None].to(device) if use_phys else None
        out = model(t_lm, t_ph)
        W.append(out["wearer_logits"][0].float().cpu().numpy())
        E.append(out["environment_logits"][0].float().cpu().numpy())
    return np.concatenate(W), np.concatenate(E)


def evaluate_condition(w, e, d, seed: int = 0, n_boot: int = 250) -> Dict[str, object]:
    y_w = d["wearer"].astype(np.int64)
    y_e = d["environment"].astype(np.int64)
    st = d["state"].astype(np.int64)
    si = d["scene_index"]

    res: Dict[str, object] = {}
    res["wearer"] = core_metrics(y_w, w)
    res["environment"] = core_metrics(y_e, e)
    pt, lo, hi = bootstrap_ci(y_w, w, si, n_boot=n_boot, seed=seed)
    res["wearer"]["auroc_ci95"] = [lo, hi]

    solo = (y_w + y_e) == 1
    if solo.sum() > 50 and len(np.unique(y_w[solo])) == 2:
        score = (w - e)[solo]
        res["wearer_vs_env_solo"] = core_metrics(y_w[solo], score)
        _, lo2, hi2 = bootstrap_ci(y_w[solo], score, si[solo], n_boot=n_boot, seed=seed + 1)
        res["wearer_vs_env_solo"]["auroc_ci95"] = [lo2, hi2]

    pred_ov = (w > 0) & (e > 0)
    true_ov = st == 3
    tp = float((pred_ov & true_ov).sum()); fp = float((pred_ov & ~true_ov).sum()); fn = float((~pred_ov & true_ov).sum())
    res["overlap_f1"] = float(2 * tp / max(2 * tp + fp + fn, 1e-9))

    pred_state = (w > 0).astype(np.int64) + 2 * (e > 0).astype(np.int64)
    conf = np.zeros((4, 4), dtype=np.int64)
    for a, b in zip(st, pred_state):
        conf[a, b] += 1
    res["state_confusion"] = conf.tolist()
    res["state_accuracy"] = float((pred_state == st).mean())
    env_only = st == 2
    if env_only.sum():
        res["false_wearer_rate_on_env_only"] = float((w[env_only] > 0).mean())
    wear = y_w == 1
    if wear.sum():
        res["missed_wearer_rate"] = float((w[wear] <= 0).mean())
    res["state_fractions"] = {ES.SC.STATE_NAMES[i]: float((st == i).mean()) for i in range(4)}
    # Workstream J: how much can raw level alone do on THESE trials?
    lr = d["physical"][:, 0]  # log_rms is feature 0
    if len(np.unique(y_w)) == 2:
        res["logrms_alone_auroc"] = ES._auroc(y_w, lr)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-suite", action="store_true")
    ap.add_argument("--generation", default="S1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-scenes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--duration-s", type=float, default=6.0)
    ap.add_argument("--conditions", default=",".join(SUITE_CONDITIONS))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=250)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    conds = [c for c in a.conditions.split(",") if c]

    if a.build_suite:
        print(f"building suite generation={a.generation} split={a.split} "
              f"n_scenes={a.n_scenes} seed={a.seed}", flush=True)
        ES.build_suite(a.generation, a.split, a.n_scenes, a.seed, conds, a.duration_s)
        print("suite built")
        if not a.checkpoint:
            return

    assert a.checkpoint, "--checkpoint required (or use --build-suite alone)"
    t0 = time.time()
    model, ck = load_checkpoint(a.checkpoint, a.device)
    tcfg = ck["config"]
    stats = None
    if tcfg["normalization"] == "global":
        stats = compute_norm_stats(DataConfig(
            generation=tcfg["generation"], n_mels=tcfg["n_mels"],
            drop_amplitude_features=tcfg.get("drop_amplitude_features", False),
            condition=tcfg.get("condition", "train_mix")))

    out: Dict[str, object] = {
        "checkpoint": str(a.checkpoint),
        "trained_on_generation": tcfg["generation"],
        "evaluated_on_generation": a.generation,
        "cross_simulator": tcfg["generation"] != a.generation,
        "split": a.split,
        "suite_seed": a.seed,
        "n_scenes_per_condition": a.n_scenes,
        "model": {
            "arch": tcfg["arch"], "variant": tcfg["variant"],
            "params": ck["params"], "receptive_field_frames": ck["receptive_field_frames"],
            "context_ms": ck["context_ms"], "global_step": ck["global_step"],
            "normalization": tcfg["normalization"], "use_physical": tcfg["use_physical"],
            "drop_amplitude_features": tcfg.get("drop_amplitude_features", False),
            "loss_mode": tcfg["loss_mode"],
        },
        "enrollment_required": False,
        "data_note": "SIMULATED GEOMETRY (LibriSpeech CC BY 4.0 + MUSAN). NOT Mentra hardware validation.",
        "conditions": {},
    }

    for c in conds:
        p = ES.suite_path(a.generation, a.split, a.seed, c, a.duration_s)
        if not p.exists():
            print(f"  [skip] no suite for {c}: {p}", flush=True)
            continue
        d = load_condition(p)
        w, e = run_model(model, ck, d, a.device, stats)
        r = evaluate_condition(w, e, d, seed=a.seed, n_boot=a.n_boot)
        if c == "normal":
            r["error_buckets"] = error_buckets(w, d["wearer"].astype(np.int64),
                                               d["state"].astype(np.int64),
                                               d["scene_index"], d["scene_meta"])
        out["conditions"][c] = r
        print(f"  {c:28s} wearer AUROC {r['wearer']['auroc']:.4f} "
              f"[{r['wearer']['auroc_ci95'][0]:.4f},{r['wearer']['auroc_ci95'][1]:.4f}]  "
              f"solo {r.get('wearer_vs_env_solo',{}).get('auroc',float('nan')):.4f}  "
              f"FRR@FAR5 {r['wearer']['frr_at_far5']:.4f}  "
              f"ovF1 {r['overlap_f1']:.3f}  logRMS-only {r.get('logrms_alone_auroc',float('nan')):.3f}",
              flush=True)

    # --- Workstream AI: calibrate on VAL only, report on test ---
    val_p = ES.suite_path(a.generation, "val", a.seed + 1, "normal", a.duration_s)
    if val_p.exists():
        dv = load_condition(val_p)
        wv, ev = run_model(model, ck, dv, a.device, stats)
        yv = dv["wearer"].astype(np.int64)
        T = fit_temperature(wv, yv)
        platt = fit_platt(wv, yv)
        if "normal" in out["conditions"]:
            dt = load_condition(ES.suite_path(a.generation, a.split, a.seed, "normal", a.duration_s))
            wt, _ = run_model(model, ck, dt, a.device, stats)
            out["calibration"] = calibration_report(wt, dt["wearer"].astype(np.int64), T, platt)
            out["calibration"]["fitted_on"] = "validation split, condition=normal"
            print("  calibration:", json.dumps({k: v for k, v in out["calibration"].items()
                                                if isinstance(v, float)}, indent=None), flush=True)
    else:
        out["calibration"] = {"status": "SKIPPED", "reason": f"no val suite at {val_p}"}

    # --- Workstream AS: robustness summary used for model selection ---
    def g(cond, path, default=float("nan")):
        cur = out["conditions"].get(cond, {})
        for k in path:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        return cur if isinstance(cur, float) else default

    rob = {
        "normal_auroc": g("normal", ["wearer", "auroc"]),
        "level_matched_auroc": g("level_matched", ["wearer", "auroc"]),
        "random_gain_auroc": g("random_gain", ["wearer", "auroc"]),
        "bystander_close_auroc": g("bystander_close", ["wearer", "auroc"]),
        "shouting_bystander_auroc": g("shouting_bystander", ["wearer", "auroc"]),
        "ood_geometry_auroc": g("ood_geometry", ["wearer", "auroc"]),
        "normal_frr_at_far5": g("normal", ["wearer", "frr_at_far5"]),
        "bystander_close_false_wearer": out["conditions"].get("bystander_close", {}).get("false_wearer_rate_on_env_only"),
        "shouting_false_wearer": out["conditions"].get("shouting_bystander", {}).get("false_wearer_rate_on_env_only"),
        "overlap_f1_normal": out["conditions"].get("normal", {}).get("overlap_f1"),
        "params": ck["params"],
        "context_ms": ck["context_ms"],
    }
    vals = [v for k, v in rob.items() if k.endswith("auroc") and isinstance(v, float) and not np.isnan(v)]
    rob["robust_mean_auroc"] = float(np.mean(vals)) if vals else float("nan")
    rob["worst_condition_auroc"] = float(np.min(vals)) if vals else float("nan")
    out["robustness_summary"] = rob
    out["runtime_seconds"] = time.time() - t0

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = a.out or f"eval_{Path(a.checkpoint).parent.parent.name}_on{a.generation}{('_' + a.tag) if a.tag else ''}.json"
    p = RESULTS / name
    p.write_text(json.dumps(out, indent=2))
    print(f"\nrobustness: {json.dumps(rob, indent=2)}")
    print("wrote", p)


if __name__ == "__main__":
    main()
