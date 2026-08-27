"""GeoWearNet deterministic multi-condition evaluation suite.

Workstreams S (suite + bootstrap CI), T (cross-simulator), U (held-out
geometry), AI (calibration), AJ (FAR-prioritised operating points),
AO (error buckets), AH (E0 vs E1), AT (no overfitting to the sim test set).

DESIGN
------
The suite is BUILT ONCE per (generation, split, seed) and cached to disk as
un-normalised features plus per-scene physical metadata. Every model is then
scored on the IDENTICAL trials. This matters for three reasons:
  * comparability -- E0, E1, every ablation and every architecture see the
    same acoustics, not just the same distribution;
  * cost -- scene synthesis is the expensive part (Workstream D), and a
    150-model sweep would otherwise re-simulate everything 150 times;
  * Workstream AT -- the FINAL held-out suite uses a different seed and
    different generation than anything tuned against, and is built only once.

STATISTICS
----------
Bootstrap resampling is at the SCENE level, never the frame level: frames
within a scene are heavily correlated, so frame-level bootstrap would produce
absurdly tight and dishonest confidence intervals.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import scenes as SC
from .data import DataConfig, GeoWearNetDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "evaluation/geowearnet/suites"

# Per-scene physical covariates recorded for error bucketing (Workstream AO).
META_KEYS = [
    "tir_db", "wearer_range_m", "wearer_drr_db", "env_range_m", "env_drr_db",
    "env_angle_deg", "rt60_s", "head_radius_m", "noise_snr_db",
    "recording_gain_db", "clip_fraction", "compressor_ratio", "mix_rms_db",
]

# Workstream S -- the standard suite.
SUITE_CONDITIONS = [
    "normal", "level_matched", "random_gain", "wearer_quiet", "wearer_loud",
    "bystander_close", "bystander_close_loud", "bystander_loud",
    "shouting_bystander", "heavy_overlap", "multi_talker", "noisy",
    "high_rt60", "eq_mismatch", "compressed_clipped", "ood_geometry",
    "ood_geometry_level_matched",
]


def scene_meta_vector(sc: SC.Scene) -> np.ndarray:
    m = sc.meta
    envs = m["environment"] or [{}]
    dev = m["device"]
    vals = {
        "tir_db": m["tir_db"],
        "wearer_range_m": m["wearer"]["range_m"],
        "wearer_drr_db": m["wearer"]["drr_db"],
        "env_range_m": float(np.mean([e["range_m"] for e in envs])),
        "env_drr_db": float(np.mean([e["drr_db"] for e in envs])),
        "env_angle_deg": float(np.mean([e["angle_deg"] for e in envs])),
        "rt60_s": m["wearer"]["rt60_s"],
        "head_radius_m": m["wearer"]["head_radius_m"],
        "noise_snr_db": dev["noise_snr_db"] if dev["noise_snr_db"] is not None else 99.0,
        "recording_gain_db": dev["recording_gain_db"],
        "clip_fraction": m["clip_fraction"],
        "compressor_ratio": dev["compressor_ratio"],
        "mix_rms_db": 20.0 * np.log10(m["mix_rms"] + 1e-12),
    }
    return np.array([vals[k] for k in META_KEYS], dtype=np.float32)


def suite_path(generation: str, split: str, seed: int, condition: str, duration_s: float) -> Path:
    return SUITE_DIR / f"{generation}_{split}_s{seed}_d{duration_s:g}" / f"{condition}.npz"


def build_condition(
    generation: str, split: str, condition: str, n_scenes: int, seed: int,
    duration_s: float = 6.0, force: bool = False,
) -> Path:
    p = suite_path(generation, split, seed, condition, duration_s)
    if p.exists() and not force:
        return p
    cfg = DataConfig(split=split, generation=generation, condition=condition,
                     duration_s=duration_s, normalization="none", seed=seed)
    ds = GeoWearNetDataset(cfg)
    LM, PH, W, E, ST, SI, MET = [], [], [], [], [], [], []
    for i in range(n_scenes):
        sc = ds.scene_for(i)
        lm, ph = ds.featurize(sc.audio)
        t = min(len(lm), len(ph), len(sc.wearer))
        LM.append(lm[:t]); PH.append(ph[:t])
        W.append(sc.wearer[:t]); E.append(sc.environment[:t]); ST.append(sc.state[:t])
        SI.append(np.full(t, i, dtype=np.int32))
        MET.append(scene_meta_vector(sc))
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        log_mel=np.concatenate(LM).astype(np.float32),
        physical=np.concatenate(PH).astype(np.float32),
        wearer=np.concatenate(W).astype(np.int8),
        environment=np.concatenate(E).astype(np.int8),
        state=np.concatenate(ST).astype(np.int8),
        scene_index=np.concatenate(SI),
        scene_meta=np.stack(MET),
        meta_keys=np.array(META_KEYS),
        info=np.array(json.dumps({
            "generation": generation, "split": split, "condition": condition,
            "n_scenes": n_scenes, "seed": seed, "duration_s": duration_s,
            "note": "SIMULATED GEOMETRY. Not Mentra hardware data.",
        })),
    )
    return p


def build_suite(generation: str, split: str = "test", n_scenes: int = 200, seed: int = 4242,
                conditions: Sequence[str] = tuple(SUITE_CONDITIONS), duration_s: float = 6.0,
                force: bool = False, verbose: bool = True) -> Dict[str, Path]:
    out = {}
    for c in conditions:
        p = build_condition(generation, split, c, n_scenes, seed, duration_s, force)
        out[c] = p
        if verbose:
            with np.load(p) as z:
                print(f"  {generation}/{c:28s} {len(z['wearer']):8d} frames  {p.stat().st_size/1e6:6.1f} MB", flush=True)
    return out


def load_condition(p: Path) -> Dict[str, np.ndarray]:
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files if k not in ("info", "meta_keys")}
        d["_info"] = json.loads(str(z["info"]))
    return d


# ---------------------------------------------------------------------------
# metrics with scene-level bootstrap
# ---------------------------------------------------------------------------
def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(np.int8)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _far_frr(y: np.ndarray, s: np.ndarray):
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    n_pos, n_neg = max(int(ys.sum()), 1), max(int(len(ys) - ys.sum()), 1)
    return (1 - ys).cumsum() / n_neg, 1.0 - ys.cumsum() / n_pos


def frr_at_far(y: np.ndarray, s: np.ndarray, t: float) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    far, frr = _far_frr(y, s)
    return float(frr[min(int(np.searchsorted(far, t)), len(frr) - 1)])


def far_at_recall(y: np.ndarray, s: np.ndarray, recall: float) -> float:
    """Workstream AJ: what false-accept rate do we pay for a target wearer recall?"""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    far, frr = _far_frr(y, s)
    tpr = 1.0 - frr
    i = int(np.searchsorted(tpr, recall))
    return float(far[min(i, len(far) - 1)])


def core_metrics(y: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    out = {"auroc": _auroc(y, s), "positive_rate": float(y.mean()), "n": int(len(y))}
    for t in (0.20, 0.10, 0.05, 0.01):
        out[f"frr_at_far{int(t*100)}"] = frr_at_far(y, s, t)
    for r in (0.90, 0.95, 0.975):
        out[f"far_at_recall{int(r*1000)}"] = far_at_recall(y, s, r)
    far, frr = _far_frr(y, s)
    i = int(np.argmin(np.abs(far - frr)))
    out["eer"] = float((far[i] + frr[i]) / 2)
    return out


def bootstrap_ci(
    y: np.ndarray, s: np.ndarray, scene_index: np.ndarray,
    fn=_auroc, n_boot: int = 300, seed: int = 0, alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Scene-level (cluster) bootstrap. Frame-level bootstrap would be wrong:
    frames inside one scene share geometry, speaker and device, so they are not
    independent draws."""
    point = fn(y, s)
    uniq = np.unique(scene_index)
    if len(uniq) < 8:
        return point, float("nan"), float("nan")
    groups = {u: np.flatnonzero(scene_index == u) for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[u] for u in pick])
        v = fn(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 20:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


# ---------------------------------------------------------------------------
# calibration (Workstream AI)
# ---------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """1-D temperature scaling fitted on VALIDATION logits only."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.concatenate([np.linspace(0.20, 3.0, 57), np.linspace(3.2, 12.0, 45)]):
        nll = binary_nll(logits / t, y)
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def fit_platt(logits: np.ndarray, y: np.ndarray, iters: int = 200, lr: float = 0.5) -> Tuple[float, float]:
    """Platt scaling p = sigmoid(a*logit + b), fitted by gradient descent on
    validation logits only."""
    a, b = 1.0, 0.0
    x = logits.astype(np.float64)
    yy = y.astype(np.float64)
    n = len(x)
    xs = float(np.std(x) + 1e-6)
    for _ in range(iters):
        p = sigmoid(a * x + b)
        ga = float(np.dot(p - yy, x) / n) / (xs**2)
        gb = float(np.mean(p - yy))
        a -= lr * ga
        b -= lr * gb
    return a, b


def binary_nll(logits: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(sigmoid(logits), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((sigmoid(logits) - y) ** 2))


def ece(logits: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error, equal-width bins."""
    p = sigmoid(logits)
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(p)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(e)


def calibration_report(logits: np.ndarray, y: np.ndarray, temperature: float,
                       platt: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
    rep = {
        "raw_nll": binary_nll(logits, y), "raw_brier": brier(logits, y), "raw_ece": ece(logits, y),
        "temperature": float(temperature),
        "temp_nll": binary_nll(logits / temperature, y),
        "temp_brier": brier(logits / temperature, y),
        "temp_ece": ece(logits / temperature, y),
    }
    if platt is not None:
        a, b = platt
        rep.update({"platt_a": float(a), "platt_b": float(b),
                    "platt_nll": binary_nll(a * logits + b, y),
                    "platt_brier": brier(a * logits + b, y),
                    "platt_ece": ece(a * logits + b, y)})
    rep["note"] = "Calibration cannot fix ranking; AUROC is unchanged by any monotone rescaling."
    return rep


# ---------------------------------------------------------------------------
# error buckets (Workstream AO)
# ---------------------------------------------------------------------------
def error_buckets(
    w_logit: np.ndarray, w_lab: np.ndarray, state: np.ndarray,
    scene_index: np.ndarray, scene_meta: np.ndarray, n_bins: int = 4,
) -> Dict[str, dict]:
    """Quantitative error rates bucketed by each physical covariate.

    Reports the two product-critical errors:
      false_wearer -- an environment-only frame scored as wearer
      missed_wearer -- a wearer frame scored as not-wearer
    """
    per_frame = scene_meta[scene_index]
    env_only = state == 2
    wearer_any = w_lab == 1
    out: Dict[str, dict] = {}
    for k, key in enumerate(META_KEYS):
        v = per_frame[:, k]
        finite = np.isfinite(v)
        if finite.sum() < 100 or len(np.unique(v[finite])) < n_bins:
            continue
        qs = np.quantile(v[finite], np.linspace(0, 1, n_bins + 1))
        qs[-1] += 1e-6
        buckets = []
        for i in range(n_bins):
            m = finite & (v >= qs[i]) & (v < qs[i + 1])
            if m.sum() < 50:
                continue
            fw = m & env_only
            mw = m & wearer_any
            buckets.append({
                "range": [float(qs[i]), float(qs[i + 1])],
                "n_frames": int(m.sum()),
                "false_wearer_rate": float((w_logit[fw] > 0).mean()) if fw.sum() else None,
                "n_env_only": int(fw.sum()),
                "missed_wearer_rate": float((w_logit[mw] <= 0).mean()) if mw.sum() else None,
                "n_wearer": int(mw.sum()),
                "auroc_within_bucket": _auroc(w_lab[m], w_logit[m]),
            })
        if buckets:
            out[key] = {"bins": buckets}
    return out
