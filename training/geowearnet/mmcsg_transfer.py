"""Workstream X/AZ -- zero-shot sim-to-real transfer evaluation.

Runs a GeoWearNet checkpoint trained ENTIRELY on the synthetic S1/S2/mixed
simulator directly on real MMCSG recordings (real Aria smart-glasses
hardware, real humans, real rooms), with ZERO fine-tuning and ZERO domain
adaptation. This is the first non-simulated evidence produced anywhere in
this campaign, so the honesty rules matter more than usual here:

  * MMCSG's Aria glasses are NOT Mentra hardware. This is the closest
    available real wearable proxy, not a substitute for the real Mentra pilot
    (Workstreams AA-AD, still blocked on hardware access).
  * The feature pipeline (log-mel, 64 bins, 16 kHz, 25 ms/10 ms; physical
    scalars; normalisation) is IDENTICAL byte-for-byte to training -- reuses
    `acoustics.log_mel`, `features.extract_features` and
    `evaluate._normalize` rather than re-implementing anything, specifically
    so a poor result cannot be blamed on a feature-pipeline mismatch.
  * Normalisation stats are the exact cached training-time stats for that
    checkpoint's generation (`training/geowearnet/cache/norm_stats_*.npz`),
    never recomputed on MMCSG -- recomputing on the real data would leak
    domain-specific statistics into what is supposed to be a zero-shot test.
  * The single raw microphone channel is picked by `channel_study()`
    (measured on the MMCSG train split, never `eval`) rather than assumed.
  * Metrics reuse `evalsuite.core_metrics` / `bootstrap_ci` so numbers are
    directly comparable to the simulated-suite tables in the campaign report,
    with recordings (not frames) as the bootstrap cluster unit -- frames
    inside one conversation share a room, a channel and two people, so they
    are not independent draws.

Run:
    python3 -m training.geowearnet.mmcsg_transfer \\
        --checkpoint training/geowearnet/runs/<run>/checkpoints/best.pt \\
        --root /usr/data/923873155/mmcsg/MMCSG --split eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .acoustics import log_mel
from .data import CACHE_DIR, physical_feature_indices
from .evalsuite import bootstrap_ci, core_metrics
from .features import extract_features
from .mmcsg_adapter import (MMCSGConfig, channel_study, check_available,
                            discover, frame_labels, read_audio, read_transcript)
from .model_zoo import ARCHS, ZooConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "evaluation/geowearnet/results"


def load_checkpoint(path: Path):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    mc = ck["model_config"]
    cfg = ZooConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in mc.items()})
    model = ARCHS[cfg.arch](cfg)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg, ck


def load_norm_stats(ck: dict) -> Optional[Dict[str, np.ndarray]]:
    tc = ck.get("config", {})
    if tc.get("normalization", "global") != "global":
        return None
    gen = tc.get("generation", "S1")
    drop_amp = int(bool(tc.get("drop_amplitude_features", False)))
    p = CACHE_DIR / f"norm_stats_{gen}_64_{drop_amp}_train_mix.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


@torch.no_grad()
def run_recording(model, cfg: ZooConfig, mode: str, stats, audio_mono: np.ndarray):
    from .evaluate import _normalize  # reuse the exact eval-time normalisation

    lm = log_mel(audio_mono, n_mels=cfg.n_mels)
    if cfg.use_physical_features:
        idx = physical_feature_indices(drop_amplitude=False)
        ph = extract_features(audio_mono)[:, idx]
    else:
        ph = np.zeros((len(lm), 0), np.float32)
    t = min(len(lm), len(ph)) if cfg.use_physical_features else len(lm)
    lm, ph = lm[:t], ph[:t]
    lm, ph = _normalize(lm, ph, mode, stats)
    t_lm = torch.from_numpy(np.ascontiguousarray(lm, dtype=np.float32))[None]
    t_ph = (torch.from_numpy(np.ascontiguousarray(ph, dtype=np.float32))[None]
            if cfg.use_physical_features else None)
    out = model(t_lm, t_ph)
    return out["wearer_logits"][0].numpy(), out["environment_logits"][0].numpy(), t


def transfer_eval(checkpoint: Path, root: Path, split: str = "eval",
                  channel: Optional[int] = None, max_recordings: Optional[int] = None,
                  n_boot: int = 200, seed: int = 0) -> Dict[str, object]:
    avail = check_available(root)
    if avail["status"] != "AVAILABLE":
        return {"status": "BLOCKED", "availability": avail}

    model, cfg, ck = load_checkpoint(checkpoint)
    mode = ck.get("config", {}).get("normalization", "global")
    stats = load_norm_stats(ck)
    if mode == "global" and stats is None:
        return {"status": "BLOCKED", "error": "no cached norm_stats for this checkpoint's generation"}

    if channel is None:
        cs = channel_study(root, split="train")  # measured on train, never eval
        channel = cs["recommended_single_channel"]
        channel_source = "measured_on_train_split"
    else:
        cs = None
        channel_source = "user_specified"

    mcfg = MMCSGConfig(root=root, split=split, channels=(channel,))
    recs = discover(mcfg)
    if max_recordings:
        recs = recs[:max_recordings]

    W: List[np.ndarray] = []
    E: List[np.ndarray] = []
    YW: List[np.ndarray] = []
    YE: List[np.ndarray] = []
    SI: List[np.ndarray] = []
    per_recording = []
    for i, rec in enumerate(recs):
        try:
            x, sr = read_audio(mcfg, rec)
        except Exception as ex:
            per_recording.append({"recording_id": rec, "error": repr(ex)})
            continue
        mono = x[:, 0]
        w, e, t = run_recording(model, cfg, mode, stats, mono)
        words = read_transcript(mcfg, rec)
        yw, ye = frame_labels(words, t)
        W.append(w); E.append(e); YW.append(yw); YE.append(ye)
        SI.append(np.full(t, i, dtype=np.int64))
        rec_auroc = float("nan")
        if len(np.unique(yw)) == 2:
            rec_auroc = core_metrics(yw.astype(np.int64), w)["auroc"]
        per_recording.append({"recording_id": rec, "n_frames": int(t), "wearer_auroc": rec_auroc})

    if not W:
        return {"status": "BLOCKED", "error": "no recordings scored"}

    Wc = np.concatenate(W); Ec = np.concatenate(E)
    YWc = np.concatenate(YW).astype(np.int64); YEc = np.concatenate(YE).astype(np.int64)
    SIc = np.concatenate(SI)

    result: Dict[str, object] = {
        "status": "SCORED",
        "checkpoint": str(checkpoint),
        "checkpoint_name": ck.get("config", {}).get("name"),
        "trained_on_generation": ck.get("config", {}).get("generation"),
        "arch": cfg.arch,
        "params": model.count_parameters(),
        "normalization": mode,
        "mmcsg_root": str(root),
        "mmcsg_split": split,
        "channel_used": channel,
        "channel_source": channel_source,
        "n_recordings": len(recs),
        "n_recordings_scored": int(len(W)),
        "n_frames": int(len(Wc)),
        "wearer_positive_rate": float(YWc.mean()),
        "environment_positive_rate": float(YEc.mean()),
    }
    result["wearer"] = core_metrics(YWc, Wc)
    pt, lo, hi = bootstrap_ci(YWc, Wc, SIc, n_boot=n_boot, seed=seed)
    result["wearer"]["auroc_ci95"] = [lo, hi]

    solo = (YWc + YEc) == 1
    if solo.sum() > 50 and len(np.unique(YWc[solo])) == 2:
        score = (Wc - Ec)[solo]
        result["wearer_vs_env_solo"] = core_metrics(YWc[solo], score)
        _, lo2, hi2 = bootstrap_ci(YWc[solo], score, SIc[solo], n_boot=n_boot, seed=seed + 1)
        result["wearer_vs_env_solo"]["auroc_ci95"] = [lo2, hi2]

    aurocs = [r["wearer_auroc"] for r in per_recording if not np.isnan(r.get("wearer_auroc", float("nan")))]
    result["per_recording_auroc_median"] = float(np.median(aurocs)) if aurocs else None
    result["per_recording_auroc_p10_p90"] = ([float(np.percentile(aurocs, 10)), float(np.percentile(aurocs, 90))]
                                             if aurocs else None)
    if cs is not None:
        result["channel_study"] = cs
    result["per_recording"] = per_recording
    result["note"] = ("MMCSG Aria glasses are NOT Mentra hardware -- this is the closest available "
                      "real wearable proxy, not Mentra product validation. Zero fine-tuning, zero "
                      "domain adaptation: this checkpoint has never seen a single real sample.")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="eval", choices=["train", "dev", "eval"])
    ap.add_argument("--channel", type=int, default=None)
    ap.add_argument("--max-recordings", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = transfer_eval(Path(a.checkpoint), Path(a.root), a.split, a.channel,
                        a.max_recordings, a.n_boot, a.seed)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_recording", "channel_study")}, indent=2))

    out_name = a.out or f"geowearnet_mmcsg_transfer_{Path(a.checkpoint).parent.parent.name}_on{a.split}.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / out_name).write_text(json.dumps(res, indent=2))
    print("wrote", RESULTS / out_name)


if __name__ == "__main__":
    main()
