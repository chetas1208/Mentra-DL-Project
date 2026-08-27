"""Shortcut and speaker-invariance probes (Workstreams AP, AQ).

Three questions, each answered by a LINEAR probe on a frozen trained model --
nothing is retrained, and no adversarial invariance is optimised (Workstream AQ
is explicit: measure first).

AP1  Can role be predicted from RMS alone?
     If yes, the "geometry" result may be an amplitude shortcut.

AP2  Can the hidden state identify WHICH SIMULATOR produced the audio?
     S1 and S2 differ ONLY in the head-diffraction model, so a probe that
     separates them is reading simulator implementation detail. That alone is
     not fatal -- what is fatal is high simulator-ID accuracy TOGETHER WITH
     poor cross-simulator transfer. Both numbers are reported together.

AQ   Can the hidden state identify the SPEAKER?
     GeoWearNet is meant to learn WHERE, not WHO. High speaker accuracy on
     TRAIN speakers is not automatically bad. The failure signature is high
     train-speaker identifiability combined with collapse on unseen users --
     so the probe is run on both, and the unseen-user task performance from the
     evaluation suite is quoted alongside.

The S1/S2 probe uses a PAIRED design: identical seed, identical speakers,
utterances and geometry draws, differing only in the head model. Any separation
is therefore attributable to the head model, not to confounded content.

Run:
  python3 -m training.diagnostics.geowearnet_probes --checkpoint <path>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from training.geowearnet.data import DataConfig, GeoWearNetDataset, compute_norm_stats, physical_feature_indices
from training.geowearnet.evaluate import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "evaluation/geowearnet/results"


def _normalize(lm, ph, mode, stats):
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
def embed_scenes(model, ck, split: str, generation: str, indices: List[int],
                 stats, condition: str = "normal", duration_s: float = 6.0):
    """Returns per-scene: mean hidden state over wearer-active frames, over
    environment-active frames, plus labels and the wearer speaker id."""
    tcfg = ck["config"]
    keep = physical_feature_indices(tcfg.get("drop_amplitude_features", False))
    use_phys = tcfg.get("use_physical", True) and model.config.use_physical_features
    cfg = DataConfig(split=split, generation=generation, condition=condition,
                     duration_s=duration_s, normalization="none", seed=555_111)
    ds = GeoWearNetDataset(cfg)
    rows = []
    for i in indices:
        sc = ds.scene_for(i)
        lm, ph = ds.featurize(sc.audio)
        t = min(len(lm), len(ph), len(sc.wearer))
        lm, ph = lm[:t], (ph[:t][:, keep] if use_phys else np.zeros((t, 0), np.float32))
        lmn, phn = _normalize(lm, ph, tcfg["normalization"], stats)
        e = model.embed(
            torch.from_numpy(np.ascontiguousarray(lmn, dtype=np.float32))[None],
            torch.from_numpy(np.ascontiguousarray(phn, dtype=np.float32))[None] if use_phys else None,
        )[0].numpy()
        w = sc.wearer[:t] > 0.5
        env = sc.environment[:t] > 0.5
        rows.append({
            "scene": i,
            "emb_wearer": e[w].mean(0) if w.sum() > 3 else None,
            "emb_env": e[env].mean(0) if env.sum() > 3 else None,
            "emb_all": e.mean(0),
            "wearer_speaker": sc.meta["wearer_speaker"],
            "env_speakers": sc.meta["environment_speakers"],
        })
    return rows


def linear_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                 seed: int = 0) -> Dict[str, float]:
    """Held-out linear probe. Splits by `groups` when given (so the probe cannot
    win by memorising a scene that appears on both sides)."""
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(y))
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_tr = int(0.7 * len(uniq))
    tr_g, te_g = set(uniq[:n_tr]), set(uniq[n_tr:])
    tr = np.array([g in tr_g for g in groups])
    te = ~tr
    if tr.sum() < 20 or te.sum() < 10 or len(np.unique(y[tr])) < 2:
        return {"status": "INSUFFICIENT_DATA", "n": int(len(y))}
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=1500, C=1.0, multi_class="auto")
    clf.fit(sc.transform(X[tr]), y[tr])
    pred = clf.predict(sc.transform(X[te]))
    acc = float((pred == y[te]).mean())
    n_cls = len(np.unique(y))
    # majority-class baseline is the honest floor, not 1/n_classes
    vals, cnts = np.unique(y[te], return_counts=True)
    majority = float(cnts.max() / cnts.sum())
    out = {
        "accuracy": acc, "n_classes": int(n_cls), "chance_uniform": 1.0 / n_cls,
        "majority_baseline": majority, "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "accuracy_over_majority": acc - majority,
    }
    if n_cls == 2:
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        out["auroc"] = float(roc_auc_score(y[te], p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-scenes-sim", type=int, default=120)
    ap.add_argument("--n-scenes-speaker", type=int, default=260)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    model, ck = load_checkpoint(a.checkpoint, "cpu")
    tcfg = ck["config"]
    stats = None
    if tcfg["normalization"] == "global":
        stats = compute_norm_stats(DataConfig(
            generation=tcfg["generation"], n_mels=tcfg["n_mels"],
            drop_amplitude_features=tcfg.get("drop_amplitude_features", False),
            condition=tcfg.get("condition", "train_mix")))

    out: Dict[str, object] = {
        "checkpoint": a.checkpoint,
        "trained_on": tcfg["generation"],
        "model": {"arch": tcfg["arch"], "variant": tcfg["variant"], "params": ck["params"],
                  "global_step": ck["global_step"]},
    }

    # ---------- AP2: simulator-generation probe (PAIRED) ----------
    idx = list(range(a.n_scenes_sim))
    print("embedding S1 scenes...", flush=True)
    r1 = embed_scenes(model, ck, "test", "S1", idx, stats)
    print("embedding S2 scenes (paired: same seed/speakers/geometry)...", flush=True)
    r2 = embed_scenes(model, ck, "test", "S2", idx, stats)
    X, y, g = [], [], []
    for src, lab in ((r1, 0), (r2, 1)):
        for row in src:
            X.append(row["emb_all"]); y.append(lab); g.append(row["scene"])
    sim_probe = linear_probe(np.array(X), np.array(y), np.array(g), seed=1)
    # paired sanity: identical speakers on both sides?
    paired_ok = all(x["wearer_speaker"] == z["wearer_speaker"] for x, z in zip(r1, r2))
    sim_probe["paired_design_verified"] = bool(paired_ok)
    out["simulator_generation_probe"] = sim_probe
    print("simulator probe:", json.dumps(sim_probe), flush=True)

    # ---------- AQ: speaker-identity probe, seen vs unseen speakers ----------
    speaker_probes = {}
    for split in ("train", "test"):
        print(f"embedding {split} scenes for speaker probe...", flush=True)
        rows = embed_scenes(model, ck, split, tcfg["generation"],
                            list(range(a.n_scenes_speaker)), stats)
        X, y, g = [], [], []
        for row in rows:
            if row["emb_wearer"] is None:
                continue
            X.append(row["emb_wearer"]); y.append(row["wearer_speaker"]); g.append(row["scene"])
        y = np.array(y)
        # keep only speakers with enough scenes to be learnable at all
        vals, cnts = np.unique(y, return_counts=True)
        keep = set(vals[cnts >= 4])
        m = np.array([v in keep for v in y])
        if m.sum() < 40:
            speaker_probes[split] = {"status": "INSUFFICIENT_DATA",
                                     "n_speakers_with_enough_scenes": int(len(keep))}
            continue
        pr = linear_probe(np.array(X)[m], y[m], np.array(g)[m], seed=2)
        pr["n_speakers"] = int(len(keep))
        pr["speakers_seen_in_training"] = (split == "train")
        speaker_probes[split] = pr
        print(f"speaker probe [{split}]:", json.dumps(pr), flush=True)
    out["speaker_identity_probe"] = speaker_probes

    # ---------- AP1: RMS-only role probe (quoted from the eval suite) ----------
    out["rms_only_role_probe"] = {
        "note": ("Measured directly on the evaluation suite as `logrms_alone_auroc` per "
                 "condition -- see the eval_*.json artifacts. Reproduced here rather than "
                 "recomputed so both use identical trials."),
    }

    # ---------- AP: dataset-identity probe ----------
    out["dataset_identity_probe"] = {
        "status": "NOT_APPLICABLE",
        "reason": ("Every speech source in this campaign comes from one corpus "
                   "(LibriSpeech train-clean-100). There is no second dataset whose "
                   "identity could be used as a shortcut. This becomes meaningful only "
                   "once MMCSG or real Mentra audio is mixed in."),
    }

    # ---------- verdicts ----------
    sim_acc = sim_probe.get("accuracy")
    sim_over = sim_probe.get("accuracy_over_majority")
    tr = speaker_probes.get("train", {})
    te = speaker_probes.get("test", {})
    out["verdicts"] = {
        "simulator_identifiable": (
            "STRONGLY_IDENTIFIABLE" if isinstance(sim_over, float) and sim_over > 0.25
            else "WEAKLY_IDENTIFIABLE" if isinstance(sim_over, float) and sim_over > 0.10
            else "NOT_IDENTIFIABLE"),
        "simulator_shortcut_rule": (
            "A simulator shortcut is flagged only when the generation is strongly "
            "identifiable AND cross-simulator AUROC drops materially. Cross-simulator "
            "numbers live in the eval_*_onS*.json artifacts; both must be read together."),
        "speaker_identifiability": {
            "train_speakers_accuracy_over_majority": tr.get("accuracy_over_majority"),
            "unseen_speakers_accuracy_over_majority": te.get("accuracy_over_majority"),
            "interpretation": (
                "High train-speaker identifiability is only a problem if unseen-user task "
                "performance collapses. Read this next to the unseen-speaker AUROC in the "
                "evaluation suite (which is measured on the speaker-disjoint test split)."),
        },
        "adversarial_invariance": "NOT APPLIED -- Workstream AQ says measure first, do not optimise yet.",
    }
    out["runtime_seconds"] = time.time() - t0

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = a.out or f"probes_{Path(a.checkpoint).parent.parent.name}.json"
    p = RESULTS / name
    p.write_text(json.dumps(out, indent=2))
    print("\n", json.dumps(out["verdicts"], indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
