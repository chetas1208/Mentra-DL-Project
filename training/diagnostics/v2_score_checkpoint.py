#!/usr/bin/env python3
"""V2 evaluation autopsy: core scoring engine (phases 2-3 infrastructure).

Loads one MentraWearNet checkpoint, runs it over the fixed validation suite
(evaluation/manifests/validation_suite_v2.npz, 144 clips) PLUS the TIR
negative-verification trials built by build_tir_negative_trials.py (32
clips, evaluation/manifests/validation_suite_v2_tir_negatives.npz) -- all
held-out-speaker, validation-only data (zero overlap with the 211 training
speakers, zero overlap with the day1_public final test set).

For every frame of every clip, records: clip_id, frame_index, true_wearer,
true_environment, true_state, wearer_logit, environment_logit,
wearer_probability, environment_probability, tir, target_speaker (wearer_id),
interferer_speaker, hard_negative flag, slice name, clip_is_target (whether
the enrolled wearer is actually present anywhere in this clip -- used for
clip-level verification EER).

Also computes, per clip, the 15 candidate aggregation-function scores used
by the Phase 12 aggregation sweep, so that sweep can run without re-scoring.

Saves two arrays to a single .npz:
  - frame-level: evaluation/results/mentrawearnet_v2_<tag>_frame_logits.npz
  - the same file also holds the clip-level aggregation table

Usage:
    .venv/bin/python3 training/diagnostics/v2_score_checkpoint.py \\
        --checkpoint training/checkpoints/mentrawearnet_v2_final_5000.pt \\
        --tag step5000 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from training.models.mentrawearnet import MentraWearNet
from training.train import align_labels_to_frames

VAL_SUITE = "evaluation/manifests/validation_suite_v2.npz"
VAL_META = "evaluation/manifests/validation_suite_v2_metadata.json"
TIR_NEG_SUITE = "evaluation/manifests/validation_suite_v2_tir_negatives.npz"
TIR_NEG_META = "evaluation/manifests/validation_suite_v2_tir_negatives_metadata.json"
HARD_PAIRS_VAL = "evaluation/manifests/hard_negative_pairs_val.json"

STATE_NAMES = ["00", "10", "01", "11"]  # index = 2*wearer + env... no: encode below explicitly


def state_code(wearer: np.ndarray, env: np.ndarray) -> np.ndarray:
    """0='00' (silence), 1='10' (wearer-only), 2='01' (environment-only), 3='11' (overlap)."""
    w = (wearer > 0.5).astype(np.int64)
    e = (env > 0.5).astype(np.int64)
    return w * 1 + e * 2  # w=0,e=0->0 ; w=1,e=0->1 ; w=0,e=1->2 ; w=1,e=1->3
    # code 0='00', 1='10', 2='01', 3='11' -- matches STATE_NAMES ordering below


STATE_NAMES = ["00", "10", "01", "11"]


def load_clip_list():
    """Merges validation_suite_v2.npz (144 clips, clip_is_target inferred
    from slice) with the TIR negative trials (32 clips, clip_is_target=False
    by construction). Returns a list of dicts with raw waveforms + meta."""
    clips = []

    d = np.load(VAL_SUITE, allow_pickle=True)
    meta = json.loads(Path(VAL_META).read_text())
    assert len(meta) == len(d["mixtures"])
    for i, m in enumerate(meta):
        slice_name = m["slice"]
        # clip_is_target: does the ENROLLED wearer_id actually speak in this
        # clip? True for every slice except clean_solo_environment_* (content
        # speaker is other_id, not wearer_id).
        clip_is_target = not slice_name.startswith("clean_solo_environment")
        clips.append({
            "mixture": d["mixtures"][i], "enrollment": d["enrollments"][i],
            "wearer_activity": d["wearer_activities"][i], "environment_activity": d["environment_activities"][i],
            "meta": {**m, "clip_is_target": clip_is_target, "is_tir_negative_trial": False},
        })

    d2 = np.load(TIR_NEG_SUITE, allow_pickle=True)
    meta2 = json.loads(Path(TIR_NEG_META).read_text())
    assert len(meta2) == len(d2["mixtures"])
    for i, m in enumerate(meta2):
        clips.append({
            "mixture": d2["mixtures"][i], "enrollment": d2["enrollments"][i],
            "wearer_activity": d2["wearer_activities"][i], "environment_activity": d2["environment_activities"][i],
            "meta": {**m, "is_tir_negative_trial": True},
        })

    return clips


@torch.no_grad()
def score_clip(model, mixture, enrollment, device):
    # object-dtype npz rows can come back as object-dtype 1D arrays (e.g.
    # when every clip in a batch happens to share the same sample length,
    # numpy silently stacks the outer object array into a 2D object array
    # instead of an array-of-arrays, and per-row slices inherit dtype=object)
    # -- force a real float32 array regardless of how it was stored.
    mixture = np.asarray(mixture, dtype=np.float32)
    enrollment = np.asarray(enrollment, dtype=np.float32)
    mix_t = torch.from_numpy(mixture).unsqueeze(0).float().to(device)
    mix_len = torch.tensor([len(mixture)]).to(device)
    enr_t = torch.from_numpy(enrollment).unsqueeze(0).float().to(device)
    enr_len = torch.tensor([len(enrollment)]).to(device)
    wearer_embedding = model.encode_enrollment(enr_t, enr_len)
    out = model.process_with_embedding(mix_t, mix_len, wearer_embedding)
    wearer_logit = out["wearer_logits"][0].cpu().numpy()
    env_logit = out["environment_logits"][0].cpu().numpy()
    return wearer_logit, env_logit


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clip_aggregations(wearer_logit, env_logit, wearer_prob, env_prob,
                       global_wearer_threshold_logit, common_activity_threshold_logit):
    """The 15 candidate clip-level aggregation scores from Phase 12. All
    return a single scalar per clip; "higher = more wearer-like"."""
    dominance = wearer_logit - env_logit
    common = (wearer_logit + env_logit) / 2.0
    n = len(wearer_logit)
    sorted_prob = np.sort(wearer_prob)[::-1]
    sorted_dom = np.sort(dominance)[::-1]
    k10 = max(1, int(np.ceil(0.10 * n)))
    k25 = max(1, int(np.ceil(0.25 * n)))

    # longest continuous wearer-positive run (frame count / n, i.e. fraction of clip)
    pos = wearer_prob > 0.5
    longest = 0
    cur = 0
    for p in pos:
        cur = cur + 1 if p else 0
        longest = max(longest, cur)
    longest_run_frac = longest / n

    speech_active = common > common_activity_threshold_logit
    if speech_active.any():
        speech_gated_mean_logit = wearer_logit[speech_active].mean()
        frac_speech_active_dominant = (dominance[speech_active] > 0).mean()
    else:
        speech_gated_mean_logit = wearer_logit.mean()  # fallback: no frame passed the gate
        frac_speech_active_dominant = 0.0

    lse = np.log(np.mean(np.exp(wearer_logit - wearer_logit.max()))) + wearer_logit.max()  # mean-normalized logsumexp

    return {
        "mean_wearer_prob": wearer_prob.mean(),
        "mean_wearer_logit": wearer_logit.mean(),
        "median_wearer_logit": np.median(wearer_logit),
        "max_wearer_prob": wearer_prob.max(),
        "top10pct_mean_prob": sorted_prob[:k10].mean(),
        "top25pct_mean_prob": sorted_prob[:k25].mean(),
        "p75_wearer_prob": np.percentile(wearer_prob, 75),
        "p90_wearer_prob": np.percentile(wearer_prob, 90),
        "frac_above_calibrated_threshold": (wearer_logit > global_wearer_threshold_logit).mean(),
        "longest_run_frac": longest_run_frac,
        "logsumexp_wearer_logit": lse,
        "speech_gated_mean_wearer_logit": speech_gated_mean_logit,
        "mean_dominance_logit": dominance.mean(),
        "top25pct_dominance_logit": sorted_dom[:k25].mean(),
        "frac_speech_active_dominant": frac_speech_active_dominant,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True, help="short label used in output filenames, e.g. step4600")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="evaluation/results")
    args = ap.parse_args()

    model = MentraWearNet().to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded {args.checkpoint} (step={ckpt.get('step')}, final_loss={ckpt.get('final_loss')})")

    hard_pairs = json.loads(Path(HARD_PAIRS_VAL).read_text())

    clips = load_clip_list()
    print(f"scoring {len(clips)} clips ({sum(1 for c in clips if not c['meta']['is_tir_negative_trial'])} val_suite + "
          f"{sum(1 for c in clips if c['meta']['is_tir_negative_trial'])} tir_negative) ...")

    # --- pass 1: score every clip, collect frame-level arrays ---
    frame_rows = []  # list of dict-of-scalars (will vectorize after)
    clip_records = []
    all_wearer_logit_frames = []  # for computing global thresholds afterward
    all_common_frames = []
    all_true_wearer_frames = []

    per_clip_cache = []
    for cid, c in enumerate(clips):
        wearer_logit, env_logit = score_clip(model, c["mixture"], c["enrollment"], args.device)
        target_frames = len(wearer_logit)
        w_true = align_labels_to_frames(c["wearer_activity"], target_frames)
        e_true = align_labels_to_frames(c["environment_activity"], target_frames)
        wearer_prob = sigmoid(wearer_logit)
        env_prob = sigmoid(env_logit)
        per_clip_cache.append((cid, wearer_logit, env_logit, wearer_prob, env_prob, w_true, e_true))
        all_wearer_logit_frames.append(wearer_logit)
        all_common_frames.append((wearer_logit + env_logit) / 2.0)
        all_true_wearer_frames.append(w_true)
        if (cid + 1) % 40 == 0:
            print(f"  scored {cid+1}/{len(clips)} clips")

    # --- global validation-only calibration constants (Youden-J thresholds) ---
    all_wl = np.concatenate(all_wearer_logit_frames)
    all_common = np.concatenate(all_common_frames)
    all_wt = np.concatenate(all_true_wearer_frames)

    def youden_threshold(scores, labels):
        order = np.argsort(scores)
        thresholds = np.unique(scores)
        best_j, best_t = -1.0, float(np.median(scores))
        # subsample thresholds for speed if huge
        cand = thresholds if len(thresholds) < 2000 else np.quantile(thresholds, np.linspace(0, 1, 2000))
        for t in cand:
            pred = scores > t
            tp = np.sum((labels > 0.5) & pred)
            fn = np.sum((labels > 0.5) & ~pred)
            fp = np.sum((labels <= 0.5) & pred)
            tn = np.sum((labels <= 0.5) & ~pred)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            j = tpr - fpr
            if j > best_j:
                best_j, best_t = j, t
        return best_t

    global_wearer_threshold_logit = youden_threshold(all_wl, all_wt)
    # speech-vs-silence label for common-activity threshold: speech = wearer OR env true (frame-level, reconstruct is_speech per frame)
    all_et = np.concatenate([align_labels_to_frames(c["environment_activity"], len(wl))
                              for c, wl in zip(clips, all_wearer_logit_frames)])
    all_is_speech = ((all_wt > 0.5) | (all_et > 0.5)).astype(np.float32)
    common_activity_threshold_logit = youden_threshold(all_common, all_is_speech)
    print(f"global wearer-logit Youden threshold: {global_wearer_threshold_logit:.4f}")
    print(f"common-activity-logit Youden threshold (speech vs silence): {common_activity_threshold_logit:.4f}")

    # --- pass 2: build frame rows + clip aggregation table using the calibrated thresholds ---
    for cid, wearer_logit, env_logit, wearer_prob, env_prob, w_true, e_true in per_clip_cache:
        c = clips[cid]
        m = c["meta"]
        n = len(wearer_logit)
        state = state_code(w_true, e_true)
        other_id = m.get("other_id")
        hn = False
        if other_id is not None:
            hn = other_id in hard_pairs.get(m["wearer_id"], [])
        for fi in range(n):
            frame_rows.append((
                cid, fi, w_true[fi], e_true[fi], state[fi],
                wearer_logit[fi], env_logit[fi], wearer_prob[fi], env_prob[fi],
                m.get("tir_db", np.nan) if m.get("tir_db") is not None else np.nan,
                m["wearer_id"], other_id if other_id is not None else "",
                hn, m["slice"], m["is_tir_negative_trial"], m.get("noisy", False),
                m.get("duration_s", np.nan),
            ))
        agg = clip_aggregations(wearer_logit, env_logit, wearer_prob, env_prob,
                                 global_wearer_threshold_logit, common_activity_threshold_logit)
        clip_records.append({
            "clip_id": cid, "slice": m["slice"], "tir_db": m.get("tir_db", np.nan),
            "wearer_id": m["wearer_id"], "other_id": other_id,
            "clip_is_target": m.get("clip_is_target", not m["is_tir_negative_trial"]),
            "duration_s": m.get("duration_s", np.nan), "noisy": m.get("noisy", False),
            "n_frames": n, "is_tir_negative_trial": m["is_tir_negative_trial"],
            **agg,
        })

    # --- save frame-level as structured npz (columnar) ---
    fr = frame_rows
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = out_dir / f"mentrawearnet_v2_{args.tag}_frame_logits.npz"
    np.savez(
        frame_path,
        clip_id=np.array([r[0] for r in fr], dtype=np.int32),
        frame_index=np.array([r[1] for r in fr], dtype=np.int32),
        true_wearer=np.array([r[2] for r in fr], dtype=np.float32),
        true_environment=np.array([r[3] for r in fr], dtype=np.float32),
        true_state=np.array([r[4] for r in fr], dtype=np.int32),
        wearer_logit=np.array([r[5] for r in fr], dtype=np.float32),
        environment_logit=np.array([r[6] for r in fr], dtype=np.float32),
        wearer_probability=np.array([r[7] for r in fr], dtype=np.float32),
        environment_probability=np.array([r[8] for r in fr], dtype=np.float32),
        tir=np.array([r[9] for r in fr], dtype=np.float32),
        target_speaker=np.array([r[10] for r in fr], dtype=object),
        interferer_speaker=np.array([r[11] for r in fr], dtype=object),
        hard_negative=np.array([r[12] for r in fr], dtype=bool),
        slice_name=np.array([r[13] for r in fr], dtype=object),
        is_tir_negative_trial=np.array([r[14] for r in fr], dtype=bool),
        noisy=np.array([r[15] for r in fr], dtype=bool),
        duration_s=np.array([r[16] for r in fr], dtype=np.float32),
        state_names=np.array(STATE_NAMES, dtype=object),
        global_wearer_threshold_logit=np.array([global_wearer_threshold_logit], dtype=np.float32),
        common_activity_threshold_logit=np.array([common_activity_threshold_logit], dtype=np.float32),
    )
    print(f"saved frame-level data: {frame_path} ({len(fr)} frames)")

    clip_path = out_dir / f"mentrawearnet_v2_{args.tag}_clip_scores.npz"
    keys = list(clip_records[0].keys())
    arrs = {}
    for k in keys:
        vals = [r[k] for r in clip_records]
        if isinstance(vals[0], (str,)) or vals[0] is None:
            arrs[k] = np.array([v if v is not None else "" for v in vals], dtype=object)
        elif isinstance(vals[0], (bool, np.bool_)):
            arrs[k] = np.array(vals, dtype=bool)
        else:
            arrs[k] = np.array(vals, dtype=np.float64)
    np.savez(clip_path, **arrs)
    print(f"saved clip-level aggregation table: {clip_path} ({len(clip_records)} clips)")

    meta_summary = {
        "checkpoint": args.checkpoint, "step": ckpt.get("step"), "final_loss": ckpt.get("final_loss"),
        "n_clips": len(clips), "n_frames": len(fr),
        "global_wearer_threshold_logit": float(global_wearer_threshold_logit),
        "common_activity_threshold_logit": float(common_activity_threshold_logit),
    }
    (out_dir / f"mentrawearnet_v2_{args.tag}_score_summary.json").write_text(json.dumps(meta_summary, indent=2))
    print(json.dumps(meta_summary, indent=2))


if __name__ == "__main__":
    main()
