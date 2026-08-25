#!/usr/bin/env python3
"""V3C-MA0: adapter behavior + speaker-drift safety diagnostics.

For a trained STATIC_ADAPTER or MIXTURE_AWARE checkpoint, generates the
same clean/TIR-bucketed examples as v3a_frame_dataset.py (same generator,
same held-out validation speakers), runs each through the model, and
records per-frame:
  - gate (mean/std over the E=256 adapter output dims)
  - ||delta_t|| (L2 norm over E)
  - cos(e_t, e) -- how far the adapted embedding has moved from the static
    enrollment embedding, in direction terms
  - for state-01 (environment-only) frames specifically: cos(e_t, e) [target
    similarity] vs cos(e_t, e_other) [environment-speaker similarity, where
    e_other = model.encode_enrollment() run on the INTERFERER's own audio --
    same backbone/encode_speaker path, same embedding space, just a
    different speaker's reference clip]. This is the speaker-drift safety
    test: does the adapted conditioning vector drift toward whoever's
    actually talking (bad) or stay anchored to the enrolled wearer (safe)?

Aggregates by true 4-state (00/10/01/11) and by TIR bucket
(clean/tir+5/tir+0/tir-5/tir-10), matching v3a_analyze.py's bucket
convention so tables line up with the rest of the report.

Output: evaluation/results/v3c_ma0_<tag>_drift_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from training.models.mentrawearnet import MentraWearNet
from training.train import align_labels_to_frames
from training.diagnostics.v2_frame_dataset import (
    build_overlap_bucket_example, build_clean_example, state_code,
    VAL_MANIFEST, GLOBAL_SEED, TIR_LEVELS,
)
from training.data.mixture_generator import SpeakerPool


@torch.no_grad()
def score_one(model, mixture, enrollment, other_enrollment, device):
    mix_t = torch.from_numpy(np.asarray(mixture, dtype=np.float32)).unsqueeze(0).to(device)
    mix_len = torch.tensor([len(mixture)]).to(device)
    enr_t = torch.from_numpy(np.asarray(enrollment, dtype=np.float32)).unsqueeze(0).to(device)
    enr_len = torch.tensor([len(enrollment)]).to(device)
    other_t = torch.from_numpy(np.asarray(other_enrollment, dtype=np.float32)).unsqueeze(0).to(device)
    other_len = torch.tensor([len(other_enrollment)]).to(device)

    wearer_embedding = model.encode_enrollment(enr_t, enr_len)          # [1, E] -- e (target/wearer)
    other_embedding = model.encode_enrollment(other_t, other_len)        # [1, E] -- environment speaker's own identity

    out = model.process_with_embedding(mix_t, mix_len, wearer_embedding)
    e_t = out["adapted_embedding"][0]        # [T, E]
    e = out["static_embedding"][0]            # [E]
    gate = out["adapter_gate"][0]             # [T, E]
    delta = out["adapter_delta"][0]           # [T, E]
    e_other = other_embedding[0]              # [E]

    e_t_n = torch.nn.functional.normalize(e_t, dim=-1)
    e_n = torch.nn.functional.normalize(e, dim=0)
    e_other_n = torch.nn.functional.normalize(e_other, dim=0)

    cos_e_t_e = (e_t_n @ e_n).cpu().numpy()               # [T] -- target similarity
    cos_e_t_other = (e_t_n @ e_other_n).cpu().numpy()      # [T] -- environment-speaker similarity
    gate_mean_t = gate.mean(dim=-1).cpu().numpy()          # [T]
    gate_std_t = gate.std(dim=-1).cpu().numpy()            # [T]
    delta_norm_t = delta.norm(dim=-1).cpu().numpy()        # [T]

    return cos_e_t_e, cos_e_t_other, gate_mean_t, gate_std_t, delta_norm_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adapter-mode", required=True, choices=["static", "mixture"])
    ap.add_argument("--adapter-hidden-dim", type=int, default=32)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--examples-per-tir", type=int, default=40)
    ap.add_argument("--examples-clean", type=int, default=80)
    ap.add_argument("--total-s", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--out-dir", default="evaluation/results")
    args = ap.parse_args()

    model = MentraWearNet(aux_losses=True, adapter_mode=args.adapter_mode,
                           adapter_hidden_dim=args.adapter_hidden_dim).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"loaded {args.checkpoint} (step={ckpt.get('step')})  missing={missing}  unexpected={unexpected}")
    assert missing == [] and unexpected == [], "adapter checkpoint should load with NO missing/unexpected keys"
    model.eval()

    pool = SpeakerPool(VAL_MANIFEST)
    rng = random.Random(args.seed)

    rows_state = []   # (true_state, bucket, cos_e_t_e, cos_e_t_other or nan, gate_mean, gate_std, delta_norm)

    def process(mixture, enrollment, w_act, e_act, wearer_id, other_id, bucket):
        other_enrollment = pool.enrollment_clip(other_id, rng)
        cos_te, cos_to, gm, gs, dn = score_one(model, mixture, enrollment, other_enrollment, args.device)
        n = len(cos_te)
        wt = align_labels_to_frames(w_act, n)
        et = align_labels_to_frames(e_act, n)
        st = state_code(wt, et)
        for i in range(n):
            rows_state.append((int(st[i]), bucket, float(cos_te[i]), float(cos_to[i]),
                                float(gm[i]), float(gs[i]), float(dn[i])))

    for _ in range(args.examples_clean):
        mixture, enrollment, w_act, e_act, wearer_id, other_id = build_clean_example(pool, rng, args.total_s)
        process(mixture, enrollment, w_act, e_act, wearer_id, other_id, "clean")

    for tir in TIR_LEVELS:
        for _ in range(args.examples_per_tir):
            mixture, enrollment, w_act, e_act, wearer_id, other_id = build_overlap_bucket_example(pool, rng, tir, args.total_s)
            process(mixture, enrollment, w_act, e_act, wearer_id, other_id, f"tir{tir:+d}")
        print(f"  tir={tir}: done")

    rows = np.array(rows_state, dtype=[
        ("state", "i4"), ("bucket", "U10"), ("cos_e_t_e", "f4"), ("cos_e_t_other", "f4"),
        ("gate_mean", "f4"), ("gate_std", "f4"), ("delta_norm", "f4"),
    ])

    def describe(x):
        if len(x) == 0:
            return {"n": 0}
        return {"n": int(len(x)), "mean": float(np.mean(x)), "std": float(np.std(x)),
                "median": float(np.median(x))}

    STATE_NAMES = ["00", "10", "01", "11"]
    by_state = {}
    for code, name in enumerate(STATE_NAMES):
        mask = rows["state"] == code
        by_state[name] = {
            "gate_mean": describe(rows["gate_mean"][mask]),
            "gate_std": describe(rows["gate_std"][mask]),
            "delta_norm": describe(rows["delta_norm"][mask]),
            "cos_e_t_e": describe(rows["cos_e_t_e"][mask]),
        }

    buckets = ["clean"] + [f"tir{t:+d}" for t in TIR_LEVELS]
    by_bucket = {}
    for b in buckets:
        mask = rows["bucket"] == b
        by_bucket[b] = {
            "gate_mean": describe(rows["gate_mean"][mask]),
            "delta_norm": describe(rows["delta_norm"][mask]),
            "cos_e_t_e": describe(rows["cos_e_t_e"][mask]),
        }

    # THE speaker-drift safety test: state-01 (environment-only) frames only.
    mask01 = rows["state"] == 2  # STATE_NAMES index 2 == "01"
    target_sim = rows["cos_e_t_e"][mask01]
    env_sim = rows["cos_e_t_other"][mask01]
    margin = target_sim - env_sim  # positive = stays anchored to wearer (safe), negative = drifts to interferer (FAILURE)
    drift_report = {
        "n_state01_frames": int(mask01.sum()),
        "target_similarity_cos_e_t_e": describe(target_sim),
        "environment_similarity_cos_e_t_other": describe(env_sim),
        "margin_target_minus_environment": describe(margin),
        "frac_frames_drifted_toward_interferer": float((margin < 0).mean()) if len(margin) else float("nan"),
    }

    result = {
        "tag": args.tag, "checkpoint": args.checkpoint, "adapter_mode": args.adapter_mode,
        "source_step": ckpt.get("step"),
        "by_state": by_state,
        "by_bucket": by_bucket,
        "speaker_drift_state01": drift_report,
        "n_total_frames": int(len(rows)),
    }
    out_path = Path(args.out_dir) / f"v3c_ma0_{args.tag}_drift_diagnostics.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved {out_path}")
    print(json.dumps(drift_report, indent=2))


if __name__ == "__main__":
    main()
