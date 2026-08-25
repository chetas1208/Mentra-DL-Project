#!/usr/bin/env python3
"""V3A evaluation: canonical joint-logit frame dataset, extending
v2_frame_dataset.py (same held-out validation speakers, same synthetic
SILENCE/WEARER/ENVIRONMENT/OVERLAP construction, same seed/example counts by
default) so V3A's frame metrics are directly comparable to V2's already-saved
mentrawearnet_v2_<tag>_analysis.json numbers -- apples to apples, same
protocol, only the checkpoint (and therefore the model's aux_losses=True
construction, needed to load a --aux-losses checkpoint) differs.

Additions over v2_frame_dataset.py (V3A is specifically about these):
  - captures out["similarity"] -- the raw frame-speaker/enrollment cosine
    metric branch (frame_speaker_projection / enrollment_similarity_projection
    fused via einsum in process_with_embedding()), the actual thing V3A's
    --speaker-disc-loss was trying to move off chance. This tensor is
    ALWAYS returned regardless of aux_losses (see mentrawearnet.py), so this
    diagnostic is directly comparable to what V2/M0's checkpoint would have
    produced too (R0 already measured that baseline at ~0.45-0.51 AUROC,
    cited in the final report rather than recomputed here).
  - captures four_state_logits softmax probabilities (only present when
    aux_losses=True, i.e. only for V3A-style checkpoints) for the
    solo-target (class 1 = wearer-only) precision/recall/F1 diagnostic.

Output: evaluation/results/mentrawearnet_v3a_<tag>_joint_frame_dataset.npz
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
import torch.nn.functional as F

from training.data.mixture_generator import (
    SAMPLE_RATE, SAMPLES_PER_FRAME, SpeakerPool, _fit_or_loop, _rms,
)
from training.models.mentrawearnet import MentraWearNet
from training.train import align_labels_to_frames
from training.diagnostics.v2_frame_dataset import (
    build_overlap_bucket_example, build_clean_example, state_code, sigmoid,
    VAL_MANIFEST, HARD_PAIRS_VAL, GLOBAL_SEED, TIR_LEVELS, STATE_NAMES,
)


@torch.no_grad()
def score(model, mixture, enrollment, device):
    mix_t = torch.from_numpy(np.asarray(mixture, dtype=np.float32)).unsqueeze(0).to(device)
    mix_len = torch.tensor([len(mixture)]).to(device)
    enr_t = torch.from_numpy(np.asarray(enrollment, dtype=np.float32)).unsqueeze(0).to(device)
    enr_len = torch.tensor([len(enrollment)]).to(device)
    wearer_embedding = model.encode_enrollment(enr_t, enr_len)
    out = model.process_with_embedding(mix_t, mix_len, wearer_embedding)
    wl = out["wearer_logits"][0].cpu().numpy()
    el = out["environment_logits"][0].cpu().numpy()
    sim = out["similarity"][0].cpu().numpy()
    four_state_prob = None
    if "four_state_logits" in out:
        four_state_prob = F.softmax(out["four_state_logits"][0], dim=0).cpu().numpy()  # [4, T]
    return wl, el, sim, four_state_prob


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--examples-per-tir", type=int, default=40)
    ap.add_argument("--examples-clean", type=int, default=80)
    ap.add_argument("--total-s", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--out-dir", default="evaluation/results")
    ap.add_argument("--adapter-mode", default=None, choices=["static", "mixture"],
                     help="V3C-MA0: build the model with a MixtureAwareEnrollmentAdapter. "
                          "Omit (default) for exact prior behavior.")
    ap.add_argument("--adapter-hidden-dim", type=int, default=32)
    args = ap.parse_args()

    model = MentraWearNet(aux_losses=True, adapter_mode=args.adapter_mode,
                           adapter_hidden_dim=args.adapter_hidden_dim).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"loaded {args.checkpoint} (step={ckpt.get('step')})  missing={missing}  unexpected={unexpected}")
    model.eval()

    pool = SpeakerPool(VAL_MANIFEST)
    hard_pairs = json.loads(Path(HARD_PAIRS_VAL).read_text())
    rng = random.Random(args.seed)

    rows = []
    clip_id = 0

    for _ in range(args.examples_clean):
        mixture, enrollment, w_act, e_act, wearer_id, other_id = build_clean_example(pool, rng, args.total_s)
        wl, el, sim, fsp = score(model, mixture, enrollment, args.device)
        n = len(wl)
        wt = align_labels_to_frames(w_act, n)
        et = align_labels_to_frames(e_act, n)
        st = state_code(wt, et)
        hn = other_id in hard_pairs.get(wearer_id, [])
        wp, ep = sigmoid(wl), sigmoid(el)
        for fi in range(n):
            rows.append((clip_id, fi, wt[fi], et[fi], st[fi], wl[fi], el[fi], wp[fi], ep[fi],
                         np.nan, wearer_id, other_id, hn, "clean", False, sim[fi],
                         float(fsp[1, fi]) if fsp is not None else np.nan,
                         int(np.argmax(fsp[:, fi])) if fsp is not None else -1))
        clip_id += 1

    for tir in TIR_LEVELS:
        for _ in range(args.examples_per_tir):
            mixture, enrollment, w_act, e_act, wearer_id, other_id = build_overlap_bucket_example(pool, rng, tir, args.total_s)
            wl, el, sim, fsp = score(model, mixture, enrollment, args.device)
            n = len(wl)
            wt = align_labels_to_frames(w_act, n)
            et = align_labels_to_frames(e_act, n)
            st = state_code(wt, et)
            hn = other_id in hard_pairs.get(wearer_id, [])
            wp, ep = sigmoid(wl), sigmoid(el)
            for fi in range(n):
                rows.append((clip_id, fi, wt[fi], et[fi], st[fi], wl[fi], el[fi], wp[fi], ep[fi],
                             float(tir), wearer_id, other_id, hn, f"tir{tir:+d}", False, sim[fi],
                             float(fsp[1, fi]) if fsp is not None else np.nan,
                             int(np.argmax(fsp[:, fi])) if fsp is not None else -1))
            clip_id += 1
        print(f"  tir={tir}: done ({clip_id} clips so far)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mentrawearnet_v3a_{args.tag}_joint_frame_dataset.npz"
    np.savez(
        out_path,
        clip_id=np.array([r[0] for r in rows], dtype=np.int32),
        frame_index=np.array([r[1] for r in rows], dtype=np.int32),
        true_wearer=np.array([r[2] for r in rows], dtype=np.float32),
        true_environment=np.array([r[3] for r in rows], dtype=np.float32),
        true_state=np.array([r[4] for r in rows], dtype=np.int32),
        wearer_logit=np.array([r[5] for r in rows], dtype=np.float32),
        environment_logit=np.array([r[6] for r in rows], dtype=np.float32),
        wearer_probability=np.array([r[7] for r in rows], dtype=np.float32),
        environment_probability=np.array([r[8] for r in rows], dtype=np.float32),
        tir=np.array([r[9] for r in rows], dtype=np.float32),
        target_speaker=np.array([r[10] for r in rows], dtype=object),
        interferer_speaker=np.array([r[11] for r in rows], dtype=object),
        hard_negative=np.array([r[12] for r in rows], dtype=bool),
        bucket=np.array([r[13] for r in rows], dtype=object),
        is_tir_negative_trial=np.array([r[14] for r in rows], dtype=bool),
        similarity=np.array([r[15] for r in rows], dtype=np.float32),
        four_state_solo_prob=np.array([r[16] for r in rows], dtype=np.float32),
        four_state_argmax=np.array([r[17] for r in rows], dtype=np.int32),
        state_names=np.array(STATE_NAMES, dtype=object),
        n_clips=np.array([clip_id]),
    )
    print(f"saved {out_path} ({len(rows)} frames across {clip_id} clips)")


if __name__ == "__main__":
    main()
