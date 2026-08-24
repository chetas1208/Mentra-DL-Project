#!/usr/bin/env python3
"""P0 evaluation: apples-to-apples comparison of four clip-level score types
on the SAME held-out validation pool (validation_suite_v2.npz +
validation_suite_v2_tir_negatives.npz -- the artifacts the recent V2 autopsy
built and used, held-out validation speakers, zero overlap with training).

Four score types, all computed from ONE consistent pipeline in this script:
  1. S_original     -- SpeakerNet baseline protocol: cosine(enrollment,
                        backbone.encode_speaker() run directly on the
                        window's raw mixed audio). Re-establishes the
                        SpeakerNet-alone reference on THIS pool (the
                        historical 2.13%/4.17%/13.33%/20.83%/29.17% numbers
                        were measured on a different pool, day1_public).
  2. V2_mean_pooled  -- what the existing mentrawearnet_v2_step4800 checkpoint
                        does today: mean(sigmoid(wearer_logit)) over valid
                        frames. Matches the historical
                        scripts/model/evaluate_mentrawearnet.py /
                        v2_analyze.py phase2_clip_eer "mean_wearer_prob"
                        protocol exactly.
  3. S_target_pool   -- the new P0 TargetAwarePooler's window embedding,
                        alone: cosine(enrollment, target-aware pooled embedding).
  4. S_fused         -- the new P0 FusionHead's output, combining
                        S_original, S_target_pool, S_frame (mean dominance
                        logit), and overlap_fraction.

Reuses existing, already-validated utilities rather than reimplementing:
  - training.diagnostics.v2_score_checkpoint.load_clip_list() for the merged
    176-clip list (144 validation_suite_v2 + 32 tir_negatives) with
    clip_is_target labels.
  - training.diagnostics.v2_analyze.compute_eer / safe_auroc /
    frr_at_far_targets for the EER/AUROC/FAR-FRR metrics themselves, and the
    EXACT SAME per-TIR pos/neg slice masks as phase2_clip_eer (so P0's
    numbers are directly comparable to the numbers already on record for
    this checkpoint).

Note: validation_suite_v2 only has TIR levels {+5, 0, -5, -10} (plus
"clean"). The historical day1_public reference additionally reports a
+10dB point -- that point does not exist in this validation pool and is
NOT re-derived here (day1_public is reserved for a single final protocol
run, never used for iteration -- see build_validation_suite.py docstring).

Usage:
    .venv/bin/python3 training/diagnostics/evaluate_p0_target_pooling.py \\
        --p0-checkpoint training/checkpoints/p0_target_pooling_latest.pt \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from training.diagnostics.v2_analyze import compute_eer, frr_at_far_targets, safe_auroc
from training.diagnostics.v2_score_checkpoint import load_clip_list
from training.models.mentrawearnet import MentraWearNet
from training.models.target_aware_pooling import FusionHead, TargetAwarePooler

RESULTS_DIR = Path("evaluation/results")
TIR_LEVELS = (5, 0, -5, -10)

# Historical reference, measured on a DIFFERENT pool (day1_public, 8 speakers)
# -- see scripts/model/evaluate_mentrawearnet.py. Cited verbatim here for
# context, never re-derived by this script (day1_public is reserved for a
# single final held-out comparison, not for iteration).
HISTORICAL_DAY1_PUBLIC_EER_PCT = {"clean": 2.13, "+10": 4.17, "+5": 4.17, "0": 13.33, "-5": 20.83, "-10": 29.17}


@torch.no_grad()
def score_clip_all(frozen_model, pooler, fusion_head, mixture, enrollment, device):
    mixture = np.asarray(mixture, dtype=np.float32)
    enrollment = np.asarray(enrollment, dtype=np.float32)
    mix_t = torch.from_numpy(mixture).unsqueeze(0).float().to(device)
    mix_len = torch.tensor([len(mixture)]).to(device)
    enr_t = torch.from_numpy(enrollment).unsqueeze(0).float().to(device)
    enr_len = torch.tensor([len(enrollment)]).to(device)

    wearer_embedding = frozen_model.encode_enrollment(enr_t, enr_len)  # [1,256]
    h_t, frame_lengths = frozen_model.backbone.encode_frames(mix_t, mix_len)  # [1,1500,T]
    out = frozen_model.process_with_embedding(mix_t, mix_len, wearer_embedding)
    wearer_logits, environment_logits = out["wearer_logits"], out["environment_logits"]

    mixture_speaker_embedding = frozen_model.backbone.encode_speaker(mix_t, mix_len)
    s_original = (wearer_embedding * mixture_speaker_embedding).sum(dim=-1)  # both L2-normalized -> cosine

    T = wearer_logits.shape[1]
    mask = (torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1)).float()
    denom = mask.sum(dim=1).clamp(min=1)
    wp = torch.sigmoid(wearer_logits)
    ep = torch.sigmoid(environment_logits)
    v2_mean_pooled = (wp * mask).sum(dim=1) / denom  # matches historical "mean_wearer_prob"
    d_t = wearer_logits - environment_logits
    s_frame = (d_t * mask).sum(dim=1) / denom
    overlap = ((wp > 0.5) & (ep > 0.5)).float() * mask
    overlap_fraction = overlap.sum(dim=1) / denom

    target_embedding = pooler(h_t, frame_lengths, wearer_embedding,
                               wearer_logits, environment_logits, wp, ep,
                               frozen_model.backbone.embedding_projection)
    s_target_pool = F.cosine_similarity(target_embedding, wearer_embedding, dim=-1)

    fusion_logit = fusion_head(s_original, s_target_pool, s_frame, overlap_fraction)
    s_fused = torch.sigmoid(fusion_logit)

    return {
        "s_original": s_original.item(),
        "v2_mean_pooled": v2_mean_pooled.item(),
        "s_target_pool": s_target_pool.item(),
        "s_fused": s_fused.item(),
    }


def eer_auroc_far_for_bucket(pos_mask, neg_mask, score, far_target=0.05):
    pos = score[pos_mask]
    neg = score[neg_mask]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return {"eer_pct": float("nan"), "auroc": float("nan"), "frr_at_far5pct": float("nan"),
                "n_pos": n_pos, "n_neg": n_neg}
    eer_pct = compute_eer(pos, neg) * 100
    mask = pos_mask | neg_mask
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    scores_subset = np.concatenate([pos, neg])
    auroc = safe_auroc(labels, scores_subset)
    frr = frr_at_far_targets(labels, scores_subset, far_targets=(far_target,))
    return {"eer_pct": eer_pct, "auroc": auroc, "frr_at_far5pct": frr[f"far_{int(far_target*100)}pct"],
            "n_pos": n_pos, "n_neg": n_neg}


def bucket_masks(slice_names: np.ndarray, clip_is_target: np.ndarray):
    """Identical construction to v2_analyze.py's phase2_clip_eer -- same
    slices, same target/impostor logic -- so results are directly comparable
    to the numbers already on record for this checkpoint."""
    buckets = {}
    buckets["clean"] = (
        np.array([s.startswith("clean_solo_wearer") for s in slice_names]) & clip_is_target,
        np.array([s.startswith("clean_solo_environment") for s in slice_names]) & ~clip_is_target,
    )
    for tir in TIR_LEVELS:
        pos = np.array([s in (f"overlap_random_tir{tir:+d}", f"overlap_hardpair_tir{tir:+d}") for s in slice_names]) & clip_is_target
        neg = np.array([s == f"tir_negative_tir{tir:+d}" for s in slice_names]) & ~clip_is_target
        buckets[f"{tir:+d}"] = (pos, neg)
    return buckets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen-checkpoint", default="training/checkpoints/mentrawearnet_v2_step4800.pt")
    ap.add_argument("--p0-checkpoint", default="training/checkpoints/p0_target_pooling_latest.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="p0")
    args = ap.parse_args()

    frozen_model = MentraWearNet().to(args.device)
    fckpt = torch.load(args.frozen_checkpoint, map_location=args.device, weights_only=False)
    frozen_model.load_state_dict(fckpt["model_state_dict"])
    for p in frozen_model.parameters():
        p.requires_grad = False
    frozen_model.eval()
    print(f"loaded frozen {args.frozen_checkpoint} (step={fckpt.get('step')})")

    p0ckpt = torch.load(args.p0_checkpoint, map_location=args.device, weights_only=False)
    p0_config = p0ckpt["config"]
    pooler = TargetAwarePooler(frame_channels=frozen_model.backbone.frame_channels,
                                embedding_dim=frozen_model.backbone.embedding_dim,
                                proj_dim=p0_config["proj_dim"], attn_hidden=p0_config["attn_hidden"]).to(args.device)
    pooler.load_state_dict(p0ckpt["pooler_state_dict"])
    pooler.eval()
    fusion_head = FusionHead().to(args.device)
    fusion_head.load_state_dict(p0ckpt["fusion_head_state_dict"])
    fusion_head.eval()
    print(f"loaded P0 pooler+fusion from {args.p0_checkpoint} "
          f"(step={p0ckpt.get('step')}, final_loss={p0ckpt.get('final_loss')}, "
          f"final_acc={p0ckpt.get('final_acc')}, n_trainable_params={p0ckpt.get('n_trainable_params')})")

    clips = load_clip_list()
    print(f"scoring {len(clips)} clips (validation_suite_v2 + tir_negatives, held-out speakers) ...")

    slice_names, clip_is_target = [], []
    scores = {"s_original": [], "v2_mean_pooled": [], "s_target_pool": [], "s_fused": []}
    for cid, c in enumerate(clips):
        r = score_clip_all(frozen_model, pooler, fusion_head, c["mixture"], c["enrollment"], args.device)
        for k in scores:
            scores[k].append(r[k])
        slice_names.append(c["meta"]["slice"])
        clip_is_target.append(bool(c["meta"].get("clip_is_target", not c["meta"]["is_tir_negative_trial"])))
        if (cid + 1) % 40 == 0:
            print(f"  scored {cid + 1}/{len(clips)}")

    slice_names = np.array(slice_names)
    clip_is_target = np.array(clip_is_target)
    for k in scores:
        scores[k] = np.array(scores[k])

    buckets = bucket_masks(slice_names, clip_is_target)

    results = {"tag": args.tag, "frozen_checkpoint": args.frozen_checkpoint, "p0_checkpoint": args.p0_checkpoint,
               "n_clips": len(clips),
               "historical_day1_public_eer_pct_reference": HISTORICAL_DAY1_PUBLIC_EER_PCT,
               "by_score_type": {}}

    header = f"{'score_type':<18}{'bucket':<8}{'EER%':>8}{'AUROC':>8}{'FRR@FAR5%':>11}{'n_pos':>7}{'n_neg':>7}"
    print("\n" + header)
    print("-" * len(header))
    for score_name, score_arr in scores.items():
        results["by_score_type"][score_name] = {}
        for bucket_name, (pos_mask, neg_mask) in buckets.items():
            r = eer_auroc_far_for_bucket(pos_mask, neg_mask, score_arr)
            results["by_score_type"][score_name][bucket_name] = r
            print(f"{score_name:<18}{bucket_name:<8}{r['eer_pct']:>8.2f}{r['auroc']:>8.4f}"
                  f"{r['frr_at_far5pct']:>11.4f}{r['n_pos']:>7}{r['n_neg']:>7}")
        print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"p0_target_pooling_eval_{args.tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
