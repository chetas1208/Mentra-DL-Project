#!/usr/bin/env python3
"""M0 Phase 0: fix the O0-vs-O1/O2 selection-bias comparison from R0's
multi-segment oracle-pooling supplement BEFORE trusting it, and implement an
evaluation-only Oracle Hierarchical (OH) pooling policy that never drops a
trial.

R0 (training/diagnostics/r0_information_loss_audit.py,
run_multisegment_oracle_supplement) found O0/O1/O2 EER on a 240-trial
multi-segment supplement, but O1/O2 EXCLUDED any trial whose oracle mask had
< MIN_FRAMES_FOR_MASK (10 frames = 100ms) frames -- 48/240 trials excluded
for O1, 82/240 for O2 (see r0_information_loss_audit.json). Comparing O0's
full-pool EER against O1/O2's excluded-subset EER is not apples-to-apples:
O1/O2 only "win" on the (easier?) subset of trials that had enough
oracle-mask duration in the first place.

This script:
  1. Regenerates the IDENTICAL 240-trial multi-segment supplement (same
     seed=20260826, same val_pool, same examples_per_tir=40,
     examples_clean=80, total_s=2.0 -- byte-identical trial construction to
     R0's run_multisegment_oracle_supplement), but keeps PER-TRIAL records
     (mask frame counts + pos/neg scores for O0/O1/O2) instead of only
     aggregated EER tables.
  2. Matched-subset check: O0 recomputed restricted to exactly the trial
     subset where O1 was valid (>=MIN_FRAMES_FOR_MASK frames), vs O1 on that
     same subset. Same for O2.
  3. OH (Oracle Hierarchical): per trial, if oracle solo-wearer (state=10)
     frame count >= solo_thr_frames, use O2-style pooling; elif oracle
     target-active (state in {10,11}, i.e. wt=1) frame count >=
     active_thr_frames, use O1-style pooling; else fall back to O0. NEVER
     excludes a trial. Threshold pair swept over {200,400,600,1000}ms x
     {200,400,600,1000}ms, selected on a calibration half (even trial index)
     by minimizing mean EER% across TIR buckets, then reported on the held-
     out eval half (odd trial index) with zero exclusions.
  4. Decision gate: if OH's held-out full-set EER shows no meaningful
     improvement over O0 on the same held-out half, STOP -- M0 (Phase 3+)
     should not be run.

Zero training of the main model. Reuses r0_information_loss_audit.py's
already-verified forward_with_ladder / masked_stats_pool / embed_from_mask /
fingerprint / verify_ladder_equivalence and v2_frame_dataset.py's
build_clean_example / build_overlap_bucket_example / state_code.

Run: .venv/bin/python3 training/diagnostics/phase0_oracle_bias_correction.py --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from training.data.mixture_generator import SpeakerPool
from training.diagnostics.r0_information_loss_audit import (
    FROZEN_CKPT, MIN_FRAMES_FOR_MASK, TIR_LEVELS, TIR_TAGS, DISPLAY_TAGS,
    embed_from_mask, fingerprint, verify_ladder_equivalence,
)
from training.diagnostics.v2_analyze import compute_eer, safe_auroc
from training.diagnostics.v2_frame_dataset import build_clean_example, build_overlap_bucket_example
from training.models.mentrawearnet import MentraWearNet
from training.train import align_labels_to_frames

VAL_MANIFEST = "evaluation/manifests/librispeech_train_clean_100_val.json"
RESULTS_DIR = Path("evaluation/results")
DURATION_CANDIDATES_MS = (200, 400, 600, 1000)
FRAMES_PER_MS = 0.1  # 10ms/frame -> 1 frame = 10ms -> frames = ms * 0.1


def ms_to_frames(ms: int) -> int:
    return max(1, round(ms * FRAMES_PER_MS))


@torch.no_grad()
def generate_supplement_detailed(model: MentraWearNet, pool: SpeakerPool, device: str,
                                  examples_per_tir: int, examples_clean: int, total_s: float, seed: int):
    """Byte-identical trial construction to R0's
    run_multisegment_oracle_supplement (same rng draw order), but records
    PER-TRIAL mask frame counts and O0/O1/O2 pos/neg scores instead of only
    an aggregated EER table."""
    rng = random.Random(seed)
    trials = []  # list of dicts

    def process(mixture, w_act, e_act, wearer_id, other_id, tir_tag, trial_idx):
        mix_t = torch.from_numpy(np.asarray(mixture, dtype=np.float32)).unsqueeze(0).to(device)
        mix_len = torch.tensor([len(mixture)], device=device)
        frame_features, frame_lengths = model.backbone.encode_frames(mix_t, mix_len)
        T = frame_features.shape[-1]
        length_mask = (torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1))
        wt = torch.from_numpy(align_labels_to_frames(w_act, T)).to(device).bool()
        et = torch.from_numpy(align_labels_to_frames(e_act, T)).to(device).bool()
        masks = {
            "O0": length_mask,                     # native, all valid frames
            "O1": length_mask & wt,                # target-active: state in {10,11}
            "O2": length_mask & wt & ~et,           # solo-wearer: state == 10 only
        }
        counts = {k: int(m.sum().item()) for k, m in masks.items()}

        pos_enr = pool.enrollment_clip(wearer_id, rng)
        impostor_candidates = [s for s in pool.speaker_ids if s not in (wearer_id, other_id)]
        impostor_id = rng.choice(impostor_candidates)
        neg_enr = pool.enrollment_clip(impostor_id, rng)

        def enr_embedding(enr):
            e_t = torch.from_numpy(np.asarray(enr, dtype=np.float32)).unsqueeze(0).to(device)
            e_len = torch.tensor([len(enr)], device=device)
            return model.encode_enrollment(e_t, e_len)

        pos_emb = enr_embedding(pos_enr)
        neg_emb = enr_embedding(neg_enr)

        scores_pos, scores_neg, valid = {}, {}, {}
        for k, mask in masks.items():
            if counts[k] < 1:
                scores_pos[k] = scores_neg[k] = float("nan")
                valid[k] = False
                continue
            emb = embed_from_mask(model, frame_features, mask)
            scores_pos[k] = F.cosine_similarity(emb, pos_emb, dim=-1).item()
            scores_neg[k] = F.cosine_similarity(emb, neg_emb, dim=-1).item()
            valid[k] = counts[k] >= MIN_FRAMES_FOR_MASK

        trials.append({
            "idx": trial_idx, "tir_tag": tir_tag, "counts": counts,
            "scores_pos": scores_pos, "scores_neg": scores_neg, "valid": valid,
        })

    idx = 0
    for _ in range(examples_clean):
        mixture, enrollment, w_act, e_act, wearer_id, other_id = build_clean_example(pool, rng, total_s)
        process(mixture, w_act, e_act, wearer_id, other_id, "clean", idx)
        idx += 1
    for tir in TIR_LEVELS:
        for _ in range(examples_per_tir):
            mixture, enrollment, w_act, e_act, wearer_id, other_id = build_overlap_bucket_example(pool, rng, tir, total_s)
            process(mixture, w_act, e_act, wearer_id, other_id, f"{tir:+d}", idx)
            idx += 1
    return trials


def eer_for_subset(trials, key, subset_pred, tir_tag=None):
    pos, neg = [], []
    for t in trials:
        if tir_tag is not None and t["tir_tag"] != tir_tag:
            continue
        if not subset_pred(t):
            continue
        sp, sn = t["scores_pos"][key], t["scores_neg"][key]
        if np.isnan(sp) or np.isnan(sn):
            continue
        pos.append(sp)
        neg.append(sn)
    if len(pos) < 2 or len(neg) < 2:
        return {"eer_pct": float("nan"), "auroc": float("nan"), "n": len(pos)}
    pos_a, neg_a = np.array(pos), np.array(neg)
    eer = compute_eer(pos_a, neg_a) * 100
    labels = np.concatenate([np.ones(len(pos_a)), np.zeros(len(neg_a))])
    scores = np.concatenate([pos_a, neg_a])
    auroc = safe_auroc(labels, scores)
    return {"eer_pct": eer, "auroc": auroc, "n": len(pos_a)}


def oh_score(trial, solo_thr_frames, active_thr_frames):
    """Returns (pos_score, neg_score, which) for OH's per-trial choice.
    NEVER returns NaN as long as O0 is valid (O0 is length_mask -> valid for
    any clip with >=1 frame, which every generated clip has)."""
    c = trial["counts"]
    if c["O2"] >= solo_thr_frames and not np.isnan(trial["scores_pos"]["O2"]):
        return trial["scores_pos"]["O2"], trial["scores_neg"]["O2"], "O2"
    if c["O1"] >= active_thr_frames and not np.isnan(trial["scores_pos"]["O1"]):
        return trial["scores_pos"]["O1"], trial["scores_neg"]["O1"], "O1"
    return trial["scores_pos"]["O0"], trial["scores_neg"]["O0"], "O0"


def eer_for_oh(trials, solo_thr_frames, active_thr_frames, tir_tag=None):
    pos, neg, which_count = [], [], {"O0": 0, "O1": 0, "O2": 0}
    for t in trials:
        if tir_tag is not None and t["tir_tag"] != tir_tag:
            continue
        sp, sn, which = oh_score(t, solo_thr_frames, active_thr_frames)
        which_count[which] += 1
        pos.append(sp)
        neg.append(sn)
    if len(pos) < 2 or len(neg) < 2:
        return {"eer_pct": float("nan"), "auroc": float("nan"), "n": len(pos), "which_count": which_count}
    pos_a, neg_a = np.array(pos), np.array(neg)
    eer = compute_eer(pos_a, neg_a) * 100
    labels = np.concatenate([np.ones(len(pos_a)), np.zeros(len(neg_a))])
    scores = np.concatenate([pos_a, neg_a])
    auroc = safe_auroc(labels, scores)
    return {"eer_pct": eer, "auroc": auroc, "n": len(pos_a), "which_count": which_count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--examples-per-tir", type=int, default=40)
    ap.add_argument("--examples-clean", type=int, default=80)
    ap.add_argument("--total-s", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--out", default="evaluation/results/phase0_oracle_bias_correction.json")
    args = ap.parse_args()
    device = args.device
    t0 = time.time()
    results = {}

    print("=" * 70)
    print("PHASE 0: oracle matched-subset check + OH hierarchical pooling")
    model = MentraWearNet().to(device)
    ckpt = torch.load(FROZEN_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    fp_before = fingerprint(model)
    print(f"loaded {FROZEN_CKPT} (step={ckpt.get('step')})")
    print(f"fingerprint before: {fp_before}")

    ladder_ok = verify_ladder_equivalence(model, device)
    assert ladder_ok
    results["ladder_equivalence_test_pass"] = ladder_ok

    print(f"\nloading VAL manifest pool ({VAL_MANIFEST}) ...")
    val_pool = SpeakerPool(VAL_MANIFEST)
    print(f"  {len(val_pool.speaker_ids)} validation speakers")

    print(f"\ngenerating {args.examples_clean} clean + {args.examples_per_tir}x4 TIR multi-segment "
          f"trials (seed={args.seed}, IDENTICAL construction to R0's supplement) ...")
    trials = generate_supplement_detailed(model, val_pool, device, args.examples_per_tir,
                                           args.examples_clean, args.total_s, args.seed)
    print(f"  -> {len(trials)} trials")

    # ---------------- matched-subset check ----------------
    print("\n" + "=" * 70)
    print("MATCHED-SUBSET CHECK (O0 vs O1, O0 vs O2, restricted to the SAME valid-trial subset)")
    header = "  " + f"{'':<18}" + "".join(f"{DISPLAY_TAGS[t]:>10}" for t in TIR_TAGS) + f"{'ALL':>10}"
    print(header)

    matched = {"O1": {}, "O2": {}}
    coverage = {"O1": {}, "O2": {}}
    for k in ("O1", "O2"):
        valid_pred = lambda t, k=k: t["valid"][k]
        row_o0 = f"  {'O0 (on ' + k + '-valid)':<18}"
        row_ok = f"  {k + ' (native subset)':<18}"
        n_valid_total, n_total = 0, 0
        for tag in list(TIR_TAGS) + [None]:
            label = tag if tag is not None else "ALL"
            r_o0 = eer_for_subset(trials, "O0", valid_pred, tag)
            r_ok = eer_for_subset(trials, k, valid_pred, tag)
            matched[k][label] = {"O0_on_subset": r_o0, k: r_ok}
            if tag is not None:
                row_o0 += f"{r_o0['eer_pct']:>10.2f}"
                row_ok += f"{r_ok['eer_pct']:>10.2f}"
            n_tag_total = sum(1 for t in trials if (tag is None or t["tir_tag"] == tag))
            n_tag_valid = sum(1 for t in trials if (tag is None or t["tir_tag"] == tag) and t["valid"][k])
            coverage[k][label] = {"n_valid": n_tag_valid, "n_total": n_tag_total,
                                   "frac_valid": n_tag_valid / n_tag_total if n_tag_total else float("nan")}
        r_o0_all = eer_for_subset(trials, "O0", valid_pred, None)
        r_ok_all = eer_for_subset(trials, k, valid_pred, None)
        row_o0 += f"{r_o0_all['eer_pct']:>10.2f}"
        row_ok += f"{r_ok_all['eer_pct']:>10.2f}"
        print(row_o0)
        print(row_ok)
        print(f"  coverage ({k}-valid): " + ", ".join(
            f"{tag}={coverage[k][tag]['n_valid']}/{coverage[k][tag]['n_total']}"
            f"({coverage[k][tag]['frac_valid']*100:.0f}%)" for tag in DISPLAY_TAGS))
        print()

    results["matched_subset_check"] = matched
    results["coverage"] = coverage

    # ---------------- OH: calibration/eval split ----------------
    print("=" * 70)
    print("OH (Oracle Hierarchical): calibration (even idx) / eval (odd idx) split, "
          "threshold grid selected on calibration, reported on held-out eval, ZERO exclusions")
    calib_trials = [t for t in trials if t["idx"] % 2 == 0]
    eval_trials = [t for t in trials if t["idx"] % 2 == 1]
    print(f"  n_calib={len(calib_trials)}  n_eval={len(eval_trials)}")

    best_combo, best_mean_eer = None, float("inf")
    grid_results = []
    for solo_ms in DURATION_CANDIDATES_MS:
        for active_ms in DURATION_CANDIDATES_MS:
            solo_f, active_f = ms_to_frames(solo_ms), ms_to_frames(active_ms)
            per_tag_eer = []
            for tag in TIR_TAGS:
                r = eer_for_oh(calib_trials, solo_f, active_f, tag)
                if not np.isnan(r["eer_pct"]):
                    per_tag_eer.append(r["eer_pct"])
            mean_eer = float(np.mean(per_tag_eer)) if per_tag_eer else float("inf")
            grid_results.append({"solo_ms": solo_ms, "active_ms": active_ms, "calib_mean_eer_pct": mean_eer})
            if mean_eer < best_mean_eer:
                best_mean_eer, best_combo = mean_eer, (solo_ms, active_ms, solo_f, active_f)
    print(f"  best combo on calibration: solo_thr={best_combo[0]}ms active_thr={best_combo[1]}ms "
          f"(calib mean EER%={best_mean_eer:.2f})")
    results["oh_threshold_grid"] = grid_results
    results["oh_best_combo_ms"] = {"solo_ms": best_combo[0], "active_ms": best_combo[1]}

    solo_f, active_f = best_combo[2], best_combo[3]
    oh_eval_table = {}
    print(f"\n  OH full-set (held-out eval half, n={len(eval_trials)}, zero exclusions) EER%:")
    print(header.replace("ALL", "ALL"))
    row_oh = f"  {'OH':<18}"
    row_o0_eval = f"  {'O0 (same eval half)':<18}"
    for tag in TIR_TAGS:
        r_oh = eer_for_oh(eval_trials, solo_f, active_f, tag)
        r_o0 = eer_for_subset(eval_trials, "O0", lambda t: True, tag)
        oh_eval_table[tag] = {"OH": r_oh, "O0": r_o0}
        row_oh += f"{r_oh['eer_pct']:>10.2f}"
        row_o0_eval += f"{r_o0['eer_pct']:>10.2f}"
    r_oh_all = eer_for_oh(eval_trials, solo_f, active_f, None)
    r_o0_all = eer_for_subset(eval_trials, "O0", lambda t: True, None)
    oh_eval_table["ALL"] = {"OH": r_oh_all, "O0": r_o0_all}
    row_oh += f"{r_oh_all['eer_pct']:>10.2f}"
    row_o0_eval += f"{r_o0_all['eer_pct']:>10.2f}"
    print(row_oh)
    print(row_o0_eval)
    print(f"  OH which-mask-used counts (ALL, eval half): {r_oh_all['which_count']}")

    results["oh_eval_holdout_table"] = oh_eval_table

    # also report OH + O0 on the FULL 240-trial set (both halves) for a
    # larger-n headline number, clearly labeled as NOT threshold-selection-
    # independent (thresholds were chosen on the calib half only, so this is
    # still leakage-free w.r.t. threshold selection, just larger n than the
    # strict eval-half-only table above).
    oh_full_table = {}
    print(f"\n  OH full-set (ALL {len(trials)} trials, thresholds still calib-only-selected) EER%:")
    row_oh_f = f"  {'OH':<18}"
    row_o0_f = f"  {'O0':<18}"
    for tag in TIR_TAGS:
        r_oh = eer_for_oh(trials, solo_f, active_f, tag)
        r_o0 = eer_for_subset(trials, "O0", lambda t: True, tag)
        oh_full_table[tag] = {"OH": r_oh, "O0": r_o0}
        row_oh_f += f"{r_oh['eer_pct']:>10.2f}"
        row_o0_f += f"{r_o0['eer_pct']:>10.2f}"
    r_oh_all_f = eer_for_oh(trials, solo_f, active_f, None)
    r_o0_all_f = eer_for_subset(trials, "O0", lambda t: True, None)
    oh_full_table["ALL"] = {"OH": r_oh_all_f, "O0": r_o0_all_f}
    row_oh_f += f"{r_oh_all_f['eer_pct']:>10.2f}"
    row_o0_f += f"{r_o0_all_f['eer_pct']:>10.2f}"
    print(row_oh_f)
    print(row_o0_f)
    print(f"  OH which-mask-used counts (ALL {len(trials)} trials): {r_oh_all_f['which_count']}")
    results["oh_full_240_table"] = oh_full_table

    # ---------------- decision gate ----------------
    print("\n" + "=" * 70)
    print("DECISION GATE")
    improvements = []
    for tag in TIR_TAGS:
        o0e = oh_eval_table[tag]["O0"]["eer_pct"]
        ohe = oh_eval_table[tag]["OH"]["eer_pct"]
        if not (np.isnan(o0e) or np.isnan(ohe)):
            improvements.append(o0e - ohe)
            print(f"  {DISPLAY_TAGS[tag]:>6}: O0={o0e:.2f}%  OH={ohe:.2f}%  delta={o0e - ohe:+.2f}pp")
    mean_improvement = float(np.mean(improvements)) if improvements else float("nan")
    n_improved = sum(1 for d in improvements if d > 0)
    print(f"  mean improvement (O0-OH) across {len(improvements)} TIR buckets (eval half): "
          f"{mean_improvement:+.2f}pp,  buckets improved: {n_improved}/{len(improvements)}")

    gate_pass = mean_improvement > 1.0 and n_improved >= (len(improvements) // 2 + 1)
    results["decision_gate"] = {
        "mean_improvement_pp_eval_half": mean_improvement,
        "n_buckets_improved": n_improved,
        "n_buckets_total": len(improvements),
        "gate_pass": bool(gate_pass),
    }
    print(f"\n  GATE: {'PASS -- proceed to Phase 1+' if gate_pass else 'FAIL -- STOP, do not train M0'}")

    fp_after = fingerprint(model)
    results["fingerprint_before"] = fp_before
    results["fingerprint_after"] = fp_after
    results["fingerprint_match"] = (fp_before == fp_after)
    assert fp_after == fp_before, "FROZEN MODEL FINGERPRINT CHANGED"
    print(f"\nfingerprint after: {fp_after}  match={fp_before == fp_after}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    print(f"\nsaved {out_path}")
    print(f"total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
