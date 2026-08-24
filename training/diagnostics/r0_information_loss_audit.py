#!/usr/bin/env python3
"""R0: pure diagnostic representation-ladder + oracle/predicted masked-pooling
audit for the frozen mentrawearnet_v2_step4800 checkpoint.

ZERO training of the main model. The checkpoint is loaded, .eval()'d once,
requires_grad=False on every existing parameter, and never touched again.
Only tiny throwaway linear/1-hidden-layer probes (section 3) are trained,
clearly separate from the frozen model's own parameters.

Sections (see docstring at top of the task spec this implements):
  1. freeze + SHA-256 fingerprint (params AND buffers) before/after
  2. representation-ladder extraction (R0..R8) via a wrapper function that
     duplicates MentraWearNet.process_with_embedding()'s math tap-by-tap --
     mentrawearnet.py is NOT modified. Verified bit-identical against the
     real process_with_embedding() call before trusting any downstream number.
  3. frozen linear/1-hidden-layer probes at every stage R0..R8, 10-vs-01
     frame classification, AUROC/AUPRC/balanced-accuracy per TIR bucket
  4. oracle masked SpeakerNet statistics pooling (O0..O3) on the SAME
     validation pool P0 used (validation_suite_v2 + tir_negatives)
  5. predicted-mask pooling (P0..P3) using the frozen V2 checkpoint's own
     frame outputs, thresholds selected on a calibration split only
  6. frame-level similarity-branch audit (best-effort)

Run: .venv/bin/python3 training/diagnostics/r0_information_loss_audit.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from training.data.mixture_generator import NoisePool, SpeakerPool
from training.diagnostics.evaluate_p0_target_pooling import bucket_masks, eer_auroc_far_for_bucket
from training.diagnostics.v2_analyze import compute_eer, safe_auprc, safe_auroc
from training.diagnostics.v2_frame_dataset import build_clean_example, build_overlap_bucket_example, state_code
from training.diagnostics.v2_score_checkpoint import load_clip_list
from training.models.mentrawearnet import MentraWearNet
from training.train import align_labels_to_frames

TRAIN_MANIFEST = "evaluation/manifests/librispeech_train_clean_100_train.json"
VAL_MANIFEST = "evaluation/manifests/librispeech_train_clean_100_val.json"
FROZEN_CKPT = "training/checkpoints/mentrawearnet_v2_step4800.pt"
RESULTS_DIR = Path("evaluation/results")
TIR_LEVELS = (5, 0, -5, -10)
# Internal tag keys match bucket_masks()'s f"{tir:+d}" convention exactly
# (tir=0 -> "+0", not "0") -- MUST match or every "0dB" bucket silently
# goes empty/NaN. DISPLAY_TAGS is only for pretty-printing "0" instead of "+0".
TIR_TAGS = ("clean", "+5", "+0", "-5", "-10")
DISPLAY_TAGS = {"clean": "clean", "+5": "+5", "+0": "0", "-5": "-5", "-10": "-10"}
STAGE_NAMES = [f"R{i}" for i in range(9)]  # R0..R8 (R9 handled separately, no probe)
MIN_FRAMES_FOR_MASK = 10  # ~100ms @ ~10ms/frame -- clips below this are marked invalid, not degenerate-scored

HISTORICAL_P0_S_ORIGINAL_EER = {"clean": 8.33, "+5": 12.50, "+0": 21.88, "-5": 25.00, "-10": 25.00}


# --------------------------------------------------------------------------
# 1. Fingerprint
# --------------------------------------------------------------------------
def fingerprint(model: nn.Module) -> str:
    """SHA-256 over every entry of state_dict() -- state_dict() includes both
    parameters AND buffers (BatchNorm running_mean/running_var/
    num_batches_tracked), which is exactly the historical bug class this
    project already hit once (buffer drift under a stray .train() call, not
    weight drift) -- see speakernet_backbone.py's SpeakerNetBackbone.train()
    docstring. Hashing name+bytes for every tensor in deterministic
    (sorted-by-name) order."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# 2. Representation ladder (read-only tap extraction, no mentrawearnet.py edits)
# --------------------------------------------------------------------------
@torch.no_grad()
def forward_with_ladder(model: MentraWearNet, waveform: torch.Tensor, lengths: torch.Tensor,
                         wearer_embedding: torch.Tensor):
    """Line-for-line duplicate of MentraWearNet.process_with_embedding()'s
    math (see training/models/mentrawearnet.py), with a tap recorded after
    every stage. Returns (taps: dict[str, Tensor[B,D,T]], wearer_logits,
    environment_logits, frame_lengths, similarity[B,T]).

    Taps: R0=raw backbone frame_features [B,1500,T], R1=post frame_projection
    [B,192,T], R2=post-FiLM, R3=post-similarity-fusion, R4..R8=TCN block
    1-5 outputs (self.tcn is nn.Sequential of 5 CausalDepthwiseSeparableBlock
    -- iterated manually here, one tap per block)."""
    frame_features, frame_lengths = model.backbone.encode_frames(waveform, lengths)  # R0 [B,1500,T]
    h = model.frame_projection(frame_features)  # R1 [B,D,T]

    e_proj = model.enrollment_projection(wearer_embedding)  # [B,D]
    h_conditioned = model.film(h, e_proj)  # R2 [B,D,T]

    frame_speaker = model.frame_speaker_projection(h_conditioned)
    enrollment_sim_space = model.enrollment_similarity_projection(e_proj)
    enrollment_sim_space = F.normalize(enrollment_sim_space, p=2, dim=-1)
    frame_speaker_norm = F.normalize(frame_speaker, p=2, dim=1)
    similarity = torch.einsum("bet,be->bt", frame_speaker_norm, enrollment_sim_space)  # [B,T]
    similarity_ch = similarity.unsqueeze(1)  # [B,1,T]

    fused = model.similarity_fusion(torch.cat([h_conditioned, similarity_ch], dim=1))  # R3 [B,D,T]

    taps = {"R0": frame_features, "R1": h, "R2": h_conditioned, "R3": fused}
    x = fused
    for i, block in enumerate(model.tcn):
        x = block(x)
        taps[f"R{4 + i}"] = x  # R4..R8
    temporal = x

    wearer_logits = model.wearer_head(temporal).squeeze(1)
    environment_logits = model.environment_head(temporal).squeeze(1)

    return taps, wearer_logits, environment_logits, frame_lengths, similarity


@torch.no_grad()
def verify_ladder_equivalence(model: MentraWearNet, device: str) -> bool:
    """TEST: forward_with_ladder() must reproduce process_with_embedding()'s
    wearer_logits/environment_logits/similarity bit-for-bit (same ops, same
    eval-mode module state, no dropout since model.eval() disables it,
    deterministic on a fixed input) -- this is what lets us trust every
    downstream ladder number without having touched mentrawearnet.py."""
    torch.manual_seed(0)
    waveform = torch.randn(2, 32000, device=device) * 0.05
    lengths = torch.tensor([32000, 24000], device=device)
    enr = torch.randn(2, 16000, device=device) * 0.05
    enr_len = torch.tensor([16000, 16000], device=device)
    wearer_embedding = model.encode_enrollment(enr, enr_len)

    out = model.process_with_embedding(waveform, lengths, wearer_embedding)
    taps, wl2, el2, fl2, sim2 = forward_with_ladder(model, waveform, lengths, wearer_embedding)

    ok = (torch.allclose(out["wearer_logits"], wl2, atol=0, rtol=0) and
          torch.allclose(out["environment_logits"], el2, atol=0, rtol=0) and
          torch.equal(out["frame_lengths"], fl2) and
          torch.allclose(out["similarity"], sim2, atol=0, rtol=0))
    print(f"[TEST] forward_with_ladder equivalence to process_with_embedding: "
          f"{'PASS (bit-identical)' if ok else 'FAIL'}")
    if not ok:
        wl_diff = (out["wearer_logits"] - wl2).abs().max().item()
        print(f"  max wearer_logits diff: {wl_diff}")
    return ok


# --------------------------------------------------------------------------
# 4/5. Masked SpeakerNet statistics pooling
# --------------------------------------------------------------------------
def masked_stats_pool(frame_features: torch.Tensor, keep_mask: torch.Tensor,
                       unbiased: bool = True, eps: float = 1e-10) -> torch.Tensor:
    """Reimplements NeMo StatsPoolLayer's masked branch (pool_mode='xvector':
    mean+std over valid frames), generalized from a prefix-length mask to an
    ARBITRARY boolean frame mask -- backbone.pooling itself only accepts a
    `length` (prefix count), not an arbitrary mask, so encode_speaker()
    can't be reused directly for masked subsets (see module docstring).
    frame_features: [B,1500,T]. keep_mask: [B,T] bool. Returns [B,3000]."""
    m = keep_mask.float().unsqueeze(1)  # [B,1,T]
    n = m.sum(dim=-1).clamp(min=1)  # [B,1]
    mean = (frame_features * m).sum(dim=-1) / n  # [B,1500]
    correction = 1 if unbiased else 0
    centered = (frame_features - mean.unsqueeze(-1)) * keep_mask.unsqueeze(1).float()
    var = centered.pow(2).sum(dim=-1) / (n - correction).clamp(min=1)
    std = var.clamp(min=eps).sqrt()
    return torch.cat([mean, std], dim=-1)  # [B,3000]


def embed_from_mask(model: MentraWearNet, frame_features: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    pooled = masked_stats_pool(frame_features, keep_mask)
    emb = model.backbone.embedding_projection(pooled)
    return F.normalize(emb, p=2, dim=-1)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    thresholds = np.unique(scores)
    if len(thresholds) > 2000:
        thresholds = np.quantile(thresholds, np.linspace(0, 1, 2000))
    best_j, best_t = -1.0, float(np.median(scores))
    labels_pos = labels > 0.5
    for t in thresholds:
        pred = scores > t
        tp = np.sum(labels_pos & pred)
        fn = np.sum(labels_pos & ~pred)
        fp = np.sum(~labels_pos & pred)
        tn = np.sum(~labels_pos & ~pred)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------
# 3. Probe dataset generation
# --------------------------------------------------------------------------
@torch.no_grad()
def generate_ladder_dataset(model: MentraWearNet, pool: SpeakerPool, device: str,
                             examples_per_tir: int, examples_clean: int, total_s: float, seed: int):
    """Generates SILENCE/WEARER/ENVIRONMENT/OVERLAP clips (build_clean_example
    for the 'clean' bucket, build_overlap_bucket_example per TIR level --
    reused verbatim from v2_frame_dataset.py, the established convention that
    produced the historical '10-vs-01 AUROC 0.78-0.87 across TIR' number),
    runs the frozen model with the ladder tap extractor, and returns:
      - stage_X: dict[str, list[np.ndarray[n_frames_kept, D]]] for R0..R8,
        one row per KEPT frame (state 10 or 01 only)
      - y: np.ndarray[N] int (1=wearer/10, 0=environment/01), for kept frames
      - tir_tag: np.ndarray[N] str, per kept frame
      - r9_dominance: np.ndarray[N] float32 (wearer_logit - environment_logit at that frame), for kept frames
      - full_state_records: list of (state[T], similarity[T], tir_tag) for
        EVERY frame of every clip (section 6 -- similarity audit needs 00/11 too)
    """
    rng = random.Random(seed)
    stage_X = {s: [] for s in STAGE_NAMES}
    y_list, tir_list, dom_list = [], [], []
    full_state_records = []

    def process_clip(mixture, enrollment, w_act, e_act, tir_tag):
        mix_t = torch.from_numpy(np.asarray(mixture, dtype=np.float32)).unsqueeze(0).to(device)
        mix_len = torch.tensor([len(mixture)], device=device)
        enr_t = torch.from_numpy(np.asarray(enrollment, dtype=np.float32)).unsqueeze(0).to(device)
        enr_len = torch.tensor([len(enrollment)], device=device)
        wearer_embedding = model.encode_enrollment(enr_t, enr_len)
        taps, wl, el, frame_lengths, similarity = forward_with_ladder(model, mix_t, mix_len, wearer_embedding)
        T = wl.shape[1]
        wt = align_labels_to_frames(w_act, T)
        et = align_labels_to_frames(e_act, T)
        state = state_code(wt, et)  # 0..3
        keep = (state == 1) | (state == 2)
        wl_np = wl[0].cpu().numpy()
        el_np = el[0].cpu().numpy()
        sim_np = similarity[0].cpu().numpy()

        full_state_records.append((state.copy(), sim_np.copy(), tir_tag))

        if keep.sum() == 0:
            return
        keep_idx = np.where(keep)[0]
        for stage in STAGE_NAMES:
            feat = taps[stage][0].transpose(0, 1).cpu().numpy()  # [T, D]
            stage_X[stage].append(feat[keep_idx])
        y_list.append((state[keep_idx] == 1).astype(np.int64))
        tir_list.append(np.array([tir_tag] * len(keep_idx)))
        dom_list.append((wl_np - el_np)[keep_idx])

    # clean bucket
    for _ in range(examples_clean):
        mixture, enrollment, w_act, e_act, wid, oid = build_clean_example(pool, rng, total_s)
        process_clip(mixture, enrollment, w_act, e_act, "clean")

    # TIR buckets
    for tir in TIR_LEVELS:
        for _ in range(examples_per_tir):
            mixture, enrollment, w_act, e_act, wid, oid = build_overlap_bucket_example(pool, rng, tir, total_s)
            process_clip(mixture, enrollment, w_act, e_act, f"{tir:+d}")

    for stage in STAGE_NAMES:
        stage_X[stage] = np.concatenate(stage_X[stage], axis=0)
    y = np.concatenate(y_list, axis=0)
    tir_tag = np.concatenate(tir_list, axis=0)
    dominance = np.concatenate(dom_list, axis=0)
    return stage_X, y, tir_tag, dominance, full_state_records


# --------------------------------------------------------------------------
# 3. Probe training
# --------------------------------------------------------------------------
def train_probe(Xtr: np.ndarray, ytr: np.ndarray, device: str, epochs: int = 30,
                 batch_size: int = 2048, hidden: int = 64, lr: float = 1e-3, seed: int = 0):
    """Tiny 1-hidden-layer probe, IDENTICAL architecture (Linear(D,64) ->
    ReLU -> Linear(64,1)) across every stage -- only D (the input dim) varies
    naturally with the representation being probed. THROWAWAY diagnostic
    artifact, not a model improvement; never touches the frozen model."""
    torch.manual_seed(seed)
    D = Xtr.shape[1]
    mu = Xtr.mean(axis=0, keepdims=True)
    sigma = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xn = (Xtr - mu) / sigma

    probe = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    Xn_t = torch.from_numpy(Xn.astype(np.float32)).to(device)
    y_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
    n = Xn_t.shape[0]
    pos_weight = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = probe(Xn_t[idx]).squeeze(-1)
            loss = loss_fn(logits, y_t[idx])
            loss.backward()
            opt.step()
    probe.eval()
    return probe, mu, sigma


@torch.no_grad()
def probe_predict(probe, mu, sigma, X: np.ndarray, device: str) -> np.ndarray:
    Xn = (X - mu) / sigma
    Xn_t = torch.from_numpy(Xn.astype(np.float32)).to(device)
    logits = probe(Xn_t).squeeze(-1)
    return torch.sigmoid(logits).cpu().numpy()


def metrics_by_tir(y_true: np.ndarray, scores: np.ndarray, tir_tag: np.ndarray) -> dict:
    out = {}
    for tag in TIR_TAGS:
        m = tir_tag == tag
        if m.sum() < 2 or len(np.unique(y_true[m])) < 2:
            out[tag] = {"auroc": float("nan"), "auprc": float("nan"), "bal_acc": float("nan"), "n": int(m.sum())}
            continue
        yt, sc = y_true[m], scores[m]
        auroc = safe_auroc(yt, sc)
        auprc = safe_auprc(yt, sc)
        bal_acc = float(balanced_accuracy_score(yt, (sc > 0.5).astype(int)))
        out[tag] = {"auroc": auroc, "auprc": auprc, "bal_acc": bal_acc, "n": int(m.sum())}
    return out


# --------------------------------------------------------------------------
# 4b. Supplementary oracle pooling on multi-segment clips (see note in main():
# validation_suite_v2's TIR-overlap POSITIVE trials are 100% state=11 for
# their FULL duration by construction (make_overlap_example sets both
# wearer_activity and environment_activity to all-ones for the whole clip --
# same fact v2_frame_dataset.py's docstring already flags). That means O2
# (true_wearer=1 AND true_environment=0) is mathematically EMPTY for every
# TIR-level positive trial in that pool -- there are no solo-wearer frames to
# select AT ALL, not a "too few frames" edge case but an exact zero by
# construction. And O1 is then identical to O0 for every clip in that pool
# (whole-clip masks always coincide with the single state that clip actually
# has). This is reported honestly in the REQUIRED table below. To actually
# test "does restricting pooling to solo-wearer frames help", this
# supplementary experiment reuses the SAME multi-segment SILENCE/WEARER/
# ENVIRONMENT/OVERLAP generator (build_clean_example/build_overlap_bucket_example,
# same VAL_MANIFEST held-out speakers, same convention already used for the
# probe eval set) which DOES produce genuine within-clip mixes of solo-wearer
# and overlap frames. Builds standard positive/negative verification trials
# (same mixture audio, enrollment = true wearer vs a random impostor) so a
# real clip-level EER is computable per mask.
# --------------------------------------------------------------------------
@torch.no_grad()
def run_multisegment_oracle_supplement(model: MentraWearNet, pool: SpeakerPool, device: str,
                                        examples_per_tir: int, examples_clean: int, total_s: float, seed: int):
    rng = random.Random(seed)
    scores_pos = {k: {tag: [] for tag in TIR_TAGS} for k in ("O0", "O1", "O2", "O3")}
    scores_neg = {k: {tag: [] for tag in TIR_TAGS} for k in ("O0", "O1", "O2", "O3")}
    excluded = {k: 0 for k in ("O0", "O1", "O2", "O3")}
    total_trials = 0

    def process(mixture, w_act, e_act, wearer_id, other_id, tir_tag):
        nonlocal total_trials
        mix_t = torch.from_numpy(np.asarray(mixture, dtype=np.float32)).unsqueeze(0).to(device)
        mix_len = torch.tensor([len(mixture)], device=device)
        frame_features, frame_lengths = model.backbone.encode_frames(mix_t, mix_len)
        T = frame_features.shape[-1]
        length_mask = (torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1))
        wt = torch.from_numpy(align_labels_to_frames(w_act, T)).to(device).bool()
        et = torch.from_numpy(align_labels_to_frames(e_act, T)).to(device).bool()
        masks = {"O0": length_mask, "O1": length_mask & wt, "O2": length_mask & wt & ~et,
                 "O3": length_mask & (wt | et)}

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
        total_trials += 1
        for k, mask in masks.items():
            if int(mask.sum().item()) < MIN_FRAMES_FOR_MASK:
                excluded[k] += 1
                continue
            emb = embed_from_mask(model, frame_features, mask)
            scores_pos[k][tir_tag].append(F.cosine_similarity(emb, pos_emb, dim=-1).item())
            scores_neg[k][tir_tag].append(F.cosine_similarity(emb, neg_emb, dim=-1).item())

    for _ in range(examples_clean):
        mixture, enrollment, w_act, e_act, wearer_id, other_id = build_clean_example(pool, rng, total_s)
        process(mixture, w_act, e_act, wearer_id, other_id, "clean")
    for tir in TIR_LEVELS:
        for _ in range(examples_per_tir):
            mixture, enrollment, w_act, e_act, wearer_id, other_id = build_overlap_bucket_example(pool, rng, tir, total_s)
            process(mixture, w_act, e_act, wearer_id, other_id, f"{tir:+d}")

    table = {}
    for k in ("O0", "O1", "O2", "O3"):
        table[k] = {}
        for tag in TIR_TAGS:
            pos = np.array(scores_pos[k][tag])
            neg = np.array(scores_neg[k][tag])
            if len(pos) < 2 or len(neg) < 2:
                table[k][tag] = {"eer_pct": float("nan"), "n_pos": len(pos), "n_neg": len(neg)}
                continue
            table[k][tag] = {"eer_pct": compute_eer(pos, neg) * 100, "n_pos": len(pos), "n_neg": len(neg)}
    return table, excluded, total_trials


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train-examples-per-tir", type=int, default=60)
    ap.add_argument("--train-examples-clean", type=int, default=100)
    ap.add_argument("--eval-examples-per-tir", type=int, default=40)
    ap.add_argument("--eval-examples-clean", type=int, default=80)
    ap.add_argument("--total-s", type=float, default=2.0)
    ap.add_argument("--out", default="evaluation/results/r0_information_loss_audit.json")
    args = ap.parse_args()
    device = args.device
    t0 = time.time()

    results = {}

    # ---------------- 1. Freeze + fingerprint ----------------
    print("=" * 70)
    print("SECTION 1: freeze + fingerprint")
    model = MentraWearNet().to(device)
    ckpt = torch.load(FROZEN_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    fp_before = fingerprint(model)
    print(f"loaded {FROZEN_CKPT} (step={ckpt.get('step')})")
    print(f"fingerprint (before diagnostic work): {fp_before}")

    # ---------------- 2. verify ladder equivalence ----------------
    print("\nSECTION 2: representation ladder equivalence test")
    ladder_ok = verify_ladder_equivalence(model, device)
    results["ladder_equivalence_test_pass"] = ladder_ok
    assert ladder_ok, "forward_with_ladder does not match process_with_embedding -- aborting"

    # ---------------- 3. probe datasets ----------------
    print("\nSECTION 3: generating probe train/eval datasets")
    print(f"  loading TRAIN manifest pool ({TRAIN_MANIFEST}) for probe-fitting data ...")
    train_pool = SpeakerPool(TRAIN_MANIFEST)
    print(f"  {len(train_pool.speaker_ids)} training speakers")
    print(f"  loading VAL manifest pool ({VAL_MANIFEST}) for held-out probe eval data ...")
    val_pool = SpeakerPool(VAL_MANIFEST)
    print(f"  {len(val_pool.speaker_ids)} validation speakers")

    print(f"  generating probe TRAIN set: {args.train_examples_clean} clean + "
          f"{args.train_examples_per_tir}x4 TIR clips (train-manifest speakers) ...")
    train_X, train_y, train_tir, train_dom, _ = generate_ladder_dataset(
        model, train_pool, device, args.train_examples_per_tir, args.train_examples_clean,
        args.total_s, seed=20260824)
    print(f"    -> {len(train_y)} probe-training frames "
          f"({(train_y == 1).sum()} wearer/10, {(train_y == 0).sum()} environment/01)")

    print(f"  generating probe EVAL set: {args.eval_examples_clean} clean + "
          f"{args.eval_examples_per_tir}x4 TIR clips (held-out VAL-manifest speakers, "
          f"same generation convention as v2_frame_dataset.py) ...")
    eval_X, eval_y, eval_tir, eval_dom, eval_full_state = generate_ladder_dataset(
        model, val_pool, device, args.eval_examples_per_tir, args.eval_examples_clean,
        args.total_s, seed=20260825)
    print(f"    -> {len(eval_y)} probe-eval frames "
          f"({(eval_y == 1).sum()} wearer/10, {(eval_y == 0).sum()} environment/01)")

    # ---------------- 3. train + eval probes per stage ----------------
    print("\nSECTION 3: training frozen linear/1-hidden-layer probes R0..R8")
    ladder_table = {}
    for stage in STAGE_NAMES:
        t1 = time.time()
        probe, mu, sigma = train_probe(train_X[stage], train_y, device)
        scores = probe_predict(probe, mu, sigma, eval_X[stage], device)
        ladder_table[stage] = metrics_by_tir(eval_y, scores, eval_tir)
        print(f"  {stage} (D={train_X[stage].shape[1]:4d}): "
              + " ".join(f"{tag}=AUROC{ladder_table[stage][tag]['auroc']:.3f}" for tag in TIR_TAGS)
              + f"  [{time.time() - t1:.1f}s]")

    # R9: no probe -- the model's own dominance logit (wearer_logit - environment_logit) IS the score
    r9_metrics = metrics_by_tir(eval_y, sigmoid_np(eval_dom), eval_tir)
    ladder_table["R9"] = r9_metrics
    print(f"  R9 (final logits, dominance=wearer_logit-environment_logit, no probe): "
          + " ".join(f"{tag}=AUROC{r9_metrics[tag]['auroc']:.3f}" for tag in TIR_TAGS))

    results["representation_ladder"] = ladder_table
    results["probe_train_n"] = int(len(train_y))
    results["probe_eval_n"] = int(len(eval_y))

    # ---------------- 4/5. oracle + predicted masked pooling on P0's validation pool ----------------
    print("\nSECTION 4/5: oracle (O0-O3) + predicted (P0-P3) masked SpeakerNet pooling")
    clips = load_clip_list()
    n_clips = len(clips)
    print(f"  scoring {n_clips} clips (validation_suite_v2 + tir_negatives, P0's exact pool) ...")

    slice_names, clip_is_target = [], []
    scores_O = {k: np.full(n_clips, np.nan) for k in ("O0", "O1", "O2", "O3")}
    scores_S_original = np.full(n_clips, np.nan)
    valid_O = {k: np.zeros(n_clips, dtype=bool) for k in ("O0", "O1", "O2", "O3")}
    o0_sanity_cosine = np.full(n_clips, np.nan)

    # per-clip cached frame-level arrays for the calibration-split predicted-mask stage
    per_clip_wp, per_clip_ep, per_clip_dom = [], [], []
    per_clip_wt, per_clip_et, per_clip_valid_mask = [], [], []
    per_clip_frame_features, per_clip_wearer_embedding, per_clip_valid_len = [], [], []

    for cid, c in enumerate(clips):
        mixture = np.asarray(c["mixture"], dtype=np.float32)
        enrollment = np.asarray(c["enrollment"], dtype=np.float32)
        mix_t = torch.from_numpy(mixture).unsqueeze(0).to(device)
        mix_len = torch.tensor([len(mixture)], device=device)
        enr_t = torch.from_numpy(enrollment).unsqueeze(0).to(device)
        enr_len = torch.tensor([len(enrollment)], device=device)

        with torch.no_grad():
            wearer_embedding = model.encode_enrollment(enr_t, enr_len)
            taps, wl, el, frame_lengths, similarity = forward_with_ladder(model, mix_t, mix_len, wearer_embedding)
            frame_features = taps["R0"]  # [1,1500,T]
            T = wl.shape[1]
            length_mask = (torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1))

            m = c["meta"]
            w_true = align_labels_to_frames(c["wearer_activity"], T)
            e_true = align_labels_to_frames(c["environment_activity"], T)
            wt = torch.from_numpy(w_true).to(device).bool()
            et = torch.from_numpy(e_true).to(device).bool()

            masks = {
                "O0": length_mask,
                "O1": length_mask & wt,
                "O2": length_mask & wt & ~et,
                "O3": length_mask & (wt | et),
            }
            for k, mask in masks.items():
                cnt = int(mask.sum().item())
                if cnt < MIN_FRAMES_FOR_MASK:
                    continue
                emb = embed_from_mask(model, frame_features, mask)
                scores_O[k][cid] = F.cosine_similarity(emb, wearer_embedding, dim=-1).item()
                valid_O[k][cid] = True

            # sanity check: O0 embedding vs actual backbone.encode_speaker() directly on the same waveform
            mixture_speaker_embedding = model.backbone.encode_speaker(mix_t, mix_len)
            scores_S_original[cid] = F.cosine_similarity(wearer_embedding, mixture_speaker_embedding, dim=-1).item()
            if valid_O["O0"][cid]:
                o0_emb = embed_from_mask(model, frame_features, masks["O0"])
                o0_sanity_cosine[cid] = F.cosine_similarity(o0_emb, mixture_speaker_embedding, dim=-1).item()

            per_clip_frame_features.append(frame_features[0].cpu())
            per_clip_wearer_embedding.append(wearer_embedding[0].cpu())
            per_clip_valid_len.append(int(frame_lengths.item()))
            per_clip_wp.append(torch.sigmoid(wl[0]).cpu().numpy())
            per_clip_ep.append(torch.sigmoid(el[0]).cpu().numpy())
            per_clip_dom.append((wl[0] - el[0]).cpu().numpy())
            per_clip_wt.append(w_true)
            per_clip_et.append(e_true)
            per_clip_valid_mask.append(length_mask[0].cpu().numpy())

        slice_names.append(m["slice"])
        clip_is_target.append(bool(m.get("clip_is_target", not m["is_tir_negative_trial"])))
        if (cid + 1) % 40 == 0:
            print(f"    scored {cid + 1}/{n_clips}")

    slice_names = np.array(slice_names)
    clip_is_target = np.array(clip_is_target)
    buckets = bucket_masks(slice_names, clip_is_target)

    def eer_table_for(score_arr, valid_arr, buckets):
        table = {}
        excluded = {}
        for bucket_name, (pos_mask, neg_mask) in buckets.items():
            pos_m = pos_mask & valid_arr
            neg_m = neg_mask & valid_arr
            r = eer_auroc_far_for_bucket(pos_m, neg_m, score_arr)
            table[bucket_name] = r
            excluded[bucket_name] = int(((pos_mask | neg_mask) & ~valid_arr).sum())
        return table, excluded

    oracle_full_table = {}
    oracle_full_excluded = {}
    for k in ("O0", "O1", "O2", "O3"):
        oracle_full_table[k], oracle_full_excluded[k] = eer_table_for(scores_O[k], valid_O[k], buckets)
    s_original_table, _ = eer_table_for(scores_S_original, np.ones(n_clips, dtype=bool), buckets)

    print("\n  ORACLE POOLING TABLE (full pool, EER%):")
    header = "  " + f"{'mask':<6}" + "".join(f"{DISPLAY_TAGS[t]:>8}" for t in TIR_TAGS)
    print(header)
    for k in ("O0", "O1", "O2", "O3"):
        row = f"  {k:<6}"
        for tag in TIR_TAGS:
            row += f"{oracle_full_table[k][tag]['eer_pct']:>8.2f}"
        print(row)
    print(f"  {'S_orig':<6}" + "".join(f"{s_original_table[tag]['eer_pct']:>8.2f}"
                                        for tag in TIR_TAGS) + "  (P0 replica, sanity)")
    print(f"  historical P0 s_original: " + " ".join(f"{tag}={HISTORICAL_P0_S_ORIGINAL_EER[tag]}" for tag in TIR_TAGS))
    valid_sanity = ~np.isnan(o0_sanity_cosine)
    print(f"  O0-vs-encode_speaker() cosine sanity check: mean={np.nanmean(o0_sanity_cosine):.6f} "
          f"min={np.nanmin(o0_sanity_cosine):.6f} (n={valid_sanity.sum()}, should be ~1.0)")

    results["oracle_pooling_full_pool"] = oracle_full_table
    results["oracle_pooling_excluded_counts"] = oracle_full_excluded
    results["s_original_sanity_table"] = s_original_table
    results["o0_vs_encode_speaker_cosine_mean"] = float(np.nanmean(o0_sanity_cosine))
    results["o0_vs_encode_speaker_cosine_min"] = float(np.nanmin(o0_sanity_cosine))

    if np.isnan(oracle_full_table["O2"]["clean"]["eer_pct"]) and np.isnan(oracle_full_table["O1"]["clean"]["eer_pct"]):
        print("  NOTE: O1/O2 are NaN on this pool by construction, not a bug -- see code comment above "
              "run_multisegment_oracle_supplement(). validation_suite_v2's overlap-TIR POSITIVE trials are "
              "100% state=11 for their full 2s duration (make_overlap_example sets wearer_activity AND "
              "environment_activity to all-ones for the whole clip), so O2 (wt=1 & et=0) is EXACTLY EMPTY "
              "for every TIR positive trial, and every NEGATIVE trial in this pool has wt=0 for its entire "
              "duration by definition of 'negative' (the enrolled wearer never appears), so O1/O2 are also "
              "exactly empty for every negative. A supplementary multi-segment experiment below (fresh "
              "synthetic clips with genuine within-clip solo-wearer/overlap mixes) tests O1 vs O2 properly.")

    print("\n  SUPPLEMENTARY: oracle pooling on multi-segment clips (genuine within-clip state mixes,"
          " fresh VAL-manifest synthetic verification trials, NOT validation_suite_v2)")
    supp_table, supp_excluded, supp_n = run_multisegment_oracle_supplement(
        model, val_pool, device, args.eval_examples_per_tir, args.eval_examples_clean, args.total_s, seed=20260826)
    print(header)
    for k in ("O0", "O1", "O2", "O3"):
        row = f"  {k:<6}"
        for tag in TIR_TAGS:
            row += f"{supp_table[k][tag]['eer_pct']:>8.2f}"
        row += f"   (excluded {supp_excluded[k]}/{supp_n} trials for <{MIN_FRAMES_FOR_MASK} mask frames)"
        print(row)
    results["oracle_pooling_multisegment_supplement"] = supp_table
    results["oracle_pooling_multisegment_supplement_excluded"] = supp_excluded
    results["oracle_pooling_multisegment_supplement_n_trials"] = supp_n

    # ---------------- 5. predicted-mask pooling, calibration split ----------------
    print("\nSECTION 5: predicted-mask (P0-P3) pooling, calibration split (even idx=calib, odd idx=eval)")
    calib_idx = np.array([i for i in range(n_clips) if i % 2 == 0])
    eval_idx = np.array([i for i in range(n_clips) if i % 2 == 1])

    def gather_frames(idx_list):
        wp_all, ep_all, dom_all, wt_all, et_all = [], [], [], [], []
        for cid in idx_list:
            valid = per_clip_valid_mask[cid]
            wp_all.append(per_clip_wp[cid][valid])
            ep_all.append(per_clip_ep[cid][valid])
            dom_all.append(per_clip_dom[cid][valid])
            wt_all.append(per_clip_wt[cid][valid])
            et_all.append(per_clip_et[cid][valid])
        return (np.concatenate(wp_all), np.concatenate(ep_all), np.concatenate(dom_all),
                np.concatenate(wt_all), np.concatenate(et_all))

    wp_c, ep_c, dom_c, wt_c, et_c = gather_frames(calib_idx)
    thr_p0 = youden_threshold(wp_c, wt_c)
    thr_p1 = youden_threshold(dom_c, wt_c)
    thr_p2_w = thr_p0
    thr_p2_e = youden_threshold(ep_c, et_c)

    # P3: top-K% by dominance per clip -- select K (calibration only) maximizing
    # frame-level F1 for predicting state==10 (true_wearer=1 AND true_environment=0)
    candidate_ks = [5, 10, 15, 20, 25, 30, 40, 50]
    best_k, best_f1 = candidate_ks[0], -1.0
    for K in candidate_ks:
        tp = fp = fn = 0
        for cid in calib_idx:
            valid = per_clip_valid_mask[cid]
            dom = per_clip_dom[cid][valid]
            wt = per_clip_wt[cid][valid]
            et = per_clip_et[cid][valid]
            n = len(dom)
            if n == 0:
                continue
            k_n = max(1, int(np.ceil(K / 100.0 * n)))
            top_idx = np.argsort(dom)[::-1][:k_n]
            pred = np.zeros(n, dtype=bool)
            pred[top_idx] = True
            true_solo = (wt > 0.5) & (et <= 0.5)
            tp += int((pred & true_solo).sum())
            fp += int((pred & ~true_solo).sum())
            fn += int((~pred & true_solo).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_k = f1, K
    print(f"  calibrated thresholds (from {len(calib_idx)} calibration clips): "
          f"P0(wearer_prob)>{thr_p0:.4f}  P1(dominance)>{thr_p1:.4f}  "
          f"P2(wearer_prob)>{thr_p2_w:.4f}&(env_prob)<{thr_p2_e:.4f}  "
          f"P3(top-K%% dominance) K={best_k}% (calib F1={best_f1:.3f})")

    results["predicted_mask_calibration"] = {
        "thr_p0_wearer_prob": thr_p0, "thr_p1_dominance": thr_p1,
        "thr_p2_wearer_prob": thr_p2_w, "thr_p2_env_prob": thr_p2_e,
        "p3_top_k_pct": best_k, "p3_calib_f1": best_f1,
        "n_calib_clips": int(len(calib_idx)), "n_eval_clips": int(len(eval_idx)),
    }

    # apply P0-P3 masks to ALL clips (compute once), then slice EER by eval_idx-only bucket for
    # a leakage-free comparison, and also report full-pool for direct visual comparison to O-table.
    scores_P = {k: np.full(n_clips, np.nan) for k in ("P0", "P1", "P2", "P3")}
    valid_P = {k: np.zeros(n_clips, dtype=bool) for k in ("P0", "P1", "P2", "P3")}

    for cid in range(n_clips):
        valid = per_clip_valid_mask[cid]
        wp = per_clip_wp[cid]
        ep = per_clip_ep[cid]
        dom = per_clip_dom[cid]
        n_valid = int(valid.sum())

        pred_masks = {
            "P0": valid & (wp > thr_p0),
            "P1": valid & (dom > thr_p1),
            "P2": valid & (wp > thr_p2_w) & (ep < thr_p2_e),
        }
        # P3: top-K% of VALID frames by dominance
        p3_mask = np.zeros_like(valid)
        if n_valid > 0:
            valid_positions = np.where(valid)[0]
            dom_valid = dom[valid_positions]
            k_n = max(1, int(np.ceil(best_k / 100.0 * n_valid)))
            top_local = np.argsort(dom_valid)[::-1][:k_n]
            p3_mask[valid_positions[top_local]] = True
        pred_masks["P3"] = p3_mask

        frame_features = per_clip_frame_features[cid].unsqueeze(0).to(device)
        wearer_embedding = per_clip_wearer_embedding[cid].unsqueeze(0).to(device)
        for k, mask_np in pred_masks.items():
            cnt = int(mask_np.sum())
            if cnt < MIN_FRAMES_FOR_MASK:
                continue
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = embed_from_mask(model, frame_features, mask_t)
                scores_P[k][cid] = F.cosine_similarity(emb, wearer_embedding, dim=-1).item()
            valid_P[k][cid] = True

    eval_mask_bool = np.zeros(n_clips, dtype=bool)
    eval_mask_bool[eval_idx] = True

    def restrict_buckets(buckets, subset_mask):
        return {name: (pos & subset_mask, neg & subset_mask) for name, (pos, neg) in buckets.items()}

    buckets_eval_only = restrict_buckets(buckets, eval_mask_bool)

    predicted_table_evalhalf = {}
    for k in ("P0", "P1", "P2", "P3"):
        predicted_table_evalhalf[k], _ = eer_table_for(scores_P[k], valid_P[k], buckets_eval_only)

    # O-table restricted to the SAME eval-half clips, for an apples-to-apples P-vs-O comparison
    oracle_evalhalf_table = {}
    for k in ("O0", "O1", "O2", "O3"):
        oracle_evalhalf_table[k], _ = eer_table_for(scores_O[k], valid_O[k], buckets_eval_only)

    print(f"\n  PREDICTED MASK TABLE (eval-half only, n={len(eval_idx)} clips, thresholds from calib-half, EER%):")
    print(header)
    for k in ("P0", "P1", "P2", "P3"):
        row = f"  {k:<6}"
        for tag in TIR_TAGS:
            row += f"{predicted_table_evalhalf[k][tag]['eer_pct']:>8.2f}"
        print(row)
    print("  (oracle O0-O3 on the SAME eval-half subset, for direct comparison)")
    for k in ("O0", "O1", "O2", "O3"):
        row = f"  {k:<6}"
        for tag in TIR_TAGS:
            row += f"{oracle_evalhalf_table[k][tag]['eer_pct']:>8.2f}"
        print(row)

    results["predicted_mask_evalhalf"] = predicted_table_evalhalf
    results["oracle_evalhalf_for_comparison"] = oracle_evalhalf_table

    # ---------------- 6. similarity-branch audit (best-effort) ----------------
    print("\nSECTION 6: frame-level similarity-branch audit (best-effort)")
    all_states, all_sims, all_tirs = [], [], []
    for state, sim, tag in eval_full_state:
        all_states.append(state)
        all_sims.append(sim)
        all_tirs.append(np.array([tag] * len(state)))
    all_states = np.concatenate(all_states)
    all_sims = np.concatenate(all_sims)
    all_tirs = np.concatenate(all_tirs)

    sim_by_state = {}
    for name, code in (("00_silence", 0), ("10_wearer", 1), ("01_environment", 2), ("11_overlap", 3)):
        vals = all_sims[all_states == code]
        if len(vals) == 0:
            continue
        sim_by_state[name] = {
            "n": int(len(vals)), "mean": float(vals.mean()), "std": float(vals.std()),
            "p25": float(np.percentile(vals, 25)), "median": float(np.median(vals)),
            "p75": float(np.percentile(vals, 75)),
        }
        print(f"  {name:<16} n={len(vals):6d}  mean={vals.mean():+.4f}  std={vals.std():.4f}  "
              f"median={np.median(vals):+.4f}  IQR=[{np.percentile(vals,25):+.4f}, {np.percentile(vals,75):+.4f}]")

    # AUROC: does the similarity branch alone separate 10 from 01?
    m1001 = (all_states == 1) | (all_states == 2)
    sim_1001_auroc = safe_auroc((all_states[m1001] == 1).astype(int), all_sims[m1001])
    # does it separate 10 from 11 (does overlap corrupt the per-frame similarity relative to solo)?
    m1011 = (all_states == 1) | (all_states == 3)
    sim_1011_auroc = safe_auroc((all_states[m1011] == 1).astype(int), all_sims[m1011])
    print(f"  similarity-branch AUROC, 10-vs-01: {sim_1001_auroc:.4f}")
    print(f"  similarity-branch AUROC, 10-vs-11 (does overlap shift it toward/away from solo-wearer?): {sim_1011_auroc:.4f}")

    results["similarity_audit"] = {
        "applicable": True,
        "note": ("A principled per-frame speaker-similarity signal DOES exist at post-FiLM/similarity-fusion "
                 "depth: MentraWearNet's own frame_speaker_projection + enrollment_similarity_projection + "
                 "cosine branch (the `similarity` tap, computed between R2-derived frame_speaker and the "
                 "projected enrollment embedding, both already part of the frozen model's forward pass). "
                 "This is NOT a raw-R0-stage embedding -- SpeakerNet's architecture only defines a speaker "
                 "embedding after full-clip statistics pooling, so there is no principled per-frame embedding "
                 "at R0 itself; the similarity tap used here is downstream of frame_projection+FiLM (R2), "
                 "which is the earliest point in the graph where a per-frame speaker-comparable vector exists."),
        "sim_by_state": sim_by_state,
        "sim_10_vs_01_auroc": sim_1001_auroc,
        "sim_10_vs_11_auroc": sim_1011_auroc,
    }

    # ---------------- fingerprint after ----------------
    fp_after = fingerprint(model)
    print("\n" + "=" * 70)
    print(f"fingerprint (after all diagnostic work): {fp_after}")
    print(f"fingerprint match: {'YES -- frozen model untouched' if fp_after == fp_before else 'NO -- MODEL CHANGED, BUG'}")
    results["fingerprint_before"] = fp_before
    results["fingerprint_after"] = fp_after
    results["fingerprint_match"] = (fp_before == fp_after)
    assert fp_after == fp_before, "FROZEN MODEL FINGERPRINT CHANGED -- diagnostic script has a bug"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    print(f"\nsaved {out_path}")
    print(f"total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
