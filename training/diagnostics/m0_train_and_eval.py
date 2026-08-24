#!/usr/bin/env python3
"""M0: precision-first four-state frame classifier trained on the FROZEN
mentrawearnet_v2_step4800 representation (R8 tap + wearer/environment/
dominance logits, all already-computed and frozen), whose predicted masks
then drive SpeakerNet's ORIGINAL statistics pooling (backbone.pooling /
backbone.embedding_projection -- no new pooling mechanism).

Covers Phases 2-9 of the M0 task spec (Phase 0/1 are separate scripts:
phase0_oracle_bias_correction.py and build_m0_eval_suite.py). Zero training
of the main MentraWearNet model -- loaded frozen, .eval()'d once,
requires_grad=False on every existing param, SHA-256 fingerprint verified
unchanged before/after (reusing r0_information_loss_audit.py's fingerprint()
and the same ladder-extraction/masked-pooling utilities it already
verified).

Run: .venv/bin/python3 training/diagnostics/m0_train_and_eval.py --device cuda:1
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_curve

from training.data.mixture_generator import SpeakerPool
from training.diagnostics.r0_information_loss_audit import (
    FROZEN_CKPT, MIN_FRAMES_FOR_MASK, TIR_LEVELS, TIR_TAGS, DISPLAY_TAGS,
    fingerprint, forward_with_ladder, masked_stats_pool, verify_ladder_equivalence,
    youden_threshold,
)
from training.diagnostics.v2_analyze import compute_eer, safe_auprc, safe_auroc
from training.diagnostics.v2_frame_dataset import build_clean_example, build_overlap_bucket_example, state_code
from training.models.mentrawearnet import MentraWearNet
from training.train import align_labels_to_frames

TRAIN_MANIFEST = "evaluation/manifests/librispeech_train_clean_100_train.json"
SUITE_NPZ = "evaluation/manifests/m0_eval_suite_v1.npz"
SUITE_META = "evaluation/manifests/m0_eval_suite_v1_metadata.json"
PHASE0_RESULTS = "evaluation/results/phase0_oracle_bias_correction.json"
RESULTS_DIR = Path("evaluation/results")
DURATION_CANDIDATES_MS = (200, 400, 600, 1000)
STATE_NAMES = ["00", "10", "01", "11"]
PRECISION_TARGETS = (0.90, 0.95, 0.975)


# ============================================================================
# Phase 5/6: weighted masked SpeakerNet statistics pooling (generalizes R0's
# masked_stats_pool from boolean masks to arbitrary non-negative frame
# weights). Mathematically correct weighted mean/variance -- NOT "multiply
# frames by weight then run unweighted stats". Uses a Bessel-style
# correction on the TOTAL weight (sum_w - 1) rather than the more elaborate
# reliability-weight formula (sum_w - sum_w^2/sum_w) precisely because that
# elaborate formula degenerates to 0/0 for 0/1 weights (sum_w - sum_w^2/sum_w
# = n - n = 0), which would NOT reduce correctly to R0's already-verified
# n-1 hard-mask formula. The (sum_w - 1) correction is the standard
# generalization used in practice and reduces EXACTLY to R0's masked_stats_pool
# when weights are 0/1 (sanity-checked below) and to backbone.pooling's
# native encode_speaker() when weights are all-1 over the valid prefix
# (sanity-checked below, cosine ~1.0, same pattern R0 already used).
# ============================================================================
def weighted_masked_stats_pool(frame_features: torch.Tensor, weights: torch.Tensor,
                                eps: float = 1e-10) -> torch.Tensor:
    """frame_features: [B,1500,T]. weights: [B,T] float >= 0 (NOT required to be 0/1)."""
    w = weights.unsqueeze(1)  # [B,1,T]
    sum_w = weights.sum(dim=-1).clamp(min=1e-6)  # [B]
    mean = (frame_features * w).sum(dim=-1) / sum_w.unsqueeze(-1)  # [B,1500]
    centered_sq = (frame_features - mean.unsqueeze(-1)).pow(2) * w  # weight the squared deviation itself
    denom = (sum_w - 1).clamp(min=1e-6)
    var = centered_sq.sum(dim=-1) / denom.unsqueeze(-1)
    std = var.clamp(min=eps).sqrt()
    return torch.cat([mean, std], dim=-1)  # [B,3000]


def embed_from_weights(model: MentraWearNet, frame_features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    pooled = weighted_masked_stats_pool(frame_features, weights)
    emb = model.backbone.embedding_projection(pooled)
    return F.normalize(emb, p=2, dim=-1)


@torch.no_grad()
def sanity_check_weighted_pool(model: MentraWearNet, device: str) -> dict:
    torch.manual_seed(1)
    waveform = torch.randn(3, 32000, device=device) * 0.05
    lengths = torch.tensor([32000, 28000, 20000], device=device)
    frame_features, frame_lengths = model.backbone.encode_frames(waveform, lengths)
    T = frame_features.shape[-1]
    length_mask = (torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1))

    # 1) weights == 0/1 hard mask must match R0's masked_stats_pool bit-for-bit
    hard_mask = length_mask & (torch.rand(3, T, device=device) > 0.4)
    hard_mask[:, 0] = True  # avoid degenerate all-false rows
    pooled_a = masked_stats_pool(frame_features, hard_mask)
    pooled_b = weighted_masked_stats_pool(frame_features, hard_mask.float())
    max_diff = (pooled_a - pooled_b).abs().max().item()

    # 2) weights == all-ones over the valid prefix must match encode_speaker() (cosine ~1.0)
    emb_w = embed_from_weights(model, frame_features, length_mask.float())
    emb_native = model.backbone.encode_speaker(waveform, lengths)
    cos = F.cosine_similarity(emb_w, emb_native, dim=-1)
    return {
        "hard_vs_weighted_max_abs_diff": max_diff,
        "weights1_vs_encode_speaker_cosine": cos.cpu().numpy().tolist(),
    }


# ============================================================================
# Streaming feasibility: rolling weighted moments (sum_w, sum_w*x, sum_w*x^2),
# standard online update. Confirmed here by replaying frame-by-frame and
# checking it reproduces weighted_masked_stats_pool's batch result.
# ============================================================================
@torch.no_grad()
def streaming_weighted_pool_replay(frame_features: torch.Tensor, weights: torch.Tensor, eps: float = 1e-10):
    """frame_features: [1500,T] weights: [T]. Frame-by-frame online update,
    causal (only sees frames 0..t at "time" t), final state compared to the
    batch weighted_masked_stats_pool result."""
    D, T = frame_features.shape
    sum_w = torch.zeros((), device=frame_features.device)
    sum_wx = torch.zeros(D, device=frame_features.device)
    sum_wx2 = torch.zeros(D, device=frame_features.device)
    for t in range(T):
        w_t = weights[t]
        x_t = frame_features[:, t]
        sum_w = sum_w + w_t
        sum_wx = sum_wx + w_t * x_t
        sum_wx2 = sum_wx2 + w_t * x_t * x_t
    mean = sum_wx / sum_w.clamp(min=1e-6)
    var = (sum_wx2 - sum_w * mean * mean) / (sum_w - 1).clamp(min=1e-6)
    std = var.clamp(min=eps).sqrt()
    return torch.cat([mean, std], dim=-1)


# ============================================================================
# Phase 3: M0 architecture
# ============================================================================
class M0MaskPredictor(nn.Module):
    """LayerNorm -> Linear(input_dim, 64) -> GELU -> Linear(64, 4). Per-frame
    logits for states [00, 10, 01, 11]. Input: frozen R8 tap (192-d) concat
    wearer_logit, environment_logit, speaker_dominance_logit (all already
    computed by the frozen model's forward pass, no new temporal network)."""

    def __init__(self, r8_dim: int = 192):
        super().__init__()
        input_dim = r8_dim + 3
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, 64)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))


def m0_input_features(r8: torch.Tensor, wl: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
    """r8: [B,D,T] (channel-first tap). wl, el: [B,T]. Returns [B,T,D+3]."""
    r8_t = r8.transpose(1, 2)  # [B,T,D]
    dom = (wl - el).unsqueeze(-1)
    return torch.cat([r8_t, wl.unsqueeze(-1), el.unsqueeze(-1), dom], dim=-1)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Class-balanced focal loss. alpha: [4] per-class weight
    (class-balanced, e.g. inverse-sqrt-frequency). Chosen over plain
    weighted CE because M0 is used as a PRECISION-first selector -- only
    frames the model is confident about get used downstream (via calibrated
    precision thresholds), and the specific failure mode we most want to
    suppress is confident false inclusion of 01 (pure-environment) frames
    into the 10/target-active selection. Focal's (1-p_t)^gamma term
    down-weights already-easy/correct examples and up-weights hard/
    misclassified ones (disproportionately state-01-near-10 confusions),
    which should sharpen precision at the high-confidence operating points
    Phase 4 reports on, more than uniform weighted CE would -- at the cost
    of slightly slower convergence, acceptable given how cheap this run is."""
    logp = F.log_softmax(logits, dim=-1)
    p_t = logp.exp().gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    logp_t = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    alpha_t = alpha[targets]
    loss = -alpha_t * ((1 - p_t) ** gamma) * logp_t
    return loss.mean()


# ============================================================================
# Live frame-dataset generation (train speakers, for M0 training) --
# extracts R8 tap + wearer/env logits + 4-state label per frame, reusing the
# already-verified forward_with_ladder (no mentrawearnet.py edits, no
# gradient through the frozen model -- @torch.no_grad()).
# ============================================================================
@torch.no_grad()
def generate_m0_training_frames(model: MentraWearNet, pool: SpeakerPool, device: str,
                                 examples_per_tir: int, examples_clean: int, total_s: float, seed: int):
    rng = random.Random(seed)
    feats, labels, clip_ids = [], [], []

    def process(mixture, enrollment, w_act, e_act, cid):
        mix_t = torch.from_numpy(np.asarray(mixture, dtype=np.float32)).unsqueeze(0).to(device)
        mix_len = torch.tensor([len(mixture)], device=device)
        enr_t = torch.from_numpy(np.asarray(enrollment, dtype=np.float32)).unsqueeze(0).to(device)
        enr_len = torch.tensor([len(enrollment)], device=device)
        wearer_embedding = model.encode_enrollment(enr_t, enr_len)
        taps, wl, el, frame_lengths, _sim = forward_with_ladder(model, mix_t, mix_len, wearer_embedding)
        T = wl.shape[1]
        valid = (torch.arange(T, device=device) < frame_lengths[0]).cpu().numpy()
        wt = align_labels_to_frames(w_act, T)
        et = align_labels_to_frames(e_act, T)
        state = state_code(wt, et)
        r8 = taps["R8"][0].cpu().numpy()  # [D,T]
        x = np.concatenate([r8.T, wl[0].cpu().numpy()[:, None], el[0].cpu().numpy()[:, None],
                             (wl[0] - el[0]).cpu().numpy()[:, None]], axis=1)  # [T, D+3]
        feats.append(x[valid])
        labels.append(state[valid])
        clip_ids.append(np.full(valid.sum(), cid, dtype=np.int64))

    cid = 0
    for _ in range(examples_clean):
        mixture, enrollment, w_act, e_act, wid, oid = build_clean_example(pool, rng, total_s)
        process(mixture, enrollment, w_act, e_act, cid)
        cid += 1
    for tir in TIR_LEVELS:
        for _ in range(examples_per_tir):
            mixture, enrollment, w_act, e_act, wid, oid = build_overlap_bucket_example(pool, rng, tir, total_s)
            process(mixture, enrollment, w_act, e_act, cid)
            cid += 1

    X = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    clip_ids = np.concatenate(clip_ids, axis=0)
    return X, y, clip_ids, cid


# ============================================================================
# Suite loading + per-clip frozen-model forward (for calibration/eval)
# ============================================================================
def load_suite():
    npz = np.load(SUITE_NPZ, allow_pickle=True)
    meta = json.loads(Path(SUITE_META).read_text())
    return npz, meta["clips"]


@torch.no_grad()
def process_suite_clip(model: MentraWearNet, m0: M0MaskPredictor, npz, clip_meta, cid, device):
    mixture = np.asarray(npz["mixture"][cid], dtype=np.float32)
    enrollment = np.asarray(npz["enrollment"][cid], dtype=np.float32)
    impostor = np.asarray(npz["impostor_enrollment"][cid], dtype=np.float32)
    w_act = npz["wearer_activity"][cid]
    e_act = npz["environment_activity"][cid]

    mix_t = torch.from_numpy(mixture).unsqueeze(0).to(device)
    mix_len = torch.tensor([len(mixture)], device=device)
    enr_t = torch.from_numpy(enrollment).unsqueeze(0).to(device)
    enr_len = torch.tensor([len(enrollment)], device=device)
    imp_t = torch.from_numpy(impostor).unsqueeze(0).to(device)
    imp_len = torch.tensor([len(impostor)], device=device)

    wearer_embedding = model.encode_enrollment(enr_t, enr_len)
    impostor_embedding = model.encode_enrollment(imp_t, imp_len)
    taps, wl, el, frame_lengths, _sim = forward_with_ladder(model, mix_t, mix_len, wearer_embedding)
    frame_features = taps["R0"]  # [1,1500,T]
    T = wl.shape[1]
    length_mask = (torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1))
    wt = torch.from_numpy(align_labels_to_frames(w_act, T)).to(device)
    et = torch.from_numpy(align_labels_to_frames(e_act, T)).to(device)
    state = (wt.bool()).long() + (et.bool()).long() * 2  # 0..3, matches state_code convention

    x = m0_input_features(taps["R8"], wl, el)
    probs = F.softmax(m0(x), dim=-1)[0]  # [T,4]

    return {
        "frame_features": frame_features, "length_mask": length_mask[0],
        "wearer_embedding": wearer_embedding, "impostor_embedding": impostor_embedding,
        "state": state.cpu().numpy(), "probs": probs, "T": T,
    }


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train-examples-per-tir", type=int, default=400)
    ap.add_argument("--train-examples-clean", type=int, default=400)
    ap.add_argument("--total-s", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", default="evaluation/results/m0_train_and_eval.json")
    args = ap.parse_args()
    device = args.device
    t_start = time.time()
    results = {}

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-m0_mask_predictor"
    run_dir = Path("training/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log(f"run_dir={run_dir}")

    # ---------------- Phase 2: freeze + fingerprint ----------------
    log("=" * 70)
    log("PHASE 2: freeze + fingerprint")
    model = MentraWearNet().to(device)
    ckpt = torch.load(FROZEN_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    fp_before = fingerprint(model)
    log(f"loaded {FROZEN_CKPT} (step={ckpt.get('step')})  fingerprint_before={fp_before}")
    ladder_ok = verify_ladder_equivalence(model, device)
    assert ladder_ok
    sanity = sanity_check_weighted_pool(model, device)
    log(f"weighted-pool sanity: hard-vs-weighted max|diff|={sanity['hard_vs_weighted_max_abs_diff']:.2e}  "
        f"weights=1 vs encode_speaker() cosine={sanity['weights1_vs_encode_speaker_cosine']}")
    results["sanity_checks"] = sanity
    results["ladder_equivalence_pass"] = ladder_ok

    # ---------------- Phase 3: M0 architecture ----------------
    log("\nPHASE 3: M0 architecture")
    m0 = M0MaskPredictor(r8_dim=192).to(device)
    n_params = sum(p.numel() for p in m0.parameters())
    log(f"M0MaskPredictor param count: {n_params}  (hard cap 100000, target <50000)")
    assert n_params < 100_000, "M0 exceeds hard param cap"
    results["m0_param_count"] = n_params

    # streaming replay correctness check (Phase 8, but cheap to do now)
    torch.manual_seed(2)
    ff_test = torch.randn(1500, 60, device=device)
    w_test = torch.rand(60, device=device)
    batch_res = weighted_masked_stats_pool(ff_test.unsqueeze(0), w_test.unsqueeze(0))[0]
    stream_res = streaming_weighted_pool_replay(ff_test, w_test)
    stream_diff = (batch_res - stream_res).abs().max().item()
    log(f"streaming-replay vs batch weighted-pool max|diff|={stream_diff:.2e}")
    results["streaming_replay_max_abs_diff"] = stream_diff

    # ---------------- Phase 4: training data + training ----------------
    log("\nPHASE 4: generating M0 training frame dataset (TRAIN manifest speakers)")
    train_pool = SpeakerPool(TRAIN_MANIFEST)
    log(f"  {len(train_pool.speaker_ids)} training speakers")
    t1 = time.time()
    X, y, clip_ids, n_clips = generate_m0_training_frames(
        model, train_pool, device, args.train_examples_per_tir, args.train_examples_clean,
        args.total_s, seed=args.seed)
    log(f"  generated {len(y)} frames from {n_clips} clips in {time.time() - t1:.1f}s")
    state_counts = {STATE_NAMES[i]: int((y == i).sum()) for i in range(4)}
    log(f"  state distribution: {state_counts}")

    # held-out-by-clip split within TRAIN manifest data, for monitoring only
    # (NOT for hyperparameter selection -- that stays on the VAL-manifest
    # calibration split later; this is purely an overfitting sanity check).
    rng_split = np.random.RandomState(0)
    clip_perm = rng_split.permutation(n_clips)
    val_clip_ids = set(clip_perm[: max(1, n_clips // 10)].tolist())
    is_heldout = np.isin(clip_ids, list(val_clip_ids))
    Xtr, ytr = X[~is_heldout], y[~is_heldout]
    Xho, yho = X[is_heldout], y[is_heldout]
    log(f"  train/heldout-by-clip frame split: {len(ytr)} / {len(yho)}")

    mu = Xtr.mean(axis=0, keepdims=True)
    sigma = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr_n = (Xtr - mu) / sigma
    Xho_n = (Xho - mu) / sigma

    class_freq = np.array([max(1, (ytr == i).sum()) for i in range(4)], dtype=np.float64)
    alpha = (1.0 / np.sqrt(class_freq))
    alpha = alpha / alpha.sum() * 4.0
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)
    log(f"  class-balanced focal-loss alpha (inverse-sqrt-frequency): {alpha.tolist()}")

    Xtr_t = torch.from_numpy(Xtr_n.astype(np.float32)).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xho_t = torch.from_numpy(Xho_n.astype(np.float32)).to(device)
    yho_t = torch.from_numpy(yho).to(device)

    opt = torch.optim.Adam(m0.parameters(), lr=args.lr)
    n = Xtr_t.shape[0]
    torch.manual_seed(args.seed)
    loss_curve = []
    t1 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            logits = m0(Xtr_t[idx])
            loss = focal_loss(logits, ytr_t[idx], alpha_t)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        with torch.no_grad():
            ho_logits = m0(Xho_t)
            ho_loss = focal_loss(ho_logits, yho_t, alpha_t).item()
            ho_acc = (ho_logits.argmax(-1) == yho_t).float().mean().item()
        rec = {"epoch": ep, "train_loss": ep_loss / n_batches, "heldout_loss": ho_loss, "heldout_acc": ho_acc}
        loss_curve.append(rec)
        if ep % 5 == 0 or ep == args.epochs - 1:
            log(f"  epoch {ep:3d}  train_loss={rec['train_loss']:.4f}  heldout_loss={ho_loss:.4f}  "
                f"heldout_acc={ho_acc:.4f}")
    log(f"  training done in {time.time() - t1:.1f}s")
    results["training_curve"] = loss_curve
    results["train_manifest_state_counts"] = state_counts

    m0.eval()
    ckpt_path = run_dir / f"m0_mask_predictor_step{args.epochs}.pt"
    torch.save({
        "model_state_dict": m0.state_dict(), "mu": mu, "sigma": sigma,
        "epochs": args.epochs, "n_params": n_params, "alpha": alpha.tolist(),
        "config": vars(args),
    }, ckpt_path)
    stable_ptr = Path("training/checkpoints/m0_mask_predictor_latest.pt")
    torch.save({
        "model_state_dict": m0.state_dict(), "mu": mu, "sigma": sigma,
        "epochs": args.epochs, "n_params": n_params, "alpha": alpha.tolist(),
        "config": vars(args),
    }, stable_ptr)
    log(f"  saved checkpoint: {ckpt_path} (and stable pointer {stable_ptr})")
    results["checkpoint_path"] = str(ckpt_path)

    mu_t = torch.from_numpy(mu.astype(np.float32)).to(device)
    sigma_t = torch.from_numpy(sigma.astype(np.float32)).to(device)

    class NormalizedM0(nn.Module):
        def __init__(self, inner, mu, sigma):
            super().__init__()
            self.inner = inner
            self.register_buffer("mu", mu)
            self.register_buffer("sigma", sigma)

        def forward(self, x):
            return self.inner((x - self.mu) / self.sigma)

    m0_full = NormalizedM0(m0, mu_t, sigma_t).to(device).eval()

    # ---------------- Phase 1 suite + Phase 5-7: calibration + eval ----------------
    log("\nPHASE 5-7: loading Phase-1 eval suite, calibrating thresholds, scoring pooling variants")
    npz, clip_meta = load_suite()
    n_suite = len(clip_meta)
    calib_idx = [i for i, m in enumerate(clip_meta) if m["split"] == "calib"]
    eval_idx = [i for i, m in enumerate(clip_meta) if m["split"] == "eval"]
    log(f"  suite: {n_suite} clips total, {len(calib_idx)} calib / {len(eval_idx)} eval")

    # NOTE on performance: this suite's clips are all generated with the
    # SAME total_s (fixed 2.0s), so every clip has IDENTICAL mixture sample
    # count and therefore IDENTICAL encoder frame count T (encode_frames'
    # frame count is a deterministic function of input length only) -- so
    # clips can be batched directly with no padding on the mixture side
    # (enrollment/impostor clips DO vary in length -> padded+lengths, same
    # pattern as training/train.py's build_batch). Batching eliminates the
    # dominant cost in an earlier version of this script: looping per-clip
    # and calling .item() on every pooling variant's cosine score forces a
    # CUDA sync each time, which under this machine's heavy GPU contention
    # (an unrelated job at 90%+ utilization) serializes the CPU behind the
    # other job's queue thousands of times over. Batched calls + deferring
    # .cpu()/.numpy() conversion to chunk/pass boundaries cut sync count by
    # ~2-3 orders of magnitude.
    def pad_batch_1d(arrays, device):
        lengths = [len(a) for a in arrays]
        maxlen = max(lengths)
        batch = np.zeros((len(arrays), maxlen), dtype=np.float32)
        for i, a in enumerate(arrays):
            batch[i, :len(a)] = a
        return torch.from_numpy(batch).to(device), torch.tensor(lengths, dtype=torch.long, device=device)

    @torch.no_grad()
    def batched_process(idx_list, chunk_size=64):
        """Yields one dict per chunk of stacked GPU tensors + numpy state."""
        for start in range(0, len(idx_list), chunk_size):
            chunk = idx_list[start:start + chunk_size]
            mixtures = np.stack([np.asarray(npz["mixture"][cid], dtype=np.float32) for cid in chunk])
            mix_t = torch.from_numpy(mixtures).to(device)
            mix_len = torch.full((len(chunk),), mixtures.shape[1], dtype=torch.long, device=device)
            enr_t, enr_len = pad_batch_1d([np.asarray(npz["enrollment"][cid], dtype=np.float32) for cid in chunk], device)
            imp_t, imp_len = pad_batch_1d([np.asarray(npz["impostor_enrollment"][cid], dtype=np.float32) for cid in chunk], device)

            wearer_embedding = model.encode_enrollment(enr_t, enr_len)
            impostor_embedding = model.encode_enrollment(imp_t, imp_len)
            taps, wl, el, frame_lengths, _sim = forward_with_ladder(model, mix_t, mix_len, wearer_embedding)
            frame_features = taps["R0"]
            T = wl.shape[1]
            length_mask = (torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1))

            state_list = []
            for cid in chunk:
                wt = align_labels_to_frames(npz["wearer_activity"][cid], T)
                et = align_labels_to_frames(npz["environment_activity"][cid], T)
                state_list.append(state_code(wt, et))
            state = np.stack(state_list)  # [B,T]

            x = m0_input_features(taps["R8"], wl, el)
            probs = F.softmax(m0_full(x), dim=-1)  # [B,T,4]

            yield {
                "frame_features": frame_features, "length_mask": length_mask,
                "wearer_embedding": wearer_embedding, "impostor_embedding": impostor_embedding,
                "state": state, "probs": probs, "T": T,
                "tir_tags": [clip_meta[cid]["tir_tag"] for cid in chunk],
            }

    def gather_all(idx_list, chunk_size=64):
        """Runs batched_process over idx_list and concatenates every chunk's
        outputs into single stacked tensors/arrays (fine for <=a few thousand
        clips at this feature size)."""
        ff_l, lm_l, we_l, ie_l, st_l, pr_l, tags = [], [], [], [], [], [], []
        for chunk in batched_process(idx_list, chunk_size):
            ff_l.append(chunk["frame_features"])
            lm_l.append(chunk["length_mask"])
            we_l.append(chunk["wearer_embedding"])
            ie_l.append(chunk["impostor_embedding"])
            st_l.append(chunk["state"])
            pr_l.append(chunk["probs"])
            tags.extend(chunk["tir_tags"])
        return {
            "frame_features": torch.cat(ff_l, dim=0), "length_mask": torch.cat(lm_l, dim=0),
            "wearer_embedding": torch.cat(we_l, dim=0), "impostor_embedding": torch.cat(ie_l, dim=0),
            "state": np.concatenate(st_l, axis=0), "probs": torch.cat(pr_l, dim=0),
            "tir_tags": np.array(tags),
        }

    t1 = time.time()
    calib = gather_all(calib_idx)
    log(f"  calib forward pass done in {time.time() - t1:.1f}s ({calib['state'].shape[0]} clips)")

    calib_length_mask_np = calib["length_mask"].cpu().numpy()

    # ---- calibration: frame-level thresholds for M0A (solo) / M0C (active) ----
    calib_p10_full = calib["probs"][:, :, 1].detach().cpu().numpy()
    calib_pactive_full = (calib["probs"][:, :, 1] + calib["probs"][:, :, 3]).detach().cpu().numpy()
    calib_p10 = calib_p10_full[calib_length_mask_np]
    calib_pactive = calib_pactive_full[calib_length_mask_np]
    calib_state = calib["state"][calib_length_mask_np]
    thr_solo = youden_threshold(calib_p10, (calib_state == 1).astype(int))
    thr_active = youden_threshold(calib_pactive, np.isin(calib_state, [1, 3]).astype(int))
    log(f"  calibrated thresholds: thr_solo(p10)={thr_solo:.4f}  thr_active(p10+p11)={thr_active:.4f}")

    # ---- calibration: precision-target thresholds (Phase 4 recall@precision) ----
    def precision_target_thresholds(scores, labels):
        prec, rec, thr = precision_recall_curve(labels, scores)
        out = {}
        for target in PRECISION_TARGETS:
            ok = np.where(prec[:-1] >= target)[0]
            if len(ok) == 0:
                out[target] = None
            else:
                # smallest threshold achieving the target precision -> maximizes recall
                best_i = ok[np.argmin(thr[ok])]
                out[target] = float(thr[best_i])
        return out

    solo_prec_thr = precision_target_thresholds(calib_p10, (calib_state == 1).astype(int))
    active_prec_thr = precision_target_thresholds(calib_pactive, np.isin(calib_state, [1, 3]).astype(int))
    log(f"  solo precision-target thresholds: {solo_prec_thr}")
    log(f"  active precision-target thresholds: {active_prec_thr}")
    results["calibrated_thresholds"] = {
        "thr_solo_youden": thr_solo, "thr_active_youden": thr_active,
        "solo_precision_target_thresholds": {str(k): v for k, v in solo_prec_thr.items()},
        "active_precision_target_thresholds": {str(k): v for k, v in active_prec_thr.items()},
    }

    def ms_to_frames(ms):
        return max(1, round(ms * 0.1))

    def hierarchical_weights(solo_sel_np, active_sel_np, length_mask_np, solo_f, active_f):
        """solo_sel_np/active_sel_np: [N,T] bool (already AND'ed with validity).
        Returns [N,T] float32: solo mask if its per-clip frame count >= solo_f,
        elif active mask if its count >= active_f, else native length_mask.
        NEVER excludes a clip (falls back to O0-style native pooling)."""
        c_solo = solo_sel_np.sum(axis=1)
        c_active = active_sel_np.sum(axis=1)
        use_solo = c_solo >= solo_f
        use_active = (~use_solo) & (c_active >= active_f)
        return np.where(use_solo[:, None], solo_sel_np,
                         np.where(use_active[:, None], active_sel_np, length_mask_np)).astype(np.float32)

    @torch.no_grad()
    def grid_search_duration_thresholds(solo_sel_np, active_sel_np):
        """16-combo grid (4x4 ms candidates), ONE batched embed_from_weights
        call per combo (not per clip) -- selects the combo minimizing mean
        EER% across TIR buckets on calib['tir_tags']."""
        best_combo, best_eer = None, float("inf")
        for solo_ms in DURATION_CANDIDATES_MS:
            for active_ms in DURATION_CANDIDATES_MS:
                solo_f, active_f = ms_to_frames(solo_ms), ms_to_frames(active_ms)
                weights_np = hierarchical_weights(solo_sel_np, active_sel_np, calib_length_mask_np, solo_f, active_f)
                weights_t = torch.from_numpy(weights_np).to(device)
                emb = embed_from_weights(model, calib["frame_features"], weights_t)
                pos = F.cosine_similarity(emb, calib["wearer_embedding"], dim=-1).detach().cpu().numpy()
                neg = F.cosine_similarity(emb, calib["impostor_embedding"], dim=-1).detach().cpu().numpy()
                per_bucket = []
                for tag in TIR_TAGS:
                    m = calib["tir_tags"] == tag
                    if m.sum() >= 2:
                        per_bucket.append(compute_eer(pos[m], neg[m]) * 100)
                if per_bucket:
                    mean_eer = float(np.mean(per_bucket))
                    if mean_eer < best_eer:
                        best_eer, best_combo = mean_eer, (solo_ms, active_ms)
        return best_combo, best_eer

    # ---- calibration: OH (oracle, ground-truth labels) duration-threshold grid ----
    o2_sel = calib_length_mask_np & (calib["state"] == 1)
    o1_sel = calib_length_mask_np & np.isin(calib["state"], [1, 3])
    best_combo, best_eer = grid_search_duration_thresholds(o2_sel, o1_sel)
    log(f"  OH duration thresholds (recalibrated on Phase-1 calib split): solo={best_combo[0]}ms "
        f"active={best_combo[1]}ms (calib mean EER%={best_eer:.2f})")
    oh_solo_f, oh_active_f = ms_to_frames(best_combo[0]), ms_to_frames(best_combo[1])
    results["oh_recalibrated_thresholds_ms"] = {"solo_ms": best_combo[0], "active_ms": best_combo[1]}

    # ---- calibration: M0E duration-threshold grid (predicted confidence) ----
    m0e_solo_sel = calib_length_mask_np & (calib_p10_full > thr_solo)
    m0e_active_sel = calib_length_mask_np & (calib_pactive_full > thr_active)
    best_combo_m0e, best_eer_m0e = grid_search_duration_thresholds(m0e_solo_sel, m0e_active_sel)
    log(f"  M0E duration thresholds: solo={best_combo_m0e[0]}ms active={best_combo_m0e[1]}ms "
        f"(calib mean EER%={best_eer_m0e:.2f})")
    m0e_solo_f, m0e_active_f = ms_to_frames(best_combo_m0e[0]), ms_to_frames(best_combo_m0e[1])
    results["m0e_thresholds_ms"] = {"solo_ms": best_combo_m0e[0], "active_ms": best_combo_m0e[1]}

    del calib  # free the cached calib frame_features

    # ---------------- eval pass: batched, minimal CUDA syncs ----------------
    log(f"\n  eval forward pass ({len(eval_idx)} clips) -- scoring O0/OH/M0A-E + accumulating frame stats ...")
    t1 = time.time()
    variant_names = ["O0", "OH", "M0A", "M0B", "M0C", "M0D", "M0E"]
    scores_pos_t = {v: [] for v in variant_names}  # lists of un-synced GPU tensors, per chunk
    scores_neg_t = {v: [] for v in variant_names}
    eval_tags_l = []

    contamination = {"solo_youden": [], "active_youden": []}
    for tname in ["solo", "active"]:
        for target in PRECISION_TARGETS:
            contamination[f"{tname}_p{target}"] = []

    p10_masked_l, pactive_masked_l, state_masked_l, pred_state_masked_l = [], [], [], []

    n_chunks_done = 0
    for chunk in batched_process(eval_idx, chunk_size=64):
        ff = chunk["frame_features"]
        length_mask = chunk["length_mask"]  # [B,T] bool GPU
        length_mask_np = length_mask.cpu().numpy()
        p10 = chunk["probs"][:, :, 1]
        p11 = chunk["probs"][:, :, 3]
        pactive = p10 + p11
        state = chunk["state"]  # [B,T] numpy
        we, ie = chunk["wearer_embedding"], chunk["impostor_embedding"]
        tags = np.array(chunk["tir_tags"])
        eval_tags_l.append(tags)

        p10_np = p10.detach().cpu().numpy()
        pactive_np = pactive.detach().cpu().numpy()
        pred_state_np = chunk["probs"].argmax(-1).cpu().numpy()
        p10_masked_l.append(p10_np[length_mask_np])
        pactive_masked_l.append(pactive_np[length_mask_np])
        state_masked_l.append(state[length_mask_np])
        pred_state_masked_l.append(pred_state_np[length_mask_np])

        def emb_score(weights_np, vname):
            w_t = torch.from_numpy(weights_np.astype(np.float32)).to(device)
            emb = embed_from_weights(model, ff, w_t)
            scores_pos_t[vname].append(F.cosine_similarity(emb, we, dim=-1))
            scores_neg_t[vname].append(F.cosine_similarity(emb, ie, dim=-1))

        # O0: native, all valid frames
        emb_score(length_mask_np.astype(np.float32), "O0")

        # OH: oracle hierarchical (ground-truth labels), recalibrated thresholds
        o2_sel_e = length_mask_np & (state == 1)
        o1_sel_e = length_mask_np & np.isin(state, [1, 3])
        w_oh = hierarchical_weights(o2_sel_e, o1_sel_e, length_mask_np, oh_solo_f, oh_active_f)
        emb_score(w_oh, "OH")

        # M0A: hard solo mask
        solo_sel_e = length_mask_np & (p10_np > thr_solo)
        emb_score(solo_sel_e.astype(np.float32), "M0A")

        # M0B: soft solo weights
        emb_score(p10_np * length_mask_np, "M0B")

        # M0C: hard target-active mask
        active_sel_e = length_mask_np & (pactive_np > thr_active)
        emb_score(active_sel_e.astype(np.float32), "M0C")

        # M0D: soft target-active weights
        emb_score(pactive_np * length_mask_np, "M0D")

        # M0E: hierarchical predicted-confidence selector
        w_m0e = hierarchical_weights(solo_sel_e, active_sel_e, length_mask_np, m0e_solo_f, m0e_active_f)
        emb_score(w_m0e, "M0E")

        # contamination bookkeeping at each operating point
        contamination["solo_youden"].append(state[solo_sel_e])
        contamination["active_youden"].append(state[active_sel_e])
        for target in PRECISION_TARGETS:
            if solo_prec_thr[target] is not None:
                sel = length_mask_np & (p10_np > solo_prec_thr[target])
                contamination[f"solo_p{target}"].append(state[sel])
            if active_prec_thr[target] is not None:
                sel = length_mask_np & (pactive_np > active_prec_thr[target])
                contamination[f"active_p{target}"].append(state[sel])

        n_chunks_done += 1

    # single sync per variant, at the very end (deferred throughout the loop above)
    eval_tags = np.concatenate(eval_tags_l)
    scores_pos = {v: {} for v in variant_names}
    scores_neg = {v: {} for v in variant_names}
    for v in variant_names:
        pos_all = torch.cat(scores_pos_t[v]).detach().cpu().numpy()
        neg_all = torch.cat(scores_neg_t[v]).detach().cpu().numpy()
        for tag in TIR_TAGS:
            m = eval_tags == tag
            scores_pos[v][tag] = pos_all[m]
            scores_neg[v][tag] = neg_all[m]

    log(f"  eval forward pass done in {time.time() - t1:.1f}s ({n_chunks_done} chunks)")

    p10_masked = np.concatenate(p10_masked_l)
    pactive_masked = np.concatenate(pactive_masked_l)
    state_masked = np.concatenate(state_masked_l)
    pred_state_masked = np.concatenate(pred_state_masked_l)

    # ---------------- Phase 4 report: confusion matrix + mask quality ----------------
    log("\nPHASE 4 REPORT: confusion matrix + mask quality (on held-out eval suite)")
    cm = confusion_matrix(state_masked, pred_state_masked, labels=[0, 1, 2, 3])
    log(f"  confusion matrix (rows=true, cols=pred, order=00,10,01,11):\n{cm}")
    results["confusion_matrix"] = cm.tolist()
    results["confusion_matrix_labels"] = STATE_NAMES

    y_solo = (state_masked == 1).astype(int)
    y_active = np.isin(state_masked, [1, 3]).astype(int)

    def prf(scores, labels, thr):
        pred = scores > thr
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        return precision, recall

    solo_quality = {
        "auroc": safe_auroc(y_solo, p10_masked), "auprc": safe_auprc(y_solo, p10_masked),
    }
    p, r = prf(p10_masked, y_solo, thr_solo)
    solo_quality["precision_at_youden"] = p
    solo_quality["recall_at_youden"] = r
    for target in PRECISION_TARGETS:
        if solo_prec_thr[target] is not None:
            p, r = prf(p10_masked, y_solo, solo_prec_thr[target])
            solo_quality[f"recall_at_precision_{target}"] = r
            solo_quality[f"actual_precision_{target}"] = p
        else:
            solo_quality[f"recall_at_precision_{target}"] = None
    log(f"  SOLO (state=10) mask quality: {solo_quality}")

    active_quality = {
        "auroc": safe_auroc(y_active, pactive_masked), "auprc": safe_auprc(y_active, pactive_masked),
    }
    p, r = prf(pactive_masked, y_active, thr_active)
    active_quality["precision_at_youden"] = p
    active_quality["recall_at_youden"] = r
    for target in PRECISION_TARGETS:
        if active_prec_thr[target] is not None:
            p, r = prf(pactive_masked, y_active, active_prec_thr[target])
            active_quality[f"recall_at_precision_{target}"] = r
            active_quality[f"actual_precision_{target}"] = p
        else:
            active_quality[f"recall_at_precision_{target}"] = None
    log(f"  TARGET-ACTIVE (10∪11) mask quality: {active_quality}")
    results["solo_mask_quality"] = solo_quality
    results["target_active_mask_quality"] = active_quality

    # contamination composition table
    contamination_table = {}
    for key, arrs in contamination.items():
        if not arrs:
            continue
        all_states = np.concatenate(arrs) if any(len(a) for a in arrs) else np.array([], dtype=int)
        if len(all_states) == 0:
            contamination_table[key] = {"n_selected": 0}
            continue
        contamination_table[key] = {
            "n_selected": int(len(all_states)),
            **{f"pct_{STATE_NAMES[i]}": float((all_states == i).mean() * 100) for i in range(4)},
        }
    log(f"  contamination table: {json.dumps(contamination_table, indent=2)}")
    results["contamination_table"] = contamination_table

    # ---------------- Phase 7: EER/AUROC + bootstrap CIs ----------------
    log("\nPHASE 7: native-pooling EER/AUROC with bootstrap 95% CIs")

    def bootstrap_eer_auroc(pos, neg, n_boot, seed=0):
        rng = np.random.RandomState(seed)
        pos, neg = np.asarray(pos), np.asarray(neg)
        n_p, n_n = len(pos), len(neg)
        if n_p < 2 or n_n < 2:
            return {"eer_pct": float("nan"), "eer_ci": [float("nan"), float("nan")],
                    "auroc": float("nan"), "auroc_ci": [float("nan"), float("nan")], "n_pos": n_p, "n_neg": n_n}
        eer0 = compute_eer(pos, neg) * 100
        labels0 = np.concatenate([np.ones(n_p), np.zeros(n_n)])
        auroc0 = safe_auroc(labels0, np.concatenate([pos, neg]))
        eers, aurocs = [], []
        for _ in range(n_boot):
            pb = pos[rng.randint(0, n_p, n_p)]
            nb = neg[rng.randint(0, n_n, n_n)]
            eers.append(compute_eer(pb, nb) * 100)
            labels_b = np.concatenate([np.ones(n_p), np.zeros(n_n)])
            aurocs.append(safe_auroc(labels_b, np.concatenate([pb, nb])))
        eers, aurocs = np.array(eers), np.array(aurocs)
        return {
            "eer_pct": eer0, "eer_ci": [float(np.percentile(eers, 2.5)), float(np.percentile(eers, 97.5))],
            "auroc": auroc0, "auroc_ci": [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))],
            "n_pos": n_p, "n_neg": n_n,
        }

    pooling_table = {v: {} for v in variant_names}
    for v in variant_names:
        for tag in TIR_TAGS:
            pooling_table[v][tag] = bootstrap_eer_auroc(scores_pos[v][tag], scores_neg[v][tag], args.n_boot)
        allp = np.concatenate([scores_pos[v][t] for t in TIR_TAGS])
        alln = np.concatenate([scores_neg[v][t] for t in TIR_TAGS])
        pooling_table[v]["ALL"] = bootstrap_eer_auroc(allp, alln, args.n_boot)

    header = f"{'variant':<8}" + "".join(f"{DISPLAY_TAGS[t]:>10}" for t in TIR_TAGS) + f"{'ALL':>10}"
    log("\n  EER% (95% CI in brackets):")
    log("  " + header)
    for v in variant_names:
        row = f"  {v:<8}"
        for tag in list(TIR_TAGS) + ["ALL"]:
            row += f"{pooling_table[v][tag]['eer_pct']:>10.2f}"
        log(row)
    results["pooling_results_table"] = pooling_table

    # ---------------- gain_recovery ----------------
    log("\nORACLE GAIN RECOVERED")
    mean_eer_by_variant = {v: np.mean([pooling_table[v][t]["eer_pct"] for t in TIR_TAGS
                                        if not np.isnan(pooling_table[v][t]["eer_pct"])]) for v in ["M0A", "M0B", "M0C", "M0D", "M0E"]}
    best_m0 = min(mean_eer_by_variant, key=mean_eer_by_variant.get)
    log(f"  best M0 variant by mean EER across buckets: {best_m0} (mean EER%={mean_eer_by_variant[best_m0]:.2f})")
    gain_recovery = {}
    for tag in TIR_TAGS:
        o0e = pooling_table["O0"][tag]["eer_pct"]
        ohe = pooling_table["OH"][tag]["eer_pct"]
        m0e_ = pooling_table[best_m0][tag]["eer_pct"]
        denom = o0e - ohe
        if np.isnan(o0e) or np.isnan(ohe) or np.isnan(m0e_) or abs(denom) < 0.5:
            gain_recovery[tag] = {"value": None, "note": "denominator too small / undefined"}
        else:
            gain_recovery[tag] = {"value": float((o0e - m0e_) / denom), "o0_eer": o0e, "oh_eer": ohe, "m0_eer": m0e_}
        log(f"  {DISPLAY_TAGS[tag]}: {gain_recovery[tag]}")
    results["best_m0_variant"] = best_m0
    results["gain_recovery"] = gain_recovery

    # ---------------- Phase 8: streaming cost ----------------
    log("\nPHASE 8: streaming feasibility + cost")
    torch.cuda.synchronize() if device.startswith("cuda") else None
    n_timing = 200
    dummy = torch.randn(1, 208, 195, device=device)
    with torch.no_grad():
        for _ in range(10):
            m0_full(dummy)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(n_timing):
            m0_full(dummy)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - t1) / n_timing * 1000
    log(f"  M0 forward pass alone: {elapsed_ms:.3f} ms/clip (T=208 frames, batch=1, device={device})")
    ram_estimate_kb = (3 * 1500 * 4) / 1024  # sum_w, sum_wx, sum_wx2 rolling buffers, fp32
    results["streaming_cost"] = {
        "m0_param_count": n_params,
        "m0_forward_ms_per_clip": elapsed_ms,
        "streaming_running_stats_kb_per_active_clip": ram_estimate_kb,
        "streaming_replay_max_abs_diff": stream_diff,
        "note": "Rolling weighted moments (sum_w, sum_w*x, sum_w*x^2 per of the 1500 backbone "
                "feature channels) update in O(1) per incoming frame and reproduce "
                "weighted_masked_stats_pool's batch result exactly (see streaming_replay check "
                "above) -- M0 itself is already a per-frame-causal MLP (no temporal receptive "
                "field), so the full M0+weighted-pooling pipeline could run online with only "
                "the frozen model's own existing causal TCN as the streaming bottleneck.",
    }
    log(json.dumps(results["streaming_cost"], indent=2))

    # ---------------- fingerprint after ----------------
    fp_after = fingerprint(model)
    results["fingerprint_before"] = fp_before
    results["fingerprint_after"] = fp_after
    results["fingerprint_match"] = (fp_before == fp_after)
    assert fp_after == fp_before, "FROZEN MODEL FINGERPRINT CHANGED"
    log(f"\nfingerprint after: {fp_after}  match={fp_before == fp_after}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    log(f"\nsaved {args.out}")
    log(f"total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
