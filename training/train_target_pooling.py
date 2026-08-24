#!/usr/bin/env python3
"""P0 experiment: train a small enrollment-aware attention pooler + fusion
head on top of the FROZEN mentrawearnet_v2_step4800.pt checkpoint, on a
genuine WINDOW-level (clip-level) objective -- testing whether trained
attention pooling can recover clip-level verification performance that the
existing checkpoint loses to naive mean-pooling (measured: clip EER 18.75%/
37.50%/45-50% clean/TIR0/TIR-10 vs frame-level 10-vs-01 AUROC 0.78-0.87).

EVERYTHING already trained is frozen: the SpeakerNet backbone (including its
embedding_projection) and the entire loaded MentraWearNet module tree
(frame_projection, FiLM, similarity branch, TCN, wearer_head,
environment_head). The frozen model is put into eval() ONCE after loading
and .train() is NEVER called on it again -- this sidesteps, by construction,
the exact BatchNorm-drift-under-train-mode bug class documented in
training/models/speakernet_backbone.py's SpeakerNetBackbone.train()
docstring (that bug required the outer model.train() to be called every
step; here the outer frozen model's .train() is simply never invoked, only
the two new small modules' .train() is).

Only two new small modules are trained: TargetAwarePooler and FusionHead
(training/models/target_aware_pooling.py), combined well under 100k params.
Random negatives only (mismatched-speaker mixtures) -- no hard-negative
mining, that's explicitly P2, out of scope here.

Follows the same run_dir / self-contained log() / step-numbered
save_checkpoint() pattern as training/train.py (added recently specifically
to stop losing checkpoint history).

Usage:
    .venv/bin/python3 training/train_target_pooling.py --device cuda --steps 1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from training.data.mixture_generator import (
    MixtureExample, NoisePool, SAMPLE_RATE, SAMPLES_PER_FRAME, SpeakerPool,
    _add_noise, _fit_or_loop, _rms, generate_example, sample_duration_s, sample_tir_db,
)
from training.models.mentrawearnet import MentraWearNet
from training.models.target_aware_pooling import FusionHead, TargetAwarePooler

POSITIVE_LABEL_THRESHOLD = 0.02  # fraction of sample-resolution wearer_activity that counts as "wearer present"


def generate_negative_example(pool: SpeakerPool, rng: random.Random, total_s: float,
                               noise_pool: NoisePool) -> MixtureExample:
    """A window built ONLY from two speakers, NEITHER of which is the
    enrolled `wearer_id` used for verification -- the enrolled wearer's
    voice is entirely absent, so wearer_activity is all-zero by construction
    (window label = 0, guaranteed, no threshold ambiguity). Same segment-
    structure variety (SILENCE / speaker-A / speaker-B / OVERLAP) and the
    same low-level helpers (_fit_or_loop, _rms, sample_tir_db, _add_noise) as
    generate_example() in mixture_generator.py, so negatives are drawn from
    the same distribution of durations/transitions/TIR as positives -- only
    the identity of who's enrolled differs. Random selection only, no
    hard-negative mining (that's P2, out of scope for P0)."""
    wearer_id = rng.choice(pool.speaker_ids)  # enrolled identity -- absent from the mixture
    others = [s for s in pool.speaker_ids if s != wearer_id]
    a_id, b_id = rng.sample(others, 2)

    n_samples = int(total_s * SAMPLE_RATE)
    n_plan_chunks = n_samples // SAMPLES_PER_FRAME
    n_samples = n_plan_chunks * SAMPLES_PER_FRAME

    states = ["SILENCE", "A", "B", "OVERLAP"]
    weights = [0.15, 0.35, 0.35, 0.15]
    n_segments = rng.randint(2, 4)
    segment_states = rng.choices(states, weights=weights, k=n_segments)
    cut_points = sorted(rng.sample(range(1, n_plan_chunks), n_segments - 1)) if n_segments > 1 else []
    boundaries = [0] + cut_points + [n_plan_chunks]

    mixture = np.zeros(n_samples, dtype=np.float32)
    wearer_activity = np.zeros(n_samples, dtype=np.float32)  # always zero -- enrolled wearer never speaks
    environment_activity = np.zeros(n_samples, dtype=np.float32)

    a_clip = pool.random_clip(a_id, "test", rng)
    b_clip = pool.random_clip(b_id, "test", rng)

    for i, state in enumerate(segment_states):
        c0, c1 = boundaries[i], boundaries[i + 1]
        if c1 <= c0:
            continue
        s0 = c0 * SAMPLES_PER_FRAME
        seg_samples = (c1 - c0) * SAMPLES_PER_FRAME
        if state == "SILENCE":
            pass
        elif state == "A":
            mixture[s0:s0 + seg_samples] = _fit_or_loop(a_clip, seg_samples, rng)
            environment_activity[s0:s0 + seg_samples] = 1.0
        elif state == "B":
            mixture[s0:s0 + seg_samples] = _fit_or_loop(b_clip, seg_samples, rng)
            environment_activity[s0:s0 + seg_samples] = 1.0
        elif state == "OVERLAP":
            a = _fit_or_loop(a_clip, seg_samples, rng)
            b = _fit_or_loop(b_clip, seg_samples, rng)
            tir_db = sample_tir_db(rng)
            b_scaled = b * ((_rms(a) / (10 ** (tir_db / 20))) / _rms(b))
            mixture[s0:s0 + seg_samples] = a + b_scaled
            environment_activity[s0:s0 + seg_samples] = 1.0

    if noise_pool is not None and noise_pool.available:
        mixture = _add_noise(mixture, noise_pool, rng)
    else:
        mixture = mixture + np.random.RandomState(rng.randint(0, 2**31)).normal(0, 1e-4, size=mixture.shape).astype(np.float32)
    peak = np.max(np.abs(mixture))
    if peak > 1.0:
        mixture = mixture / peak

    enrollment = pool.enrollment_clip(wearer_id, rng)
    return MixtureExample(
        mixture=mixture, enrollment=enrollment,
        wearer_activity=wearer_activity, environment_activity=environment_activity,
        metadata={"wearer_id": wearer_id, "other_id": None, "a_id": a_id, "b_id": b_id,
                  "is_negative_construction": True},
    )


def window_label(example: MixtureExample) -> float:
    return 1.0 if float(example.wearer_activity.mean()) > POSITIVE_LABEL_THRESHOLD else 0.0


def pad_batch(examples: list[MixtureExample], device: str):
    batch_size = len(examples)
    mix_len = max(len(e.mixture) for e in examples)
    enr_len = max(len(e.enrollment) for e in examples)

    mixtures = torch.zeros(batch_size, mix_len)
    mixture_lengths = torch.zeros(batch_size, dtype=torch.long)
    enrollments = torch.zeros(batch_size, enr_len)
    enrollment_lengths = torch.zeros(batch_size, dtype=torch.long)
    labels = torch.zeros(batch_size)

    for i, e in enumerate(examples):
        mixtures[i, :len(e.mixture)] = torch.from_numpy(e.mixture)
        mixture_lengths[i] = len(e.mixture)
        enrollments[i, :len(e.enrollment)] = torch.from_numpy(e.enrollment)
        enrollment_lengths[i] = len(e.enrollment)
        labels[i] = window_label(e)

    return (mixtures.to(device), mixture_lengths.to(device),
            enrollments.to(device), enrollment_lengths.to(device), labels.to(device))


def frozen_signature(model: torch.nn.Module) -> str:
    """Deterministic fingerprint over every parameter AND buffer of the
    frozen model (buffers matter -- BatchNorm running_mean/running_var are
    buffers, not parameters, and are exactly what drifted in the historical
    bug this project already hit once). Used to prove, before vs after
    training, that nothing in the frozen model moved."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def frozen_features(frozen_model: MentraWearNet, mixtures, mixture_lengths, enrollments, enrollment_lengths):
    """Everything derived purely from the frozen model, under no_grad (no
    trainable parameters are involved anywhere in this function -- safe to
    fully detach)."""
    wearer_embedding = frozen_model.encode_enrollment(enrollments, enrollment_lengths)  # [B,256]
    h_t, frame_lengths = frozen_model.backbone.encode_frames(mixtures, mixture_lengths)  # [B,1500,T]
    out = frozen_model.process_with_embedding(mixtures, mixture_lengths, wearer_embedding)
    wearer_logits, environment_logits = out["wearer_logits"], out["environment_logits"]
    mixture_speaker_embedding = frozen_model.backbone.encode_speaker(mixtures, mixture_lengths)  # [B,256]
    s_original = (wearer_embedding * mixture_speaker_embedding).sum(dim=-1)  # both L2-normalized -> cosine
    return wearer_embedding, h_t, frame_lengths, wearer_logits, environment_logits, s_original


def compute_scalar_features(wearer_logits, environment_logits, frame_lengths):
    B, T = wearer_logits.shape
    device = wearer_logits.device
    mask = (torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1)).float()
    denom = mask.sum(dim=1).clamp(min=1)
    d_t = wearer_logits - environment_logits
    s_frame = (d_t * mask).sum(dim=1) / denom
    wp = torch.sigmoid(wearer_logits)
    ep = torch.sigmoid(environment_logits)
    overlap = ((wp > 0.5) & (ep > 0.5)).float() * mask
    overlap_fraction = overlap.sum(dim=1) / denom
    return s_frame, overlap_fraction, wp, ep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen-checkpoint", default="training/checkpoints/mentrawearnet_v2_step4800.pt")
    ap.add_argument("--manifest", default="evaluation/manifests/librispeech_train_clean_100_train.json",
                     help="training-speaker manifest -- zero overlap with the held-out validation "
                          "manifest used by validation_suite_v2.npz (211 vs 20 speakers, verified elsewhere)")
    ap.add_argument("--noise-dir", default="evaluation/data/raw/musan/noise")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--proj-dim", type=int, default=32)
    ap.add_argument("--attn-hidden", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--checkpoint-out", default="training/checkpoints/p0_target_pooling_latest.pt",
                     help="stable pointer, always updated to the latest checkpoint; full history "
                          "lives under --run-dir, step-numbered, never overwritten")
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-p0-target-pooling"
    run_dir = Path(args.run_dir) if args.run_dir else Path("training/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "train.log", "a")

    def log(msg: str, err: bool = False):
        stream = sys.stderr if err else sys.stdout
        print(msg, file=stream, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log(f"run_dir={run_dir}  (self-contained log + full checkpoint history live here)")
    log(f"device={args.device}")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    # --- frozen model: load, freeze, eval() ONCE, never train() again ---
    frozen_model = MentraWearNet().to(args.device)
    ckpt = torch.load(args.frozen_checkpoint, map_location=args.device, weights_only=False)
    frozen_model.load_state_dict(ckpt["model_state_dict"])
    for p in frozen_model.parameters():
        p.requires_grad = False
    frozen_model.eval()
    assert not frozen_model.training
    assert all(not p.requires_grad for p in frozen_model.parameters())
    log(f"loaded frozen checkpoint {args.frozen_checkpoint} (step={ckpt.get('step')}, "
        f"final_loss={ckpt.get('final_loss')})")
    log(f"frozen model: {sum(p.numel() for p in frozen_model.parameters()):,} params, "
        f"all requires_grad=False, eval() mode")
    sig_before = frozen_signature(frozen_model)
    log(f"frozen_signature (params+buffers sha256, before training) = {sig_before}")

    # --- new trainable modules ---
    pooler = TargetAwarePooler(frame_channels=frozen_model.backbone.frame_channels,
                                embedding_dim=frozen_model.backbone.embedding_dim,
                                proj_dim=args.proj_dim, attn_hidden=args.attn_hidden).to(args.device)
    fusion_head = FusionHead().to(args.device)
    trainable_params = list(pooler.parameters()) + list(fusion_head.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    log(f"NEW trainable params: pooler={pooler.n_params:,}  fusion_head={fusion_head.n_params:,}  "
        f"total={n_trainable:,}  (budget: <200k, ideally <100k)")
    assert n_trainable < 200_000

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    pool = SpeakerPool(args.manifest)
    noise_pool = NoisePool(args.noise_dir)
    log(f"speaker pool: {len(pool.speaker_ids)} speakers from {args.manifest}")
    log(f"noise pool: {len(noise_pool.clips)} clips from {args.noise_dir}" if noise_pool.available
        else "noise pool: none (training without additive noise)")

    def save_checkpoint(completed_steps: int, final_loss: float, final_acc: float):
        payload = {
            "pooler_state_dict": pooler.state_dict(),
            "fusion_head_state_dict": fusion_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": completed_steps,
            "config": vars(args),
            "final_loss": final_loss,
            "final_acc": final_acc,
            "run_dir": str(run_dir),
            "frozen_checkpoint": args.frozen_checkpoint,
            "n_trainable_params": n_trainable,
        }
        step_path = run_dir / f"checkpoint_step{completed_steps}.pt"
        step_tmp = step_path.with_suffix(".pt.tmp")
        torch.save(payload, step_tmp)
        import os
        os.replace(step_tmp, step_path)

        stable_tmp = f"{args.checkpoint_out}.tmp"
        Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, stable_tmp)
        os.replace(stable_tmp, args.checkpoint_out)
        return step_path

    losses, accs = [], []
    t0 = time.time()
    for step in range(args.steps):
        pooler.train()
        fusion_head.train()

        batch = []
        for _ in range(args.batch_size):
            dur = sample_duration_s(rng)
            if rng.random() < 0.5:
                batch.append(generate_example(pool, rng, total_s=dur, noise_pool=noise_pool))
            else:
                batch.append(generate_negative_example(pool, rng, total_s=dur, noise_pool=noise_pool))

        mixtures, mixture_lengths, enrollments, enrollment_lengths, labels = pad_batch(batch, args.device)

        wearer_embedding, h_t, frame_lengths, wearer_logits, environment_logits, s_original = frozen_features(
            frozen_model, mixtures, mixture_lengths, enrollments, enrollment_lengths)
        s_frame, overlap_fraction, wp, ep = compute_scalar_features(wearer_logits, environment_logits, frame_lengths)

        target_embedding = pooler(h_t, frame_lengths, wearer_embedding,
                                   wearer_logits, environment_logits, wp, ep,
                                   frozen_model.backbone.embedding_projection)
        s_target_pool = F.cosine_similarity(target_embedding, wearer_embedding, dim=-1)

        fusion_logit = fusion_head(s_original, s_target_pool, s_frame, overlap_fraction)
        loss = F.binary_cross_entropy_with_logits(fusion_logit, labels)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            pred = (torch.sigmoid(fusion_logit) > 0.5).float()
            acc = (pred == labels).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

        if not torch.isfinite(loss):
            log(f"FAIL: non-finite loss at step {step}", err=True)
            sys.exit(1)

        if step % args.log_every == 0 or step == args.steps - 1:
            recent_l = losses[-args.log_every:]
            recent_a = accs[-args.log_every:]
            log(f"step {step:4d}  loss={loss.item():.4f}  loss_avg{args.log_every}={sum(recent_l)/len(recent_l):.4f}  "
                f"acc={acc:.3f}  acc_avg{args.log_every}={sum(recent_a)/len(recent_a):.3f}  "
                f"grad_norm={grad_norm:.3f}  n_pos={int(labels.sum().item())}/{len(labels)}  "
                f"elapsed={time.time()-t0:.1f}s")

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            step_path = save_checkpoint(step + 1, loss.item(), acc)
            log(f"checkpoint saved at step {step + 1} -> {step_path}")

    step_path = save_checkpoint(args.steps, losses[-1], accs[-1])
    log(f"saved final checkpoint to {args.checkpoint_out} (full history: {step_path})")
    log(f"loss: start={losses[0]:.4f} end={losses[-1]:.4f}   acc: start={accs[0]:.3f} end={accs[-1]:.3f}")

    # --- prove the frozen model did not drift ---
    sig_after = frozen_signature(frozen_model)
    log(f"frozen_signature (params+buffers sha256, after training)  = {sig_after}")
    if sig_before == sig_after:
        log("VERIFIED: frozen model signature unchanged before vs after training -- nothing drifted.")
    else:
        log("FAIL: frozen model signature CHANGED during training -- something was NOT frozen correctly!", err=True)
        sys.exit(1)

    log_file.close()


if __name__ == "__main__":
    main()
