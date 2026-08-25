#!/usr/bin/env python3
"""V3C-MA0 (mixture-aware enrollment conditioning, stage 0) training driver.

Trains ONLY a MixtureAwareEnrollmentAdapter's own parameters
(training/models/mentrawearnet.py) on top of a frozen STATIC_BASE
checkpoint (mentrawearnet_v3b_control_latest.pt). Backbone, projection,
FiLM, TCN, and every existing head stay frozen at STATIC_BASE's weights --
verified via a full param+buffer fingerprint before and after training
(see fingerprint_frozen() below).

Two branches, same script, controlled by --adapter-mode:
  static  -- STATIC_ADAPTER control (mixture information zeroed inside the
             adapter, see MixtureAwareEnrollmentAdapter's zero_mixture)
  mixture -- MIXTURE_AWARE (the real causal question)

Reuses train.py's build_batch/compute_loss/align_labels_to_frames
UNCHANGED (same loss, same weights, same margin, same label-alignment
logic) -- only the model construction, freezing, and optimizer differ.

Frozen model is .eval()'d ONCE after loading and never .train()'d again
(matches the project's established BatchNorm-safety pattern in
speakernet_backbone.py's train() override, and additionally avoids
injecting stochastic dropout noise from a frozen, non-training TCN into
gradients that only the adapter receives).

Checkpoint selection is validation-loss-based (held-out
librispeech_train_clean_100_val.json, a FIXED example set generated once
per run and reused at every eval point -- never the hard-eval-suite eval
split, never the historical tiny day1_public set), with a simple
patience-based early-stop rule.

Usage:
  .venv/bin/python3 training/train_adapter.py --adapter-mode static  --device cuda:0 --run-tag v3c_ma0_static
  .venv/bin/python3 training/train_adapter.py --adapter-mode mixture --device cuda:1 --run-tag v3c_ma0_mixture
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

import torch

from training.data.mixture_generator import SpeakerPool, generate_example, sample_duration_s, NoisePool
from training.models.mentrawearnet import MentraWearNet
from training.train import build_batch, compute_loss

STATIC_BASE_CHECKPOINT = "training/checkpoints/mentrawearnet_v3b_control_latest.pt"
TRAIN_MANIFEST = "evaluation/manifests/librispeech_train_clean_100_train.json"
VAL_MANIFEST = "evaluation/manifests/librispeech_train_clean_100_val.json"
NOISE_DIR = "evaluation/data/raw/musan/noise"

# V3A/V3B-control's exact loss weights (see mentrawearnet_v3b_control_latest.pt's
# own config.json) -- reused unchanged per spec ("same weights, same margin").
AUX_LOSS_WEIGHT = 0.25
SPEAKER_DISC_WEIGHT = 0.10
SPEAKER_DISC_MARGIN = 0.3


def fingerprint_frozen(model: MentraWearNet) -> str:
    """SHA-256 over every FROZEN (non-adapter) param + buffer, name-sorted
    for determinism. Adapter's own tensors are excluded on purpose -- they
    are SUPPOSED to change during training; this fingerprint is the
    "everything else stayed byte-identical" check."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if name.startswith("adapter."):
            continue
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    for name, b in sorted(model.named_buffers()):
        if name.startswith("adapter."):
            continue
        h.update(name.encode())
        h.update(b.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def eval_on_fixed_set(model, examples, device):
    """Mean loss + wearer/env acc over a FIXED example list (val set),
    no gradient. Returns (loss, wearer_acc, env_acc, extra_info)."""
    model.eval()
    out, wearer_targets, env_targets, frame_lengths = build_batch(examples, model, device)
    loss, wearer_acc, env_acc, extra_info = compute_loss(
        out, wearer_targets, env_targets, frame_lengths,
        aux_losses=True, aux_weight=AUX_LOSS_WEIGHT,
        speaker_disc=True, disc_weight=SPEAKER_DISC_WEIGHT, disc_margin=SPEAKER_DISC_MARGIN,
    )
    return loss.item(), wearer_acc, env_acc, extra_info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-mode", required=True, choices=["static", "mixture"])
    ap.add_argument("--adapter-hidden-dim", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=320)  # matches V3B-control exactly
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--val-set-size", type=int, default=64)
    ap.add_argument("--patience-evals", type=int, default=3,
                     help="stop early if val loss hasn't improved by more than --min-delta "
                          "for this many consecutive eval points")
    ap.add_argument("--min-delta", type=float, default=0.001)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--checkpoint-base", default=STATIC_BASE_CHECKPOINT)
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + args.run_tag
    run_dir = Path("training/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "train.log", "a")

    def log(msg: str):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    log(f"run_dir={run_dir}")
    log(f"adapter_mode={args.adapter_mode}  device={args.device}")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    val_rng = random.Random(999999 + args.seed)  # separate stream, fixed, independent of training draws

    device = args.device
    model = MentraWearNet(aux_losses=True, adapter_mode=args.adapter_mode,
                           adapter_hidden_dim=args.adapter_hidden_dim).to(device)
    log(f"adapter param count: {model.adapter.n_params:,}")

    base_ckpt = torch.load(args.checkpoint_base, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(base_ckpt["model_state_dict"], strict=False)
    base_sha256 = hashlib.sha256(Path(args.checkpoint_base).read_bytes()).hexdigest()
    log(f"loaded STATIC_BASE from {args.checkpoint_base} "
        f"(source step={base_ckpt.get('step')}, source final_loss={base_ckpt.get('final_loss'):.4f}, "
        f"file sha256={base_sha256})")
    log(f"  missing_keys (adapter.*, fresh init, expected): {missing}")
    log(f"  unexpected_keys (expected empty): {unexpected}")
    assert all(k.startswith("adapter.") for k in missing), \
        f"unexpected missing keys outside adapter.*: {missing}"
    assert unexpected == [], f"unexpected keys in checkpoint not present in model: {unexpected}"

    # Residual-init sanity check (spec: "verify this numerically", report
    # actual max|e_t - e|) -- run once on a REAL batch, not synthetic noise.
    pool = SpeakerPool(TRAIN_MANIFEST)
    noise_pool = NoisePool(NOISE_DIR)
    sanity_examples = [generate_example(pool, rng, total_s=sample_duration_s(rng), noise_pool=noise_pool)
                        for _ in range(4)]
    with torch.no_grad():
        out0, _, _, _ = build_batch(sanity_examples, model, device)
        max_diff = (out0["adapted_embedding"] - out0["static_embedding"].unsqueeze(1)).abs().max().item()
        alpha_val = model.adapter.alpha.item()
    log(f"RESIDUAL INIT CHECK: alpha={alpha_val}  max|e_t - e| at step 0 (real batch) = {max_diff}")

    model.freeze_all_except_adapter()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"trainable params: {trainable:,} / total: {total:,}")

    fp_before = fingerprint_frozen(model)
    log(f"FROZEN MODEL FINGERPRINT (before training): {fp_before}")

    # eval() ONCE, never train() again -- see module docstring.
    model.eval()

    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=args.lr)

    val_pool = SpeakerPool(VAL_MANIFEST)
    val_examples = [generate_example(val_pool, val_rng, total_s=sample_duration_s(val_rng), noise_pool=noise_pool)
                     for _ in range(args.val_set_size)]
    log(f"validation set: {len(val_examples)} FIXED examples from {VAL_MANIFEST}, reused at every eval point")

    def save_checkpoint(step, loss_val):
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": vars(args),
            "final_loss": loss_val,
            "run_dir": str(run_dir),
            "adapter_mode": args.adapter_mode,
            "static_base_checkpoint": args.checkpoint_base,
            "static_base_sha256": base_sha256,
        }
        step_path = run_dir / f"checkpoint_step{step}.pt"
        tmp = step_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.rename(step_path)
        stable = Path(f"training/checkpoints/mentrawearnet_v3c_ma0_{args.run_tag}_latest.pt")
        stable_tmp = stable.with_suffix(".pt.tmp")
        torch.save(payload, stable_tmp)
        stable_tmp.rename(stable)
        return step_path

    val_history = []  # (step, val_loss)
    best_val_loss = float("inf")
    best_step = 0
    evals_since_improvement = 0
    stopped_early = False

    # Step-0 eval (before any adapter training) -- required by the spec's
    # "evaluate at 0/250/500/..." schedule.
    val_loss0, val_wacc0, val_eacc0, val_extra0 = eval_on_fixed_set(model, val_examples, device)
    log(f"EVAL step 0  val_loss={val_loss0:.4f}  val_wearer_acc={val_wacc0:.3f}  val_env_acc={val_eacc0:.3f}"
        + "".join(f"  {k}={v:.4f}" for k, v in val_extra0.items() if isinstance(v, float)))
    val_history.append((0, val_loss0))
    best_val_loss = val_loss0
    save_checkpoint(0, val_loss0)

    losses = []
    t0 = time.time()
    for step in range(args.steps):
        batch = [generate_example(pool, rng, total_s=sample_duration_s(rng), noise_pool=noise_pool)
                 for _ in range(args.batch_size)]
        out, wearer_targets, env_targets, frame_lengths = build_batch(batch, model, device)
        loss, wearer_acc, env_acc, extra_info = compute_loss(
            out, wearer_targets, env_targets, frame_lengths,
            aux_losses=True, aux_weight=AUX_LOSS_WEIGHT,
            speaker_disc=True, disc_weight=SPEAKER_DISC_WEIGHT, disc_margin=SPEAKER_DISC_MARGIN,
        )
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0 or step == args.steps - 1:
            recent = losses[-50:]
            with torch.no_grad():
                gate_mean = out["adapter_gate"].mean().item() if "adapter_gate" in out else float("nan")
                delta_norm = out["adapter_delta"].norm(dim=-1).mean().item() if "adapter_delta" in out else float("nan")
            log(f"step {step:4d}  loss={loss.item():.4f}  loss_avg50={sum(recent)/len(recent):.4f}  "
                f"wearer_acc={wearer_acc:.3f}  env_acc={env_acc:.3f}  lr={args.lr:.2e}  "
                f"grad_norm={grad_norm:.3f}  alpha={model.adapter.alpha.item():.5f}  "
                f"gate_mean={gate_mean:.4f}  delta_norm={delta_norm:.4f}  "
                f"elapsed={time.time()-t0:.1f}s")

        if not torch.isfinite(loss):
            log(f"FAIL: non-finite loss at step {step}")
            sys.exit(1)

        completed = step + 1
        if completed % args.eval_every == 0 or completed == args.steps:
            val_loss, val_wacc, val_eacc, val_extra = eval_on_fixed_set(model, val_examples, device)
            log(f"EVAL step {completed}  val_loss={val_loss:.4f}  val_wearer_acc={val_wacc:.3f}  "
                f"val_env_acc={val_eacc:.3f}"
                + "".join(f"  {k}={v:.4f}" for k, v in val_extra.items() if isinstance(v, float)))
            val_history.append((completed, val_loss))
            save_checkpoint(completed, val_loss)

            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                best_step = completed
                evals_since_improvement = 0
            else:
                evals_since_improvement += 1
                log(f"  no val improvement > {args.min_delta} for {evals_since_improvement} eval(s) "
                    f"(best so far: step {best_step}, val_loss={best_val_loss:.4f})")
                if evals_since_improvement >= args.patience_evals:
                    log(f"EARLY STOP at step {completed}: patience ({args.patience_evals} evals) exhausted")
                    stopped_early = True
                    break

    fp_after = fingerprint_frozen(model)
    fingerprint_match = (fp_after == fp_before)
    log(f"FROZEN MODEL FINGERPRINT (after training): {fp_after}")
    log(f"FROZEN MODEL INTEGRITY: {'IDENTICAL' if fingerprint_match else 'MISMATCH -- FROZEN PARAMS CHANGED!'}")
    assert fingerprint_match, "frozen (non-adapter) params/buffers changed during adapter-only training!"

    summary = {
        "run_dir": str(run_dir),
        "adapter_mode": args.adapter_mode,
        "adapter_param_count": model.adapter.n_params,
        "alpha_init": alpha_val,
        "max_e_t_minus_e_at_init": max_diff,
        "static_base_checkpoint": args.checkpoint_base,
        "static_base_sha256": base_sha256,
        "fp_before": fp_before,
        "fp_after": fp_after,
        "fingerprint_match": fingerprint_match,
        "val_loss_history": val_history,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "stopped_early": stopped_early,
        "total_steps_run": len(losses),
        "final_alpha": model.adapter.alpha.item(),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"BEST CHECKPOINT: step {best_step}  val_loss={best_val_loss:.4f}")
    log(f"saved summary to {run_dir / 'summary.json'}")
    log_file.close()


if __name__ == "__main__":
    main()
