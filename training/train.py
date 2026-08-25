#!/usr/bin/env python3
"""Real training entrypoint for MentraWearNet (Track D). Supports both
single-process (plain `python3 training/train.py`) and real DDP across
both RTX 3090s (`torchrun --nproc_per_node=2 training/train.py`) -- the
DDP path was smoke-tested in isolation first
(scripts/training/ddp_smoke_test.py) before wiring it into the actual
training loop, matching the plan's ordering: infra proven correct before
depending on it for a real run.

Each DDP rank gets a DIFFERENT seed (base_seed + rank) so the two GPUs
generate genuinely different mixtures rather than duplicating identical
work -- using DDP with identical per-rank data would waste the second GPU
even though it looks busy.

Frame label alignment (the real bug class this project was warned about):
mixture_generator.py produces SAMPLE-resolution labels. This script aligns
them to the model's ACTUAL encoder frame count (read from a real forward
pass, not assumed from a fixed stride) via block-max pooling -- a frame is
labeled active if ANY sample in its span was active, which is the correct
direction of error for an activity-detection task (a boundary frame should
count as active rather than being averaged away).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from training.data.mixture_generator import SpeakerPool, generate_example, sample_duration_s, NoisePool
from training.models.mentrawearnet import MentraWearNet


def align_labels_to_frames(activity_samples: np.ndarray, target_frames: int) -> np.ndarray:
    """Block-max-pools a sample-resolution {0,1} activity array down to
    exactly target_frames labels, regardless of the model's actual stride."""
    n = len(activity_samples)
    edges = np.linspace(0, n, target_frames + 1).astype(int)
    out = np.zeros(target_frames, dtype=np.float32)
    for i in range(target_frames):
        span = activity_samples[edges[i]:edges[i + 1]]
        out[i] = 1.0 if (len(span) > 0 and span.max() > 0.5) else 0.0
    return out


def build_batch(examples, model: MentraWearNet, device: str):
    """Builds one batch from a given list of (already-generated) examples,
    runs a real forward pass to discover the model's actual frame count,
    then aligns labels to it. Caller decides whether `examples` is a fresh
    random draw (real training) or a FIXED reused set (tiny-overfit test --
    section 44's actual point: the same examples must repeat every step,
    or overfitting is structurally impossible no matter how correct the
    model is)."""
    batch_size = len(examples)
    mix_len = max(len(e.mixture) for e in examples)
    enr_len = max(len(e.enrollment) for e in examples)

    mixtures = torch.zeros(batch_size, mix_len)
    mixture_lengths = torch.zeros(batch_size, dtype=torch.long)
    enrollments = torch.zeros(batch_size, enr_len)
    enrollment_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, e in enumerate(examples):
        mixtures[i, :len(e.mixture)] = torch.from_numpy(e.mixture)
        mixture_lengths[i] = len(e.mixture)
        enrollments[i, :len(e.enrollment)] = torch.from_numpy(e.enrollment)
        enrollment_lengths[i] = len(e.enrollment)

    mixtures, mixture_lengths = mixtures.to(device), mixture_lengths.to(device)
    enrollments, enrollment_lengths = enrollments.to(device), enrollment_lengths.to(device)

    out = model(mixtures, mixture_lengths, enrollments, enrollment_lengths)
    target_frames = out["wearer_logits"].shape[1]

    wearer_targets = torch.zeros(batch_size, target_frames)
    env_targets = torch.zeros(batch_size, target_frames)
    for i, e in enumerate(examples):
        wearer_targets[i] = torch.from_numpy(align_labels_to_frames(e.wearer_activity, target_frames))
        env_targets[i] = torch.from_numpy(align_labels_to_frames(e.environment_activity, target_frames))

    return out, wearer_targets.to(device), env_targets.to(device), out["frame_lengths"]


def compute_loss(out, wearer_targets, env_targets, frame_lengths,
                  aux_losses: bool = False, aux_weight: float = 0.3,
                  speaker_disc: bool = False, disc_weight: float = 0.1, disc_margin: float = 0.3,
                  example_hard_mask=None):
    """BCEWithLogits, masked to valid frames (section 31 -- keep it simple,
    no auxiliary losses until the base task is proven to learn).

    V2 additions (sections 2e/2f), both OFF by default (aux_losses=False,
    speaker_disc=False): with both flags False, the computation of `loss`,
    `wearer_acc`, `env_acc` below is UNCHANGED from before this addition,
    and the returned `extra_info` dict is empty -- so
    --aux-losses/--speaker-disc-loss not being passed on the command line
    (the default) reproduces the exact original behavior of this function.

    aux_losses=True adds an any-speech BCE term (out["any_speech_logits"],
    requires the model to have been built with MentraWearNet(aux_losses=True))
    plus a 4-state cross-entropy term (out["four_state_logits"], channel
    order 0=silence/1=wearer-only/2=env-only/3=overlap), both weighted by
    aux_weight and summed INTO `loss` (so gradients flow through the shared
    trunk exactly like the main task).

    speaker_disc=True adds a non-overlap speaker-discrimination margin loss
    reusing out["frame_speaker_norm"]/out["enrollment_sim_space"] (the SAME
    tensors the model's existing similarity branch already computes) --
    pulls wearer-only frames toward the enrollment embedding (cosine sim
    -> 1) and pushes environment-only frames away from it (cosine sim
    -> <= 1 - disc_margin), weighted by disc_weight."""
    B, T = out["wearer_logits"].shape
    mask = (torch.arange(T, device=out["wearer_logits"].device).unsqueeze(0) < frame_lengths.unsqueeze(1)).float()
    wearer_loss = F.binary_cross_entropy_with_logits(out["wearer_logits"], wearer_targets, reduction="none")
    env_loss = F.binary_cross_entropy_with_logits(out["environment_logits"], env_targets, reduction="none")
    denom = mask.sum().clamp(min=1)
    wearer_loss_mean = (wearer_loss * mask).sum() / denom
    env_loss_mean = (env_loss * mask).sum() / denom
    loss = wearer_loss_mean + env_loss_mean
    with torch.no_grad():
        wearer_pred = (torch.sigmoid(out["wearer_logits"]) > 0.5).float()
        env_pred = (torch.sigmoid(out["environment_logits"]) > 0.5).float()
        wearer_acc = ((wearer_pred == wearer_targets).float() * mask).sum() / denom
        env_acc = ((env_pred == env_targets).float() * mask).sum() / denom

    # V3B diagnostics addition (Phase 7): when the caller passes
    # example_hard_mask ([B] bool -- which examples in this batch drew their
    # impostor from the hard-negative table, see mixture_generator.py's
    # generate_example(hard_pairs=..., p_hard=...)), break out wearer
    # false-positive rate (predicted wearer=1 on a frame truly NOT wearer)
    # separately for hard-impostor vs random-impostor examples. Pure
    # additive logging -- does not touch `loss`/wearer_acc/env_acc above.
    group_extra = {}
    if example_hard_mask is not None:
        with torch.no_grad():
            hard_sel = example_hard_mask.float().unsqueeze(1)  # [B, 1]
            rand_sel = 1.0 - hard_sel
            for name, sel in (("hard", hard_sel), ("rand", rand_sel)):
                gmask = mask * sel
                gdenom = gmask.sum().clamp(min=1)
                wearer_neg_mask = (wearer_targets <= 0.5).float() * gmask
                wearer_fp = ((wearer_pred > 0.5).float() * wearer_neg_mask).sum() / wearer_neg_mask.sum().clamp(min=1)
                group_extra[f"{name}_n_frames"] = float(gmask.sum().item())
                group_extra[f"{name}_wearer_acc"] = (((wearer_pred == wearer_targets).float() * gmask).sum() / gdenom).item()
                group_extra[f"{name}_env_acc"] = (((env_pred == env_targets).float() * gmask).sum() / gdenom).item()
                group_extra[f"{name}_wearer_fpr"] = wearer_fp.item()
                group_extra[f"{name}_wearer_loss"] = ((wearer_loss * gmask).sum() / gdenom).item()
                group_extra[f"{name}_env_loss"] = ((env_loss * gmask).sum() / gdenom).item()

    # V3A diagnostics addition: per-component loss values, always populated
    # (cheap, pure logging -- doesn't touch `loss`/`wearer_acc`/`env_acc`
    # above, which are numerically identical to before this change). Nothing
    # downstream reads extra_info's presence/absence to change behavior, only
    # to format an extra log string, so this is a safe additive change for
    # every existing caller (--aux-losses/--speaker-disc-loss off or on).
    extra_info = {
        "wearer_loss": wearer_loss_mean.item(),
        "env_loss": env_loss_mean.item(),
    }
    if aux_losses and "any_speech_logits" in out:
        any_speech_targets = torch.clamp(wearer_targets + env_targets, max=1.0)
        any_speech_loss = F.binary_cross_entropy_with_logits(out["any_speech_logits"], any_speech_targets, reduction="none")
        any_speech_loss = (any_speech_loss * mask).sum() / denom
        four_state_targets = wearer_targets.long() + 2 * env_targets.long()  # 0/1/2/3, see docstring
        four_state_loss = F.cross_entropy(out["four_state_logits"], four_state_targets, reduction="none")
        four_state_loss = (four_state_loss * mask).sum() / denom
        loss = loss + aux_weight * (any_speech_loss + four_state_loss)
        extra_info["any_speech_loss"] = any_speech_loss.item()
        extra_info["four_state_loss"] = four_state_loss.item()

        # V3A diagnostics: 4-state accuracy + macro-F1 over valid frames
        # (state-10 == "solo-target" -- see MEMORY step-1 finding, this head
        # already IS the solo-target signal, so its own accuracy/F1 doubles
        # as the solo-target precision/recall diagnostic requested in the
        # eval report, no separate decoder needed).
        with torch.no_grad():
            four_state_pred = out["four_state_logits"].argmax(dim=1)  # [B, T]
            valid = mask.bool()
            correct = (four_state_pred == four_state_targets).float() * mask
            four_state_acc = correct.sum() / denom
            extra_info["four_state_acc"] = four_state_acc.item()
            f1s = []
            for c in range(4):
                pred_c = (four_state_pred == c).float() * mask
                true_c = (four_state_targets == c).float() * mask
                tp = (pred_c * true_c).sum()
                fp = (pred_c * (1 - true_c)).sum()
                fn = ((1 - pred_c) * true_c).sum()
                precision = tp / (tp + fp).clamp(min=1)
                recall = tp / (tp + fn).clamp(min=1)
                f1 = (2 * precision * recall / (precision + recall).clamp(min=1e-8)) if (tp + fp + fn) > 0 else torch.tensor(0.0)
                f1s.append(f1)
            extra_info["four_state_macro_f1"] = torch.stack(f1s).mean().item()
            extra_info["four_state_solo_f1"] = f1s[1].item()  # class 1 = wearer-only = "solo-target"

            # V3B diagnostics addition (Phase 7): solo-target (class 1) P/R/F1
            # and 4-state accuracy, broken out hard- vs random-impostor.
            if example_hard_mask is not None:
                hard_sel_b = example_hard_mask.bool()
                for name, sel_b in (("hard", hard_sel_b), ("rand", ~hard_sel_b)):
                    gmask = mask * sel_b.float().unsqueeze(1)
                    gdenom = gmask.sum().clamp(min=1)
                    group_extra[f"{name}_four_state_acc"] = (((four_state_pred == four_state_targets).float() * gmask).sum() / gdenom).item()
                    pred_c = (four_state_pred == 1).float() * gmask
                    true_c = (four_state_targets == 1).float() * gmask
                    tp = (pred_c * true_c).sum()
                    fp = (pred_c * (1 - true_c)).sum()
                    fn = ((1 - pred_c) * true_c).sum()
                    precision = (tp / (tp + fp).clamp(min=1)).item()
                    recall = (tp / (tp + fn).clamp(min=1)).item()
                    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                    group_extra[f"{name}_solo_precision"] = precision
                    group_extra[f"{name}_solo_recall"] = recall
                    group_extra[f"{name}_solo_f1"] = f1

    if speaker_disc and "frame_speaker_norm" in out and "enrollment_sim_space" in out:
        frame_speaker_norm = out["frame_speaker_norm"]  # [B, E, T]
        enrollment_sim_space = out["enrollment_sim_space"]  # [B, E]
        sim = torch.einsum("bet,be->bt", frame_speaker_norm, enrollment_sim_space)  # [B, T] in [-1, 1]
        wearer_only_mask = ((wearer_targets > 0.5) & (env_targets <= 0.5)).float() * mask
        env_only_mask = ((env_targets > 0.5) & (wearer_targets <= 0.5)).float() * mask
        pull_loss = ((1.0 - sim) * wearer_only_mask).sum() / wearer_only_mask.sum().clamp(min=1)
        push_loss = (torch.clamp(sim - (1.0 - disc_margin), min=0.0) * env_only_mask).sum() / env_only_mask.sum().clamp(min=1)
        disc_term = pull_loss + push_loss
        loss = loss + disc_weight * disc_term
        extra_info["speaker_disc_loss"] = disc_term.item()

        # V3A diagnostics: raw speaker-metric cosine distribution for the
        # positive (state-10, wearer-only) vs negative (state-01, env-only)
        # populations, plus the realized margin between their means -- the
        # actual thing this experiment is trying to move off chance.
        with torch.no_grad():
            pos_vals = torch.masked_select(sim, wearer_only_mask.bool())
            neg_vals = torch.masked_select(sim, env_only_mask.bool())
            if pos_vals.numel() > 0:
                extra_info["speaker_cos_pos_mean"] = pos_vals.mean().item()
                extra_info["speaker_cos_pos_std"] = pos_vals.std().item() if pos_vals.numel() > 1 else 0.0
            if neg_vals.numel() > 0:
                extra_info["speaker_cos_neg_mean"] = neg_vals.mean().item()
                extra_info["speaker_cos_neg_std"] = neg_vals.std().item() if neg_vals.numel() > 1 else 0.0
            if pos_vals.numel() > 0 and neg_vals.numel() > 0:
                extra_info["speaker_cos_margin"] = pos_vals.mean().item() - neg_vals.mean().item()

            # V3B diagnostics addition (Phase 7/8): split the NEGATIVE
            # (env-only) cosine population by whether that example's
            # impostor was hard-mined or random -- this is the actual
            # "positive-vs-hard-negative margin" / "positive-vs-random-
            # negative margin" the causal-ablation report needs. The
            # POSITIVE population is left un-split (positive frames belong
            # to the wearer, not the impostor; splitting them by the
            # example's impostor-selection type is not a meaningful
            # distinction), so both margins share the same pos_mean above.
            if example_hard_mask is not None and pos_vals.numel() > 0:
                hard_sel_b = example_hard_mask.bool()
                for name, sel_b in (("hard", hard_sel_b), ("rand", ~hard_sel_b)):
                    g_env_only_mask = env_only_mask * sel_b.float().unsqueeze(1)
                    g_neg_vals = torch.masked_select(sim, g_env_only_mask.bool())
                    if g_neg_vals.numel() > 0:
                        group_extra[f"{name}_speaker_cos_neg_mean"] = g_neg_vals.mean().item()
                        group_extra[f"{name}_speaker_cos_neg_std"] = (g_neg_vals.std().item() if g_neg_vals.numel() > 1 else 0.0)
                        group_extra[f"{name}_speaker_cos_margin"] = pos_vals.mean().item() - g_neg_vals.mean().item()

    extra_info.update(group_extra)
    return loss, wearer_acc.item(), env_acc.item(), extra_info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="evaluation/manifests/day1_public_speakers.json")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--total-s", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze-backbone", action="store_true", default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint-out", default="training/checkpoints/mentrawearnet_v1.pt",
                     help="stable pointer path, always updated to the LATEST checkpoint -- kept for "
                          "backward compatibility with scripts that default to reading this exact path "
                          "(e.g. scripts/model/evaluate_mentrawearnet.py). Full checkpoint HISTORY is "
                          "preserved separately under --run-dir (see below); this path alone is "
                          "overwritten every save, same as before.")
    ap.add_argument("--run-dir", default=None,
                     help="Every run gets its own directory (default: "
                          "training/runs/<UTC timestamp>-<manifest stem>), so concurrent or repeated "
                          "invocations never collide or silently overwrite each other's history. Each "
                          "periodic AND final checkpoint is saved there under a step-numbered filename "
                          "(never overwritten), plus a self-contained train.log (written by this "
                          "process directly, independent of shell redirection) and a config.json dump "
                          "of every CLI arg for this run. Pass explicitly to reuse a specific directory "
                          "(e.g. across --resume invocations).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-duration", action="store_true",
                     help="use --total-s for every example instead of variable-duration sampling")
    ap.add_argument("--fixed-set-size", type=int, default=0,
                     help="if >0, generate this many examples ONCE and reuse them every step "
                          "(tiny-overfit correctness test -- section 44). If 0, generate a fresh "
                          "random batch every step (real training).")
    ap.add_argument("--noise-dir", default=None,
                     help="e.g. evaluation/data/raw/musan/noise -- additive background noise (MUSAN, CC BY 4.0)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0,
                     help="if >0, write a checkpoint every N steps (in addition to the final save) "
                          "so a killed/crashed run doesn't lose all progress since the start.")
    ap.add_argument("--resume", action="store_true",
                     help="if --checkpoint-out already exists, load model/optimizer/step from it "
                          "and continue training from the saved step instead of starting over.")
    # --- V2 additions below (sections 2e/2f/2h). All default to OFF/stage-A,
    # matching today's exact behavior when not passed on the command line. ---
    ap.add_argument("--unfreeze-stage", choices=["A", "B", "C"], default="A",
                     help="section 2h (staged unfreezing): A=fully frozen backbone (default, "
                          "matches current behavior exactly), B=unfreeze last 1 Jasper block, "
                          "C=unfreeze last 2 Jasper blocks. Only takes effect when "
                          "--freeze-backbone is set (it is, by default).")
    ap.add_argument("--backbone-lr", type=float, default=None,
                     help="section 2h: separate LR for unfrozen backbone params (stage B/C only). "
                          "Defaults to --lr / 20 if not given. No effect at stage A (no trainable "
                          "backbone params to apply it to).")
    ap.add_argument("--aux-losses", action="store_true", default=False,
                     help="section 2e: adds an any-speech BCE head + 4-state cross-entropy head, "
                          "both training-only (removed at export). OFF by default -- without this "
                          "flag, MentraWearNet is constructed identically to before this option existed.")
    ap.add_argument("--aux-loss-weight", type=float, default=0.3,
                     help="section 2e: weight applied to the summed aux-loss terms when --aux-losses is set.")
    ap.add_argument("--speaker-disc-loss", action="store_true", default=False,
                     help="section 2f: adds a non-overlap speaker-discrimination margin loss "
                          "(pulls wearer-only frames toward the enrollment embedding, pushes "
                          "environment-only frames away). OFF by default. Requires --aux-losses "
                          "(reuses the frame_speaker/enrollment_sim_space tensors that are only "
                          "returned when the model was built with aux_losses=True).")
    ap.add_argument("--speaker-disc-weight", type=float, default=0.1)
    ap.add_argument("--speaker-disc-margin", type=float, default=0.3)
    # --- V3B additions (hard-negative curriculum). All default to
    # off/None, which reproduces V2/V3A behavior exactly (see
    # mixture_generator.py's generate_example() docstring: with
    # hard_pairs=None/p_hard=0.0 it doesn't even consume an extra rng draw). ---
    ap.add_argument("--hard-pairs-json", default=None,
                     help="V3B: path to a {speaker_id: [nearest speaker_ids]} table "
                          "(training/diagnostics/hard_negative_mining.py's output, e.g. "
                          "evaluation/manifests/hard_negative_pairs_k50.json) used to draw "
                          "hard-negative impostors. Required for --p-hard-schedule to have "
                          "any effect.")
    ap.add_argument("--p-hard-schedule", default=None,
                     help="V3B: comma-separated stage list 'nsteps:p,nsteps:p,...' -- e.g. "
                          "'500:0.25,500:0.50,1000:0.80' means steps [0,500) use p_hard=0.25, "
                          "[500,1000) use 0.50, [1000,2000) use 0.80. p_hard is the probability "
                          "generate_example() draws the impostor from --hard-pairs-json instead "
                          "of uniformly at random. Steps beyond the schedule's total reuse the "
                          "final stage's p. Omit (default) or pass e.g. '2000:0.0' for an "
                          "ordinary-random-negatives control run.")
    ap.add_argument("--load-full-state-from", default=None,
                     help="V3B addition (fine-tuning LR/optimizer-resume, see report Phase 5): "
                          "loads BOTH model weights (strict=False, same convention as "
                          "--init-from-checkpoint) AND optimizer state (Adam moment estimates) "
                          "from this checkpoint path, but -- unlike --resume -- does NOT couple "
                          "the run to that checkpoint's step count or to --checkpoint-out already "
                          "existing at that path: the step counter still starts at 0 (or wherever "
                          "--resume of THIS run's own --checkpoint-out says, if combined). This "
                          "is the clean way to fine-tune from a converged optimizer state (e.g. "
                          "V3A's) without also inheriting its step count or being forced to reuse "
                          "its checkpoint path. Mutually exclusive with --init-from-checkpoint "
                          "(this flag supersedes it -- it loads weights too).")
    ap.add_argument("--init-from-checkpoint", default=None,
                     help="V3A addition: load ONLY model weights from this checkpoint path "
                          "(strict=False, so heads absent from the source -- e.g. a V2 checkpoint "
                          "trained without --aux-losses -- keep their fresh random init rather than "
                          "erroring) as this run's starting point, then proceed with a FRESH "
                          "optimizer and step counter starting at 0. Distinct from --resume, which "
                          "additionally restores optimizer state + step from --checkpoint-out and "
                          "requires that exact path to already exist; mutually exclusive with "
                          "--resume. Meaning of --resume for existing callers is unchanged.")
    args = ap.parse_args()
    if args.init_from_checkpoint and args.resume:
        ap.error("--init-from-checkpoint and --resume are mutually exclusive "
                  "(init-from-checkpoint loads weights only into a fresh run; "
                  "--resume restores optimizer+step from --checkpoint-out).")
    if args.load_full_state_from and args.init_from_checkpoint:
        ap.error("--load-full-state-from already loads weights (like --init-from-checkpoint) "
                  "plus optimizer state -- pass only one of the two.")
    if args.load_full_state_from and args.resume:
        ap.error("--load-full-state-from and --resume are mutually exclusive "
                  "(both would set the initial optimizer state from a checkpoint; "
                  "--resume also couples the run to --checkpoint-out's step count).")

    def parse_p_hard_schedule(spec: str | None):
        """Returns a list of (cumulative_end_step, p) stage boundaries, or
        None if spec is None (V2/V3A-identical behavior: p_hard=0.0 always,
        no rng draw consumed -- see generate_example())."""
        if not spec:
            return None
        stages = []
        cum = 0
        for part in spec.split(","):
            n_str, p_str = part.split(":")
            cum += int(n_str)
            stages.append((cum, float(p_str)))
        return stages

    p_hard_schedule = parse_p_hard_schedule(args.p_hard_schedule)

    def get_p_hard(step: int) -> float:
        if p_hard_schedule is None:
            return 0.0
        for end_step, p in p_hard_schedule:
            if step < end_step:
                return p
        return p_hard_schedule[-1][1]  # beyond schedule: hold final stage's p

    # DDP setup: only active when launched via torchrun (RANK/LOCAL_RANK/
    # WORLD_SIZE env vars present). Plain `python3 train.py` still works
    # exactly as before (world_size=1, rank=0, no DDP wrapper).
    is_distributed = "RANK" in os.environ
    if is_distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = args.device
    is_main = rank == 0

    # Every run gets its own directory + self-contained log file, generated
    # automatically -- no more manually renaming/redirecting log files
    # between runs (real recurring friction earlier in this project: a
    # stale log filename left open in an editor tab, a checkpoint silently
    # overwritten by the next run because everyone reused the same fixed
    # path). run_id is timestamp+manifest-based so it's unique and
    # human-readable without needing a UUID.
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + Path(args.manifest).stem
    run_dir = Path(args.run_dir) if args.run_dir else Path("training/runs") / run_id
    log_file = None

    def log(msg: str, err: bool = False):
        stream = sys.stderr if err else sys.stdout
        print(msg, file=stream, flush=True)
        if log_file is not None:
            log_file.write(msg + "\n")
            log_file.flush()

    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(run_dir / "train.log", "a")
        (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
        log(f"run_dir={run_dir}  (self-contained log + full checkpoint history live here)")

    # Each rank gets a DIFFERENT seed -- see module docstring. Without
    # this, both GPUs would generate byte-identical mixtures and DDP would
    # just be computing the same gradient twice, not actually parallelizing
    # over more data.
    rank_seed = args.seed + rank
    torch.manual_seed(rank_seed)
    import random
    rng = random.Random(rank_seed)

    if is_main:
        log(f"device={device}  world_size={world_size}  distributed={is_distributed}")
    pool = SpeakerPool(args.manifest)
    if is_main:
        log(f"speaker pool: {len(pool.speaker_ids)} speakers from {args.manifest}")

    hard_pairs = None
    if args.hard_pairs_json:
        hard_pairs = json.loads(Path(args.hard_pairs_json).read_text())
        if is_main:
            log(f"hard-negative pairs table: {len(hard_pairs)} speakers from {args.hard_pairs_json}, "
                f"p_hard schedule={p_hard_schedule}")

    noise_pool = NoisePool(args.noise_dir)
    if is_main:
        log(f"noise pool: {len(noise_pool.clips)} clips from {args.noise_dir}" if noise_pool.available
            else "noise pool: none (training without additive noise)")

    model = MentraWearNet(aux_losses=args.aux_losses).to(device)
    if args.freeze_backbone:
        # Section 2h (staged unfreezing): default stage "A" reproduces the
        # original single-line `model.freeze_backbone()` exactly. Stages
        # B/C are new opt-in behavior via --unfreeze-stage.
        if args.unfreeze_stage == "A":
            model.freeze_backbone()
        elif args.unfreeze_stage == "B":
            model.unfreeze_upper_backbone(num_upper_blocks=1)
        elif args.unfreeze_stage == "C":
            model.unfreeze_upper_backbone(num_upper_blocks=2)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main:
        log(f"trainable params: {trainable:,} / total: {total:,}"
            + (f"  (x{world_size} GPUs, effective batch = {args.batch_size * world_size})" if is_distributed else ""))

    raw_model = model  # unwrapped reference for encode_enrollment/etc if ever needed outside forward()
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    # Section 2h: separate (smaller) LR for unfrozen backbone params, when
    # any exist (stage B/C). At stage A (default), backbone_params is empty
    # -- backbone.* params are all frozen -- so this reduces to exactly one
    # param group containing every trainable param at args.lr, identical to
    # the original single-group `torch.optim.Adam([p for p in
    # model.parameters() if p.requires_grad], lr=args.lr)`.
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr / 20
    main_params = [p for n, p in raw_model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    backbone_params = [p for n, p in raw_model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    param_groups = [{"params": main_params, "lr": args.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
        if is_main:
            log(f"staged unfreezing: {len(backbone_params)} backbone tensors trainable at "
                f"backbone_lr={backbone_lr:.2e} (main lr={args.lr:.2e})")
    optimizer = torch.optim.Adam(param_groups)

    # V3A addition: init-from-checkpoint -- load WEIGHTS ONLY (strict=False)
    # on ALL ranks (same file, shared filesystem, same reasoning as --resume
    # below: loading only on rank 0 would desync the other rank's weights,
    # and DDP's constructor-time broadcast already happened before this
    # point using the fresh-init weights, so every rank must load
    # independently from the identical file to end up in sync). Optimizer
    # and step counter are untouched -- optimizer was just constructed fresh
    # above, start_step stays 0 below, exactly matching the brief's "fresh
    # optimizer, fresh step counter, preserve only the V2 prior's weights."
    if args.init_from_checkpoint:
        init_ckpt = torch.load(args.init_from_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = raw_model.load_state_dict(init_ckpt["model_state_dict"], strict=False)
        if is_main:
            log(f"INIT-FROM-CHECKPOINT: loaded weights from {args.init_from_checkpoint} "
                f"(source step={init_ckpt.get('step', '?')}, "
                f"source final_loss={init_ckpt.get('final_loss', float('nan')):.4f})")
            log(f"  missing_keys ({len(missing)}, kept at fresh init -- expected for new "
                f"--aux-losses heads not present in the source checkpoint): {missing}")
            log(f"  unexpected_keys ({len(unexpected)}, ignored): {unexpected}")

    # V3B addition: --load-full-state-from -- load BOTH weights (strict=False,
    # same convention as --init-from-checkpoint above) AND optimizer state
    # on ALL ranks, WITHOUT adopting that checkpoint's step count (start_step
    # stays 0 below, same as --init-from-checkpoint) and without requiring
    # --checkpoint-out to already exist at that path (unlike --resume). This
    # is what lets V3B fine-tune from V3A's converged Adam moment estimates
    # instead of restarting the optimizer from zero at a lower LR.
    if args.load_full_state_from:
        full_ckpt = torch.load(args.load_full_state_from, map_location=device, weights_only=False)
        missing, unexpected = raw_model.load_state_dict(full_ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(full_ckpt["optimizer_state_dict"])
        # CRITICAL: Adam.load_state_dict() restores param_groups' hyperparameters
        # (including 'lr') from the SOURCE checkpoint too, silently overwriting
        # the --lr / --backbone-lr this run was just constructed with (caught by
        # a smoke test: --lr 5e-5 here, but the source checkpoint's optimizer
        # had lr=3e-4, and after load_state_dict() the logged step-0 lr was
        # 3e-4, not 5e-5). Re-apply THIS run's intended LRs onto every param
        # group right after loading so only the moment estimates (exp_avg /
        # exp_avg_sq) transfer, not the source run's learning rate.
        for i, group in enumerate(optimizer.param_groups):
            group["lr"] = param_groups[i]["lr"]
        if is_main:
            log(f"LOAD-FULL-STATE-FROM: loaded weights + optimizer state from "
                f"{args.load_full_state_from} (source step={full_ckpt.get('step', '?')}, "
                f"source final_loss={full_ckpt.get('final_loss', float('nan')):.4f}); "
                f"this run's own step counter still starts at 0")
            log(f"  missing_keys ({len(missing)}): {missing}")
            log(f"  unexpected_keys ({len(unexpected)}): {unexpected}")
            log(f"  re-applied this run's own LR(s) after optimizer-state load: "
                f"{[g['lr'] for g in optimizer.param_groups]} (moment estimates still transferred)")

    # Resume: load model+optimizer+step from an existing checkpoint on ALL
    # ranks (same file, shared filesystem) so DDP replicas stay identical --
    # loading only on rank 0 would desync the other rank's weights.
    start_step = 0
    if args.resume and Path(args.checkpoint_out).exists():
        ckpt = torch.load(args.checkpoint_out, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        if is_main:
            log(f"RESUMED from {args.checkpoint_out} at step {start_step} "
                f"(saved final_loss={ckpt.get('final_loss', float('nan')):.4f})")
    elif args.resume and is_main:
        log(f"--resume given but {args.checkpoint_out} does not exist -- starting fresh")

    fixed_examples = None
    if args.fixed_set_size > 0:
        fixed_examples = [generate_example(pool, rng, total_s=args.total_s, noise_pool=noise_pool)
                           for _ in range(args.fixed_set_size)]
        if is_main:
            log(f"TINY-OVERFIT MODE: generated {len(fixed_examples)} fixed examples, reused every step")

    if is_main:
        Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(completed_steps: int, final_loss: float):
        # raw_model (unwrapped, pre-DDP reference), not model.state_dict() --
        # DDP prefixes every key with "module.", which would silently break
        # loading this checkpoint into a plain (non-DDP) MentraWearNet later
        # (e.g. scripts/model/evaluate_mentrawearnet.py).
        #
        # Two writes per save, both via a .tmp + os.replace so a save killed
        # mid-write never leaves a corrupt/truncated file (os.replace is
        # atomic on the same filesystem):
        #   1. a STEP-NUMBERED file under run_dir, NEVER overwritten -- this
        #      is the fix for the real problem where every periodic save used
        #      to overwrite the same fixed path, silently destroying all
        #      earlier checkpoints in a run (2000-4400 lost this way in an
        #      earlier run today).
        #   2. the stable --checkpoint-out pointer, updated to the latest
        #      state every time, kept for backward compatibility with
        #      scripts that default to reading that exact fixed path.
        payload = {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": completed_steps,
            "config": vars(args),
            "final_loss": final_loss,
            "run_dir": str(run_dir),
        }
        step_path = run_dir / f"checkpoint_step{completed_steps}.pt"
        step_tmp = step_path.with_suffix(".pt.tmp")
        torch.save(payload, step_tmp)
        os.replace(step_tmp, step_path)

        stable_tmp = f"{args.checkpoint_out}.tmp"
        torch.save(payload, stable_tmp)
        os.replace(stable_tmp, args.checkpoint_out)
        return step_path

    if start_step >= args.steps:
        if is_main:
            log(f"resumed step {start_step} already >= target --steps {args.steps}, nothing to do")
            log_file.close()
        if is_distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    losses = []
    t0 = time.time()
    for step in range(start_step, args.steps):
        model.train()
        if fixed_examples is not None:
            batch = [fixed_examples[i % len(fixed_examples)]
                     for i in range(step * args.batch_size, (step + 1) * args.batch_size)]
        else:
            # variable-duration sampling (weighted toward 0.75-1.5s, the
            # product-relevant region -- see sample_duration_s docstring)
            # unless the caller pinned a fixed --total-s explicitly.
            # V3B addition: requested_p_hard is read fresh every step from
            # the schedule (get_p_hard) and passed straight through to
            # generate_example -- everything else in this call (duration
            # sampling, noise_pool) is byte-identical to before.
            requested_p_hard = get_p_hard(step)
            batch = [generate_example(pool, rng, total_s=(args.total_s if args.fixed_duration else sample_duration_s(rng)),
                                       noise_pool=noise_pool, hard_pairs=hard_pairs, p_hard=requested_p_hard)
                     for _ in range(args.batch_size)]
        example_hard_mask = torch.tensor(
            [bool(e.metadata.get("used_hard_negative", False)) for e in batch], device=device)
        out, wearer_targets, env_targets, frame_lengths = build_batch(batch, model, device)
        loss, wearer_acc, env_acc, extra_info = compute_loss(
            out, wearer_targets, env_targets, frame_lengths,
            aux_losses=args.aux_losses, aux_weight=args.aux_loss_weight,
            speaker_disc=args.speaker_disc_loss, disc_weight=args.speaker_disc_weight,
            disc_margin=args.speaker_disc_margin,
            example_hard_mask=example_hard_mask if hard_pairs else None)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0)
        optimizer.step()

        losses.append(loss.item())
        if is_main and (step % args.log_every == 0 or step == args.steps - 1):
            recent = losses[-args.log_every:]
            # Section 2i: current LR + gradient norm appended to the existing
            # log line (grad_norm was already being computed via
            # clip_grad_norm_'s return value, just not printed before; this
            # is a pure additive change to the print statement, no control
            # flow touched). extra_info (sections 2e/2f) is empty unless
            # --aux-losses/--speaker-disc-loss are passed, so the line is
            # unchanged there for default invocations.
            extra_str = "".join(f"  {k}={v:.4f}" for k, v in extra_info.items())
            log(f"step {step:4d}  loss={loss.item():.4f}  loss_avg10={sum(recent)/len(recent):.4f}  "
                f"wearer_acc={wearer_acc:.3f}  env_acc={env_acc:.3f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  grad_norm={grad_norm:.3f}"
                f"{extra_str}  "
                f"elapsed={time.time()-t0:.1f}s")

        if not torch.isfinite(loss):
            log(f"FAIL: non-finite loss at step {step}", err=True)
            sys.exit(1)

        # Periodic checkpoint -- so a killed/crashed run (deliberate restart
        # for a bigger batch, an OOM, a preemption) only loses progress since
        # the last save-every boundary, not everything back to step 0. Every
        # save is a NEW step-numbered file (see save_checkpoint) -- nothing
        # from earlier in this run is ever overwritten.
        if is_main and args.save_every > 0 and (step + 1) % args.save_every == 0:
            step_path = save_checkpoint(step + 1, loss.item())
            log(f"checkpoint saved at step {step + 1} -> {step_path}")

    if is_main:
        step_path = save_checkpoint(args.steps, losses[-1])
        log(f"saved checkpoint to {args.checkpoint_out} (full history: {step_path})")
        log(f"loss: start={losses[0]:.4f} end={losses[-1]:.4f}")
        log_file.close()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
