#!/usr/bin/env python3
"""V2 diagnostic 1b: state distribution + trivial-baseline audit.

CPU-only, standalone. Generates a real sample of mixture examples using the
SAME manifest + generator the V1 training run uses, computes frame-level
wearer/environment activity via the SAME align_labels_to_frames() used in
training/train.py, tabulates the real fraction of frames in each of the 4
states (silence / wearer-only / environment-only / overlap), and computes
what a trivial "always predict majority class" baseline would score on each
head. This is the number training's wearer_acc/env_acc must be compared
against to mean anything.

Run: .venv/bin/python3 training/diagnostics/audit_state_distribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

torch.set_num_threads(4)  # keep modest -- V1 training's data pipeline also needs CPU

import random

from training.data.mixture_generator import SpeakerPool, NoisePool, generate_example, sample_duration_s
from training.models.mentrawearnet import MentraWearNet
from training.train import align_labels_to_frames

MANIFEST = "evaluation/manifests/librispeech_train_clean_100_train.json"
NOISE_DIR = "evaluation/data/raw/musan/noise"
N_EXAMPLES = 300
SEED = 1234


def main():
    print(f"loading speaker pool ({MANIFEST}) and noise pool ({NOISE_DIR}) ...", flush=True)
    pool = SpeakerPool(MANIFEST)
    noise_pool = NoisePool(NOISE_DIR)
    print(f"  speakers: {len(pool.speaker_ids)}   noise clips: {len(noise_pool.clips)}", flush=True)

    print("loading MentraWearNet (CPU-only, for real frame-count discovery) ...", flush=True)
    model = MentraWearNet().eval()
    print("loaded.\n", flush=True)

    rng = random.Random(SEED)

    # sample-resolution tallies
    n_silence_s = n_wearer_s = n_env_s = n_overlap_s = 0
    total_samples = 0
    # frame-resolution tallies (post align_labels_to_frames, matches what the
    # model is actually trained/scored against)
    n_silence_f = n_wearer_f = n_env_f = n_overlap_f = 0
    total_frames = 0

    print(f"generating {N_EXAMPLES} real examples (variable duration, TIR-controlled overlap, "
          f"real MUSAN noise) ...", flush=True)
    for i in range(N_EXAMPLES):
        ex = generate_example(pool, rng, total_s=sample_duration_s(rng), noise_pool=noise_pool)

        # --- sample resolution ---
        w_s, e_s = ex.wearer_activity, ex.environment_activity
        both_s = (w_s > 0.5) & (e_s > 0.5)
        n_overlap_s += int(both_s.sum())
        n_wearer_s += int(((w_s > 0.5) & ~both_s).sum())
        n_env_s += int(((e_s > 0.5) & ~both_s).sum())
        n_silence_s += int(((w_s <= 0.5) & (e_s <= 0.5)).sum())
        total_samples += len(w_s)

        # --- frame resolution: need the model's real target_frames for this
        # example's mixture length, via one real forward pass (frame count
        # only depends on length, but we go through the real backbone path
        # to match train.py's build_batch() exactly rather than assuming).
        waveform = torch.from_numpy(ex.mixture).unsqueeze(0)
        lengths = torch.tensor([len(ex.mixture)])
        with torch.no_grad():
            _, frame_lengths = model.backbone.encode_frames(waveform, lengths)
        target_frames = int(frame_lengths[0].item())

        w_f = align_labels_to_frames(ex.wearer_activity, target_frames)
        e_f = align_labels_to_frames(ex.environment_activity, target_frames)
        both_f = (w_f > 0.5) & (e_f > 0.5)
        n_overlap_f += int(both_f.sum())
        n_wearer_f += int(((w_f > 0.5) & ~both_f).sum())
        n_env_f += int(((e_f > 0.5) & ~both_f).sum())
        n_silence_f += int(((w_f <= 0.5) & (e_f <= 0.5)).sum())
        total_frames += target_frames

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{N_EXAMPLES}", flush=True)

    def report(label, silence, wearer, env, overlap, total):
        print(f"\n--- {label} (total={total}) ---")
        print(f"  SILENCE:          {silence:8d}  ({100*silence/total:5.2f}%)")
        print(f"  WEARER-only:      {wearer:8d}  ({100*wearer/total:5.2f}%)")
        print(f"  ENVIRONMENT-only: {env:8d}  ({100*env/total:5.2f}%)")
        print(f"  OVERLAP:          {overlap:8d}  ({100*overlap/total:5.2f}%)")

        # wearer head target: active in WEARER-only + OVERLAP
        wearer_active_frac = (wearer + overlap) / total
        env_active_frac = (env + overlap) / total
        # trivial majority-class baseline per head (predict whichever class
        # -- active or inactive -- is more common for that head)
        wearer_trivial_acc = max(wearer_active_frac, 1 - wearer_active_frac)
        env_trivial_acc = max(env_active_frac, 1 - env_active_frac)
        print(f"  wearer head: active_frac={wearer_active_frac:.4f}  "
              f"-> trivial majority-class baseline acc = {wearer_trivial_acc:.4f}")
        print(f"  env    head: active_frac={env_active_frac:.4f}  "
              f"-> trivial majority-class baseline acc = {env_trivial_acc:.4f}")
        # also report the "always predict 0" variant specifically, since
        # that's the simplest possible degenerate model
        wearer_always0_acc = 1 - wearer_active_frac
        env_always0_acc = 1 - env_active_frac
        print(f"  wearer head: 'always predict 0' acc = {wearer_always0_acc:.4f}")
        print(f"  env    head: 'always predict 0' acc = {env_always0_acc:.4f}")
        return wearer_trivial_acc, env_trivial_acc

    report("SAMPLE-resolution (raw generator output)",
           n_silence_s, n_wearer_s, n_env_s, n_overlap_s, total_samples)
    wearer_triv, env_triv = report(
        "FRAME-resolution (post align_labels_to_frames -- what training actually sees)",
        n_silence_f, n_wearer_f, n_env_f, n_overlap_f, total_frames)

    print("\n" + "=" * 78)
    print("COMPARISON TO TRAINING LOG")
    print("=" * 78)
    print(f"  trivial majority-class baseline: wearer_acc={wearer_triv:.3f}  env_acc={env_triv:.3f}")
    print("  training/logs/train_v1.log (steps ~1000-3100) shows wearer_acc/env_acc")
    print("  hovering ~0.58-0.65 for both heads -- compare directly against the")
    print("  trivial baseline above computed from the SAME manifest+generator.")


if __name__ == "__main__":
    main()
