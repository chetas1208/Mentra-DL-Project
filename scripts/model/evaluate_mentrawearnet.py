#!/usr/bin/env python3
"""Real evaluation of a trained MentraWearNet checkpoint against the
measured SpeakerNet baseline (docs/MODEL_SELECTION.md: clean 2.13% EER,
TIR 0/-5/-10 -> 13.3/20.8/29.2% EER). Uses the SAME held-out 8-speaker
day1_public set as the historical regression benchmark (frozen, never
used for training) plus fresh synthetic TIR-controlled overlap examples
built the same way as training data, so the comparison is apples-to-apples
with the original SpeakerNet overlap experiment
(scripts/run_overlap_experiment.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from training.data.mixture_generator import SpeakerPool, load_wav
from training.models.mentrawearnet import MentraWearNet

TIR_LEVELS = [10, 5, 0, -5, -10]


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def mix_at_tir(target, interferer, tir_db):
    n = min(len(target), len(interferer))
    t, i = target[:n], interferer[:n]
    scale = (_rms(t) / (10 ** (tir_db / 20))) / _rms(i)
    mixture = t + i * scale
    peak = np.max(np.abs(mixture))
    return mixture / peak if peak > 1.0 else mixture


@torch.no_grad()
def score_clip(model, samples, sr, wearer_embedding, device):
    waveform = torch.from_numpy(samples).unsqueeze(0).float().to(device)
    lengths = torch.tensor([len(samples)]).to(device)
    out = model.process_with_embedding(waveform, lengths, wearer_embedding)
    wearer_prob = torch.sigmoid(out["wearer_logits"])
    # Frame-level -> single clip-level score: mean probability over valid frames.
    return wearer_prob.mean().item()


def compute_eer(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    allscores = np.unique(np.concatenate([pos, neg]))
    diffs = [(abs((neg >= t).mean() - (pos < t).mean()), (neg >= t).mean(), (pos < t).mean()) for t in allscores]
    diffs.sort(key=lambda x: x[0])
    return (diffs[0][1] + diffs[0][2]) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="training/checkpoints/mentrawearnet_v1.pt")
    ap.add_argument("--manifest", default="evaluation/manifests/day1_public_speakers.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = MentraWearNet().to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint from {args.checkpoint} (trained {ckpt.get('step', '?')} steps, "
          f"final_loss={ckpt.get('final_loss', '?'):.4f})" if isinstance(ckpt.get('final_loss'), float)
          else f"loaded checkpoint from {args.checkpoint}")

    pool = SpeakerPool(args.manifest)
    speaker_ids = pool.speaker_ids

    print("\n=== Clean rotation EER (matches original SpeakerNet protocol) ===")
    pos, neg = [], []
    for wearer_id in speaker_ids:
        enroll_segments = [torch.from_numpy(load_wav(p)).unsqueeze(0).float().to(args.device)
                            for p in pool.speakers[wearer_id]["enroll"]]
        enroll_lengths = [torch.tensor([e.shape[1]]).to(args.device) for e in enroll_segments]
        embs = [model.encode_enrollment(e, l) for e, l in zip(enroll_segments, enroll_lengths)]
        # NOT L2-normalized -- forward() (the training path) feeds the raw
        # encode_enrollment() output straight into enrollment_projection with
        # no normalization step. Normalizing here shifts the input scale
        # FiLM/enrollment_projection were trained on, which silently breaks
        # conditioning (real bug: was producing ~50% EER, chance level, even
        # for a checkpoint that visibly overfit its training set).
        wearer_embedding = torch.stack(embs).mean(dim=0)

        for test_id in speaker_ids:
            for test_path in pool.speakers[test_id]["test"]:
                samples = load_wav(test_path)
                score = score_clip(model, samples, 16000, wearer_embedding, args.device)
                (pos if test_id == wearer_id else neg).append(score)

    clean_eer = compute_eer(pos, neg)
    print(f"clean EER: {clean_eer*100:.2f}%  (SpeakerNet baseline: 2.13%)")

    print("\n=== Overlap / TIR EER (matches scripts/run_overlap_experiment.py protocol) ===")
    import random
    rng = random.Random(42)
    for tir in TIR_LEVELS:
        pos_tir, neg_tir = [], []
        for wearer_id in speaker_ids:
            enroll_segments = [torch.from_numpy(load_wav(p)).unsqueeze(0).float().to(args.device)
                                for p in pool.speakers[wearer_id]["enroll"]]
            enroll_lengths = [torch.tensor([e.shape[1]]).to(args.device) for e in enroll_segments]
            embs = [model.encode_enrollment(e, l) for e, l in zip(enroll_segments, enroll_lengths)]
            wearer_embedding = torch.stack(embs).mean(dim=0)  # raw scale, matches training -- see note above

            others = [s for s in speaker_ids if s != wearer_id]
            for wearer_clip_path in pool.speakers[wearer_id]["test"][:3]:
                interferer_id = rng.choice(others)
                interferer_path = rng.choice(pool.speakers[interferer_id]["test"])
                mixture = mix_at_tir(load_wav(wearer_clip_path), load_wav(interferer_path), tir)
                pos_tir.append(score_clip(model, mixture, 16000, wearer_embedding, args.device))

            for _ in range(3):
                a_id, b_id = rng.sample(others, 2)
                a = load_wav(rng.choice(pool.speakers[a_id]["test"]))
                b = load_wav(rng.choice(pool.speakers[b_id]["test"]))
                mixture = mix_at_tir(a, b, tir)
                neg_tir.append(score_clip(model, mixture, 16000, wearer_embedding, args.device))

        eer = compute_eer(pos_tir, neg_tir)
        baseline = {10: 4.17, 5: 4.17, 0: 13.33, -5: 20.83, -10: 29.17}[tir]
        delta = eer * 100 - baseline
        verdict = "BETTER" if delta < -0.5 else ("WORSE" if delta > 0.5 else "~SAME")
        print(f"TIR {tir:+3d}dB: EER={eer*100:6.2f}%  (SpeakerNet baseline: {baseline:5.2f}%, delta={delta:+.2f}pp, {verdict})")


if __name__ == "__main__":
    main()
