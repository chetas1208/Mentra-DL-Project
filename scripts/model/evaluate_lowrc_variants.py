#!/usr/bin/env python3
"""Real lookahead + accuracy measurement for C0 and LowRC variants (Track
B sections 10-12). For each configured budget: (1) measure actual future
dependency empirically via the same prefix-A/future-B/future-C method used
for the causality test, (2) run the real 8-speaker rotation experiment to
measure EER, so lookahead-vs-accuracy tradeoff is measured, not assumed.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from training.models.streaming_speakernet_backbone import StreamingSpeakerNetBackbone

VARIANTS = {"C0 (0ms)": 0, "LowRC-1 (~10ms)": 1, "LowRC-5 (~50ms)": 5,
            "LowRC-10 (~100ms)": 10, "LowRC-16 (~160ms)": 16}


def load_wav(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


def measure_lookahead(backbone, sr=16000):
    torch.manual_seed(0)
    a = torch.randn(1, sr)
    b = torch.randn(1, sr)
    c = torch.randn(1, sr)
    ab, ac = torch.cat([a, b], 1), torch.cat([a, c], 1)
    with torch.no_grad():
        frames_a, _ = backbone.encode_frames(a, torch.tensor([a.shape[1]]))
        t_a = frames_a.shape[-1]
        frames_ab, _ = backbone.encode_frames(ab, torch.tensor([ab.shape[1]]))
        frames_ac, _ = backbone.encode_frames(ac, torch.tensor([ac.shape[1]]))
    diff = (frames_ab[:, :, :t_a] - frames_ac[:, :, :t_a]).abs()
    per_frame_max = diff.max(dim=1).values[0]
    nonzero = (per_frame_max > 1e-4).nonzero().squeeze(-1)
    measured_frames = (t_a - nonzero.min().item()) if len(nonzero) > 0 else 0
    return {"max_diff": diff.max().item(), "measured_lookahead_frames": measured_frames,
            "measured_lookahead_ms": measured_frames * 10}


def measure_rotation_eer(backbone, manifest_path="evaluation/manifests/day1_public_speakers.json"):
    manifest = json.loads(Path(manifest_path).read_text())
    speakers = manifest["speakers"]
    speaker_ids = sorted(speakers.keys())

    def embed(samples, sr):
        with torch.no_grad():
            emb = backbone.encode_speaker(torch.from_numpy(samples).unsqueeze(0), torch.tensor([len(samples)]))
        return emb.squeeze(0).numpy()

    pos, neg = [], []
    for wearer_id in speaker_ids:
        enroll_embs = [embed(*load_wav(p)) for p in speakers[wearer_id]["enroll"]]
        wearer_emb = np.mean(enroll_embs, axis=0)
        wearer_emb /= np.linalg.norm(wearer_emb)
        for test_id in speaker_ids:
            for test_path in speakers[test_id]["test"]:
                samples, sr = load_wav(test_path)
                score = float(np.dot(embed(samples, sr), wearer_emb))
                (pos if test_id == wearer_id else neg).append(score)

    pos, neg = np.array(pos), np.array(neg)
    allscores = np.unique(np.concatenate([pos, neg]))
    diffs = [(abs((neg >= t).mean() - (pos < t).mean()), (neg >= t).mean(), (pos < t).mean()) for t in allscores]
    diffs.sort(key=lambda x: x[0])
    eer = (diffs[0][1] + diffs[0][2]) / 2
    return eer


if __name__ == "__main__":
    print(f"{'variant':22s} {'cfg_ms':>8s} {'measured_ms':>12s} {'max_diff':>10s} {'rotation_EER':>13s}")
    for name, budget_frames in VARIANTS.items():
        t0 = time.time()
        backbone = StreamingSpeakerNetBackbone(max_lookahead_frames=budget_frames)
        backbone.eval()
        lookahead_result = measure_lookahead(backbone)
        eer = measure_rotation_eer(backbone)
        elapsed = time.time() - t0
        print(f"{name:22s} {budget_frames*10:>7d}ms {lookahead_result['measured_lookahead_ms']:>11d}ms "
              f"{lookahead_result['max_diff']:>10.4f} {eer*100:>12.2f}%  ({elapsed:.0f}s)")
