#!/usr/bin/env python3
"""Overlap / target-to-interferer-ratio experiment (sprint spec section 14/65).

This is the escalation-gate experiment: does plain speaker verification
(current baseline: SpeakerNet) survive a second speaker talking at the same
time as the wearer? If wearer_score collapses under overlap, that's the
evidence needed to justify building a custom TS-VAD model (MentraWearNet).
If it holds up, custom-model work isn't justified yet.

Method: for each wearer, mix a held-out wearer-only test clip with a
different speaker's test clip (the "interferer") at controlled TIR, scale by
RMS energy. Score the mixture against the wearer's enrollment embedding.
Two conditions:
  - POSITIVE: wearer + interferer mixture (wearer IS present -- score should
    stay high if the model is overlap-robust)
  - NEGATIVE: interferer_A + interferer_B mixture, neither is the wearer
    (score should stay low regardless of TIR -- sanity check against false
    accepts from any two-voice mixture)
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.sherpa_onnx.detector import SherpaOnnxWearerDetector

TIR_DB_LEVELS = [10, 5, 0, -5, -10]


def load_wav(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def mix_at_tir(target: np.ndarray, interferer: np.ndarray, tir_db: float) -> np.ndarray:
    """Scales interferer so 20*log10(rms(target)/rms(interferer_scaled)) == tir_db,
    then sums truncated-to-shorter-length signals."""
    n = min(len(target), len(interferer))
    t = target[:n]
    i = interferer[:n]
    target_rms = rms(t)
    interferer_rms = rms(i)
    desired_interferer_rms = target_rms / (10 ** (tir_db / 20))
    scale = desired_interferer_rms / interferer_rms
    mixture = t + i * scale
    peak = np.max(np.abs(mixture))
    if peak > 1.0:
        mixture = mixture / peak  # avoid clipping distortion confounding the result
    return mixture


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                     default=Path("evaluation/manifests/day1_public_speakers.json"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    manifest = json.loads(args.manifest.read_text())
    speakers = manifest["speakers"]
    speaker_ids = sorted(speakers.keys())

    detector = SherpaOnnxWearerDetector(args.model)
    print(f"[{args.model_name}] dim={detector.dim}")

    rows = []
    for wearer_id in speaker_ids:
        enroll_segments = [load_wav(p) for p in speakers[wearer_id]["enroll"]]
        detector.enroll(enroll_segments)

        others = [s for s in speaker_ids if s != wearer_id]

        # POSITIVE: wearer + interferer
        wearer_test_files = speakers[wearer_id]["test"][:3]  # 3 wearer clips
        for wearer_clip_path in wearer_test_files:
            interferer_id = rng.choice(others)
            interferer_clip_path = rng.choice(speakers[interferer_id]["test"])
            w_samples, sr = load_wav(wearer_clip_path)
            i_samples, _ = load_wav(interferer_clip_path)

            for tir in TIR_DB_LEVELS:
                mixture = mix_at_tir(w_samples, i_samples, tir)
                result = detector.process(mixture, sr)
                rows.append({
                    "model": args.model_name, "condition": "WEARER_PLUS_INTERFERER",
                    "wearer_speaker": wearer_id, "interferer_speaker": interferer_id,
                    "tir_db": tir, "score": result["wearer_score"],
                    "expected": "HIGH (wearer present)",
                })

        # NEGATIVE: interferer_A + interferer_B, neither is wearer
        for _ in range(3):
            a_id, b_id = rng.sample(others, 2)
            a_clip = load_wav(rng.choice(speakers[a_id]["test"]))[0]
            b_clip = load_wav(rng.choice(speakers[b_id]["test"]))[0]
            for tir in TIR_DB_LEVELS:
                mixture = mix_at_tir(a_clip, b_clip, tir)
                result = detector.process(mixture, sr)
                rows.append({
                    "model": args.model_name, "condition": "TWO_NONWEARER_SPEAKERS",
                    "wearer_speaker": wearer_id, "interferer_speaker": f"{a_id}+{b_id}",
                    "tir_db": tir, "score": result["wearer_score"],
                    "expected": "LOW (wearer absent)",
                })
        print(f"[{args.model_name}] wearer={wearer_id} done")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
