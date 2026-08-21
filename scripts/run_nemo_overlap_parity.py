#!/usr/bin/env python3
"""Level C parity, overlap condition: does NeMo reproduce the ONNX
overlap-degradation pattern (the actual escalation-gate evidence)? If NeMo
performs dramatically better under overlap, the two pipelines aren't truly
equivalent and MentraWearNet's initialization needs re-examining.
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
from research.parity.nemo_speakernet import NemoSpeakerNetWrapper

TIR_DB_LEVELS = [10, 5, 0, -5, -10]


def load_wav(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


def rms(x): return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def mix_at_tir(target, interferer, tir_db):
    n = min(len(target), len(interferer))
    t, i = target[:n], interferer[:n]
    scale = (rms(t) / (10 ** (tir_db / 20))) / rms(i)
    mixture = t + i * scale
    peak = np.max(np.abs(mixture))
    return mixture / peak if peak > 1.0 else mixture


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                     default=Path("evaluation/manifests/day1_public_speakers.json"))
    ap.add_argument("--out-csv", type=Path,
                     default=Path("evaluation/results/parity_overlap_nemo.csv"))
    ap.add_argument("--seed", type=int, default=42)  # same seed as the ONNX overlap run
    args = ap.parse_args()

    rng = random.Random(args.seed)
    manifest = json.loads(args.manifest.read_text())
    speakers = manifest["speakers"]
    speaker_ids = sorted(speakers.keys())

    model = NemoSpeakerNetWrapper()

    rows = []
    for wearer_id in speaker_ids:
        enroll_embs = [model.embed(*load_wav(p)) for p in speakers[wearer_id]["enroll"]]
        wearer_emb = np.mean(enroll_embs, axis=0)
        wearer_emb = wearer_emb / np.linalg.norm(wearer_emb)

        others = [s for s in speaker_ids if s != wearer_id]
        wearer_test_files = speakers[wearer_id]["test"][:3]
        for wearer_clip_path in wearer_test_files:
            interferer_id = rng.choice(others)
            interferer_clip_path = rng.choice(speakers[interferer_id]["test"])
            w_samples, sr = load_wav(wearer_clip_path)
            i_samples, _ = load_wav(interferer_clip_path)
            for tir in TIR_DB_LEVELS:
                mixture = mix_at_tir(w_samples, i_samples, tir)
                emb = model.embed(mixture, sr)
                score = float(np.dot(emb, wearer_emb))
                rows.append({"condition": "WEARER_PLUS_INTERFERER", "wearer_speaker": wearer_id,
                             "interferer_speaker": interferer_id, "tir_db": tir, "score": score})

        for _ in range(3):
            a_id, b_id = rng.sample(others, 2)
            a_clip = load_wav(rng.choice(speakers[a_id]["test"]))[0]
            b_clip = load_wav(rng.choice(speakers[b_id]["test"]))[0]
            for tir in TIR_DB_LEVELS:
                mixture = mix_at_tir(a_clip, b_clip, tir)
                emb = model.embed(mixture, sr)
                score = float(np.dot(emb, wearer_emb))
                rows.append({"condition": "TWO_NONWEARER_SPEAKERS", "wearer_speaker": wearer_id,
                             "interferer_speaker": f"{a_id}+{b_id}", "tir_db": tir, "score": score})
        print(f"wearer={wearer_id} done")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
