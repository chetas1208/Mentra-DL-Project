#!/usr/bin/env python3
"""Level C parity: rerun the exact rotation experiment with the NeMo
wrapper substituted for ONNX, to see whether the ~0.98 embedding-geometry
correlation translates into matching verification behavior (EER), which is
what actually matters for the product decision.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.parity.nemo_speakernet import NemoSpeakerNetWrapper


def load_wav(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                     default=Path("evaluation/manifests/day1_public_speakers.json"))
    ap.add_argument("--out-csv", type=Path,
                     default=Path("evaluation/results/parity_rotation_nemo.csv"))
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    speakers = manifest["speakers"]
    speaker_ids = sorted(speakers.keys())

    model = NemoSpeakerNetWrapper()

    rows = []
    for wearer_id in speaker_ids:
        enroll_embs = []
        for p in speakers[wearer_id]["enroll"]:
            samples, sr = load_wav(p)
            enroll_embs.append(model.embed(samples, sr))
        wearer_emb = np.mean(enroll_embs, axis=0)
        wearer_emb = wearer_emb / np.linalg.norm(wearer_emb)

        for test_id in speaker_ids:
            for test_path in speakers[test_id]["test"]:
                samples, sr = load_wav(test_path)
                t0 = time.perf_counter()
                emb = model.embed(samples, sr)
                infer_ms = (time.perf_counter() - t0) * 1000
                score = float(np.dot(emb, wearer_emb))
                rows.append({
                    "model": "nemo_speakernet",
                    "wearer_speaker": wearer_id, "test_speaker": test_id,
                    "test_file": test_path,
                    "ground_truth": "WEARER" if test_id == wearer_id else "ENVIRONMENT",
                    "score": score, "infer_ms": round(infer_ms, 2),
                })
        print(f"wearer={wearer_id} done")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
