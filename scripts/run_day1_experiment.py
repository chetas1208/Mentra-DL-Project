#!/usr/bin/env python3
"""Day 1 rotation experiment: same-speaker vs different-speaker score
distributions, using real sherpa-onnx models against real LibriSpeech audio.

For each model, rotates the "wearer" role across every prepared speaker
(section 10 of the sprint spec: tests whether the pipeline is generic, not
tuned to one identity). For each rotation: enroll from that speaker's
enroll_*.wav (mean of L2-normalized embeddings), then score every prepared
speaker's test_*.wav against it — full utterance duration, no truncation.

Ground truth: test file's speaker == wearer speaker -> WEARER, else ENVIRONMENT.
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
from research.sherpa_onnx.detector import SherpaOnnxWearerDetector


def load_wav(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                     default=Path("evaluation/manifests/day1_public_speakers.json"))
    ap.add_argument("--model", required=True, help="path to sherpa-onnx speaker embedding .onnx")
    ap.add_argument("--model-name", required=True, help="short label for results, e.g. cam++")
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    speakers = manifest["speakers"]
    speaker_ids = sorted(speakers.keys())

    print(f"loading model {args.model_name} from {args.model}")
    detector = SherpaOnnxWearerDetector(args.model)
    print(f"embedding dim: {detector.dim}")

    rows = []
    for wearer_id in speaker_ids:
        enroll_segments = [load_wav(p) for p in speakers[wearer_id]["enroll"]]
        t0 = time.perf_counter()
        detector.enroll(enroll_segments)
        enroll_ms = (time.perf_counter() - t0) * 1000
        print(f"[{args.model_name}] wearer={wearer_id} enrolled from "
              f"{len(enroll_segments)} files in {enroll_ms:.1f}ms")

        for test_id in speaker_ids:
            for test_path in speakers[test_id]["test"]:
                samples, sr = load_wav(test_path)
                duration_s = len(samples) / sr
                t0 = time.perf_counter()
                result = detector.process(samples, sr)
                infer_ms = (time.perf_counter() - t0) * 1000
                rtf = (infer_ms / 1000) / duration_s if duration_s > 0 else float("nan")

                rows.append({
                    "model": args.model_name,
                    "wearer_speaker": wearer_id,
                    "test_speaker": test_id,
                    "test_file": test_path,
                    "ground_truth": "WEARER" if test_id == wearer_id else "ENVIRONMENT",
                    "score": result["wearer_score"],
                    "duration_s": round(duration_s, 3),
                    "infer_ms": round(infer_ms, 2),
                    "rtf": round(rtf, 4),
                })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
