#!/usr/bin/env python3
"""Duration-vs-accuracy curve (sprint spec section 12/46/93).

Truncates each test utterance to fixed durations, rescoring against the same
full-length enrollment, to find the shortest window that meets the accuracy
bar -- this directly drives the product's rolling-window design choice.
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

DURATIONS_S = [0.5, 1.0, 2.0, 3.0, None]  # None = full utterance


def load_wav(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                     default=Path("evaluation/manifests/day1_public_speakers.json"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    speakers = manifest["speakers"]
    speaker_ids = sorted(speakers.keys())

    detector = SherpaOnnxWearerDetector(args.model)
    print(f"[{args.model_name}] dim={detector.dim}")

    rows = []
    for wearer_id in speaker_ids:
        enroll_segments = [load_wav(p) for p in speakers[wearer_id]["enroll"]]
        detector.enroll(enroll_segments)

        for test_id in speaker_ids:
            for test_path in speakers[test_id]["test"]:
                samples, sr = load_wav(test_path)
                full_len = len(samples)
                for dur in DURATIONS_S:
                    if dur is None:
                        clip = samples
                        dur_label = round(full_len / sr, 3)
                    else:
                        n = int(dur * sr)
                        if n > full_len:
                            continue  # utterance shorter than this window
                        clip = samples[:n]
                        dur_label = dur

                    t0 = time.perf_counter()
                    result = detector.process(clip, sr)
                    infer_ms = (time.perf_counter() - t0) * 1000

                    rows.append({
                        "model": args.model_name,
                        "wearer_speaker": wearer_id,
                        "test_speaker": test_id,
                        "test_file": test_path,
                        "ground_truth": "WEARER" if test_id == wearer_id else "ENVIRONMENT",
                        "window_s": dur_label,
                        "score": result["wearer_score"],
                        "infer_ms": round(infer_ms, 2),
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
