#!/usr/bin/env python3
"""Offline evaluation harness (sprint spec section 16).

Reads a session manifest (list of {file, label, session, condition, snr}),
runs a chosen WearerDetector backend over each labeled WAV segment, and
writes per-item predictions + aggregate metrics.

NOT STARTED / RESEARCH ONLY: no backend is wired in yet. This is scaffolding
so a real run just has to plug in a `predict_fn`.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

VALID_LABELS = {"WEARER", "ENVIRONMENT"}


def load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        items = json.load(f)
    for item in items:
        if item["label"] not in VALID_LABELS:
            raise ValueError(f"bad label {item['label']!r} in {item['file']}")
    return items


def run(manifest_path: Path, audio_dir: Path, out_path: Path, predict_fn) -> None:
    items = load_manifest(manifest_path)
    rows = []
    for item in items:
        wav_path = audio_dir / item["file"]
        if not wav_path.exists():
            raise FileNotFoundError(wav_path)
        pred = predict_fn(wav_path)
        rows.append({
            "file": item["file"],
            "session": item.get("session"),
            "condition": item.get("condition"),
            "ground_truth": item["label"],
            "prediction": pred["prediction"],
            "confidence": pred.get("confidence"),
            "latency_ms": pred.get("latency_ms"),
            "correct": pred["prediction"] == item["label"],
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} predictions to {out_path}")


def _no_backend(_wav_path: Path):
    raise NotImplementedError(
        "no WearerDetector backend wired in yet — pass --backend or edit "
        "evaluate.py to import one from research/<candidate>/"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("out_csv", type=Path)
    args = parser.parse_args()

    try:
        run(args.manifest, args.audio_dir, args.out_csv, _no_backend)
    except NotImplementedError as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        sys.exit(1)
