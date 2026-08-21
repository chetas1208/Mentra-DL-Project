#!/usr/bin/env python3
"""SpeakerNet ONNX vs NeMo parity test (sprint spec sections 10, 13, 14, 15).

Level A (config) and Level B (embedding) parity here; Level C (rotation/
duration/overlap re-run) is scripts/run_day1_experiment.py etc. pointed at
the NeMo wrapper, run separately once this passes.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.parity.onnx_speakernet import OnnxSpeakerNetWrapper
from research.parity.nemo_speakernet import NemoSpeakerNetWrapper


def load_wav(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                     default=Path("evaluation/manifests/day1_public_speakers.json"))
    ap.add_argument("--onnx-model", default="models/sherpa-speaker/nemo_en_speakerverification_speakernet.onnx")
    ap.add_argument("--out-csv", type=Path, default=Path("evaluation/results/parity_embeddings.csv"))
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    speakers = manifest["speakers"]

    print("loading ONNX wrapper...")
    onnx_model = OnnxSpeakerNetWrapper(args.onnx_model)
    print(f"  dim={onnx_model.dim}")

    print("loading NeMo wrapper (downloads checkpoint on first run)...")
    nemo_model = NemoSpeakerNetWrapper()
    print(f"  dim={nemo_model.dim}")

    if onnx_model.dim != nemo_model.dim:
        print(f"WARNING: embedding dimension mismatch onnx={onnx_model.dim} nemo={nemo_model.dim} "
              "-- expected if the ONNX export changed output layer, investigate before trusting parity")

    all_files = []
    for sid, entry in speakers.items():
        for p in entry["enroll"] + entry["test"]:
            all_files.append((sid, p))

    rows = []
    embeddings_onnx = {}
    embeddings_nemo = {}
    for sid, path in all_files:
        samples, sr = load_wav(path)
        e_onnx = onnx_model.embed(samples, sr)
        e_nemo = nemo_model.embed(samples, sr)
        embeddings_onnx[path] = e_onnx
        embeddings_nemo[path] = e_nemo

        cos = float(np.dot(e_onnx[:min(len(e_onnx), len(e_nemo))],
                            e_nemo[:min(len(e_onnx), len(e_nemo))])) if len(e_onnx) == len(e_nemo) else None
        rows.append({
            "speaker": sid, "file": path,
            "onnx_dim": len(e_onnx), "nemo_dim": len(e_nemo),
            "onnx_vs_nemo_cosine": cos,
        })
        print(f"  {path}: onnx_dim={len(e_onnx)} nemo_dim={len(e_nemo)} cos={cos}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # pairwise geometry comparison (Level B, section 13): does the SAME PAIR
    # of files get a similar cosine score under both models, even if raw
    # embedding dims/values differ due to export transforms?
    files = list(embeddings_onnx.keys())
    pair_rows = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            a, b = files[i], files[j]
            ea_o, eb_o = embeddings_onnx[a], embeddings_onnx[b]
            ea_n, eb_n = embeddings_nemo[a], embeddings_nemo[b]
            if len(ea_o) != len(eb_o) or len(ea_n) != len(eb_n):
                continue
            cos_onnx = float(np.dot(ea_o, eb_o))
            cos_nemo = float(np.dot(ea_n, eb_n)) if len(ea_n) == len(eb_n) else None
            pair_rows.append({"file_a": a, "file_b": b, "cos_onnx": cos_onnx, "cos_nemo": cos_nemo})

    pair_csv = args.out_csv.parent / "parity_pairwise_geometry.csv"
    if pair_rows:
        with open(pair_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pair_rows[0].keys())
            writer.writeheader()
            writer.writerows(pair_rows)

        valid = [r for r in pair_rows if r["cos_nemo"] is not None]
        if valid:
            a = np.array([r["cos_onnx"] for r in valid])
            b = np.array([r["cos_nemo"] for r in valid])
            pearson = float(np.corrcoef(a, b)[0, 1])
            print(f"\npairwise score geometry: n_pairs={len(valid)} Pearson(cos_onnx, cos_nemo)={pearson:.4f}")
        else:
            print("\ncould not compute pairwise geometry -- embedding dims never matched")

    print(f"\nwrote {args.out_csv} and {pair_csv}")


if __name__ == "__main__":
    main()
