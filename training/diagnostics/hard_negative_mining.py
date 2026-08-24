#!/usr/bin/env python3
"""V2 diagnostic/infrastructure 2c: hard-negative speaker mining.

Loads the (already-fixed -- see speakernet_backbone.py's encode_speaker()
docstring for the ReLU bug that was found and fixed) SpeakerNetBackbone,
embeds every speaker's enrollment audio in a given manifest, computes
pairwise cosine similarity between speaker embeddings, and for each speaker
finds its K nearest (most similar / hardest-to-distinguish) other speakers.

This is standalone tooling: it is NOT wired into training/data/
mixture_generator.py's default random interferer sampling. It is used by:
  - training/diagnostics/build_validation_suite.py (2b), for the
    "hard-speaker-pair overlap" validation slice.
  - training/data/curriculum_sampler.py (2d)'s phase E, for
    hard-negative-speaker curriculum sampling.

CLI usage (builds and saves the training-manifest lookup table):
    .venv/bin/python3 training/diagnostics/hard_negative_mining.py \\
        --manifest evaluation/manifests/librispeech_train_clean_100_train.json \\
        --out evaluation/manifests/hard_negative_pairs.json --k 5 --device cpu

Also importable as a module (used this way by build_validation_suite.py to
compute a SEPARATE table for the held-out validation manifest, so the
validation suite is fully reproducible from a fixed seed + manifest without
depending on a pre-built JSON file existing on disk).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from training.data.mixture_generator import SpeakerPool, load_wav
from training.models.mentrawearnet import MentraWearNet


@torch.no_grad()
def embed_all_speakers(pool: SpeakerPool, model: MentraWearNet, device: str,
                        batch_size: int = 8) -> dict:
    """Returns {speaker_id: np.ndarray[256]} -- mean-pooled, L2-normalized
    embedding over each speaker's enroll-list clips, matching the same
    averaging convention scripts/model/evaluate_mentrawearnet.py uses for
    the wearer_embedding (mean of per-clip encode_enrollment() outputs).
    Modest batching (a handful of speakers at a time via a simple loop, not
    parallelized across all cores) -- this is meant to stay CPU-friendly
    alongside a concurrently-running training job, not to be maximally fast."""
    embeddings = {}
    for i, speaker_id in enumerate(pool.speaker_ids):
        clips = [load_wav(p) for p in pool.speakers[speaker_id]["enroll"]]
        embs = []
        for clip in clips:
            wf = torch.from_numpy(clip).unsqueeze(0).float().to(device)
            length = torch.tensor([len(clip)]).to(device)
            embs.append(model.encode_enrollment(wf, length))
        speaker_emb = torch.stack(embs).mean(dim=0)  # [1, 256], raw (matches training/eval convention)
        speaker_emb = torch.nn.functional.normalize(speaker_emb, p=2, dim=-1)  # normalize HERE for cosine-sim comparison only
        embeddings[speaker_id] = speaker_emb.squeeze(0).cpu().numpy()
        if (i + 1) % 50 == 0:
            print(f"  embedded {i+1}/{len(pool.speaker_ids)} speakers", flush=True)
    return embeddings


def compute_hard_negative_pairs(embeddings: dict, k: int = 5) -> dict:
    """Returns {speaker_id: [nearest_speaker_ids in descending similarity, len k]}.
    Pure numpy cosine similarity (embeddings are already L2-normalized by
    embed_all_speakers, so cosine similarity = dot product)."""
    speaker_ids = list(embeddings.keys())
    mat = np.stack([embeddings[s] for s in speaker_ids])  # [S, 256]
    sim = mat @ mat.T  # [S, S] cosine similarity (embeddings L2-normalized)
    np.fill_diagonal(sim, -np.inf)  # exclude self

    result = {}
    for i, speaker_id in enumerate(speaker_ids):
        order = np.argsort(-sim[i])[:k]
        result[speaker_id] = [speaker_ids[j] for j in order]
    return result, sim, speaker_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="evaluation/manifests/librispeech_train_clean_100_train.json")
    ap.add_argument("--out", default="evaluation/manifests/hard_negative_pairs.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"loading speaker pool from {args.manifest} ...")
    pool = SpeakerPool(args.manifest)
    print(f"  {len(pool.speaker_ids)} speakers")

    print("loading MentraWearNet backbone ...")
    model = MentraWearNet().to(args.device)
    model.eval()

    print(f"embedding all {len(pool.speaker_ids)} speakers' enrollment audio ...")
    embeddings = embed_all_speakers(pool, model, args.device)

    print(f"computing top-{args.k} hard-negative pairs ...")
    hard_pairs, sim, speaker_ids = compute_hard_negative_pairs(embeddings, k=args.k)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(hard_pairs, indent=2))
    print(f"saved {args.out}")

    # sanity check: report the globally hardest pair and a random pair, with
    # their similarity, for a human plausibility check (real output, not
    # fabricated -- see the caller's report for what these actually were).
    off_diag = sim.copy()
    i_max, j_max = np.unravel_index(np.argmax(off_diag), off_diag.shape)
    print(f"\nglobally hardest pair: {speaker_ids[i_max]} <-> {speaker_ids[j_max]}  "
          f"cos_sim={off_diag[i_max, j_max]:.4f}")
    rng = np.random.RandomState(0)
    ri, rj = rng.choice(len(speaker_ids), 2, replace=False)
    print(f"random pair for comparison: {speaker_ids[ri]} <-> {speaker_ids[rj]}  "
          f"cos_sim={sim[ri, rj]:.4f}")
    print(f"pairwise cosine similarity stats (off-diagonal): "
          f"min={sim[~np.eye(len(speaker_ids), dtype=bool)].min():.4f}  "
          f"max={sim[~np.eye(len(speaker_ids), dtype=bool)].max():.4f}  "
          f"mean={sim[~np.eye(len(speaker_ids), dtype=bool)].mean():.4f}")


if __name__ == "__main__":
    main()
