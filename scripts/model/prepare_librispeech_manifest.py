#!/usr/bin/env python3
"""Speaker-disjoint train/val/test manifest from LibriSpeech train-clean-100
(sprint spec: "never split by waveform, split by speaker" -- zero identity
overlap between splits). Produces the same {speakers: {id: {enroll, test}}}
shape as evaluation/manifests/day1_public_speakers.json, so
training/data/mixture_generator.py's SpeakerPool works unmodified against
either the tiny 8-speaker set or this larger real corpus.

The original day1_public 8 speakers (drawn from test-clean) are a disjoint
corpus already (test-clean vs train-clean-100 are different LibriSpeech
speaker pools) -- kept frozen as the historical regression set per the plan.
"""
import argparse
import json
import random
from pathlib import Path


def build_speaker_entry(speaker_dir: Path, n_enroll: int, n_test: int, rng: random.Random):
    flacs = sorted(speaker_dir.glob("*/*.flac"))
    if len(flacs) < n_enroll + n_test:
        return None
    flacs = flacs[:]
    rng.shuffle(flacs)
    return {
        "enroll": [str(p) for p in flacs[:n_enroll]],
        "test": [str(p) for p in flacs[n_enroll:n_enroll + n_test]],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--librispeech-root", type=Path,
                     default=Path("evaluation/data/raw/LibriSpeech/train-clean-100"))
    ap.add_argument("--out-dir", type=Path, default=Path("evaluation/manifests"))
    ap.add_argument("--n-enroll", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--val-speakers", type=int, default=20)
    ap.add_argument("--test-speakers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    speaker_dirs = sorted(d for d in args.librispeech_root.iterdir() if d.is_dir())
    print(f"found {len(speaker_dirs)} speaker directories in {args.librispeech_root}")

    entries = {}
    for sdir in speaker_dirs:
        entry = build_speaker_entry(sdir, args.n_enroll, args.n_test, rng)
        if entry is not None:
            entries[sdir.name] = entry
    print(f"{len(entries)} speakers have enough utterances (>= {args.n_enroll + args.n_test})")

    speaker_ids = list(entries.keys())
    rng.shuffle(speaker_ids)

    val_ids = set(speaker_ids[:args.val_speakers])
    test_ids = set(speaker_ids[args.val_speakers:args.val_speakers + args.test_speakers])
    train_ids = set(speaker_ids) - val_ids - test_ids

    assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids), \
        "speaker split overlap -- this must never happen"

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        manifest = {
            "source": "LibriSpeech train-clean-100 (OpenSLR-12, CC BY 4.0)",
            "split": split_name,
            "speakers": {sid: entries[sid] for sid in sorted(ids)},
        }
        out_path = args.out_dir / f"librispeech_train_clean_100_{split_name}.json"
        out_path.write_text(json.dumps(manifest, indent=2))
        print(f"{split_name}: {len(ids)} speakers -> {out_path}")

    print("\nspeaker-disjoint check: train/val/test share zero speaker IDs (asserted above, passed)")


if __name__ == "__main__":
    main()
