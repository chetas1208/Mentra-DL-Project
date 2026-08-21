#!/usr/bin/env python3
"""Day 1 public-data prep (sprint spec, "CONTINUE DAY 1" continuation).

Selects N speakers from LibriSpeech test-clean, converts their utterances to
Mentra-format WAV (16kHz mono PCM16), and splits each speaker's utterances
into enroll/test with no overlap. Role assignment (who is "wearer" for a
given experiment) happens later in evaluate_day1.py, not here — this script
just prepares per-real-speaker audio + a master manifest.

Source: LibriSpeech test-clean (OpenSLR-12), CC BY 4.0.
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

TARGET_SR = 16000


def convert_to_mentra_wav(src_flac: Path, dst_wav: Path) -> None:
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src_flac),
            "-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le",
            str(dst_wav),
        ],
        check=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--librispeech-root", type=Path,
                     default=Path("evaluation/data/raw/LibriSpeech/test-clean"))
    ap.add_argument("--out-dir", type=Path,
                     default=Path("evaluation/data/day1_public"))
    ap.add_argument("--manifest-out", type=Path,
                     default=Path("evaluation/manifests/day1_public_speakers.json"))
    ap.add_argument("--num-speakers", type=int, default=8)
    ap.add_argument("--enroll-per-speaker", type=int, default=3)
    ap.add_argument("--test-per-speaker", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    speaker_dirs = sorted(
        d for d in args.librispeech_root.iterdir() if d.is_dir()
    )
    needed = args.enroll_per_speaker + args.test_per_speaker

    eligible = []
    for sdir in speaker_dirs:
        flacs = sorted(sdir.glob("*/*.flac"))
        if len(flacs) >= needed:
            eligible.append((sdir.name, flacs))

    if len(eligible) < args.num_speakers:
        raise RuntimeError(
            f"only {len(eligible)} speakers have >= {needed} utterances, "
            f"need {args.num_speakers}"
        )

    rng.shuffle(eligible)
    chosen = eligible[: args.num_speakers]

    manifest = {"speakers": {}, "source": "LibriSpeech test-clean (OpenSLR-12, CC BY 4.0)"}

    for speaker_id, flacs in chosen:
        flacs = flacs[:]
        rng.shuffle(flacs)
        enroll_files = flacs[: args.enroll_per_speaker]
        test_files = flacs[args.enroll_per_speaker: needed]

        entry = {"enroll": [], "test": []}
        for i, f in enumerate(enroll_files):
            dst = args.out_dir / speaker_id / f"enroll_{i:02d}.wav"
            convert_to_mentra_wav(f, dst)
            entry["enroll"].append(str(dst))
        for i, f in enumerate(test_files):
            dst = args.out_dir / speaker_id / f"test_{i:02d}.wav"
            convert_to_mentra_wav(f, dst)
            entry["test"].append(str(dst))

        manifest["speakers"][speaker_id] = entry
        print(f"speaker {speaker_id}: {len(entry['enroll'])} enroll, {len(entry['test'])} test")

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote manifest: {args.manifest_out} ({len(chosen)} speakers)")


if __name__ == "__main__":
    main()
