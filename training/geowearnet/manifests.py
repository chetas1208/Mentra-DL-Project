"""GeoWearNet speaker-disjoint manifests (Workstream G).

The repo's existing LibriSpeech manifests
(`evaluation/manifests/librispeech_train_clean_100_{train,val,test}.json`) are
IMMUTABLE (they belong to the SpeakerNet/MentraWearNet lineage) and only list
~9 utterances per speaker (2,259 total) because they were built for
enrollment/verification trials, not for simulation training.

GeoWearNet needs a much larger utterance pool but must keep EXACTLY the same
speaker partition so that no cross-branch comparison is ever confounded by a
different split. This module therefore:

  1. reads the speaker IDs (only the IDs) from the three immutable manifests;
  2. re-scans `evaluation/data/raw/LibriSpeech/train-clean-100` for ALL flac
     files belonging to those speakers;
  3. writes NEW manifests under `evaluation/geowearnet/manifests/`, never
     touching the originals.

Guarantees asserted at build time and re-asserted by `verify()`:
  * speaker sets are pairwise disjoint (train/val/test intersections == 0);
  * utterance path sets are pairwise disjoint (no accidental clip reuse);
  * every speaker in the new manifest came from the corresponding old one;
  * the union of new speakers == the union of old speakers (nothing invented).

Split unit is the SPEAKER/PERSON, per Workstream G.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFESTS = {
    "train": REPO_ROOT / "evaluation/manifests/librispeech_train_clean_100_train.json",
    "val": REPO_ROOT / "evaluation/manifests/librispeech_train_clean_100_val.json",
    "test": REPO_ROOT / "evaluation/manifests/librispeech_train_clean_100_test.json",
}
LIBRISPEECH_ROOT = REPO_ROOT / "evaluation/data/raw/LibriSpeech/train-clean-100"
OUT_DIR = REPO_ROOT / "evaluation/geowearnet/manifests"
SPLITS = ("train", "val", "test")


def out_path(split: str) -> Path:
    return OUT_DIR / f"geowearnet_librispeech_{split}.json"


def _source_speaker_ids(split: str) -> List[str]:
    with open(SOURCE_MANIFESTS[split]) as f:
        m = json.load(f)
    return sorted(m["speakers"].keys())


def build() -> Dict[str, dict]:
    assert LIBRISPEECH_ROOT.exists(), f"missing corpus: {LIBRISPEECH_ROOT}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Full corpus index: speaker -> [rel paths]
    corpus: Dict[str, List[str]] = defaultdict(list)
    for flac in sorted(LIBRISPEECH_ROOT.rglob("*.flac")):
        spk = flac.relative_to(LIBRISPEECH_ROOT).parts[0]
        corpus[spk].append(str(flac.relative_to(REPO_ROOT)))

    manifests: Dict[str, dict] = {}
    for split in SPLITS:
        spks = _source_speaker_ids(split)
        speakers = {}
        for s in spks:
            utts = sorted(corpus.get(s, []))
            if not utts:
                raise RuntimeError(f"speaker {s} from {split} has no flac files on disk")
            speakers[s] = utts
        manifests[split] = {
            "source": "LibriSpeech train-clean-100 (CC BY 4.0)",
            "split": split,
            "split_unit": "speaker",
            "derived_from": str(SOURCE_MANIFESTS[split].relative_to(REPO_ROOT)),
            "note": (
                "Speaker partition copied verbatim from the immutable manifest above; "
                "utterance lists re-expanded to ALL utterances of those speakers. "
                "The source manifest was not modified."
            ),
            "n_speakers": len(speakers),
            "n_utterances": sum(len(v) for v in speakers.values()),
            "speakers": speakers,
        }

    verify(manifests)
    for split in SPLITS:
        p = out_path(split)
        with open(p, "w") as f:
            json.dump(manifests[split], f, indent=1)
    return manifests


def verify(manifests: Dict[str, dict] | None = None) -> dict:
    """Workstream G integrity check. Raises on any violation."""
    if manifests is None:
        manifests = {s: json.load(open(out_path(s))) for s in SPLITS}

    spk = {s: set(manifests[s]["speakers"].keys()) for s in SPLITS}
    utt = {s: {u for v in manifests[s]["speakers"].values() for u in v} for s in SPLITS}

    report = {"n_speakers": {}, "n_utterances": {}, "speaker_intersections": {}, "utterance_intersections": {}}
    for s in SPLITS:
        report["n_speakers"][s] = len(spk[s])
        report["n_utterances"][s] = len(utt[s])

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        si = len(spk[a] & spk[b])
        ui = len(utt[a] & utt[b])
        report["speaker_intersections"][f"{a}|{b}"] = si
        report["utterance_intersections"][f"{a}|{b}"] = ui
        assert si == 0, f"SPEAKER LEAKAGE {a}/{b}: {si} shared speakers"
        assert ui == 0, f"UTTERANCE LEAKAGE {a}/{b}: {ui} shared utterances"

    # every new speaker must trace back to the same split's immutable manifest
    for s in SPLITS:
        src = set(_source_speaker_ids(s))
        assert spk[s] == src, f"split {s} speaker set diverged from immutable manifest"
    report["split_matches_immutable_source"] = True
    report["total_speakers"] = len(spk["train"] | spk["val"] | spk["test"])
    report["total_utterances"] = sum(len(utt[s]) for s in SPLITS)
    return report


def manifest_hash(split: str) -> str:
    """Stable content hash of a split manifest, for checkpoint provenance."""
    p = out_path(split)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_speaker_index(split: str) -> Dict[str, List[str]]:
    """{speaker_id: [absolute utterance paths]}"""
    with open(out_path(split)) as f:
        m = json.load(f)
    return {s: [str(REPO_ROOT / u) for u in v] for s, v in m["speakers"].items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
        print("built:", *[str(out_path(s)) for s in SPLITS], sep="\n  ")
    rep = verify()
    print(json.dumps(rep, indent=2))
    print("hashes:", {s: manifest_hash(s)[:16] for s in SPLITS})


if __name__ == "__main__":
    main()
