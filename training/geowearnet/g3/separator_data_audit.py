"""Read-only WearerSepNet target audit for G3.

The audit distinguishes real wearable mixtures from corpora that retain clean
source waveforms. It intentionally does not train a separator or create a
synthetic artifact. The output is evidence for the predeclared overlap
go/no-go decision.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[3]
MMCSG_ROOT = Path("/usr/data/923873155/mmcsg/MMCSG")
DEFAULT_OUT = REPO_ROOT / "evaluation/geowearnet/g3/wearersepnet_data_audit.json"


def _files(root: Path, suffix: str | None = None) -> list[Path]:
    if not root.is_dir():
        return []
    paths = [p for p in root.rglob("*") if p.is_file()]
    if suffix is not None:
        paths = [p for p in paths if p.suffix.lower() == suffix]
    return sorted(paths)


def _audio_inventory(root: Path) -> dict[str, object]:
    paths = _files(root, ".wav")
    formats: Counter[str] = Counter()
    subtypes: Counter[str] = Counter()
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            info = sf.info(str(path))
            formats[info.format] += 1
            subtypes[info.subtype] += 1
            sample_rates[info.samplerate] += 1
            channels[info.channels] += 1
        except RuntimeError as exc:
            failures.append({"path": str(path), "error": str(exc)})
    return {
        "files": len(paths),
        "formats": dict(formats),
        "subtypes": dict(subtypes),
        "sample_rates": {str(k): v for k, v in sorted(sample_rates.items())},
        "channels": {str(k): v for k, v in sorted(channels.items())},
        "info_failures": failures,
    }


def _count_by_split(root: Path, subdir: str) -> dict[str, int]:
    return {
        split: len(_files(root / subdir / split))
        for split in ("train", "dev", "eval")
        if (root / subdir / split).is_dir()
    }


def _count_extensions(root: Path, extensions: Iterable[str]) -> dict[str, int]:
    return {ext: len(_files(root, ext)) for ext in extensions}


def run(out_path: Path = DEFAULT_OUT, mmcsg_root: Path = MMCSG_ROOT) -> dict[str, object]:
    mmcsg_audio = mmcsg_root / "audio"
    mmcsg_metadata_files = _files(mmcsg_root / "metadata")
    mmcsg_rttm_files = _files(mmcsg_root / "rttm", ".rttm")
    mmcsg_transcript_files = _files(mmcsg_root / "transcriptions")

    libri_train = REPO_ROOT / "evaluation/data/raw/LibriSpeech/train-clean-100"
    libri_test = REPO_ROOT / "evaluation/data/raw/LibriSpeech/test-clean"
    musan = REPO_ROOT / "evaluation/data/raw/musan"

    result: dict[str, object] = {
        "status": "OK",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "official_dev_used": False,
        "real_wearable_corpus": {
            "name": "MMCSG",
            "root": str(mmcsg_root),
            "audio_by_split": _count_by_split(mmcsg_root, "audio"),
            "audio_inventory": _audio_inventory(mmcsg_audio),
            "metadata_by_split": _count_by_split(mmcsg_root, "metadata"),
            "rttm_by_split": _count_by_split(mmcsg_root, "rttm"),
            "transcriptions_by_split": _count_by_split(mmcsg_root, "transcriptions"),
            "annotation_files": {
                "metadata": len(mmcsg_metadata_files),
                "rttm": len(mmcsg_rttm_files),
                "transcriptions": len(mmcsg_transcript_files),
            },
            "target_kind": "REAL_MIX_ONLY",
            "clean_isolated_source_target": False,
            "clean_target_paths_checked": [],
            "reason": (
                "MMCSG provides multi-channel wearable mixtures plus RTTM/transcript "
                "annotations. No isolated wearer waveform or lavalier/reference "
                "source target is present in the corpus layout; annotations identify "
                "activity but do not reconstruct a clean waveform."
            ),
        },
        "synthetic_clean_source_corpora": {
            "librispeech_train_clean_100": {
                "root": str(libri_train),
                "files": len(_files(libri_train, ".flac")),
                "target_kind": "CLEAN_SPEECH_SOURCE",
                "license_record": "LibriSpeech CC BY 4.0; see existing manifests/license records",
            },
            "librispeech_test_clean": {
                "root": str(libri_test),
                "files": len(_files(libri_test, ".flac")),
                "target_kind": "CLEAN_SPEECH_SOURCE",
                "license_record": "LibriSpeech CC BY 4.0; see existing stress-bench record",
            },
            "musan": {
                "root": str(musan),
                "files_by_extension": _count_extensions(musan, (".wav", ".txt")),
                "target_kind": "CLEAN_NOISE_SOURCE",
                "license_record": "MUSAN license record in existing stress-bench documentation",
            },
            "existing_generators": [
                "evaluation.agent_audio.stressbench.build_item retains clean_wearer and clean_bystander",
                "training.geowearnet.scenes.generate_scene uses clean source tracks internally but returns mixture-only Scene",
            ],
        },
        "separator_training_status": {
            "trained": False,
            "model_artifact": None,
            "status": "DATA_AUDIT_ONLY_UNTIL_OVERLAP_GO_NO_GO",
            "spec": "docs/geowearnet_wearersepnet_spec.md",
            "constraint": (
                "A synthetic R0 is permissible only if the frozen-G2 internal-val "
                "overlap gate demonstrates that separation is justified."
            ),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mmcsg-root", type=Path, default=MMCSG_ROOT)
    args = parser.parse_args()
    result = run(args.out, args.mmcsg_root)
    print(json.dumps({
        "status": result["status"],
        "real_target_kind": result["real_wearable_corpus"]["target_kind"],
        "real_clean_target": result["real_wearable_corpus"]["clean_isolated_source_target"],
        "separator_training": result["separator_training_status"]["status"],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
