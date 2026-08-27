"""Freeze the G2 parent artifacts before any G3 experiment runs.

G3 must be reproducible from this manifest and must never overwrite a G2
result. The manifest contains only source paths, metadata, and hashes; it does
not copy MMCSG data into the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[3]
G2_SELECTION = REPO_ROOT / "evaluation/geowearnet/mmcsg/results/g2_final_selection.json"
OUT = REPO_ROOT / "evaluation/geowearnet/g3/g2_frozen_parent.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def freeze(selection_path: Path = G2_SELECTION, out_path: Path = OUT) -> Dict[str, object]:
    selection = json.loads(selection_path.read_text())
    freeze_info = selection.get("freeze", {})
    if selection.get("status") != "EXECUTED" or not freeze_info.get("copy_verified"):
        raise RuntimeError(f"G2 selection is not an executed, copy-verified selection: {selection_path}")

    checkpoint = _repo_path(freeze_info["frozen_path"])
    expected_hash = freeze_info.get("verified_sha256_after_copy") or freeze_info.get("sha256")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256(checkpoint)
    if expected_hash and checkpoint_hash != expected_hash:
        raise RuntimeError(
            f"G2 checkpoint hash mismatch: expected {expected_hash}, got {checkpoint_hash}"
        )

    artifacts = {
        "selected_checkpoint": checkpoint,
        "final_selection": selection_path,
        "official_final_dev": REPO_ROOT / "evaluation/geowearnet/mmcsg/results/g2_final_dev_evaluation.json",
        "runtime_normalization_stats": REPO_ROOT / "training/geowearnet/mmcsg/cache/norm_stats_real_ch2_0.npz",
        "feature_config": REPO_ROOT / "training/geowearnet/features.py",
        "model_architecture": REPO_ROOT / "training/geowearnet/model_zoo.py",
        "export_artifact": REPO_ROOT / "evaluation/geowearnet/results/geowearnet_g2_final_deploy.json",
        "streaming_validation": REPO_ROOT / "evaluation/geowearnet/mmcsg/results/streaming_real_audio.json",
    }
    records: Dict[str, object] = {}
    for name, path in artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing G2 parent artifact {name}: {path}")
        records[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }

    checkpoint_data = torch_load_checkpoint_metadata(checkpoint)
    result = {
        "status": "G2_PARENT_FROZEN",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_selection": str(selection_path.relative_to(REPO_ROOT)),
        "selected_model": selection.get("selection", {}).get("selected", {}).get("name"),
        "selection_metrics": {
            key: selection.get("selection", {}).get("selected", {}).get(key)
            for key in ("solo_auroc", "false_wearer_rate", "params", "context_ms")
        },
        "checkpoint_metadata": checkpoint_data,
        "artifacts": records,
        "g3_write_boundary": "evaluation/geowearnet/g3/ and training/geowearnet/g3/ only; G2 artifacts immutable",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def torch_load_checkpoint_metadata(path: Path) -> Dict[str, object]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "config": checkpoint.get("config", {}),
        "model_config": checkpoint.get("model_config", {}),
        "params": checkpoint.get("params"),
        "receptive_field_frames": checkpoint.get("receptive_field_frames"),
        "context_ms": checkpoint.get("context_ms"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=G2_SELECTION)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    result = freeze(args.selection, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
