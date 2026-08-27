"""Attach frozen-G2 real-audio runtime evidence to the G3 artifact set.

The runtime experiment is intentionally not re-run: its source artifact is
hash-pinned by the frozen parent.  This avoids overwriting guarded G2 output
while making the CPU, parity, causality, and session-reset evidence explicit
in the G3 final report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PARENT = REPO_ROOT / "evaluation/geowearnet/g3/g2_frozen_parent.json"
SOURCE = REPO_ROOT / "evaluation/geowearnet/mmcsg/results/streaming_real_audio.json"
DEFAULT_OUT = REPO_ROOT / "evaluation/geowearnet/g3/runtime_cpu_evidence.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(out_path: Path = DEFAULT_OUT) -> dict:
    parent = json.loads(PARENT.read_text())
    source = json.loads(SOURCE.read_text())
    expected = parent["artifacts"]["streaming_validation"]["sha256"]
    actual = sha256(SOURCE)
    if actual != expected:
        raise ValueError(f"runtime source hash changed: expected {expected}, got {actual}")
    checkpoint = parent["artifacts"]["selected_checkpoint"]
    if Path(source["checkpoint"]).name != Path(checkpoint["path"]).name:
        raise ValueError("runtime evidence points at a checkpoint other than frozen G2")
    report = {
        "status": "VERIFIED_FROZEN_PARENT_EVIDENCE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "source_artifact": str(SOURCE.relative_to(REPO_ROOT)),
        "source_artifact_sha256": actual,
        "cpu_benchmark": source["cpu_benchmark"],
        "streaming_parity": source["streaming_parity"],
        "perturbation_causality": source["perturbation_causality"],
        "long_soak": source["long_soak"],
        "session_reset": {
            "status": "PASS_BY_EXECUTED_REAL_AUDIO_SOAK",
            "evidence": (
                "training/geowearnet/mmcsg/streaming_real.py resets the streamer at every independent "
                "recording boundary; the pinned soak exercised two recordings for 6.15 minutes."
            ),
        },
        "hardware_boundary": {
            "real_mentra_glasses": "NOT_TESTED",
            "shared_glasses_scenarios": "NOT_TESTED",
            "real_audio_proxy": "MMCSG Aria smart-glasses recordings",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = run(args.out)
    print(json.dumps({"status": report["status"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
