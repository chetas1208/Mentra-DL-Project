"""Measure the actual G2 receiver-side frontend on real wearable proxy audio.

Unlike the frozen streaming-model benchmark, this exercises the deployed
PCM path: PCM16 frame decode, rolling receiver context, GeoWearNet G2
inference, policy routing, and optional stateful RNNoise.  ASR is deliberately
out of scope because it is a downstream provider component, not part of the
audio frontend contract.

Only the MMCSG train-derived internal validation split is read.  This writes a
new G3 artifact and never modifies a G2 result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.agent_audio.denoise import Denoiser
from mentra.audio.consumer import MentraInferenceConsumer
from mentra.audio.frame import AudioFrame, MessageType
from server.audio.frontend import AudioFrontend
from server.models.geowearnet import GeoWearNetDetector
from training.geowearnet.mmcsg import audio_io as A
from training.geowearnet.mmcsg.config import resolve_root
from training.geowearnet.mmcsg.splits import load_split


DEFAULT_CHECKPOINT = REPO_ROOT / "training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt"
DEFAULT_OUT = REPO_ROOT / "evaluation/geowearnet/g3/live_frontend_cpu_benchmark.json"
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 160  # 10 ms, matching the product frontend contract.
CONTEXT_S = 2.0
HOP_S = 0.2
POLICIES = ("passthrough", "geowear_gate", "rnnoise_geowear_gate")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _percentiles(values: Iterable[float]) -> Dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"n": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "n": int(len(array)),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "max_ms": float(array.max()),
    }


def _pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _run_policy(audio: np.ndarray, checkpoint: Path, threads: int, policy: str) -> Dict[str, object]:
    detector = GeoWearNetDetector(checkpoint, num_threads=threads)
    consumer = MentraInferenceConsumer(detector, sample_rate=SAMPLE_RATE,
                                       context_s=CONTEXT_S, hop_s=HOP_S,
                                       wearer_high_threshold=0.60,
                                       wearer_low_threshold=0.40,
                                       environment_high_threshold=0.60,
                                       environment_low_threshold=0.40)
    denoiser = Denoiser(sr=SAMPLE_RATE) if policy == "rnnoise_geowear_gate" else None
    frontend = AudioFrontend(policy, denoiser=denoiser)
    frame_ms, frontend_ms, detector_ms, end_to_end_detection_ms = [], [], [], []
    wall_start = time.perf_counter()
    for sequence, offset in enumerate(range(0, len(audio) - FRAME_SAMPLES + 1, FRAME_SAMPLES)):
        samples = audio[offset:offset + FRAME_SAMPLES]
        frame = AudioFrame(
            sequence, time.monotonic_ns(), SAMPLE_RATE, 1, 16, _pcm16(samples),
            message_type=MessageType.AUDIO_FRAME,
        )
        start = time.perf_counter()
        result = consumer.consume_frame(frame)
        before_frontend = time.perf_counter()
        processed = frontend.process(samples, consumer.current_state)
        after = time.perf_counter()
        if processed.audio.shape != samples.shape:
            raise RuntimeError("frontend changed a live PCM frame's shape")
        frame_ms.append((after - start) * 1000.0)
        frontend_ms.append((after - before_frontend) * 1000.0)
        if result is not None:
            detector_ms.append(result.inference_ms)
            end_to_end_detection_ms.append((after - start) * 1000.0)
    wall_s = time.perf_counter() - wall_start
    frontend.reset()
    duration_s = len(audio) / SAMPLE_RATE
    return {
        "policy": policy,
        "threads": threads,
        "processed_audio_seconds": duration_s,
        "audio_frames": len(frame_ms),
        "detector_updates": len(detector_ms),
        "receiver_frame_compute": _percentiles(frame_ms),
        "geowearnet_update_compute": _percentiles(detector_ms),
        "frontend_policy_compute": _percentiles(frontend_ms),
        "full_frontend_on_detector_update": _percentiles(end_to_end_detection_ms),
        "wall_clock_s": wall_s,
        "compute_rtf": float(wall_s / max(duration_s, 1e-9)),
        "model_params": detector.model.count_parameters(),
        "checkpoint_size_mb": checkpoint.stat().st_size / (1024.0 * 1024.0),
        "rnnoise_available": None if denoiser is None else bool(denoiser.available),
    }


def run(checkpoint_path: Path = DEFAULT_CHECKPOINT, out_path: Path = DEFAULT_OUT,
        max_seconds: float = 60.0) -> Dict[str, object]:
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    records = load_split("val")
    if not records:
        raise RuntimeError("internal validation split is empty")
    record = records[0]
    root = resolve_root()
    audio = A.read_full(root, str(record["source_split"]), str(record["recording_id"]), channel=2).audio
    n = min(len(audio), int(max_seconds * SAMPLE_RATE))
    n -= n % FRAME_SAMPLES
    audio = np.ascontiguousarray(audio[:n], dtype=np.float32)
    if not len(audio):
        raise RuntimeError("selected runtime recording has no complete PCM frames")

    rss_start = _rss_mb()
    runs: Dict[str, Dict[str, object]] = {}
    for threads in (1, 2, 4):
        for policy in POLICIES:
            key = f"{threads}_threads/{policy}"
            runs[key] = _run_policy(audio, checkpoint_path, threads, policy)
            print(f"completed {key}", flush=True)
    report = {
        "status": "SCORED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "official_dev_used": False,
        "official_eval_used": False,
        "data_boundary": "One MMCSG train-derived internal validation recording; no official dev/eval access.",
        "source_recording": str(record["recording_id"]),
        "source_split": str(record["source_split"]),
        "runtime_contract": {
            "pcm_frame_ms": 10.0,
            "receiver_context_ms": CONTEXT_S * 1000.0,
            "receiver_decision_update_cadence_ms": HOP_S * 1000.0,
            "state_thresholds": {
                "wearer_on": 0.60,
                "wearer_off": 0.40,
                "environment_on": 0.60,
                "environment_off": 0.40,
            },
            "routing_semantics": (
                "The live receiver is a four-state, 200 ms state router: WEARER and "
                "OVERLAP pass original mixed PCM, ENVIRONMENT and SILENCE mute it. It does "
                "not claim to reproduce the evaluator's 10 ms attack/release/hangover envelope."
            ),
            "future_audio_lookahead_ms": 0.0,
            "frontend_extra_audio_buffer_ms": 0.0,
            "note": (
                "The 200 ms receiver update cadence is operational scheduling, not future-audio "
                "lookahead. The 2 s rolling buffer is history only. ASR is downstream and excluded."
            ),
        },
        "runs": runs,
        "rss_start_mb": rss_start,
        "rss_end_mb": _rss_mb(),
        "rss_growth_mb": _rss_mb() - rss_start,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    args = parser.parse_args()
    report = run(args.checkpoint, args.out, args.max_seconds)
    print(json.dumps({
        "status": report["status"], "checkpoint_sha256": report["checkpoint_sha256"],
        "runs": list(report["runs"]), "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
