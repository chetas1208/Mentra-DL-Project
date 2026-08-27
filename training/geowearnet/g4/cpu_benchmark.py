"""G4 -- CPU cost of the WS1 routing change.

G3 benchmarked the live route with the coarse binary gate. WS1 replaced that
with a real 10 ms envelope, so the honest thing is to re-measure rather than
assume the change was free. This isolates the routing stage: same audio, same
frames, binary gate vs envelope vs envelope-with-pre-roll, no model in the
loop, so the number is the envelope's own cost and not the detector's.

The end-to-end real-time factor of the WHOLE live path (features + model +
envelope + PCM16) is measured separately in
``training/geowearnet/g4/parity.py``'s cadence sweep.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from server.audio.frontend import AudioFrontend
from server.audio.streaming_gate import StreamingGateRouter, named_gate_policy

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "evaluation/geowearnet/g4/routing_cpu_benchmark.json"
SR = 16000
HOP = 160


def _bench(frontend: AudioFrontend, audio: np.ndarray, n_frames: int,
           update_every: int) -> Dict[str, float]:
    frontend.reset()
    rng = np.random.default_rng(7)
    timings: List[float] = []
    for i in range(n_frames):
        block = audio[i * HOP:(i + 1) * HOP]
        t0 = time.perf_counter()
        if i % update_every == 0:
            frontend.update_probabilities(float(rng.random()), float(rng.random()))
        frontend.process(block, "WEARER")
        timings.append((time.perf_counter() - t0) * 1000.0)
    x = np.asarray(timings)
    return {
        "p50_ms_per_10ms_frame": float(np.percentile(x, 50)),
        "p95_ms_per_10ms_frame": float(np.percentile(x, 95)),
        "max_ms_per_10ms_frame": float(np.max(x)),
        "real_time_factor": float(np.mean(x) / 10.0),
        "n_frames": int(n_frames),
    }


def run(out_path: Path = DEFAULT_OUT, seconds: float = 60.0) -> Dict[str, object]:
    n_frames = int(seconds * 100)
    rng = np.random.default_rng(3)
    audio = (rng.standard_normal(n_frames * HOP) * 0.1).astype(np.float32)
    policy = named_gate_policy("A_balanced")

    rows = {
        "geowear_gate_binary_g3": _bench(AudioFrontend("geowear_gate"), audio, n_frames, 20),
        "geowear_envelope_preroll_0ms": _bench(
            AudioFrontend("geowear_envelope", router=StreamingGateRouter(policy, sr=SR)),
            audio, n_frames, 20),
        "geowear_envelope_preroll_150ms": _bench(
            AudioFrontend("geowear_envelope",
                          router=StreamingGateRouter(policy, sr=SR, preroll_ms=150.0)),
            audio, n_frames, 20),
    }
    report = {
        "status": "MEASURED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "workstream": "G4_ROUTING_CPU",
        "scope": ("routing stage only -- no model, no features, no ASR. Compares the G3 "
                  "binary state mute against the WS1 10 ms envelope on identical frames."),
        "sample_rate": SR,
        "frame_ms": 10.0,
        "audio_seconds": seconds,
        "rows": rows,
        "preroll_memory_bytes_150ms": int(0.150 * SR * 4),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.seconds), indent=2))


if __name__ == "__main__":
    main()
