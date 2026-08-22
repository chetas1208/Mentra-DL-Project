#!/usr/bin/env python3
"""HPC-side receiver entrypoint (matches the user's requested `mentra
audio-server` shape, minimal argparse -- no CLI framework added per repo
convention). Starts MentraRemoteReceiver, enrolls a wearer from local
LibriSpeech audio (real hardware enrollment replaces this once the source
fork is resolved), and prints live predictions as frames arrive from
whatever's connected on the other end of the SSH tunnel.

Usage (on the HPC):
    python3 scripts/mentra/run_receiver.py --listen 127.0.0.1 --port 8765

Then from the laptop:
    ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 \\
        -o ServerAliveCountMax=3 -L 127.0.0.1:8765:127.0.0.1:8765 USER@HPC_HOST
    python3 scripts/mentra/live_replay_demo.py  # (pointed at ws://127.0.0.1:8765
                                                  #  once it's parameterized -- see
                                                  #  docs/MENTRA_REMOTE_AUDIO.md)
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mentra.audio.consumer import MentraInferenceConsumer
from mentra.audio.frame import AudioFrame
from research.sherpa_onnx.detector import SherpaOnnxWearerDetector
from server.audio.remote_receiver import MentraRemoteReceiver

import numpy as np
import soundfile as sf


def load_wav(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listen", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="models/sherpa-speaker/nemo_en_speakerverification_speakernet.onnx")
    ap.add_argument("--context-s", type=float, default=2.0)
    ap.add_argument("--hop-s", type=float, default=0.2)
    args = ap.parse_args()

    print(f"loading SpeakerNet from {args.model} (CPU)")
    detector = SherpaOnnxWearerDetector(args.model)
    consumer = MentraInferenceConsumer(detector, context_s=args.context_s, hop_s=args.hop_s)

    manifest = json.loads(Path("evaluation/manifests/day1_public_speakers.json").read_text())
    speakers = manifest["speakers"]
    wearer_id = sorted(speakers.keys())[0]
    enroll_segments = [load_wav(p) for p in speakers[wearer_id]["enroll"]]
    consumer.enroll(enroll_segments)
    print(f"enrolled placeholder wearer={wearer_id} from {len(enroll_segments)} LibriSpeech clips "
          f"-- REPLACE with real Mentra enrollment once the hardware source is resolved")

    def on_frame(frame: AudioFrame):
        result = consumer.consume_frame(frame)
        if result:
            print(f"score={result.wearer_score:+.3f}  state={result.state:11s}  "
                  f"ctx={result.context_ms:.0f}ms  infer={result.inference_ms:.1f}ms  "
                  f"capture->prediction={result.capture_to_prediction_ms:.1f}ms", flush=True)

    receiver = MentraRemoteReceiver(args.listen, args.port, jitter_target_ms=20, jitter_max_ms=80)
    print(f"MentraRemoteReceiver listening on {args.listen}:{args.port}")
    print(f"waiting for a connection (e.g. from an SSH -L tunnel) ...")
    await receiver.serve(on_frame)


if __name__ == "__main__":
    asyncio.run(main())
