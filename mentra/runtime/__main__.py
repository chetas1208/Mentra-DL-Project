#!/usr/bin/env python3
"""python -m mentra.runtime -- CLI entrypoint for the receiver side.
Thin wrapper around the same logic as scripts/mentra/run_receiver.py
(kept as a convenience script too); this is the `python -m mentra.runtime`
shape referenced in docs/MENTRA_REMOTE_AUDIO.md and earlier planning.

Only --source remote is implemented. --source desktop / --source android
are placeholders for MentraDesktopPcmSource / the Android relay, neither
of which exists yet (blocked on the hardware-source fork).
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mentra.audio.consumer import MentraInferenceConsumer
from mentra.audio.frame import AudioFrame
from research.sherpa_onnx.detector import SherpaOnnxWearerDetector
from server.audio.remote_receiver import MentraRemoteReceiver

import numpy as np
import soundfile as sf

MODEL_PATHS = {
    "speakernet": "models/sherpa-speaker/nemo_en_speakerverification_speakernet.onnx",
    "resnet34_lm": "models/sherpa-speaker/wespeaker_en_voxceleb_resnet34_LM.onnx",
    "campplus": "models/sherpa-speaker/wespeaker_en_voxceleb_CAM++.onnx",
}


def load_wav(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


async def run_remote(args):
    model_path = MODEL_PATHS.get(args.model, args.model)
    print(f"loading {args.model} from {model_path} (CPU)")
    detector = SherpaOnnxWearerDetector(model_path)
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
    print("waiting for a connection ...")
    try:
        await receiver.serve(on_frame)
    except OSError as e:
        print(f"BLOCKED: could not bind {args.listen}:{args.port} -- {e}", file=sys.stderr)
        print("(if another receiver is already running on this port -- e.g. from "
              "scripts/mentra/run_receiver.py -- stop that one first, or use --port to pick a different one)",
              file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["remote", "desktop", "android"], default="remote")
    ap.add_argument("--listen", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="speakernet", help=f"one of {list(MODEL_PATHS)} or a direct .onnx path")
    ap.add_argument("--device", default="cpu", choices=["cpu"], help="cpu only -- GPU inference is out of scope here")
    ap.add_argument("--context-s", type=float, default=2.0)
    ap.add_argument("--hop-s", type=float, default=0.2)
    args = ap.parse_args()

    if args.source != "remote":
        print(f"BLOCKED: --source {args.source} is not implemented yet "
              f"(blocked on the hardware-source fork -- see docs/MENTRA_REMOTE_AUDIO.md). "
              f"Only --source remote (the WebSocket receiver) works right now.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_remote(args))


if __name__ == "__main__":
    main()
