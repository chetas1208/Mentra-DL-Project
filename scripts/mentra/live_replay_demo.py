#!/usr/bin/env python3
"""Real end-to-end demo: real audio -> real network transport -> real
SpeakerNet inference -> live WEARER/ENVIRONMENT scores (sprint spec
section 15/23: "wire it now, not blocked on hardware source"). Uses real
LibriSpeech WAV files (not synthetic noise) streamed through the ACTUAL
WebSocketAudioTransport / MentraRemoteReceiver already proven in
scripts/mentra/test_remote_audio.py -- paced at real-time (10ms per
frame, matching Mentra's actual PCM cadence) rather than sent instantly.

This proves the receiver->model wiring works end to end. It does NOT prove
real Mentra hardware works -- that's a separate, still-open item (no
Bluetooth hardware in this dev session, see docs/MENTRA_REMOTE_AUDIO.md).
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mentra.audio.consumer import MentraInferenceConsumer
from mentra.audio.frame import AudioFrame, Codec
from mentra.audio.transport.websocket_transport import WebSocketAudioTransport
from research.sherpa_onnx.detector import SherpaOnnxWearerDetector
from server.audio.remote_receiver import MentraRemoteReceiver

HOST = "127.0.0.1"
PORT = 19198
FRAME_MS = 10
SAMPLES_PER_FRAME = int(16000 * FRAME_MS / 1000)


def load_wav(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


def float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype("<i2")
    return int16.tobytes()


async def stream_wav_realtime(client: WebSocketAudioTransport, wav_path: str, label: str):
    samples, sr = load_wav(wav_path)
    assert sr == 16000, f"expected 16kHz, got {sr}"
    n_frames = len(samples) // SAMPLES_PER_FRAME
    print(f"streaming {label} ({wav_path}): {n_frames} frames, {len(samples)/sr:.2f}s")
    for i in range(n_frames):
        chunk = samples[i * SAMPLES_PER_FRAME:(i + 1) * SAMPLES_PER_FRAME]
        frame = AudioFrame(
            sequence_number=0,  # transport assigns real sequence numbers
            capture_timestamp_ns=time.monotonic_ns(),
            sample_rate=16000, channels=1, bits_per_sample=16,
            payload=float32_to_pcm16_bytes(chunk), codec=Codec.PCM16,
        )
        await client.send_frame(frame)
        await asyncio.sleep(FRAME_MS / 1000)  # real-time pacing, matches actual Mentra cadence


async def main():
    manifest = json.loads(Path("evaluation/manifests/day1_public_speakers.json").read_text())
    speakers = manifest["speakers"]
    wearer_id = sorted(speakers.keys())[0]
    other_id = sorted(speakers.keys())[1]

    detector = SherpaOnnxWearerDetector("models/sherpa-speaker/nemo_en_speakerverification_speakernet.onnx")
    consumer = MentraInferenceConsumer(detector, context_s=2.0, hop_s=0.2)

    enroll_segments = [load_wav(p) for p in speakers[wearer_id]["enroll"]]
    consumer.enroll(enroll_segments)
    print(f"enrolled wearer={wearer_id} from {len(enroll_segments)} real clips")

    live_results = []

    def on_frame(frame: AudioFrame):
        result = consumer.consume_frame(frame)
        if result:
            live_results.append(result)
            print(f"  t={result.frame_capture_timestamp_ns/1e6:.0f}ms  "
                  f"score={result.wearer_score:.3f}  state={result.state:11s}  "
                  f"context={result.context_ms:.0f}ms  inference={result.inference_ms:.1f}ms  "
                  f"capture->prediction={result.capture_to_prediction_ms:.1f}ms")

    receiver = MentraRemoteReceiver(HOST, PORT, jitter_target_ms=20, jitter_max_ms=80)
    server_task = asyncio.create_task(receiver.serve(on_frame))
    await asyncio.sleep(0.2)

    client = WebSocketAudioTransport(f"ws://{HOST}:{PORT}", heartbeat_interval_s=2, dead_connection_s=10)
    await client.connect()

    print("\n=== streaming WEARER's own held-out test speech ===")
    await stream_wav_realtime(client, speakers[wearer_id]["test"][0], "WEARER speech")

    print("\n=== streaming a DIFFERENT speaker's test speech ===")
    await stream_wav_realtime(client, speakers[other_id]["test"][0], "ENVIRONMENT speech")

    await asyncio.sleep(0.3)
    await client.close()
    await asyncio.sleep(0.3)
    server_task.cancel()

    print(f"\nconsumer stats: {consumer.stats}")
    print(f"total live predictions made: {len(live_results)}")

    if live_results:
        latencies = [r.capture_to_prediction_ms for r in live_results]
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        inference_times = sorted(r.inference_ms for r in live_results)
        print(f"\ncapture->prediction latency: p50={p50:.1f}ms p95={p95:.1f}ms")
        print(f"inference-only time: p50={inference_times[len(inference_times)//2]:.1f}ms "
              f"p95={inference_times[int(len(inference_times)*0.95)]:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
