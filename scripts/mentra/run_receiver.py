#!/usr/bin/env python3
"""HPC-side receiver entrypoint. Starts MentraRemoteReceiver, enrolls a
wearer from local LibriSpeech audio (real hardware enrollment replaces
this once the source fork is resolved), and streams back four things per
connection: live DETECTION results, gated WEARER_PCM / ENVIRONMENT_PCM
audio (silence outside the matching state -- NOT true source separation,
see docs/MENTRA_REMOTE_AUDIO.md), and live TRANSCRIPT text from a
lightweight streaming ASR model (sherpa-onnx, same library as SpeakerNet).

Usage:
    python3 scripts/mentra/run_receiver.py --listen 127.0.0.1 --port 8765
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mentra.audio.consumer import MentraInferenceConsumer, pcm16_bytes_to_float32
from mentra.audio.frame import AudioFrame, MessageType
from research.sherpa_onnx.detector import SherpaOnnxWearerDetector
from server.audio.remote_receiver import MentraRemoteReceiver

import numpy as np
import sherpa_onnx
import soundfile as sf

ASR_MODEL_DIR = "models/sherpa-asr/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"


def load_wav(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


def load_asr_recognizer(num_threads: int = 2) -> sherpa_onnx.OnlineRecognizer:
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=f"{ASR_MODEL_DIR}/encoder.onnx",
        decoder=f"{ASR_MODEL_DIR}/decoder.onnx",
        joiner=f"{ASR_MODEL_DIR}/joiner.onnx",
        tokens=f"{ASR_MODEL_DIR}/tokens.txt",
        num_threads=num_threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
    )


class ConnectionSession:
    """Per-connection state: one ASR stream, one last-emitted transcript,
    kept separate from MentraInferenceConsumer (which is currently shared
    across connections -- fine for a single-wearer demo, would need to
    become per-connection too for multi-user support, not needed yet)."""

    def __init__(self, asr_recognizer: sherpa_onnx.OnlineRecognizer):
        self.asr_recognizer = asr_recognizer
        self.asr_stream = asr_recognizer.create_stream()
        self.last_transcript = ""


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listen", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="models/sherpa-speaker/nemo_en_speakerverification_speakernet.onnx")
    ap.add_argument("--context-s", type=float, default=2.0)
    ap.add_argument("--hop-s", type=float, default=0.2)
    ap.add_argument("--asr-threads", type=int, default=2)
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

    print(f"loading streaming ASR from {ASR_MODEL_DIR} (CPU, {args.asr_threads} threads)")
    asr_recognizer = load_asr_recognizer(args.asr_threads)
    session = ConnectionSession(asr_recognizer)
    print("ASR loaded -- transcribing continuously (not gated to ENVIRONMENT state; "
          "gating ASR input by classification state is a real next step, not done here)")

    # Mutable dict (not a plain variable) so on_playback_mode's closure can
    # rebind it and on_frame sees the update -- single shared value across
    # connections, matching this receiver's documented single-wearer-demo
    # scope (same pattern as the module-level `consumer`/`session` state).
    playback_mode = {"value": "both"}

    def on_frame(frame: AudioFrame):
        responses: list[AudioFrame] = []

        if frame.sample_rate != 16000 or frame.bits_per_sample != 16 or frame.channels != 1:
            return None
        samples_f32 = pcm16_bytes_to_float32(frame.payload)

        # 1. Detection -- only produces a new result on hop boundaries,
        # but current_state is readable every frame for gating below.
        result = consumer.consume_frame(frame)
        if result:
            print(f"score={result.wearer_score:+.3f}  state={result.state:11s}  "
                  f"ctx={result.context_ms:.0f}ms  infer={result.inference_ms:.1f}ms", flush=True)
            payload = json.dumps({
                "wearer_score": result.wearer_score,
                "state": result.state,
                "context_ms": result.context_ms,
                "inference_ms": result.inference_ms,
                "echoed_capture_timestamp_ns": str(frame.capture_timestamp_ns),
            }).encode("utf-8")
            responses.append(AudioFrame(
                sequence_number=0, capture_timestamp_ns=frame.capture_timestamp_ns,
                sample_rate=16000, channels=1, bits_per_sample=16,
                payload=payload, message_type=MessageType.DETECTION,
            ))

        # 2. Gated audio -- every frame, not just on hops. NOT source
        # separation: this is the original mixed PCM, passed through when
        # the current state matches, silence otherwise. See section on
        # "classified/gated streams vs true separation" in
        # docs/MENTRA_REMOTE_AUDIO.md.
        #
        # The client tells us (via SET_PLAYBACK_MODE, see on_playback_mode
        # below) which of these two streams it actually wants. Previously
        # both were always built and sent regardless, and the client just
        # silently zeroed the gain on whichever one it didn't want -- real,
        # measurable waste (WEARER_PCM/ENVIRONMENT_PCM are the same size as
        # the input AUDIO_FRAME, so unwanted-stream traffic roughly doubled
        # this connection's outbound bandwidth for no benefit). Skipping the
        # unwanted AudioFrame entirely here means it's never encoded, never
        # queued, never sent -- not just muted client-side.
        state = consumer.current_state
        silence_payload = b"\x00" * len(frame.payload)
        if playback_mode["value"] in ("both", "wearer"):
            wearer_payload = frame.payload if state == "WEARER" else silence_payload
            responses.append(AudioFrame(
                sequence_number=frame.sequence_number, capture_timestamp_ns=frame.capture_timestamp_ns,
                sample_rate=16000, channels=1, bits_per_sample=16,
                payload=wearer_payload, message_type=MessageType.WEARER_PCM,
            ))
        if playback_mode["value"] in ("both", "environment"):
            env_payload = frame.payload if state == "ENVIRONMENT" else silence_payload
            responses.append(AudioFrame(
                sequence_number=frame.sequence_number, capture_timestamp_ns=frame.capture_timestamp_ns,
                sample_rate=16000, channels=1, bits_per_sample=16,
                payload=env_payload, message_type=MessageType.ENVIRONMENT_PCM,
            ))

        # 3. ASR -- fed continuously (not gated by state; a real next step
        # would only feed ENVIRONMENT-state audio, transcribing the wearer's
        # own speech too isn't the product goal, but that gating isn't
        # implemented yet -- documented here, not silently done).
        session.asr_stream.accept_waveform(16000, samples_f32)
        while session.asr_recognizer.is_ready(session.asr_stream):
            session.asr_recognizer.decode_stream(session.asr_stream)
        text = session.asr_recognizer.get_result(session.asr_stream)
        if text != session.last_transcript:
            session.last_transcript = text
            responses.append(AudioFrame(
                sequence_number=0, capture_timestamp_ns=frame.capture_timestamp_ns,
                sample_rate=16000, channels=1, bits_per_sample=16,
                payload=json.dumps({"text": text}).encode("utf-8"),
                message_type=MessageType.TRANSCRIPT,
            ))

        return responses if responses else None

    receiver = MentraRemoteReceiver(args.listen, args.port, jitter_target_ms=20, jitter_max_ms=80)
    print(f"MentraRemoteReceiver listening on {args.listen}:{args.port}")
    def on_enroll(frame: AudioFrame):
        # Replaces the placeholder LibriSpeech enrollment with the real
        # connected user's own voice -- the fix for "live scores look
        # wrong": the model was comparing the tester's voice against a
        # stranger's embedding until this fires.
        samples_f32 = pcm16_bytes_to_float32(frame.payload)
        consumer.enroll([(samples_f32, 16000)])
        duration_s = len(samples_f32) / 16000
        print(f"REAL ENROLLMENT received: {duration_s:.1f}s of audio -- "
              f"wearer embedding replaced (was placeholder, now real)", flush=True)

    def on_playback_mode(mode: str):
        if mode not in ("both", "wearer", "environment", "muted"):
            print(f"ignoring unknown playback mode {mode!r}", flush=True)
            return
        playback_mode["value"] = mode
        print(f"playback mode set to {mode!r} -- server will stop sending "
              f"the unselected stream(s) entirely", flush=True)

    print("waiting for a connection ...")
    await receiver.serve(on_frame, on_enroll, on_playback_mode)


if __name__ == "__main__":
    asyncio.run(main())
