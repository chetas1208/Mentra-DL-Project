#!/usr/bin/env python3
"""HPC-side receiver entrypoint -- multi-model, one audio pipeline.

Both live models are loaded once at startup and offered to every connecting
client through a real model catalog:

    speakernet     -- the identity/enrollment baseline (WHO is speaking)
    geowearnet_g2  -- the frozen, enrollment-free wearable-domain checkpoint
                      (IS speech coming from the wearer position)

A client selects ONE of them per session via the SESSION_CONFIG control
message; a client that selects nothing gets the default model, which stays
``MENTRA_MODEL`` (default ``speakernet``) exactly as before. Model choice is
SESSION state -- two concurrent sessions can run different models without a
restart and without touching each other.

The audio path is unchanged and model-agnostic: binary AudioFrame transport ->
jitter buffer -> LiveSession -> selected model -> DETECTION / gated
WEARER_PCM / ENVIRONMENT_PCM (silence outside the matching state -- NOT true
source separation, see ``docs/MENTRA_REMOTE_AUDIO.md``) -> TRANSCRIPT.

Usage:
    python3 scripts/mentra/run_receiver.py --listen 127.0.0.1 --port 8765
"""
import argparse
import asyncio
import errno
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from server.audio.live_session import LiveSessionConfig, LiveSessionFactory
from server.audio.frontend import ENVELOPE_POLICIES, SUPPORTED_POLICIES
from server.audio.remote_receiver import MentraRemoteReceiver
from server.models.capabilities import (DEFAULT_MODEL, LIVE_SERVABLE_MODELS,
                                        active_model_id)
from server.models.registry import ModelRegistry

import numpy as np
import sherpa_onnx
import soundfile as sf

ASR_MODEL_DIR = "models/sherpa-asr/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
REPO_ROOT = Path(__file__).resolve().parents[2]
G2_SELECTION = REPO_ROOT / "evaluation/geowearnet/mmcsg/results/g2_final_selection.json"
PLACEHOLDER_ENROLLMENT_MANIFEST = "evaluation/manifests/day1_public_speakers.json"


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


def load_placeholder_enrollment():
    """The historical LibriSpeech placeholder embedding SpeakerNet starts from.

    Unchanged behaviour: every SpeakerNet session begins on this placeholder
    and the connected user's own ENROLL_AUDIO replaces it -- now for that
    session only, rather than for the whole process.
    """
    manifest_path = Path(PLACEHOLDER_ENROLLMENT_MANIFEST)
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    speakers = manifest["speakers"]
    wearer_id = sorted(speakers.keys())[0]
    segments = [load_wav(p) for p in speakers[wearer_id]["enroll"]]
    print(f"placeholder SpeakerNet enrollment: wearer={wearer_id} from "
          f"{len(segments)} LibriSpeech clips -- replaced per session by real ENROLL_AUDIO")
    return segments


def resolve_g2_checkpoint(explicit: str | None) -> str | None:
    """Resolve the frozen G2 checkpoint from the executed final-selection
    artifact. Returns None if neither an explicit path nor a valid artifact
    exists -- GeoWearNet then simply reports ready=false and SpeakerNet is
    unaffected."""
    if explicit:
        return explicit
    if not G2_SELECTION.is_file():
        return None
    try:
        selection = json.loads(G2_SELECTION.read_text())
    except json.JSONDecodeError:
        return None
    freeze = selection.get("freeze", {})
    if selection.get("status") != "EXECUTED" or not freeze.get("copy_verified"):
        return None
    frozen = freeze.get("frozen_path")
    if not frozen:
        return None
    path = Path(frozen)
    return str(path if path.is_absolute() else REPO_ROOT / path)


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listen", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="models/sherpa-speaker/nemo_en_speakerverification_speakernet.onnx")
    ap.add_argument(
        "--default-model",
        default=None,
        choices=LIVE_SERVABLE_MODELS,
        help="model served to clients that never send a session_config. "
             "Defaults to MENTRA_MODEL, which defaults to speakernet.",
    )
    ap.add_argument("--context-s", type=float, default=2.0)
    ap.add_argument("--hop-s", type=float, default=0.2)
    ap.add_argument("--asr-threads", type=int, default=2)
    ap.add_argument("--no-asr", action="store_true",
                    help="skip loading the streaming ASR model (smoke tests / benchmarks)")
    ap.add_argument(
        "--bind-retry-s", type=float, default=0.0,
        help="keep retrying the listen bind for this many seconds if the port is "
             "still held. Used for a load-then-swap production cutover: the "
             "replacement finishes loading every model FIRST, then takes the "
             "port the instant the old process releases it.",
    )
    ap.add_argument(
        "--audio-policy",
        choices=SUPPORTED_POLICIES,
        default=os.environ.get("MENTRA_AUDIO_POLICY", "passthrough"),
        help="live audio policy; default preserves the existing passthrough path. "
             "The *_envelope policies use the offline evaluator's own gate envelope "
             "(attack/release/hangover/crossfade/pre-roll) instead of the coarse binary mute.",
    )
    ap.add_argument(
        "--gate-policy",
        default=os.environ.get("MENTRA_GATE_POLICY", "A_balanced"),
        help="named GatePolicy variant used by the *_envelope audio policies",
    )
    ap.add_argument(
        "--gate-preroll-ms",
        type=float,
        default=float(os.environ.get("MENTRA_GATE_PREROLL_MS", "0")),
        help="bounded pre-roll for the *_envelope policies. Real added algorithmic "
             "delay, reported in the startup banner -- not lookahead.",
    )
    ap.add_argument(
        "--capture-dir",
        default=os.environ.get("MENTRA_CAPTURE_DIR"),
        help="enable research capture: write per-session raw/derived PCM + metadata "
             "under this directory. Off unless explicitly set.",
    )
    ap.add_argument(
        "--geowearnet-checkpoint",
        default=os.environ.get("GEOWEARNET_G2_CHECKPOINT"),
        help="G2 checkpoint; defaults to the frozen checkpoint in the final-selection artifact",
    )
    ap.add_argument("--geowearnet-threads", type=int, default=1)
    ap.add_argument("--speaker-threads", type=int, default=1)
    args = ap.parse_args()

    # Backward compatibility: MENTRA_MODEL still picks the DEFAULT model for
    # clients that do not select one. It is no longer the only way to choose,
    # and it no longer decides what the process is able to serve.
    default_model = args.default_model or active_model_id()
    if default_model not in LIVE_SERVABLE_MODELS:
        raise SystemExit(
            f"MENTRA_MODEL/--default-model={default_model!r} has no live runtime; "
            f"choose one of {LIVE_SERVABLE_MODELS} (default {DEFAULT_MODEL})")

    g2_checkpoint = resolve_g2_checkpoint(args.geowearnet_checkpoint)
    print(f"loading model registry (default={default_model}) ...")
    print(f"  speakernet    <- {args.model}")
    print(f"  geowearnet_g2 <- {g2_checkpoint or 'NOT CONFIGURED'}")

    registry = ModelRegistry(
        speaker_model_path=args.model,
        geowearnet_checkpoint=g2_checkpoint,
        default_model=default_model,
        audio_policy=args.audio_policy,
        speaker_threads=args.speaker_threads,
        geowearnet_threads=args.geowearnet_threads,
        context_s=args.context_s, hop_s=args.hop_s,
        speaker_enrollment=load_placeholder_enrollment(),
    )
    registry.load_all()
    for entry in registry.catalog()["models"]:
        status = "READY" if entry["ready"] else f"UNAVAILABLE ({entry['unavailableReason']})"
        print(f"  model {entry['id']:<14s} {entry['displayName']:<16s} {status} "
              f"requiresEnrollment={entry['requiresEnrollment']} "
              f"experimental={entry['experimental']}")
    if not registry.is_ready(default_model):
        raise SystemExit(
            f"default model {default_model!r} failed to load: "
            f"{registry.get(default_model).failure_reason}")
    g2 = registry.models.get("geowearnet_g2")
    if g2 is not None and g2.ready:
        print(f"GeoWearNet G2 checkpoint sha256 verified: {g2.checkpoint_sha256}")

    denoiser_factory = None
    if args.audio_policy in ("rnnoise_geowear_gate", "rnnoise_geowear_envelope"):
        from evaluation.agent_audio.denoise import Denoiser

        def denoiser_factory():  # noqa: F811 -- deliberate per-session factory
            return Denoiser(sr=16000)

    print(f"audio policy={args.audio_policy} (source separation supported: false)", flush=True)
    if args.audio_policy in ENVELOPE_POLICIES:
        print(f"gate envelope={args.gate_policy} preroll={args.gate_preroll_ms}ms "
              f"(real algorithmic delay, lookahead 0ms)", flush=True)

    asr_recognizer = None
    if not args.no_asr:
        print(f"loading streaming ASR from {ASR_MODEL_DIR} (CPU, {args.asr_threads} threads)")
        asr_recognizer = load_asr_recognizer(args.asr_threads)
        print("ASR loaded -- passthrough keeps continuous input; an opt-in gated audio "
              "policy routes the frontend output to ASR")

    # ---- G4 WS6: research capture -------------------------------------------
    # Entirely opt-in. Without --capture-dir the sink is None, sessions are
    # given no capture callbacks, and the live path is byte-for-byte unchanged.
    capture_sink = None
    if args.capture_dir:
        from mentra.capture.sink import CaptureSink

        capture_sink = CaptureSink(Path(args.capture_dir))
        print(f"research capture ENABLED -> {Path(args.capture_dir).resolve()} "
              f"(raw native-rate PCM + derived 16 kHz PCM + metadata + validation; "
              f"stays local, never uploaded)", flush=True)

    session_config = LiveSessionConfig(
        audio_policy=args.audio_policy,
        gate_policy_name=args.gate_policy,
        gate_preroll_ms=args.gate_preroll_ms,
        denoiser_factory=denoiser_factory,
        asr_recognizer=asr_recognizer,
        capture_sink=capture_sink,
        log=lambda message: print(message, flush=True),
    )
    factory = LiveSessionFactory(registry, session_config)

    receiver = MentraRemoteReceiver(args.listen, args.port, jitter_target_ms=20,
                                    jitter_max_ms=80,
                                    capabilities=registry.capabilities_for(default_model),
                                    session_factory=factory)
    deadline = time.monotonic() + args.bind_retry_s
    while True:
        try:
            print(f"MentraRemoteReceiver listening on {args.listen}:{args.port}")
            print(f"default model={default_model} "
                  f"catalog={[m['id'] for m in registry.catalog()['models']]}")
            print("waiting for a connection ...", flush=True)
            await receiver.serve(
                on_capture_meta=capture_sink.on_capture_meta if capture_sink is not None else None,
                on_capture_raw=capture_sink.on_capture_raw if capture_sink is not None else None,
            )
            return
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or time.monotonic() >= deadline:
                raise
            print(f"port {args.port} still held; retrying bind "
                  f"({deadline - time.monotonic():.0f}s left) ...", flush=True)
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
