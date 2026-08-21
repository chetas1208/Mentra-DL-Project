# Mentra Wearer-vs-Environment Speech Detection

Sprint: 2026-08-21 → 2026-09-04.

Goal: given continuous 16kHz/16-bit/mono PCM from Mentra Live's Bluetooth mic
(`onMicPcm`), classify each speech region as `WEARER` or `ENVIRONMENT`.
No transcription, no localization, no camera/IMU. Phone does all compute.

Locked constraints (2026-08-21):
- Training data = public/open corpora only (LibriSpeech, DNS5 Personalized,
  CHiME-9 ECHI, EasyCom). Real Mentra recordings are eval/calibration/test
  data only, never required for training. See `docs/DATA_COLLECTION.md`.
- Critical inference path = fully local, no file/cloud/network stage. No
  WAV round-trip, no WebSocket, no transcription in the decision path. WAV
  recording is a debug-only branch off the live stream.

Decision hierarchy (don't skip rungs):
1. Commercial off-the-shelf engine (Picovoice Eagle)
2. Open-source pretrained speaker verification (sherpa-onnx + CAM++/WeSpeaker/ECAPA)
3. Domain adaptation / fine-tuning
4. Lightweight custom speaker model
5. Target speaker extraction (WeSep / VoiceFilter-Lite style)
6. Distillation
7. Train from scratch

Status labels used everywhere in this repo: `VERIFIED`, `IMPLEMENTED`, `TESTED`,
`PARTIAL`, `BLOCKED`, `RESEARCH ONLY`, `NOT STARTED`. Never claim more.

## Layout

- `android/` — Kotlin/Compose app, Mentra Bluetooth SDK integration.
- `research/` — per-candidate spike code (eagle, sherpa_onnx, wespeaker, speechbrain, tse).
- `evaluation/` — offline harness: manifests, scoring scripts, metrics, results.
- `recordings/` — local-only labeled Mentra audio. Never committed (see `.gitignore`).
- `models/` — model weights/checksums where license permits, else download scripts.
- `docs/` — RESEARCH_REPORT, ARCHITECTURE, BENCHMARKS, DATA_COLLECTION, FINAL_RECOMMENDATION.

## Current status (checked 2026-08-21)

Dev environment: **no Bluetooth stack, no Mentra Live, no Android
phone/adb** — verified via `lsusb`/`bluetoothctl`/`adb devices`, all empty.
2x RTX 3090 + 20 CPU cores + 125GB RAM confirmed present (`nvidia-smi`).
So: hardware-dependent items (live PCM, live enrollment, Bluetooth
playback routing) are blocked in this session regardless of what's
connected elsewhere; GPU training and offline model work are fully
unblocked and underway. See `docs/TODO.md` for the full breakdown.

Offline wearer-verification baseline is running for real: 4 pretrained
sherpa-onnx speaker models benchmarked against real LibriSpeech audio
(8 speakers, full rotation). See `docs/RESEARCH_REPORT.md` and
`docs/MODEL_SELECTION.md` for measured results.
