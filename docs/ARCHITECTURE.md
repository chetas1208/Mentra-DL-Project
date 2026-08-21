# Architecture

Status: LOCKED (design decision, 2026-08-21) / NOT STARTED (implementation).

## Two hard rules (locked)

1. **No file/cloud/network stage in the critical inference path.** No WAV
   round-trip, no WebSocket to a server, no transcription step during live
   classification. WAV recording exists only as a debug sink branching off
   the live stream, never in the decision path.
2. **Training data is public/open corpora only.** Real Mentra recordings are
   evaluation/calibration/debugging data — never required training data. See
   `docs/DATA_COLLECTION.md`.

## Runtime pipeline (locked)

```
Mentra Live
      |
      | BLE
      v
PCM16 @ 16kHz mono  (onMicPcm — SDK also exposes LC3 frames, PCM used first
      |                to avoid a second decoder in the inference path)
      v
lock-free ring buffer (audio/PcmRingBuffer)
      |
      v
Silero VAD (cheap gate — skip expensive path during silence)
      |
      v
rolling speech window (500ms-1s target, benchmark 250ms-2s to find
      |                 shortest window that meets accuracy bar)
      v
speaker encoder (CAM++ / ECAPA512 / ResNet34 via sherpa-onnx or WeSpeaker
      |            ONNX — pretrained, not trained by us for prototype)
      v
current_embedding                    wearer_embedding
      |                              (computed once at enrollment,
      |                               cached in RAM/local storage —
      |                               never recomputed per frame)
      +---------------+----------------------+
                       v
              cosine similarity
                       v
              score calibration
                       v
              hysteresis (T_high enter / T_low remain)
                       v
      +----------------+----------------+----------------+
      v                v                v                v
   WEARER         ENVIRONMENT        SILENCE          OVERLAP (4th state,
                                                        research-only until
                                                        proven necessary)
```

## State machine (revised)

Four states, not three: `SILENCE`, `WEARER`, `ENVIRONMENT`, `OVERLAP`
(`UNCERTAIN` dropped in favor of calibrated hysteresis handling ambiguity).
Overlap is real (wearer + environment speaking simultaneously) — don't force
it into a binary choice. Public API can collapse `OVERLAP` per product need,
but the internal model should support two independent sigmoid heads
(`wearer_active`, `other_active`) rather than one softmax over 2 classes,
if/when we go past pretrained-embedding verification (see custom-model
fallback below).

## Interfaces (implemented, unverified against real hardware)

- `WearerDetector` — `android/.../inference/WearerDetector.kt`
- `VoiceActivityDetector` — `android/.../audio/VoiceActivityDetector.kt`
- `PcmRingBuffer` — `android/.../audio/PcmRingBuffer.kt`

Backends to implement, in priority order (decision hierarchy, no skipping
rungs without measured evidence): `SherpaWearerDetector` (CAM++/WeSpeaker,
pretrained, first target), `EagleWearerDetector` (commercial comparison),
`CustomOnnxWearerDetector` (only if verification fails — see fallback).
None implemented yet.

## Enrollment vs training (two different things — don't conflate)

```
TRAINING (offline, once, public data)         ENROLLMENT (runtime, per user)
LibriSpeech + DNS5 Personalized                 wearer speaks 5-30s
  + CHiME-9 ECHI + EasyCom (research)             through Mentra mic
        |                                              |
        v                                              v
  generic speaker encoder                     wearer_embedding (local only,
  (pretrained WeSpeaker/sherpa-onnx              on-device, never uploaded)
   first — no training needed for prototype)
```

## Custom-model fallback (only past the escalation gate)

If pretrained speaker verification fails (overlap, domain mismatch, latency),
next step is a tiny causal wearer-activity model conditioned on the wearer
embedding (FiLM/cross-attention), not generic source separation first:

```
wearer enrollment -> speaker encoder -> E
                                         |
Mentra audio -> log-mel -> tiny causal net (conditioned on E)
                                         |
                              two sigmoid heads
                            wearer_active | other_active
```

Training data for this (if reached): public corpora only, per the locked
rule above — LibriSpeech for mix-and-swap identity roles, DNS5 Personalized
for enrollment/interferer/noise mechanics, CHiME-9 ECHI / EasyCom for
glasses-realistic acoustics (research license only — see LICENSES.md before
any commercial path uses these).
