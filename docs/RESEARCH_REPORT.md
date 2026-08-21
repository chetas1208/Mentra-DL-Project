# Research Report

Status: NOT STARTED. This file is the deliverable required by spec section 49.
Fill in only with measured results, cited sources, or explicit
`NOT TESTED` / `PAPER REPORTED` / `VENDOR REPORTED` / `OUR MEASUREMENT` labels.

## Question 1 — Does an existing off-the-shelf solution work today?

Answer: NOT TESTED YET.

- Candidate: Picovoice Eagle
- Status: BLOCKED — no AccessKey, no Mentra PCM recordings yet.

## Question 2 — Existing models needing only an integration harness?

Answer: NOT TESTED YET.

- Candidates: sherpa-onnx (CAM++/WeSpeaker/ECAPA), SpeechBrain ECAPA-TDNN.

## Question 3 — Can existing cloud/large models be distilled?

Answer: NOT RESEARCHED YET. Gate: only relevant if Q1/Q2 fail (see escalation
gate, README decision hierarchy).

## Question 4 — If a new model must be trained, what's required?

Answer: N/A — no evidence yet that training is necessary. Do not fill in
speculatively (see project rule against inventing training projects).

## Comparison Table

| Solution | Tested | Mentra Live | Real-time | CPU | Enrollment | Overlap | Size | Accuracy | License | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Eagle | No | No | — | — | — | — | — | — | — | NOT TESTED |
| sherpa-onnx/CAM++ | No | No | — | — | — | — | — | — | — | NOT TESTED |
| WeSpeaker | No | No | — | — | — | — | — | — | — | NOT TESTED |
| SpeechBrain ECAPA | No | No | — | — | — | — | — | — | — | NOT TESTED |
| TSE (WeSep/VF-Lite) | No | No | — | — | — | — | — | — | — | NOT TESTED |

## Dataset survey (locked strategy, 2026-08-21 — RESEARCH ONLY, not yet verified/downloaded)

Training = public data only. Mentra recordings = eval/calibration/test only.
See `docs/ARCHITECTURE.md` and `docs/DATA_COLLECTION.md`.

| Dataset | Role | License note | Status |
|---|---|---|---|
| LibriSpeech | clean speech, mix-and-swap wearer/interferer roles | CC BY 4.0, ~1000h | NOT DOWNLOADED |
| VoxCeleb2 | speaker identity pretraining | original Oxford URLs reported unavailable — not needed if using pretrained WeSpeaker checkpoint | NOT USED (superseded by pretrained checkpoint) |
| WeSpeaker pretrained (CAM++/ECAPA/ResNet34, ONNX) | ready-made speaker encoder, no training needed for prototype | check WeSpeaker repo license per model | NOT DOWNLOADED |
| DNS5 Personalized | enrollment + interferer + noise + RIR mechanics | check MS DNS Challenge license terms before commercial use | NOT DOWNLOADED |
| CHiME-9 ECHI | real smart-glasses (Meta Aria) acoustics, 18h/118 speakers | DUA-gated, academic/research only — commercial needs org contact | NOT DOWNLOADED |
| EasyCom | head-worn AR glasses, 6ch, 5h18m | CC BY-NC 4.0 — research only | NOT DOWNLOADED |

Priority for prototype (Days 1-2): pretrained WeSpeaker/sherpa-onnx checkpoint,
zero training. Fine-tuning corpora only enter play past the escalation gate.

## SYSTEM VERDICT

BLOCKED — no hardware/API keys connected to dev environment yet.
