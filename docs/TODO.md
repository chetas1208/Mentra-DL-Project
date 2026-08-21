# Master TODO

Consolidates all three master-prompt rounds (Day-1 continuation, model
research phase, production ASR/TTS/Bluetooth phase). Every item traceable
to its source section so nothing gets silently dropped. Status legend:

`[x]` done and verified · `[~]` in progress · `[ ]` not started ·
`[BLOCKED: reason]` cannot proceed here.

## Environment reality (checked 2026-08-21, do not re-litigate without re-checking)

- [x] Bluetooth: **NONE**. No `bluetoothctl`, no `bluetooth.service`, no BT adapter on this box.
- [x] Mentra Live / Android phone: **NOT connected here**. No `adb`, no USB Mentra device (lsusb checked).
- [x] GPU: **2x RTX 3090, 24GB each, driver 535.288, CUDA 12.2** — real, confirmed via `nvidia-smi`.
- [x] CPU/RAM: 20 cores, 125GB RAM, 540GB free disk.
- [x] Internet: unrestricted (pypi, github, openslr all reachable).

Implication: everything requiring live Mentra PCM, live enrollment, Bluetooth
media routing, or Android on-device testing is **hardware-blocked in this
session** regardless of what's connected on your side. GPU training work is
fully unblocked. This file will be updated the moment that changes.

---

## Phase 1 — Wearer-detection baseline (offline, public data) — Day-1 continuation prompt

- [x] venv + sherpa-onnx + numpy/soundfile/scipy installed and verified importable.
- [x] Real pretrained models downloaded from `k2-fsa/sherpa-onnx` release `speaker-recongition-models`: CAM++, CAM++_LM, ResNet34_LM, TitaNet-small (NeMo). Recorded in `models/README.md` — **license column still needs filling in** (§48/§84 of both prompts).
- [x] LibriSpeech test-clean (OpenSLR-12, CC BY 4.0) downloaded, extracted, 8 speakers selected, converted to 16kHz mono PCM16 WAV (`scripts/prepare_day1_public_data.py`).
- [x] Source-agnostic `PcmSource`/`WavPcmSource` abstraction (`research/common/pcm_source.py`) — §2 of Day-1 prompt.
- [x] `SherpaOnnxWearerDetector` (`research/sherpa_onnx/detector.py`) wrapping real sherpa-onnx API, verified against official example script.
- [x] Rotation experiment (every speaker as wearer once) run for 4 models — 384 real scored rows each: CAM++, CAM++_LM, ResNet34_LM, TitaNet-small.
- [x] Real metrics computed (ROC-AUC, EER, balanced accuracy, precision/recall/F1) — `scripts/compute_day1_metrics.py`.
- [x] **Constraint added (2026-08-21): ≤10M params, CPU-only, continuous.** Exact param counts computed via `scripts/count_onnx_params.py` (real ONNX initializer sums, not file-size estimates). Disqualifies TitaNet-small (10.03M). New winner: `nemo_en_speakerverification_speakernet.onnx` (5.85M params, EER 2.08%, AUC 0.9994 — matches TitaNet-small's accuracy at 58% the size). See `docs/MODEL_SELECTION.md`.
- [x] Duration curve for TitaNet-small (now disqualified, kept as reference: EER flat ~2% from 1.0s+, collapses to 25% at 0.5s).
- [x] Duration curve for SpeakerNet — wins at every window length (2.08% EER @2.0s, 6.25% @1.0s).
- [x] Duration curve for ResNet34_LM — loses to SpeakerNet at every window (8.97% EER full-length vs 2.13%). Ruled out.
- [x] **Final in-budget pick: SpeakerNet (5.85M params).** Target ≥2.0s rolling window. See `docs/MODEL_SELECTION.md`.
- [ ] Same experiment for CAM++/CAM++_LM duration curve (low priority — already losing badly at full length, param budget doesn't rescue it).
- [ ] Noise experiment (20/10/5/0 dB SNR mixtures) — §13 of Day-1 prompt. Not started.
- [x] **Overlap/TIR experiment — GATE PASSED.** SpeakerNet EER degrades from 2.13% (clean) to 13.3%/20.8%/29.2% at TIR 0/-5/-10dB (realistic "someone talking at the wearer" scenarios). Real, material weakness — justifies starting MentraWearNet per the user's own stop-condition. See `docs/MODEL_SELECTION.md`.
- [ ] Streaming simulator (`research/streaming_sim/`) feeding chunked audio instead of whole files — §17. Not started.
- [ ] Multiple window/hop configs (500ms/100ms, 1s/200ms, 2s/250ms) on the streaming simulator — §18. Not started.
- [ ] Wire `evaluation/scripts/evaluate.py` to a real backend (currently structural only) — §26. Superseded in practice by `run_day1_experiment.py`; should still reconcile the two.
- [ ] Unit tests: PcmRingBuffer, cosine similarity, normalization, manifest parsing, metric computation, threshold behavior — §27. Not started.
- [ ] Picovoice Eagle: still `BLOCKED: ACCESSKEY`. Not chased further per explicit instruction to not block on it.

## Phase 2 — Research paper archive — model-research prompt §5-§26, §72-§86

- [ ] `research/papers/index.md` + `research/papers/bib/mentra_audio.bib` with real metadata (title/authors/year/arXiv/DOI/license/local PDF) for: CAM++, ECAPA-TDNN, WeSpeaker, TitaNet, TS-VAD, Continuous TSE, VoiceFilter-Lite, SpeakerBeam, SpeakerBeam-SS, DENSE, USEF-TSE, Listen to Extract, WeSep 2024, WeSep 2026, Target Conversation Extraction, DNS Challenge 2023, TEA-PSE 3.0, NPU-Elevoc PSE, CHiME smart-glasses papers, EasyCom paper, ESPnet-SPK. **Not started** — 20 papers, needs its own pass; will fetch real arXiv metadata, not fabricate.
- [ ] `research/papers/notes/<paper>.md` per-paper breakdown (problem/architecture/conditioning/causal/size/latency/dataset/losses/results/code/Mentra relevance) — §72. Not started, depends on above.
- [ ] `research/RESEARCH_MATRIX.md` cross-paper comparison table — §74. Not started.
- [ ] Clone official repos: Mentra Bluetooth SDK Starter Kit, sherpa-onnx (have via pip, not cloned), WeSpeaker, WeSep — §28. Not started.
- [ ] `scripts/download_research_assets.sh` — §27. Not started.
- [ ] `research/assets_manifest.json` with checksums — §83. Not started.

## Phase 3 — Custom model (MentraWearNet) — gate PASSED, Phase-1 (parity) in progress

- [x] **Gate check PASSED** (2026-08-21) — overlap experiment showed real SpeakerNet degradation (2.13%→13.3%/20.8%/29.2% EER at TIR 0/-5/-10dB). See `docs/MODEL_SELECTION.md`, `Decisions.md`.
- [x] **SpeakerNet parity — PASS_WITH_EXPLAINED_DIFFERENCE (measured 2026-08-21).** torch 2.5.1+cu121 + nemo_toolkit[asr] 3.0.0 installed in existing `.venv`. NeMo checkpoint downloaded (5,854,328 params, sha256 recorded, `models/manifests/speakernet_nemo.json`). Level A (params/dim): exact match. Level B (raw embedding cosine): ~0.92 mean, not ~1.0 — unexplained numerically but doesn't affect behavior. Level B (pairwise geometry Pearson): 0.9816. Level C (rotation EER): 2.08% both ONNX and NeMo, exact match. Level C (overlap EER curve): matches within small-sample noise at every TIR (4.17/4.17, 4.17/4.17, 12.5/13.3, 25.0/20.8, 29.2/29.2). Full writeup: `docs/SPEAKERNET_PARITY.md`.
- [x] Real infra bug hit and fixed: NeMo's `ModelPT.__init__` crashes touching CUDA even for CPU-only checkpoint loading, because the driver reports a CUDA runtime version torch's cu121 build doesn't like. Fixed with `CUDA_VISIBLE_DEVICES=""` for all parity work (legitimate — parity/CPU-inference doesn't need GPU). **This same issue will block actual GPU training later and needs a real fix then** (driver/torch-build mismatch), not just an env-var workaround.
- [ ] Duration-curve (0.5/1.0/2.0s) parity — not run, Level C full-length + overlap results already gave enough evidence; can backfill if a discrepancy surfaces later.
- [ ] SpeakerNetBackbone wrapper (frame-level features before pooling) — next concrete step, not started.
- [ ] MentraWearNet module skeleton, param-count assertion (<10M), causality test, training smoke test — blocked on backbone extraction.
- [ ] Architecture doc: dual-sigmoid head (wearer_active, environment_active), FiLM conditioning first, cross-attention (USEF-TSE-style) and dynamic embedding (DENSE-style) as ablations — already sketched in `docs/ARCHITECTURE.md`, needs its own doc if this phase starts.
- [ ] Training data pipeline: LibriSpeech role-swapped mixtures + DNS5 Personalized + CHiME-9 ECHI (research license) + EasyCom (CC BY-NC, research only) — public data only, per locked rule in `docs/DATA_COLLECTION.md`.
- [ ] GPU training script (DDP across the 2x3090, bf16/fp16) — infrastructure only, not started, correctly gated.
- [ ] ONNX export + numerical-equivalence check + INT8 quantization with before/after FAR/FRR — not started, gated.

## Phase 4 — Production real-time pipeline — production-system prompt

**Everything in this phase needs either live Mentra hardware or is pure software architecture that can be built/tested against `WavReplayAudioSource` first.** Splitting accordingly.

### 4a. Can build/test now (no hardware needed)
- [ ] `AudioSource` / `PcmFrame` interface + `WavReplayAudioSource`, `SyntheticTestAudioSource` — §7.
- [ ] `WearerDetector` Kotlin interface already exists (`android/.../inference/WearerDetector.kt`); extend with `wearerProbability`/`environmentProbability`/four-state output per §15 of production prompt (currently has `wearerScore`/`confidence` — needs revision).
- [ ] Bounded `PcmRingBuffer` + `StreamFramer` — ring buffer exists (`android/.../audio/PcmRingBuffer.kt`), framer does not.
- [ ] Threading/queue design with backpressure metrics (queue depth, backlog ms, drop counts) — design doc only until there's a real pipeline to attach it to.
- [ ] `AsrEngine` / `TtsEngine` interfaces — not started.
- [ ] `IdentityEnglishTransformer` text-pipeline slot — not started, trivial once ASR/TTS exist.
- [ ] Transcript stability / commit-layer logic (stable prefix vs unstable suffix) — design only, not started.
- [ ] Config schema (`audio:`/`wearer:`/`asr:`/`tts:`/`runtime:` YAML) — not started.
- [ ] Model manifest format under `models/manifests/` — not started.
- [ ] ASR model candidates via sherpa-onnx: download 2+ CPU-appropriate streaming models, benchmark WER/RTF/latency on **public** test data (not Mentra) — can start now.
- [ ] TTS model candidates via sherpa-onnx Piper/VITS (e.g. `en_US-amy-medium`): download, benchmark RTF/time-to-first-audio — can start now.
- [ ] CPU-only enforcement + `CUDA_VISIBLE_DEVICES=""` test mode — can build now, verify on this box's CPU.

### 4b. Hardware-blocked — `BLOCKED: no Mentra/Bluetooth/Android in this session`
- [ ] `MentraAudioSource` live implementation and PCM callback verification (rate/cadence/drop measurement).
- [ ] Live wearer enrollment through actual Mentra mic.
- [ ] Live WEARER/ENVIRONMENT score timeline from real speech.
- [ ] `SystemBluetoothAudioSink` — OS Bluetooth media route playback to Mentra speaker (laptop pairing test).
- [ ] Feedback/echo test and suppression (`setOwnAppAudioPlaying` or equivalent) — needs real playback+mic loop to observe.
- [ ] End-to-end latency measurement (speech-in to synthesized-audio-out through real Mentra).
- [ ] Offline acceptance test (disconnect internet, run full pipeline) — needs the above first.
- [ ] Android port of any of this — needs a phone.

### 4c. Structural/process items
- [ ] Monorepo restructure to `apps/ / runtime/ / training/ / evaluation/ / models/ / research/ / docs/ / scripts/` — **holding off**: premature to restructure before Phase 1's overlap result decides whether Phase 3/4 even happen this sprint. Current flat structure (`android/`, `research/`, `evaluation/`, `docs/`) already separates concerns adequately for where the project actually is.
- [ ] `docs/SYSTEM_DESIGN.md`, `RUNTIME_ARCHITECTURE.md`, `LATENCY_REPORT.md` — not started, most inputs (real latency numbers) don't exist yet.
- [ ] `docs/MODEL_SELECTION.md` — **next concrete deliverable**, can be written now with the real Phase-1 numbers already in hand.

---

## Immediate next 3 actions (my recommendation, not yet executed)

1. Finish duration curves (in progress) + run the overlap/interfering-speaker experiment — this is the single result that determines whether Phase 3/4's TS-VAD work is justified at all. Everything else is secondary until this is measured.
2. Write `docs/MODEL_SELECTION.md` from real Phase-1 numbers (TitaNet-small currently winning by a wide margin — EER 2.1% vs ResNet34_LM 8.8% vs CAM++ family 31-38%).
3. Correct the two hardware-status errors that arrived in prompts this session (Mentra "connected", implicitly-assumed Bluetooth) in `README.md`/`ARCHITECTURE.md` so the docs don't silently inherit a false claim.

Still not committing (per explicit instruction) — this file included.
