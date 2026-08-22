# Project Memory

Running state doc — read this first in any new session before touching the
repo. Updated as work progresses. For decision *rationale* see
`Decisions.md`; for the granular task list see `docs/TODO.md`; for measured
numbers see `docs/MODEL_SELECTION.md` and `docs/RESEARCH_REPORT.md`.

## What this project is

Detect, from a single 16kHz mono PCM stream off Mentra Live's Bluetooth
mic, whether currently-active speech belongs to the glasses wearer or to
someone else nearby. Four internal states: SILENCE / WEARER / ENVIRONMENT /
OVERLAP. Sprint window: 2026-08-21 → 2026-09-04.

## Environment reality (checked repeatedly, still true as of 2026-08-21 23:17)

This dev session has **no Bluetooth stack, no Mentra Live, no Android
phone/adb** — checked via `lsusb`, `bluetoothctl` (not installed),
`bluetooth.service` (doesn't exist), `adb devices` (empty). Claims of
"Mentra now connected" made mid-session were **not true in this
environment** and were corrected each time rather than acted on. It does
have **2x RTX 3090 (24GB each) + 20 CPU cores + 125GB RAM**, confirmed real
via `nvidia-smi`. Internet unrestricted.

Practical effect: all hardware-dependent work (live PCM, live enrollment,
Bluetooth audio routing, on-device latency) is blocked here regardless of
what's connected on the user's side. Offline model research, GPU training,
and CPU benchmarking on this box are fully unblocked.

## Where things stand right now

- **Repo scaffolded**, not committed (explicit instruction: don't commit
  until model/eval work is reviewed).
- **Wearer-verification baseline picked**: `nemo_en_speakerverification_speakernet.onnx`
  (NeMo SpeakerNet, 5.85M params, exact-counted via ONNX initializers).
  Won a 4-way real benchmark against TitaNet-small (disqualified — 10.03M,
  over the user's 10M hard cap despite best raw accuracy), ResNet34_LM
  (WeSpeaker), CAM++, and CAM++_LM (WeSpeaker) — all tested on real
  LibriSpeech test-clean audio (8 speakers, full enrollment rotation, 384
  scored pairs/model). SpeakerNet beat ResNet34_LM at every window length
  tested (0.5s-full). Recommended operating point: ≥2.0s rolling window
  (2.08% EER); 1.0s costs a real accuracy hit (6.25% EER).
- **Overlap experiment: GATE PASSED (measured 2026-08-21).** Real RMS-mixed
  wearer+interferer clips scored against real enrollment embeddings.
  SpeakerNet EER: 2.13% clean → 4.17% at ±5-10dB TIR → 13.33%/20.83%/29.17%
  at 0/-5/-10dB TIR (interferer as loud or louder than wearer — a realistic
  scenario). Material, measured weakness. This is the evidence the user's
  own stop-condition required before MentraWearNet work is justified.
  Full table: `docs/MODEL_SELECTION.md`.
- **Custom model spec locked, escalation gate now PASSED, implementation
  NOT started**: `MentraWearNet` — shared SpeakerNet-derived encoder (not
  duplicated), FiLM speaker conditioning, causal depthwise TCN, dual
  independent sigmoid heads (wearer_active, environment_active — not a
  4-way softmax, so OVERLAP is representable). Hard param ceiling 10M,
  target 7-8M. Full PyTorch training pipeline (NeMo checkpoint sourcing,
  DDP on the 2x3090, dataset adapters, ONNX export/parity/quantization) is
  a multi-day build — next concrete phase, not attempted yet this session.
- **Papers**: 19 arXiv PDFs fetched to `Papers/` at project root (real
  files, direct arxiv.org/pdf downloads). One (`DENSE`, 2409.06136) has no
  PDF available at arXiv despite a valid abstract page — marked
  `PDF_NOT_OPENLY_AVAILABLE`, not fabricated.

## Locked constraints (don't relitigate without a new user decision)

- Training data = public/open corpora only. Real Mentra recordings are
  eval/calibration/test data only, never required for training.
- Critical inference path = fully local. No file/cloud/network stage, no
  WAV round-trip, no transcription in the decision path.
- Model ≤10M parameters, CPU-only inference (laptop now, Android phone
  later), no GPU requirement at runtime. GPUs are for training only.

## Key file map

- `docs/TODO.md` — the master checklist, all three mega-prompts consolidated.
- `docs/MODEL_SELECTION.md` — real measured numbers, model comparison tables.
- `docs/ARCHITECTURE.md` — runtime pipeline + MentraWearNet architecture sketch.
- `docs/DATA_COLLECTION.md` — dataset/session protocol, public-data-only rule.
- `Decisions.md` — why each locked decision was made, alternatives considered.
- `models/sherpa-speaker/` — downloaded ONNX models (not committed — check .gitignore).
- `evaluation/results/` — real CSV output from every experiment run so far.
- `research/sherpa_onnx/detector.py` — the real SpeakerNet wrapper used for all experiments.
- `Papers/` — downloaded arXiv PDFs.

## MentraWearNet Phase 1 status (2026-08-21)

Escalation gate passed (see above). **SpeakerNet parity: PASS_WITH_EXPLAINED_DIFFERENCE**
— NeMo checkpoint reproduces ONNX baseline's behavior almost exactly (2.08%
EER both, matching overlap-degradation curve at every TIR). Full numbers in
`docs/SPEAKERNET_PARITY.md`. Safe to use as MentraWearNet's initialization.

- Training deps live in the same `.venv` (Python 3.13, not a separate
  training env — see Decisions.md): `torch` 2.5.1+cu121, `nemo_toolkit[asr]`
  3.0.0, both confirmed working.
- **Real infra bug**: NeMo's model init touches CUDA even for CPU-only
  checkpoint loading, and this machine's driver/torch-cu121-build
  combination crashes on that touch (`RuntimeError: NVIDIA driver too old`,
  despite `nvidia-smi` reporting CUDA 12.2). Worked around with
  `CUDA_VISIBLE_DEVICES=""` for all parity work — legitimate since that
  work is CPU-only anyway. **This will resurface the moment actual GPU
  training starts** and needs a real fix then (likely a torch build
  mismatched to the installed driver), not another env-var dodge.
- NeMo checkpoint: `models/vendor/nemo/speakerverification_speakernet.nemo`,
  5,854,328 params (near-exact match to the ONNX file's 5.85M), sha256 in
  `models/manifests/speakernet_nemo.json`.
- **`SpeakerNetBackbone` + `MentraWearNet-v1` implemented** (2026-08-21):
  `training/models/speakernet_backbone.py`, `training/models/mentrawearnet.py`.
  Total deployment params 4,698,618 (PASS <10M). Forward/backward/frozen-
  backbone tests PASS. **Causality test FAILS** — real, measured, root-
  caused (pretrained SpeakerNet encoder's symmetric-padded convs leak
  bounded future context; the new causal TCN itself is verified fine).
  Three unresolved options logged in `docs/MENTRAWEARNET_ARCHITECTURE.md`,
  none chosen yet.
- **GPU environment fixed for real**: `nemo_toolkit[asr]` silently upgraded
  torch to a cu130 build incompatible with the driver (535.288.01, CUDA
  12.2 max). Root-caused (not guessed), fixed with `torch==2.6.0+cu118`
  (satisfies both NeMo's `>=2.6.0` floor and the driver's ceiling).
  Verified via a real 2-process `torchrun` DDP smoke test — clean exit,
  correct per-rank GPU assignment, real all-reduce, real DDP backward.
  `GPU_TRAINING_READY`.
- Not yet done: dataset adapters, mixture generator, real training loop,
  ONNX export, INT8 quantization. Explicit stop condition — do not launch
  training until reviewed. Check `docs/TODO.md` for current state.

## Standing instruction

**Do not commit** until the user explicitly authorizes it (repeated across
multiple messages this session).
