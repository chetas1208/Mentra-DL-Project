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

## Product pivot: web app (2026-08-22)

Dropped the live-hardware-connection chase per explicit direction ("DROP
this idea of connecting. Delete related code too"). Deleted:
`android/.../mentra/MentraPcmSource.kt`, `scripts/mentra/check_desktop_audio.sh`,
`docs/MENTRA_SDK_NOTES.md`, `vendor/` (cloned SDK). Kept the transport/
protocol/consumer layer (`mentra/audio/*`, `server/audio/*`) — it's reused
as-is by the web frontend, unchanged in spirit from the original design
("same binary protocol regardless of source").

**New architecture**: browser mic (`getUserMedia` → `AudioWorklet` → PCM16)
speaks the same binary WebSocket protocol directly to the Python backend.
No native bridge, no laptop Bluetooth dependency for the browser-mic path.
Real Mentra hardware audio remains a separate, still-unresolved question —
this pivot sidesteps it rather than solving it, by using the browser as
the audio source instead of trying to bridge Mentra's mic_pcm through a
laptop.

**Live infrastructure, real and working (2026-08-22)**:
- Backend: `scripts/mentra/run_receiver.py`, HPC-local, `127.0.0.1:8765`.
- Tunnel: Cloudflare quick tunnel (account-less, ephemeral) —
  `cloudflared tunnel --url http://localhost:8765`. WebSocket handshake
  verified working through it end-to-end.
- Frontend: Nuxt 4 + Tailwind, `web/` directory, deployed to Vercel at
  `https://mentra-dl-project.vercel.app`. Instrument-panel design with a
  bipolar center-out meter as the signature element.
- GitHub: `github.com/chetas1208/Mentra-DL-Project` (private), both
  backend and frontend pushed.

**Known fragility**: both the receiver and the Cloudflare tunnel are
background processes tied to this session's lifecycle — the receiver died
once already mid-session for reasons outside anything the user or their
SSH command did. If the tunnel dies, the deployed Vercel app's backend URL
goes stale until both are restarted and the Vercel env var
(`NUXT_PUBLIC_BACKEND_WS_URL`) is updated to match the new random
`trycloudflare.com` subdomain (quick tunnels don't have stable URLs).

**Still real enrollment is a placeholder** (LibriSpeech clip, not a real
wearer) — disclosed directly in the UI, not hidden.

## Standing instruction

**Do not commit** until the user explicitly authorizes it (repeated across
multiple messages this session).

## Standing instruction: parallel processing from now onwards (2026-08-27)

**From this point forward, run independent GPU/CPU jobs concurrently
instead of sequentially by default**, not just for the GeoWearNet G2
ablation campaign that prompted this. Applies to any future training
campaign, sweep, or batch of independent experiments on this box.

Why this is safe/correct here specifically: the GeoWearNet/G2 models are
tiny (~29K params for the CRNN, sub-6M for SpeakerNet-derived models) —
`nvidia-smi` confirms GPU compute utilization stays near 0% even during
"active" training. The real bottleneck is CPU-side audio I/O/feature
dataloading, not GPU compute, so multiple training jobs can safely share
each of the 2x RTX 3090s with large memory headroom to spare. Concretely:
`training/geowearnet/mmcsg/campaign_ablations.py` now takes a
`--concurrency` flag (jobs per GPU, default raised from 1 to 3) with
`num_workers` per job reduced 6→3 to compensate.

**Caveats to re-check before blindly parallelizing a NEW workload**:
- Verify the same "GPU compute idle" signature via `nvidia-smi` first —
  don't assume every future model is this small.
- This shared host runs other users' jobs too (`uptime` load average was
  ~80 during this run) — parallelizing further than ~3 jobs/GPU risks
  diminishing or negative returns from CPU thrashing, not another
  speedup.
- When killing/restarting a scheduler to change concurrency, verify with
  `ps -ef`/`pstree` that old child processes actually died — a `pkill -f`
  pattern that doesn't match the exact `-m module.path` invocation can
  leave orphaned processes writing into the same log files as the new run
  (this happened once this session; caught and cleaned up via explicit
  PID kills, not assumed).

## GeoWearNet G1 -> G2 campaign (2026-08-27, live)

Single long-running background agent (autonomous, no routine check-ins by
design) took the G1 simulation campaign to completion, then self-initiated
a G2 campaign against the real MMCSG smart-glasses dataset (CC BY-NC 4.0,
license re-verification in progress — see `docs/geowearnet_data_license_state.md`)
without being re-briefed, per its own "don't stop at milestones" mandate.

**Verified real** (checked directly via `ps`/`nvidia-smi`, not just agent
self-report): G2 ablation campaign running now —
`training/geowearnet/mmcsg/campaign_ablations.py --run --concurrency 3`,
real `train_real.py` jobs across both GPUs (baseline_60m, random_channel,
response_augment, level_normalized, spectral_only, ctx100, ctx1000; ctx250/
500/680 queued), initialized from real G1 checkpoints under
`training/geowearnet/runs/geowearnet_e1_s1_arch_crnn_20260825_150233_83e5/`
etc. Own watcher polls `training/logs/ablation_watch.log` every 60s,
prints `ABLATION_CAMPAIGN_COMPLETE` when done.

**Agent-reported, not yet independently spot-checked by reading the raw
JSON**: strong zero-shot SIM->MMCSG transfer (~0.9+ AUROC on the one
predeclared official-dev look); real-data sample-efficiency curve
plateaus ~60min; multichannel spatial features give +0.05-0.06 AUROC over
best single channel (channel 2); `IDENTITY_SHORTCUT_SUSPECTED` flagged
honestly; one MMCSG recording (`1302664060426140_0001_3375_22000`) has a
genuine isolated SELF/OTHER RTTM label swap (documented, raw file
untouched).

**Product north star correction (2026-08-27, from user)**: GeoWearNet is
the control signal, not the end product. Real target: raw Mentra mic ->
[optional machinery denoise] -> GeoWearNet (wearer/environment activity,
no enrollment) -> non-overlap routing now, WearerSepNet (geometry-
conditioned target extraction) later for the overlap case -> wearer-
dominant PCM -> ASR/agent. Audio-only — no camera, no IMU in the always-on
path (camera stays an explicit agent tool-call). Eval axis that actually
matters: wearer transcription retention + bystander leakage rate +
command correctness, not raw AUROC. Source separation (WearerSepNet)
stays gated — spec/design work is fine, no GPU training until G2 lands a
checkpoint and this gets explicit go-ahead.

**Parallel P1 lane launched (2026-08-27)**: second background agent
building the product-facing evaluation/routing infra so the moment G2
picks a final checkpoint, its real value is immediately measurable — not
another ablation table. Scope: `evaluation/agent_audio/` harness (wearer
WER/TER, bystander leakage rate, false-command rate, overlap-conditioned
metrics), a no-training "GeoWear Gate v0" streaming router (hysteresis/
attack/release, NOT a separator), an oracle-gate upper bound using ground-
truth MMCSG labels, an optional RNNoise baseline for machinery noise, a
synthetic "autobody stress bench," the one-shot final-G2-selection
pipeline (Pareto selection -> frozen checkpoint -> single guarded official-
dev eval -> export -> backend flip -> report regen) with a dev-reuse
guard, campaign process hardening (PID files, stale-process
detection) after the earlier zombie-process incident, an
`known_annotation_anomalies.json` registry for the MMCSG label swap, and a
WearerSepNet *design spec only* (no training). Explicitly forbidden from
touching the running GPU jobs, final MMCSG dev set, frontend, or
committing anything.

Both agents are autonomous background processes on this box — check
`ps aux | grep -E "geowearnet|train_real|campaign_ablations"` and
`nvidia-smi` for real current state rather than trusting any cached
status (including this note) once time has passed.

## GeoWearNet G3 — campaign converged (2026-08-27, verified real)

GPU campaign finished for real (confirmed: `ps aux` shows no
`train_real`/`campaign_ablations` processes, both GPUs 0% util). Final
model selected and frozen (read-only, checksummed):
`training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt` — TCN,
132,678 params, sha256 `07c43c3d9e37...68ef` (verified by hand, matches).

**Key evidence-based decisions from `evaluation/geowearnet/g3/g3_final_summary.json`
and `docs/geowearnet_g3_report.md`** (agent-reported, spot-checked file
existence/hash/timestamps only, not every number re-derived):
- Identity causality: `IDENTITY_INFORMATION_PRESENT_BUT_NOT_CAUSAL` —
  decodable in internal reps but a controlled real-MMCSG audit (671,575
  frames, 38 recordings, train/val wearer-identity-disjoint) found no
  material AUROC/Brier/log-loss improvement from adding self-speaker
  identity after acoustic/noise controls.
- Product value: raw bystander leakage 62.09% -> predicted GeoWear Gate
  7.04% (wearer WER 28.27%, activity F1 79.06%). Oracle-vs-predicted gap
  quantifies real detector-improvement headroom separately from the
  overlap-is-unsolvable-by-gating ceiling.
- **WearerSepNet: correctly NOT started** — predeclared gate required
  overlap to cause >=50% of residual leakage; measured 6.90%. Muting
  overlap would cost 3.99pp wearer deletion (< 5pp threshold). This is a
  real evidence-based negative result, not unfinished work — matches
  [[project_mentra_product_north_star]]'s "separator only after G2 lands
  + explicit go-ahead" gate, and the evidence says not yet warranted.
- Live receiver (`mentra/audio/consumer.py`, `server/audio/frontend.py`,
  `scripts/mentra/run_receiver.py`) audited and fixed: was silently
  collapsing GeoWearNet's real wearer+environment dual scores into a
  binary result; now emits all 4 states (WEARER/ENVIRONMENT/OVERLAP/
  SILENCE) correctly, resets all state (consumer/RNNoise/ASR/transcript/
  playback-mode) at session boundaries, OVERLAP passes raw mixed PCM
  (never fabricates separation), capability payload now truthfully
  advertises `supportsSourceSeparation=false` + marks opt-in gate
  policies `experimental`. Live smoke-tested on a separate port
  (18768) without touching the existing production receiver (8765).
- CPU runtime: 2-thread GeoWear-gate update 8.3ms p50/13.4ms p95, RTF
  0.048; +RNNoise 11.4ms p50, RTF 0.181 — real-time on CPU with margin.
- Full test suite 225 passed (+38 web), `git diff --check` clean.

**Real remaining limit (external, not solvable locally): Mentra hardware
validation.** Everything above is MMCSG-wearable-proxy-validated, not
Mentra-device-validated. Next step needing the user specifically: a real
Mentra pilot recording session per `docs/geowearnet_capture_protocol.md`.

**No commit was made** (consistent with every campaign brief's rule) —
all of this is real, verified, uncommitted working-tree state.

## GeoWearNet G4 completed (2026-08-27) — real live-path bug found+fixed

Real finding, not just infra: the live receiver's actual first-word
retention was **19.74%**, not the 59.21% G3 reported (that number was from
the *offline* evaluator, never measured on the live path until G4). Root
cause: 200ms detector cadence, not gate logic (isolated via router-only vs
cadence-only vs full-path parity decomposition). Fixed via a predeclared-
rule-selected 150ms pre-roll -> 68.42% first-250ms retention, -1.13pp
leakage cost; 100ms cadence + 150ms pre-roll reaches 81.58%. New:
`server/audio/streaming_gate.py` (wraps the offline `GeoWearGate`, don't
reimplement), two new opt-in policies, binary default unchanged.

**Capture-path correction applied for real** (see
[[project_mentra_product_north_star]]'s "Architecture north star" section):
capture tool is `web/app/pages/capture.vue` (extends the existing real
browser->WebSocket->receiver chain), NOT a server-side Bluetooth/adb tool —
G4 self-corrected after an orchestrator mistake mid-campaign.

**Follow-up done directly by the orchestrator (not another big agent, per
explicit "stop autonomous coding, do the necessary prep" instruction)**:
- `mentra/audio/mentraos_source.py` — MentraOS Path A (`session.mic.onAudioChunk`,
  base64 PCM/LC3) + Path B (`mic_pcm` raw) decode adapter, targets the same
  `AudioFrame` contract the browser path already uses. LC3 refuses rather
  than silently mis-decoding; unreported/unmeasured sample rate is always
  tagged `assumed_format=True` rather than guessed silently. 11 tests,
  `tests/audio/test_mentraos_source.py`, all passing on synthetic payloads
  (no hardware needed for this part).
- `docs/geowearnet_capture_protocol.md` — added a frozen **G4-PILOT-V1**
  section: 5 people x 7-10min (not 20), exact per-condition duration table,
  mandatory shared-glasses swap sequence, AGC check procedure (run FIRST,
  ~20dB level-change comparison), zero-shot-only two runtime configs to
  test (200ms/0ms vs 100ms/150ms — no fresh sweep on 5 people), a 4-way
  minimal product comparison (not another 30-condition matrix), and the
  real command-script (with False Agent Command Rate as the headline
  product metric, distinct from raw token leakage).
- Recovery snapshot at `/home/923873155/mentra_snapshots/G4_PRE_HARDWARE_SNAPSHOT_<UTC-timestamp>/`
  (outside the repo, survives tree mangling) — full `git diff`, untracked-
  file backup (1898 files, excludes checkpoints/npz/runs — those stay
  hash-verified in place, not duplicated), critical-file hashes, 316-test
  pass snapshot. Make a fresh one before any future long autonomous run
  touches this dirty tree again.

**Explicit standing instruction from the user (2026-08-27): STOP training
models for now.** No G5 architecture search, no separator training, no
more MMCSG ablations, no identity work, no bigger model. Frozen checkpoint
(`g2_selected_07c43c3d9e37.pt`) stays as-is. The only real next step is
physical: a Mentra Live + phone + 5 people through the now-real capture
path, per G4-PILOT-V1 above. Software is ready; more autonomous coding
without hardware would be avoidance of the actual experiment, not
progress — do not launch another giant campaign to fill the wait.

## GeoWearNet G4 launched (2026-08-27) — real-Mentra-hardware gate

G4's headline question (does frozen G2 work on real Mentra Live audio)
**cannot be answered in this dev environment** — no physical Mentra
device, no paired phone, no Bluetooth/adb, verified twice (see
"Environment reality" above, still true). Agent briefed to do everything
achievable without hardware for real (live-routing envelope fix for the
known offline-evaluator-vs-live-200ms-router mismatch, first-word
retention/pre-roll tuning on internal data, live/offline parity
measurement, MentraOS audio-path desk research, capture-tool build+self-
test) and cleanly mark every hardware-dependent workstream
BLOCKED_NO_MENTRA_HARDWARE with an exact unblock spec, rather than
faking results on relabeled MMCSG/synthetic audio. Expected honest
verdict: `MENTRA_NOT_TESTED`. Real next action once this converges: the
user needs to physically supply a Mentra Live + phone + some pilot
capture minutes via the tool this campaign builds — everything else is
already staged to consume that the moment it exists.
