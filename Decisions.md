# Decisions Log

Dated, reasoned record of locked project decisions — why, not just what.
For current state see `MEMORY.md`; for the task list see `docs/TODO.md`.

## 2026-08-21 — Training data: public corpora only, Mentra data = eval/calibration/test only

**Decision**: no proprietary Mentra recording may be required training data.
Training draws from LibriSpeech, DNS5 Personalized, CHiME-9 ECHI (research
license), EasyCom (CC BY-NC, research license).

**Why**: user's explicit correction — earlier draft implicitly assumed
Mentra recordings would be collected and used for training. Locked
specifically so a commercial deployment path isn't blocked later by having
trained on data with no clear collection/consent/licensing story.

**Alternatives considered**: collecting a large Mentra-specific dataset
first. Rejected — slower, and the whole point of picking a
pretrained-embedding-based approach is to avoid needing that.

## 2026-08-21 — Critical inference path: fully local, no file/cloud/network stage

**Decision**: no WAV round-trip, no WebSocket, no transcription step in the
live decision path. WAV recording exists only as a debug tap off the live
stream.

**Why**: user's explicit correction to an earlier draft. Matches the
product requirement (near-real-time, on-device) and avoids designing
something that can't actually ship on a phone.

## 2026-08-21 — Decision hierarchy: don't skip rungs

**Decision**: commercial off-the-shelf → open pretrained speaker
verification → domain adaptation → lightweight custom model → target
speaker extraction → distillation → train from scratch. Escalate only on
measured evidence the current rung fails.

**Why**: user's standing instruction across every prompt this session:
"find the simplest existing solution," "do not train a new model simply
because training is interesting," explicit stop-conditions/escalation
gates repeated in every subsequent mega-prompt. This is the single most
consistent instruction across the whole session — treat any future prompt
that seems to skip a rung as needing the same measured-evidence gate
applied before acting, not as an automatic override.

## 2026-08-21 — Environment reality: no Bluetooth/Mentra/Android here; GPUs confirmed real

**Decision**: don't act on hardware-connection claims without verifying
first (`lsusb`, `bluetoothctl`, `adb devices`, `nvidia-smi`). Corrected
"Mentra now connected" claims twice this session — each time verified false
in this specific dev environment.

**Why**: repeated instructions across every prompt in this session to never
fabricate a hardware-dependent result ("no fake completion," "never say
works on Mentra unless tested"). A claim of hardware being connected is
exactly the kind of thing that needs verifying, not assuming — especially
since it changes which phase of work is unblocked.

**What this doesn't mean**: it doesn't mean distrust everything the user
says. The GPU claim (2x RTX 3090) was checked the same way and turned out
to be true — verification cuts both ways, it isn't presumption of error.

## 2026-08-21 — Baseline model: SpeakerNet (NeMo), not CAM++/ResNet34/TitaNet

**Decision**: `nemo_en_speakerverification_speakernet.onnx` (5.85M params)
is the wearer-verification baseline and the initialization point for any
future custom model.

**Why**: measured, not assumed. Real 4-way rotation experiment on
LibriSpeech test-clean (8 speakers, 384 scored pairs/model) showed
TitaNet-small had the best raw accuracy (EER 2.08%) but at 10.03M params —
over the user's later-added ≤10M hard cap. Among in-budget candidates,
SpeakerNet (5.85M) matched TitaNet-small's accuracy almost exactly and beat
ResNet34_LM and CAM++/CAM++_LM at every window length tested (0.5s-full).
This directly contradicted the prior expectation that CAM++ would win on
efficiency/accuracy tradeoff (per the CAM++ paper's design intent) — real
measurement overrode a reasonable-sounding prior.

**Alternatives rejected**: TitaNet-small (over param budget once that
constraint was added), CAM++/CAM++_LM (weak on this test, EER 31-38%),
ResNet34_LM (beaten at every duration by SpeakerNet despite similar size).

## 2026-08-21 — Hard parameter budget: ≤10M, target 7-8M for any custom model

**Decision**: any custom model (MentraWearNet) must stay under 10M
parameters, with an automated test that fails the build if it doesn't.
Target design budget 6.8-8.0M.

**Why**: user's explicit requirement — "max 10 mill parameters,"
"lightweight, run on CPU relentlessly." This single constraint already
disqualified the best-measured model (TitaNet-small) and changed the
recommendation — proof the constraint is load-bearing, not decorative.

## 2026-08-21 — MentraWearNet architecture locked, but implementation gated on overlap evidence

**Decision**: if a custom model is built, it's `MentraWearNet` — one shared
SpeakerNet-derived encoder (not duplicated), FiLM speaker conditioning,
explicit cosine-similarity feature, causal depthwise TCN, two independent
sigmoid heads (wearer_active, environment_active — not a 4-way softmax).
Full PyTorch training pipeline (NeMo checkpoint sourcing, DDP training on
the 2x3090, dataset adapters, ONNX export/parity/INT8) is specced but
**not started**.

**Why not started yet**: the user's own prompt contains its stop condition
("MentraWearNet is successful only if it materially improves at least one
major weakness... If custom model is worse: do NOT ship it merely because
it is custom") and escalation gate ("Do NOT build a separator unless
MentraWearNet fails materially under overlap" — same logic applies one
level up: do NOT build MentraWearNet unless SpeakerNet fails materially
under overlap). No overlap measurement existed yet when this architecture
was specced. Building a multi-day training pipeline before that measurement
exists would violate the user's own repeatedly-stated principle
("measurements decide the product," not enthusiasm for a bigger build).
Overlap experiment is running now (`scripts/run_overlap_experiment.py`) —
its result is what unblocks or further defers this.

## 2026-08-21 — Escalation gate PASSED: proceed from SpeakerNet verification baseline to custom MentraWearNet TS-VAD architecture

**Decision**: build MentraWearNet (custom target-speaker activity model).

**Evidence** (measured, `evaluation/results/overlap_speakernet.csv`):
SpeakerNet clean EER 2.13%; TIR 0dB → 13.3%; TIR -5dB → 20.8%; TIR -10dB →
29.2%.

**Reason**: overlapping environmental speech at equal or higher volume than
the wearer is an ordinary product condition, not an edge case. A 6-14x EER
degradation under that ordinary condition is a material weakness per the
user's own stop-condition ("MentraWearNet is successful only if it
materially improves at least one major weakness... better overlap
detection"). This satisfies the gate that was explicitly left open in the
prior SpeakerNet-selection decision above.

**Immediate next step is NOT training**: per the user's explicit
instruction, the mandatory next milestone is proving the trainable
PyTorch/NeMo SpeakerNet reproduces the ONNX baseline's behavior closely
enough to trust as MentraWearNet's initialization (embedding parity,
rotation/duration/overlap parity) — building custom heads on an unverified
backbone would make any later result uninterpretable.

## 2026-08-21 — Training environment: same venv as inference, not a separate one

**Decision**: use the existing `.venv` (Python 3.13) for both sherpa-onnx
CPU inference work and PyTorch/NeMo training work, rather than the
separate training-only environment the spec called for.

**Why**: attempted to create a Python 3.12 venv (thinking it safer for NeMo
compatibility) but `python3.12-venv` isn't installed system-wide and
there's no passwordless sudo to install it. Tested whether PyTorch has
Python 3.13 wheels instead — it does (`torch==2.5.1+cu121`, confirmed
`cuda.is_available()==True`, 2 GPUs visible) — so the blocker turned out to
be avoidable rather than fundamental. Kept in one venv since it works;
revisit only if NeMo itself turns out to be Python-3.13-incompatible (being
installed now, not yet confirmed).

## 2026-08-21 — MentraWearNet-v1 core implemented; causality gap found and documented, not hidden

**Decision**: shipped `SpeakerNetBackbone` + `MentraWearNet-v1` as real,
tested modules (4,698,618 total deployment params, well under the 10M cap).
Did NOT proceed to launch training this session, per the explicit stop
condition (backbone extraction + structural tests first).

**Evidence**: forward/backward/frozen-backbone tests all PASS. Causality
test FAILS — the pretrained SpeakerNet encoder's symmetric-padded
convolutions leak a small, bounded amount of future audio context (max
diff 0.0098, growing measurably toward the future boundary, confirming the
custom causal TCN is not the source).

**Why recorded as FAIL, not smoothed over**: the alternative was either
skip the test (fabrication by omission) or declare a loose tolerance to
pass it (fabrication by redefinition). Neither is honest. The real finding
— non-causality is bounded and localized to the reused pretrained
backbone, not the new code — is more useful than a fake PASS, and leaves
three concrete resolution options open in `docs/MENTRAWEARNET_ARCHITECTURE.md`
rather than a false sense of "done."

## 2026-08-21 — GPU environment: root-caused and fixed, not worked around

**Decision**: pinned `torch==2.6.0+cu118` for real GPU training capability,
replacing the `CUDA_VISIBLE_DEVICES=""` workaround used for CPU-only
parity work.

**Evidence**: `nemo_toolkit[asr]`'s own install silently upgraded torch to
a `cu130` build the installed driver (535.288.01, CUDA 12.2 max) can't run
— confirmed via the actual PyTorch warning message, not inferred. Pinning
back to the previously-working `2.5.1+cu121` failed because
`nemo-toolkit==3.0.0` requires `torch>=2.6.0`. `torch==2.6.0+cu118`
satisfies both constraints (NeMo's version floor, driver's CUDA ceiling).
Verified with a real 2-process `torchrun` DDP job completing cleanly, not
just `torch.cuda.is_available()==True`.

**Why this matters going forward**: the user explicitly flagged that the
CPU-only workaround "is not a training fix" and required a genuine
diagnosis before any 2x3090 run. This is that diagnosis, done before
training was attempted, not discovered mid-run.

## 2026-08-21 — Papers archived from arXiv, not fabricated when unavailable

**Decision**: 19 of 20 requested papers downloaded as real PDFs to
`Papers/`. `DENSE` (arXiv 2409.06136) has a valid abstract page but no PDF
available at arXiv — recorded as `PDF_NOT_OPENLY_AVAILABLE`, not
substituted or faked.

**Why**: standing instruction against fabrication. A missing PDF is a fact
worth recording accurately, not a gap to paper over.

## 2026-08-27 — Parallel processing (concurrent GPU jobs) from now onwards

**Decision**: default to running independent training/eval jobs
concurrently on this box rather than one-at-a-time per GPU, going forward
for all future campaigns, not only the one that prompted it.

**Trigger**: user asked to "utilize more gpu to go faster" while the
GeoWearNet G2 real-training ablation campaign
(`training/geowearnet/mmcsg/campaign_ablations.py`) was running only 2
jobs total (1 per GPU) sequentially through a 10-experiment queue.

**Evidence before deciding**: `nvidia-smi --query-gpu=utilization.gpu`
showed 0% GPU compute utilization during active training of both
in-flight jobs. This is a real, measured fact, not an assumption — it
means "more GPU" in the literal sense (more compute) wasn't available to
give; the actual constraint was CPU-side dataloading (~29K-parameter
model, feature extraction/resampling dominates wall-clock, not matrix
math). Confirmed via `ps -ef` that each job spawns 6 dataloader worker
processes, and `uptime` showed the shared host already under heavy load
(~65-80) from other users' jobs.

**Action taken**: added a `--concurrency` flag to `campaign_ablations.py`
(jobs per GPU, 1→3), reduced `num_workers` per job 6→3 to compensate, and
relaunched. Net effect: 6-7 concurrent jobs instead of 2, better
overlapping I/O wait with GPU's large idle memory/compute headroom (both
3090s were using only 0.5-3.4GB of 24GB).

**Complication found and fixed, not glossed over**: the first
kill-and-restart left some old child processes alive (a `pkill -f` pattern
mismatch), which kept writing into the same log files as the new run and
briefly doubled real GPU/CPU load. Diagnosed with `ps -ef`/`pstree`
(showing stale PIDs reparented to init), fixed with explicit PID-targeted
`kill -9`, and verified via a clean `pstree` that only the new process
tree remained before declaring it fixed.

**Two independent, pre-existing bugs surfaced by this change (not caused
by it) were also fixed while here**, because the concurrency change is
what got the `spectral_only` (WS31 drop-amplitude-features) ablation far
enough to actually run and hit them:
1. `training/geowearnet/mmcsg/norm_stats.py`:
   `compute_real_norm_stats()` accepted a `drop_amplitude` semantic via
   its cache filename (`path_for()`) but never actually threaded the flag
   into the `RealDataConfig` used to compute the stats — so the cached
   file for the drop-amplitude condition silently contained stats for the
   full (non-dropped) 14-feature set, causing a broadcast shape error at
   training start. Fixed by passing `drop_amplitude` through to
   `RealDataConfig` inside `compute_real_norm_stats()`.
2. `training/geowearnet/mmcsg/train_real.py`: sim-pretrained `init_from`
   used `model.load_state_dict(strict=False)`, which only tolerates
   missing/unexpected *keys*, not shape-mismatched tensors for keys
   present in both state dicts — so the drop-amplitude ablation's
   narrower physical-feature head couldn't load G1's checkpoint at all
   (hard `RuntimeError`, not a graceful skip). Fixed by filtering out any
   checkpoint tensor whose shape doesn't match the current model's shape
   before calling `load_state_dict`, logging exactly which keys were
   skipped, so the rest of the simulation-pretrained network still
   transfers correctly and only the genuinely incompatible layer is left
   at random init.

**Why recorded as a standing decision, not a one-off**: the same
GPU-idle/CPU-bound signature will very likely recur for any other small
GeoWearNet/SpeakerNet-scale model trained on this box, so defaulting to
checking `nvidia-smi` utilization first and parallelizing when GPU compute
is idle (rather than assuming single-job-per-GPU is optimal) is the
correct default posture going forward — not something to re-derive from
scratch each time. See `MEMORY.md`'s matching entry for the operational
caveats (re-verify per workload, shared-host load ceiling, kill/restart
process hygiene).
