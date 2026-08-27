# Mentra Product Audio Architecture — the north star

**Status:** architecture of record for every Mentra audio experiment from
2026-08-27 onward. Stages 1-4 exist and are measured. Stage 5 (WearerSepNet)
is **specified only, not built, not trained** — see
`docs/geowearnet_wearersepnet_spec.md`.

---

## The one-sentence statement

> **GeoWearNet is a control signal, not the product.**

The product is: *the voice agent hears the wearer, and only the wearer, in
real time, on shared glasses, with no enrollment.* GeoWearNet is one stage
inside the pipeline that delivers that. A campaign that improves GeoWearNet's
AUROC but does not improve what the agent hears has not improved the product.

**GeoWearNet is not a separator.** It emits per-frame probabilities. It never
produces a waveform, never estimates a mask, and cannot pull two overlapping
voices apart. Anything that claims to do that is Stage 5, which does not
exist yet.

---

## The customer problem

An autobody technician, wearing shared glasses, over impact wrenches and a
shop radio, with a coworker talking two metres away, says:

> "Hey glasses, what part is this? Do we have a replacement in stock?"

The agent must act on *that*. It must not act on the coworker. It must not
miss half the technician's words because a compressor kicked in.

Three distinct problems hide inside the phrase "background noise", and they
need three different solutions:

| # | Problem | Nature | Solved by |
|---|---|---|---|
| 1 | Machinery / environmental noise | non-speech interference | classic speech enhancement (Stage 2) |
| 2 | **Other humans talking** | *valid speech* that is not the wearer's | wearer/environment detection + routing (Stages 3-4) |
| 3 | Wearer and coworker talking **simultaneously** | two valid speech signals in one waveform | target extraction (Stage 5, unbuilt) |

Problem 2 is why GeoWearNet exists at all: **a denoiser will not suppress a
coworker.** To any noise suppressor, a nearby human talking is exactly the
signal it is designed to preserve. No amount of denoising solves competing
speech.

---

## The pipeline

```
   raw Mentra mic PCM  (16 kHz mono, absolute amplitude preserved)
            │
            ▼
   [1] capture / transport                    server/audio, mentra/audio
            │                                 no AGC, no normalisation, no
            │                                 cloud round-trip
            ▼
   [2] optional machinery denoising           RNNoise (frozen, BSD-3)
            │                                 evaluation/agent_audio/denoise.py
            │                                 ~10 ms latency, CPU-native
            │                                 handles problem 1 ONLY
            ▼
   [3] GEOWEARNET                             training/geowearnet/
            │                                 enrollment-free, causal, ~29K params
            │   ├─► P(wearer)      per 10 ms frame
            │   └─► P(environment) per 10 ms frame
            │                                 handles problem 2's DETECTION half
            ▼
   [4] GEOWEAR GATE  ("Wearer-Routed Audio")  evaluation/agent_audio/gate.py
            │                                 per-frame gain on the ORIGINAL
            │                                 waveform. Hysteresis, attack/
            │                                 release, hangover, crossfade.
            │                                 NOT separation.
            │
            │   state 00 silence      → mute
            │   state 10 wearer only  → pass          ← solved
            │   state 01 env only     → mute          ← solved
            │   state 11 overlap      → ??? ──────────┐
            │                                         │
            ▼                                         ▼
   wearer-dominant PCM                    [5] WEARERSEPNET  (SPEC ONLY —
            │                                  NOT BUILT, NOT TRAINED)
            │                                  geometry-conditioned target
            │◄─────────────────────────────────  extraction for state 11
   ▼
   [6] ASR / voice agent                   evaluation/agent_audio/asr.py
                                           provider-agnostic interface
```

### Current live integration status

The receiver has an opt-in G2 runtime path:

```bash
MENTRA_MODEL=geowearnet_g2 \
  .venv/bin/python scripts/mentra/run_receiver.py --listen 127.0.0.1 --port 8765
```

It loads the copy-verified frozen checkpoint selected by the G2 campaign,
reuses the train-derived real-MMCSG normalization statistics, skips
enrollment, and advertises ready=true only after that contract loads. The
default MENTRA_MODEL=speakernet path remains unchanged. The live audio policy
also defaults to passthrough; the two explicit opt-in routing policies are
selected with MENTRA_AUDIO_POLICY=geowear_gate or
MENTRA_AUDIO_POLICY=rnnoise_geowear_gate. These policies are gain routing and
optional denoising, not source separation. G3's controlled identity
counterfactuals classify the old identity concern as
IDENTITY_INFORMATION_PRESENT_BUT_NOT_CAUSAL, but this remains an integration
smoke path, not Mentra-hardware validation.

### Why the gate operates on the original waveform

Stage 4 multiplies the *unprocessed* signal by a gain envelope. It does not
resynthesise. Consequences, all deliberate:

- **In states 00/10/01 it is exact.** Passing wearer-only audio through a
  gain of 1.0 is bit-identical to the microphone. There is no processing
  artefact for ASR to trip over, which is why the oracle-gated wearer WER is
  so close to the clean-speech floor.
- **In state 11 it is provably limited.** A scalar gain applied to a mixture
  of two voices scales both. Pass, and the bystander leaks; mute, and the
  wearer's simultaneous words are deleted. There is no third option. This is
  not an implementation shortcoming — it is what "gain-only" means.

That bound is the entire justification for Stage 5, and it is why the
evaluation lane measures **how much of the residual error actually lives in
state 11** before anyone builds a separator.

---

## Locked constraints

These are not preferences; changing one invalidates the deployment model.

- **No enrollment.** The glasses are shared between workers on a shift. A
  system needing a 3-second voice enrollment per user (TargetVoice-style
  target-speaker extraction) is the wrong shape for this deployment. This is
  the direct justification for GeoWearNet's enrollment-free, device-relative
  design, and it binds Stage 5 too.
- **Audio only.** No camera, no IMU in the always-on path. The camera is an
  explicit agent tool invoked on request ("what part is this?" → an
  intentional photo), never a continuous signal for deciding who is speaking.
  MMCSG ships IMU data and this project deliberately does not use it.
- **Fully local, streaming, causal.** No cloud stage, no file round-trip, no
  transcription in the decision path. Every model in stages 2-5 is causal
  with zero lookahead.
- **CPU-only inference, ≤10M params total.** Laptop now, phone later. GPUs
  are for training only.
- **Absolute amplitude preserved end to end.** GeoWearNet's physical cues
  depend on near-field proximity gain; any AGC or per-file normalisation
  upstream destroys the signal it classifies on.

---

## How this is measured

The evaluation axis is **not AUROC**. It is what reaches the agent.
Implemented in `evaluation/agent_audio/`:

| Metric | Direction | What it catches |
|---|---|---|
| **Bystander leakage rate (BLR)** | ↓ | the agent heard the coworker |
| Wearer WER / TER | ↓ | the agent misheard the technician |
| Wearer deletion rate | ↓ | over-aggressive gating ate the technician's words |
| **False agent command rate** | ↓ | the agent *acted* on someone who is not the wearer |
| Wearer command retention | ↑ | the technician had to repeat themselves |
| Speaker attribution error | ↓ | timestamp-level leakage, even for garbled words |
| Latency (algorithmic + measured) | ↓ | it stopped feeling real-time |

Perceptual metrics (PESQ, DNSMOS, SI-SDR, speaker similarity) are
**secondary** and are explicitly forbidden from overriding downstream agent
accuracy in any acceptance decision. A pipeline that sounds better and
transcribes worse has made the product worse.

The six-way comparison that answers "is a separator needed":
`RAW` / `RNNOISE` / `GEOWEAR_GATE` / `RNNOISE→GEOWEAR_GATE` / `ORACLE_GATE` /
`RNNOISE→ORACLE_GATE`, on identical windows
(`evaluation/agent_audio/run_matrix.py`).

**The oracle gate is the load-bearing measurement.** It is the same gate with
the same policy fed ground-truth activity instead of predictions. Therefore:

- `RAW → ORACLE_GATE` = everything detection-plus-routing can possibly deliver.
- `ORACLE_GATE → GEOWEAR_GATE` = what is currently lost to *detector error*
  (fixable by a better detector).
- whatever the oracle *still* gets wrong = the *routing architecture's own
  ceiling*, which is overlap, which is Stage 5's territory.

---

## Sequencing (do not skip ahead)

1. **G2 must land a checkpoint** — prove enrollment-free wearer/environment
   detection survives sim→real-wearable transfer. In progress.
2. **Measure the product gap with the oracle bound** — before building a
   separator, establish that a large fraction of residual error genuinely
   requires one. This is the P1 lane's purpose.
3. **Real Mentra pilot** (`docs/geowearnet_capture_protocol.md`) — confirm
   the MMCSG-derived result holds on actual hardware in actual shops.
4. **Only then**, and only with explicit go-ahead, consider WearerSepNet.

A geometry-conditioned separator built before wearer detection is proven
reliable is built on sand. Stage 5 stays gated: design and specification work
is permitted, GPU training is not.

---

## Related documents

- `docs/geowearnet_wearersepnet_spec.md` — Stage 5 design, acceptance
  metrics, and conditioning study (P1.14/P1.15/P1.16). **Spec only.**
- `docs/geowearnet_capture_protocol.md` — real Mentra product bench (P1.17).
- `docs/geowearnet_g2_mmcsg_report.md` — the detector campaign's own results.
- `evaluation/geowearnet/mmcsg/known_annotation_anomalies.json` — known
  corpus label defects that affect aggregate numbers (P1.13).
