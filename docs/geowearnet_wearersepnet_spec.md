# WearerSepNet — design specification

> ## STATUS: SPECIFICATION ONLY.
> **Not built. Not trained. No GPU time has been or may be spent on this.**
> This document exists so that *if* the overlap measurement justifies a
> separator, the design work is already done and the acceptance criteria were
> fixed **before** anyone saw a training curve. Writing acceptance metrics
> after seeing results is how projects talk themselves into shipping
> something that does not help.
>
> Building this requires an explicit go-ahead that has not been given.

Covers P1.14 (design), P1.15 (acceptance metrics), P1.16 (conditioning study
design).

---

## 1. Why this might be needed — and the test that decides

Stage 4 of `docs/geowearnet_product_architecture.md` is a **gain-only
router**. It multiplies the microphone signal by a per-frame scalar. In the
four activity states:

| State | Meaning | Router's options | Outcome |
|---|---|---|---|
| 00 | silence | mute | solved |
| 10 | wearer only | pass (gain 1.0) | solved, bit-exact |
| 01 | environment only | mute | solved |
| **11** | **both talking** | pass **or** mute | **unsolvable** |

In state 11 a scalar gain scales *both* voices identically. Pass → the
bystander leaks into the agent. Mute → the wearer's simultaneous words are
deleted. There is no third option available to a router. Only a system that
estimates a **time-frequency mask or a resynthesised waveform** can keep one
voice and discard the other from the same samples.

### The go/no-go criterion (predeclared)

WearerSepNet is justified **only if both** of the following hold on real
wearable audio, measured with the oracle gate (`evaluation/agent_audio/`,
`overlap.py::separator_value_estimate`):

1. **Overlap dominates the residual leakage.** With a *perfect* detector,
   ≥ 50% of remaining bystander leakage is attributable to state 11.
   If leakage is mostly in state 01, the problem is detection, not
   separation — invest in the detector.
2. **Muting overlap is genuinely expensive.** Switching the oracle gate from
   pass-overlap to mute-overlap must cost ≥ 5 percentage points of wearer
   word retention. If muting overlap costs almost nothing, then *just mute
   overlap* — ship that, skip the separator entirely.

Both conditions failing, or either failing, means **do not build this**.
The corner a separator is trying to reach — low leakage *and* low deletion
simultaneously — is only worth reaching if a router provably cannot get near
it.

---

## 2. Interface contract

Fixed now so the rest of the system can be written against it.

```python
class WearerSepNet(Protocol):
    """Streaming, causal, enrollment-free wearer-dominant extraction."""

    def reset(self) -> None:
        """Clear all recurrent/streaming state. Called at stream start."""

    def process_frame(
        self,
        pcm_frame: np.ndarray,        # (hop,) float32, 16 kHz mono, 10 ms
        p_wearer: float,              # GeoWearNet P(wearer)      this frame
        p_environment: float,         # GeoWearNet P(environment) this frame
        geo_features: np.ndarray | None = None,
                                      # (P,) GeoWearNet frame representation:
                                      # physical/geometry features, or an
                                      # intermediate embedding. See §4.
    ) -> np.ndarray:                  # (hop,) float32 wearer-dominant PCM
        ...
```

**Input:** mixed 16 kHz mono PCM + the GeoWearNet frame representation.
**Output:** wearer-dominant waveform, same rate, same length, same alignment.

Hard requirements, inherited from the product constraints and non-negotiable:

- **Streaming and strictly causal.** Zero lookahead. Verified numerically the
  same way `model_zoo.check_causality()` verifies GeoWearNet: perturb a future
  input, assert no past output changes.
- **CPU-deployable.** Laptop now, phone later. Must run faster than real time
  on one to two CPU threads alongside GeoWearNet and ASR.
- **Audio only.** No camera, no IMU.
- **No enrollment.** No speaker embedding, no reference utterance, no
  per-user state that survives a session. The glasses are shared.
- **Absolute amplitude preserved.** Must not internally normalise away the
  near-field level cue, and must not emit a signal whose absolute scale
  differs from the input's wearer component (ASR and downstream gating both
  depend on it).
- **Graceful in non-overlap.** In states 00/10/01 the router already has an
  exact answer. The separator must not make those states *worse*. Strongly
  preferred: bypass it entirely outside state 11 (see §6).

---

## 3. Budget

| Quantity | Target | Hard limit |
|---|---|---|
| WearerSepNet parameters | 0.5M – 3M | — |
| **Entire system** (GeoWearNet + WearerSepNet + any enhancement) | — | **< 10M** |
| Algorithmic latency added | ≤ 20 ms | ≤ 32 ms |
| CPU real-time factor (1 thread) | < 0.3 | < 1.0 |

GeoWearNet currently costs ~29K parameters, so essentially the entire budget
is available. The hard 10M ceiling may only be exceeded if a measurement —
not an intuition — shows the smaller model is capacity-limited on the
acceptance metrics in §5.

---

## 4. Conditioning — what "geometry-conditioned" means here

The distinguishing idea is that WearerSepNet is conditioned on
**device-relative geometry**, supplied by GeoWearNet, rather than on
**speaker identity**, supplied by an enrollment clip. That is what makes it
compatible with shared glasses.

Three candidate conditioning signals, to be compared empirically (§7), not
chosen by argument:

- **C1 — probabilities only.** `(p_wearer, p_environment)` per frame. Two
  scalars. Cheapest, and the weakest hypothesis: it tells the separator
  *when* to extract but nothing about *what* the wearer sounds like on this
  device.
- **C2 — physical feature vector.** The 14 physical/geometry scalars
  GeoWearNet already computes (`training/geowearnet/features.py`) — level,
  spectral tilt, harmonicity and friends. These carry the near-field
  proximity signature that separates wearer from bystander in the first
  place.
- **C3 — intermediate embedding.** GeoWearNet's penultimate hidden state.
  Richest, but couples the two models: retraining the detector then
  invalidates the separator. Only justified if C1/C2 measurably fall short.

Conditioning is injected via **FiLM-style feature-wise modulation** (per-frame
scale and shift on the separator's hidden channels), which is cheap, streaming-
friendly, and does not change the separator's shape when the conditioning
width changes.

---

## 5. Acceptance metrics (P1.15) — fixed before training exists

### Primary — recognition and agent correctness. These decide acceptance.

| Metric | Direction | Notes |
|---|---|---|
| Wearer TER / WER | ↓ | measured on state-11 windows specifically, and overall |
| **Bystander leakage rate** | ↓ | the headline number; recall of bystander words in the ASR hypothesis |
| Wearer activity F1 | ↑ | must not degrade the detector's usable output |
| **False agent command rate** | ↓ | the expensive failure: the agent acts for the wrong person |
| Wearer command retention | ↑ | |
| Wearer deletion rate | ↓ | guards against a separator that "wins" by muting everything |

### Secondary — perceptual and signal-domain. Diagnostic only.

SI-SDR / SDR (only where a clean wearer source genuinely exists — the
synthetic bench and any lavalier-referenced real capture), DNSMOS, speaker
similarity, STOI, PESQ where the conditions for their validity are actually
met.

> ### The overriding rule
> **Perceptual metrics must NEVER override downstream agent accuracy in the
> acceptance decision.**
>
> A model that improves DNSMOS/PESQ while increasing wearer WER or bystander
> leakage has made the product **worse** and must be rejected, regardless of
> how much better it sounds. Separators are notoriously good at producing
> pleasant-sounding output containing processing artefacts that ASR hates.
> Perceptual scores are permitted as *diagnostics* — to explain a result —
> never as the criterion.

### Acceptance thresholds (predeclared)

Against the **oracle-gate pass-overlap** baseline on the same windows:

- Bystander leakage rate: **≥ 40% relative reduction**.
- Wearer WER: **no regression** (≤ +1 point absolute).
- Wearer deletion rate: **no regression**.
- False agent command rate: **no regression**.
- Latency and RTF within §3.

Failing any of these = not accepted, regardless of SI-SDR.

---

## 6. Candidate architecture families (P1.14 — listed, NOT selected)

No architecture is chosen here. Choosing one before the go/no-go criterion in
§1 is even evaluated would be exactly the "build the complicated thing before
knowing it is needed" failure this whole lane exists to prevent.

**A — STFT mask + causal TCN.**
Causal STFT, a causal dilated TCN predicting a real or complex ratio mask,
inverse STFT with overlap-add. Latency = window length. Best understood, most
predictable CPU cost, easiest to keep strictly causal, mask is inspectable.
Weakest at very low SNR and adds musical-noise artefacts that can hurt ASR.

**B — small spectral CRNN.**
Convolutional frequency frontend + GRU over time, predicting a magnitude
mask. Very close in shape to GeoWearNet's existing winning CRNN, so it shares
the frontend and the streaming implementation already written in
`training/geowearnet/deploy.py::StreamingCRNN`. Unbounded temporal context
through the recurrent state; harder to bound worst-case behaviour.

**C — light ConvTasNet-style time-domain model.**
Learned encoder/decoder with a temporal convolutional separator, causal
variant. Strongest separation quality per parameter in the literature; no
phase-reconstruction problem. Highest CPU cost per parameter and the hardest
to keep inside the latency budget.

Selection, if it ever happens, is by measurement on the §5 acceptance suite
under the fixed §3 budget — not by reputation.

---

## 7. Conditioning study design (P1.16 — design only, do NOT run)

The experiment that measures what the detector is actually worth to
extraction. Three arms, identical architecture, identical data, identical
budget, identical seeds — the **only** difference is the conditioning input.

| Arm | Conditioning | Question it answers |
|---|---|---|
| **A** | none (separator alone) | Can a small separator do this unaided? This is the null hypothesis and it must be run — if A ≈ B, GeoWearNet contributes nothing to extraction and the coupling is unjustified complexity. |
| **B** | **predicted** GeoWearNet output | The deployable configuration. This is the number that would ship. |
| **C** | **oracle** ground-truth activity | The ceiling. Bounds how much B could improve if the detector were perfect. |

Readouts:

- **B − A** = the real, deployable value of conditioning on GeoWearNet.
  If small, the detector is not helping extraction and the two should be
  decoupled.
- **C − B** = headroom recoverable by improving the *detector* rather than
  the separator. If large, the next investment is detection, not separation.
- **C − A** = the value of activity information in principle, independent of
  current detector quality.

Each arm is scored on the full §5 primary suite, on state-11 windows
specifically and overall, with recording-level bootstrap confidence intervals
(`evaluation/agent_audio/metrics.py::bootstrap_ci_rate`) — three arms on one
dataset will produce differences small enough that point estimates alone
would mislead.

Run arm C first: if the *oracle-conditioned* separator cannot clear the §5
acceptance thresholds, no amount of detector improvement will make arms A or
B clear them either, and the whole line of work stops there having spent one
experiment instead of three.

---

## 8. What would make this document wrong

Recorded deliberately, so it can be checked rather than defended:

- If the P1.6 overlap measurement shows state 11 carries only a small share
  of residual leakage, §1's go/no-go fails and this document should be
  archived, not implemented.
- If muting overlap turns out to cost almost no wearer speech, the correct
  product decision is a one-line policy change
  (`GatePolicy.overlap_policy = "mute"`), not a new model.
- If real Mentra hardware turns out to expose multiple microphones, spatial
  beamforming may dominate any single-channel separator, and the entire
  architecture question reopens on better terms. See
  `docs/geowearnet_mentra_mic_question.md`.
