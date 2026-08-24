# Source Separation Research Track (scoping document)

Status: **RESEARCH ONLY**. No implementation code was written or modified
to produce this document, and none should be written on the basis of it.

## Gate — read this before anything else

**Do not start separator implementation.** The currently-training
MentraWearNet detection model
(`training/checkpoints/mentrawearnet_v1.pt`, 5000 steps, real
LibriSpeech+MUSAN data) has not yet been evaluated with
`scripts/model/evaluate_mentrawearnet.py` against the measured SpeakerNet
TIR baseline (13.33% / 20.83% / 29.17% EER at TIR 0 / -5 / -10 dB, from
`docs/SPEAKERNET_PARITY.md` and `docs/MODEL_SELECTION.md`). Per this
project's standing principle — measurements decide, not speculation — a
separator build should not begin until that evaluation exists and the
detection model is shown to actually beat (or at least not regress
against) that baseline. This document is groundwork for a future track:
architecture options, an evaluation protocol, a gating control-flow
sketch, and a blocked hardware experiment. It is not a build plan to
execute now.

---

## 1. Verification of input claims

Before writing any of the content below, six specific factual claims
handed to the user (source unknown — possibly an AI-generated proposal
summary) were checked against real search results. A second, more
detailed batch of twelve further claims arrived mid-task (see 1.7–1.18)
and was verified the same way before being merged into this document.
Findings:

| # | Claim | Status |
|---|---|---|
| 1 | CHiME-10 "SG-TSE" task, two tracks (wearer / interlocutor) | **VERIFIED** |
| 2 | SpeakerBeam-SS, Conv-TasNet+SSM, 78% RTF reduction | **VERIFIED** |
| 3 | REAL-TSE Challenge 2026, online low-latency track | **VERIFIED** |
| 4 | TargetVoice (2025), 10MB+12MB, 3s enrollment | **VERIFIED** |
| 5 | Gated overlap-only separation, ~halves WER at 44% extra compute | **VERIFIED** |
| 6 | CHiME-9 wearable TSE baseline, causal GridNet + FiLM | **VERIFIED** |
| 7 | REAL-TSE online track: ≤100ms latency, verified via perturbation testing | **VERIFIED** |
| 8 | CARTSE: winning REAL-TSE online system, exact pipeline + metrics | **VERIFIED** |
| 9 | DeepSound "SA-Mamba": speaker-conditioned Mamba, exact params/latency/ablation | **VERIFIED** |
| 10 | CARTSE: ~38% target-absent training + target-silent loss | **VERIFIED** |
| 11 | CARTSE: offline teacher → pseudo-label → quality-filter (~37.9h) → student | **VERIFIED** |
| 12 | MERL: winning offline submission, multi-stage + real far-field adaptation | **VERIFIED** |
| 13 | CARTSE: mixture/enrollment similarity as WER predictor + channel-gap augmentation | **VERIFIED** |
| 14 | REAL-TSE organizers found DNSMOS-OVRL gaming, switched to DNSMOS-P808 | **VERIFIED** |
| 15 | SonicAGI: bounded lookahead, 96ms latency figure | **VERIFIED** (terminology correction, see 1.15) |
| 16 | CHiME-9 2026 wearable system, 19.875ms whole latency, multichannel beamforming | **VERIFIED** |
| 17 | CHiME-9 2026 band-split recurrent system, 20ms max latency, subband conditioning | **VERIFIED** |
| 18 | DeepSound: lower STFT bands carry more speaker info, justifies 257→36 bands | **VERIFIED** |

All 18 verified with citable sources — a notably higher hit rate than the
original "assume fabricated until proven" prior should predict by
default, worth stating plainly rather than downplaying, while the
verification discipline that produced it (primary sources, not
specificity-as-evidence) is what should carry forward regardless.
Sections 2-6 of this document build on these where relevant, with any
additional engineering judgment clearly marked as such rather than
attributed to a source.

### 1.1 CHiME-10 "SG-TSE" — VERIFIED

CHiME-10 (announced ~July 2026 per the CHiME challenge site's news
section) consists of three tasks: **Task 1 — ECHI-2**, **Task 2 —
URGENT**, and **Task 3 — SG-TSE**. The SG-TSE (Smart-Glasses Target
Speech Extraction) task page describes it as "extracting a desired
speaker from realistic multichannel, multi-person recordings captured by
smart glasses," with exactly two tracks:

- **Track 1 — Wearer's Speech Enhancement**: enhance the wearer's own
  speech for robust ASR and interactive agents.
- **Track 2 — Interlocutor Extraction**: extract a specified interlocutor
  for selective listening applications.

The dataset is described as bilingual, combining real human-worn
conversations with HATS-based (Head-and-Torso Simulator) replay
recordings providing paired mixture/clean-target signals on smart-glasses
platforms. Evaluation emphasizes robustness to noise, reverberation,
overlapping speech, and changing speaker positions, plus generalization
from controlled HATS recordings to real conversations. Specific metrics
and baseline system were not published on the fetched page at the time of
this check.

Source: [chimechallenge.org — CHiME-10 Task 3 (SG-TSE)](https://www.chimechallenge.org/challenges/chime10/task3/index), [chimechallenge.org — CHiME-10 index](https://www.chimechallenge.org/challenges/chime10/index)

This is directly relevant: the SG-TSE two-track split (wearer's own
speech vs. a specific interlocutor) maps closely onto this project's
wearer-vs-environment framing, though MentraWearNet currently treats
"environment" as an undifferentiated class rather than extracting one
specific interlocutor among several.

### 1.2 SpeakerBeam-SS — VERIFIED

Real Interspeech 2024 paper: "SpeakerBeam-SS: Real-time Target Speaker
Extraction with Lightweight Conv-TasNet and State Space Modeling" (Sato,
Moriya, Mimura, Horiguchi, Ochiai, Ashihara, Ando, Shinayama, Delcroix).
It adds state-space modeling (S4D) to a causal Conv-TasNet-based target
speaker extractor so that fewer dilated convolutional layers are needed
to capture long-term dependency, reducing model complexity. The paper
reports a **78% reduction in real-time factor relative to a conventional
causal Conv-TasNet TSE baseline, while matching its separation
performance**, achieved partly by enlarging the TasNet frontend
encoder's window/stride (compensating the resulting quality loss with a
larger frontend) to cut compute further.

Source: [ISCA Archive — SpeakerBeam-SS](https://www.isca-archive.org/interspeech_2024/sato24_interspeech.html) ([PDF](https://www.isca-archive.org/interspeech_2024/sato24_interspeech.pdf)), independent reimplementation: [OpenSpeakerBeam-SS](https://github.com/helloooideeeeea/OpenSpeakerBeam-SS)

This is the strongest verified precedent for combining a lightweight
time-domain separator with SSM blocks for causal streaming TSE — directly
informs candidate architecture B below.

### 1.3 REAL-TSE Challenge 2026 — VERIFIED

"SLT 2026 REAL-TSE Challenge: Real-world Target Speaker Extraction from
Conversational Recordings," a satellite challenge of IEEE SLT 2026. Given
a multi-speaker mixture and one or more target-speaker enrollment
utterances, systems must recover only the target speech. Unlike
simulated read-speech benchmarks it uses real Mandarin/English
conversational recordings with natural overlap, reverberation, noise,
channel mismatch, and conversational dynamics. It defines two tracks: an
**Online track** (low-latency streaming extraction) and an **Offline
track** (full-context processing). Evaluation metrics: **Token Error
Rate (TER)**, **Speaker Similarity (SpkSim)**, **DNSMOS**, and
**target-speaker activity F1**. (TER is the challenge's actual metric
name; it is functionally an ASR-error-rate metric, close to but not
verbatim "ASR WER" as phrased in the original claim.) Submission deadline
for both tracks was June 25, 2026 (AoE); system-description reports were
due July 1, 2026.

Source: [arXiv:2607.15198 — SLT 2026 REAL-TSE Challenge](https://arxiv.org/abs/2607.15198), [real-tse.github.io/challenge](https://real-tse.github.io/challenge/)

### 1.4 TargetVoice (2025) — VERIFIED

"TargetVoice: Single Channel Low-Latency Target Speaker Extraction"
(Pallala, Chennupati, Padmanaban, Pogula, Ravuri, Ellanki, Rajamani,
Ambati), Interspeech 2025. A lightweight single-channel TSE system for
edge devices (call centers, conference calls, hands-free communication,
smart speakers). Confirmed specifications from the ISCA archive abstract:
**speaker encoder ≈ 10MB**, **extraction block ≈ 12MB**, **≈6G MACs**,
and a **single 3-second enrollment utterance**. Architecture: a
20ms-window/10ms-hop STFT-domain encoder (Hanning window) feeding a
convolutional layer, LSTM, an inter-LSTM, and a self-attention module;
the encoder speaker embedding is fused with the mixture's time-frequency
representation in the extraction block.

Source: [ISCA Archive — TargetVoice](https://www.isca-archive.org/interspeech_2025/pallala25_interspeech.html) ([PDF](https://www.isca-archive.org/interspeech_2025/pallala25_interspeech.pdf))

This is a directly comparable existing system at almost exactly this
project's target scale (10MB ≈ 2.6M fp32 params for the encoder-sized
component; 12MB ≈ 3.1M fp32 params for the extractor) — useful as an
external sanity check that a 1-3M-parameter conditioned extractor is a
credible, previously-shipped design point, not wishful sizing.

### 1.5 Gated overlap-only separation, ~halving WER at 44% extra compute — VERIFIED

"Lightweight Target-Speaker-Based Overlap Transcription for Practical
Streaming ASR" (arXiv:2506.20288; also published at TSD 2025). The system
runs a speaker-independent (SI) ASR path continuously, plus a compact
binary overlap classifier trained on frozen SI-model outputs at
negligible cost. Only when overlap is detected does it switch to a
speaker-conditioned (SC) path — an ASR model that uses FiLM to inject a
target-speaker embedding, trained on synthetically mixed data, applied
only to the flagged overlap segments. On a Czech television-debate test
set with 16% overlap, this reduced overlap-segment WER from **68.0%
(baseline SI-only) to 35.78%** (a ~47% relative reduction — close to but
not exactly "halving," so stated precisely here rather than rounded up),
while increasing total computational load by **exactly 44%** (relative
HW cost 1.44x) — because the expensive SC path only runs on the minority
of frames flagged as overlapping.

Source: [arXiv:2506.20288](https://arxiv.org/abs/2506.20288), [ACM DL / TSD 2025](https://dl.acm.org/doi/10.1007/978-3-032-02548-7_1)

This is direct, verified precedent for the gated-execution design in
section 4 below (cheap path always-on, expensive conditioned path
triggered by a cheap detector) — not just plausible engineering logic but
a measured result from a closely analogous system.

### 1.6 CHiME-9 wearable TSE baseline, causal GridNet + FiLM — VERIFIED

CHiME-9 Task 2 is **ECHI** ("Enhancing Conversations to address Hearing
Impairment"), targeting hearing-aid and smart-glasses devices (4-channel
hearing aids + 7-channel Aria glasses in the released dataset), aiming to
produce a single-channel enhanced stream for each conversational partner
of the device wearer (i.e., the opposite framing direction from
MentraWearNet — CHiME-9 ECHI extracts the *other* speakers for the
wearer's benefit; MentraWearNet currently detects the *wearer's own*
activity). Its baseline system is confirmed to use: a convolutional
encoder over the noisy device audio and a participant's clean speech
passage, a speaker encoder producing a target-speaker embedding,
**FiLM to infuse that embedding into the noisy-audio representation**,
and a **causal GridNet block** (TF-GridNet with modified padding to make
it causal) processing the FiLM-conditioned representation, followed by a
convolutional decoder producing the target speaker's spectrogram. Speaker
embedding + causal GridNet stages can be repeated (multi-stage).

Source: [chimechallenge.org — CHiME-9 Task 2 baseline](https://www.chimechallenge.org/challenges/chime9/task2/baseline), [chimechallenge.org — CHiME-9 Task 2 (ECHI)](https://www.chimechallenge.org/current/task2/index)

This is a second verified precedent (alongside SpeakerBeam-SS) for
FiLM-based target-speaker conditioning feeding a causal
mask/separation network — structurally similar to how MentraWearNet
already conditions its TCN on the wearer embedding via FiLM (see
`docs/MENTRAWEARNET_ARCHITECTURE.md`), which is encouraging for reusing
that conditioning path for a separator rather than inventing a new one.

### 1.7 REAL-TSE 100ms online-track bound + perturbation testing — VERIFIED

REAL-TSE's Track 1 (Online) requires "end-to-end algorithmic latency must
not exceed 100 ms," verified with a perturbation-based response-delay
test because organizers found "analytical estimates and measured delays
did not always agree" — exactly why this project's own lookahead-sweep
methodology (section 5 below) measures empirically rather than trusting
an architectural figure.

Source: [arXiv:2607.15198](https://arxiv.org/abs/2607.15198), [real-tse.github.io/challenge](https://real-tse.github.io/challenge/)

### 1.8 CARTSE — VERIFIED

CARTSE (Li Li, Shogo Seki, CyberAgent Inc.) ranked **#1** on the official
REAL-TSE Track 1 (Online) leaderboard: TER=0.699, Timing F1=0.848,
SIM=0.504, DNSMOS-P808=3.069. Pipeline: complex STFT (512-pt/32ms
window, 128-sample/8ms hop) → time-frequency enrollment map
(enrollment-magnitude × mixture-magnitude attention map) → non-uniform
band-split (32 bands) → speaker embedding conditioning (fine-tuned
ECAPA-TDNN, element-wise multiplication) → causal BSRNN (6 repeats of
band-RNN + unidirectional temporal RNN) → complex band mask → iSTFT.
Effective future dependency, measured via the challenge's own
perturbation methodology: mean **22.9ms** (range 22.2–23.7ms) — well
inside the 100ms bound.

One nuance: CARTSE's own paper reports its *self-measured* DNSMOS P.835
overall (OVRL) = 3.436, a different sub-metric from the leaderboard's
official P808 = 3.069 — a live instance of the OVRL-vs-P808 divergence
flagged in 1.14. Any number pulled from a system paper should be checked
against which DNSMOS variant it actually reports.

Source: [CARTSE-Track1.pdf](https://real-tse.github.io/assets/pdf/CARTSE-Track1.pdf), [leaderboard](https://real-tse.github.io/challenge/)

### 1.9 DeepSound "SA-Mamba" — VERIFIED

"Speaker-Aware State Space Modeling for Streaming Target Speaker
Extraction" (DeepSound). Injects the target-speaker condition directly
into a Mamba block's selective-parameter generation (input/output
matrices and discretization step all depend on speaker condition, not
just externally fused after temporal modeling). Ranked **#2** on REAL-TSE
Track 1: **15.89M params, 30.01 G MACs/s**, beating both official BSRNN
baselines (25.05M/27.24M) on every metric (TER=0.713, F1=0.849,
SIM=0.524, OVRL=2.151). Effective future dependency: mean **46.7ms**
(range 44.5–49.9ms). Ablation (SI-SNRi): causal TCN **8.26**, causal GRU
**10.52**, SA-Mamba **13.63**; embedding-multiplication conditioning
9.23 vs. cross-attention conditioning 11.25 SI-SNRi.

Source: [DeepSound-Track1.pdf](https://real-tse.github.io/assets/pdf/DeepSound-Track1.pdf)

### 1.10 CARTSE target-absent training + target-silent loss — VERIFIED

CARTSE trains with distractor enrollment (p=0.35) and noise-only mixtures
(p=0.05), totaling **38.25%** target-absent examples. Loss is split:
target-active L_TA (SI-SDR + multi-resolution-STFT, active frames only)
and a separate target-silent term L_TS that directly penalizes output
energy on target-absent frames.

Source: [CARTSE-Track1.pdf](https://real-tse.github.io/assets/pdf/CARTSE-Track1.pdf), Sections III-A, IV-A

### 1.11 CARTSE offline-teacher → pseudo-label → quality-filter → streaming student — VERIFIED

CARTSE's real-data fine-tuning corpus comes from its own non-causal
offline system acting as teacher on real meetings (AISHELL-4, AliMeeting,
AMI, CHiME-6 train splits), quality-filtered (positive OVRL gain,
absolute OVRL≥2.2, VAD precision≥0.80, ASR TER<0.6), yielding **7,710
clips, 37.9 hours**, then used to fine-tune the streaming student in two
stages.

Source: [CARTSE-Track1.pdf](https://real-tse.github.io/assets/pdf/CARTSE-Track1.pdf), Section III-C

### 1.12 MERL: winning offline submission, multi-stage + real far-field adaptation — VERIFIED

MERL ranked **#1** on REAL-TSE Track 2 (Offline): TER=0.613, F1=0.861,
SIM=0.538, DNSMOS-P808=3.371. Four-stage curriculum: (1) fully-overlapped
Libri2Mix-style pretraining, (2) noisy simulated multi-talker
conversations, (3) simulated far-field mixtures from real corpora, (4)
**real far-field mixtures** trained against pseudo-targets from a causal
Wiener-filter projection of close-talk mic onto the far-field channel,
ASR-filtered for quality.

Source: [arXiv:2607.09043](https://arxiv.org/abs/2607.09043)

### 1.13 CARTSE: mixture/enrollment similarity as WER predictor + channel-gap augmentation — VERIFIED

Confirmed verbatim: "the learned mixture–enrolment similarity is the
strongest WER predictor (Spearman ρ −0.52)." Channel-gap augmentation
applies a simulated device response independently to the enrollment
(spectral tilt ±4dB/oct, cubic-spline random EQ ±6dB, Butterworth
low/high-pass) — enrollment channel ≠ mixture channel on almost every
training sample.

Source: [CARTSE-Track1.pdf](https://real-tse.github.io/assets/pdf/CARTSE-Track1.pdf), Appendix A

### 1.14 REAL-TSE: DNSMOS-OVRL gaming, switch to DNSMOS-P808 — VERIFIED

Organizers found "some submissions achieved unusually high OVRL scores
that did not align with listening impressions," and that "OVRL is weakly
correlated with human MOS ... whereas P808 shows consistently higher
agreement." Independently corroborated by MERL, who ran their own
adversarial-attack demonstration showing DNSMOS-OVRL and speaker
similarity can both be driven to near-maximal values with an
imperceptible waveform perturbation, without moving TER or F1 at all —
i.e. both metrics are trivially gameable independent of actual quality.
CARTSE's own OVRL=3.436 vs. official P808=3.069 (1.8) is a real-world
instance of the same gap.

Source: [arXiv:2607.15198](https://arxiv.org/abs/2607.15198), [arXiv:2607.09043](https://arxiv.org/abs/2607.09043)

### 1.15 SonicAGI: bounded lookahead, 96ms — VERIFIED, with a terminology correction

SonicAGI's "SwiftNet-Lookahead" uses one chunked bidirectional-LSTM
lookahead module (max lookahead 80ms) before a strictly causal iterative
separator, so lookahead doesn't accumulate across iterations. The paper's
own arithmetic: 8ms STFT delay + 80ms lookahead = **88ms algorithmic
latency**; +8ms hop buffer = **96ms total system latency**. The original
claim's phrase "96ms total algorithmic latency" conflates two numbers the
source paper keeps separate — the 96ms figure is real, but it's *total
system* latency, not *algorithmic* latency. Also: SonicAGI's paper
self-reports "ranks second," but the official leaderboard's final
verified rankings (after removing non-compliant submissions) place it
**#3** tied with WasedaM — the leaderboard is the source of truth.

Source: [arXiv:2607.11083](https://arxiv.org/abs/2607.11083), [leaderboard](https://real-tse.github.io/challenge/)

### 1.16 CHiME-9 2026, 19.875ms, multichannel beamforming — VERIFIED

Tu et al. (Anhui University / IACAS), "A Low-Latency Multi-Stage MIMO
System with Cross-Beam Interaction for CHiME-9 Task 2 (ECHI)": "the whole
latency of our system is only 19.875 ms," derived as
τ=(L_syn+H)/f_s=(212+106)/16000. Uses real multichannel spatial
processing: a region-wise MIMO front-end (5 azimuth-sector estimates from
7-channel Aria-glasses input) drives spatial covariance estimation for an
online beamformer, followed by cross-beam interference suppression — not
a mono system, which matters directly for the deprioritization note
below.

Source: [ISCA CHiME 2026 — tu26_chime.pdf](https://www.isca-archive.org/chime_2026/tu26_chime.pdf)

### 1.17 CHiME-9 2026, 20ms max algorithmic latency, band-split recurrent, subband conditioning — VERIFIED

Jhou et al. (NTU/Academia Sinica), "MBSRNN": "a chunk-based inference
strategy ... ensures a maximum algorithmic latency of 20 ms." Multichannel
(7ch/4ch) input, 33 subbands, with band-split processing applied to the
*enrollment* itself so subband-level speaker characteristics condition
the network via FiLM. This 20ms figure is also the CHiME-9 Task 2 (ECHI)
organizer-level hard requirement — "only systems meeting the 20ms
algorithmic latency constraint will be ranked."

Source: [ISCA CHiME 2026 — jhou26_chime.pdf](https://www.isca-archive.org/chime_2026/jhou26_chime.pdf), [chimechallenge.org — Task 2 rules](https://www.chimechallenge.org/challenges/chime9/task2/rules)

### 1.18 DeepSound: low frequency bands carry more speaker info — VERIFIED

"The low-frequency region contains the fundamental frequency, low-order
harmonics, and timbre-related speaker cues ... the high-frequency region
mainly reflects consonant details, sibilants, and noise textures, and can
be appropriately merged." Justifies partitioning 257 STFT bins into 36
non-uniform bands, weighted [2×8, 4×8, 8×6, 11×13, 18×1] — fine (2-bin)
resolution below 500Hz, progressively coarser to 8kHz.

Source: [DeepSound-Track1.pdf](https://real-tse.github.io/assets/pdf/DeepSound-Track1.pdf), Section II.B

---

## 2. Candidate separator architectures

All three options are scoped against: total system budget ≤10M params
(current detection model already uses 4,698,618 — see
`docs/MENTRAWEARNET_ARCHITECTURE.md` — leaving roughly 5.3M headroom, of
which the brief asks for only 1-3M to be spent on the separator itself,
preserving margin), causal or bounded-lookahead streaming operation, CPU
inference on the Mentra Live target, and reuse of (a) the existing
192-D post-FiLM temporal representation and (b) the existing wearer
enrollment embedding — explicitly not a second full speaker encoder.

### A. STFT magnitude/phase-sensitive mask separator

Compute an STFT of the input mixture (short window, e.g. 20-25ms, to
bound latency), feed magnitude (or magnitude+phase-sensitive target)
frames alongside the existing 192-D representation and FiLM-conditioned
wearer embedding into a small mask-prediction head (a few Conv1d/GRU
layers over frequency bins, conditioned via a second FiLM application
using the same wearer embedding already computed for detection), predict
a per-bin gain mask for the wearer stream (and optionally
`1 - mask` or a separately predicted mask for environment), then
resynthesize with the mixture's phase (or a lightly refined phase if
budget allows).

- **Rough param count**: mask-prediction network on top of an existing
  192-D representation is small — a 2-3 layer Conv1d/GRU stack over
  ~257 freq bins (512-point STFT) with modest hidden width (128-192)
  lands in roughly **0.5-1.5M params**, well under the 1-3M budget,
  leaving room for a second (environment) mask head or a small
  post-filter.
- **Fit for CPU + budget**: best fit of the three. STFT/iSTFT are cheap,
  well-optimized, and CPU-friendly; the mask head is small enough to
  leave real margin under the 10M total. This is consistent with
  TargetVoice's approach (section 1.4) of an STFT-domain encoder with a
  moderate model size (~12MB extractor) achieving real-time low-latency
  operation on comparable hardware constraints, though TargetVoice's own
  extractor alone is already near the top of this project's entire
  proposed separator budget — an argument for a leaner design that
  reuses MentraWearNet's already-computed representation rather than
  recomputing acoustic features from scratch as TargetVoice does.
- **Streaming/lookahead**: STFT window size sets a hard latency floor
  (e.g., a 25ms window with 50% overlap implies ~12.5ms minimum
  algorithmic latency before any model compute); causal masking
  (predict frame t's mask from frames ≤t) adds no further lookahead
  beyond the STFT window itself. This is the simplest of the three to
  keep strictly causal or to run at the same lookahead budgets already
  measured for the detection model (see section 5).
- **Likely first to implement**: yes — smallest, cheapest, most reuse of
  existing infrastructure (the project already computes frame-level
  representations; STFT-domain masking is the standard "cheapest
  correct" separation baseline in the literature, e.g. VoiceFilter-style
  approaches), and lowest architectural risk.

### B. Lightweight time-domain conditioned separator (Conv-TasNet/SSM-style)

A learned time-domain encoder (not STFT) feeding a target-speaker
FiLM-conditioned separation network — verified as a viable design point
at small scale specifically because of SpeakerBeam-SS (section 1.2),
which demonstrated that adding state-space modeling blocks to a causal
Conv-TasNet-style separator cuts real-time factor by 78% versus a
standard causal Conv-TasNet TSE baseline while matching accuracy, and
because CHiME-9's causal-GridNet+FiLM baseline (section 1.6) and
SpeakerBeam-SS both confirm FiLM-based target-speaker conditioning is a
proven, lightweight mechanism for this class of model — which
MentraWearNet already implements for detection and could extend for
separation with a second FiLM application rather than a new mechanism.

- **Rough param count**: full Conv-TasNet-scale time-domain separators
  are typically several million to >10M params on their own; the
  budget-fitting version here would need to be substantially pruned
  (fewer separable-conv blocks, narrower bottleneck, SSM blocks in place
  of some dilated-conv stacks per SpeakerBeam-SS's approach) and would
  need to consume the existing 192-D representation directly rather than
  learning its own encoder — plausible in the **1-3M range** only if the
  learned encoder/decoder pair is shared with or replaced by the
  existing MentraWearNet frame representation, otherwise it risks
  exceeding budget on the encoder/decoder alone.
- **Fit for CPU + budget**: workable but tighter margin than option A —
  time-domain separators typically need a learned encoder/decoder (added
  parameter and compute cost) unless that step is eliminated by reusing
  the existing 192-D representation, which is architecturally attractive
  here specifically because it avoids duplicating what SpeakerNet's
  backbone already computes. SSM blocks (per SpeakerBeam-SS) are the
  verified lever for keeping this affordable on CPU: they reduced RTF by
  78% versus a plain causal Conv-TasNet TSE baseline in the cited paper,
  which is the whole reason this option is worth considering rather than
  ruled out on cost alone.
- **Streaming/lookahead**: naturally causal if built with causal
  convolutions/SSM state (no future context needed by construction,
  matching this project's existing TCN causal-padding convention) —
  potentially the lowest-latency option since it can avoid a fixed STFT
  window if the encoder itself is small and short-kernel, but only if
  the encoder/decoder are kept causal too (note the project's own
  causality post-mortem in `docs/MENTRAWEARNET_ARCHITECTURE.md` — the
  reused pretrained SpeakerNet backbone itself was found to leak a small
  amount of future context because its Jasper blocks use symmetric,
  non-causal padding; any separator built directly on that
  192-D representation inherits the same caveat unless that upstream
  issue is separately resolved).

### C. Hybrid spectral + TCN/SSM separator

Combine option A's STFT front-end (cheap, CPU-friendly, well-understood
latency) with option B's SSM/TCN-based conditioned mask refinement in
place of a purely convolutional mask head — i.e., STFT magnitude in,
FiLM-conditioned causal TCN or small SSM stack predicts the mask (instead
of a plain Conv1d/GRU stack as in option A), STFT magnitude out.

- **Rough param count**: between A and B — an STFT front-end keeps the
  encoder/decoder cost near zero (unlike B), while the SSM/TCN mask head
  can be sized similarly to MentraWearNet's existing TCN (688K trainable
  params for the whole current head, including two output heads and a
  FiLM/similarity branch) — plausibly **1-2M params** for a
  correspondingly-scaled separator mask head, comfortably inside budget.
- **Fit for CPU + budget**: good — inherits STFT's latency-cheapness and
  the SSM lever's demonstrated compute efficiency (SpeakerBeam-SS, 78%
  RTF reduction) without committing to a learned time-domain
  encoder/decoder pair.
- **Streaming/lookahead**: same STFT-window latency floor as option A,
  plus whatever lookahead the TCN/SSM mask head is configured for
  (0ms if causal-padded like the existing TCN, or a small bounded amount
  if a LowRC-style variant is used — see section 5).
- **Assessment**: this is the most defensible middle ground —
  it is the option this document would recommend evaluating first if
  option A's simpler mask head turns out to be accuracy-limited, since it
  keeps the cheap STFT front end while borrowing the one concretely
  verified efficiency trick (SSM blocks in place of stacked dilated
  convs) from the literature, rather than inventing a new time-domain
  encoder under tight budget pressure.

**Recommended sequencing (research judgment, not a claim from any
source)**: prototype A first — cheapest, lowest risk, reuses the most
existing infrastructure, and gives a real SI-SDRi/DNSMOS number to decide
whether B/C's added complexity is justified. Only escalate to C (and
treat B as a fallback if C's SSM component proves hard to get correct
under the causal/CPU constraints) if A's separation quality is
insufficient once measured against the protocol in section 3 — again,
measurements deciding, not front-loaded architecture speculation.

---

## 3. Evaluation protocol

A future separator needs to be evaluated on separation quality, not just
detection accuracy. Metrics grouped by what's already available in this
project's toolchain versus what needs new tooling:

### Separation-quality metrics

| Metric | Purpose | Tooling status |
|---|---|---|
| SI-SDRi (scale-invariant SDR improvement) | Core separation quality vs. mixture | Computable today with `mir_eval` (Python, pip-installable) or a hand-rolled SI-SDR function (a few lines of numpy) — **no new tooling needed**, standard practice |
| Speaker similarity (extracted stream vs. wearer enrollment) | Confirms the "wearer" stream is actually the wearer, not leakage | Computable today by re-embedding the extracted stream with the **existing SpeakerNet backbone** (`training/models/speakernet_backbone.py`, already used for enrollment) and cosine-comparing to the enrollment embedding — reuses existing infra, **no new model needed** |
| DNSMOS (perceptual quality, no reference needed) | Non-intrusive quality score matching what REAL-TSE (section 1.3) and typical TSE literature report | **Needs new tooling** — no DNSMOS ONNX model or wrapper currently in this repo; would need to vendor/download the standard DNSMOS ONNX model (Microsoft's public release) and add a small inference wrapper |
| Downstream ASR WER on extracted stream | Task-relevant quality (does separation help actual transcription) | **Needs new tooling** — no ASR model currently wired into this repo's evaluation path; would need to pick and integrate a CPU-feasible ASR model (e.g., a small Whisper variant) purely for evaluation, not deployment |

### Detection metrics (already exist, reuse as-is)

Wearer/environment/overlap F1, false-accept rate (FAR), false-reject rate
(FRR) — these are the existing MentraWearNet detection metrics computed
by `scripts/model/evaluate_mentrawearnet.py` and should continue to be
tracked alongside separation metrics, since the gated design in section 4
means separation quality is conditional on detection (specifically
overlap-detection) quality — a separator can't be evaluated fairly
independent of the detector that decides when to invoke it.

### System metrics (already exist, reuse as-is)

CPU latency, real-time factor (RTF), RAM, lookahead (ms), and — new for
a two-stream system — **source-switch latency** (time from a real
wearer/environment role change in the audio to the separator's output
reflecting it), which does not have an existing measurement script and
would need one, but can follow the same measurement pattern as the
existing lookahead test (`scripts/model/evaluate_lowrc_variants.py`'s
prefix-A/future-B/future-C diffing technique) applied to separator output
instead of detection logits.

### Summary: needs new tooling vs. reuses existing

- **Reuses existing tooling/models**: SI-SDRi (mir_eval or hand-rolled),
  speaker similarity (existing SpeakerNet backbone), all detection
  metrics, CPU latency/RTF/RAM/lookahead measurement patterns.
- **Needs new tooling**: DNSMOS (vendor an ONNX model + wrapper), ASR WER
  (integrate an evaluation-only ASR model), source-switch latency (new
  script, but same measurement technique as existing lookahead tooling).

---

## 4. Conditional/gated execution design

The separator should not run continuously — it should be invoked only
when the existing detection model's per-frame wearer and environment
activity heads both indicate activity (i.e., overlap), since separation
is presumably a small minority of total speaking time and is
computationally more expensive than detection alone.

Sketch of control flow, built entirely on what already exists
(`training/models/mentrawearnet.py`'s two independent sigmoid heads —
`wearer_logits` and `environment_logits` per frame, per
`docs/MENTRAWEARNET_ARCHITECTURE.md`):

```
for each incoming frame (or small frame block):
    wearer_active      = sigmoid(wearer_logits[t])      > wearer_threshold
    environment_active = sigmoid(environment_logits[t])  > environment_threshold
    overlap = wearer_active AND environment_active

    if not overlap:
        # cheap path: detection heads already tell us which single
        # stream is active (or neither) -- pass audio through untouched,
        # tagged with the active-speaker label. No separator invocation.
        emit(audio_frame, label=active_stream_or_silence)
    else:
        # expensive path: only entered on flagged overlap frames.
        wearer_stream, environment_stream = separator(
            audio_frame, wearer_enrollment_embedding, mentrawearnet_representation
        )
        emit(wearer_stream, label="wearer")
        emit(environment_stream, label="environment")
```

This pattern is justified on its own logic, independent of any external
result: the detection heads already exist and are cheap (they are the
current 4.7M-param model, already sized for real-time CPU inference), the
separator is presumably the more expensive addition, and overlap is
presumably the minority case in typical wearer-glasses usage — so gating
the expensive path behind the cheap detector's overlap signal should
reduce average compute versus always running the separator, without
requiring the separator to also perform detection. Section 1.5's verified
result (arXiv:2506.20288: switching to a speaker-conditioned path only on
detected-overlap frames cut overlap-segment WER by ~47% relative at only
44% extra total compute) is cited here as independent, closely-analogous
supporting precedent for exactly this pattern — not as the basis for the
pattern's own logic, which stands on its own regardless of that specific
number.

Practical notes for later (not decided here): threshold selection for
`wearer_threshold`/`environment_threshold` should reuse whatever
operating point the detection model's own FAR/FRR evaluation settles on
(see `docs/TODO.md` gate status); a hysteresis or minimum-dwell-time rule
on the overlap flag is likely needed in practice to avoid rapidly
toggling the separator on/off across a noisy detection boundary, but that
is an implementation detail to prototype once the detector itself is
evaluated, not something to design further in this research document.

---

## 5. Lookahead/latency research note

This project already evaluates its detection model across a lookahead
sweep — `scripts/model/evaluate_lowrc_variants.py` measures both actual
empirical future-dependency (via prefix-A/future-B/future-C output
diffing) and accuracy (EER) for a **C0 (strict causal, 0ms)** variant and
four **LowRC (bounded lookahead)** variants at approximately 10ms, 50ms,
100ms, and 160ms of lookahead. Any future separator should be evaluated
the same way, across the same style of lookahead sweep, with the same
goal already established for detection: find the **smallest lookahead
that does not cost significant accuracy**, not zero lookahead as an
end in itself. This is a principle carried over from the existing
detection-model methodology, not new research — see
`docs/MENTRAWEARNET_ARCHITECTURE.md`'s causality section for why
strict 0ms causality has real costs (the reused pretrained SpeakerNet
backbone itself already has a small measured non-causal leak, and forcing
strict causality by retraining the backbone's padding was noted there as
a real option but one that trades away pretrained-weight reuse).

---

## 6. Energy/proximity hypothesis test protocol — BLOCKED

**Status: BLOCKED pending real Mentra Live hardware audio capture.**
Per `docs/MENTRA_REMOTE_AUDIO.md`, this project currently only has
browser-microphone capture end to end (`web/app/app.vue` carries an
explicit, load-bearing disclosure banner: "Audio source: this browser's
microphone — not Mentra Live glasses"), and no confirmed way yet to read
`mic_pcm` from real Mentra Live hardware (Bluetooth terminates on an edge
device; whether that edge device can actually access the glasses'
microphone as a PCM source is itself unresolved, per that doc's "Known
limitations" section). The protocol below is written now so it is ready
to run the moment real hardware capture exists — it should not be
attempted with browser-mic-only data, since the entire point is
measuring how a glasses-mounted microphone's directional/proximity
response separates wearer speech from environment speech at controlled
distances, which a laptop/browser mic cannot approximate.

### Hypothesis

Raw signal energy (RMS), or a simple derived feature such as spectral
tilt, may alone be sufficient to discriminate wearer speech from
environment speech in a meaningful fraction of real-world cases, given
that a glasses-mounted mic sits close to the wearer's mouth and far from
most environment speakers. If true even partially, this would inform how
much the detection/separation stack needs to rely on the full learned
speaker-conditioned pipeline versus a cheap energy-based pre-filter.

### Recording protocol

Two independent variables, recorded on real Mentra Live hardware once
available:

- **Wearer loudness** (4 levels): quiet, normal, loud, whisper.
- **Environment-speaker distance and loudness** (6 distances × normal
  loudness at minimum, ideally repeated at a second loudness level if
  time allows): 10cm, 25cm, 50cm, 1m, 2m, 3m.

For each of the wearer-loudness × environment-distance cells, record a
short clip (target: a fixed, consistent sentence or reading passage per
cell, e.g. 10-15 seconds) of the wearer speaking alone, the environment
speaker speaking alone (at that distance/loudness), and both overlapping
— so per-source ground truth is unambiguous per clip and the energy
feature can be measured under known-source conditions before testing it
on overlap.

### Decision metric

For each recorded clip, compute RMS (and optionally spectral tilt) per
frame, and compute an **AUROC** treating "is this frame wearer speech"
as the binary label and the energy feature as the score, separately per
wearer-loudness level and per environment-distance/loudness combination
(and pooled across all combinations for an overall number). An AUROC near
0.5 means energy alone carries no discriminative signal at that
condition (expected at close environment-speaker distances or loud
environment/quiet wearer combinations); AUROC near 1.0 means the simple
feature is already highly discriminative there (plausibly true at large
distances, e.g. 2-3m). The shape of AUROC as a function of distance and
relative loudness is the actual research output — it tells you where
along that curve a cheap energy pre-filter would help versus where the
full learned pipeline is doing real work that a simple feature cannot
replace.

This experiment does not require any new model or architecture — it is
a measurement task that only needs real captured audio and standard
signal-processing/statistics tooling (numpy for RMS/spectral tilt,
scikit-learn or a hand-rolled ROC/AUROC for the metric). The only real
blocker is the hardware capture path itself.

---

## 7. Candidate architecture: "MentraSepNet-R1"

Naming is a placeholder, not a commitment. This is a testable hypothesis
architecture assembled from verified precedent (section 1), explicitly
designed to be ablated in the experiment campaign (section 9) rather than
assumed correct.

**Pipeline**:

1. **Complex STFT front-end**: 512-point FFT, 32ms window, 8ms
   (128-sample) hop. This specific framing is this project's own
   engineering reasoning, not attributed to any source: an 8ms hop is
   close to (not identical to) this project's existing detection-model
   frame rate per `docs/SPEAKERNET_CONTEXT_REPORT.md` — reconciling the
   two hop sizes is an open implementation question — and 32ms is a
   conservative middle ground between CARTSE's 24ms and DeepSound's 32ms
   (16ms-hop) choices.
2. **Non-uniform frequency bands, 24–32 bands**: coarser high-frequency
   bands cost less compute per unit of spectral information carried,
   which holds on its own engineering merits and is further supported by
   the DeepSound low-frequency-speaker-info finding (1.18). CARTSE used
   32 bands, DeepSound 36; 24–32 is proposed here, sized down from both
   for this project's tighter <10M ceiling (both REAL-TSE entries target
   ≥15M-param systems).
3. **Proximity/energy feature branch (hypothesis-branch, not
   assumed-good)**: ~12–16 scalars (RMS, band-energy ratios, spectral
   tilt/centroid/flux/flatness, crest factor) into a tiny MLP (a few
   thousand params). This is **this project's own hypothesis**,
   inherited from section 6's blocked energy/proximity experiment. None
   of the 18 verified claims in section 1 mention or support this
   specific branch — it should be treated as a clean scientific test
   (ablate in E3; delete if it doesn't measurably help).
4. **Target conditioning**: the existing SpeakerNet embedding
   conditioning (already how MentraWearNet works today, FiLM-based) plus,
   optionally, a **TF enrollment map** in CARTSE's style (1.8) — this
   second path is only justified because CARTSE verified it as effective;
   if dropped, MentraSepNet-R1 falls back to single-path conditioning,
   already proven in this project's existing detection model.
5. **Temporal core — an experiment, not a foregone choice**: causal TCN
   (already in production) vs. S4D (verified via SpeakerBeam-SS, 1.2) vs.
   a small speaker-conditioned SSM/Mamba (verified via DeepSound
   SA-Mamba, 1.9 — though DeepSound's 13.63 vs. 10.52 vs. 8.26 dB
   SI-SNRi ordering favoring Mamba was measured at 15.89M total params,
   ~3× this project's separator sub-budget, and may not hold at this
   project's much smaller scale — worth testing, not assuming transfers
   down).
6. **Complex mask decoder**: predicts a complex band mask per frequency
   band (as both CARTSE and DeepSound do), applied to the input STFT,
   reconstructed via iSTFT.

---

## 8. Parameter budget table

| Component | Params | Status |
|---|---:|---|
| Shared backbone (frozen SpeakerNet-derived) | ~4.0M | Already exists — matches current MentraWearNet's measured 4,009,848-param backbone |
| Conditioning / projection | ~0.3M | New |
| Energy/proximity branch (hypothesis) | <0.02M | New, ablate-first |
| Temporal core | 0.5–1.0M | New — size depends on E4's TCN/S4D/SSM choice |
| Activity heads (existing wearer/environment sigmoid heads) | <0.05M | Already exists |
| Complex-mask decoder | 0.7–1.2M | New |
| **Target total** | **~5.5–6.6M** | Preferred ≤7M, hard ceiling <10M (this project's existing constraint, unchanged) |

This is a more granular breakdown of section 2's framing: the existing
detection model already spends 4,698,618 params against the 10M ceiling,
and section 2 already scoped a 1–3M separator addition — this table is
that same budget broken into named sub-components for planning, not a
change to the ceiling.

---

## 9. Controlled experiment campaign E0–E6

Each experiment isolates one variable against the same data/eval protocol
(section 3) so a result can be attributed to a specific design choice
rather than a bundle of simultaneous changes.

- **E0 — baseline (existing)**: current activity-only MentraWearNet, no
  separation. Per the gate, this must itself first be evaluated against
  the SpeakerNet TIR baseline before any of E1–E6 start.
- **E1 — does separation learn at all**: basic complex-STFT +
  non-uniform-band + mask decoder (reuse the existing causal TCN, no
  fancy temporal core). Purpose: confirm mask-based separation produces a
  measurable SI-SDRi gain over the mixture at all before spending budget
  on anything more sophisticated.
- **E2 — dual target conditioning**: add the TF enrollment map alongside
  the existing SpeakerNet embedding conditioning, *only if* judged worth
  testing (verified as CARTSE precedent, 1.8) — otherwise skip E2 and use
  E1's single-path conditioning as the baseline for E3 onward.
- **E3 — energy/proximity branch ablation**: add the proximity/energy
  branch (section 7, item 3) on top of E1/E2's best configuration. Keep
  it only if it measurably helps TIR 0/-5/-10dB SI-SDRi over the same
  configuration without it. **If it doesn't help, delete it** — treat as
  a clean scientific test, not an assumed win.
- **E4 — temporal-core bake-off**: causal TCN vs. S4D vs. a small
  speaker-conditioned SSM/Mamba, same data/model otherwise. Tests whether
  DeepSound's ablation ordering (1.9) transfers down to this project's
  much smaller parameter budget, rather than assuming it does.
- **E5 — target-absent training**: 30–40% of examples with no target
  speaker present (following CARTSE's ~38.25% split, 1.10), independent
  enrollment channel/EQ/reverb augmentation (following CARTSE's
  channel-gap augmentation, 1.13), plus an explicit target-silent-output
  loss (following CARTSE's L_TS, 1.10) so the model learns silence rather
  than hallucinating the enrolled speaker when absent.
- **E6 — real-data adaptation (later-stage, BLOCKED)**: offline noncausal
  teacher → pseudo-label real Mentra recordings → quality-filter →
  streaming-student adaptation, following CARTSE (1.11) and MERL (1.12)
  precedent. **Blocked on the same hardware constraint as section 6**:
  this project has no confirmed way to capture real Mentra Live PCM yet
  (browser-mic capture only), so there is no real Mentra audio to
  pseudo-label. Written now so it's ready the moment real hardware
  capture exists.

---

## 10. Latency target

No verified external hard number exists for what *this project's own*
separator latency ceiling should be — the two real numbers found in this
research (100ms, 20ms) come from other challenges with different tasks
and hardware, and neither should be imported as this project's own
requirement without saying so explicitly:

- **100ms** is REAL-TSE Track 1's (Online) hard bound — a **mono**,
  single-channel task, the closest verified analog to Mentra's own
  mono-mic setup. CARTSE runs at a measured mean 22.9ms (1.8); DeepSound
  at 46.7ms (1.9).
- **20ms** is CHiME-9 Task 2 (ECHI)'s hard organizer requirement — but
  ECHI is **multichannel** (4–7ch wearable input), and both verified
  20ms-class systems (1.16, 1.17) rely on multichannel spatial
  information Mentra's single mic does not have. Importing "20ms" without
  importing the multichannel input that made it achievable would be
  citing the number without its supporting context.

**Recommendation**: set this project's own target from its own existing
groundwork rather than either imported number. This project already has
a lookahead-sweep methodology and a configured script for it
(`scripts/model/evaluate_lowrc_variants.py`, variants C0/0ms,
LowRC-1/~10ms, LowRC-5/~50ms, LowRC-10/~100ms, LowRC-16/~160ms — see
section 5). Once that sweep is run, its result — the smallest lookahead
that does not cost material accuracy — should set MentraSepNet-R1's
target, informed by (not overridden by) the 22.9–46.7ms range verified
as achievable at the mono, single-channel REAL-TSE task.

**Three latency numbers that are not the same number**, and conflating
them is a common mistake worth flagging explicitly:

1. **Historical context window** — how much *past* audio the model
   conditions on. Can be long (seconds) at no interactivity cost, since
   past audio is already available. `docs/SPEAKERNET_CONTEXT_REPORT.md`
   measured the existing frozen backbone's own per-feature normalization
   as depending on the *entire* buffered history — a historical-context
   dependency, not a future-lookahead one, though it does mean a rolling
   buffer's length changes the backbone's numerical behavior.
2. **Algorithmic future lookahead** — how many ms of *future* samples
   must be buffered before a frame can be emitted. This is the number
   that actually matters for interactivity, and what all the verified
   figures in section 1 (22.9ms, 46.7ms, 96ms, 19.875ms, 20ms) measure.
3. **Wall-clock compute latency** — actual CPU inference time per frame,
   independent of algorithmic/buffering latency. A model can have 0ms
   algorithmic lookahead and still be too slow for real time on CPU if
   its per-frame compute exceeds the frame period. No REAL-TSE or
   CHiME-9 paper surveyed here reports a comparable CPU wall-clock figure
   for this project's actual target hardware — this remains an open
   measurement this project must make itself.

---

## 11. What to deprioritize

One sentence each on why it doesn't address this project's actual
bottleneck: a CPU-only, mono-audio, <10M-parameter, low-lookahead
separator running on (eventually) Mentra Live hardware.

- **Generic speaker-verification backbone research**: this project
  already has a measured, parity-tested backbone (SpeakerNet, 4.0M
  params, `docs/SPEAKERNET_PARITY.md`) — swapping it risks losing that
  parity work for marginal gain on a task (verification) that isn't the
  bottleneck (separation under overlap).
- **Large foundation audio models**: every verified low-latency system in
  this research used small, purpose-built causal architectures (15–27M
  params at the largest) — foundation-model-scale audio models aren't
  CPU-real-time-feasible at this project's parameter ceiling.
- **Multichannel beamforming/localization**: Mentra's mic is mono. Both
  verified CHiME-9 20ms-class systems (1.16, 1.17) depend on multichannel
  spatial covariance/phase information that does not exist in a
  single-mic signal — explicitly not transferable without a hardware
  change this project does not have.
- **Diffusion-based separation**: iterative denoising-style inference
  costs multiple forward passes per output frame, conflicting with a hard
  per-frame CPU wall-clock budget; none of the verified low-latency
  winning systems here (CARTSE, DeepSound, MBSRNN, the MIMO system) use a
  diffusion-style decoder.
- **Separators over ~20M params**: even DeepSound's REAL-TSE runner-up,
  at 15.89M params, is already >2× this project's entire <10M ceiling for
  backbone + separator combined — a >20M-param separator isn't a
  scaled-up version of this plan, it's a different project.
- **Cloud GPU inference dependency**: this project's locked constraint is
  fully local CPU inference with no network stage in the critical path;
  none of the verified systems here were designed for cloud-dependent
  streaming inference — they were all built to meet a bounded on-device
  latency budget.
- **Full end-to-end ASR**: the bottleneck is *separating* wearer speech
  from environment speech before transcription happens — an end-to-end
  ASR model doesn't solve that. Every verified system here treats ASR
  purely as an *evaluation* tool for scoring separation quality (TER),
  never as the separator itself.

---

## Gate (restated)

No separator implementation should begin until:

1. `training/checkpoints/mentrawearnet_v1.pt` (5000 training steps, real
   LibriSpeech+MUSAN data) has been evaluated with
   `scripts/model/evaluate_mentrawearnet.py`.
2. That evaluation has been compared against the measured SpeakerNet TIR
   baseline (13.33% / 20.83% / 29.17% EER at TIR 0 / -5 / -10 dB).
3. The result of that comparison — not speculation, not this document —
   decides whether and how a separator track proceeds.

This document is groundwork for that future decision: verified external
context (section 1), architecture options sized to budget (section 2), an
evaluation protocol (section 3), a gated-execution design (section 4), a
carried-over lookahead methodology note (section 5), and a blocked
hardware experiment ready to run once real glasses audio capture exists
(section 6). It is not authorization to start building.
