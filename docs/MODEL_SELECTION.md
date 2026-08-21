# Model Selection — Wearer-Verification Baseline

Status: `OUR MEASUREMENT` for everything in this file. Test set: 8 LibriSpeech
test-clean speakers, full rotation (each speaker enrolled once, all 8
speakers' test utterances scored against it → 384 scored pairs per model,
48 true-wearer / 336 true-environment). Public data only, per locked
training-data rule (`docs/DATA_COLLECTION.md`). **Threshold selected on the
same data used for reporting — not a calibration/test split. Numbers here
are a first read, not a final calibrated product threshold.** Small-n
caveat: 8 speakers is enough to catch a large effect (which this is) but
too few to trust the EER to the decimal point.

## Full-utterance comparison

Param counts are exact (summed ONNX initializer tensor sizes, not estimated
from file size) via `onnx.load()` — see `scripts/count_onnx_params.py`.

| Model | Params (exact) | ≤10M? | Size | ROC-AUC | EER | Balanced Acc | Verdict |
|---|---:|:---:|---:|---:|---:|---:|---|
| TitaNet-small (NeMo) | **10.03M** | ✗ just over | 40.3MB | **0.9986** | **2.08%** | 98.96% | best accuracy, breaks the param budget |
| ResNet34_LM (WeSpeaker) | **6.63M** | ✓ | 26.5MB | 0.9654 | 8.78% | 91.96% | best accuracy **within** the 10M budget |
| CAM++ (WeSpeaker, non-LM) | 7.24M | ✓ | 29.3MB | 0.7506 | 31.25% | 71.43% | weak on this test |
| CAM++_LM (WeSpeaker) | 7.24M | ✓ | 29.3MB | 0.6859 | 37.80% | 65.63% | weakest — LM training didn't help here |

Source: `evaluation/results/day1_{campplus,campplus_lm,resnet34_lm,titanet_small}.csv`,
computed via `scripts/compute_day1_metrics.py`.

## Constraint update (2026-08-21): ≤10M params, CPU-only, continuous operation

User requirement: model must be lightweight enough to run continuously on
CPU, hard cap 10M parameters. This **eliminates TitaNet-small** (10.03M —
over by 30K params, not a rounding tolerance worth arguing) despite it
being the clear accuracy winner above.

**Update: found a better in-budget candidate.** Checked one more model from
the same sherpa-onnx release that hadn't been tested yet —
`nemo_en_speakerverification_speakernet.onnx` (NeMo SpeakerNet, smaller/older
than TitaNet, 23.4MB). Real rotation-experiment result:

| Model | Params (exact) | ≤10M? | ROC-AUC | EER | Balanced Acc |
|---|---:|:---:|---:|---:|---:|
| **SpeakerNet (NeMo)** | **5.85M** | ✓ | **0.9994** | **2.08%** | 98.81% |
| TitaNet-small (NeMo) | 10.03M | ✗ | 0.9986 | 2.08% | 98.96% |
| ResNet34_LM (WeSpeaker) | 6.63M | ✓ | 0.9654 | 8.78% | 91.96% |
| CAM++ (WeSpeaker) | 7.24M | ✓ | 0.7506 | 31.25% | 71.43% |
| CAM++_LM (WeSpeaker) | 7.24M | ✓ | 0.6859 | 37.80% | 65.63% |

**SpeakerNet matches TitaNet-small's accuracy almost exactly (same EER,
near-identical AUC) at 58% of its parameter count, comfortably inside the
10M budget.** This is now the leading candidate — supersedes the
ResNet34_LM recommendation above. Source:
`evaluation/results/day1_speakernet.csv`.

## SpeakerNet duration curve (measured)

| Window | n_wearer | n_env | wearer mean | env mean | EER |
|---:|---:|---:|---:|---:|---:|
| 0.5s | 48 | 336 | 0.313 | 0.236 | 39.14% |
| 1.0s | 48 | 336 | 0.521 | 0.249 | 6.25% |
| 2.0s | 48 | 336 | 0.672 | 0.244 | 2.08% |
| 3.0s | 39 | 273 | 0.732 | 0.248 | 0.00% (small n — 39 positives, don't over-read a perfect number) |
| full | 47 | 329 | 0.790 | 0.241 | 2.13% |

**Real tradeoff, not a free lunch:** SpeakerNet matches TitaNet-small's
accuracy at full utterance length, but needs ~2.0s of audio to get there —
at 1.0s it's still 6.25% EER, worse than TitaNet-small's 2.08% at the same
1.0s mark. Smaller model converges slower with less context. If sub-1.5s
decision latency matters more than the last ~4M params, this reopens the
question — worth weighing against product latency targets, not an
automatic win either direction.

## ResNet34_LM duration curve (measured) — SpeakerNet wins outright

| Window | n_wearer | n_env | wearer mean | env mean | EER |
|---:|---:|---:|---:|---:|---:|
| 0.5s | 48 | 336 | 0.412 | 0.370 | 47.92% |
| 1.0s | 48 | 336 | 0.683 | 0.541 | 25.00% |
| 2.0s | 48 | 336 | 0.799 | 0.593 | 10.42% |
| 3.0s | 39 | 273 | 0.849 | 0.618 | 5.13% |
| full | 47 | 329 | 0.873 | 0.635 | 8.97% |

**Not a tradeoff — SpeakerNet beats ResNet34_LM at every single window
length**, while also having fewer params (5.85M vs 6.63M):

| Window | SpeakerNet EER | ResNet34_LM EER |
|---:|---:|---:|
| 0.5s | 39.14% | 47.92% |
| 1.0s | 6.25% | 25.00% |
| 2.0s | 2.08% | 10.42% |
| 3.0s | 0.00%* | 5.13% |
| full | 2.13% | 8.97% |

(*3.0s SpeakerNet number is small-n, per caveat above — don't read it as
literally perfect.)

## Final recommendation (under the ≤10M param / CPU-only constraint)

**`nemo_en_speakerverification_speakernet.onnx` — 5.85M params, 23.4MB.**
Best accuracy at every measured window length among the four in-budget
candidates tested, and the only one whose full-length accuracy rivals the
disqualified 10M+ TitaNet-small. Target a **≥2.0s rolling analysis window**
for this model (2.08% EER) — 1.0s still costs a real accuracy hit (6.25%
EER) but may be worth it if the product needs sub-1.5s decisions; that's a
latency-vs-accuracy call for the product owner, not something to decide
silently here.

Still open before this is truly final: noise robustness (untested), CPU
latency profiling in a realistic streaming loop (current numbers are
single-shot calls on a 20-core workstation, not phone-representative), and
license verification (`models/README.md` — currently NOT VERIFIED for
commercial use).

## Overlap experiment (measured, escalation gate for MentraWearNet)

Synthetic mixtures: real wearer test clip + real interferer test clip
(different LibriSpeech speaker), RMS-scaled to target TIR, scored against
the wearer's real enrollment embedding. Negative control: two non-wearer
speakers mixed the same way (score should stay low regardless of TIR — it
does, ~0.27-0.29 flat, confirming the model isn't just reacting to "two
voices present").

| TIR | wearer+interferer mean score | EER vs no-wearer-present pool |
|---:|---:|---:|
| clean, no overlap (reference) | 0.790 | 2.13% |
| +10dB (interferer quiet) | 0.678 | 4.17% |
| +5dB | 0.603 | 4.17% |
| 0dB (equal loudness) | 0.523 | **13.33%** |
| -5dB | 0.449 | **20.83%** |
| -10dB (interferer louder) | 0.377 | **29.17%** |

Source: `evaluation/results/overlap_speakernet.csv`, `scripts/run_overlap_experiment.py`.

**Reading**: degradation is graceful, not catastrophic collapse to random —
even at -10dB TIR the model still separates the classes better than chance.
But at TIR ≤0dB (a realistic scenario — someone talking at the wearer at
normal or raised volume while the wearer speaks softly), EER is 6-14x worse
than the clean baseline. This is a real, material weakness, not a marginal
one.

**Escalation-gate verdict: PASSED.** Per the user's own stop-condition
("MentraWearNet is successful only if it materially improves at least one
major weakness... better overlap detection... without unacceptable CPU
cost"), this overlap result is the evidence needed to justify starting
MentraWearNet — plain speaker verification has a real, measurable weakness
at realistic TIR that a purpose-built temporal/activity model has a
plausible shot at improving (exposing frame-level features before global
pooling, per the architecture spec in `docs/ARCHITECTURE.md`, is a direct
response to this exact failure mode).

**What this does NOT establish**: whether MentraWearNet actually will
improve on it — that's an empirical question the training pipeline has to
answer, not something the gate result can predict. It also doesn't cover
real acoustic overlap (RIR-mixed, not RMS-summed) or noise-plus-overlap
compound conditions.

**This contradicts the "CAM++ should win on efficiency/accuracy tradeoff"
prior from the CAM++ paper's VoxCeleb benchmarks.** Two honest explanations,
neither confirmed yet: (1) CAM++'s reported strength is on standard VoxCeleb
trial pairs, not necessarily this specific 8-speaker/3-enrollment-utterance
setup — small enrollment sets may suit some architectures' embedding
geometry better than others; (2) possible mismatch between how sherpa-onnx
feeds CAM++ vs how WeSpeaker/NeMo intended it (worth a sanity check against
raw WeSpeaker/NeMo Python inference before fully trusting the CAM++ numbers
specifically). Do not treat the CAM++ result as final without that check.

## Duration curve (TitaNet-small only so far — the current leader)

| Window | n_wearer | n_env | wearer mean score | env mean score | EER |
|---:|---:|---:|---:|---:|---:|
| 0.5s | 48 | 336 | 0.257 | 0.030 | **25.15%** |
| 1.0s | 48 | 336 | 0.566 | 0.051 | 2.08% |
| 2.0s | 48 | 336 | 0.721 | 0.064 | 2.08% |
| 3.0s | 39 | 273 | 0.769 | 0.063 | 2.56% |
| full | 47 | 329 | 0.816 | 0.066 | 2.13% |

(n drops slightly at 3.0s/full because a few LibriSpeech test utterances are
shorter than 3s.)

**Reading:** EER is flat from 1.0s to full-utterance (~2-2.6%), but falls
off a cliff below 1.0s (25% EER at 500ms — unusable). This is the load-bearing
number for the product's rolling-window design: **target ≥1.0s analysis
window**, not 250-500ms as originally hoped in the sprint spec's stretch
goal. Sub-second decisions on this model are not reliable.

Source: `evaluation/results/duration_titanet_small.csv`, `scripts/run_duration_experiment.py`.

## Not yet measured (see `docs/TODO.md`)

- ResNet34_LM duration curve (only TitaNet-small run so far).
- Noise robustness (SNR sweep).
- Overlap / interfering-speaker robustness — **this is the result that
  actually decides whether Phase 3 (TS-VAD custom model) is justified.**
  Nothing below has touched overlapping speech yet.
- CPU latency/RTF under realistic streaming conditions (current numbers are
  single-shot embedding calls on 20-core CPU, not representative of a phone).
- Cross-check of CAM++ result against native WeSpeaker/NeMo inference.

## Current recommendation

TitaNet-small is the clear leader on this first test and the natural
default backend to carry into the overlap experiment next. Do not lock it
in yet — overlap and noise results could change the picture, and the CAM++
anomaly deserves a sanity check before being trusted as a real architecture
finding rather than an integration artifact.
