# GeoWearNet Dataset & Claim License Audit

Generated 2026-08-25 as part of the GeoWearNet Phase 3 gate. Every item below was
checked against a real web source before being written here (WebSearch + WebFetch,
this session). Nothing in this document repeats the research brief's characterizations
without independent verification. Where a claim could not be confirmed from a primary
source, it is marked **COULD NOT VERIFY** or **UNKNOWN LICENSE — DO NOT USE** rather
than softened into something that reads as confirmed.

## Claim-by-claim verification

### 1. "2026 single-microphone own-voice-detection study, 90.02% accuracy"
**VERIFIED**, with one correction: the brief's framing implies this is a
smart-glasses study; it is actually a **hearing-aid** study.
- Source: Å. Bech Jensen et al. (arXiv), *"Single Microphone Own Voice Detection
  based on Simulated Transfer Functions for Hearing Aids"*,
  https://arxiv.org/abs/2603.02724 (also PDF/HTML at arxiv.org/pdf/2603.02724 and
  arxiv.org/html/2603.02724), dated March 2026.
- Measured numbers confirmed: 95.52% accuracy on simulated head-and-torso test data
  (full-length utterances), 90.02% accuracy on 1-second segments of the same
  simulated data, and 80.00% accuracy on **real-world recordings** after a
  compensation step (no real-data fine-tuning).
- Relevance caveat for GeoWearNet: this is a conformer classifier trained on
  simulated acoustic transfer functions (ATFs) for a hearing-aid form factor, using
  the same "distinguish own voice from external speaker via spatial propagation
  cues, no speaker-dependent cues" hypothesis GeoWearNet is testing — genuinely
  relevant prior art for the core hypothesis — but it is not glasses-specific and
  the 90.02%/80.00% numbers are hearing-aid-geometry numbers, not glasses-geometry
  numbers. Do not cite this as glasses-specific evidence.

### 2. "2025 smart-glasses wearer-speech system with side-talk detector, 18% relative WER reduction"
**VERIFIED.**
- Source: *"Multi-Channel Differential ASR for Robust Wearer Speech Recognition on
  Smart Glasses"*, arXiv:2509.14430, https://arxiv.org/abs/2509.14430, published
  2025-09-17.
- Confirmed: the system combines a beamformer, microphone selection, and a
  lightweight side-talk detection model, and reports up to 18.0% relative WER
  reduction.
- Relevance caveat: this is a **multi-channel** (beamforming) system, not a
  single-channel device-relative-geometry classifier. GeoWearNet's E0/E1 scope in
  this repo is single-channel (Mentra Live's channel layout is unknown — see
  Phase 27/28 in the brief). Do not present this WER number as evidence for a
  single-mic approach.

### 3. MMCSG (Meta's smart-glasses conversational dataset)
**Existence VERIFIED. Exact license terms for the audio dataset itself COULD NOT
BE FULLY VERIFIED** — and the one concrete license text found **contradicts** the
brief's specific claim that it "permits certain research, development and
commercial uses."
- Existence: real dataset, Meta AI / FAIR, used for the CHiME-8 MMCSG Challenge.
  Sources: https://ai.meta.com/datasets/mmcsg-dataset/ ,
  https://ai.meta.com/datasets/mmcsg-downloads/ ,
  ISCA paper "The CHiME-8 MMCSG Challenge: Multi-modal conversations in smart
  glasses", https://www.isca-archive.org/chime_2024/zmolikova24_chime.html ,
  code at https://github.com/facebookresearch/MMCSG .
- License finding: the `ai.meta.com` dataset/download pages reference "our Data
  License" by name but the fetched page content (JS-rendered nav shell) did not
  surface the actual license text in this session's fetch. The one place actual
  license text was found — the `facebookresearch/MMCSG` GitHub repository — is
  licensed **CC BY-NC 4.0 (Attribution-NonCommercial)**, i.e. non-commercial only,
  which directly contradicts the brief's assumption that commercial use is
  permitted.
- Additional real constraint independent of the license question: downloading
  MMCSG requires going through Meta's dataset request/download flow (an
  account/agreement step), which this session cannot complete on the user's
  behalf per this task's own instructions.
- **Verdict: UNKNOWN LICENSE FOR THE ACTUAL AUDIO DATA — DO NOT USE** until a
  human confirms the exact Meta Data License Agreement text (not just the
  adjacent code repo's CC BY-NC 4.0) and completes the account/download
  agreement themselves. Do not treat the brief's "permits commercial use"
  characterization as fact — it is unverified and the nearest real evidence
  points the other way.

### 4. "SLT 2026 SmartGlasses Challenge"
**Existence VERIFIED. License terms VERIFIED** and match the brief's
characterization.
- Existence: real. IEEE SLT 2026 grand challenge, Mandarin-focused egocentric
  multi-talker speech dataset (106 hours, 4-channel, 714 sessions). Sources:
  https://arxiv.org/abs/2608.12034 ("The SLT 2026 SmartGlasses Challenge:
  Benchmarking Egocentric Multi-Talker Speech Recognition and Understanding with
  Audio-Language Models"), https://attend.ieee.org/slt-2026/grand-challenges/ ,
  official site https://aslp-lab.github.io/SmartGlasses/ , and a system-description
  paper https://arxiv.org/pdf/2607.17867 .
- License terms confirmed from the official challenge site: dataset use is
  **"strictly limited to non-commercial academic research,"** explicitly
  restricted to "participating in the IEEE SLT 2026 SmartGlasses Challenge and
  subsequent related academic research after the challenge concludes." Commercial
  use, product development, and redistribution to unregistered parties are
  explicitly prohibited. Access requires team registration (Google Form / Tencent
  Form) and agreeing to challenge rules; IP remains with the organizing
  committee.
- **Verdict: real dataset, real license, matches the brief's claim — but it is
  non-commercial-academic-research-only, gated behind challenge registration this
  session cannot complete, and is Mandarin-language egocentric-conversation data,
  not Mentra-device data. Confirmed usable only for a human-driven, registered,
  non-commercial academic use — not usable for this task right now. DO NOT
  DOWNLOAD without the user completing registration themselves.**

### 5. NasoVoce
**Existence VERIFIED. License/availability terms COULD NOT VERIFY.**
- Existence: real. *"NasoVoce: A Nose-Mounted Low-Audibility Speech Interface for
  Always-Available Speech Interaction,"* CHI 2026,
  https://dl.acm.org/doi/10.1145/3772318.3791397 , preprint
  https://arxiv.org/abs/2603.10324 . Confirmed: 45 participants, gender-balanced,
  ages 25-55, ~2.3 hours each (~104 hours total), paired MEMS-microphone +
  vibration-sensor signals, noise-augmented with DEMAND at -10 to +10 dB RMS.
- Relevance caveat: this is a **nose-bridge contact/vibration sensor** interface,
  not a smart-glasses air-microphone geometry dataset — it measures a physically
  different signal (bone/tissue vibration) than what GeoWearNet's air-microphone
  hypothesis needs.
- No public download link, license, or availability statement was found in this
  session's search (ACM/arXiv listing only; no GitHub/HuggingFace release located).
- **Verdict: UNKNOWN LICENSE — DO NOT USE.** Even if it were licensable, its
  sensing modality does not match GeoWearNet's air-microphone assumption, so it
  would need a separate feasibility judgment even if access were resolved.

## Repo-internal datasets (for contrast — real, in-repo, already used by this project)

| Dataset | License | Verified how |
|---|---|---|
| LibriSpeech train-clean-100 | CC BY 4.0 (per `evaluation/data/raw/LibriSpeech/LICENSE.TXT`, OpenSLR-12) | Already in repo; manifest at `evaluation/manifests/librispeech_train_clean_100_train.json` states "OpenSLR-12, CC BY 4.0" |
| MUSAN | Free to use for research (OpenSLR-17); see `evaluation/data/raw/musan/README` | Already in repo |

## Bottom line for this task

No externally sourced device-relative (glasses-geometry) audio dataset is both
(a) verifiably real, (b) verifiably license-clear for this project's use, and
(c) actually downloadable by this session. MMCSG's license is unresolved/likely
non-commercial and requires an account step; the SLT 2026 SmartGlasses Challenge
data is real but explicitly non-commercial-academic-only and gated behind human
registration; NasoVoce is a different sensing modality with no located license.
**This confirms Phase 2/21's expected honest state: only LibriSpeech-based
simulation is available in this repo for E1 work in this execution.** This is
reported as `BLOCKED_ON_DEVICE_RELATIVE_DATA` for E1-REAL; see the final report
for the resulting `PROCEED_TO_E1_SIM` decision.
