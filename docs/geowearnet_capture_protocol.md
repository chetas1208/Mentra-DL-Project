# Mentra Pilot Capture Protocol — informed by GeoWearNet G2 (MMCSG)

> **Update (G4, 2026-08-27): the tooling this protocol assumed now exists.**
> A research capture mode was built and self-tested end to end in G4 WS6 —
> `web/app/pages/capture.vue` (client) plus `mentra/capture/` and
> `scripts/mentra/run_receiver.py --capture-dir` (server). It records the raw
> native-rate stream, the derived 16 kHz stream, session/wearer/condition
> metadata and an automatic validation report per take, and it refuses to let
> non-Mentra audio be labelled as Mentra evidence. Read
> `docs/geowearnet_g4_mentra_report.md` (the "Next action" and
> BLOCKED_NO_MENTRA_HARDWARE sections) and
> `docs/geowearnet_g4_mentraos_audio_path.md` (the AGC check and the
> `bypassVad` / `useGlassesMic` settings) alongside this document before
> running a session. This file remains the sizing rationale; those two are the
> operational instructions.

**Status: proposal, not yet executed.** No Mentra hardware capture has happened.
Everything below is a recommendation derived from measured MMCSG (Aria smart
glasses) evidence in `docs/geowearnet_g2_mmcsg_report.md`. MMCSG is a real
*wearable-domain proxy*, not Mentra hardware — treat every number here as a
sizing estimate, not a guarantee that Mentra will behave identically.

## Why a pilot, and why now

G2's two headline findings jointly motivate a **small, fast** Mentra pilot
rather than either "skip real data entirely" or "collect a huge corpus first":

1. **Zero-shot transfer from pure simulation to real Aria audio is strong**
   (speech-active SELF-vs-OTHER AUROC up to ~0.97 on official MMCSG dev, zero
   real training data). This means a Mentra pilot does not need to bootstrap
   GeoWearNet from nothing — a simulation-pretrained checkpoint is a
   reasonable starting point to fine-tune, or even to deploy zero-shot as a
   first pass while pilot data is being collected.
2. **Simulation pretraining measurably helps real-data sample efficiency**,
   especially in the first 5-15 minutes of real per-condition data (large
   gaps in overlap-F1 and false-wearer-rate-on-environment-only at low
   budgets), and the real-MMCSG scaling curve is **largely flat from ~60
   minutes of total real audio onward** (AUROC 0.90-0.91 from 15m through
   "full" ≈ 6-8h; overlap-F1 keeps improving somewhat through ~60m then
   plateaus). This means a pilot sized in the tens of minutes-per-subject
   range, across a reasonable number of subjects, is likely to already show
   whether Aria→Mentra domain transfer works — a multi-hour-per-subject
   collection is not obviously justified by the MMCSG curve alone.

## G4-PILOT-V1 (frozen 2026-08-27) — the actual first pilot, supersedes sizing below for pilot #1 only

Decided by the project owner after G4's runtime findings (live first-word
retention was 19.74% before a fix, 68.42% after a 150ms pre-roll — see
`docs/geowearnet_g4_mentra_report.md`). This section is the one to follow for
the *first* session; "Recommended minimum pilot" below remains the target for
scaling up afterward if this one is promising. **Do not improvise once
hardware arrives — this is frozen, not a starting point for negotiation.**

**Participants: 5 people, not 20.** The only question pilot #1 answers is
whether Aria/MMCSG-trained GeoWearNet transfers to Mentra *at all*. If yes,
scale to 10-20 per the section below. Starting bigger than 5 for a
zero-shot-transfer question is not justified by anything measured yet.

**Per person: ~7-10 minutes**, ~40-50 minutes total for the pilot. Condition
breakdown (approximate, adjust live rather than rigidly enforcing to the
second):

| Condition | Approx. duration |
|---|---:|
| Wearer normal | 60s |
| Wearer soft | 45s |
| Wearer loud | 30s |
| Wearer near-whisper | 30s |
| Bystander normal | 60s |
| Bystander close/loud | 60s |
| Wearer + bystander alternating | 60s |
| Real overlap (both talking at once) | 60s |
| Machinery/noise + wearer | 60s |
| Machinery + bystander | 45s |
| Movement/head turning | 30-60s |

**Mandatory: the shared-glasses swap sequence**, run at least once, ideally
between every pair of consecutive subjects:

```
Person A wears glasses -> speaks -> session reset
Person B wears SAME glasses -> speaks -> session reset
Person C ...
```

No enrollment at any point. This directly tests the actual product
hypothesis (device-relative detection, not identity), not just per-person
accuracy — see `docs/geowearnet_product_architecture.md` / the
`project_mentra_product_north_star` memory note for why this is the point of
the whole GeoWearNet track, not an afterthought.

### AGC/compression check — run this FIRST, before collecting condition data

Five minutes, before the 5-person pilot proper. Per
`docs/geowearnet_g4_mentraos_audio_path.md`'s "AGC detection procedure":
play a fixed sound source (tone or speech clip) through a speaker at two
known playback levels **~15-20 dB apart**, through the exact production-
relevant capture path (`capture.vue` -> `run_receiver.py --capture-dir`),
and compare:

```
physical level change (~20 dB)   vs.   captured RMS change
```

Also record peak, crest factor, clipping fraction, and rough spectral shape
for both takes (the existing `mentra/capture/validation.py` numbers cover
most of this automatically — check its output rather than computing by
hand). Interpretation:

- **~20dB in -> ~20dB out**: no material AGC/compression in the path,
  amplitude-sensitive GeoWearNet features can be trusted as-is.
- **~20dB in -> a few dB out**: a compressor/AGC sits somewhere in the chain.
  This does **not** kill GeoWearNet — G2/G3 already show geometry isn't
  purely amplitude-based — but it changes which features to trust and what
  augmentation the eventual adaptation pass (if needed) should include.

Do not skip this and infer AGC behavior from folklore/forum claims — the
G4 desk research explicitly found the "16kHz/16-bit/mono, no AGC" claims
floating around are **not actually in the documentation read**, only in
secondhand summaries. Measure it.

### Zero-shot run, immediately after capture (no training)

Before touching any adaptation:

1. Run `training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt`
   (`g2_ablation_ctx1000`, TCN, 132,678 params) with **zero gradient
   updates** on the captured audio.
2. Tag the result artifact permanently as `MENTRA_ZERO_SHOT_PRE_ADAPTATION`
   — never overwrite it, exactly like G2's `ZERO_SHOT_PRE_MMCSG_TRAINING`
   tag on the official MMCSG dev look.
3. Only test the **two runtime configs G4 already validated on internal
   data**, not a fresh sweep on 5 users:
   - 200ms detector cadence + 0ms pre-roll (current shipped default)
   - 100ms detector cadence + 150ms pre-roll (G4's selected config,
     68.42%→81.58% first-250ms retention range on internal MMCSG data)
   The point of pilot #1 is verifying whether the *existing* runtime
   conclusion transfers to Mentra, not re-deriving it from 5 people.

### First product comparison — keep it small

Only these four, not another wide matrix:

```
RAW
RNNoise
GeoWear (100ms cadence + 150ms pre-roll)
RNNoise + GeoWear (100ms cadence + 150ms pre-roll)
```

Report per pipeline: wearer speech retained, bystander leakage rate, false
agent command rate (see the Command script section below — this is the
headline product metric, not another AUROC number). If GeoWear gets
anywhere near MMCSG's measured 62%→7% leakage reduction on real Mentra
audio, that is a major result on its own; don't dilute the first read with
a 30-condition table.

### Command script — real commands, not generic speech

Wearer says (realistic autobody-shop phrasing, reuse across takes):
- "Hey glasses, what part is this?"
- "Do we have a replacement in stock?"
- "What is the torque specification?"
- "Show me the repair instructions."

Bystander deliberately says competing/dangerous phrases at the same time or
right after:
- "Hey glasses."
- "Cancel that."
- "Call John."
- "Open inventory."
- "Stop."

A bystander's stray leaked noun is an annoyance; a bystander's phrase
triggering the agent is a product failure. Track these separately:
**False Agent Command Rate** (bystander phrase interpreted as a real
command) is the headline metric here, distinct from raw token-level
bystander leakage.

### Decision after pilot #1

One of: `MENTRA_ZERO_SHOT_STRONG` / `MENTRA_ZERO_SHOT_PROMISING` /
`MENTRA_ZERO_SHOT_PARTIAL` / `MENTRA_DOMAIN_GAP` (predeclared, don't move the
goalposts after seeing results). Strong or promising -> proceed toward the
10-20 subject pilot below and shipping hardening, no fine-tuning just
because it's possible. Partial/domain-gap -> run the domain-gap autopsy
(compare raw acoustic distributions: RMS/peak/crest/clipping/spectral
tilt/LF-HF ratio between MMCSG and Mentra captures) before considering a
bigger model or more data. **Separator (WearerSepNet) stays closed** unless
pilot #1's overlap-condition results show overlap is a dominant source of
remaining leakage — G3's MMCSG-domain number was 6.90% of oracle residual
leakage, nowhere near the ~50% that would justify reopening that gate; this
must be re-measured on real Mentra data, not assumed to carry over.

---

## Recommended minimum pilot

- **Subjects:** at least **10**, ideally 15-20. MMCSG's own scaling curve is
  measured across MANY distinct wearers even at small time budgets (each
  MMCSG recording is one wearer/one conversation, ~3 minutes on average), so
  "60 minutes of real data" in the G2 curve already means roughly 20 distinct
  people, not one person for an hour. Breadth across wearers (voice,
  physiology, head/glasses fit) is the axis GeoWearNet's core hypothesis is
  actually about — prioritize more subjects over more minutes per subject.
- **Minutes per subject:** **10-20 minutes** of natural conversational audio
  per subject as a starting target, informed by the point where MMCSG's
  per-budget metrics stop improving sharply (roughly the 15-30 minute mark on
  the pooled curve). This is intentionally on the small side so the pilot can
  run soon; scale up only if held-out results are still improving at this size.
- **Total target:** roughly **3-6 hours** of real Mentra audio across all
  subjects for the first pilot — enough to populate a scratch-vs-pretrained
  comparison and a coarse channel/context check, not a production-scale corpus.

## Conditions to include (per subject, not just once)

Modeled on MMCSG's own condition coverage plus GeoWearNet's own simulated
adversarial suite (`training/geowearnet/evalsuite.py`):

- **Wearer speech**, normal conversational level.
- **Wearer speech, soft/quiet** (tests the amplitude-shortcut risk found in
  G2's anti-loudness audit — real E0 turned out to be more loudness-dependent
  than the simulator predicted, so under-representing quiet wearer speech in
  a pilot would hide a real failure mode).
- **Bystander/environment speech, normal level.**
- **Bystander/environment speech, LOUD** (shouting-bystander-style condition;
  G2's hard-OTHER audit shows E0's false-accept rate on OTHER speech at or
  above the wearer's own typical loudness is 97-100% — this is exactly the
  condition a real pilot must stress-test).
- **Overlapping speech** (both wearer and bystander talking at once) — real
  MMCSG shows ~11% of conversation time is genuine overlap; a pilot with zero
  overlap coverage would silently drop the hardest, most realistic case.
- **Movement** (walking, head turning) — untested by MMCSG (its recordings
  are seated/stationary conversations) and untested by the current simulator;
  this is a genuine coverage gap the pilot should close.
- **Multiple rooms/environments** — at least 2-3 acoustically distinct spaces
  per subject if feasible (quiet room, moderately reverberant room, and
  outdoors or a noisy space), to probe RT60/noise generalisation the way
  G1's simulated `high_rt60`/`noisy` suites did synthetically.

## What must be captured alongside audio

- Raw PCM at Mentra's native capture rate, archived before any on-device
  normalisation, AGC, or noise suppression the device/OS applies (see
  `docs/geowearnet_mentra_mic_question.md` for the exact settings to record).
- Ground-truth segment timing for wearer vs. bystander speech (a simple
  push-to-mark protocol or post-hoc annotation is sufficient at this scale —
  MMCSG's own RTTM-style segment format is a reasonable template, see
  `training/geowearnet/mmcsg/labels.py`).
- Participant/session identifiers, kept separate from audio, so a
  person-disjoint train/val/test split (mirroring `training/geowearnet/mmcsg/splits.py`)
  can be built from day one — do not repeat the mistake of only discovering
  the need for a disjoint split after data collection.

## What the pilot should answer first

In priority order, directly mirroring the three headline G2 questions:

1. Does a G1/G2 simulation-pretrained checkpoint transfer zero-shot to
   Mentra audio at all (even roughly)? This requires zero new training and
   can be checked the moment the first few minutes of pilot audio exist.
2. Does a short (10-20 minute) per-subject fine-tune on real Mentra audio,
   starting from the MMCSG/simulation-pretrained checkpoint, reach usable
   accuracy? Compare against the same amount of data trained from scratch.
3. Is Mentra's actual single microphone behaving like MMCSG's best channel,
   a middle channel, or its worst channel, in relative (not absolute) terms —
   i.e., does the channel-selection methodology in `training/geowearnet/mmcsg/e0_census.py`
   even apply if Mentra is single-mic (no channel selection needed), or does
   it turn out Mentra exposes more than one raw channel (see the multi-mic
   question doc)?

---

# P1.17 — REAL MENTRA PRODUCT BENCH (customer conditions)

**Added 2026-08-27 by the P1 product-evaluation lane.** Everything above sizes
a pilot around the *research* question (does sim→Mentra transfer work).
This section adds the *product* question: **does the voice agent hear the
technician correctly, in the places Mentra is actually going to be worn.**

The two are not the same capture. The section above is conversational audio
sufficient to fine-tune a detector. This section is a **product bench**: a
fixed, repeatable set of conditions that the pipeline in
`docs/geowearnet_product_architecture.md` is scored on end-to-end, using the
metrics in `evaluation/agent_audio/` (wearer WER/TER, **bystander leakage
rate**, wearer command retention, false agent command rate).

## MODALITY LOCK — audio only

> **Camera OFF. IMU OFF. Audio only.**

This is a hard constraint on the capture, not a default that can be relaxed
for convenience:

- **Camera is OFF** for the entire always-on path. The camera remains an
  explicit agent *tool* — "hey glasses, what part is this?" triggers an
  intentional, user-initiated photo — and is never a continuous background
  signal used to decide who is speaking. Do not record video "just in case";
  a recorded video stream invites a future model to quietly depend on it and
  destroys the audio-only claim.
- **IMU is OFF** as a model input. MMCSG happens to ship IMU data and this
  project deliberately does not use it, specifically to keep the
  production-relevant branch audio-only. The Mentra bench must not introduce
  an IMU dependency that MMCSG work was careful to avoid.
- **No enrollment.** Nothing in this capture may require a per-wearer voice
  enrollment, account switch, or "say this phrase to calibrate" step. The
  glasses are shared between workers on a shift; any protocol step that
  assumes a known wearer invalidates the deployment model.

If a session accidentally captures camera or IMU data, that session's
non-audio streams must be deleted before the data enters the evaluation set,
and the deletion recorded.

## Target environments (first Mentra pilot)

Four environments, in priority order. The first is the reference customer
scenario; the rest test whether the result generalises across the noise
*character*, not just the noise *level*.

| # | Environment | Dominant interference | Why it is on the list |
|---|---|---|---|
| 1 | **Autobody shop** | impact wrenches, air compressors, grinders, shop radio, coworkers | the reference customer scenario; impulsive + continuous + music + competing speech all at once |
| 2 | **Warehouse** | forklifts, conveyor hum, PA announcements, distant shouted speech | large reverberant volume; PA speech is *speech-like interference from a loudspeaker*, a distinct failure mode from a nearby human |
| 3 | **Construction** | intermittent high-SPL impact, generators, outdoor wind | highest absolute levels; wind noise on a head-worn mic is its own problem |
| 4 | **Restaurant / retail** | dense multi-talker babble, music, clatter | the hardest *competing speech* case: many simultaneous bystanders, none of them loud individually |

## Conditions to capture, per environment

Each condition is a separately-marked take so it can be scored independently.
The first six mirror the synthetic scenarios in
`evaluation/agent_audio/stressbench.py`, so synthetic and real results are
directly comparable — that comparison is the *only* way to find out how much
the synthetic bench was worth.

**Wearer speech level (3 takes)**
1. **Quiet wearer** — technician speaking softly, head down, close work.
2. **Normal wearer** — ordinary conversational level.
3. **Loud wearer** — raised voice over machinery (Lombard effect is real and
   changes spectrum, not just level; a level-only simulation will not
   reproduce it).

**Competing speech (3 takes)**
4. **Nearby coworker** — one bystander at ~1-2 m, normal level, taking turns
   with the wearer.
5. **Loud coworker** — bystander at or above the wearer's level. G2's
   hard-OTHER audit puts E0's false-accept rate at 97-100% in this condition;
   it is the single most likely source of bystander leakage.
6. **Overlap** — wearer and coworker speaking simultaneously, deliberately,
   for at least 30 s of genuine overlap per environment. This is the state a
   gain-only router provably cannot solve (see P1.6 in
   `docs/geowearnet_product_architecture.md`), so it must be measured, not
   avoided.

**Machinery / background (3 takes)**
7. **Continuous machinery** — compressor, HVAC, conveyor: stationary noise,
   the case classic denoisers handle well.
8. **Impulsive machinery** — impact wrench, hammering, dropped tools:
   non-stationary, the case classic denoisers handle badly and which can
   masquerade as speech onsets.
9. **Music / radio** — shop radio or PA music. Spectrally speech-like and
   continuous; it is the condition most likely to be mistaken for a bystander.

**Wearer/device dynamics (2 takes)**
10. **Movement** — walking the bay, turning the head, bending over a vehicle.
    Untested by MMCSG (seated) and by the current simulator.
11. **Multiple rooms** — at least two acoustically distinct spaces per site
    (e.g. open bay vs. parts room vs. outdoors), same wearer, same script.

## Command script (per take)

Every take must include at least three **agent commands** actually spoken by
the wearer, drawn from `evaluation/agent_audio/commands.py::COMMAND_SET`:

- "Hey glasses, what part is this?"
- "Hey glasses, do we have a replacement in stock?"
- "Show me the repair procedure."
- "Hey glasses, what is the torque spec for this bolt?"

And, in the competing-speech takes, at least two **command-shaped bystander
phrases** spoken by the coworker (`BYSTANDER_PHRASES`), e.g. the coworker
saying "hey glasses, order a new one". This is what makes **false agent
command rate** measurable on real audio. Without deliberately planting these,
the most expensive failure mode in the product cannot be scored at all.

This closes the one gap the synthetic bench cannot fill: no local corpus
contains a human saying these exact phrases, and no TTS model is present in
this environment, so `stressbench.py` currently measures command retention
using *phrase surrogates* drawn from real LibriSpeech transcripts. Real
capture of the real wording is the fix.

## Ground truth required for product scoring

Beyond the segment timing already listed above, the product metrics need:

- **Per-speaker reference transcripts** (wearer and bystander separately) —
  this is what `bystander_leakage_rate` and `wearer_wer` are computed
  against. MMCSG's word-level TSV (`start, end, word, speaker∈{0,1}`) is the
  template; `training/geowearnet/mmcsg/labels.py` already parses that format,
  so matching it means zero new parsing code.
- **A clean(-ish) wearer reference** where obtainable — e.g. a simultaneous
  close-talk lavalier on the wearer, recorded on a separate channel. This is
  optional but high-value: it is the only way to compute waveform-domain
  metrics (SI-SDR, STOI) on real data, and it is what a future WearerSepNet
  would be trained and validated against. If a lav is used it must be
  archived as a *reference*, never fed to the model.
- **Marked command timestamps**, so command retention can be scored without
  hand-alignment.

## Sizing

Per environment: 11 conditions × ~2 minutes = ~22 minutes per wearer.
With 3 wearers per environment and 4 environments, ≈ **4.5 hours** total.
That is the same order as the research pilot above and can share sessions
with it where the conditions overlap — but the *takes must be marked
separately*, because product scoring is per-condition and a single
unsegmented recording cannot be broken down after the fact.

## Explicit non-goals for this first pilot

- Not a production data-collection effort. Sized to answer the domain-transfer
  question fast, not to train a final shippable model.
- Not a substitute for eventual larger-scale collection if this pilot shows
  transfer works but real-Mentra accuracy is still climbing at this size —
  in that case, scale up using the same MMCSG-style scaling-curve methodology
  (`training/geowearnet/mmcsg/campaign_real.py`) rather than guessing a bigger
  number.
