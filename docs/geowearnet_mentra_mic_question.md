# Technical note — Mentra Live microphone access

**Workstream AE.** For the Mentra hardware/firmware/SDK owner. This is a
question note, not a request for work. **No GeoWearNet work is blocked on the
answer** — the entire E1 track is single-channel mono by design, because
Mentra Live's channel layout is currently unknown to us. A positive answer
would open a second, stronger research direction; a negative answer changes
nothing about the current plan.

> **Update (G4 WS4, 2026-08-27) — partially answered from public sources.**
> Mentra Live *is* publicly described as having multiple microphones, with one
> "set aside" for a dedicated channel that is LC3-compressed and sent directly
> to the Mentra app, running alongside the standard Bluetooth audio path
> ([Making Mentra Live](https://mentraglass.com/blogs/blog/making-mentra-live)).
> However, **no public API exposes multi-channel or per-microphone audio**: the
> SDK offers a single mic stream (`mic_pcm` / `mic_lc3`) plus
> `setPreferredMic()` to choose a *source*, not to receive several. Sample
> rate, bit depth and channel count are UNSPECIFIED in the documentation.
> Full sourced write-up: `docs/geowearnet_g4_mentraos_audio_path.md`.
> The question below therefore still stands for the hardware/firmware owner —
> what is now known is that the answer is not in the public docs, and that the
> single-channel design remains the correct assumption until someone with the
> device says otherwise.

---

## The primary question

> **Can Mentra Live expose the individual raw microphone channels before mono
> mixing / beamforming / DSP?**

---

## If — and only if — multiple microphones exist and access is possible

These follow-ups matter, roughly in order of impact on what we could build:

1. **Sample rate per channel.** Is every channel captured at the same rate?
   What is it natively, before any resampling the SDK performs?
2. **Synchronisation.** Are the channels sample-synchronous (same ADC clock)?
   If not, what is the worst-case inter-channel skew, and is it constant or
   drifting? Sub-sample inter-channel timing is the single most valuable
   property for device-relative source localisation — it is what makes
   "where did this sound come from" tractable at all.
3. **DSP applied before SDK exposure.** What processing sits between the ADC
   and the buffer we receive? Specifically: high-pass filtering, noise
   suppression, echo cancellation, dereverberation, spectral enhancement.
4. **AGC.** Is automatic gain control applied per channel independently, or as
   a single shared gain across all channels? *Shared* AGC preserves
   inter-channel level ratios and is far better for our purposes; *per-channel*
   AGC destroys exactly the cue we would want.
5. **Beamforming / mixing stage.** If a mono stream is produced, is it a fixed
   linear mix, an adaptive beamformer, or channel selection? Is the mono stream
   derived from the raw channels we would receive, or from a separately
   processed path?
6. **Channel geometry.** Physical positions of the microphones relative to the
   frame — spacing in millimetres and orientation. Even approximate values
   (e.g. "two mics, ~110 mm apart, both on the right temple") would be enough
   to constrain the acoustic model.

---

## Why we are asking

GeoWearNet tests whether wearer speech can be distinguished from environment
speech by **how the sound reached the glasses** rather than by **who is
speaking** — no enrollment, no voice profile. In simulation the usable
single-channel cues are the direct-to-reverberant ratio and near-field head
diffraction. Both survive on one microphone, which is why the current work is
mono.

Multiple synchronous raw channels would add inter-channel time and level
differences, which are a far more direct measurement of source position. That
is a materially different and stronger problem setup.

**Two properties would matter more than the channel count itself:** shared
(not per-channel) AGC, and sample-synchronous capture. A pair of microphones
with independent AGC and unspecified skew is worth much less than a single
clean raw channel.

---

## Update — measured evidence from real MMCSG smart-glasses audio (G2 campaign, 2026-08-27)

The question above was written from simulation alone. G2 has since run the actual
measurement on real Aria smart-glasses audio (MMCSG corpus, 7 raw microphone
channels, train-derived validation split, never tuned on official dev):

| signal | AUROC (SELF-vs-OTHER, speech-active frames) |
|---|---|
| best single raw microphone (E0 physical features) | **0.834** |
| naive 7-mic average (single derived waveform) | 0.792 (**worse** than best single mic) |
| simple spatial features only (inter-channel level differences + short-lag cross-correlation, no beamforming) | **0.886** (+0.053 over best single mic) |
| spatial features + best single channel combined | **0.894** (+0.060 over best single mic) |
| official Meta 13-beam / mouth-directed-beam frontend | **not reproducible** — the downloaded corpus does not include Meta's published per-beam filter-sum weights, real Aria RIRs, or the MCAC simulator resources, and this campaign does not fabricate them |

**Reading of this evidence:** raw multi-mic access is worth something concrete —
a ~5-6 AUROC-point gain from dirt-cheap spatial features (no beamforming, no
model, just per-frame level-difference and cross-correlation scalars fed to a
linear classifier) — but it is a **moderate**, not dramatic, gain. It does not
by itself justify treating raw-channel access as an urgent blocker; it is a
legitimate secondary research enabler. A caveat: this measures Aria's specific
7-mic geometry, not Mentra's (unknown) microphone count/placement, so the
*actual* Mentra number could differ in either direction — recorded honestly as
what MMCSG evidence can and cannot tell us here (see `docs/geowearnet_g2_mmcsg_report.md`
for the full writeup and methodology).

This still does not block anything: the deployed GeoWearNet path remains
single-channel by design, and the zero-shot/real-training results (also in the
G2 report) show a single raw channel is already strong on its own.

## What we need in the meantime, regardless of the answer

The mono path already works end to end in the production console. For the real
Mentra pilot (Workstreams AA–AD) what actually matters is:

- raw PCM at the device's native rate, archived before any normalisation;
- the **actual** applied capture settings (`autoGainControl`,
  `noiseSuppression`, `echoCancellation`), read back and recorded rather than
  assumed — the web console already reads these from `track.getSettings()`;
- confirmation of whether the OS or browser forces AGC on despite our
  requesting it off. If it does, that is not a blocker: it becomes a documented
  covariate we train against in the simulator's device chain.
