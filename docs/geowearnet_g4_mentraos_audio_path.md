# MentraOS audio path — what is actually documented (G4 WS4 / WS5)

Status: **DESK RESEARCH ONLY — NOT VERIFIED AGAINST A DEVICE**

Every statement below is sourced from public documentation or public
repositories, fetched 2026-08-27. Nothing here was confirmed by running code
on Mentra hardware, because no Mentra Live device exists in this development
environment. Where the public documentation is silent, this file says
**UNSPECIFIED** rather than guessing — a guessed sample rate is exactly the
kind of thing that quietly invalidates a domain-gap analysis six weeks later.

No proprietary Bluetooth behaviour was reverse-engineered. Only published
docs and public repos were read.

## Sources

| # | Source | URL |
|---|---|---|
| S1 | MentraOS docs — Audio chunks (app/cloud SDK) | https://docs.mentraglass.com/app-devs/core-concepts/microphone/audio-chunks |
| S2 | MentraOS docs — Audio manager (`session.speaker`) | https://docs.mentraglass.com/app-devs/reference/managers/audio-manager |
| S3 | Mentra Bluetooth SDK Starter Kit (repo) | https://github.com/Mentra-Community/Mentra-Bluetooth-SDK-Starter-Kit |
| S4 | Edge_AI_SmartGlasses — `docs/audio-guide.md` | https://github.com/Mentra-Community/Edge_AI_SmartGlasses/blob/main/docs/audio-guide.md |
| S5 | Mentra blog — "Making Mentra Live" | https://mentraglass.com/blogs/blog/making-mentra-live |

## The two audio paths

The public material describes **two distinct ways** an application can reach
glasses microphone audio. They have different processing chains, and which one
a pilot uses materially changes what the captured audio is.

### Path A — cloud/app SDK (`session.mic`)

Glasses mic → phone (MentraOS app) → MentraOS cloud → app backend (S1).

* Delivery: `session.mic.onAudioChunk()`; a separate
  `session.mic.onVoiceActivity()` gives VAD status only, no audio (S1).
* Payload: **"Base64-encoded audio. PCM or LC3 depending on the phone's mic
  mode"** (S1). So the encoding is not fixed by the API — it depends on the
  phone-side configuration at runtime.
* Sample rate: the `AudioChunkData` object carries a sample rate **"when
  reported"** — i.e. it is optional and may be absent (S1).
* Permission: `MICROPHONE` must be declared in the miniapp manifest (S1).
* Channels, bit depth, chunk cadence, and any AGC/noise-suppression stage:
  **UNSPECIFIED** in the documentation.

### Path B — Bluetooth SDK on the phone (`mic_pcm` / `mic_lc3`)

Glasses mic → BLE → an Android/iOS/React-Native app on the phone (S3, S4).
This is the shorter chain and therefore the *least processed* audio an
application is documented to be able to reach.

* Raw callbacks: `onMicPcm` / `'mic_pcm'` (PCM frames) and `onMicLc3` /
  `'mic_lc3'` (LC3 compressed frames) (S4).
* Payload type: `ByteArray` (Android), `Data` (iOS), `event.pcm` /
  `event.lc3` (React Native) (S4).
* Control: `setMicState(enabled, useGlassesMic, bypassVad)` and
  `setPreferredMic(...)` with an `AUTO` mode (S4). `useGlassesMic` selects the
  glasses microphone over the phone's; `bypassVad` controls whether voice
  activity detection gates the stream — **for a research capture this should
  be set to bypass VAD**, otherwise the corpus silently loses exactly the
  low-level and onset audio a first-word study cares about.
* Local transcription is available separately (`LocalTranscriptionEvent` with
  `text` / `isFinal`) (S4).
* Sample rate, bit depth, channel count, base64-vs-binary framing:
  **the audio guide explicitly does not specify them** (S4, verified by
  reading the raw file).

> **Unverified claim, recorded as such.** A search-engine summary of S3
> asserted `mic_pcm` delivers "base64 pcm_s16le at 16 kHz, 16-bit, mono".
> That string was **not** found in the documentation actually read (S4), so
> this project treats 16 kHz/16-bit/mono as an UNCONFIRMED hypothesis to be
> measured on a real device, not as a specification. This distinction matters:
> if the true native rate is 48 kHz, every capture that assumes 16 kHz is
> silently resampled and the domain-gap analysis is measuring the resampler.

### Not an input path — `session.speaker` (AudioManager)

S2 documents `session.speaker` as **output only**: URL playback, TTS, and live
PCM streaming *to* the device (signed 16-bit little-endian mono, 16000 /
24000 / 48000 Hz, default 16000). It explicitly redirects to `session.mic`
for microphone input. Any plan that treats "AudioManager" as a capture API is
reading the wrong page.

## Mentra Live hardware, as publicly described (S5)

* Processor: MTK 8766. Camera: 12 MP. WiFi 5 GHz (1080p streaming).
  Bluetooth classic audio plus BLE GATT.
* **Multiple microphones.** The blog describes one microphone "set aside" for
  a dedicated channel that is compressed with **LC3** and sent directly to the
  Mentra app, running *simultaneously* with the standard Bluetooth audio path
  used for calls and media.
* Consequence for this project: the always-on app-facing audio is described as
  an **LC3-compressed, dedicated single channel**, not the raw multi-mic array.

## Multi-microphone question (answers G4's MULTI-MIC item)

**No public documentation exposes multi-channel or per-microphone audio to an
application.** S5 confirms multiple microphones physically exist; S3/S4 expose
a single mic stream plus `setPreferredMic()` to choose a *source*, not to
receive several. GeoWearNet's audio-only, single-channel design is therefore
the right shape for this device on current public evidence — but the G2 finding
that multichannel spatial features added +0.05–0.06 AUROC on MMCSG's 7-channel
Aria audio **cannot be transferred to Mentra Live** unless a multi-channel path
turns out to exist. Verifying that requires the device and the SDK.

## Rawest available audio — the target capture spec (WS5)

This is what a capture tool should aim to record, in priority order. Fields
marked *(measure)* cannot be filled in from documentation and must be read off
a real device during the first pilot.

| Field | Target | Status |
|---|---|---|
| Path | Bluetooth SDK `mic_pcm` on the phone (Path B) | preferred; shortest chain |
| Fallback path | `session.mic.onAudioChunk()` (Path A) | more processing stages |
| Encoding | uncompressed PCM (`mic_pcm`), never the LC3 branch if PCM is offered | LC3 is lossy; decoding it adds a codec artefact to every sample |
| Sample rate | *(measure)* — record what the API reports; do not assume 16 kHz | UNSPECIFIED in docs |
| Bit depth | *(measure)*, expected signed 16-bit | UNSPECIFIED |
| Channels | *(measure)*, expected mono | UNSPECIFIED |
| `useGlassesMic` | **true** — the glasses mic, not the phone's | documented (S4) |
| `bypassVad` | **true** — VAD gating would delete onsets | documented (S4) |
| `setPreferredMic` | record the resolved value | documented (S4) |
| AGC / noise suppression / DSP | *(measure)* — compare a fixed played-back stimulus at two levels | UNSPECIFIED anywhere |
| Glasses firmware version | record | operator reads it from the app |
| MentraOS app version | record | operator reads it from the app |
| SDK version | record | from the build |
| Phone model + OS version | record | affects Path A encoding |

**AGC detection procedure** (since no documentation answers it): play a fixed
tone or speech file through a speaker at two known levels differing by ~20 dB,
capture both, and compare the ratio of measured RMS to the ratio of the
played levels. A ratio materially below 1:1 means a compressor/AGC sits in the
chain, which would break GeoWearNet's amplitude-sensitive physical features.
This is a five-minute test and belongs in the first pilot session.

## What is BLOCKED and exactly why

* **BLOCKED_NO_MENTRA_HARDWARE** — every *(measure)* row above. Requires a
  physical Mentra Live plus a paired phone.
* **NOT BLOCKED** — SDK documentation access. The docs at
  `docs.mentraglass.com` and the Mentra-Community repositories were read
  without credentials. No account, registration, or API key was needed for
  anything in this document.
* **PARTIALLY BLOCKED** — running the Bluetooth SDK (Path B) requires building
  an Android/iOS/React-Native app and pairing it with a device. That is a real
  engineering task, not a documentation gap, and it is only worth doing if the
  browser capture path (below) proves insufficient.

## How this project actually captures audio today

This project does **not** need Path A or Path B to run a pilot. It already has
a working, deployed, tested chain that is device-agnostic:

    input device paired to the OPERATOR's own machine
      -> browser: getUserMedia + AudioWorklet at the native rate
         (web/app/composables/useMicCapture.ts — AGC/NS/EC requested off and
          the GRANTED settings read back from the track, never assumed)
      -> MTRA binary WebSocket protocol
      -> server/audio/remote_receiver.py
      -> mentra/capture/ (raw native-rate WAV + derived 16 kHz WAV +
         metadata + automatic validation)

The `web/app/pages/capture.vue` research-capture mode built in G4 WS6 records
exactly that. Whether the Mentra Live microphone reaches the browser as a
selectable input device on the operator's machine depends on how that machine
exposes the paired glasses (Bluetooth headset/HFP profile), and **that is the
one thing that must be checked with the device in hand.** If it does, a pilot
can start immediately with no SDK work at all; if it does not, Path B and a
phone-side app become necessary.

Note the tradeoff if the browser route works: the classic Bluetooth headset
profile is likely to apply its own processing and a narrower band than the
dedicated LC3 app channel. That would be a *different* audio domain from what
a shipped MentraOS app would receive — usable for a first smoke test, but the
capture metadata must record the route so the two are never pooled.
