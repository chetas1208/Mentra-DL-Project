# GeoWearNet G4 — Mentra convergence campaign

Status: **MENTRA_NOT_TESTED — SOFTWARE LANE COMPLETE**

G4 asked whether the frozen G2 detector works on real Mentra Live audio. That
question is **not answered here and could not be**, because no physical Mentra
Live, no paired phone and no human pilot subject exists in this development
environment. No corpus, synthetic, or browser-microphone audio was relabelled
as Mentra audio anywhere in this campaign — in code, in artifacts, or in this
document. `mentra.capture.session.assert_mentra_hardware` enforces that in
code, and a test asserts the refusal actually fires.

What G4 *did* complete is everything that did not need the device — and one of
those results materially changes what the project believed about its own live
path.

Machine-readable summary: `evaluation/geowearnet/g4/g4_final_summary.json`.

---

## Headline findings

1. **The live receiver was not running the router the product numbers
   described.** G3 flagged this as a known discrepancy. G4 measured its size:
   the shipped live path retained the first 250 ms of a wearer's first word
   only **19.74%** of the time, against the **59.21%** G3 reported. The
   published figure was measured on the offline dense probability source; the
   live receiver's own figure was three times worse and had never been
   measured.
2. **The gap is the detector's 200 ms cadence, not the envelope.** With
   identical probabilities the new live router is **bit-identical** to the
   offline evaluator (100% of items, gain MAE 0.0). Feeding the offline gate
   the live 200 ms held probabilities reproduces essentially the entire
   divergence on its own.
3. **A bounded pre-roll fixes most of it, and costs less leakage, not more.**
   150 ms of pre-roll takes live first-250 ms retention from 19.74% to
   **68.42%** while bystander word exposure *falls* from 9.07% to 7.93%.
4. **Combining a faster cadence with pre-roll nearly closes the gap.** Through
   the real live components, 100 ms cadence + 150 ms pre-roll reaches **81.58%**
   first-250 ms retention at a real-time factor of 0.120 — against 17.11% for
   the shipped 200 ms/no-pre-roll configuration on the same measurement.
5. **The capture tool is built and self-tested end to end** through the real
   transport and the real receiver — but on synthetic input, and it has never
   seen a Mentra device.

---

## Frozen parent

| | |
|---|---|
| Path | `training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt` |
| SHA-256 | `07c43c3d9e37dbd490ae0477f46ff515e1e19e8fe0a20687ec9d13a8555c68ef` (re-verified) |
| Architecture | TCN, 132,678 parameters, 1000 ms context |
| Role in G4 | `G4_ZERO_SHOT_PARENT` — **unmodified**. No G4 training was run. |

No new model id was registered. There is no `geowearnet_g4` checkpoint,
because there is no Mentra data to adapt to, and inventing the id would imply
a Mentra-adapted model that does not exist.

---

## WS1 — live routing equivalence

`server/audio/streaming_gate.py` introduces `StreamingGateRouter`, which
**wraps the offline evaluator's own `GeoWearGate`** rather than
reimplementing it, and drives it at 10 ms from a live PCM stream whose
probabilities arrive on a slower cadence. Two new audio policies —
`geowear_envelope` and `rnnoise_geowear_envelope` — expose it. The G3 binary
policies (`geowear_gate`, `rnnoise_geowear_gate`) are kept **unchanged** so
previously reported live behaviour stays reproducible, and the default
remains `passthrough`.

Equivalence is asserted, not assumed: with the same probabilities and zero
pre-roll the router's output is bit-identical to `GeoWearGate.apply()` for
every predeclared policy and every block size
(`tests/audio/test_streaming_gate.py`).

**Routing CPU cost** (`evaluation/geowearnet/g4/routing_cpu_benchmark.json`,
routing stage only, 60 s of audio):

| Policy | p50 / 10 ms frame | p95 | RTF |
|---|---:|---:|---:|
| `geowear_gate` (G3 binary) | 0.0027 ms | 0.0035 ms | 0.00027 |
| `geowear_envelope`, pre-roll 0 | 0.0286 ms | 0.0434 ms | 0.0030 |
| `geowear_envelope`, pre-roll 150 ms | 0.0293 ms | 0.0445 ms | 0.0030 |

The envelope costs about 10x the binary mute and is still 0.3% of real time.
Pre-roll adds 9.6 kB of buffer and no measurable compute.

---

## WS2 — first-word retention

38 GeoWearNet-internal MMCSG validation windows (12.67 minutes, official-train
derived, wearer-disjoint from G2 inner train). 76 first-word events, 54 of them
wake-word events, 1,059 bystander words. **These are gate-exposure proxies —
mean applied gain over labelled word spans — not ASR word error**, the same
measurement class G3 used, so the before/after numbers are comparable.

Artifact: `evaluation/geowearnet/g4/first_word_retention.json`.

### The reproduction, and the correction

| Probability source | policy `A_balanced`, pre-roll 0 | first-250 ms retention |
|---|---|---:|
| Offline dense (what G3 measured) | G3 reported 59.21% | **60.53%** (reproduced) |
| Offline dense, `E_no_smoothing` control | G3 reported 73.68% | **73.68%** (exact) |
| **Live 200 ms held (what the receiver does)** | never previously measured | **19.74%** |

The control reproduces exactly and the balanced policy to within 1.3 pp, so
this is the same measurement — applied to the path that actually ships.

### Pre-roll sweep, live probability source, `A_balanced`

| Pre-roll (ms) | first 100 ms | first 250 ms | first token | wake word | bystander exposure |
|---:|---:|---:|---:|---:|---:|
| 0 | 10.53% | 19.74% | 23.68% | 29.63% | 9.07% |
| 50 | 18.42% | 31.58% | 38.16% | 42.59% | 8.78% |
| 100 | 31.58% | 44.74% | 51.32% | 57.41% | 8.40% |
| **150** | **46.05%** | **68.42%** | **69.74%** | **77.78%** | **7.93%** |
| 200 | 71.05% | 76.32% | 77.63% | 79.63% | 7.08% |

> **Two live baselines appear in this report and they are not contradictory.**
> WS2's 19.74% uses the live 200 ms held signal *reconstructed* from the dense
> pass (validated to 4.8e-7 probability error, above). WS3's cadence sweep
> reports 17.11% for the same configuration measured through the *actual*
> `MentraInferenceConsumer`. The 2.6 pp gap is the residual cost of the
> reconstruction, and it is reported rather than harmonised away. Both agree
> the shipped path retains under a fifth of first words.

**Added algorithmic delay is exactly the pre-roll** and is reported as such;
it is a delay line, never lookahead — no sample is consulted before it would
have been available.

Bystander exposure *decreases* with pre-roll. That is not a free lunch, it is
a mechanism: shifting the envelope earlier also shifts the 200 ms hangover
tail earlier, so the gate closes sooner relative to a bystander's turn. It is
reported because it was measured, not because it flatters the change.

Varying the transition parameters instead of the pre-roll did almost nothing:
`A_fast_attack` (no 10 ms ramp) was identical to `A_balanced` at every
pre-roll, and `A_early_open` (ON threshold 0.60→0.50) gained ~1–2 pp of
retention while *increasing* leakage. **The onset problem is a timing problem,
not a threshold or ramp problem.**

### Pareto selection

The rule was declared before any result existed: maximise first-token
retention under the *live* probability source, subject to bystander exposure
rising no more than 2.0 pp above the `A_balanced`/pre-roll-0 baseline and
added delay ≤ 150 ms; `E_no_smoothing` excluded as a control.

**Selected: `A_balanced` + 150 ms pre-roll.** First-token retention +46.05 pp,
first-250 ms +48.68 pp, bystander exposure −1.13 pp, added delay 150 ms.
200 ms scored higher but violates the declared delay cap; the cap was not
relaxed after seeing the numbers.

**Nothing was flipped on.** The live default remains `passthrough`.

---

## WS3 — live/offline parity

The same 38 internal-val windows, decomposed so that "live differs from
offline" is attributable rather than merely true. Artifact:
`evaluation/geowearnet/g4/live_offline_parity.json`.

| Configuration | Bit-identical | Gate-state agreement | Gain MAE | Median onset delay |
|---|---:|---:|---:|---:|
| **B** dense probs + new router (isolates the router) | **100% (38/38)** | **1.0000** | **0.0** | 0 ms |
| **C** live 200 ms probs + offline gate (isolates cadence) | 2.6% | 0.9070 | 0.0428 | +130 ms |
| **D** real consumer + real frontend + real PCM16 (the live path) | 2.6% | 0.9044 | 0.0432 | +140 ms |

C ≈ D, so essentially all divergence is the detector cadence. PCM16
quantisation is reported separately at 48.2 dB error-SNR (max sample
difference 3.2e-5) so rounding is never mistaken for a routing difference.

One real, bounded convention difference was found while establishing B and is
excluded deliberately rather than silently: the feature extractor emits
slightly fewer frames than the audio contains, and over that trailing 20 ms
the offline `apply_gain_envelope` holds the last gain while the router keeps
advancing its release ramp. The comparison is made over the region where both
are defined; the divergence is confined to those final frames and is pinned by
`test_offline_tail_convention_differs_and_only_at_the_tail`.

The rolling-window shortcut used throughout was verified rather than assumed:
scoring a real 2 s rolling window and comparing its final frame against the
dense full-recording pass gave a max absolute probability difference of
**4.8e-7**, independently re-confirming G3's streaming-parity claim.

### Detector cadence × pre-roll, through the real live path

| Cadence | Pre-roll | State agreement | first-250 ms retention | Median onset delay | RTF (whole live path) |
|---:|---:|---:|---:|---:|---:|
| 200 ms (shipped) | 0 | 0.9044 | 17.11% | +140 ms | 0.065 |
| 200 ms | 100 ms | 0.9044 | 47.37% | +40 ms | 0.065 |
| 200 ms | 150 ms | 0.9044 | 63.16% | 0 ms | 0.065 |
| 100 ms | 0 | 0.9298 | 34.21% | +70 ms | 0.120 |
| **100 ms** | **150 ms** | **0.9298** | **81.58%** | −50 ms | **0.120** |
| 50 ms | 0 | 0.9479 | 42.11% | +40 ms | 0.222 |
| 50 ms | 150 ms | 0.9479 | 85.53% | −90 ms | 0.222 |

RTF is the whole live path (features, model, envelope, PCM16 encode) at
2 threads on this shared host. All configurations are comfortably real-time.
Cadence and pre-roll are complementary, not redundant: cadence improves *state
agreement* (0.9044 → 0.9479), pre-roll improves *which audio a correct state
lands on*. Only pre-roll can make the gate open before the onset.

**Recommended live configuration when a gate is enabled at all:** 100 ms
detector cadence with 150 ms pre-roll — 81.58% first-250 ms retention at RTF
0.120 and 150 ms of declared delay. This is a recommendation supported by
measurement, **not** a default that G4 changed.

---

## WS4 / WS5 — MentraOS audio path

Full write-up with sources: **`docs/geowearnet_g4_mentraos_audio_path.md`**.

- Public docs were readable **without any credentials** — SDK documentation
  access is *not* blocked.
- Two documented app-facing audio paths: the cloud SDK
  (`session.mic.onAudioChunk()`, "base64-encoded audio, PCM **or LC3**
  depending on the phone's mic mode", sample rate reported only "when
  reported") and the phone-side Bluetooth SDK (`mic_pcm` / `mic_lc3`, with
  `setMicState(enabled, useGlassesMic, bypassVad)`).
- `session.speaker` / AudioManager is **output only** and is not a capture API.
- **Sample rate, bit depth, channel count and any AGC/noise-suppression stage
  are UNSPECIFIED in the documentation.** A widely-repeated
  "16 kHz/16-bit/mono" claim was *not* found in the docs actually read and is
  recorded as an unconfirmed hypothesis to measure on a device.
- Mentra Live is publicly described as having multiple microphones with one
  dedicated to an LC3-compressed app channel — but **no public API exposes
  multi-channel audio**. G2's +0.05–0.06 AUROC from MMCSG multichannel spatial
  features must not be assumed to transfer.
- For a research capture: `useGlassesMic=true`, `bypassVad=true` (VAD gating
  would delete exactly the onsets WS2 is about), and prefer the PCM branch
  over LC3.

---

## WS6 — research capture

Built as an extension of the project's **existing, working, deployed** capture
chain, not as a parallel mechanism:

    input device paired to the OPERATOR's machine
      -> browser: getUserMedia + AudioWorklet at the NATIVE rate
         (AGC/NS/EC requested off, GRANTED settings read back from the track)
      -> MTRA binary WebSocket protocol
      -> server/audio/remote_receiver.py
      -> mentra/capture/

New: `web/app/pages/capture.vue` and `web/app/composables/useResearchCapture.ts`
(client), three additive protocol messages (`CAPTURE_META`,
`CAPTURE_RAW_PCM`, `CAPTURE_ACK`), a native-rate tap added to
`useMicCapture.ts` *before* the resampler, and `mentra/capture/`
(`session.py`, `validation.py`, `sink.py`) on the server. Enabled only by
`run_receiver.py --capture-dir`; without it the live path is unchanged.

Each take writes `raw.wav` (native rate), `derived_16k.wav` (exactly what the
model consumed), `metadata.json` and `validation.json`. Validation grades
readability, sample rate, channels, duration, RMS, peak, clipping, speech
activity against the recording's own noise floor, and **cross-stream duration
consistency** — the one failure mode invisible in either file alone. A failed
take is still written to disk, because a flagged file beats a vanished one
while the wearer is still in the room.

**Honesty is enforced in code.** `source_kind` is declared by the operator;
anything other than a Mentra device stamps `is_mentra_hardware_audio: false`,
and `assert_mentra_hardware()` refuses such a session outright.

### Self-test result

`python3 scripts/mentra/capture_selftest.py` runs a real WebSocket client
against a real `MentraRemoteReceiver` and the real `CaptureSink` — the same
object `run_receiver.py` uses — on an isolated port:

```
validation: PASS (0 fail / 0 warn)
raw: 48000 Hz, 1 ch, 6.000 s     derived: 16000 Hz, 1 ch, 6.000 s
source_kind: SYNTHETIC_TEST_INPUT_NOT_MENTRA_AUDIO
is_mentra_hardware_audio: false
mentra_guard_refused_synthetic_capture: true
CAPTURE_SELFTEST: PASS
```

The input is LibriSpeech from this repo's own public evaluation data. **This
proves the capture tool works and says nothing whatsoever about Mentra Live.**

A full live smoke test was also run against `run_receiver.py` itself on port
18772 (production 8765 untouched) with
`MENTRA_MODEL=geowearnet_g2 MENTRA_AUDIO_POLICY=geowear_envelope
MENTRA_GATE_PREROLL_MS=150 --hop-s 0.1 --capture-dir`: capabilities advertised
`requiresEnrollment=false`, `experimental=true`,
`supportsSourceSeparation=false`; DETECTION frames carried real
`gate_gain`/`gate_state`/`gate_preroll_ms`; the capture session closed PASS;
and session state reset with no wearer identity retained.

---

## WS39 — shipping gate

A candidate may be called **MENTRA-PILOT-VALIDATED** only when **all** hold:

| # | Criterion | Met? |
|---|---|---|
| 1 | Audio came from a real Mentra microphone (`assert_mentra_hardware` passes) | **NO** |
| 2 | Multiple unseen users tested (≥5, wearer-disjoint) | **NO** |
| 3 | No voice enrollment used at any point | yes (architecturally) |
| 4 | Camera OFF | yes (never used) |
| 5 | IMU OFF | yes (never used) |
| 6 | Real bystander speech present in the evaluation | **NO** |
| 7 | Real environmental/machinery noise present | **NO** |
| 8 | Agent-facing metrics measured (wearer WER, bystander leakage, command accuracy, false-command rate) | **NO** — on Mentra audio |
| 9 | Real end-to-end latency measured on the device chain | **NO** |

**The gate is NOT met, and cannot be met in this environment.** Criteria 1, 2,
6, 7, 8 and 9 all require the physical device and human subjects.

Current honest status: **WEARABLE_VALIDATED_EXPERIMENTAL** (MMCSG/Aria proxy,
from G2/G3), plus **LIVE_ROUTING_VERIFIED** (G4 software lane).

---

## WS40 — experimental ship configuration (specified, not enabled)

```bash
MENTRA_MODEL=geowearnet_g2 \
MENTRA_AUDIO_POLICY=geowear_envelope \
MENTRA_GATE_POLICY=A_balanced \
MENTRA_GATE_PREROLL_MS=150 \
python3 scripts/mentra/run_receiver.py --listen 127.0.0.1 --port <unused> --hop-s 0.1
```

Reports `experimental=true`, `requiresEnrollment=false`,
`supportsSourceSeparation=false`, `audioPolicy=geowear_envelope`.

**The live default was not changed**: `MENTRA_MODEL` still defaults to
`speakernet` and `MENTRA_AUDIO_POLICY` to `passthrough`. Enabling this on a
real deployment is **not** justified yet — the routing is now verified, but
nothing about the *device* is.

---

## BLOCKED_NO_MENTRA_HARDWARE — the complete list

Machine-readable: `training/geowearnet/g4/blocked.py`, mirrored into
`evaluation/geowearnet/g4/g4_final_summary.json`. 21 named workstreams; none
was attempted with substitute data.

### The base requirement, shared by nearly all of them

- 1x physical **Mentra Live** (or another MentraOS-compatible model, recorded as such)
- 1x paired phone running the MentraOS app (model + OS version recorded)
- A laptop whose browser can select the glasses microphone as an input device
  — **or** a phone-side app built on the Mentra Bluetooth SDK if it cannot
- **No account or credentials needed** for the browser capture route
- Software already ready: `web/app/pages/capture.vue`,
  `run_receiver.py --capture-dir`, `mentra/capture/`

### The first pilot everything else depends on

**5 people × 5–10 minutes each (25–50 minutes total)**, the *same* physical
glasses passed between all five (this is simultaneously the shared-glasses
test). No enrollment, camera off, IMU off. Per person: `quiet_solo`,
`bystander_1m` (including deliberate overlap), `noise_machinery`,
`quiet_wearer`, and `agent_commands` — 10 scripted commands spoken by the
wearer **and the same 10 spoken by the bystander**, which is what makes
false-command-rate measurable at all.

### The items

| ID | Workstream | Status | Extra beyond the base pilot |
|---|---|---|---|
| G4-B01 | Five-person zero-shot smoke | BLOCKED | — |
| G4-B02 | Shared-glasses test | BLOCKED | software half already done + tested |
| G4-B03 | G4 zero-shot eval of the frozen parent | BLOCKED | per-frame activity labels (close mics or 4–8 h annotation) |
| G4-B04 | Pipeline matrix on Mentra audio | BLOCKED | same labels |
| G4-B05 | Agent command bench | BLOCKED | scripted commands from both speakers |
| G4-B06 | Autobody conditions on hardware | BLOCKED | a real workshop, +10–15 min/wearer |
| G4-B07 | Distance / quiet-wearer stress | BLOCKED | bystander at 0.5/1/3 m; wearer normal/soft/turned away |
| G4-B08 | Real agent ASR on Mentra captures | BLOCKED | per-speaker reference transcripts |
| G4-B09 | False-agent-command rate | BLOCKED | bystander-spoken commands |
| G4-B10 | Mentra zero-shot verdict | BLOCKED | depends on B03/B04/B05 |
| G4-B11 | Domain-gap autopsy | BLOCKED | true native-rate raw stream; moderate new code |
| G4-B12 | Small Mentra adaptation | NOT_APPLICABLE / BLOCKED | 60+ labelled minutes; gated on B10 |
| G4-B13 | Mentra learning curve | BLOCKED | 60+ labelled minutes, sliceable |
| G4-B14 | Cross-wearer generalisation | BLOCKED | 5 people detects a large effect only |
| G4-B15 | Session variability | BLOCKED | same wearers on 2 separate days |
| G4-B16 | Multi-microphone question | **PARTIALLY ANSWERED** | desk research done; device inspection remains |
| G4-B17 | Overlap re-evaluation on Mentra | BLOCKED | overlap-marked labels |
| G4-B18 | Separator-gate re-check | BLOCKED | depends on B17; **gate stays CLOSED** |
| G4-B19 | Live receiver vs a real device | BLOCKED | one 10-minute live session |
| G4-B20 | Real end-to-end latency | BLOCKED | one session, ideally through the deployed tunnel |
| G4-B21 | Glasses-swap soak | BLOCKED | 60+ min continuous, ≥6 wearer swaps |

For most of these the answer to "what engineering remains once data lands?" is
**none** — the harness, metrics, gate and capture tool already exist. B03 needs
a small labelled-item adapter; B11 needs a distribution-comparison script whose
design deliberately waits until the real audio format is known.

---

## Standing findings carried forward, explicitly caveated

- **Overlap.** G3 measured overlap contributing **6.90%** of oracle residual
  leakage on MMCSG, with muting it costing 3.99 pp of wearer deletion.
  **NOT YET RE-VERIFIED ON REAL MENTRA AUDIO** (G4-B17).
- **Separator gate.** **CLOSED**, unchanged from G3. G4 produced no new
  evidence bearing on it and did not reopen it.
- **Identity.** `IDENTITY_INFORMATION_PRESENT_BUT_NOT_CAUSAL` (G3), untouched.
- **Soak.** G3's software soak plus G4's envelope/router reset assertions show
  no state leaks between wearers *in this process*. Neither says anything
  about a physical device staying connected (G4-B21).

---

## Next action

**Supply one physical Mentra Live plus a paired phone, and run the five-person
pilot through the capture mode built in this campaign.** Concretely:

1. Pair the Mentra Live to a laptop and confirm the glasses microphone appears
   as a selectable browser input device. If it does not, the phone-side
   Bluetooth SDK route is required and that is a separate build.
2. Start the receiver with capture enabled on a non-production port:
   `python3 scripts/mentra/run_receiver.py --port <unused> --capture-dir captures/pilot01`
3. Open the research capture page, select the glasses microphone, declare
   `source_kind = MENTRA_LIVE_GLASSES_MIC`, record the glasses firmware and
   MentraOS app versions, and label each condition.
4. Record 5 people × 5–10 minutes across the five conditions, passing the same
   glasses between them.
5. Before anything else, run the AGC check from
   `docs/geowearnet_g4_mentraos_audio_path.md` — a fixed stimulus at two levels
   ~20 dB apart. If a compressor sits in the chain, GeoWearNet's
   amplitude-sensitive physical features are compromised and that changes the
   whole plan.

That single session unblocks G4-B01, B02, B03, B19 and B20 immediately, and
B04–B09 and B17 as soon as the takes are labelled.

---

## Tests

**305 Python tests pass** (up from G3's 225; +80 added by G4), plus **38 web
tests** and a clean `nuxt typecheck`. The 4 pre-existing ONNX export warnings
are unchanged.

New coverage: router↔offline bit-identity across policies and block sizes,
pre-roll delay-line semantics and its offline equivalent, bounded/quantised
pre-roll, frame-alignment contract, session reset and no-retained-identity
invariants, no-source-separation invariant on OVERLAP, capability
policy-list anti-drift, no-`geowearnet_g4`-model invariant, capture metadata
contract, validation of silent/clipped/short/wrong-rate/corrupt/non-finite
takes, cross-stream duration mismatch, the Mentra-hardware guard in both
directions, the capture path over the real transport, WS2/WS3 measurement
definitions, the predeclared selection rule, and the blocked registry's own
completeness.

---

## Guards

| | |
|---|---|
| Official MMCSG dev split used | **No** |
| Official MMCSG eval split used | **No** |
| Frozen G2 checkpoint modified | **No** (hash re-verified) |
| Raw MMCSG data modified | **No** |
| SpeakerNet / MentraWearNet touched | **No** |
| WearerSepNet training started | **No** (gate still CLOSED) |
| Live default audio policy changed | **No** (still `passthrough`) |
| Mentra audio fabricated or relabelled | **No** |
| Commit / push made | **No** |
