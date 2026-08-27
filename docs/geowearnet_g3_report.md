# GeoWearNet G3 report

Status: **COMPLETE_WITH_GATED_NEGATIVE_FINDINGS**

G3 is complete as a gated campaign. The frozen G2 parent was not changed, no G3 official MMCSG dev look was spent, and no WearerSepNet model was trained because the predeclared overlap gate failed.

## Executive result

- Identity verdict: **IDENTITY_INFORMATION_PRESENT_BUT_NOT_CAUSAL**.
- Real decision-conditioned identity audit: **NO_MATERIAL_HELD_RECORDING_IMPROVEMENT** across 671,575 internal-val frames after acoustic/noise controls.
- Product frontend verdict: **GATING_PRODUCT_VALUE_CONFIRMED**; the gate reduces internal-val BLR from 62.09% to 7.04%, while remaining experimental and opt-in.
- Frozen parent: g2_ablation_ctx1000, tcn, 132678 parameters, 1000 ms context.
- Final-G2 internal-val predicted gate: wearer WER 28.27%, BLR 7.04%, deletion 19.09%, activity F1 79.06%.
- Final-G2 oracle pass-overlap: wearer WER 21.50%, BLR 5.04%, deletion 14.92%.
- WearerSepNet decision: **WEARER_SEPARATOR_NOT_STARTED**; overlap contributes 6.90% of oracle residual leakage, below the 50% threshold, and muting costs 3.99%, below the 5-point threshold.

## Frozen parent and evaluation boundaries

- Checkpoint: training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt.
- SHA-256: 07c43c3d9e37dbd490ae0477f46ff515e1e19e8fe0a20687ec9d13a8555c68ef.
- Freeze manifest: evaluation/geowearnet/g3/g2_frozen_parent.json.
- G3 identity and product measurements used the frozen checkpoint and did not touch the official MMCSG dev/eval splits.
- The G2 final-dev artifact remains the sole guarded official-dev result.

## Identity causality

The causal experiment used 60 same-source role flips on synthetic test speakers disjoint from synthetic training speakers. Role-flip accuracy was 1.000; grouped leave-identity-out accuracy was 1.000; geometry dominance was 362.1x.

Speaker identity was still decodable in physical, log-mel, model-input, early, late, and final-pre-head representations. That is information presence, not shortcut causality. Same-speaker content control and same-geometry speaker swap both had median absolute probability deltas near 0.0005–0.0006, while the same-source role flip median was about 0.229. The old closed-set/session probe is retained as context and is explicitly not treated as causal evidence.

Artifact: evaluation/geowearnet/g3/identity_causality_report.json.

### Real decision-conditioned audit

The frozen checkpoint was then scored over 671,575 frames from all 38 internal-validation recordings. These wearers are disjoint from G2 inner training (34 train versus 9 validation identities; overlap 0). Overall solo AUROC was 0.947; false-wearer rate on environment-only frames was 10.63%; wearer-only miss rate was 8.06%.

Across the nine held-out wearer identities, median false-wearer rate was 6.11% (p90 20.66%). A held-recording cross-validation model gained no material predictive value from self-speaker identity after physical-feature and noise-category controls (mean expanded-minus-control AUROC -0.006; log-loss +0.021). This is deliberately an observational error analysis, not a causal counterfactual. The previously identified RTTM role-swap suspicion was retained and flagged, not relabelled or excluded.

Artifact: evaluation/geowearnet/g3/real_identity_error_audit.json.

## Product and overlap bottleneck

The primary product matrix is the final G2 checkpoint on 38 GeoWearNet-internal validation windows (12.67 minutes), not the earlier 28,950-parameter proxy.

| Pipeline | Wearer WER | BLR | Wearer deletion | Activity F1 | Algorithmic latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAW | 89.53% | 62.09% | 11.49% | n/a | 0.0 ms |
| RNNOISE | 84.99% | 55.22% | 12.14% | n/a | 10.5 ms |
| GEOWEAR_GATE | 28.27% | 7.04% | 19.09% | 79.06% | 35.0 ms |
| RNNOISE_GEOWEAR_GATE | 47.08% | 16.96% | 17.61% | 70.10% | 45.5 ms |
| ORACLE_GATE | 21.50% | 5.04% | 14.92% | 93.03% | 35.0 ms |
| RNNOISE_ORACLE_GATE | 22.61% | 5.22% | 15.48% | 93.03% | 45.5 ms |
| ORACLE_GATE_MUTE_OVERLAP | 25.21% | 4.70% | 18.91% | 84.19% | 35.0 ms |

The exact upper-bound overlap contribution is 6.90%; muting overlap removes only 0.35% BLR and costs 3.99% wearer deletion. The token-bag attribution independently puts overlap at 11.89% of leaked bystander words, with the remaining leakage dominated by environment-only frames.

The final-G2 policy sweep supports explicit routing tradeoffs. The default remains passthrough; GeoWear gating is opt-in. The synthetic stress pass covers quiet wearer, loud bystander, overlap, machinery/music, and high-reverberation scenarios, but it is not wearable-hardware evidence.

On that 10-item synthetic stress pass, predicted gating had wearer WER 55.15%, BLR 61.84%, wearer-command retention 88.89%, and false-agent-command rate 66.67%. This poor predicted-vs-oracle gap is a detector/generalization priority.

### First-word and transition exposure

On the same 38 internal-val windows, the balanced causal gate opened within 500 ms for 91.03% of 78 wearer onsets (median 100 ms). Its first-word gate-exposure proxy passed 59.21% of 76 labelled first-word events. The no-smoothing control increased that proxy to 73.68%, but no post-hoc retuning was adopted from this internal split. These are router-exposure measurements, not ASR word-error claims.

Artifact: evaluation/geowearnet/g3/transition_frontend_audit.json.

## WearerSepNet status

MMCSG is recorded as REAL_MIX_ONLY: 7-channel 48 kHz wearable mixtures with RTTM/transcript annotations and no isolated clean wearer target. LibriSpeech and MUSAN provide clean synthetic sources, and the existing stress bench retains clean source pairs, but no R0 was trained because the final-G2 internal-val go/no-go failed both criteria.

Data audit: evaluation/geowearnet/g3/wearersepnet_data_audit.json. Design spec remains go/no-go gated and source-separation capability remains false.

## Runtime and policy

Added the modular AudioFrontend.process boundary in server/audio/frontend.py. Supported policies are passthrough, GeoWear gate, and RNNoise plus GeoWear gate. Passthrough is the default and preserves the existing SpeakerNet/default behavior. The capability payload now reports the active audioPolicy and experimental status; it continues to report supportsSourceSeparation=false. GeoWearNet now exposes all four receiver states: WEARER and OVERLAP pass the original mixed PCM, while ENVIRONMENT and SILENCE mute it. RNNoise retains its recurrent state within a live stream and resets explicitly at session boundaries; independent evaluation windows reset it before each comparison.

The opt-in G2 receiver loaded the frozen checkpoint, initialized ASR, advertised model=geowearnet_g2, requiresEnrollment=False, and ready=True on an unused localhost port. The existing receiver on port 8765 was preserved. A sequential A→B software swap now resets rolling PCM, classifier state, RNNoise, ASR stream, transcript deduplication, telemetry, and playback routing at connection/discontinuity boundaries without retaining a wearer identity.

Frozen-parent CPU evidence on real MMCSG audio: 1/2/4-thread p50 was 3.64/3.30/3.19 ms per 1 s chunk; streaming parity passed (max wearer-logit difference 4.68e-06); future-audio perturbation had zero pre-T effect; and the session-reset soak streamed 6.15 real minutes with 0 NaN/Inf outputs and 0.78 MB RSS growth.

The actual PCM receiver path was then measured over 30 seconds of real MMCSG proxy audio at 10 ms frames: at two CPU threads, GeoWear+gate update compute was p50/p95 8.30/13.38 ms (RTF 0.048); RNNoise+GeoWear+gate was 11.42/15.55 ms (RTF 0.181). The frontend has zero future-audio lookahead and zero extra audio buffer; the receiver makes a new detector decision every 200 ms, which is scheduling cadence rather than lookahead. This live route is explicitly a four-state, 200 ms router with 0.60/0.40 on/off thresholds; it passes overlap as mixed PCM and does not claim exact equivalence to the evaluator's 10 ms attack/release/hangover envelope. ASR was intentionally excluded as a downstream provider.

Artifact: evaluation/geowearnet/g3/live_frontend_cpu_benchmark.json.

## License, hardware, and verification

The current MMCSG record is in docs/geowearnet_data_license_state.md. It records the exact current Meta agreement as permitting research and commercial-purpose software/model development subject to the agreement, while prohibiting raw redistribution and imposing deletion/termination obligations. This project record is factual, not legal advice. The authoritative source is https://ai.meta.com/datasets/mmcsg-downloads/.

Mentra hardware validation remains NOT TESTED. G3 used CPU identity/product evaluation; no G3 training was run. The full Python suite passed **225 tests** with 4 existing ONNX-export warnings; focused shared-session/frontend/runtime coverage passed 15 tests and the web capability contract passed 38 tests. Async transport passed 3/3 after adding pytest-asyncio>=0.24,<2 and asyncio_mode = auto.

## Negative findings and next action

- Representation-level identity remains highly decodable, so decodability alone is not evidence of a causal shortcut.
- The original closed-set identity/session probe was not leave-person-out and did not test geometry counterfactuals.
- On real wearer-disjoint internal validation, adding self-speaker identity after physical-feature and noise controls did not improve held-recording false-wearer prediction; the real result is observational, not a causal counterfactual.
- First-word metrics are gate-exposure proxies, not ASR word-error measurements. The no-smoothing control exposed more first-word audio, but no post-hoc gate retuning was applied on internal validation.
- Final-G2 internal-val overlap is only 6.9% of oracle residual leakage; muting overlap costs 3.99 percentage points, so neither separator go-no-go criterion passes.
- MMCSG has no isolated clean wearer waveform target; synthetic sources exist but do not justify training after the product gate fails.
- Final-G2 predicted gating remains materially worse than oracle gating on the synthetic stress bench; this is a detector/generalization finding, not a separator justification.
- No adversarial identity-invariance branch was run because the controlled evidence did not confirm identity as a causal harmful shortcut.
- No G3 official MMCSG dev evaluation was performed and no G2 artifact was overwritten.
- Real Mentra-glasses, shared-glasses, and multi-user session validation remain untested; MMCSG Aria audio is a proxy only.
- The A-to-B shared-glasses test proves software state reset only; it is not a physical multi-user Mentra validation.
- The live receiver is a coarse 200 ms four-state router, not a claim of exact equivalence to the evaluator's 10 ms attack/release/hangover gate envelope.

## Promotion, blockers, and next actions

- Promotion verdict: **WEARABLE_VALIDATED_EXPERIMENTAL**. The identity blocker is resolved as IDENTITY_INFORMATION_PRESENT_BUT_NOT_CAUSAL; this does not imply Mentra validation.
- Hardware verdict: **MENTRA_NOT_TESTED**. The A→B session-reset result is software-only and does not replace a physical shared-glasses test.
- Separator verdict: **WEARER_SEPARATOR_NOT_STARTED**. The highest-value next step is detector/generalization work on environment-only false-wearer errors and real Mentra capture validation. A separator should remain unstarted until a future real capture or a revised product measurement makes the predeclared overlap gate pass.
- External blocker: physical Mentra audio capture is not available in this workspace. The audio-only capture/product-bench protocol is prepared in docs/geowearnet_capture_protocol.md.

## Key changed artifacts

- training/geowearnet/g3/{real_identity_audit.py, transition_frontend.py, runtime_evidence.py, live_frontend_runtime.py, finalize.py}
- mentra/audio/consumer.py, server/audio/remote_receiver.py, server/audio/frontend.py, evaluation/agent_audio/denoise.py
- server/models/capabilities.py, scripts/mentra/run_receiver.py, web/app/utils/modelCapabilities.ts
- evaluation/geowearnet/g3/* and tests/audio/test_session_reset.py

No commit or push was made. The existing dirty worktree was preserved.
