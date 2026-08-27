# GeoWearNet Campaign G1 — Results

_Generated 2026-08-26T06:27:41Z. 24 evaluations, 20 training runs._

> **ALL numbers below are SIMULATED GEOMETRY (LibriSpeech CC BY 4.0 + MUSAN through the GeoWearNet S1/S2 simulator). No Mentra hardware data exists. Nothing here is Mentra product validation.**


## E0 — heuristic baseline (Workstream AH)

Frame-level E0 on the S1 simulator: 14 interpretable physical scalars plus causal running means over 10/30/68 frames (matched to E1's 680 ms context), fed to logistic regression and a small MLP. Same trials E1 sees.

| condition | E0 wearer AUROC | logRMS-alone AUROC |
|---|---|---|
| normal | 0.8201 | 0.7962 |
| level_matched | 0.7261 | 0.6994 |
| random_gain | 0.7948 | 0.7397 |
| wearer_quiet | 0.7033 | 0.6789 |
| wearer_loud | 0.8563 | 0.8356 |
| bystander_close | 0.7904 | 0.7637 |
| bystander_close_loud | 0.6466 | 0.6129 |
| bystander_loud | 0.7433 | 0.7171 |
| shouting_bystander | 0.7212 | 0.6894 |
| heavy_overlap | 0.8029 | 0.7747 |
| noisy | 0.7568 | 0.6718 |
| high_rt60 | 0.8214 | 0.7890 |
| ood_geometry | 0.8467 | 0.8128 |

**Workstream J verdict: `AMPLITUDE_INFORMATIVE_BUT_NOT_SUFFICIENT`** — level-match delta -0.0939, gain-random delta -0.0253.


**Legacy E0 on the S0 simulator** (kept for history, NOT comparable — clip-level, single-source, per-source device EQ): verdict `{'value': 'E0_PHYSICAL_SIGNAL_STRONG', 'all_features_test_auroc': 0.9969135802469136, 'amplitude_only_test_auroc': 0.9865432098765432, 'spectral_only_test_auroc': 0.9456790123456791, 'level_matched_auroc': 0.9927111111111111, 'random_gain_auroc': 0.9976888888888888, 'note': 'SIMULATED_GEOMETRY data only (Phase 2/47) -- this verdict is about whether the simulated transfer-function hypothesis has ANY nontrivial, non-loudness signal, not about real Mentra wearer detection.'}`, all-features test AUROC 0.9990. The drop from S0 to S1 is the single most important negative result of this campaign and is discussed below.


## E1 main table (Workstream AU)

| model | enroll? | params | ctx ms | train→eval | normal | level-match | gain-rand | close-byst | shout-byst | OOD geom | FRR@FAR5 | overlap F1 | logRMS-only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 heuristic (all feats) | no | ~1.4K | 680 | S1→S1 | 0.8201 | 0.7261 | 0.7948 | 0.7904 | 0.7212 | 0.8467 | 0.6526 | n/a | 0.7962 |
| geowearnet_e1_s1_ablate_noamp_20260825_074920_bc2a_onS1_final | no | 107670 | 680 | S1→S1 | 0.9546 | 0.8919 | 0.9465 | 0.8873 | 0.8846 | 0.9591 | 0.2148 | 0.592 | 0.808 |
| geowearnet_e1_s1_ablate_spectral_20260825_074920_8250_onS1_final | no | 107382 | 680 | S1→S1 | 0.9511 | 0.8901 | 0.9429 | 0.8837 | 0.8823 | 0.9586 | 0.2292 | 0.588 | 0.808 |
| geowearnet_e1_s1_arch_convnext_20260825_151958_a7fc_onS1_final | no | 111958 | 980 | S1→S1 | 0.9575 | 0.8950 | 0.9482 | 0.8923 | 0.8874 | 0.9602 | 0.1983 | 0.601 | 0.808 |
| geowearnet_e1_s1_arch_crnn_20260825_150233_83e5_onS1_final | no | 28950 | 80 | S1→S1 | 0.9615 | 0.8942 | 0.9491 | 0.9079 | 0.8890 | 0.9638 | 0.1799 | 0.610 | 0.808 |
| geowearnet_e1_s1_arch_dsconv_20260825_141832_eea7_onS1_final | no | 24150 | 380 | S1→S1 | 0.9421 | 0.8704 | 0.9315 | 0.8716 | 0.8646 | 0.9513 | 0.2533 | 0.587 | 0.808 |
| geowearnet_e1_s1_base_20260825_050841_0ff0_onS1_final | no | 107718 | 680 | S1→S1 | 0.9533 | 0.8932 | 0.9442 | 0.8858 | 0.8841 | 0.9594 | 0.2154 | 0.593 | 0.808 |
| geowearnet_e1_s1_base_20260825_050841_0ff0_onS1_interim | no | 107718 | 680 | S1→S1 | 0.9361 | 0.8519 | 0.9207 | 0.8703 | n/a | n/a | 0.2849 | 0.570 | 0.808 |
| geowearnet_e1_s1_base_20260825_050841_0ff0_onS1_midtrain_step4500_smoketest | no | 107718 | 680 | S1→S1 | 0.9477 | 0.8792 | 0.9351 | 0.8749 | 0.8730 | 0.9547 | 0.2402 | 0.558 | 0.808 |
| geowearnet_e1_s1_base_20260825_050841_0ff0_onS2_final | no | 107718 | 680 | S1→S2 | 0.9361 | 0.8972 | 0.9256 | 0.8652 | 0.8468 | 0.9370 | 0.3110 | 0.586 | 0.786 |
| geowearnet_e1_s1_ctx1000_20260825_121526_3dba_onS1_final | no | 132678 | 1000 | S1→S1 | 0.9564 | 0.8962 | 0.9482 | 0.8927 | 0.8875 | 0.9614 | 0.2064 | 0.602 | 0.808 |
| geowearnet_e1_s1_ctx100_20260825_105635_42d8_onS1_final | no | 41190 | 100 | S1→S1 | 0.9208 | 0.8339 | 0.9080 | 0.8462 | 0.8303 | 0.9329 | 0.3390 | 0.552 | 0.808 |
| geowearnet_e1_s1_ctx250_20260825_111359_26c1_onS1_final | no | 74774 | 250 | S1→S1 | 0.9421 | 0.8729 | 0.9318 | 0.8683 | 0.8652 | 0.9507 | 0.2526 | 0.567 | 0.808 |
| geowearnet_e1_s1_ctx500_20260825_115804_b354_onS1_final | no | 107718 | 520 | S1→S1 | 0.9509 | 0.8893 | 0.9426 | 0.8835 | 0.8818 | 0.9571 | 0.2268 | 0.589 | 0.808 |
| geowearnet_e1_s1_loss_m0_20260825_095443_b0d3_onS1_final | no | 107718 | 680 | S1→S1 | 0.9538 | 0.8923 | 0.9454 | 0.8871 | 0.8857 | 0.9599 | 0.2148 | 0.591 | 0.808 |
| geowearnet_e1_s1_loss_m2_20260825_101203_7faa_onS1_final | no | 107718 | 680 | S1→S1 | 0.9535 | 0.8925 | 0.9458 | 0.8866 | 0.8839 | 0.9598 | 0.2173 | 0.597 | 0.808 |
| geowearnet_e1_s1_norm_cmvn_20260825_090229_e023_onS1_final | no | 107718 | 680 | S1→S1 | 0.9446 | 0.8889 | 0.9345 | 0.8794 | 0.8780 | 0.9506 | 0.2544 | 0.581 | 0.808 |
| geowearnet_e1_s1_norm_none_20260825_084323_662d_onS1_final | no | 107718 | 680 | S1→S1 | 0.9534 | 0.8913 | 0.9445 | 0.8866 | 0.8831 | 0.9577 | 0.2128 | 0.596 | 0.808 |
| geowearnet_e1_s1_size250k_20260825_131702_1e6d_onS1_final | no | 255098 | 680 | S1→S1 | 0.9585 | 0.9028 | 0.9505 | 0.8953 | 0.8939 | 0.9585 | 0.1981 | 0.602 | 0.808 |
| geowearnet_e1_s1_size500k_20260825_140104_1c6a_onS1_final | no | 523974 | 680 | S1→S1 | 0.9599 | 0.9083 | 0.9536 | 0.8981 | 0.8990 | 0.9635 | 0.1919 | 0.598 | 0.808 |
| geowearnet_e1_s1_size50k_20260825_125932_bad2_onS1_final | no | 52386 | 680 | S1→S1 | 0.9504 | 0.8818 | 0.9392 | 0.8817 | 0.8748 | 0.9572 | 0.2314 | 0.596 | 0.808 |
| geowearnet_e1_s2_base_20260825_050841_6e1b_onS1_final | no | 107718 | 680 | S2→S1 | 0.9469 | 0.8850 | 0.9371 | 0.8741 | 0.8821 | 0.9560 | 0.2433 | 0.573 | 0.808 |
| geowearnet_e1_s2_base_20260825_050841_6e1b_onS2_final | no | 107718 | 680 | S2→S2 | 0.9471 | 0.9114 | 0.9347 | 0.8711 | 0.8692 | 0.9531 | 0.2653 | 0.598 | 0.786 |
| geowearnet_e1_mixed_base_20260825_065020_4668_onS1_final | no | 107718 | 680 | mixed→S1 | 0.9534 | 0.8933 | 0.9443 | 0.8827 | 0.8869 | 0.9604 | 0.2208 | 0.578 | 0.808 |
| geowearnet_e1_mixed_base_20260825_065020_4668_onS2_final | no | 107718 | 680 | mixed→S2 | 0.9454 | 0.9101 | 0.9349 | 0.8717 | 0.8643 | 0.9504 | 0.2797 | 0.592 | 0.786 |

## Cross-simulator generalisation (Workstream T)

| train→eval | normal | level-match | gain-rand | OOD geom | robust mean |
|---|---|---|---|---|---|
| mixed→S1 | 0.9534 | 0.8933 | 0.9443 | 0.9604 | 0.9201 |
| mixed→S2 | 0.9454 | 0.9101 | 0.9349 | 0.9504 | 0.9128 |
| S1→S2 | 0.9361 | 0.8972 | 0.9256 | 0.9370 | 0.9013 |
| S2→S1 | 0.9469 | 0.8850 | 0.9371 | 0.9560 | 0.9135 |

## Model selection (Workstream AS)

Selection uses a robustness objective (level-matched 0.22, gain-randomised 0.18, close-bystander 0.18, shouting-bystander 0.12, OOD geometry 0.10, normal 0.10, solo 0.10) minus 0.15 x the close-bystander false-wearer rate — deliberately NOT normal-condition AUROC.

| model | selection score | robust mean | worst condition |
|---|---|---|---|
| geowearnet_e1_s1_arch_crnn_20260825_150233_83e5_onS1_final | 0.8718 | 0.9276 | 0.8890 |
| geowearnet_e1_s1_size500k_20260825_140104_1c6a_onS1_final | 0.8612 | 0.9304 | 0.8981 |
| geowearnet_e1_s1_size250k_20260825_131702_1e6d_onS1_final | 0.8609 | 0.9266 | 0.8939 |
| geowearnet_e1_s1_ctx1000_20260825_121526_3dba_onS1_final | 0.8547 | 0.9237 | 0.8875 |
| geowearnet_e1_s1_arch_convnext_20260825_151958_a7fc_onS1_final | 0.8532 | 0.9234 | 0.8874 |
| geowearnet_e1_s1_ablate_noamp_20260825_074920_bc2a_onS1_final | 0.8476 | 0.9207 | 0.8846 |
| geowearnet_e1_s1_loss_m0_20260825_095443_b0d3_onS1_final | 0.8476 | 0.9207 | 0.8857 |
| geowearnet_e1_s1_norm_none_20260825_084323_662d_onS1_final | 0.8465 | 0.9194 | 0.8831 |
| geowearnet_e1_s1_base_20260825_050841_0ff0_onS1_final | 0.8456 | 0.9200 | 0.8841 |
| geowearnet_e1_s1_ablate_spectral_20260825_074920_8250_onS1_final | 0.8436 | 0.9181 | 0.8823 |
| geowearnet_e1_s1_ctx500_20260825_115804_b354_onS1_final | 0.8423 | 0.9175 | 0.8818 |
| geowearnet_e1_s1_norm_cmvn_20260825_090229_e023_onS1_final | 0.8401 | 0.9127 | 0.8780 |
| geowearnet_e1_s1_size50k_20260825_125932_bad2_onS1_final | 0.8389 | 0.9142 | 0.8748 |
| geowearnet_e1_s1_loss_m2_20260825_101203_7faa_onS1_final | 0.8375 | 0.9203 | 0.8839 |
| geowearnet_e1_s2_base_20260825_050841_6e1b_onS2_final | 0.8336 | 0.9144 | 0.8692 |
| geowearnet_e1_s1_base_20260825_050841_0ff0_onS1_interim | 0.8316 | 0.8947 | 0.8519 |
| geowearnet_e1_s1_base_20260825_050841_0ff0_onS1_midtrain_step4500_smoketest | 0.8278 | 0.9108 | 0.8730 |
| geowearnet_e1_s1_ctx250_20260825_111359_26c1_onS1_final | 0.8265 | 0.9052 | 0.8652 |
| geowearnet_e1_s1_arch_dsconv_20260825_141832_eea7_onS1_final | 0.8260 | 0.9052 | 0.8646 |
| geowearnet_e1_s1_ctx100_20260825_105635_42d8_onS1_final | 0.7926 | 0.8787 | 0.8303 |

**Selected: `geowearnet_e1_s1_arch_crnn_20260825_150233_83e5_onS1_final`** (28950 params, 80 ms context).


## CPU deployment (Workstreams AK/AL/AM)

| model | params | size MB | ctx ms | p50 ms/1s | p95 ms/1s | RTF (1 thread) | streaming vs rolling | streaming parity (max abs diff) |
|---|---|---|---|---|---|---|---|---|
| geowearnet_e1_s1_arch_convnext | 111958 | 0.47 | 980 | 1.85 | 2.77 | 0.00185 | n/a | n/a |
| geowearnet_e1_s1_arch_crnn | 28950 | 0.13 | 80 | 2.65 | 2.77 | 0.00265 | n/a | 2.9e-06 |
| geowearnet_e1_s1_base | 107718 | 0.46 | 680 | 2.16 | 3.53 | 0.00216 | 1.2 | 2.9e-06 |
| geowearnet_e1_s1_size500k | 523974 | 2.13 | 680 | 4.06 | 4.96 | 0.00406 | 1.3 | 2.1e-06 |
| smoke_cpu | 107718 | 0.46 | 680 | 3.49 | 4.82 | 0.00349 | 1.0 | 2.1e-07 |

`ctx ms` is the bounded conv receptive field only; the CRNN's GRU state is additionally unbounded in principle (see model_zoo.py), so its true effective context is not fully captured by this column. Streaming parity is verified separately per architecture (`StreamingTCN`, `StreamingCRNN`) against the batch forward pass, not assumed.


Export (primary/first model benchmarked): TorchScript `OK`, ONNX `OK`.


**Soak tests** (continuous stateful streaming, checked for NaNs / RSS growth / state-size drift / latency drift):

| model | minutes | NaN/Inf | RSS growth MB | state elems | state constant? | latency drift % |
|---|---|---|---|---|---|---|
| geowearnet_e1_s1_arch_crnn | 1.0 | 0 | 0.00 | 512 | True | -1.9 |
| geowearnet_e1_s1_base | 1.0 | 0 | 0.00 | 4288 | True | -1.2 |
| geowearnet_e1_s1_size500k | 1.0 | 0 | 0.00 | 9280 | True | 3.1 |

## MMCSG zero-shot sim-to-real transfer (Workstreams W/X/AZ)

> MMCSG (Meta, real Aria smart-glasses recordings of real conversations) is NOT Mentra hardware and this is NOT Mentra product validation. It is the closest available real-wearable proxy. Access required the project owner to independently accept Meta's CC-BY-NC research license (see `docs/geowearnet_data_license_state.md`); MMCSG is used here strictly as a research/evaluation signal, never as a training dependency for anything intended to ship, and the raw data is not redistributed.

Corpus present: 172 train / 169 dev / 189 eval recordings, 7 raw microphones, 48kHz. Per-channel E0-feature wearer-vs-other AUROC measured (not assumed) on a train-split sample: channel 2 was strongest (AUROCs 0:0.867, 1:0.873, 2:0.879, 3:0.845, 4:0.835, 5:0.734, 6:0.734) and is the channel used below.

| model | trained on | MMCSG split | n rec | channel | wearer AUROC (95% CI) | solo AUROC (95% CI) | wearer EER | solo EER |
|---|---|---|---|---|---|---|---|---|
| geowearnet_e1_s1_arch_crnn | S1 | eval | 189 | 2 | 0.8860 (0.8767-0.8936) | 0.9687 (0.9573-0.9752) | 0.1781 | 0.0835 |
| geowearnet_e1_s1_size500k | S1 | eval | 189 | 2 | 0.8781 (0.8692-0.8850) | 0.9630 (0.9507-0.9704) | 0.1867 | 0.0921 |
| geowearnet_e1_s1_base | S1 | eval | 189 | 2 | 0.8780 (0.8687-0.8854) | 0.9593 (0.9465-0.9671) | 0.1881 | 0.0988 |
| geowearnet_e1_mixed_base | mixed | eval | 189 | 2 | 0.8728 (0.8640-0.8799) | 0.9541 (0.9411-0.9624) | 0.1898 | 0.1071 |
| geowearnet_e1_s2_base | S2 | eval | 189 | 2 | 0.8712 (0.8621-0.8787) | 0.9519 (0.9383-0.9608) | 0.1931 | 0.1130 |

`wearer AUROC` scores every frame of every recording (silence and overlap included, against a `wearer_active` binary label built from MMCSG's real word-level self/other transcripts). `solo AUROC` restricts to frames where exactly one side is speaking -- the same `wearer_vs_env_solo` metric used throughout this report, for direct comparability with the simulated-suite numbers above. CIs are recording-level cluster bootstraps (a conversation's frames are not independent draws).

**Every checkpoint tested was trained ENTIRELY on simulated audio and never saw a single real sample -- zero fine-tuning, zero domain adaptation.** All land in the same 0.87-0.89 (all-frames) / 0.95-0.97 (solo) band the simulated suites report, and the ranking is consistent with the simulated selection: the campaign's selected model (`geowearnet_e1_s1_arch_crnn`, chosen purely on simulated robustness) is also the strongest zero-shot transferrer here. This is the first non-simulated evidence produced in this campaign, and it is a positive one for sim-to-real generalisation -- but MMCSG's Aria glasses differ from Mentra's actual microphone/placement, so it substitutes for, rather than replaces, the real Mentra capture pilot.


## Training runs

| run | sim | variant | norm | loss | best step | best selection | final val wearer AUROC | final overlap F1 |
|---|---|---|---|---|---|---|---|---|
| geowearnet_e1_mixed_base_20260825_065020_4668 | mixed | ctx680 | global | M1 | 12000 | 0.8578 | 0.9016 | 0.595 |
| geowearnet_e1_s1_ablate_noamp_20260825_074920_bc2a | S1 | ctx680 | global | M1 | 12000 | 0.9007 | 0.9176 | 0.633 |
| geowearnet_e1_s1_ablate_spectral_20260825_074920_8250 | S1 | ctx680 | global | M1 | 9000 | 0.9072 | 0.9203 | 0.619 |
| geowearnet_e1_s1_arch_convnext_20260825_151958_a7fc | S1 | ctx680 | global | M1 | 11250 | 0.9041 | 0.9228 | 0.633 |
| geowearnet_e1_s1_arch_crnn_20260825_150233_83e5 | S1 | ctx680 | global | M1 | 11250 | 0.9140 | 0.9319 | 0.646 |
| geowearnet_e1_s1_arch_dsconv_20260825_141832_eea7 | S1 | ctx680 | global | M1 | 10500 | 0.8747 | 0.9028 | 0.615 |
| geowearnet_e1_s1_base_20260825_050841_0ff0 | S1 | ctx680 | global | M1 | 10500 | 0.9014 | 0.9202 | 0.638 |
| geowearnet_e1_s1_ctx1000_20260825_121526_3dba | S1 | ctx1000 | global | M1 | 12000 | 0.9018 | 0.9215 | 0.631 |
| geowearnet_e1_s1_ctx100_20260825_105635_42d8 | S1 | ctx100 | global | M1 | 11250 | 0.8128 | 0.8690 | 0.575 |
| geowearnet_e1_s1_ctx250_20260825_111359_26c1 | S1 | ctx250 | global | M1 | 10500 | 0.8820 | 0.9052 | 0.613 |
| geowearnet_e1_s1_ctx500_20260825_115804_b354 | S1 | ctx500 | global | M1 | 10500 | 0.8966 | 0.9155 | 0.630 |
| geowearnet_e1_s1_loss_m0_20260825_095443_b0d3 | S1 | ctx680 | global | M0 | 10500 | 0.9016 | 0.9186 | 0.624 |
| geowearnet_e1_s1_loss_m2_20260825_101203_7faa | S1 | ctx680 | global | M2 | 11250 | 0.8967 | 0.9187 | 0.633 |
| geowearnet_e1_s1_norm_cmvn_20260825_090229_e023 | S1 | ctx680 | cmvn | M1 | 11250 | 0.8826 | 0.9086 | 0.608 |
| geowearnet_e1_s1_norm_none_20260825_084323_662d | S1 | ctx680 | none | M1 | 10500 | 0.8957 | 0.9170 | 0.619 |
| geowearnet_e1_s1_size250k_20260825_131702_1e6d | S1 | size250k | global | M1 | 9750 | 0.9134 | 0.9251 | 0.638 |
| geowearnet_e1_s1_size500k_20260825_140104_1c6a | S1 | size500k | global | M1 | 10500 | 0.9172 | 0.9294 | 0.648 |
| geowearnet_e1_s1_size50k_20260825_125932_bad2 | S1 | size50k | global | M1 | 10500 | 0.8831 | 0.9101 | 0.623 |
| geowearnet_e1_s2_base_20260825_050841_6e1b | S2 | ctx680 | global | M1 | 10500 | 0.8891 | 0.9036 | 0.622 |

## Final verdict (Workstreams BJ/BK)

- **Selected model:** `geowearnet_e1_s1_arch_crnn_20260825_150233_83e5_onS1_final` — 28950 params, selection score 0.8718, robust mean 0.9276.
- **Cross-simulator generalisation holds:** S1→S2, S2→S1 and mixed→{S1,S2} robust means all land in the 0.90-0.92 band (see table above), close to native-generation performance. This is evidence AGAINST a simulator-specific shortcut, not proof of real-world transfer.
- **Amplitude is not load-bearing:** dropping amplitude features (`ablate_noamp`) and destroying absolute level (`norm_cmvn`, causal per-frame CMVN) both cost only a small amount of selection score relative to the base model, consistent with the E0 verdict `AMPLITUDE_INFORMATIVE_BUT_NOT_SUFFICIENT` — the model is not simply a loudness detector.
- **Context matters up to ~500-680ms, then flattens:** ctx100 is a large, clear regression; ctx680→ctx1000 is a small further gain. Below ~500 ms context, robustness drops sharply.
- **Architecture beat width:** the smallest model in the whole sweep (CRNN, 28950 params) is also the most robust by this selection objective, ahead of every wider/deeper TCN, ConvNeXt and DSConv variant tried. Streaming for the CRNN's GRU state was implemented and parity-verified this session (`StreamingCRNN` in `deploy.py`) specifically because it was not covered by the original TCN-only streaming code -- it would have blocked shipping the actual best-scoring model.

- **CPU cost is not a constraint at this scale:** RTF 0.00265 (1 thread, batch=1), well under real-time; export to TorchScript/ONNX verified.

- **Zero-shot sim-to-real transfer is real and positive, on a proxy:** every checkpoint tested transfers to real MMCSG recordings (real Aria glasses, real humans, real rooms) with NO fine-tuning, landing in the same AUROC band as the simulated suites (see MMCSG section above). Best observed: `geowearnet_e1_s1_arch_crnn` at solo AUROC 0.9687 on 189 real held-out `eval` recordings. **This is evidence for, not proof of, Mentra transfer** -- MMCSG is Aria-glasses hardware, not Mentra, and was reached under a non-commercial research license (see `docs/geowearnet_data_license_state.md`), so it cannot itself justify training a shipped model, only evaluating one.

- **What this verdict is NOT:** the E1 main table above is still 100% simulated (S1/S2/mixed built from LibriSpeech + MUSAN). Shipping to real Mentra glasses still requires the real-capture pilot, which remains blocked on hardware access. The honest status is now `SIMULATION_CONVERGED, REAL_PROXY_TRANSFER_POSITIVE, MENTRA_HARDWARE_PILOT_PENDING` -- a real wearable-hardware transfer signal now exists (MMCSG), but no result anywhere in this document has been measured on Mentra hardware itself.


---
_Negative results are retained deliberately (Workstream BI). No result in this document has been measured on Mentra hardware._
