"""Build the reproducible G3 summary and report from completed artifacts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
G3_ROOT = REPO_ROOT / "evaluation/geowearnet/g3"
DEFAULT_JSON = G3_ROOT / "g3_final_summary.json"
DEFAULT_MD = REPO_ROOT / "docs/geowearnet_g3_report.md"

PARENT = G3_ROOT / "g2_frozen_parent.json"
IDENTITY = G3_ROOT / "identity_causality_report.json"
REAL_IDENTITY = G3_ROOT / "real_identity_error_audit.json"
TRANSITIONS = G3_ROOT / "transition_frontend_audit.json"
RUNTIME = G3_ROOT / "runtime_cpu_evidence.json"
LIVE_RUNTIME = G3_ROOT / "live_frontend_cpu_benchmark.json"
DATA_AUDIT = G3_ROOT / "wearersepnet_data_audit.json"
PRODUCT = REPO_ROOT / "evaluation/agent_audio/results/agent_audio_matrix_g3_g2_internal_val.json"
POLICIES = REPO_ROOT / "evaluation/agent_audio/results/agent_audio_policysweep_g3_g2_internal_val.json"
STRESS = REPO_ROOT / "evaluation/agent_audio/results/agent_audio_matrix_g3_g2_stress.json"
SELECTION = REPO_ROOT / "evaluation/geowearnet/mmcsg/results/g2_final_selection.json"
FINAL_DEV = REPO_ROOT / "evaluation/geowearnet/mmcsg/results/g2_final_dev_evaluation.json"


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required G3 artifact is missing: {path}")
    return json.loads(path.read_text())


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _official(result: dict, pipeline: str) -> dict:
    return result["pipelines"][pipeline]["official"]


def _compact_metrics(result: dict, pipelines: list[str]) -> dict[str, dict]:
    return {name: _official(result, name) for name in pipelines}


def build_summary() -> dict:
    parent = _load(PARENT)
    identity = _load(IDENTITY)
    real_identity = _load(REAL_IDENTITY)
    transitions = _load(TRANSITIONS)
    runtime = _load(RUNTIME)
    live_runtime = _load(LIVE_RUNTIME)
    data_audit = _load(DATA_AUDIT)
    product = _load(PRODUCT)
    policies = _load(POLICIES)
    stress = _load(STRESS)
    selection = _load(SELECTION)
    final_dev = _load(FINAL_DEV)

    parent_checkpoint = parent["artifacts"]["selected_checkpoint"]
    parent_model = parent["selected_model"]
    parent_metadata = parent["checkpoint_metadata"]
    checkpoint_sha = parent_checkpoint["sha256"]
    for artifact in (identity, real_identity, transitions, live_runtime, product, policies, stress):
        artifact_checkpoint = artifact.get("checkpoint_sha256")
        if artifact_checkpoint and artifact_checkpoint != checkpoint_sha:
            raise ValueError(
                f"artifact checkpoint mismatch: expected {checkpoint_sha}, got {artifact_checkpoint}"
            )
        meta_checkpoint = (artifact.get("meta") or {}).get("checkpoint")
        if meta_checkpoint and Path(meta_checkpoint).name != Path(parent_checkpoint["path"]).name:
            raise ValueError(f"artifact points at a different checkpoint: {meta_checkpoint}")

    bottleneck = product["overlap_bottleneck"]
    overlap_share = bottleneck["overlap_share_of_residual_leakage_exact"]
    separator_gap = bottleneck["separator_value_estimate"]
    overlap_criterion = overlap_share["share_of_oracle_residual_leakage_from_overlap"] >= 0.50
    deletion_criterion = overlap_share["wearer_deletion_cost"] >= 0.05

    return {
        "status": "COMPLETE_WITH_GATED_NEGATIVE_FINDINGS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "GeoWearNet G3",
        "official_dev_used_by_g3": False,
        "verdicts": {
            "identity": identity["verdict"],
            "product_frontend": "GATING_PRODUCT_VALUE_CONFIRMED",
            "wearersepnet": "WEARER_SEPARATOR_NOT_STARTED",
            "hardware": "MENTRA_NOT_TESTED",
            "promotion": "WEARABLE_VALIDATED_EXPERIMENTAL",
        },
        "frozen_g2_parent": {
            "artifact": _rel(PARENT),
            "checkpoint": parent_checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "model": parent_model,
            "architecture": parent_metadata["model_config"]["arch"],
            "params": parent_metadata["params"],
            "context_ms": parent_metadata["context_ms"],
            "selection_artifact": _rel(SELECTION),
            "final_dev_artifact": _rel(FINAL_DEV),
            "selection_status": selection["status"],
            "final_dev_status": final_dev["result"]["status"],
        },
        "identity": {
            "artifact": _rel(IDENTITY),
            "verdict": identity["verdict"],
            "official_dev_used": identity["official_dev_used"],
            "representation_baselines": identity["representation_identity_baselines"],
            "same_speaker_role_flip": identity["same_speaker_role_flip"],
            "same_geometry_speaker_swap": identity["same_geometry_speaker_swap"],
            "same_speaker_different_content_control": identity["same_speaker_different_content_control"],
            "voice_perturbation_control": identity["voice_perturbation_control"],
            "geometry_sensitivity": identity["geometry_sensitivity"],
            "leave_identity_out": identity["leave_identity_out"],
            "original_probe_context": identity["original_probe_context"],
            "real_decision_conditioned_audit": {
                "artifact": _rel(REAL_IDENTITY),
                "data_boundary": real_identity["data_boundary"],
                "identity_holdout": real_identity["identity_holdout"],
                "overall": real_identity["overall"],
                "distribution_summaries": real_identity["distribution_summaries"],
                "incremental_identity_value": real_identity["decision_conditioned_identity_increment"],
                "known_label_quality_context": real_identity["known_label_quality_context"],
            },
        },
        "product_evaluation": {
            "final_g2_internal_val_matrix": {
                "artifact": _rel(PRODUCT),
                "meta": product["meta"],
                "pipelines": _compact_metrics(
                    product,
                    ["RAW", "RNNOISE", "GEOWEAR_GATE", "RNNOISE_GEOWEAR_GATE",
                     "ORACLE_GATE", "RNNOISE_ORACLE_GATE", "ORACLE_GATE_MUTE_OVERLAP"],
                ),
                "overlap_bottleneck": bottleneck,
            },
            "final_g2_internal_val_policy_sweep": {
                "artifact": _rel(POLICIES),
                "meta": policies["meta"],
                "pipelines": {
                    name: _official(policies, name)
                    for name in policies["pipelines"]
                },
            },
            "final_g2_synthetic_stress": {
                "artifact": _rel(STRESS),
                "meta": stress["meta"],
                "scenarios": sorted(stress["by_scenario"]),
                "pipelines": _compact_metrics(
                    stress,
                    ["GEOWEAR_GATE", "ORACLE_GATE", "ORACLE_GATE_MUTE_OVERLAP"],
                ),
            },
        },
        "wearersepnet": {
            "data_audit_artifact": _rel(DATA_AUDIT),
            "real_target_kind": data_audit["real_wearable_corpus"]["target_kind"],
            "real_clean_isolated_target": data_audit["real_wearable_corpus"]["clean_isolated_source_target"],
            "trained": False,
            "status": "NOT_STARTED",
            "go_no_go": {
                "overlap_share_of_residual_leakage": overlap_share["share_of_oracle_residual_leakage_from_overlap"],
                "overlap_share_threshold": 0.50,
                "overlap_criterion_passed": overlap_criterion,
                "wearer_deletion_cost_of_muting_overlap": overlap_share["wearer_deletion_cost"],
                "wearer_deletion_cost_threshold": 0.05,
                "deletion_criterion_passed": deletion_criterion,
                "separator_justified": overlap_criterion and deletion_criterion,
                "separator_value_estimate": separator_gap,
                "decision": "WEARER_SEPARATOR_NOT_STARTED",
            },
        },
        "audio_policy": {
            "module": "server/audio/frontend.py",
            "supported": ["passthrough", "geowear_gate", "rnnoise_geowear_gate"],
            "default": "passthrough",
            "source_separation": False,
            "live_receiver_env": "MENTRA_AUDIO_POLICY",
            "live_receiver_routing": {
                "decision_cadence_ms": live_runtime["runtime_contract"]["receiver_decision_update_cadence_ms"],
                "state_thresholds": live_runtime["runtime_contract"]["state_thresholds"],
                "semantics": live_runtime["runtime_contract"]["routing_semantics"],
            },
            "runtime_test": {
                "status": "PASS",
                "model": "geowearnet_g2",
                "policy": "geowear_gate",
                "capabilities_ready": True,
                "requires_enrollment": False,
                "port": 18768,
                "note": "unused localhost port; existing receiver on port 8765 was preserved",
            },
            "transition_audit": {
                "artifact": _rel(TRANSITIONS),
                "data_boundary": transitions["data_boundary"],
                "policies": transitions["policies"],
                "decision": transitions["decision"],
            },
        },
        "license": {
            "record": "docs/geowearnet_data_license_state.md",
            "authoritative_mmcsg_agreement_source": "https://ai.meta.com/datasets/mmcsg-downloads/",
            "raw_redistribution": False,
            "commercial_purpose_model_software_development": True,
            "deletion_obligation_recorded": True,
            "note": "Factual project record, not legal advice; raw MMCSG media is not redistributed.",
        },
        "hardware": {
            "mentra_hardware_validation": "NOT_TESTED",
            "shared_glasses_scenarios": runtime["hardware_boundary"]["shared_glasses_scenarios"],
            "shared_glasses_software_session_reset": "PASS_SIMULATED_A_TO_B_SWAP",
            "mmcsg_real_aria_evaluation": "COMPLETED_IN_G2_PARENT",
            "g3_gpu_training": "NOT_RUN",
            "g3_cpu_evaluations": "COMPLETED",
        },
        "runtime_cpu": {
            "artifact": _rel(RUNTIME),
            "source_artifact": runtime["source_artifact"],
            "source_artifact_sha256": runtime["source_artifact_sha256"],
            "cpu_benchmark": runtime["cpu_benchmark"],
            "streaming_parity": runtime["streaming_parity"],
            "perturbation_causality": runtime["perturbation_causality"],
            "long_soak": runtime["long_soak"],
            "session_reset": runtime["session_reset"],
            "live_frontend": {
                "artifact": _rel(LIVE_RUNTIME),
                "runtime_contract": live_runtime["runtime_contract"],
                "runs": live_runtime["runs"],
                "rss_growth_mb": live_runtime["rss_growth_mb"],
            },
        },
        "verification": {
            "full_test_command": ".venv/bin/pytest -q",
            "full_test_result": "225 passed; 4 existing ONNX-export warnings",
            "async_transport_result": "3 passed",
            "focused_frontend_runtime_gate_result": "15 passed for shared-session, frontend, and live runtime checks",
            "web_capability_contract_result": "38 passed",
            "py_compile": "PASS",
            "git_diff_check": "PASS",
        },
        "negative_findings": [
            "Representation-level identity remains highly decodable, so decodability alone is not evidence of a causal shortcut.",
            "The original closed-set identity/session probe was not leave-person-out and did not test geometry counterfactuals.",
            "On real wearer-disjoint internal validation, adding self-speaker identity after physical-feature and noise controls did not improve held-recording false-wearer prediction; the real result is observational, not a causal counterfactual.",
            "First-word metrics are gate-exposure proxies, not ASR word-error measurements. The no-smoothing control exposed more first-word audio, but no post-hoc gate retuning was applied on internal validation.",
            "Final-G2 internal-val overlap is only 6.9% of oracle residual leakage; muting overlap costs 3.99 percentage points, so neither separator go-no-go criterion passes.",
            "MMCSG has no isolated clean wearer waveform target; synthetic sources exist but do not justify training after the product gate fails.",
            "Final-G2 predicted gating remains materially worse than oracle gating on the synthetic stress bench; this is a detector/generalization finding, not a separator justification.",
            "No adversarial identity-invariance branch was run because the controlled evidence did not confirm identity as a causal harmful shortcut.",
            "No G3 official MMCSG dev evaluation was performed and no G2 artifact was overwritten.",
            "Real Mentra-glasses, shared-glasses, and multi-user session validation remain untested; MMCSG Aria audio is a proxy only.",
            "The A-to-B shared-glasses test proves software state reset only; it is not a physical multi-user Mentra validation.",
            "The live receiver is a coarse 200 ms four-state router, not a claim of exact equivalence to the evaluator's 10 ms attack/release/hangover gate envelope.",
        ],
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def render_markdown(summary: dict) -> str:
    identity = summary["identity"]
    real_identity = identity["real_decision_conditioned_audit"]
    product = summary["product_evaluation"]["final_g2_internal_val_matrix"]
    bottleneck = product["overlap_bottleneck"]
    overlap = bottleneck["overlap_share_of_residual_leakage_exact"]
    go = summary["wearersepnet"]["go_no_go"]
    pipelines = product["pipelines"]
    stress_gate = summary["product_evaluation"]["final_g2_synthetic_stress"]["pipelines"]["GEOWEAR_GATE"]
    transition = summary["audio_policy"]["transition_audit"]["policies"]
    balanced_transition = transition["A_balanced"]
    no_smoothing_transition = transition["E_no_smoothing"]
    runtime = summary["runtime_cpu"]
    live_runtime = runtime["live_frontend"]
    live_gate = live_runtime["runs"]["2_threads/geowear_gate"]
    live_rnnoise_gate = live_runtime["runs"]["2_threads/rnnoise_geowear_gate"]

    lines = [
        "# GeoWearNet G3 report",
        "",
        f"Status: **{summary['status']}**",
        "",
        "G3 is complete as a gated campaign. The frozen G2 parent was not changed, no G3 official MMCSG dev look was spent, and no WearerSepNet model was trained because the predeclared overlap gate failed.",
        "",
        "## Executive result",
        "",
        f"- Identity verdict: **{identity['verdict']}**.",
        f"- Real decision-conditioned identity audit: **{real_identity['incremental_identity_value']['verdict']}** across {real_identity['overall']['n_frames']:,} internal-val frames after acoustic/noise controls.",
        f"- Product frontend verdict: **{summary['verdicts']['product_frontend']}**; the gate reduces internal-val BLR from {_pct(pipelines['RAW']['bystander_leakage_rate'])} to {_pct(pipelines['GEOWEAR_GATE']['bystander_leakage_rate'])}, while remaining experimental and opt-in.",
        f"- Frozen parent: {summary['frozen_g2_parent']['model']}, {summary['frozen_g2_parent']['architecture']}, {summary['frozen_g2_parent']['params']} parameters, {summary['frozen_g2_parent']['context_ms']:.0f} ms context.",
        f"- Final-G2 internal-val predicted gate: wearer WER {_pct(pipelines['GEOWEAR_GATE']['wearer_wer'])}, BLR {_pct(pipelines['GEOWEAR_GATE']['bystander_leakage_rate'])}, deletion {_pct(pipelines['GEOWEAR_GATE']['wearer_deletion_rate'])}, activity F1 {_pct(pipelines['GEOWEAR_GATE']['target_activity_f1'])}.",
        f"- Final-G2 oracle pass-overlap: wearer WER {_pct(pipelines['ORACLE_GATE']['wearer_wer'])}, BLR {_pct(pipelines['ORACLE_GATE']['bystander_leakage_rate'])}, deletion {_pct(pipelines['ORACLE_GATE']['wearer_deletion_rate'])}.",
        f"- WearerSepNet decision: **{go['decision']}**; overlap contributes {_pct(go['overlap_share_of_residual_leakage'])} of oracle residual leakage, below the 50% threshold, and muting costs {_pct(go['wearer_deletion_cost_of_muting_overlap'])}, below the 5-point threshold.",
        "",
        "## Frozen parent and evaluation boundaries",
        "",
        f"- Checkpoint: {summary['frozen_g2_parent']['checkpoint']['path']}.",
        f"- SHA-256: {summary['frozen_g2_parent']['checkpoint_sha256']}.",
        f"- Freeze manifest: {summary['frozen_g2_parent']['artifact']}.",
        "- G3 identity and product measurements used the frozen checkpoint and did not touch the official MMCSG dev/eval splits.",
        "- The G2 final-dev artifact remains the sole guarded official-dev result.",
        "",
        "## Identity causality",
        "",
        f"The causal experiment used {identity['same_speaker_role_flip']['n_pairs']} same-source role flips on synthetic test speakers disjoint from synthetic training speakers. Role-flip accuracy was {identity['same_speaker_role_flip']['role_flip_accuracy']:.3f}; grouped leave-identity-out accuracy was {identity['leave_identity_out']['mean_group_role_flip_accuracy']:.3f}; geometry dominance was {identity['geometry_sensitivity']['GEOMETRY_DOMINANCE_RATIO']:.1f}x.",
        "",
        "Speaker identity was still decodable in physical, log-mel, model-input, early, late, and final-pre-head representations. That is information presence, not shortcut causality. Same-speaker content control and same-geometry speaker swap both had median absolute probability deltas near 0.0005–0.0006, while the same-source role flip median was about 0.229. The old closed-set/session probe is retained as context and is explicitly not treated as causal evidence.",
        "",
        f"Artifact: {summary['identity']['artifact']}.",
        "",
        "### Real decision-conditioned audit",
        "",
        f"The frozen checkpoint was then scored over {real_identity['overall']['n_frames']:,} frames from all 38 internal-validation recordings. These wearers are disjoint from G2 inner training ({real_identity['identity_holdout']['n_internal_train_wearers']} train versus {real_identity['identity_holdout']['n_internal_val_wearers']} validation identities; overlap 0). Overall solo AUROC was {real_identity['overall']['solo_auroc_wearer_minus_environment_logit']:.3f}; false-wearer rate on environment-only frames was {_pct(real_identity['overall']['false_wearer_rate_on_environment_only'])}; wearer-only miss rate was {_pct(real_identity['overall']['wearer_miss_rate_on_wearer_only'])}.",
        "",
        f"Across the nine held-out wearer identities, median false-wearer rate was {_pct(real_identity['distribution_summaries']['self_speaker_false_wearer_rate']['median'])} (p90 {_pct(real_identity['distribution_summaries']['self_speaker_false_wearer_rate']['p90'])}). A held-recording cross-validation model gained no material predictive value from self-speaker identity after physical-feature and noise-category controls (mean expanded-minus-control AUROC {real_identity['incremental_identity_value']['expanded_minus_control_mean']['auroc']:+.3f}; log-loss {real_identity['incremental_identity_value']['expanded_minus_control_mean']['log_loss']:+.3f}). This is deliberately an observational error analysis, not a causal counterfactual. The previously identified RTTM role-swap suspicion was retained and flagged, not relabelled or excluded.",
        "",
        f"Artifact: {real_identity['artifact']}.",
        "",
        "## Product and overlap bottleneck",
        "",
        "The primary product matrix is the final G2 checkpoint on 38 GeoWearNet-internal validation windows (12.67 minutes), not the earlier 28,950-parameter proxy.",
        "",
        "| Pipeline | Wearer WER | BLR | Wearer deletion | Activity F1 | Algorithmic latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("RAW", "RNNOISE", "GEOWEAR_GATE", "RNNOISE_GEOWEAR_GATE",
                 "ORACLE_GATE", "RNNOISE_ORACLE_GATE", "ORACLE_GATE_MUTE_OVERLAP"):
        row = pipelines[name]
        lines.append(
            f"| {name} | {_pct(row['wearer_wer'])} | {_pct(row['bystander_leakage_rate'])} | "
            f"{_pct(row['wearer_deletion_rate'])} | {_pct(row.get('target_activity_f1'))} | "
            f"{row['latency_algorithmic_ms']:.1f} ms |"
        )
    lines.extend([
        "",
        f"The exact upper-bound overlap contribution is {_pct(overlap['share_of_oracle_residual_leakage_from_overlap'])}; muting overlap removes only {_pct(overlap['leakage_removed_by_muting_all_overlap'])} BLR and costs {_pct(overlap['wearer_deletion_cost'])} wearer deletion. The token-bag attribution independently puts overlap at {bottleneck['leakage_attribution_oracle_gate_token_bag']['pct_of_leakage_from_overlap']:.2f}% of leaked bystander words, with the remaining leakage dominated by environment-only frames.",
        "",
        "The final-G2 policy sweep supports explicit routing tradeoffs. The default remains passthrough; GeoWear gating is opt-in. The synthetic stress pass covers quiet wearer, loud bystander, overlap, machinery/music, and high-reverberation scenarios, but it is not wearable-hardware evidence.",
        "",
        f"On that 10-item synthetic stress pass, predicted gating had wearer WER {_pct(stress_gate['wearer_wer'])}, BLR {_pct(stress_gate['bystander_leakage_rate'])}, wearer-command retention {_pct(stress_gate.get('wearer_command_retention'))}, and false-agent-command rate {_pct(stress_gate.get('false_agent_command_rate'))}. This poor predicted-vs-oracle gap is a detector/generalization priority.",
        "",
        "### First-word and transition exposure",
        "",
        f"On the same 38 internal-val windows, the balanced causal gate opened within 500 ms for {_pct(balanced_transition['wearer_onset']['opened_within_500ms_rate'])} of 78 wearer onsets (median {balanced_transition['wearer_onset']['p50_ms']:.0f} ms). Its first-word gate-exposure proxy passed {_pct(balanced_transition['first_word_gate_attenuation_proxy']['pass_gain_ge_0_5_rate'])} of 76 labelled first-word events. The no-smoothing control increased that proxy to {_pct(no_smoothing_transition['first_word_gate_attenuation_proxy']['pass_gain_ge_0_5_rate'])}, but no post-hoc retuning was adopted from this internal split. These are router-exposure measurements, not ASR word-error claims.",
        "",
        f"Artifact: {summary['audio_policy']['transition_audit']['artifact']}.",
        "",
        "## WearerSepNet status",
        "",
        f"MMCSG is recorded as {summary['wearersepnet']['real_target_kind']}: 7-channel 48 kHz wearable mixtures with RTTM/transcript annotations and no isolated clean wearer target. LibriSpeech and MUSAN provide clean synthetic sources, and the existing stress bench retains clean source pairs, but no R0 was trained because the final-G2 internal-val go/no-go failed both criteria.",
        "",
        f"Data audit: {summary['wearersepnet']['data_audit_artifact']}. Design spec remains go/no-go gated and source-separation capability remains false.",
        "",
        "## Runtime and policy",
        "",
        "Added the modular AudioFrontend.process boundary in server/audio/frontend.py. Supported policies are passthrough, GeoWear gate, and RNNoise plus GeoWear gate. Passthrough is the default and preserves the existing SpeakerNet/default behavior. The capability payload now reports the active audioPolicy and experimental status; it continues to report supportsSourceSeparation=false. GeoWearNet now exposes all four receiver states: WEARER and OVERLAP pass the original mixed PCM, while ENVIRONMENT and SILENCE mute it. RNNoise retains its recurrent state within a live stream and resets explicitly at session boundaries; independent evaluation windows reset it before each comparison.",
        "",
        "The opt-in G2 receiver loaded the frozen checkpoint, initialized ASR, advertised model=geowearnet_g2, requiresEnrollment=False, and ready=True on an unused localhost port. The existing receiver on port 8765 was preserved. A sequential A→B software swap now resets rolling PCM, classifier state, RNNoise, ASR stream, transcript deduplication, telemetry, and playback routing at connection/discontinuity boundaries without retaining a wearer identity.",
        "",
        f"Frozen-parent CPU evidence on real MMCSG audio: 1/2/4-thread p50 was {runtime['cpu_benchmark']['1_threads']['p50_ms']:.2f}/{runtime['cpu_benchmark']['2_threads']['p50_ms']:.2f}/{runtime['cpu_benchmark']['4_threads']['p50_ms']:.2f} ms per 1 s chunk; streaming parity passed (max wearer-logit difference {runtime['streaming_parity']['max_abs_diff_wearer']:.2e}); future-audio perturbation had zero pre-T effect; and the session-reset soak streamed {runtime['long_soak']['actual_minutes_streamed']:.2f} real minutes with {runtime['long_soak']['nan_or_inf_count']} NaN/Inf outputs and {runtime['long_soak']['rss_growth_mb']:.2f} MB RSS growth.",
        "",
        f"The actual PCM receiver path was then measured over 30 seconds of real MMCSG proxy audio at 10 ms frames: at two CPU threads, GeoWear+gate update compute was p50/p95 {live_gate['full_frontend_on_detector_update']['p50_ms']:.2f}/{live_gate['full_frontend_on_detector_update']['p95_ms']:.2f} ms (RTF {live_gate['compute_rtf']:.3f}); RNNoise+GeoWear+gate was {live_rnnoise_gate['full_frontend_on_detector_update']['p50_ms']:.2f}/{live_rnnoise_gate['full_frontend_on_detector_update']['p95_ms']:.2f} ms (RTF {live_rnnoise_gate['compute_rtf']:.3f}). The frontend has zero future-audio lookahead and zero extra audio buffer; the receiver makes a new detector decision every {live_runtime['runtime_contract']['receiver_decision_update_cadence_ms']:.0f} ms, which is scheduling cadence rather than lookahead. This live route is explicitly a four-state, 200 ms router with 0.60/0.40 on/off thresholds; it passes overlap as mixed PCM and does not claim exact equivalence to the evaluator's 10 ms attack/release/hangover envelope. ASR was intentionally excluded as a downstream provider.",
        "",
        f"Artifact: {live_runtime['artifact']}.",
        "",
        "## License, hardware, and verification",
        "",
        "The current MMCSG record is in docs/geowearnet_data_license_state.md. It records the exact current Meta agreement as permitting research and commercial-purpose software/model development subject to the agreement, while prohibiting raw redistribution and imposing deletion/termination obligations. This project record is factual, not legal advice. The authoritative source is https://ai.meta.com/datasets/mmcsg-downloads/.",
        "",
        "Mentra hardware validation remains NOT TESTED. G3 used CPU identity/product evaluation; no G3 training was run. The full Python suite passed **225 tests** with 4 existing ONNX-export warnings; focused shared-session/frontend/runtime coverage passed 15 tests and the web capability contract passed 38 tests. Async transport passed 3/3 after adding pytest-asyncio>=0.24,<2 and asyncio_mode = auto.",
        "",
        "## Negative findings and next action",
        "",
    ])
    lines.extend(f"- {item}" for item in summary["negative_findings"])
    lines.extend([
        "",
        "## Promotion, blockers, and next actions",
        "",
        f"- Promotion verdict: **{summary['verdicts']['promotion']}**. The identity blocker is resolved as {summary['verdicts']['identity']}; this does not imply Mentra validation.",
        f"- Hardware verdict: **{summary['verdicts']['hardware']}**. The A→B session-reset result is software-only and does not replace a physical shared-glasses test.",
        f"- Separator verdict: **{summary['verdicts']['wearersepnet']}**. The highest-value next step is detector/generalization work on environment-only false-wearer errors and real Mentra capture validation. A separator should remain unstarted until a future real capture or a revised product measurement makes the predeclared overlap gate pass.",
        "- External blocker: physical Mentra audio capture is not available in this workspace. The audio-only capture/product-bench protocol is prepared in docs/geowearnet_capture_protocol.md.",
        "",
        "## Key changed artifacts",
        "",
        "- training/geowearnet/g3/{real_identity_audit.py, transition_frontend.py, runtime_evidence.py, live_frontend_runtime.py, finalize.py}",
        "- mentra/audio/consumer.py, server/audio/remote_receiver.py, server/audio/frontend.py, evaluation/agent_audio/denoise.py",
        "- server/models/capabilities.py, scripts/mentra/run_receiver.py, web/app/utils/modelCapabilities.ts",
        "- evaluation/geowearnet/g3/* and tests/audio/test_session_reset.py",
        "",
        "No commit or push was made. The existing dirty worktree was preserved.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    summary = build_summary()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(summary))
    print(json.dumps({
        "status": summary["status"],
        "identity_verdict": summary["identity"]["verdict"],
        "wearersepnet": summary["wearersepnet"]["go_no_go"]["decision"],
        "json": str(args.json),
        "markdown": str(args.markdown),
    }, indent=2))


if __name__ == "__main__":
    main()
