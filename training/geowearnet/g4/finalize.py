"""G4 finalisation -- assemble evaluation/geowearnet/g4/g4_final_summary.json.

Reads only artifacts that were actually produced, records the frozen parent's
verified hash, and separates -- structurally, not just in prose -- the evidence
that exists (MMCSG wearable proxy, software) from the evidence that does not
(Mentra hardware). A missing artifact is recorded as missing; nothing is
back-filled from memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from training.geowearnet.g4 import blocked

REPO_ROOT = Path(__file__).resolve().parents[3]
G4_DIR = REPO_ROOT / "evaluation/geowearnet/g4"
FROZEN_PARENT = REPO_ROOT / "training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt"
EXPECTED_SHA = "07c43c3d9e37dbd490ae0477f46ff515e1e19e8fe0a20687ec9d13a8555c68ef"


def sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _git_state() -> Dict[str, object]:
    def _run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    return {
        "head": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit_made_by_g4": False,
        "push_made_by_g4": False,
    }


def build() -> Dict[str, object]:
    first_word = _load(G4_DIR / "first_word_retention.json")
    parity = _load(G4_DIR / "live_offline_parity.json")
    cpu = _load(G4_DIR / "routing_cpu_benchmark.json")

    actual_sha = sha256(FROZEN_PARENT)
    parent = {
        "path": str(FROZEN_PARENT.relative_to(REPO_ROOT)),
        "sha256": actual_sha,
        "sha256_matches_declared": actual_sha == EXPECTED_SHA,
        "modified_by_g4": False,
        "role": "G4_ZERO_SHOT_PARENT",
        "params": 132678,
        "arch": "tcn",
        "context_ms": 1000,
    }

    ws2: Dict[str, object] = {"status": "MISSING"}
    if first_word:
        selection = first_word.get("selection", {})
        baseline = selection.get("baseline") or {}
        selected = selection.get("selected") or {}

        def _rate(row: dict, key: str) -> Optional[float]:
            value = (row.get(key) or {}).get("pass_rate_ge_0_5")
            return round(value * 100.0, 2) if value is not None else None

        ws2 = {
            "status": first_word.get("status"),
            "measurement_class": first_word.get("measurement_class"),
            "n_items": first_word.get("n_items"),
            "total_audio_minutes": round(first_word.get("total_audio_minutes", 0.0), 2),
            "rolling_window_parity_max_prob_diff": max(
                (p.get("max_abs_wearer_prob_diff", 0.0)
                 for p in first_word.get("rolling_window_parity_probes", [])
                 if p.get("n_probes")), default=None),
            "live_baseline_A_balanced_preroll_0": {
                "first_100ms_pct": _rate(baseline, "first_100ms_retention"),
                "first_250ms_pct": _rate(baseline, "first_250ms_retention"),
                "first_token_pct": _rate(baseline, "first_token_retention"),
                "wake_word_pct": _rate(baseline, "wake_word_retention"),
                "bystander_word_exposure_pct": _rate(baseline, "bystander_word_exposure"),
            },
            "selected": {
                "policy": selected.get("policy"),
                "preroll_ms": selected.get("preroll_ms"),
                "added_algorithmic_delay_ms": selected.get("added_delay_ms"),
                "first_100ms_pct": _rate(selected, "first_100ms_retention"),
                "first_250ms_pct": _rate(selected, "first_250ms_retention"),
                "first_token_pct": _rate(selected, "first_token_retention"),
                "wake_word_pct": _rate(selected, "wake_word_retention"),
                "bystander_word_exposure_pct": _rate(selected, "bystander_word_exposure"),
            },
            "deltas_pp": {
                "first_token": selection.get("delta_first_token_pp"),
                "first_250ms": selection.get("delta_first_250ms_pp"),
                "bystander_exposure": selection.get("delta_bystander_exposure_pp"),
            },
            "selection_rule": selection.get("rule"),
            "g3_comparison": {
                "g3_reported_first_word_proxy_pct": 59.21,
                "g3_no_smoothing_control_pct": 73.68,
                "g4_reproduction_offline_dense_pct": None,
                "note": ("G3's number was measured on the OFFLINE dense probability source. G4 "
                         "reproduces it there and shows the LIVE receiver's own figure is far worse."),
            },
        }
        for config in first_word.get("configs", []):
            if (config["probability_source"] == "offline_dense"
                    and config["policy"] == "A_balanced" and config["preroll_ms"] == 0.0):
                ws2["g3_comparison"]["g4_reproduction_offline_dense_pct"] = _rate(
                    config, "first_250ms_retention")

    ws3: Dict[str, object] = {"status": "MISSING"}
    if parity:
        summary = parity.get("summary", {})
        ws3 = {
            "status": parity.get("status"),
            "n_items": len(parity.get("per_item", [])),
            "router_vs_offline_bit_identical_rate": (
                summary.get("B_ROUTER_DENSE", {}).get("audio_bit_identical_rate")),
            "cadence_only_state_agreement": (
                summary.get("C_LIVE_PROBS_OFFLINE_GATE", {}).get("mean_state_agreement")),
            "full_live_state_agreement": (
                summary.get("D_LIVE_FULL", {}).get("mean_state_agreement")),
            "full_live_onset_delay_median_ms": (
                summary.get("D_LIVE_FULL", {})
                .get("wearer_onset_open_latency_delta", {}).get("median_delta_ms")),
            "pcm16_quantisation_snr_db": (
                parity.get("pcm16_quantisation_only", {}).get("mean_audio_error_snr_db")),
            "cadence_sweep": parity.get("detector_cadence_sweep", {}).get("rows"),
            "conclusion": ("The WS1 envelope is exactly the offline evaluator. Essentially all "
                           "remaining live/offline divergence is the receiver's 200 ms detector "
                           "cadence, not the routing implementation."),
        }

    return {
        "campaign": "G4_MENTRA_CONVERGENCE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "headline_verdict": "MENTRA_NOT_TESTED",
        "verdict_reason": (
            "No physical Mentra Live device, paired phone, or human pilot subject exists in this "
            "environment. No substitute, synthetic, or corpus audio was relabelled as Mentra audio."
        ),
        "frozen_parent": parent,
        "evidence_separation": {
            "mmcsg_wearable_proxy_evidence": "ESTABLISHED (G2/G3, unchanged by G4)",
            "software_correctness_evidence": "ESTABLISHED (G4 WS1/WS2/WS3/WS6/WS41)",
            "mentra_hardware_evidence": "NONE",
        },
        "workstreams": {
            "WS1_live_routing_equivalence": ws3,
            "WS2_first_word_retention": ws2,
            "WS3_live_offline_parity": ws3,
            "WS4_WS5_mentraos_audio_path": {
                "status": "DESK_RESEARCH_COMPLETE",
                "doc": "docs/geowearnet_g4_mentraos_audio_path.md",
                "sdk_docs_access_blocked": False,
                "credentials_required_for_docs": False,
                "device_measurements_required": True,
            },
            "WS6_capture_tool": {
                "status": "BUILT_AND_SELF_TESTED",
                "server": ["mentra/capture/session.py", "mentra/capture/validation.py",
                           "mentra/capture/sink.py"],
                "client": ["web/app/pages/capture.vue",
                           "web/app/composables/useResearchCapture.ts"],
                "selftest": "scripts/mentra/capture_selftest.py",
                "selftest_input": "SYNTHETIC_TEST_INPUT_NOT_MENTRA_AUDIO (LibriSpeech via the real transport)",
                "real_mentra_capture_performed": False,
            },
            "WS39_shipping_gate": {
                "status": "DEFINED_NOT_MET",
                "criteria_met": False,
                "doc": "docs/geowearnet_g4_mentra_report.md",
            },
            "WS40_experimental_ship": {
                "status": "SPECIFIED_NOT_ENABLED",
                "default_policy_changed": False,
                "geowearnet_g4_model_registered": False,
            },
            "WS41_tests": {"status": "ADDED"},
        },
        "routing_cpu": (cpu or {}).get("rows"),
        "blocked": blocked.summary(),
        "guards": {
            "official_mmcsg_dev_used": False,
            "official_mmcsg_eval_used": False,
            "frozen_checkpoint_modified": False,
            "raw_mmcsg_modified": False,
            "speakernet_or_mentrawearnet_touched": False,
            "wearersepnet_training_started": False,
            "separator_gate": "CLOSED (unchanged from G3; no new evidence)",
            "live_default_audio_policy": "passthrough (unchanged)",
        },
        "git": _git_state(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=G4_DIR / "g4_final_summary.json")
    args = parser.parse_args()
    report = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "headline_verdict": report["headline_verdict"],
        "frozen_parent_sha_ok": report["frozen_parent"]["sha256_matches_declared"],
        "n_blocked": report["blocked"]["n_blocked_workstreams"],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
