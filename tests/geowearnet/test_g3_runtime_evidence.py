from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "evaluation/geowearnet/g3/runtime_cpu_evidence.json"
LIVE_REPORT = REPO_ROOT / "evaluation/geowearnet/g3/live_frontend_cpu_benchmark.json"


def test_runtime_evidence_keeps_frozen_parent_cpu_and_causality_results():
    report = json.loads(REPORT.read_text())

    assert report["status"] == "VERIFIED_FROZEN_PARENT_EVIDENCE"
    assert {"1_threads", "2_threads", "4_threads"} <= set(report["cpu_benchmark"])
    assert report["streaming_parity"]["pass"] is True
    assert report["perturbation_causality"]["causality_pass"] is True
    assert report["long_soak"]["nan_or_inf_count"] == 0
    assert report["session_reset"]["status"] == "PASS_BY_EXECUTED_REAL_AUDIO_SOAK"


def test_live_frontend_benchmark_covers_every_supported_policy_and_cpu_count():
    report = json.loads(LIVE_REPORT.read_text())

    assert report["status"] == "SCORED"
    assert report["official_dev_used"] is False
    assert report["official_eval_used"] is False
    assert report["runtime_contract"]["future_audio_lookahead_ms"] == 0.0
    assert report["runtime_contract"]["frontend_extra_audio_buffer_ms"] == 0.0
    assert report["runtime_contract"]["state_thresholds"] == {
        "wearer_on": 0.60,
        "wearer_off": 0.40,
        "environment_on": 0.60,
        "environment_off": 0.40,
    }
    assert "four-state, 200 ms state router" in report["runtime_contract"]["routing_semantics"]
    for threads in (1, 2, 4):
        for policy in ("passthrough", "geowear_gate", "rnnoise_geowear_gate"):
            run = report["runs"][f"{threads}_threads/{policy}"]
            assert run["detector_updates"] > 0
            assert run["compute_rtf"] < 1.0
