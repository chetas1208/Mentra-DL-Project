from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "evaluation/geowearnet/g3/identity_causality_report.json"


def test_g3_identity_report_contains_causal_counterfactuals():
    report = json.loads(REPORT.read_text())

    assert report["official_dev_used"] is False
    assert report["verdict"] in {
        "IDENTITY_SHORTCUT_CLEARED",
        "IDENTITY_INFORMATION_PRESENT_BUT_NOT_CAUSAL",
        "IDENTITY_SHORTCUT_CONFIRMED",
        "IDENTITY_SHORTCUT_UNRESOLVED",
    }
    assert report["same_speaker_role_flip"]["n_pairs"] >= 6
    assert report["same_geometry_speaker_swap"]["n_pairs"] == report["same_speaker_role_flip"]["n_pairs"]
    assert report["same_speaker_different_content_control"]["n_pairs"] > 0
    assert report["leave_identity_out"]["per_speaker"]
    assert "final_pre_head" in report["representation_identity_baselines"]


def test_g3_identity_report_keeps_old_probe_as_noncausal_context():
    report = json.loads(REPORT.read_text())
    original = report["original_probe_context"]["original_result"]

    assert original["verdict"] == "IDENTITY_SHORTCUT_SUSPECTED"
    assert report["original_probe_context"]["not_tested_by_original"]
