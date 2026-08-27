from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "evaluation/geowearnet/g3/real_identity_error_audit.json"


def test_real_identity_audit_is_wearer_disjoint_and_decision_conditioned():
    report = json.loads(REPORT.read_text())

    assert report["status"] == "SCORED"
    assert report["official_dev_used"] is False
    assert report["official_eval_used"] is False
    assert report["identity_holdout"]["wearer_identity_overlap_train_val"] == 0
    assert report["overall"]["n_environment_only_frames"] > 0
    assert report["per_self_speaker"]
    assert report["per_recording_session"]
    assert report["decision_conditioned_identity_increment"]["status"] == "SCORED"
    assert report["decision_conditioned_identity_increment"]["verdict"] in {
        "MATERIAL_INCREMENTAL_ASSOCIATION",
        "NO_MATERIAL_HELD_RECORDING_IMPROVEMENT",
    }


def test_real_identity_audit_retains_known_label_anomaly_without_relabeling():
    report = json.loads(REPORT.read_text())

    assert report["known_label_quality_context"]["status"] == "RETAINED_NOT_RELABELED_OR_EXCLUDED"
