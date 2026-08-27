from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "evaluation/geowearnet/g3/transition_frontend_audit.json"


def test_transition_audit_uses_internal_val_and_labels_first_word_metric_honestly():
    report = json.loads(REPORT.read_text())

    assert report["status"] == "SCORED"
    assert report["official_dev_used"] is False
    assert report["official_eval_used"] is False
    assert report["n_items"] == 38
    balanced = report["policies"]["A_balanced"]
    first_word = balanced["first_word_gate_attenuation_proxy"]
    assert balanced["wearer_onset"]["n_events"] > 0
    assert first_word["n_first_words"] > 0
    assert "not ASR word error" in first_word["definition"]
    assert 0.0 <= first_word["pass_gain_ge_0_5_rate"] <= 1.0
