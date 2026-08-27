"""G4 WS41 -- the blocked-workstream registry must stay actionable.

The registry is the campaign's main deliverable for everything hardware
depends on. Its value is entirely in being specific, so these tests fail if an
entry decays into a vague "needs hardware" note, or if a blocked item quietly
starts claiming a Mentra result.
"""
from __future__ import annotations

import re

import pytest

from training.geowearnet.g4 import blocked

FORBIDDEN_VAGUENESS = re.compile(r"\bTBD\b|\bTODO\b|\bfigure out\b|\bsomehow\b", re.IGNORECASE)


def test_every_entry_names_a_reason_a_requirement_and_a_consumer():
    for entry in blocked.entries():
        for field in ("id", "workstream", "status", "why", "needs", "consumes",
                      "then_run", "engineering_remaining"):
            assert entry.get(field), f"{entry.get('id')} is missing {field}"
        assert len(str(entry["why"])) > 40, f"{entry['id']}: 'why' is too vague to act on"
        assert not FORBIDDEN_VAGUENESS.search(str(entry["why"]))
        assert not FORBIDDEN_VAGUENESS.search(str(entry["then_run"]))


def test_entry_ids_are_unique_and_well_formed():
    ids = [e["id"] for e in blocked.entries()]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"G4-B\d{2}", i) for i in ids)


def test_every_status_is_an_explicit_blocked_or_partial_state():
    for entry in blocked.entries():
        status = entry["status"]
        assert (status == blocked.BLOCK_REASON
                or status.endswith(blocked.BLOCK_REASON)
                or status == "PARTIALLY_ANSWERED_BY_DESK_RESEARCH"), status


def test_no_entry_claims_a_mentra_result():
    """A blocked item must never carry a measured Mentra number."""
    for entry in blocked.entries():
        text = str(entry).lower()
        for phrase in ("we measured on mentra", "mentra wer", "mentra auroc",
                       "on mentra audio we", "mentra-validated"):
            assert phrase not in text, f"{entry['id']} appears to claim a Mentra result"


def test_the_first_pilot_is_concrete_enough_to_book():
    pilot = blocked.PILOT_SMOKE
    assert pilot["people"] == 5
    assert pilot["minutes_each"]
    assert len(pilot["conditions_per_person"]) >= 4
    # the non-negotiable product constraints must be spelled out in the protocol
    joined = " ".join(pilot["constraints"]).lower()
    assert "no voice enrollment" in joined
    assert "camera off" in joined
    assert "imu off" in joined
    assert "same physical glasses" in joined


def test_base_hardware_requirement_lists_ready_software_and_credential_state():
    base = blocked.BASE_HARDWARE_REQUIREMENT
    assert base["device"] and base["phone"]
    assert base["software_already_ready"], "the point is that no engineering blocks the pilot"
    assert "none needed" in base["account_or_credentials"].lower()
    assert any("capture.vue" in s for s in base["software_already_ready"])
    assert any("--capture-dir" in s for s in base["software_already_ready"])


def test_summary_is_complete_and_names_one_highest_value_unblock():
    summary = blocked.summary()
    assert summary["n_blocked_workstreams"] == len(blocked.entries()) == 21
    assert "single_highest_value_unblock" in summary
    assert "Mentra Live" in summary["single_highest_value_unblock"]
    assert "five people" in summary["single_highest_value_unblock"]


def test_separator_gate_entry_is_still_closed():
    entry = next(e for e in blocked.entries() if e["id"] == "G4-B18")
    assert entry["current_state"].startswith("CLOSED")


@pytest.mark.parametrize("entry_id", ["G4-B12", "G4-B13"])
def test_adaptation_entries_are_gated_on_a_zero_shot_verdict_first(entry_id):
    entry = next(e for e in blocked.entries() if e["id"] == entry_id)
    text = str(entry).lower()
    assert "zero-shot" in text or "60" in text
