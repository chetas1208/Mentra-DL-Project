"""P1.19 -- agent command metric tests (P1.8)."""
from __future__ import annotations

import pytest

from evaluation.agent_audio.commands import (BYSTANDER_PHRASES, COMMAND_SET, WAKE_PHRASE,
                                             aggregate_command_metrics, command_metrics,
                                             fuzzy_contains)
from evaluation.agent_audio.asr import tokenize


def test_fuzzy_contains_exact():
    m = fuzzy_contains(tokenize("hey glasses what part is this"),
                       tokenize("what part is this"))
    assert m.present and m.best_error == 0.0


def test_fuzzy_contains_tolerates_one_asr_error():
    m = fuzzy_contains(tokenize("hey glasses what part is thus"),
                       tokenize("what part is this"))
    assert m.present, "a single ASR error must not destroy command detection"


def test_fuzzy_contains_rejects_unrelated_text():
    m = fuzzy_contains(tokenize("i am heading out for lunch"),
                       tokenize("what part is this"))
    assert not m.present


def test_fuzzy_contains_empty_hypothesis():
    assert not fuzzy_contains([], tokenize("order a new one")).present


# ---------------------------------------------------------------------------
# the two headline command metrics
# ---------------------------------------------------------------------------
def test_wearer_command_retention_perfect_pipeline():
    m = command_metrics("hey glasses what part is this",
                        wearer_commands=["hey glasses what part is this"],
                        bystander_commands=[])
    assert m["wearer_command_retention"] == 1.0
    assert m["n_wearer_commands_retained"] == 1


def test_wearer_command_lost_by_over_aggressive_gating():
    m = command_metrics("", wearer_commands=["hey glasses what part is this"],
                        bystander_commands=[])
    assert m["wearer_command_retention"] == 0.0


def test_false_agent_command_rate_detects_bystander_command():
    """The expensive failure: a coworker's command reaches the agent."""
    m = command_metrics("hey glasses order a new one",
                        wearer_commands=[],
                        bystander_commands=["hey glasses order a new one"])
    assert m["false_agent_command_rate"] == 1.0
    assert m["n_false_agent_commands"] == 1


def test_false_agent_command_rate_zero_when_bystander_suppressed():
    m = command_metrics("hey glasses what part is this",
                        wearer_commands=["hey glasses what part is this"],
                        bystander_commands=["did you see the game last night"])
    assert m["false_agent_command_rate"] == 0.0
    assert m["wearer_command_retention"] == 1.0


def test_ideal_pipeline_retains_wearer_and_blocks_bystander():
    """The product target state, expressed as a test."""
    m = command_metrics("hey glasses what part is this",
                        wearer_commands=["hey glasses what part is this"],
                        bystander_commands=["hey glasses order a new one"])
    assert m["wearer_command_retention"] == 1.0
    assert m["false_agent_command_rate"] == 0.0


# ---------------------------------------------------------------------------
# wake phrase
# ---------------------------------------------------------------------------
def test_wake_phrase_retained_when_wearer_said_it():
    m = command_metrics("hey glasses take a picture of this",
                        wearer_commands=["hey glasses take a picture of this"],
                        bystander_commands=[])
    assert m["wake_phrase_in_hypothesis"] and m["wake_phrase_retained"]
    assert not m["wake_phrase_falsely_injected"]


def test_wake_phrase_falsely_injected_from_bystander():
    m = command_metrics("hey glasses take a picture of this",
                        wearer_commands=[],
                        bystander_commands=["hey glasses take a picture of this"])
    assert m["wake_phrase_falsely_injected"] is True


def test_wake_phrase_absent_when_muted():
    m = command_metrics("", wearer_commands=["hey glasses order a new one"],
                        bystander_commands=[])
    assert not m["wake_phrase_in_hypothesis"]


# ---------------------------------------------------------------------------
# undefined cases + aggregation
# ---------------------------------------------------------------------------
def test_metrics_undefined_when_no_commands_present():
    m = command_metrics("some chatter", [], [])
    assert m["wearer_command_retention"] is None
    assert m["false_agent_command_rate"] is None


def test_aggregate_command_metrics_micro_averages():
    rows = [
        {"n_wearer_commands": 2, "n_wearer_commands_retained": 2,
         "n_bystander_commands": 2, "n_false_agent_commands": 0,
         "wake_phrase_falsely_injected": False},
        {"n_wearer_commands": 1, "n_wearer_commands_retained": 0,
         "n_bystander_commands": 1, "n_false_agent_commands": 1,
         "wake_phrase_falsely_injected": True},
    ]
    a = aggregate_command_metrics(rows)
    assert a["wearer_command_retention"] == pytest.approx(2 / 3)
    assert a["false_agent_command_rate"] == pytest.approx(1 / 3)
    assert a["n_wake_phrase_falsely_injected"] == 1


# ---------------------------------------------------------------------------
# the phrase sets themselves
# ---------------------------------------------------------------------------
def test_command_set_covers_the_briefed_examples():
    texts = " | ".join(c["text"] for c in COMMAND_SET)
    assert "what part is this" in texts
    assert "replacement in stock" in texts
    assert "repair procedure" in texts


def test_bystander_set_contains_command_shaped_traps():
    shaped = [b for b in BYSTANDER_PHRASES if b["command_shaped"]]
    assert len(shaped) >= 3, "need bystander phrases that LOOK like commands"


def test_wake_phrase_constant_is_used_by_command_set():
    assert any(c["text"].startswith(WAKE_PHRASE) for c in COMMAND_SET)
