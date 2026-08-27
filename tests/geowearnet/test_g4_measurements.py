"""G4 WS41 -- the measurement logic behind the WS2/WS3 numbers.

These do not re-run the campaign (that needs the corpus and the frozen
checkpoint); they pin the definitions, so a future edit cannot quietly change
what "first-word retention" or "the live probability source" means and leave
the reported numbers looking comparable when they are not.
"""
from __future__ import annotations

import numpy as np
import pytest

from training.geowearnet.g4.first_word import (MAX_ADDED_DELAY_MS, PREROLL_MS,
                                               bystander_word_events, first_word_events,
                                               gate_policies, select_pareto, _mean_gain)
from training.geowearnet.g4.probsource import live_cadence, shift_gain_for_preroll


# --- the live probability source -------------------------------------------
def test_live_cadence_holds_the_last_decision_and_starts_silent():
    dense_w = np.arange(60, dtype=np.float32) / 60.0
    dense_e = np.zeros(60, dtype=np.float32)
    held_w, held_e = live_cadence(dense_w, dense_e, hop_ms=200.0)

    assert held_w.shape == dense_w.shape
    # the receiver cannot decide anything before its first 200 ms hop
    assert np.all(held_w[:20] == 0.0)
    # frames 20..39 hold the decision published at frame 19
    assert np.all(held_w[20:40] == dense_w[19])
    assert np.all(held_w[40:60] == dense_w[39])
    assert np.all(held_e == 0.0)


def test_live_cadence_hop_changes_how_often_the_decision_refreshes():
    dense = np.arange(60, dtype=np.float32) / 60.0
    fast, _ = live_cadence(dense, dense, hop_ms=50.0)
    slow, _ = live_cadence(dense, dense, hop_ms=200.0)
    assert len(np.unique(fast)) > len(np.unique(slow))


# --- pre-roll ---------------------------------------------------------------
def test_shift_gain_for_preroll_moves_the_envelope_earlier():
    gain = np.arange(20, dtype=np.float32)
    shifted = shift_gain_for_preroll(gain, 50.0)   # 5 frames
    assert shifted.shape == gain.shape
    assert np.array_equal(shifted[:15], gain[5:])
    assert np.all(shifted[15:] == gain[-1])
    assert np.array_equal(shift_gain_for_preroll(gain, 0.0), gain)


def test_preroll_grid_stays_within_the_declared_delay_budget():
    """The selection rule caps added delay; the grid may probe beyond it, but
    at least one probed value must be selectable or the rule is vacuous."""
    assert min(PREROLL_MS) == 0.0
    assert any(p <= MAX_ADDED_DELAY_MS for p in PREROLL_MS if p > 0)


# --- event definitions ------------------------------------------------------
def test_first_word_events_pick_the_first_word_after_each_onset():
    wearer = np.zeros(300, dtype=np.float32)
    wearer[100:150] = 1.0          # onset at 1.00 s
    wearer[250:280] = 1.0          # onset at 2.50 s
    words = [(1.02, 1.25, "hey"), (1.30, 1.45, "there"), (2.52, 2.70, "what")]

    events = first_word_events(wearer, words)
    assert [e["word"] for e in events] == ["hey", "what"]
    # a turn preceded by >= 1 s of wearer silence is a wake candidate
    assert events[0]["is_wake"] is True
    assert events[1]["is_wake"] is True


def test_first_word_events_marks_a_quick_resumption_as_not_a_wake():
    wearer = np.zeros(300, dtype=np.float32)
    wearer[100:150] = 1.0
    wearer[170:200] = 1.0          # only 200 ms later
    words = [(1.02, 1.20, "hey"), (1.72, 1.90, "and")]
    events = first_word_events(wearer, words)
    assert events[1]["is_wake"] is False


def test_first_word_events_ignores_onsets_with_no_labelled_word():
    wearer = np.zeros(200, dtype=np.float32)
    wearer[50:80] = 1.0
    assert first_word_events(wearer, []) == []


def test_bystander_events_exclude_overlapped_words():
    """Overlapped bystander words are the routing architecture's ceiling, not
    a cost of pre-roll -- counting them would misattribute G3's overlap
    finding to this change."""
    wearer = np.zeros(200, dtype=np.float32)
    env = np.zeros(200, dtype=np.float32)
    env[10:40] = 1.0               # bystander alone
    env[100:130] = 1.0             # bystander overlapped with the wearer
    wearer[100:130] = 1.0
    words = [(0.10, 0.35, "alone"), (1.00, 1.25, "overlapped")]

    events = bystander_word_events(wearer, env, words)
    assert [e["word"] for e in events] == ["alone"]


def test_mean_gain_is_bounded_by_the_available_envelope():
    gain = np.ones(100, dtype=np.float32)
    assert _mean_gain(gain, 0.0, 0.25) == pytest.approx(1.0)
    assert _mean_gain(gain, 5.0, 6.0) is None   # entirely past the end


# --- the predeclared selection rule ----------------------------------------
def _config(policy: str, preroll: float, token_rate: float, leak_rate: float,
            source: str = "live_200ms_held") -> dict:
    summary = lambda rate: {"n": 10, "mean": rate, "p50": rate,  # noqa: E731
                            "pass_rate_ge_0_5": rate}
    return {
        "policy": policy, "preroll_ms": preroll, "probability_source": source,
        "added_delay_ms": preroll,
        "first_100ms_retention": summary(token_rate),
        "first_250ms_retention": summary(token_rate),
        "first_token_retention": summary(token_rate),
        "wake_word_retention": summary(token_rate),
        "bystander_word_exposure": summary(leak_rate),
    }


def test_selection_rejects_a_config_that_exceeds_the_delay_budget():
    configs = [
        _config("A_balanced", 0.0, 0.20, 0.10),
        _config("A_balanced", 200.0, 0.95, 0.10),   # best, but too much delay
        _config("A_balanced", 150.0, 0.70, 0.10),
    ]
    selection = select_pareto(configs)
    assert selection["status"] == "SELECTED"
    assert selection["selected"]["preroll_ms"] == 150.0


def test_selection_rejects_a_config_that_leaks_materially_more():
    configs = [
        _config("A_balanced", 0.0, 0.20, 0.10),
        _config("A_balanced", 100.0, 0.99, 0.20),   # +10 pp leakage: refused
        _config("A_balanced", 50.0, 0.40, 0.11),
    ]
    selection = select_pareto(configs)
    assert selection["selected"]["preroll_ms"] == 50.0


def test_selection_never_picks_the_no_smoothing_control():
    configs = [
        _config("A_balanced", 0.0, 0.20, 0.10),
        _config("E_no_smoothing", 0.0, 0.99, 0.10),
    ]
    selection = select_pareto(configs)
    assert selection["selected"]["policy"] == "A_balanced"


def test_selection_uses_the_live_source_not_the_offline_one():
    configs = [
        _config("A_balanced", 0.0, 0.20, 0.10),
        _config("A_balanced", 100.0, 0.60, 0.10),
        _config("A_balanced", 100.0, 0.99, 0.10, source="offline_dense"),
    ]
    selection = select_pareto(configs)
    assert selection["selected"]["probability_source"] == "live_200ms_held"
    assert selection["baseline"]["probability_source"] == "live_200ms_held"


def test_gate_policy_grid_is_the_declared_one():
    names = [p.name for p in gate_policies()]
    assert names == ["A_balanced", "A_fast_attack", "A_early_open", "E_no_smoothing"]
