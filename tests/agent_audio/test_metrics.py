"""P1.19 -- deterministic tests for the product metrics.

Closed-form transcripts with hand-computed expected values. No ASR, no
corpus, no model.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.agent_audio.asr import ASRWord, normalize_text, tokenize
from evaluation.agent_audio.metrics import (activity_prf, aggregate, attribute_hyp_words,
                                            bootstrap_ci_rate, bystander_leakage_rate,
                                            cer, energy_leakage, wer,
                                            wearer_transcription_metrics)


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def test_normalization_is_case_and_punctuation_insensitive():
    assert normalize_text("Hey, Glasses!  What part IS this?") == "hey glasses what part is this"


def test_normalization_keeps_apostrophes():
    assert tokenize("don't") == ["don't"]


def test_normalization_splits_hyphens():
    assert tokenize("re-torque") == ["re", "torque"]


# ---------------------------------------------------------------------------
# WER / CER
# ---------------------------------------------------------------------------
def test_wer_perfect_match_is_zero():
    w = wer("the quick brown fox", "the quick brown fox")
    assert w["wer"] == 0.0 and w["hit_rate"] == 1.0


def test_wer_counts_substitution_deletion_insertion():
    # ref: a b c d   hyp: a x c d e  -> 1 sub, 0 del, 1 ins over 4 ref words
    w = wer("a b c d", "a x c d e")
    assert w["n_sub"] == 1 and w["n_del"] == 0 and w["n_ins"] == 1
    assert w["wer"] == pytest.approx(0.5)


def test_wer_full_deletion():
    w = wer("a b c d", "")
    assert w["del_rate"] == 1.0 and w["wer"] == 1.0 and w["hit_rate"] == 0.0


def test_wer_empty_reference_counts_insertions():
    w = wer("", "spurious words")
    assert w["n_ref_words"] == 0 and w["n_ins"] == 2


def test_cer_is_finer_grained_than_wer():
    """At heavy suppression WER saturates at 1.0 while CER still discriminates
    -- the stated reason TER is reported alongside WER."""
    ref = "torque specification"
    near = wer(ref, "torqu specificaton")
    assert near["wer"] == 1.0            # both words wrong
    assert 0.0 < cer(ref, "torqu specificaton") < 0.2


# ---------------------------------------------------------------------------
# BYSTANDER LEAKAGE RATE -- the first-class metric
# ---------------------------------------------------------------------------
def test_blr_zero_when_bystander_absent_from_hypothesis():
    r = bystander_leakage_rate("did you see the game last night", "what part is this")
    assert r["bystander_leakage_rate"] == 0.0
    assert r["defined"] is True and r["n_other_ref_words"] == 7


def test_blr_one_when_bystander_fully_leaks():
    r = bystander_leakage_rate("hello there friend", "hello there friend")
    assert r["bystander_leakage_rate"] == 1.0


def test_blr_partial_leak():
    r = bystander_leakage_rate("a b c d", "a b x y")
    assert r["n_other_words_leaked"] == 2
    assert r["bystander_leakage_rate"] == pytest.approx(0.5)


def test_blr_undefined_when_bystander_silent():
    """No bystander words => no leakage possible. Must be flagged undefined so
    aggregation does not average in a meaningless zero."""
    r = bystander_leakage_rate("", "wearer said something")
    assert r["defined"] is False and r["n_other_ref_words"] == 0


def test_lexical_coincidence_floor_detects_shared_vocabulary():
    """BLR's floor: words the wearer ALSO said appear in a perfectly clean
    wearer-only hypothesis and score as leakage. Without this floor, a BLR of
    0.06 looks like residual leakage when it may be complete suppression."""
    from evaluation.agent_audio.metrics import lexical_coincidence_floor
    f = lexical_coincidence_floor("the part is here", "the bolt is loose")
    # "the" and "is" are shared -> 2 of the bystander's 4 words
    assert f["lexical_coincidence_floor"] == pytest.approx(0.5)
    assert f["defined"] is True


def test_lexical_floor_zero_for_disjoint_vocabulary():
    from evaluation.agent_audio.metrics import lexical_coincidence_floor
    f = lexical_coincidence_floor("alpha bravo", "charlie delta")
    assert f["lexical_coincidence_floor"] == 0.0


def test_lexical_floor_undefined_when_bystander_silent():
    from evaluation.agent_audio.metrics import lexical_coincidence_floor
    assert lexical_coincidence_floor("anything", "")["defined"] is False


def test_clean_wearer_only_hypothesis_scores_blr_at_the_floor():
    """The property that makes the floor meaningful: a hypothesis containing
    ONLY wearer speech still scores BLR == the floor, not 0."""
    from evaluation.agent_audio.metrics import lexical_coincidence_floor
    self_t, other_t = "the part is here", "the bolt is loose"
    floor = lexical_coincidence_floor(self_t, other_t)["lexical_coincidence_floor"]
    blr = bystander_leakage_rate(other_t, self_t)["bystander_leakage_rate"]
    assert blr == pytest.approx(floor)


def test_blr_is_recall_not_precision():
    """Leakage must not be diluted by the wearer talking a lot: the same
    bystander leak in a longer hypothesis is the same harm."""
    short = bystander_leakage_rate("secret phrase", "secret phrase")
    long = bystander_leakage_rate("secret phrase", "many other wearer words secret phrase more words")
    assert short["bystander_leakage_rate"] == long["bystander_leakage_rate"] == 1.0


# ---------------------------------------------------------------------------
# wearer retention / deletion
# ---------------------------------------------------------------------------
def test_wearer_retention_and_deletion_are_complementary_on_pure_deletion():
    m = wearer_transcription_metrics("one two three four", "one two")
    assert m["wearer_deletion_rate"] == pytest.approx(0.5)
    assert m["wearer_retention"] == pytest.approx(0.5)


def test_wearer_metrics_carry_raw_counts_for_aggregation():
    """Regression: `aggregate` micro-averages corpus WER from n_sub/n_del/
    n_ins. When `wearer_transcription_metrics` omitted them, every corpus
    WER silently came out as exactly 0.0 while per-item WER looked fine."""
    m = wearer_transcription_metrics("a b c d", "a x c")
    for k in ("n_sub", "n_del", "n_ins", "n_self_ref_words", "n_self_words_retained"):
        assert k in m, f"{k} missing -> corpus aggregation would silently read 0"
    a = aggregate([m])
    assert a["wearer_wer"] == pytest.approx(m["wearer_wer"])
    assert a["wearer_wer"] > 0.0


def test_aggregate_of_single_row_matches_that_row():
    m = wearer_transcription_metrics("one two three", "one")
    m.update(bystander_leakage_rate("four five", "four"))
    a = aggregate([m])
    assert a["wearer_deletion_rate"] == pytest.approx(m["wearer_deletion_rate"])
    assert a["bystander_leakage_rate"] == pytest.approx(m["bystander_leakage_rate"])


def test_over_aggressive_gating_shows_as_deletion_not_substitution():
    m = wearer_transcription_metrics("alpha bravo charlie", "")
    assert m["wearer_deletion_rate"] == 1.0
    assert m["wearer_sub_rate"] == 0.0


# ---------------------------------------------------------------------------
# timestamp attribution
# ---------------------------------------------------------------------------
def test_attribution_assigns_words_to_true_speaker():
    n = 100
    w = np.zeros(n, np.float32); w[:50] = 1.0
    e = np.zeros(n, np.float32); e[50:] = 1.0
    words = [ASRWord("mine", 0.05, 0.25), ASRWord("theirs", 0.60, 0.80)]
    a = attribute_hyp_words(words, w, e)
    assert a["attribution_counts"]["self"] == 1
    assert a["attribution_counts"]["other"] == 1
    assert a["speaker_attribution_error_rate"] == pytest.approx(0.5)


def test_attribution_detects_overlap():
    n = 100
    w = np.ones(n, np.float32)
    e = np.ones(n, np.float32)
    a = attribute_hyp_words([ASRWord("both", 0.1, 0.3)], w, e)
    assert a["attribution_counts"]["overlap"] == 1


def test_attribution_clean_pipeline_has_zero_error():
    n = 100
    w = np.ones(n, np.float32)
    e = np.zeros(n, np.float32)
    a = attribute_hyp_words([ASRWord("x", 0.1, 0.2), ASRWord("y", 0.3, 0.4)], w, e)
    assert a["speaker_attribution_error_rate"] == 0.0


# ---------------------------------------------------------------------------
# gate activity P/R/F1
# ---------------------------------------------------------------------------
def test_activity_prf_perfect():
    t = np.array([0, 0, 1, 1, 1, 0], dtype=bool)
    m = activity_prf(t.astype(float), t)
    assert m["target_activity_f1"] == pytest.approx(1.0)


def test_activity_prf_always_open_has_perfect_recall_poor_precision():
    t = np.array([0, 0, 1, 1], dtype=bool)
    m = activity_prf(np.ones(4), t)
    assert m["target_activity_recall"] == pytest.approx(1.0)
    assert m["target_activity_precision"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# energy leakage
# ---------------------------------------------------------------------------
def test_energy_leakage_detects_bystander_only_energy():
    hop = 160
    n = 20
    w = np.zeros(n, np.float32); w[:10] = 1.0
    e = np.zeros(n, np.float32); e[10:] = 1.0
    audio = np.zeros(n * hop, dtype=np.float32)
    audio[10 * hop:] = 1.0                    # all energy in bystander-only frames
    r = energy_leakage(audio, w, e)
    assert r["energy_leak_fraction"] == pytest.approx(1.0)


def test_energy_leakage_zero_for_perfectly_gated_audio():
    hop = 160
    n = 20
    w = np.zeros(n, np.float32); w[:10] = 1.0
    e = np.zeros(n, np.float32); e[10:] = 1.0
    audio = np.zeros(n * hop, dtype=np.float32)
    audio[: 10 * hop] = 1.0
    r = energy_leakage(audio, w, e)
    assert r["energy_leak_fraction"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def test_aggregate_micro_averages_by_word_count():
    """A 1-word recording must not outweigh a 100-word one."""
    rows = [
        {"n_self_ref_words": 100, "n_del": 0, "n_sub": 0, "n_ins": 0, "n_self_words_retained": 100,
         "n_other_ref_words": 100, "n_other_words_leaked": 0},
        {"n_self_ref_words": 1, "n_del": 1, "n_sub": 0, "n_ins": 0, "n_self_words_retained": 0,
         "n_other_ref_words": 1, "n_other_words_leaked": 1},
    ]
    a = aggregate(rows)
    assert a["wearer_deletion_rate"] == pytest.approx(1 / 101)
    assert a["bystander_leakage_rate"] == pytest.approx(1 / 101)


def test_bootstrap_ci_brackets_point_estimate():
    num = [1, 2, 0, 3, 1, 0, 2, 1]
    den = [10] * 8
    pt, lo, hi = bootstrap_ci_rate(num, den, n_boot=500, seed=0)
    assert lo <= pt <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_bootstrap_ci_is_deterministic_for_fixed_seed():
    num, den = [1, 2, 3], [10, 10, 10]
    a = bootstrap_ci_rate(num, den, n_boot=200, seed=7)
    b = bootstrap_ci_rate(num, den, n_boot=200, seed=7)
    assert a == b
