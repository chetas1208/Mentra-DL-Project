"""P1.19 -- tests for overlap analysis (P1.6), the synthetic stress bench
(P1.7), the pipeline matrix (P1.5), and the WearerSepNet spec-only guarantee
(P1.14)."""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.agent_audio import stressbench as SB
from evaluation.agent_audio.gate import GatePolicy
from evaluation.agent_audio.overlap import (bystander_leakage_ceiling,
                                            overlap_attributable_leakage,
                                            separator_value_estimate, state_array,
                                            state_occupancy, wearer_deletion_ceiling,
                                            word_states)
from evaluation.agent_audio.pipelines import (PIPELINES, EvalItem, PipelineContext,
                                              run_pipeline)

SR = 16000


# ---------------------------------------------------------------------------
# P1.6 overlap analysis
# ---------------------------------------------------------------------------
def test_state_array_encodes_four_states():
    w = np.array([0, 1, 0, 1], np.float32)
    e = np.array([0, 0, 1, 1], np.float32)
    assert list(state_array(w, e)) == [0, 1, 2, 3]


def test_state_occupancy_sums_to_one():
    w = np.array([0, 1, 0, 1] * 10, np.float32)
    e = np.array([0, 0, 1, 1] * 10, np.float32)
    occ = state_occupancy(w, e)["occupancy"]
    assert sum(occ.values()) == pytest.approx(1.0)
    assert occ["11_overlap"] == pytest.approx(0.25)


def test_word_states_uses_dominant_state_over_span():
    w = np.zeros(100, np.float32); w[:50] = 1
    e = np.zeros(100, np.float32); e[50:] = 1
    st = word_states([(0.0, 0.4, "mine"), (0.6, 0.9, "theirs")], w, e)
    assert st == [1, 2]


def test_overlap_attribution_identifies_state_11_share():
    """The P1.6 headline number."""
    rows = [{"leaked_words_by_state": [0, 0, 2, 8],
             "other_ref_words_by_state": [0, 0, 10, 10]}]
    r = overlap_attributable_leakage(rows)
    assert r["total_leaked_words"] == 10
    assert r["pct_of_leakage_from_overlap"] == pytest.approx(80.0)
    assert r["fraction_of_leakage_from_env_only"] == pytest.approx(0.2)


def test_overlap_attribution_reports_per_state_leakage_rate():
    """Distinguishes 'overlap is leaky' from 'overlap is merely common'."""
    rows = [{"leaked_words_by_state": [0, 0, 1, 9],
             "other_ref_words_by_state": [0, 0, 100, 10]}]
    r = overlap_attributable_leakage(rows)
    assert r["leakage_rate_by_state"]["11_overlap"] == pytest.approx(0.9)
    assert r["leakage_rate_by_state"]["01_environment_only"] == pytest.approx(0.01)


def test_overlap_attribution_handles_zero_leakage():
    r = overlap_attributable_leakage([{"leaked_words_by_state": [0, 0, 0, 0],
                                       "other_ref_words_by_state": [0, 0, 5, 5]}])
    assert r["pct_of_leakage_from_overlap"] is None


def test_received_words_by_state_separates_pure_leak_from_ambiguous():
    """Timestamp attribution: state-01 words are unambiguously leakage;
    state-11 words cannot be assigned to a speaker without a separator."""
    from evaluation.agent_audio.overlap import received_words_by_state
    r = received_words_by_state([{"hyp_words_by_state": [1, 50, 20, 5]}])
    assert r["pure_leakage_words_env_only"] == 20
    assert r["ambiguous_words_overlap"] == 5
    assert r["max_share_of_non_wearer_content_from_overlap"] == pytest.approx(5 / 25)
    assert r["frac_by_state"]["10_wearer_only"] == pytest.approx(50 / 76)


def test_received_words_by_state_handles_empty():
    from evaluation.agent_audio.overlap import received_words_by_state
    r = received_words_by_state([{"hyp_words_by_state": [0, 0, 0, 0]}])
    assert r["n_hyp_words"] == 0


def test_overlap_share_upper_bound_is_optimistic_toward_the_separator():
    """The bound must ASSUME every overlap word is a bystander word, so that a
    small value is strong evidence against needing a separator."""
    from evaluation.agent_audio.overlap import received_words_by_state
    r = received_words_by_state([{"hyp_words_by_state": [0, 100, 90, 10]}])
    assert r["max_share_of_non_wearer_content_from_overlap"] == pytest.approx(0.1)


def test_aggregate_carries_all_attribution_fractions():
    """Regression: only `attribution_frac_other` was listed in the aggregate
    key set, so self/overlap/none silently vanished from every summary."""
    from evaluation.agent_audio.metrics import aggregate
    row = {"attribution_frac_self": 0.5, "attribution_frac_other": 0.3,
           "attribution_frac_overlap": 0.1, "attribution_frac_none": 0.1}
    a = aggregate([row])
    for k in row:
        assert k in a, f"{k} dropped by aggregate()"


def test_leakage_ceiling_is_overlap_share_of_bystander_words():
    rows = [{"other_ref_words_by_state": [0, 0, 30, 10]}]
    c = bystander_leakage_ceiling(rows)
    assert c["min_possible_bystander_leakage_rate_pass_overlap"] == pytest.approx(0.25)
    assert c["min_possible_bystander_leakage_rate_mute_overlap"] == 0.0


def test_wearer_deletion_ceiling_is_cost_of_muting_overlap():
    rows = [{"self_ref_words_by_state": [0, 80, 0, 20]}]
    c = wearer_deletion_ceiling(rows)
    assert c["min_wearer_deletion_rate_if_overlap_muted"] == pytest.approx(0.2)
    assert c["min_wearer_deletion_rate_if_overlap_passed"] == 0.0


def test_separator_value_estimate_quantifies_the_corner_gap():
    """A separator is only worth building if BOTH gaps are large."""
    v = separator_value_estimate(
        {"bystander_leakage_rate": 0.40, "wearer_deletion_rate": 0.02},
        {"bystander_leakage_rate": 0.05, "wearer_deletion_rate": 0.25})
    assert v["leakage_reducible_by_muting"] == pytest.approx(0.35)
    assert v["wearer_deletion_cost_of_muting"] == pytest.approx(0.23)
    assert v["separator_target_corner"] == {"bystander_leakage_rate": 0.05,
                                            "wearer_deletion_rate": 0.02}


def test_separator_value_estimate_shows_no_value_when_muting_is_free():
    """If muting overlap costs nothing, just mute -- no separator needed."""
    v = separator_value_estimate(
        {"bystander_leakage_rate": 0.40, "wearer_deletion_rate": 0.02},
        {"bystander_leakage_rate": 0.05, "wearer_deletion_rate": 0.021})
    assert v["wearer_deletion_cost_of_muting"] < 0.01


# ---------------------------------------------------------------------------
# P1.7 synthetic stress bench
# ---------------------------------------------------------------------------
def _bench_available():
    return SB.availability_report()["status"] == "AVAILABLE"


def test_bench_scenarios_cover_the_briefed_list():
    names = {s.name for s in SB.scenario_suite()}
    for required in ("wearer_only", "coworker_only", "wearer_plus_machinery",
                     "wearer_plus_loud_coworker", "wearer_coworker_overlap",
                     "wearer_machinery_coworker", "quiet_wearer", "loud_bystander",
                     "music_radio", "high_reverberation"):
        assert required in names, f"missing required scenario {required}"


def test_bench_is_labelled_synthetic_everywhere():
    """P1.7: must NEVER be presented as real autobody validation."""
    assert "NOT real autobody validation" in SB.DISCLAIMER
    assert "SYNTHETIC" in SB.DISCLAIMER
    assert SB.availability_report()["DISCLAIMER"] == SB.DISCLAIMER


def test_energy_vad_returns_zeros_for_silence():
    """Regression: a purely relative threshold marks EVERY frame of an
    all-zero signal active, which silently inverted the labels for
    single-speaker scenarios and made `coworker_only` unbuildable."""
    assert SB.energy_vad(np.zeros(SR, np.float32)).sum() == 0.0


def test_energy_vad_detects_real_speech_region():
    x = np.zeros(SR, np.float32)
    x[SR // 2:] = np.random.default_rng(0).normal(0, 0.1, SR // 2)
    v = SB.energy_vad(x)
    assert v[:40].sum() == 0 and v[60:].mean() > 0.8


def test_bench_is_deterministic():
    if not _bench_available():
        pytest.skip("local corpora unavailable")
    a = SB.build_item(SB.scenario_suite()[0], seed=42, duration_s=12.0)
    b = SB.build_item(SB.scenario_suite()[0], seed=42, duration_s=12.0)
    assert np.array_equal(a.audio, b.audio), "same seed must give identical audio"
    assert a.self_text == b.self_text


def test_bench_preserves_clean_sources():
    """Required so leakage/retention can be measured exactly, not inferred."""
    if not _bench_available():
        pytest.skip("local corpora unavailable")
    it = SB.build_item(SB.scenario_suite()[4], seed=7, duration_s=12.0)
    assert it is None or (it.clean_wearer is not None and it.clean_bystander is not None)


def test_overlap_scenario_actually_produces_state_11():
    if not _bench_available():
        pytest.skip("local corpora unavailable")
    sc = [s for s in SB.scenario_suite() if s.name == "wearer_coworker_overlap"][0]
    got = False
    for seed in range(20):
        it = SB.build_item(sc, seed=seed, duration_s=12.0)
        if it is None:
            continue
        st = state_array(it.wearer_active, it.env_active)
        if (st == 3).mean() > 0.05:
            got = True
            break
    assert got, "the overlap scenario never produced meaningful state 11"


def test_sequential_scenario_produces_little_overlap():
    if not _bench_available():
        pytest.skip("local corpora unavailable")
    sc = [s for s in SB.scenario_suite() if s.name == "wearer_plus_loud_coworker"][0]
    it = SB.build_item(sc, seed=3, duration_s=12.0)
    if it is None:
        pytest.skip("no fitting utterance pair at this seed")
    st = state_array(it.wearer_active, it.env_active)
    assert (st == 3).mean() < 0.10, "turn-taking scenario should be mostly non-overlap"


def test_bench_audio_does_not_clip():
    if not _bench_available():
        pytest.skip("local corpora unavailable")
    for it in SB.build_suite(seeds_per_scenario=1, duration_s=12.0):
        assert np.max(np.abs(it.audio)) <= 1.0 + 1e-6, f"{it.scenario} clipped"


def test_bench_transcript_matches_untruncated_audio():
    """Regression: utterances longer than the window were truncated in AUDIO
    while keeping their FULL transcript, silently inflating every deletion
    and WER number."""
    if not _bench_available():
        pytest.skip("local corpora unavailable")
    libri = SB._librispeech_index()
    rng = np.random.default_rng(0)
    u = SB._pick_fitting(libri, rng, max_dur_s=5.0)
    assert u is not None and u["duration_s"] <= 5.0


# ---------------------------------------------------------------------------
# P1.5 pipeline matrix mechanics
# ---------------------------------------------------------------------------
def _item(n_frames=100):
    hop = 160
    w = np.zeros(n_frames, np.float32); w[:50] = 1
    e = np.zeros(n_frames, np.float32); e[50:] = 1
    return EvalItem(item_id="t", audio=np.ones(n_frames * hop, np.float32),
                    wearer_active=w, env_active=e, self_text="a b", other_text="c d")


def test_all_six_required_pipelines_are_defined():
    for required in ("RAW", "RNNOISE", "GEOWEAR_GATE", "RNNOISE_GEOWEAR_GATE",
                     "ORACLE_GATE", "RNNOISE_ORACLE_GATE"):
        assert required in PIPELINES


def test_raw_pipeline_is_bit_identical_passthrough():
    it = _item()
    ctx = PipelineContext(predictor=None, policy=GatePolicy())
    p = run_pipeline("RAW", it, ctx)
    assert np.array_equal(p.audio, it.audio)
    assert p.latency["latency_algorithmic_ms"] == 0.0


def test_oracle_pipeline_uses_ground_truth_not_the_detector():
    it = _item()
    ctx = PipelineContext(predictor=None, policy=GatePolicy())
    p = run_pipeline("ORACLE_GATE", it, ctx)
    assert p.detector_input == "oracle"
    assert p.p_wearer is not None
    assert np.array_equal(p.p_wearer, it.wearer_active)


def test_oracle_gate_suppresses_bystander_only_region():
    it = _item()
    ctx = PipelineContext(predictor=None,
                          policy=GatePolicy(name="t", attack_ms=0.0, release_ms=0.0,
                                            min_state_ms=0.0, hangover_ms=0.0))
    p = run_pipeline("ORACLE_GATE", it, ctx)
    assert p.gate_gain[:50].mean() > 0.9
    assert p.gate_gain[55:].mean() < 0.1


def test_denoised_pipelines_record_detector_input_as_denoised():
    it = _item()
    ctx = PipelineContext(predictor=None, policy=GatePolicy())
    p = run_pipeline("RNNOISE_GEOWEAR_GATE", it, ctx)
    assert p.detector_input in ("denoised", "raw")
    if ctx.denoiser.available:
        assert p.detector_input == "denoised", (
            "pipeline order must be denoise -> detect, and must be recorded honestly")


def test_gated_pipelines_report_nonzero_algorithmic_latency():
    it = _item()
    ctx = PipelineContext(predictor=None, policy=GatePolicy())
    p = run_pipeline("GEOWEAR_GATE", it, ctx)
    assert p.latency["latency_algorithmic_ms"] > 0.0
    assert p.latency["gate_lookahead_ms"] == 0.0


# ---------------------------------------------------------------------------
# P1.14 -- the spec-only guarantee
# ---------------------------------------------------------------------------
def test_wearersepnet_is_not_implemented():
    """This test MUST keep passing. If it fails, someone trained a separator
    without the go-ahead the spec requires."""
    from training.geowearnet import wearersepnet_stub as WS
    m = WS.build()
    assert m.is_implemented is False
    assert m.n_parameters == 0
    assert WS.SELECTED_ARCHITECTURE is None, "no architecture may be selected yet"
    assert m.describe()["status"] == "SPEC_ONLY_NO_MODEL_TRAINED"


def test_wearersepnet_stub_is_an_honest_passthrough():
    from training.geowearnet import wearersepnet_stub as WS
    m = WS.build()
    x = np.random.default_rng(0).normal(0, 0.1, WS.HOP).astype(np.float32)
    assert np.array_equal(m.process_frame(x, 0.9, 0.1), x)


def test_wearersepnet_stub_satisfies_the_declared_interface():
    from training.geowearnet import wearersepnet_stub as WS
    assert isinstance(WS.build(), WS.WearerSepNetInterface)


def test_wearersepnet_budget_constants_match_the_spec():
    from training.geowearnet import wearersepnet_stub as WS
    assert WS.SYSTEM_PARAM_HARD_LIMIT == 10_000_000
    assert WS.PARAM_TARGET_MIN == 500_000 and WS.PARAM_TARGET_MAX == 3_000_000
