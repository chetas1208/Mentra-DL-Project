"""P1.19 -- tests for the final-dev guard (P1.11), the G2 final-selection
pipeline (P1.10), campaign PID isolation (P1.12), and the annotation-anomaly
registry (P1.13).

All of these run against SYNTHETIC fixtures in tmp_path. Nothing here reads
the official MMCSG dev split, launches training, or touches the live
campaign -- the selection pipeline is exercised in dry-run mode only, which
is exactly what the brief asks for ("build and unit-test the pipeline's
logic ... without invoking the real dev-evaluation step for real").
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from training.geowearnet.mmcsg import campaign_procs as CP
from training.geowearnet.mmcsg import dev_guard as DG
from training.geowearnet.mmcsg import final_selection as FS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ckpt(tmp_path: Path, name: str, content: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


def _ablation_status(tmp_path: Path, experiments: dict) -> Path:
    p = tmp_path / "ablation_campaign_status.json"
    p.write_text(json.dumps({"experiments": experiments}))
    return p


def _exp(name, status="DONE", solo=0.90, fwr=0.10, params=29000, run_dir=None):
    d = {"status": status, "params": params}
    if status == "DONE":
        d["run_dir"] = str(run_dir) if run_dir else f"/nonexistent/{name}"
        d["best"] = {"selection": solo, "metrics": {
            "wearer_vs_env_solo_auroc": solo,
            "false_wearer_rate_on_env_only": fwr,
            "wearer_auroc": solo - 0.02,
        }}
    return d


# ===========================================================================
# P1.11 -- FINAL DEV GUARD
# ===========================================================================
def test_dev_guard_allows_the_first_evaluation(tmp_path):
    led = tmp_path / "ledger.json"
    ck = _ckpt(tmp_path, "a.pt", b"model-a")
    d = DG.check(ck, "first look", path=led)
    assert d["decision"] == DG.ALLOW_FIRST and d["allowed"]


def test_dev_guard_refuses_a_second_different_checkpoint(tmp_path):
    """The core anti-fishing property."""
    led = tmp_path / "ledger.json"
    a = _ckpt(tmp_path, "a.pt", b"model-a")
    b = _ckpt(tmp_path, "b.pt", b"model-b")
    DG.record(DG.check(a, "first", path=led), path=led)
    d = DG.check(b, "second", path=led)
    assert d["decision"] == DG.REFUSE_DIFFERENT_CHECKPOINT
    assert not d["allowed"]


def test_dev_guard_enforce_raises_on_refusal(tmp_path):
    led = tmp_path / "ledger.json"
    a = _ckpt(tmp_path, "a.pt", b"model-a")
    b = _ckpt(tmp_path, "b.pt", b"model-b")
    DG.record(DG.check(a, "first", path=led), path=led)
    with pytest.raises(DG.DevGuardRefusal):
        DG.enforce(b, "second", path=led)


def test_dev_guard_allows_rerunning_the_same_checkpoint(tmp_path):
    """Reproduction is not fishing: the same weights cannot yield a new
    selection opportunity."""
    led = tmp_path / "ledger.json"
    a = _ckpt(tmp_path, "a.pt", b"model-a")
    DG.record(DG.check(a, "first", path=led), path=led)
    d = DG.check(a, "reproduce", path=led)
    assert d["decision"] == DG.ALLOW_SAME_CHECKPOINT and d["allowed"]


def test_dev_guard_identifies_checkpoints_by_content_not_path(tmp_path):
    """Copying/renaming a checkpoint must not sneak a second look past it."""
    led = tmp_path / "ledger.json"
    a = _ckpt(tmp_path, "a.pt", b"identical-bytes")
    a2 = _ckpt(tmp_path, "renamed.pt", b"identical-bytes")
    DG.record(DG.check(a, "first", path=led), path=led)
    d = DG.check(a2, "sneaky", path=led)
    assert d["decision"] == DG.ALLOW_SAME_CHECKPOINT, "content hash must defeat renaming"


def test_dev_guard_override_requires_written_reason(tmp_path):
    led = tmp_path / "ledger.json"
    a = _ckpt(tmp_path, "a.pt", b"model-a")
    b = _ckpt(tmp_path, "b.pt", b"model-b")
    DG.record(DG.check(a, "first", path=led), path=led)
    with pytest.raises(DG.DevGuardRefusal):
        DG.check(b, "second", override=True, override_reason="  ", path=led)


def test_dev_guard_override_is_permitted_and_recorded(tmp_path):
    led = tmp_path / "ledger.json"
    a = _ckpt(tmp_path, "a.pt", b"model-a")
    b = _ckpt(tmp_path, "b.pt", b"model-b")
    DG.record(DG.check(a, "first", path=led), path=led)
    d = DG.check(b, "second", override=True,
                 override_reason="eval bug found, prior selection invalidated", path=led)
    assert d["decision"] == DG.ALLOW_OVERRIDE and d["allowed"]
    DG.record(d, path=led)
    st = DG.status(path=led)
    assert st["n_overrides"] == 1 and st["n_real_dev_evaluations"] == 2


def test_dev_guard_dry_runs_do_not_spend_the_dev_look(tmp_path):
    led = tmp_path / "ledger.json"
    a = _ckpt(tmp_path, "a.pt", b"model-a")
    b = _ckpt(tmp_path, "b.pt", b"model-b")
    DG.record(DG.check(a, "dry", path=led), dry_run=True, path=led)
    d = DG.check(b, "real first look", path=led)
    assert d["decision"] == DG.ALLOW_FIRST, "a dry run must not consume the real look"


def test_dev_guard_status_reports_unspent_ledger(tmp_path):
    st = DG.status(path=tmp_path / "nope.json")
    assert st["dev_split_spent"] is False and st["n_real_dev_evaluations"] == 0


# ===========================================================================
# P1.10 -- FINAL SELECTION
# ===========================================================================
def test_selection_blocks_while_campaign_incomplete(tmp_path):
    sp = _ablation_status(tmp_path, {
        "a": _exp("a", "DONE"), "b": _exp("b", "RUNNING")})
    rep = FS.run_pipeline(dry_run=True, status_path=sp, out_path=tmp_path / "o.json")
    assert rep["status"] == "BLOCKED_CAMPAIGN_INCOMPLETE"
    assert "b" in rep["completeness"]["blocking_reason"]


def test_selection_provides_provisional_view_when_blocked(tmp_path):
    sp = _ablation_status(tmp_path, {
        "a": _exp("a", "DONE"), "b": _exp("b", "QUEUED")})
    rep = FS.run_pipeline(dry_run=True, status_path=sp, out_path=tmp_path / "o.json")
    assert "PROVISIONAL" in rep["provisional_selection"]


def test_completeness_verdicts(tmp_path):
    assert FS.validate_completeness({"experiments": {"a": _exp("a")}})["verdict"] == \
        "CAMPAIGN_COMPLETE"
    assert FS.validate_completeness({"experiments": {
        "a": _exp("a"), "b": _exp("b", "RUNNING")}})["verdict"] == "CAMPAIGN_STILL_RUNNING"
    assert FS.validate_completeness({"experiments": {
        "a": _exp("a"), "b": _exp("b", "FAILED")}})["verdict"] == \
        "CAMPAIGN_COMPLETE_WITH_FAILURES"


def test_pareto_picks_dominating_candidate(tmp_path):
    """b dominates a on BOTH axes -> b must win."""
    rd = tmp_path / "rb"; (rd / "checkpoints").mkdir(parents=True)
    (rd / "checkpoints/best.pt").write_bytes(b"b")
    ra = tmp_path / "ra"; (ra / "checkpoints").mkdir(parents=True)
    (ra / "checkpoints/best.pt").write_bytes(b"a")
    sel = FS.pareto_select({"experiments": {
        "a": _exp("a", solo=0.90, fwr=0.20, run_dir=ra),
        "b": _exp("b", solo=0.95, fwr=0.10, run_dir=rd)}})
    assert sel["status"] == "SELECTED"
    assert sel["selected"]["name"] == "b"
    assert sel["pareto_front"] == ["b"]


def test_pareto_front_keeps_genuine_tradeoffs(tmp_path):
    ra = tmp_path / "ra"; (ra / "checkpoints").mkdir(parents=True)
    (ra / "checkpoints/best.pt").write_bytes(b"a")
    rb = tmp_path / "rb"; (rb / "checkpoints").mkdir(parents=True)
    (rb / "checkpoints/best.pt").write_bytes(b"b")
    sel = FS.pareto_select({"experiments": {
        "a": _exp("a", solo=0.95, fwr=0.20, run_dir=ra),   # better AUROC
        "b": _exp("b", solo=0.90, fwr=0.05, run_dir=rb)}})  # better leakage
    assert set(sel["pareto_front"]) == {"a", "b"}, "neither dominates the other"


def test_selection_excludes_deployment_ineligible_controls(tmp_path):
    rd = tmp_path / "r"; (rd / "checkpoints").mkdir(parents=True)
    (rd / "checkpoints/best.pt").write_bytes(b"x")
    sel = FS.pareto_select({"experiments": {
        "g2_ablation_level_normalized": _exp("x", solo=0.99, fwr=0.01, run_dir=rd),
        "g2_ablation_baseline_60m": _exp("y", solo=0.90, fwr=0.10, run_dir=rd)}})
    assert sel["selected"]["name"] == "g2_ablation_baseline_60m"
    assert "g2_ablation_level_normalized" in sel["ineligible"]


def test_selection_criterion_is_predeclared_in_code():
    assert FS.PARETO_SPEC["predeclared"] is True
    assert FS.PARETO_SPEC["maximise"] == "wearer_vs_env_solo_auroc"
    assert FS.PARETO_SPEC["minimise"] == "false_wearer_rate_on_env_only"


def test_dry_run_never_invokes_the_dev_evaluation(tmp_path):
    rd = tmp_path / "r"; (rd / "checkpoints").mkdir(parents=True)
    (rd / "checkpoints/best.pt").write_bytes(b"weights")
    sp = _ablation_status(tmp_path, {"a": _exp("a", solo=0.9, fwr=0.1, run_dir=rd)})
    led = tmp_path / "ledger.json"
    orig = DG.LEDGER
    DG.LEDGER = led
    try:
        rep = FS.run_pipeline(dry_run=True, status_path=sp, out_path=tmp_path / "o.json")
    finally:
        DG.LEDGER = orig
    assert rep["status"] == "DRY_RUN_OK"
    for stage in ("streaming_real", "identity_probe", "cpu_benchmark_and_export",
                  "report_regeneration"):
        assert rep["downstream"][stage]["status"] == "PLANNED_NOT_RUN"
    assert rep["downstream"]["dev_evaluation"]["status"] == "PLANNED_NOT_RUN"
    assert rep["downstream"]["capability_update"]["status"] == "PLANNED_NOT_RUN"
    assert not led.exists(), "dry-run planning must not touch the dev ledger"


def test_existing_final_dev_artifact_is_idempotent_for_same_checkpoint(tmp_path):
    ck = _ckpt(tmp_path, "selected.pt", b"selected-model")
    artifact = tmp_path / "g2_final_dev_evaluation.json"
    artifact.write_text(json.dumps({
        "checkpoint": str(ck),
        "evaluated_utc": "2026-08-27T00:00:00Z",
        "result": {
            "wearer_all_frames": {"auroc": 0.95},
            "speech_active_10_vs_01": {"auroc": 0.98},
            "overlap_f1": 0.40,
            "state_accuracy": 0.81,
            "n_frames": 123,
        },
    }))
    original = FS.FINAL_DEV_ARTIFACT
    FS.FINAL_DEV_ARTIFACT = artifact
    try:
        result = FS.final_dev_evaluation(ck, dry_run=False)
    finally:
        FS.FINAL_DEV_ARTIFACT = original
    assert result["status"] == "ALREADY_RECORDED"
    assert result["headline"]["n_frames"] == 123
    assert json.loads(artifact.read_text())["result"]["n_frames"] == 123


def test_existing_final_dev_artifact_refuses_different_checkpoint(tmp_path):
    selected = _ckpt(tmp_path, "selected.pt", b"selected-model")
    other = _ckpt(tmp_path, "other.pt", b"other-model")
    artifact = tmp_path / "g2_final_dev_evaluation.json"
    artifact.write_text(json.dumps({"checkpoint": str(selected), "result": {}}))
    original = FS.FINAL_DEV_ARTIFACT
    FS.FINAL_DEV_ARTIFACT = artifact
    try:
        result = FS.final_dev_evaluation(other, dry_run=False)
    finally:
        FS.FINAL_DEV_ARTIFACT = original
    assert result["status"] == "REFUSED_ARTIFACT_EXISTS"
    assert json.loads(artifact.read_text())["checkpoint"] == str(selected)


def test_dry_run_does_not_write_a_frozen_checkpoint(tmp_path):
    rd = tmp_path / "r"; (rd / "checkpoints").mkdir(parents=True)
    ck = rd / "checkpoints/best.pt"
    ck.write_bytes(b"weights")
    m = FS.freeze_checkpoint(ck, dest_dir=tmp_path / "frozen", dry_run=True)
    assert m["dry_run"] is True
    assert not Path(m["frozen_path"]).exists()


def test_freeze_copies_and_verifies_hash(tmp_path):
    rd = tmp_path / "r"; (rd / "checkpoints").mkdir(parents=True)
    ck = rd / "checkpoints/best.pt"
    ck.write_bytes(b"weights-abc")
    m = FS.freeze_checkpoint(ck, dest_dir=tmp_path / "frozen", dry_run=False)
    assert Path(m["frozen_path"]).exists()
    assert m["copy_verified"] is True
    assert ck.exists(), "freezing must never remove or alter the source run's checkpoint"


def test_skip_dev_flag_avoids_touching_dev(tmp_path):
    rd = tmp_path / "r"; (rd / "checkpoints").mkdir(parents=True)
    (rd / "checkpoints/best.pt").write_bytes(b"w")
    sp = _ablation_status(tmp_path, {"a": _exp("a", run_dir=rd)})
    rep = FS.run_pipeline(dry_run=True, skip_dev=True, status_path=sp,
                          out_path=tmp_path / "o.json")
    assert rep["downstream"]["dev_evaluation"]["status"] == "SKIPPED_BY_FLAG"
    assert "dev_guard_decision" not in rep["downstream"]


# ===========================================================================
# P1.12 -- CAMPAIGN PID ISOLATION
# ===========================================================================
def test_registry_creates_unique_paths_per_campaign(tmp_path):
    a = CP.CampaignRegistry("camp_a", root=tmp_path)
    b = CP.CampaignRegistry("camp_b", root=tmp_path)
    a.open(); b.open()
    assert a.log_path("exp") != b.log_path("exp"), "log files must never collide"
    assert a.output_dir("exp") != b.output_dir("exp"), "output dirs must never collide"
    assert a.run_id("exp") != b.run_id("exp")


def test_registry_verifies_live_process_identity(tmp_path):
    reg = CP.CampaignRegistry("c", root=tmp_path)
    reg.open()
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        rec = reg.register("myexp", p.pid, "cuda:0", "python -c sleep")
        v = CP.verify(rec)
        # our sleeper's cmdline does not contain "myexp", so identity check
        # correctly reports a mismatch -- that IS the fingerprint working
        assert v["alive"] is True
        assert v["state"] in ("ALIVE", "PID_REUSED")
        assert CP.proc_start_time(p.pid) == rec.start_time
    finally:
        p.kill(); p.wait()


def test_dead_process_is_detected_as_dead(tmp_path):
    reg = CP.CampaignRegistry("c", root=tmp_path)
    reg.open()
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    time.sleep(0.2)
    rec = CP.PidRecord(name="e", pid=p.pid, ppid=os.getpid(),
                       start_time=999999, campaign_id="c", gpu="cuda:0",
                       run_id="e__c", log="l", output_dir="o", cmd="c",
                       launched_utc="now")
    v = CP.verify(rec)
    assert v["state"] in ("DEAD", "PID_REUSED"), "a finished process must not read as ALIVE"


def test_pid_reuse_is_detected_via_start_time(tmp_path):
    """A recycled PID must never be mistaken for our job -- otherwise teardown
    would kill an innocent process."""
    rec = CP.PidRecord(name="python", pid=os.getpid(), ppid=os.getppid(),
                       start_time=1, campaign_id="c", gpu="cuda:0",
                       run_id="r", log="l", output_dir="o", cmd="c",
                       launched_utc="now")
    v = CP.verify(rec)
    assert v["state"] == "PID_REUSED"
    assert any("start_time" in r for r in v["mismatch_reasons"])


def test_claim_refuses_when_a_live_job_holds_the_experiment(tmp_path):
    """The structural fix for the zombie incident: two live jobs can no longer
    be pointed at the same log/output directory."""
    reg = CP.CampaignRegistry("c", root=tmp_path)
    reg.open()
    p = subprocess.Popen([sys.executable, "-c",
                          "import time; time.sleep(30)  # holder_exp"])
    try:
        # register with a name that appears in the cmdline so identity holds
        rec = reg.register("holder_exp", p.pid, "cuda:0", "x")
        assert CP.verify(rec)["state"] == "ALIVE"
        claim = reg.claim("holder_exp")
        assert claim["ok"] is False
        assert claim["reason"] == "EXPERIMENT_ALREADY_RUNNING"
    finally:
        p.kill(); p.wait()


def test_claim_reclaims_after_process_dies(tmp_path):
    reg = CP.CampaignRegistry("c", root=tmp_path)
    reg.open()
    p = subprocess.Popen([sys.executable, "-c", "pass  # gone_exp"])
    p.wait(); time.sleep(0.2)
    reg.register("gone_exp", p.pid, "cuda:0", "x")
    claim = reg.claim("gone_exp")
    assert claim["ok"] is True


def test_find_stale_reports_running_records_whose_process_vanished(tmp_path):
    reg = CP.CampaignRegistry("c", root=tmp_path)
    reg.open()
    p = subprocess.Popen([sys.executable, "-c", "pass  # vanish_exp"])
    p.wait(); time.sleep(0.2)
    reg.register("vanish_exp", p.pid, "cuda:0", "x")
    stale = reg.find_stale()
    assert len(stale) == 1 and stale[0]["record"]["name"] == "vanish_exp"


def test_status_reports_pid_ppid_gpu_runid_log_output_state(tmp_path):
    """The brief's explicit --status requirement."""
    reg = CP.CampaignRegistry("c", root=tmp_path)
    reg.open()
    reg.register("e", os.getpid(), "cuda:1", "cmd")
    row = reg.status()["experiments"][0]
    for k in ("pid", "ppid", "gpu", "run_id", "log", "output_dir",
              "recorded_state", "verified_state"):
        assert k in row, f"--status must show {k}"


def test_terminate_tree_is_dry_run_by_default(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        r = CP.terminate_tree(p.pid)
        assert r["result"] == "PLANNED_NOT_SENT"
        assert p.poll() is None, "dry run must not actually signal the process"
    finally:
        p.kill(); p.wait()


def test_terminate_tree_actually_terminates_when_asked(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"],
                         start_new_session=True)
    r = CP.terminate_tree(p.pid, grace_s=5.0, dry_run=False)
    assert r["result"] in ("TERMINATED", "KILLED")
    p.wait(timeout=5)


def test_terminate_tree_handles_already_gone_process():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait(); time.sleep(0.2)
    r = CP.terminate_tree(p.pid, dry_run=False)
    assert r["result"] in ("ALREADY_GONE", "TERMINATED", "KILLED")


# ===========================================================================
# P1.13 -- ANNOTATION ANOMALY REGISTRY
# ===========================================================================
def test_registry_lists_the_known_anomaly():
    from evaluation.agent_audio import anomalies as A
    ids = A.anomalous_recording_ids()
    assert "1302664060426140_0001_3375_22000" in ids


def test_registry_entry_has_required_provenance_fields():
    from evaluation.agent_audio import anomalies as A
    e = A.anomalies()[0]
    for k in ("observed_behavior", "model_independent_evidence", "detection_script",
              "date_identified", "status", "handling_policy", "classification"):
        assert k in e, f"anomaly entry missing {k}"
    assert e["handling_policy"]["raw_data"].startswith("UNMODIFIED")


def test_is_anomalous_matches_sliced_window_ids():
    from evaluation.agent_audio import anomalies as A
    assert A.is_anomalous("1302664060426140_0001_3375_22000@5.0+20.0")
    assert not A.is_anomalous("some_other_recording@5.0+20.0")


def test_split_rows_returns_both_views():
    """P1.13: a caller physically cannot obtain one view without the other."""
    from evaluation.agent_audio import anomalies as A
    rows = [{"recording_id": "1302664060426140_0001_3375_22000"},
            {"recording_id": "normal_rec"}]
    allr, clean = A.split_rows_by_anomaly(rows)
    assert len(allr) == 2 and len(clean) == 1


def test_harness_always_emits_official_and_anomaly_excluded():
    from evaluation.agent_audio.harness import aggregate_rows
    rows = [{"recording_id": "1302664060426140_0001_3375_22000",
             "n_self_ref_words": 10, "n_del": 5, "n_sub": 0, "n_ins": 0,
             "n_self_words_retained": 5, "n_other_ref_words": 10, "n_other_words_leaked": 9},
            {"recording_id": "normal", "n_self_ref_words": 10, "n_del": 1, "n_sub": 0,
             "n_ins": 0, "n_self_words_retained": 9, "n_other_ref_words": 10,
             "n_other_words_leaked": 1}]
    agg = aggregate_rows(rows, n_boot=20)
    assert "official" in agg and "anomaly_excluded" in agg
    assert agg["anomaly_reporting"]["n_rows_excluded_as_anomalous"] == 1
    # the anomaly genuinely moves the number -> excluding it must differ
    assert agg["official"]["bystander_leakage_rate"] != \
        agg["anomaly_excluded"]["bystander_leakage_rate"]


def test_product_report_reads_current_overlap_artifact_key():
    from evaluation.agent_audio.report import _overlap_section

    section = _overlap_section({
        "state_occupancy": {"occupancy": {}},
        "overlap_bottleneck": {
            "leakage_attribution_oracle_gate_token_bag": {
                "pct_of_leakage_from_overlap": 75.0,
                "leaked_words_by_state": {"11_overlap": 3.0},
                "other_ref_words_by_state": {"11_overlap": 4.0},
                "leakage_rate_by_state": {"11_overlap": 0.75},
            }
        },
    })
    assert "75.0%" in section
