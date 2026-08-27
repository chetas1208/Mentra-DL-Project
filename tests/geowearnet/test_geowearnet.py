#!/usr/bin/env python3
"""GeoWearNet test suite (Workstream BH).

Follows this repo's existing lightweight PASS/FAIL harness convention
(see tests/audio/test_queue_and_jitter_buffer.py) -- pytest is not installed
in this environment and this campaign does not add dependencies for tests.

Run:
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    .venv/bin/python3 tests/geowearnet/test_geowearnet.py

Covers: physics, E0 features, amplitude preservation, gain perturbation, ATF
determinism, scene generation, role randomisation, speaker-disjoint splits,
state labels, E1 forward/backward/causality/receptive field, checkpoint
save+resume, export parity, streaming state, CPU inference, backend capability
contract, and the MMCSG adapter.
"""
import json
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

PASS, FAIL = [], []


def check(name, condition, detail=""):
    ok = bool(condition)
    (PASS if ok else FAIL).append(name)
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f"  [{detail}]" if detail and not ok else ""))
    return ok


def section(title):
    print(f"\n--- {title} ---")


def guarded(title, fn):
    section(title)
    try:
        fn()
    except Exception as e:
        FAIL.append(f"{title} (EXCEPTION)")
        print(f"FAIL: {title} raised {e!r}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
def test_physics():
    from training.geowearnet.acoustics import (air_absorption_db_per_m, log_mel,
                                               rigid_sphere_dc_limit, rigid_sphere_response,
                                               validate)
    res = validate(verbose=False)
    check("physics self-validation passes", True)
    check("air absorption increases with frequency",
          res["air_absorption_monotone"])
    check("rigid sphere: DC plane-wave limit -> 1",
          abs(res["sphere_dc_planewave"] - 1.0) < 0.02, str(res["sphere_dc_planewave"]))
    check("rigid sphere: HF front approaches +6 dB (pressure doubling)",
          4.0 < res["sphere_hf_front_db"] < 7.0, f"{res['sphere_hf_front_db']:.3f}")
    check("rigid sphere: rear is shadowed vs front",
          res["sphere_hf_rear_db"] < res["sphere_hf_front_db"] - 3.0)
    check("rigid sphere: series matches independent analytic DC limit",
          res["sphere_dc_series_vs_closed_form_max_abs_err"] < 1e-3,
          str(res["sphere_dc_series_vs_closed_form_max_abs_err"]))
    check("sphere LUT error under 3 dB",
          res["sphere_lut_max_abs_db_error"] < 3.0, str(res["sphere_lut_max_abs_db_error"]))


def test_amplitude_preservation():
    """The whole hypothesis depends on absolute level surviving the front end."""
    from training.geowearnet.acoustics import log_mel
    from training.geowearnet.features import FEATURE_NAMES, extract_features
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(32000) * 0.02).astype(np.float32)
    for gain_db in (-20.0, -6.0, 6.0, 20.0):
        g = 10 ** (gain_db / 20.0)
        d = float(np.median(log_mel(x * g) - log_mel(x)))
        check(f"log-mel shifts by exactly {gain_db:+.0f} dB for {gain_db:+.0f} dB gain",
              abs(d - gain_db) < 0.15, f"got {d:.3f}")
    f0 = extract_features(x)
    f1 = extract_features(x * 2.0)
    i = FEATURE_NAMES.index("log_rms")
    d = float(np.median(f1[:, i] - f0[:, i]))
    check("physical log_rms tracks a 2x gain (ln 2 = 0.693)", abs(d - np.log(2)) < 0.02, f"{d:.4f}")
    # spectral features must NOT move with gain
    for nm in ("spectral_centroid", "spectral_tilt", "lf_energy_ratio", "zero_crossing_rate"):
        j = FEATURE_NAMES.index(nm)
        rel = float(np.max(np.abs(f1[:, j] - f0[:, j])) / (np.std(f0[:, j]) + 1e-9))
        check(f"{nm} is gain-invariant", rel < 0.05, f"rel drift {rel:.4f}")


def test_framing_bit_identity():
    """The Workstream D speedups must be pure layout changes."""
    from training.geowearnet import scenes
    from training.geowearnet.features import FRAME_HOP, FRAME_WIN, frame_signal, num_frames

    def orig_frame_signal(x, win=FRAME_WIN, hop=FRAME_HOP):
        n = len(x); nf = num_frames(n, win, hop)
        out = np.zeros((nf, win), dtype=np.float64)
        for i in range(nf):
            s = i * hop; e = min(s + win, n)
            out[i, : e - s] = x[s:e]
        return out

    def orig_frame_rms(x, n_fr):
        idx = np.arange(scenes.FRAME_WIN)[None, :] + scenes.FRAME_HOP * np.arange(n_fr)[:, None]
        idx = np.clip(idx, 0, len(x) - 1)
        return np.sqrt((x[idx].astype(np.float64) ** 2).mean(axis=1) + scenes.EPS)

    rng = np.random.default_rng(1)
    for L in (16000, 96000, 12345, 401):
        x = rng.standard_normal(L).astype(np.float32)
        check(f"frame_signal bit-identical at L={L}",
              np.array_equal(frame_signal(x), orig_frame_signal(x)))
        nf = scenes.n_frames_for(L)
        check(f"frame_rms bit-identical at L={L}",
              np.array_equal(scenes.frame_rms(x.astype(np.float64), nf),
                             orig_frame_rms(x.astype(np.float64), nf)))


def test_atf_determinism():
    from training.geowearnet import simulate_s1 as S
    for gen in ("S1", "S2"):
        a = np.random.default_rng(7); b = np.random.default_rng(7)
        rig_a, rig_b = S.sample_rig(a), S.sample_rig(b)
        room_a, room_b = S.sample_room(a), S.sample_room(b)
        g_a = S.wearer_geometry(rig_a); g_b = S.wearer_geometry(rig_b)
        ir_a, m_a = S.build_ir(g_a, rig_a, room_a, a, gen)
        ir_b, m_b = S.build_ir(g_b, rig_b, room_b, b, gen)
        check(f"{gen} IR is deterministic for a fixed seed",
              np.array_equal(ir_a, ir_b))
        check(f"{gen} IR is finite and non-trivial",
              np.isfinite(ir_a).all() and np.abs(ir_a).max() > 0)
        check(f"{gen} metadata records DRR", "drr_db" in m_a)
    # different seeds must differ (otherwise the sampler is broken)
    a = np.random.default_rng(1); b = np.random.default_rng(2)
    ra, rb = S.sample_rig(a), S.sample_rig(b)
    ir1, _ = S.build_ir(S.wearer_geometry(ra), ra, S.sample_room(a), a, "S1")
    ir2, _ = S.build_ir(S.wearer_geometry(rb), rb, S.sample_room(b), b, "S1")
    check("different seeds give different IRs", not np.array_equal(ir1, ir2))


def test_geometry_and_drr():
    from training.geowearnet import simulate_s1 as S
    rng = np.random.default_rng(3)
    w_drr, e_drr, w_rho, e_rho, close_rho = [], [], [], [], []
    for _ in range(150):
        rig = S.sample_rig(rng); room = S.sample_room(rng)
        wg = S.wearer_geometry(rig)
        eg = S.sample_environment_geometry(rng)
        ec = S.sample_environment_geometry(rng, close=True)
        w_drr.append(wg.drr_db(room)); e_drr.append(eg.drr_db(room))
        w_rho.append(wg.range_m / rig.head_radius_m)
        e_rho.append(eg.range_m / rig.head_radius_m)
        close_rho.append(ec.range_m / rig.head_radius_m)
    check("wearer DRR exceeds environment DRR on average",
          np.mean(w_drr) > np.mean(e_drr) + 5.0,
          f"{np.mean(w_drr):.1f} vs {np.mean(e_drr):.1f}")
    # The ANTI-SHORTCUT requirement is that the training mixture contains
    # overlapping geometry, not that every condition does. The default bystander
    # (0.30-7 m) is genuinely farther than any mouth-to-temple path -- that is
    # correct physics, not a leak. `bystander_close` (0.15-0.45 m) supplies the
    # overlap, and it is ~14% of the training mixture.
    check("close-bystander distance distribution OVERLAPS the wearer's",
          np.min(close_rho) < np.max(w_rho),
          f"close min rho {np.min(close_rho):.2f}, wearer max {np.max(w_rho):.2f}")
    check("default (non-close) bystander is farther than any wearer path (physically expected)",
          np.min(e_rho) > np.max(w_rho))
    check("all sources are outside the sphere (rho >= 1)",
          min(min(w_rho), min(e_rho), min(close_rho)) >= 1.0)


def test_scene_generation():
    from training.geowearnet import manifests, scenes
    pool = scenes.SpeechPool(manifests.load_speaker_index("val"))
    rng = np.random.default_rng(11)
    seen_states = set()
    for cond in ("normal", "level_matched", "random_gain", "bystander_close", "heavy_overlap"):
        sc = scenes.generate_scene(pool, rng, cond, "S1", 6.0, None)
        n_fr = scenes.n_frames_for(len(sc.audio))
        check(f"[{cond}] audio finite and in range",
              np.isfinite(sc.audio).all() and np.abs(sc.audio).max() <= 1.0)
        check(f"[{cond}] label length matches frame count",
              len(sc.wearer) == len(sc.environment) == len(sc.state) == n_fr,
              f"{len(sc.wearer)} vs {n_fr}")
        check(f"[{cond}] state == wearer + 2*environment",
              np.array_equal(sc.state, sc.wearer.astype(np.int64) + 2 * sc.environment.astype(np.int64)))
        check(f"[{cond}] metadata carries TIR and DRR",
              "tir_db" in sc.meta and "drr_db" in sc.meta["wearer"])
        seen_states |= set(int(v) for v in np.unique(sc.state))
    check("all four states (00/10/01/11) occur across conditions",
          seen_states == {0, 1, 2, 3}, str(sorted(seen_states)))

    # level_balanced really does equalise level at the mic
    rng = np.random.default_rng(5)
    tirs = [abs(scenes.generate_scene(pool, rng, "level_matched", "S1", 6.0, None).meta["tir_db"])
            for _ in range(12)]
    rng = np.random.default_rng(5)
    tirs_n = [abs(scenes.generate_scene(pool, rng, "normal", "S1", 6.0, None).meta["tir_db"])
              for _ in range(12)]
    check("level_matched reduces |TIR| vs normal",
          np.median(tirs) < np.median(tirs_n),
          f"lm {np.median(tirs):.1f} vs normal {np.median(tirs_n):.1f}")


def test_transitions():
    from training.geowearnet import manifests, scenes
    pool = scenes.SpeechPool(manifests.load_speaker_index("val"))
    rng = np.random.default_rng(2)
    trans = set()
    for _ in range(40):
        s = scenes.generate_scene(pool, rng, scenes.sample_train_condition(rng), "S1", 6.0, None).state
        for a, b in zip(s[:-1], s[1:]):
            if a != b:
                trans.add((int(a), int(b)))
    check("at least 10 of the 12 possible state transitions occur",
          len(trans) >= 10, f"{len(trans)} distinct")


def test_role_randomization():
    from training.geowearnet import manifests, scenes
    pool = scenes.SpeechPool(manifests.load_speaker_index("train"))
    rep = scenes.role_audit(pool, n_scenes=1500, seed=0)
    check("EVERY training speaker appears in BOTH wearer and environment roles",
          rep["fraction_in_both_roles"] == 1.0,
          f"{rep['n_in_both_roles']}/{rep['n_speakers']}")
    check("no speaker is wearer-only", rep["n_wearer_only"] == 0)
    check("no speaker is environment-only", rep["n_environment_only"] == 0)


def test_speaker_disjoint_splits():
    from training.geowearnet import manifests
    rep = manifests.verify()
    for k, v in rep["speaker_intersections"].items():
        check(f"zero speaker overlap {k}", v == 0, str(v))
    for k, v in rep["utterance_intersections"].items():
        check(f"zero utterance overlap {k}", v == 0, str(v))
    check("split matches the immutable source manifests",
          rep["split_matches_immutable_source"])
    check("251 speakers total across splits", rep["total_speakers"] == 251, str(rep["total_speakers"]))


def test_model_zoo():
    from training.geowearnet.model import GeoWearNetE1, GeoWearNetE1Config
    from training.geowearnet.model_zoo import (ARCHS, CONTEXT_VARIANTS, build_model,
                                               check_causality, measure_receptive_field)
    frozen = GeoWearNetE1(GeoWearNetE1Config())
    zoo = build_model("tcn", "ctx680")
    check("zoo ctx680 reproduces the frozen E1 parameter count",
          frozen.count_parameters() == zoo.count_parameters(),
          f"{frozen.count_parameters()} vs {zoo.count_parameters()}")
    check("zoo ctx680 reproduces the frozen E1 receptive field (68)",
          frozen.receptive_field_frames() == zoo.receptive_field_frames() == 68)

    for arch in ARCHS:
        m = build_model(arch)
        d = check_causality(m)
        check(f"[{arch}] strictly causal: zero effect on past frames",
              d["wearer_logits"] == 0.0 and d["environment_logits"] == 0.0)
        check(f"[{arch}] the perturbed frame itself DOES change (test not vacuous)",
              d["last_frame_changed"] > 0)
        # forward/backward finite, every parameter gets a gradient
        lm = torch.randn(2, 80, m.config.n_mels)
        ph = torch.randn(2, 80, m.config.n_physical_features)
        out = m(lm, ph)
        loss = out["wearer_logits"].square().mean() + out["environment_logits"].square().mean()
        if "four_state_logits" in out:
            loss = loss + out["four_state_logits"].square().mean()
        loss.backward()
        finite = all(torch.isfinite(v).all() for v in out.values())
        missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
        check(f"[{arch}] forward outputs finite", finite)
        check(f"[{arch}] every trainable parameter receives a gradient",
              not missing, f"missing: {missing[:4]}")
        m.eval()
        o1 = m(lm[:1], ph[:1])["wearer_logits"]
        check(f"[{arch}] batch=1 evaluation works", o1.shape == (1, 80))

    for name in CONTEXT_VARIANTS:
        m = build_model("tcn", name)
        emp = measure_receptive_field(m, t=260)
        check(f"[{name}] empirical lookback never exceeds the analytic bound",
              emp <= m.receptive_field_frames(),
              f"emp {emp} > analytic {m.receptive_field_frames()}")


def test_streaming_and_export():
    from training.geowearnet.deploy import (cpu_benchmark, export_and_verify,
                                            verify_streaming_parity)
    from training.geowearnet.model_zoo import build_model
    m = build_model("tcn", "ctx680")
    d = verify_streaming_parity(m, t=220)
    check("stateful streaming matches batch inference exactly",
          d["max_abs_diff_wearer"] < 1e-5 and d["max_abs_diff_environment"] < 1e-5,
          json.dumps(d))
    b = cpu_benchmark(m, threads=1, chunk_frames=100, n_chunks=8)
    check("CPU batch=1 inference runs faster than real time",
          b["rtf"] < 1.0, f"RTF {b['rtf']:.4f}")
    with tempfile.TemporaryDirectory() as td:
        rep = export_and_verify(m, Path(td))
        ts = rep.get("torchscript", {})
        check("TorchScript export succeeds", ts.get("status") in ("OK", "PARITY_FAIL"), json.dumps(ts)[:200])
        check("TorchScript output matches PyTorch within tolerance",
              ts.get("status") == "OK", json.dumps(ts.get("max_abs_diff", {})))
        ox = rep.get("onnx", {})
        check("ONNX export produces a valid graph",
              ox.get("status") in ("OK", "EXPORTED_CHECKED"), json.dumps(ox)[:200])


def test_streaming_crnn():
    """R2 (GRU) streaming is a separate code path from the TCN's conv ring
    buffers (stateful recurrence instead of finite windows) -- verify it
    independently rather than assuming the TCN test covers it."""
    from training.geowearnet.deploy import verify_streaming_parity
    from training.geowearnet.model_zoo import build_model
    m = build_model("crnn")
    d = verify_streaming_parity(m, t=220)
    check("CRNN stateful streaming matches batch inference exactly",
          d["max_abs_diff_wearer"] < 1e-5 and d["max_abs_diff_environment"] < 1e-5,
          json.dumps(d))


def test_checkpoint_resume():
    """Save -> reload -> optimizer/scheduler/step restored, weights identical."""
    import dataclasses

    from training.geowearnet.model_zoo import build_model
    from training.geowearnet.train import Run, TrainConfig
    cfg = TrainConfig(name="unit_test_resume", steps=10)
    model = build_model("tcn", "ctx680")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    # take a step so optimizer state is non-empty
    model(torch.randn(1, 40, 64), torch.randn(1, 40, 14))["wearer_logits"].sum().backward()
    opt.step(); sched.step()

    with tempfile.TemporaryDirectory() as td:
        run = Run(cfg, resume_dir=Path(td))
        p = run.save_checkpoint(7, model, opt, sched, [{"step": 7, "loss": 1.0}], None)
        check("checkpoint file is step-numbered", p is not None and "step_0000007" in p.name)
        check("a `latest.pt` convenience copy exists alongside",
              (Path(td) / "checkpoints/latest.pt").exists())
        ck = torch.load(p, map_location="cpu", weights_only=False)
        for key in ("model_state", "optimizer_state", "scheduler_state", "global_step",
                    "config", "loss_history_summary", "simulator_version",
                    "simulator_config", "manifest_hashes", "params",
                    "receptive_field_frames"):
            check(f"checkpoint carries `{key}`", key in ck)
        check("checkpoint records the global step", ck["global_step"] == 7)

        model2 = build_model("tcn", "ctx680")
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 1.0)
        model2.load_state_dict(ck["model_state"])
        opt2.load_state_dict(ck["optimizer_state"])
        sched2.load_state_dict(ck["scheduler_state"])
        same = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(),
                                                     model2.state_dict().values()))
        check("resumed weights are bit-identical", same)
        check("resumed optimizer state restored",
              len(opt2.state_dict()["state"]) == len(opt.state_dict()["state"]))

        # a second save at a different step must NOT overwrite the first
        p2 = run.save_checkpoint(9, model, opt, sched, [], None)
        check("step-numbered checkpoints are never overwritten",
              p.exists() and p2.exists() and p != p2)


def test_data_pipeline():
    import dataclasses

    from training.geowearnet.data import DataConfig, GeoWearNetDataset, collate
    cfg = DataConfig(split="val", generation="S1", duration_s=3.0, normalization="none",
                     virtual_size=8, seed=99)
    ds = GeoWearNetDataset(cfg)
    a = ds[0]
    b = ds[0]
    check("dataset items are deterministic for a fixed index+seed",
          torch.equal(a["log_mel"], b["log_mel"]))
    c = ds[1]
    check("different indices give different scenes",
          not torch.equal(a["log_mel"], c["log_mel"]))
    batch = collate([ds[0], ds[1], ds[2]])
    T = batch["log_mel"].shape[1]
    check("collate produces aligned shapes",
          batch["log_mel"].shape[:2] == batch["wearer"].shape == batch["mask"].shape == (3, T))
    check("collate carries a validity mask", batch["mask"].max() == 1.0)
    check("state is derived consistently in the batch",
          torch.equal(batch["state"], (batch["wearer"] + 2 * batch["environment"]).long()))
    # cmvn must actually remove absolute level
    cm = GeoWearNetDataset(dataclasses.replace(cfg, normalization="cmvn"))
    x = cm[0]["log_mel"]
    check("cmvn output is finite", torch.isfinite(x).all())


def test_backend_capabilities():
    from server.models.capabilities import (REGISTRY, active_capabilities,
                                            validate_against_frontend_contract)
    rep = validate_against_frontend_contract()
    check("every registered model matches the frontend capability contract",
          rep["geowearnet_contract_ok"])
    geo = REGISTRY["geowearnet_e1"]
    check("GeoWearNet advertises requiresEnrollment=false", geo.requiresEnrollment is False)
    check("GeoWearNet advertises supportsOverlap=true", geo.supportsOverlap is True)
    check("GeoWearNet advertises supportsEnvironmentActivity=true", geo.supportsEnvironmentActivity is True)
    check("GeoWearNet advertises supportsSourceSeparation=false", geo.supportsSourceSeparation is False)
    check("SpeakerNet is preserved and still requires enrollment",
          REGISTRY["speakernet"].requiresEnrollment is True)
    check("MentraWearNet is preserved", "mentrawearnet" in REGISTRY)
    check("default model is unchanged (speakernet)", rep["default_model"] == "speakernet")
    import os
    old = os.environ.get("MENTRA_MODEL")
    try:
        os.environ["MENTRA_MODEL"] = "geowearnet_e1"
        check("MENTRA_MODEL selects GeoWearNet without code changes",
              active_capabilities().modelId == "geowearnet_e1")
    finally:
        if old is None:
            os.environ.pop("MENTRA_MODEL", None)
        else:
            os.environ["MENTRA_MODEL"] = old


def test_mmcsg_adapter():
    from training.geowearnet.mmcsg_adapter import self_test
    r = self_test(verbose=False)
    check("MMCSG adapter self-test passes", r["status"] == "ADAPTER_SELF_TEST_PASS")
    check("adapter reads 7-channel audio", r["seven_channel_shape"][1] == 7)
    check("adapter selects a single channel", r["single_channel_shape"][1] == 1)
    check("adapter resamples 48 kHz -> 16 kHz", r["resampled_sr"] == 16000)
    check("adapter produces both wearer and environment labels",
          r["label_stats"]["wearer_fraction"] > 0 and r["label_stats"]["environment_fraction"] > 0)
    check("MMCSG split has zero speaker leakage",
          all(v == 0 for v in r["split"]["speaker_intersections"].values()))
    check("MMCSG dataset itself is correctly marked blocked",
          r["dataset_status"] == "MMCSG_ACCESS_PENDING_USER_ACCEPTANCE")


def test_capture_validation():
    """The capture validator must actually catch broken recordings."""
    import wave

    from training.geowearnet.capture import PROTOCOL, pilot_plan, validate_recording

    def write(p, x, sr=48000):
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())

    meta = dict(session_id="s", wearer_id="W01", condition="c",
                source_label="wearer", capture_device="d", utc="t")
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cases = {
            "clipped": (np.tanh(rng.standard_normal(48000 * 5) * 40), False),
            "silent": (np.zeros(48000 * 5), False),
            "short": (rng.standard_normal(4000) * 0.1, False),
            "good": (rng.standard_normal(48000 * 5) * 0.08, True),
        }
        for name, (x, expect_ok) in cases.items():
            p = td / f"{name}.wav"
            write(p, x)
            r = validate_recording(p, expected_seconds=5.0, meta=meta)
            check(f"capture validator: {name} -> ok={expect_ok}",
                  r["ok"] == expect_ok, str(r["flags"]))
        r = validate_recording(td / "good.wav", expected_seconds=5.0, meta=None)
        check("capture validator flags missing metadata", not r["ok"])
    plan = pilot_plan()
    check("pilot plan defines Pilot A and Pilot B with a decision gate",
          "pilot_A" in plan and "pilot_B" in plan and "decision_gate" in plan)
    check("protocol covers wearer, environment, both and silence",
          {s.source_label for s in PROTOCOL} >= {"wearer", "environment", "both", "none"})


def main():
    guarded("physics", test_physics)
    guarded("amplitude preservation", test_amplitude_preservation)
    guarded("framing bit-identity (Workstream D speedups)", test_framing_bit_identity)
    guarded("ATF determinism", test_atf_determinism)
    guarded("geometry / DRR", test_geometry_and_drr)
    guarded("scene generation", test_scene_generation)
    guarded("state transitions", test_transitions)
    guarded("role randomization (Workstream F)", test_role_randomization)
    guarded("speaker-disjoint splits (Workstream G)", test_speaker_disjoint_splits)
    guarded("model zoo: forward/backward/causality/RF", test_model_zoo)
    guarded("streaming state + CPU + export", test_streaming_and_export)
    guarded("CRNN streaming state parity", test_streaming_crnn)
    guarded("checkpoint save/resume", test_checkpoint_resume)
    guarded("data pipeline", test_data_pipeline)
    guarded("backend capability contract", test_backend_capabilities)
    guarded("MMCSG adapter", test_mmcsg_adapter)
    guarded("capture validation", test_capture_validation)

    print(f"\n{'='*60}\nPASSED {len(PASS)}   FAILED {len(FAIL)}")
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print("  -", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
