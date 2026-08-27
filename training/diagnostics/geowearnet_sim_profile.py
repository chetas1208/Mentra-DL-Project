"""Workstream D -- simulator throughput profile + correctness smoke test.

Run:  python3 -m training.diagnostics.geowearnet_sim_profile
(always with OMP_NUM_THREADS=2 etc -- see the campaign environment notes)
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.geowearnet import manifests, scenes
from training.geowearnet.acoustics import log_mel, validate as validate_physics

OUT = Path(__file__).resolve().parents[2] / "evaluation/geowearnet/results/geowearnet_sim_profile.json"


def main() -> None:
    res = {}
    print("== physics validation ==")
    res["physics"] = validate_physics(verbose=False)

    idx = manifests.load_speaker_index("train")
    pool = scenes.SpeechPool(idx)
    npool = scenes.NoisePool()
    print(f"speakers={len(pool.speakers)} noise_files={len(npool.paths)}")
    res["pool"] = {"n_speakers": len(pool.speakers), "n_noise_files": len(npool.paths)}

    # --- Workstream F audit ---
    res["role_audit"] = scenes.role_audit(pool, n_scenes=1500, seed=0)
    print("role audit:", res["role_audit"])

    # --- correctness + throughput per generation ---
    for gen in ("S1", "S2"):
        rng = np.random.default_rng(7)
        # warm caches
        scenes.generate_scene(pool, rng, "normal", gen, 8.0, npool)
        n = 40
        t0 = time.perf_counter()
        state_hist = np.zeros(4)
        tirs, drr_w, drr_e, clip = [], [], [], []
        for _ in range(n):
            c = scenes.sample_train_condition(rng)
            sc = scenes.generate_scene(pool, rng, c, gen, 8.0, npool)
            state_hist += np.bincount(sc.state, minlength=4)
            tirs.append(sc.meta["tir_db"])
            drr_w.append(sc.meta["wearer"]["drr_db"])
            drr_e.append(float(np.mean([m["drr_db"] for m in sc.meta["environment"]])))
            clip.append(sc.meta["clip_fraction"])
            assert np.isfinite(sc.audio).all(), "non-finite audio"
            assert len(sc.wearer) == len(sc.environment) == len(sc.state)
        dt = time.perf_counter() - t0
        # feature cost, separately
        t1 = time.perf_counter()
        for _ in range(20):
            lm = log_mel(sc.audio)
        dt_feat = (time.perf_counter() - t1) / 20

        res[gen] = {
            "scenes_per_sec_single_core": n / dt,
            "audio_seconds_per_sec_single_core": n * 8.0 / dt,
            "ms_per_scene": 1000 * dt / n,
            "log_mel_ms_per_8s": 1000 * dt_feat,
            "state_fractions": {scenes.STATE_NAMES[i]: float(state_hist[i] / state_hist.sum()) for i in range(4)},
            "tir_db": {"mean": float(np.mean(tirs)), "p5": float(np.percentile(tirs, 5)),
                       "p95": float(np.percentile(tirs, 95))},
            "drr_wearer_db_mean": float(np.mean(drr_w)),
            "drr_environment_db_mean": float(np.mean(drr_e)),
            "mean_clip_fraction": float(np.mean(clip)),
            "log_mel_frames": int(lm.shape[0]),
            "log_mel_dim": int(lm.shape[1]),
        }
        print(gen, json.dumps(res[gen], indent=2))

    # --- transition coverage (Workstream H): every state pair must occur ---
    rng = np.random.default_rng(11)
    trans = defaultdict(int)
    for _ in range(60):
        sc = scenes.generate_scene(pool, rng, scenes.sample_train_condition(rng), "S1", 8.0, npool)
        s = sc.state
        for a, b in zip(s[:-1], s[1:]):
            if a != b:
                trans[f"{scenes.STATE_NAMES[a]}->{scenes.STATE_NAMES[b]}"] += 1
    res["state_transitions_observed"] = dict(sorted(trans.items()))
    res["n_distinct_transitions"] = len(trans)
    print("transitions:", res["n_distinct_transitions"], dict(sorted(trans.items())))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
