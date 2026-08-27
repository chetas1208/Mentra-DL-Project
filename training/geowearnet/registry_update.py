"""Refresh the GeoWearNet experiment registry from what is actually on disk.

Workstream A + the orchestrator's durability requirement: a fresh continuation
of this campaign must be able to read
`evaluation/geowearnet/experiment_registry.json` and know the real state
without any memory of the conversation. So this script DERIVES state from
files (run dirs, summaries, result JSONs, campaign status) rather than from
anything hand-written.

Run:  python3 -m training.geowearnet.registry_update
"""
from __future__ import annotations

import json
from pathlib import Path

from .registry import REPO_ROOT, Registry, refresh_fingerprints

RUNS = REPO_ROOT / "training/geowearnet/runs"
RESULTS = REPO_ROOT / "evaluation/geowearnet/results"
CAMPAIGN_STATUS = REPO_ROOT / "evaluation/geowearnet/campaign_status.json"

# Files produced by the campaign that should always be fingerprinted.
CAMPAIGN_ARTIFACTS = [
    ("sim_s1_s2", "training/geowearnet/simulate_s1.py", "S1/S2 simulator"),
    ("acoustics", "training/geowearnet/acoustics.py", "Physics primitives (ISO 9613-1, rigid sphere, mel)"),
    ("scenes", "training/geowearnet/scenes.py", "Scene generator + conditions"),
    ("data_pipeline", "training/geowearnet/data.py", "Torch dataset / normalisation"),
    ("model_zoo", "training/geowearnet/model_zoo.py", "Context/size/architecture variants"),
    ("trainer", "training/geowearnet/train.py", "E1 trainer (DDP/AMP/resume)"),
    ("campaign_scheduler", "training/geowearnet/campaign.py", "Experiment queue scheduler"),
    ("geo_manifests_builder", "training/geowearnet/manifests.py", "Speaker-disjoint manifest builder"),
    ("manifest_geo_train", "evaluation/geowearnet/manifests/geowearnet_librispeech_train.json", "GeoWearNet train manifest"),
    ("manifest_geo_val", "evaluation/geowearnet/manifests/geowearnet_librispeech_val.json", "GeoWearNet val manifest"),
    ("manifest_geo_test", "evaluation/geowearnet/manifests/geowearnet_librispeech_test.json", "GeoWearNet test manifest"),
    ("license_state", "docs/geowearnet_data_license_state.md", "Workstream B license record"),
    ("mmcsg_adapter", "training/geowearnet/mmcsg_adapter.py", "MMCSG ingestion/manifest/channel-study adapter"),
    ("mmcsg_transfer", "training/geowearnet/mmcsg_transfer.py", "Workstream X/AZ zero-shot sim-to-real transfer eval"),
    ("e0_s1_frame_results", "evaluation/geowearnet/results/geowearnet_e0_s1_frame_results.json", "E0 frame-level on S1"),
    ("sim_profile", "evaluation/geowearnet/results/geowearnet_sim_profile.json", "Workstream D throughput profile"),
]


def main() -> None:
    reg = refresh_fingerprints()
    for key, path, role in CAMPAIGN_ARTIFACTS:
        reg.fingerprint_file(key, path, role=role)

    # --- training runs discovered on disk ---
    if RUNS.exists():
        for d in sorted(RUNS.iterdir()):
            if not d.is_dir():
                continue
            cfg_p, sum_p = d / "config.json", d / "summary.json"
            cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
            summary = json.loads(sum_p.read_text()) if sum_p.exists() else None
            ckpts = sorted((d / "checkpoints").glob("step_*.pt")) if (d / "checkpoints").exists() else []
            metrics = d / "metrics.jsonl"
            last_val = None
            n_val = 0
            if metrics.exists():
                for line in metrics.read_text().splitlines():
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    if o.get("type") == "val":
                        last_val = {k: v for k, v in o.items() if k != "state_confusion"}
                        n_val += 1
            status = "DONE" if summary else ("RUNNING" if ckpts or metrics.exists() else "STARTED")
            reg.upsert_experiment(
                d.name,
                kind="training_run",
                run_dir=str(d.relative_to(REPO_ROOT)),
                config=cfg,
                status=status,
                n_step_checkpoints=len(ckpts),
                checkpoint_steps=[int(p.stem.split("_")[1]) for p in ckpts][-8:],
                best=(summary or {}).get("best"),
                params=(summary or {}).get("params"),
                final_step=(summary or {}).get("final_step"),
                n_validations=n_val,
                last_validation=last_val,
                simulator_generation=cfg.get("generation"),
                seed=cfg.get("seed"),
                data_note="SIMULATED GEOMETRY (LibriSpeech CC BY 4.0 + MUSAN); NOT Mentra data",
            )

    # --- campaign scheduler state ---
    if CAMPAIGN_STATUS.exists():
        reg.data["campaign_status"] = json.loads(CAMPAIGN_STATUS.read_text())

    # --- standalone result artifacts ---
    if RESULTS.exists():
        for p in sorted(RESULTS.glob("*.json")):
            reg.fingerprint_file(f"result::{p.stem}", str(p.relative_to(REPO_ROOT)), role="evaluation artifact")

    # Storage hygiene: drop `result::*` fingerprints for files deleted since the
    # last refresh, so the registry never claims an artifact exists when it doesn't.
    stale = [k for k, v in reg.data["artifacts"].items()
            if k.startswith("result::") and not (REPO_ROOT / v.get("path", "")).exists()]
    for k in stale:
        del reg.data["artifacts"][k]

    reg.save()
    if stale:
        print(f"pruned {len(stale)} stale result artifact(s): {stale}")
    exps = reg.data.get("experiments", {})
    print(f"registry updated: {len(reg.data['artifacts'])} artifacts, {len(exps)} experiments")
    for k, v in exps.items():
        b = v.get("best") or {}
        print(f"  {v.get('status','?'):8s} {k}  best_sel={b.get('selection')} step={b.get('step')}")


if __name__ == "__main__":
    main()
