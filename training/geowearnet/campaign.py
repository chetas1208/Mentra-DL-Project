"""GeoWearNet campaign scheduler (Workstreams BE, BF, AR).

Runs a queue of independent training experiments across the two 3090s, one
experiment per GPU (never two conflicting DDP jobs). A failing experiment is
recorded and the queue continues -- per Workstream AR, an individual failure
must not take the campaign down.

Naming (Workstream BF) is deterministic and descriptive; runs are never
overwritten because `train.py` timestamps every run directory.

Usage
  python3 -m training.geowearnet.campaign --queue main --gpus 0,1
  python3 -m training.geowearnet.campaign --list
  python3 -m training.geowearnet.campaign --status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
PY = str(REPO_ROOT / ".venv/bin/python3")
STATUS = REPO_ROOT / "evaluation/geowearnet/campaign_status.json"
LOG_DIR = REPO_ROOT / "training/logs"

# Small thread counts everywhere: this is a SHARED, heavily contended box and
# BLAS oversubscription has already burned hours here once.
CHILD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

COMMON = dict(steps=12000, batch_size=24, num_workers=5, duration_s=6.0,
              val_every=750, val_scenes=96, ckpt_every=3000, log_every=100,
              early_stop_patience=6)


def _exp(name: str, **kw) -> dict:
    d = {"name": name, **COMMON}
    d.update(kw)
    return d


# --------------------------------------------------------------------------
# Queues
# --------------------------------------------------------------------------
QUEUES: Dict[str, List[dict]] = {
    # Workstream K -- base models per simulator generation.
    "main": [
        _exp("geowearnet_e1_s1_base", generation="S1"),
        _exp("geowearnet_e1_s2_base", generation="S2"),
        _exp("geowearnet_e1_mixed_base", generation="mixed"),
    ],
    # Workstream N (feature ablation) + O (normalisation ablation) + M (loss).
    "ablate": [
        _exp("geowearnet_e1_s1_ablate_spectral", generation="S1", use_physical=False),
        _exp("geowearnet_e1_s1_ablate_noamp", generation="S1", drop_amplitude_features=True),
        _exp("geowearnet_e1_s1_norm_none", generation="S1", normalization="none"),
        _exp("geowearnet_e1_s1_norm_cmvn", generation="S1", normalization="cmvn"),
        _exp("geowearnet_e1_s1_loss_m0", generation="S1", loss_mode="M0"),
        _exp("geowearnet_e1_s1_loss_m2", generation="S1", loss_mode="M2"),
    ],
    # Workstream P -- context sweep.
    "context": [
        _exp("geowearnet_e1_s1_ctx100", generation="S1", variant="ctx100"),
        _exp("geowearnet_e1_s1_ctx250", generation="S1", variant="ctx250"),
        _exp("geowearnet_e1_s1_ctx500", generation="S1", variant="ctx500"),
        _exp("geowearnet_e1_s1_ctx1000", generation="S1", variant="ctx1000"),
    ],
    # Workstream Q -- size sweep (context fixed at the E1 default).
    "size": [
        _exp("geowearnet_e1_s1_size50k", generation="S1", variant="size50k"),
        _exp("geowearnet_e1_s1_size250k", generation="S1", variant="size250k"),
        _exp("geowearnet_e1_s1_size500k", generation="S1", variant="size500k"),
    ],
    # Workstream R -- architecture ablation.
    "arch": [
        _exp("geowearnet_e1_s1_arch_dsconv", generation="S1", arch="dsconv"),
        _exp("geowearnet_e1_s1_arch_crnn", generation="S1", arch="crnn"),
        _exp("geowearnet_e1_s1_arch_convnext", generation="S1", arch="convnext"),
    ],
}


def build_cmd(exp: dict, device: str) -> List[str]:
    cmd = [PY, "-m", "training.geowearnet.train", "--device", device]
    for k, v in exp.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            cmd.append(flag if v else "--no-" + k.replace("_", "-"))
        else:
            cmd += [flag, str(v)]
    return cmd


def load_status() -> dict:
    if STATUS.exists():
        return json.loads(STATUS.read_text())
    return {"experiments": {}, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


_lock = threading.Lock()


def update_status(name: str, **fields) -> None:
    with _lock:
        s = load_status()
        e = s["experiments"].get(name, {})
        e.update(fields)
        e["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        s["experiments"][name] = e
        s["updated_utc"] = e["updated_utc"]
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATUS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(s, indent=2))
        os.replace(tmp, STATUS)


def run_slot(exps: List[dict], device: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for exp in exps:
        name = exp["name"]
        log = LOG_DIR / f"campaign_{name}.log"
        cmd = build_cmd(exp, device)
        update_status(name, status="RUNNING", device=device, log=str(log),
                      cmd=" ".join(cmd), config=exp)
        t0 = time.time()
        env = {**os.environ, **CHILD_ENV}
        try:
            with open(log, "w") as f:
                f.write(" ".join(cmd) + "\n\n")
                f.flush()
                rc = subprocess.call(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, env=env)
        except Exception as ex:  # never let one experiment kill the queue
            update_status(name, status="FAILED", error=repr(ex), seconds=time.time() - t0)
            continue
        # locate the run directory this experiment produced
        run_dir = None
        cands = sorted((REPO_ROOT / "training/geowearnet/runs").glob(f"{name}_*"),
                       key=lambda p: p.stat().st_mtime)
        if cands:
            run_dir = str(cands[-1])
        summary = None
        if run_dir and (Path(run_dir) / "summary.json").exists():
            summary = json.loads((Path(run_dir) / "summary.json").read_text())
        update_status(
            name,
            status="DONE" if rc == 0 else "FAILED",
            returncode=rc,
            seconds=time.time() - t0,
            run_dir=run_dir,
            best=(summary or {}).get("best"),
            params=(summary or {}).get("params"),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", nargs="*", default=["main"])
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--steps", type=int, default=None,
                    help="override COMMON steps for every experiment in this invocation")
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    a = ap.parse_args()

    if a.list:
        for q, exps in QUEUES.items():
            print(f"{q}: {[e['name'] for e in exps]}")
        return
    if a.status:
        print(json.dumps(load_status(), indent=2))
        return

    exps: List[dict] = []
    for q in a.queue:
        exps += [dict(e) for e in QUEUES[q]]
    # Budget overrides. Applied uniformly so every compared model gets the SAME
    # training budget -- an ablation run on a different budget is not an ablation.
    for e in exps:
        if a.steps is not None:
            e["steps"] = a.steps
        if a.val_every is not None:
            e["val_every"] = a.val_every
        if a.ckpt_every is not None:
            e["ckpt_every"] = a.ckpt_every
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    slots: List[List[dict]] = [[] for _ in gpus]
    for i, e in enumerate(exps):
        slots[i % len(gpus)].append(e)
        update_status(e["name"], status="QUEUED", config=e)

    print(f"campaign: {len(exps)} experiments over {len(gpus)} GPUs")
    for g, s in zip(gpus, slots):
        print(f"  cuda:{g} -> {[e['name'] for e in s]}")

    threads = [threading.Thread(target=run_slot, args=(s, f"cuda:{g}"), daemon=False)
               for g, s in zip(gpus, slots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("campaign queue complete")
    print(json.dumps(load_status(), indent=2))


if __name__ == "__main__":
    main()
