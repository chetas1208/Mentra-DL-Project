"""GeoWearNet experiment registry (Workstream A).

Machine-readable, append/update-only record of every GeoWearNet artifact and
experiment. This is the campaign's PRIMARY DURABLE STATE: if a session is
interrupted, a fresh continuation should be able to read
`evaluation/geowearnet/experiment_registry.json` plus `ls`/`ps` and know
exactly what has been done.

Design rules:
  * Never duplicate large files -- store path + size + sha256 + mtime only.
  * Never delete an entry. Statuses move forward (PLANNED -> RUNNING ->
    DONE / FAILED / BLOCKED); superseded entries keep their record.
  * Real values only. No placeholders. If a number is not measured yet the
    field is simply absent, never 0.0 or "TBD".

Usage (library):
    from training.geowearnet.registry import Registry
    reg = Registry()
    reg.fingerprint_file("e0_impl", "training/diagnostics/geowearnet_e0.py")
    reg.upsert_experiment("geowearnet_e1_s1_base", status="RUNNING", ...)
    reg.save()

Usage (CLI):
    python3 -m training.geowearnet.registry --refresh-fingerprints
    python3 -m training.geowearnet.registry --show
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "evaluation/geowearnet/experiment_registry.json"


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_record(rel_path: str) -> Dict[str, Any]:
    p = REPO_ROOT / rel_path
    rec: Dict[str, Any] = {"path": rel_path, "exists": p.exists()}
    if p.exists() and p.is_file():
        st = p.stat()
        rec["size_bytes"] = st.st_size
        rec["mtime_utc"] = _dt.datetime.fromtimestamp(
            st.st_mtime, _dt.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        rec["sha256"] = sha256_file(p)
    return rec


def git_fingerprint() -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "head_commit": head,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_dirty": bool(status),
        "working_tree_status_sha256": sha256_text(status) if status is not None else None,
        "note": "NO COMMITS ARE MADE BY THIS CAMPAIGN -- all work stays uncommitted on disk.",
    }


class Registry:
    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        if path.exists():
            with open(path) as f:
                self.data = json.load(f)
        else:
            self.data = {
                "schema_version": 1,
                "campaign": "GEOWEARNET CONVERGENCE CAMPAIGN G1",
                "created_utc": utcnow(),
                "commit_authorization": "WITHHELD -- do not git add/commit/push",
                "frozen_baseline_commit": None,
                "artifacts": {},
                "experiments": {},
                "workstreams": {},
                "notes": [],
            }
        self.data["updated_utc"] = utcnow()

    # ---- artifacts ----
    def fingerprint_file(self, key: str, rel_path: str, role: str = "", note: str = "") -> Dict[str, Any]:
        rec = file_record(rel_path)
        if role:
            rec["role"] = role
        if note:
            rec["note"] = note
        rec["fingerprinted_utc"] = utcnow()
        self.data["artifacts"][key] = rec
        return rec

    # ---- experiments ----
    def upsert_experiment(self, exp_id: str, **fields: Any) -> Dict[str, Any]:
        exps = self.data.setdefault("experiments", {})
        cur = exps.get(exp_id, {"experiment_id": exp_id, "created_utc": utcnow()})
        for k, v in fields.items():
            if v is not None:
                cur[k] = v
        cur["updated_utc"] = utcnow()
        exps[exp_id] = cur
        return cur

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get("experiments", {}).get(exp_id)

    # ---- workstreams ----
    def set_workstream(self, letter: str, status: str, note: str = "", artifacts: Optional[list] = None) -> None:
        ws = self.data.setdefault("workstreams", {})
        entry = ws.get(letter, {})
        entry.update({"status": status, "updated_utc": utcnow()})
        if note:
            entry["note"] = note
        if artifacts:
            entry["artifacts"] = artifacts
        ws[letter] = entry

    def add_note(self, text: str) -> None:
        self.data.setdefault("notes", []).append({"utc": utcnow(), "text": text})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_utc"] = utcnow()
        self.data["git"] = git_fingerprint()
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, sort_keys=False)
        os.replace(tmp, self.path)


# Files frozen by Workstream A. (key, path, role)
FROZEN_ARTIFACTS = [
    # --- GeoWearNet current state ---
    ("e0_impl", "training/diagnostics/geowearnet_e0.py", "E0 physical-heuristic baseline implementation"),
    ("e0_results", "evaluation/geowearnet/results/geowearnet_e0_results.json", "E0 measured results (SIMULATED S0 geometry)"),
    ("e0_run_log", "training/logs/geowearnet_e0_run.log", "E0 run log"),
    ("e1_model_impl", "training/geowearnet/model.py", "E1 causal-TCN architecture"),
    ("e1_smoke_test", "training/diagnostics/geowearnet_e1_smoke_test.py", "E1 smoke test (forward/backward/causality)"),
    ("sim_s0", "training/geowearnet/simulate.py", "S0 simulator (synthetic decaying-noise IR) -- IMMUTABLE historical baseline"),
    ("features_e0", "training/geowearnet/features.py", "E0/physical scalar feature extractor"),
    ("dataset_s0", "training/geowearnet/dataset.py", "S0 dataset builder"),
    ("license_audit", "docs/geowearnet_dataset_licenses.md", "Dataset/claim license audit (Workstream B re-verifies)"),
    ("baseline_registry_phase0", "evaluation/baselines/geowearnet_baseline_registry.json", "Earlier Phase-0 baseline fingerprint"),
    # --- immutable reference baselines (DO NOT MODIFY) ---
    ("speakernet_nemo", "models/vendor/nemo/speakerverification_speakernet.nemo", "SpeakerNet reference checkpoint (IMMUTABLE)"),
    ("speakernet_onnx", "models/sherpa-speaker/nemo_en_speakerverification_speakernet.onnx", "SpeakerNet ONNX (IMMUTABLE)"),
    ("mentrawearnet_impl", "training/models/mentrawearnet.py", "MentraWearNet architecture (IMMUTABLE for this campaign)"),
    ("mentrawearnet_v2_final", "training/checkpoints/mentrawearnet_v2_final_5000.pt", "MentraWearNet V2 reference checkpoint (IMMUTABLE)"),
    ("mentrawearnet_v3b_control", "training/checkpoints/mentrawearnet_v3b_control_latest.pt", "MentraWearNet V3B control (IMMUTABLE)"),
    ("mentrawearnet_v3b_hard", "training/checkpoints/mentrawearnet_v3b_hard_latest.pt", "MentraWearNet V3B hard-negative (IMMUTABLE)"),
    # --- manifests (speaker-disjoint LibriSpeech splits reused by GeoWearNet) ---
    ("manifest_ls_train", "evaluation/manifests/librispeech_train_clean_100_train.json", "Speaker-disjoint train manifest"),
    ("manifest_ls_val", "evaluation/manifests/librispeech_train_clean_100_val.json", "Speaker-disjoint val manifest"),
    ("manifest_ls_test", "evaluation/manifests/librispeech_train_clean_100_test.json", "Speaker-disjoint test manifest"),
]


def refresh_fingerprints() -> Registry:
    reg = Registry()
    reg.data["frozen_baseline_commit"] = git_fingerprint()["head_commit"]
    for key, path, role in FROZEN_ARTIFACTS:
        reg.fingerprint_file(key, path, role=role)
    reg.save()
    return reg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-fingerprints", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.refresh_fingerprints:
        reg = refresh_fingerprints()
        missing = [k for k, v in reg.data["artifacts"].items() if not v.get("exists")]
        print(f"registry: {len(reg.data['artifacts'])} artifacts fingerprinted -> {reg.path}")
        if missing:
            print("MISSING:", missing)
    if args.show:
        reg = Registry()
        print(json.dumps(reg.data, indent=2)[:8000])


if __name__ == "__main__":
    main()
