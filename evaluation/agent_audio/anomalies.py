"""P1.13 -- accessor + verifier for the known annotation anomaly registry.

The registry itself is data, not code:
    evaluation/geowearnet/mmcsg/known_annotation_anomalies.json

This module gives the harness two things:
  1. `anomalous_recording_ids()` so every aggregate can be emitted twice --
     an `official` figure over all recordings with labels exactly as the
     corpus ships them, and an `anomaly_excluded` sensitivity figure. Both
     are always reported; neither replaces the other. The registry never
     causes a recording to be silently dropped.
  2. `verify()` which RE-RUNS the model-independent RMS evidence against the
     raw corpus and checks the registry's recorded numbers still hold, so a
     stale or fabricated entry cannot survive unnoticed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "evaluation/geowearnet/mmcsg/known_annotation_anomalies.json"


def load_registry(path: Path = REGISTRY_PATH) -> Dict[str, object]:
    if not Path(path).exists():
        return {"schema_version": "1.0", "anomalies": [], "status": "REGISTRY_MISSING"}
    return json.loads(Path(path).read_text())


def anomalies(status: Optional[str] = "CONFIRMED_ANNOTATION_DEFECT") -> List[Dict[str, object]]:
    reg = load_registry()
    out = list(reg.get("anomalies", []))
    if status:
        out = [a for a in out if a.get("status") == status]
    return out


def anomalous_recording_ids(status: Optional[str] = "CONFIRMED_ANNOTATION_DEFECT") -> Set[str]:
    return {str(a["recording_id"]) for a in anomalies(status)}


def is_anomalous(recording_id: str) -> bool:
    """Matches windows/slices too: `slice_recording` names windows
    '<rid>@<start>+<dur>', so a substring check on the base id is correct."""
    base = str(recording_id).split("@")[0]
    return base in anomalous_recording_ids()


def split_rows_by_anomaly(rows: List[Dict[str, object]], id_key: str = "recording_id"):
    """-> (all_rows, rows_without_anomalies). Both are returned so callers
    physically cannot report one without having the other."""
    clean = [r for r in rows if not is_anomalous(str(r.get(id_key, "")))]
    return rows, clean


# ---------------------------------------------------------------------------
# verification against raw corpus (no model involved)
# ---------------------------------------------------------------------------
def verify(channel: int = 2, tolerance: float = 0.05) -> Dict[str, object]:
    """Recompute each entry's self/other RMS evidence from the raw corpus and
    compare against what the registry claims. Never writes to the corpus."""
    from training.geowearnet.mmcsg.config import resolve_root
    from training.geowearnet.mmcsg.error_analysis import self_other_rms_ratio

    root = resolve_root()
    results = []
    for a in anomalies(status=None):
        rid = a["recording_id"]
        split = a.get("corpus_split", "train")
        try:
            got = self_other_rms_ratio(root, split, rid, channel)
        except Exception as ex:
            results.append({"id": a["id"], "recording_id": rid,
                            "status": "VERIFY_ERROR", "error": repr(ex)})
            continue
        claimed = float(a["model_independent_evidence"]["this_recording"])
        actual = float(got.get("self_over_other_ratio", float("nan")))
        ok = abs(actual - claimed) <= tolerance * max(abs(claimed), 1e-9)
        results.append({
            "id": a["id"], "recording_id": rid, "channel": channel,
            "claimed_self_over_other_ratio": claimed,
            "recomputed_self_over_other_ratio": actual,
            "matches_within_tolerance": bool(ok),
            "status": "VERIFIED" if ok else "MISMATCH",
        })
    return {
        "registry": str(REGISTRY_PATH),
        "n_entries": len(results),
        "all_verified": all(r.get("status") == "VERIFIED" for r in results) if results else False,
        "entries": results,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--channel", type=int, default=2)
    a = ap.parse_args()
    if a.verify:
        print(json.dumps(verify(a.channel), indent=2))
    else:
        print(json.dumps({"anomalous_recording_ids": sorted(anomalous_recording_ids()),
                          "n_entries": len(anomalies(status=None))}, indent=2))
