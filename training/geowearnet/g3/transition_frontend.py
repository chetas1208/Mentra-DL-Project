"""Causal gate onset and first-word-exposure audit for frozen G2.

The product matrix reports recognition and leakage over full windows.  This
small companion measurement isolates the onset behaviour that those aggregate
metrics can hide.  It does not call ASR and therefore does *not* claim word
recognition deletion: it measures whether the first labelled wearer word is
actually exposed to the gate (gain >= 0.5) and labels that limitation plainly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from evaluation.agent_audio import mmcsg_bridge as B
from evaluation.agent_audio.gate import GatePolicy
from evaluation.agent_audio.pipelines import PipelineContext, run_pipeline
from evaluation.agent_audio.run_matrix import mmcsg_items


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = REPO_ROOT / "training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt"
DEFAULT_OUT = REPO_ROOT / "evaluation/geowearnet/g3/transition_frontend_audit.json"
FRAME_MS = 10.0
FIRST_WORD_WINDOW_S = 1.5
INITIAL_WORD_EXPOSURE_S = 0.25
ONSET_WINDOW_FRAMES = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rising_edges(active: np.ndarray) -> np.ndarray:
    x = np.asarray(active, dtype=bool)
    return np.flatnonzero(np.logical_and(x, np.concatenate(([True], ~x[:-1]))))


def first_word_exposure(gain: np.ndarray, wearer_active: np.ndarray,
                        self_words: Sequence[tuple[float, float, str]]) -> List[dict]:
    """Return one first-word gate-exposure result per wearer speech onset."""
    out = []
    for onset in rising_edges(wearer_active):
        onset_s = onset * FRAME_MS / 1000.0
        candidates = [
            word for word in self_words
            if word[1] > onset_s and word[0] < onset_s + FIRST_WORD_WINDOW_S
        ]
        if not candidates:
            continue
        start, end, _ = min(candidates, key=lambda word: (word[0], word[1]))
        first = max(start, onset_s, 0.0)
        last = min(max(end, first), first + INITIAL_WORD_EXPOSURE_S)
        a = max(0, int(np.floor(first * 100.0)))
        b = min(len(gain), max(a + 1, int(np.ceil(last * 100.0))))
        exposure = float(np.mean(gain[a:b])) if b > a else 0.0
        out.append({
            "onset_frame": int(onset),
            "initial_word_gain": exposure,
            "initial_word_passes": bool(exposure >= 0.5),
        })
    return out


def onset_metrics(gain: np.ndarray, wearer_active: np.ndarray,
                  env_active: np.ndarray) -> Dict[str, List[float] | int]:
    wearer_latency, env_initial_gain, pre_onset_gain = [], [], []
    for onset in rising_edges(wearer_active):
        end = min(len(gain), onset + ONSET_WINDOW_FRAMES)
        opened = np.flatnonzero(gain[onset:end] >= 0.5)
        if len(opened):
            wearer_latency.append(float(opened[0] * FRAME_MS))
        else:
            wearer_latency.append(float("inf"))
        start = max(0, onset - 25)
        if onset > start:
            pre_onset_gain.append(float(np.mean(gain[start:onset])))
    environment_only = np.logical_and(env_active.astype(bool), ~wearer_active.astype(bool))
    for onset in rising_edges(environment_only):
        end = min(len(gain), onset + 25)
        if end > onset:
            env_initial_gain.append(float(np.mean(gain[onset:end])))
    return {
        "wearer_open_latency_ms": wearer_latency,
        "environment_initial_gain": env_initial_gain,
        "pre_wearer_onset_gain": pre_onset_gain,
    }


def _summary(values: Iterable[float], *, latency: bool = False) -> Dict[str, object]:
    x = np.asarray(list(values), dtype=np.float64)
    if latency:
        finite = x[np.isfinite(x)]
        return {
            "n_events": int(len(x)),
            "opened_within_500ms_rate": float(np.mean(x <= 500.0)) if len(x) else None,
            "not_opened_within_1s_rate": float(np.mean(~np.isfinite(x))) if len(x) else None,
            "p50_ms": float(np.median(finite)) if len(finite) else None,
            "p95_ms": float(np.percentile(finite, 95)) if len(finite) else None,
        }
    return {
        "n_events": int(len(x)),
        "mean": float(np.mean(x)) if len(x) else None,
        "p50": float(np.median(x)) if len(x) else None,
        "p95": float(np.percentile(x, 95)) if len(x) else None,
    }


def _policy_set() -> List[GatePolicy]:
    # A is the predeclared balanced policy used in the product matrix.  E is a
    # no-smoothing control, not a post-hoc optimisation candidate.
    return [
        GatePolicy(name="A_balanced"),
        GatePolicy(name="E_no_smoothing", attack_ms=0.0, release_ms=0.0,
                   min_state_ms=0.0, hangover_ms=0.0),
    ]


def run(checkpoint_path: Path = DEFAULT_CHECKPOINT, out_path: Path = DEFAULT_OUT,
        n_recordings: int = 38, window_s: float = 20.0) -> Dict[str, object]:
    torch.set_num_threads(1)
    predictor = B.GeoWearNetPredictor(checkpoint_path)
    items = mmcsg_items(n_recordings, windows_per_rec=1, window_s=window_s)
    policies: Dict[str, dict] = {}
    for policy in _policy_set():
        context = PipelineContext(predictor=predictor, policy=policy)
        wearer_latency, env_gain, pre_gain, word_results = [], [], [], []
        for index, record in enumerate(items, start=1):
            item = record["item"]
            processed = run_pipeline("GEOWEAR_GATE", item, context, policy=policy)
            assert processed.gate_gain is not None
            metrics = onset_metrics(processed.gate_gain, item.wearer_active, item.env_active)
            wearer_latency.extend(metrics["wearer_open_latency_ms"])
            env_gain.extend(metrics["environment_initial_gain"])
            pre_gain.extend(metrics["pre_wearer_onset_gain"])
            word_results.extend(first_word_exposure(
                processed.gate_gain, item.wearer_active, record.get("self_words", []),
            ))
            context._pred_cache.clear()
            print(f"[{policy.name} {index}/{len(items)}] {item.item_id}", flush=True)
        word_gain = [row["initial_word_gain"] for row in word_results]
        policies[policy.name] = {
            "policy": policy.to_json(),
            "wearer_onset": _summary(wearer_latency, latency=True),
            "environment_only_onset_initial_gain": _summary(env_gain),
            "pre_wearer_onset_gate_exposure": _summary(pre_gain),
            "first_word_gate_attenuation_proxy": {
                "definition": (
                    "For the first labelled wearer word following each wearer RTTM onset, mean gate gain "
                    "over its initial <=250 ms. This is exposure to the audio router, not ASR word error."
                ),
                **_summary(word_gain),
                "n_first_words": len(word_results),
                "pass_gain_ge_0_5_rate": (
                    float(np.mean([row["initial_word_passes"] for row in word_results]))
                    if word_results else None
                ),
            },
        }
    report = {
        "status": "SCORED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "official_dev_used": False,
        "official_eval_used": False,
        "data_boundary": "38 MMCSG internal-val windows derived only from official train; wearer-disjoint from G2 inner train.",
        "n_items": len(items),
        "total_audio_minutes": float(sum(row["item"].duration_s for row in items) / 60.0),
        "policies": policies,
        "decision": (
            "No post-hoc gate retuning was applied. A_balanced remains the measured product policy; "
            "E_no_smoothing is a causal-control measurement only, and receiver default remains passthrough."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-recordings", type=int, default=38)
    parser.add_argument("--window-s", type=float, default=20.0)
    args = parser.parse_args()
    report = run(args.checkpoint, args.out, args.n_recordings, args.window_s)
    print(json.dumps({
        "status": report["status"], "checkpoint_sha256": report["checkpoint_sha256"],
        "policies": list(report["policies"]), "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
