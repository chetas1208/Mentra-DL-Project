"""Decision-conditioned identity/error audit on frozen-G2 internal validation.

This is deliberately an *observational* complement to
``identity_causality.py``.  It uses the 38-recording, wearer-disjoint MMCSG
internal validation split (which comes only from the official train split),
never the guarded official dev/eval splits.  It answers the practical
question left open by a representation probe: do real false-wearer decisions
cluster by person after controlling for the observable acoustic features and
noise label available to the detector?

The answer is not used to tune G2 or to make a causal claim.  The synthetic
same-waveform geometry counterfactual remains the causal identity test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..evalsuite import core_metrics
from ..features import FEATURE_NAMES, extract_features
from ..mmcsg import audio_io as A
from ..mmcsg.config import resolve_root
from ..mmcsg.labels import frame_labels_from_rttm, read_rttm
from ..mmcsg.norm_stats import path_for
from ..mmcsg.splits import load_split
from ..mmcsg_transfer import load_checkpoint, run_recording


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = REPO_ROOT / "training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt"
DEFAULT_OUT = REPO_ROOT / "evaluation/geowearnet/g3/real_identity_error_audit.json"
KNOWN_LABEL_ANOMALY = "1302664060426140_0001_3375_22000"
MAX_ENV_FRAMES_PER_RECORDING = 1_500


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(checkpoint: dict) -> Dict[str, np.ndarray] | None:
    config = checkpoint.get("config", {})
    if config.get("normalization", "global") != "global":
        return None
    stats_path = path_for(
        config.get("channel", 2), bool(config.get("drop_amplitude_features", False)),
        bool(config.get("level_normalize", False)),
    )
    if not stats_path.is_file():
        raise FileNotFoundError(f"missing frozen-G2 normalization stats: {stats_path}")
    with np.load(stats_path) as values:
        return {key: values[key].copy() for key in values.files}


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def error_metrics(wearer_logit: np.ndarray, env_logit: np.ndarray,
                  wearer_label: np.ndarray, env_label: np.ndarray) -> Dict[str, object]:
    """Decision/error metrics using G2's predeclared logit-zero operating point."""
    n = min(len(wearer_logit), len(env_logit), len(wearer_label), len(env_label))
    w, e = wearer_logit[:n], env_logit[:n]
    yw, ye = wearer_label[:n].astype(bool), env_label[:n].astype(bool)
    solo = np.logical_xor(yw, ye)
    env_only = np.logical_and(~yw, ye)
    wearer_only = np.logical_and(yw, ~ye)
    score = w - e
    solo_auroc = None
    if solo.sum() and len(np.unique(yw[solo])) == 2:
        solo_auroc = _finite_or_none(core_metrics(yw[solo].astype(np.int8), score[solo])["auroc"])
    return {
        "n_frames": int(n),
        "n_solo_frames": int(solo.sum()),
        "n_environment_only_frames": int(env_only.sum()),
        "n_wearer_only_frames": int(wearer_only.sum()),
        "solo_auroc_wearer_minus_environment_logit": solo_auroc,
        "false_wearer_rate_on_environment_only": (
            float((w[env_only] > 0.0).mean()) if env_only.any() else None
        ),
        "wearer_miss_rate_on_wearer_only": (
            float((w[wearer_only] <= 0.0).mean()) if wearer_only.any() else None
        ),
    }


def _concatenate(rows: Sequence[dict], key: str) -> np.ndarray:
    arrays = [np.asarray(row[key]) for row in rows if len(row[key])]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)


def _group_metrics(rows: Sequence[dict], group_key: str) -> Dict[str, dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    out: Dict[str, dict] = {}
    for name, members in sorted(grouped.items()):
        metrics = error_metrics(
            _concatenate(members, "wearer_logit"), _concatenate(members, "env_logit"),
            _concatenate(members, "wearer_label"), _concatenate(members, "env_label"),
        )
        out[name] = {
            "n_recordings": len(members),
            "recording_ids": [str(member["recording_id"]) for member in members],
            "contains_known_label_anomaly": any(member["known_label_anomaly"] for member in members),
            **metrics,
        }
    return out


def distribution_summary(groups: Dict[str, dict], metric: str) -> Dict[str, object]:
    values = np.asarray([
        row[metric] for row in groups.values() if row.get(metric) is not None
    ], dtype=np.float64)
    if not len(values):
        return {"n_groups": 0, "median": None, "p10": None, "p90": None, "worst": None}
    return {
        "n_groups": int(len(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "worst": float(np.min(values)) if "auroc" in metric else float(np.max(values)),
    }


def _sample_evenly(indices: np.ndarray, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    # Deterministic spacing avoids a random seed becoming an unreported source
    # of selection variance while keeping every part of a conversation present.
    return indices[np.linspace(0, len(indices) - 1, limit, dtype=np.int64)]


def _encoded_design(numeric: np.ndarray, noise: np.ndarray, speaker: np.ndarray,
                    train: np.ndarray, test: np.ndarray, include_speaker: bool):
    scaler = StandardScaler().fit(numeric[train])
    pieces_train = [sparse.csr_matrix(scaler.transform(numeric[train]))]
    pieces_test = [sparse.csr_matrix(scaler.transform(numeric[test]))]
    for values in (noise, speaker) if include_speaker else (noise,):
        encoder = OneHotEncoder(handle_unknown="ignore")
        pieces_train.append(encoder.fit_transform(values[train, None]))
        pieces_test.append(encoder.transform(values[test, None]))
    return sparse.hstack(pieces_train, format="csr"), sparse.hstack(pieces_test, format="csr")


def identity_incremental_value(numeric: np.ndarray, noise: np.ndarray, speaker: np.ndarray,
                               target: np.ndarray, recording: np.ndarray) -> Dict[str, object]:
    """Held-recording CV: identity is tested only after acoustic/noise controls.

    Unknown speaker categories are deliberately ignored in a held fold.  That
    makes this a conservative test of repeat-person information, rather than
    an accidental lookup table for the same recording.
    """
    unique_recordings = np.unique(recording)
    if len(unique_recordings) < 5 or len(np.unique(target)) < 2:
        return {"status": "INSUFFICIENT_DATA"}
    folds = GroupKFold(n_splits=min(5, len(unique_recordings)))
    fold_rows = []
    for fold, (train, test) in enumerate(folds.split(numeric, target, groups=recording), start=1):
        if len(np.unique(target[train])) < 2 or len(np.unique(target[test])) < 2:
            continue
        row: Dict[str, object] = {"fold": fold, "n_train": int(len(train)), "n_test": int(len(test))}
        for name, include_speaker in (("acoustic_noise_controls", False), ("plus_self_speaker", True)):
            x_train, x_test = _encoded_design(numeric, noise, speaker, train, test, include_speaker)
            classifier = LogisticRegression(C=0.25, max_iter=500, random_state=0)
            classifier.fit(x_train, target[train])
            probability = classifier.predict_proba(x_test)[:, 1]
            row[name] = {
                "auroc": float(roc_auc_score(target[test], probability)),
                "brier": float(brier_score_loss(target[test], probability)),
                "log_loss": float(log_loss(target[test], probability, labels=[0, 1])),
            }
        fold_rows.append(row)
    if not fold_rows:
        return {"status": "INSUFFICIENT_VALID_FOLDS"}
    deltas = {
        metric: float(np.mean([
            row["plus_self_speaker"][metric] - row["acoustic_noise_controls"][metric]
            for row in fold_rows
        ]))
        for metric in ("auroc", "brier", "log_loss")
    }
    material = deltas["auroc"] >= 0.01 or deltas["log_loss"] <= -0.005
    return {
        "status": "SCORED",
        "method": (
            "grouped held-recording cross-validation of false-wearer decisions on environment-only "
            "frames; base model uses physical features plus noise category, expanded model also uses "
            "self-speaker identity. Unknown held-fold identities are ignored."
        ),
        "n_frames": int(len(target)),
        "n_recordings": int(len(unique_recordings)),
        "n_self_speakers": int(len(np.unique(speaker))),
        "folds": fold_rows,
        "expanded_minus_control_mean": deltas,
        "verdict": "MATERIAL_INCREMENTAL_ASSOCIATION" if material else "NO_MATERIAL_HELD_RECORDING_IMPROVEMENT",
        "interpretation": (
            "This is an observational association test, not a causal identity-shortcut test. "
            "The same-source geometry counterfactual is reported separately."
        ),
    }


def run(checkpoint_path: Path = DEFAULT_CHECKPOINT, out_path: Path = DEFAULT_OUT,
        max_env_frames_per_recording: int = MAX_ENV_FRAMES_PER_RECORDING) -> Dict[str, object]:
    torch.set_num_threads(1)
    model, model_config, checkpoint = load_checkpoint(checkpoint_path)
    model.eval()
    train_config = checkpoint.get("config", {})
    stats = _stats(checkpoint)
    channel = int(train_config.get("channel", 2))
    root = resolve_root()
    records = load_split("val")
    train_records = load_split("train")
    train_wearers = {str(row["self_speaker"]) for row in train_records}
    val_wearers = {str(row["self_speaker"]) for row in records}
    if train_wearers & val_wearers:
        raise RuntimeError("internal validation wearer identities overlap training; refusing audit")

    rows: List[dict] = []
    decision_numeric: List[np.ndarray] = []
    decision_noise: List[np.ndarray] = []
    decision_speaker: List[np.ndarray] = []
    decision_target: List[np.ndarray] = []
    decision_recording: List[np.ndarray] = []
    for index, record in enumerate(records, start=1):
        split, rid = str(record["source_split"]), str(record["recording_id"])
        audio = A.read_full(root, split, rid, channel=channel).audio
        wearer_logit, env_logit, n_frames = run_recording(
            model, model_config, train_config.get("normalization", "global"), stats, audio,
        )
        segments = read_rttm(root / "rttm" / split / f"{rid}.rttm")
        wearer_label, env_label = frame_labels_from_rttm(segments, n_frames)
        physical = extract_features(audio)[:n_frames]
        n = min(len(wearer_logit), len(env_logit), len(wearer_label), len(env_label), len(physical))
        wearer_logit, env_logit = wearer_logit[:n], env_logit[:n]
        wearer_label, env_label, physical = wearer_label[:n], env_label[:n], physical[:n]
        row = {
            "recording_id": rid,
            "session_id": rid,  # MMCSG exports no session key distinct from recording id.
            "self_speaker": str(record["self_speaker"]),
            "other_speaker": str(record["other_speaker"]),
            "noise_category": str(record["noise_category"]),
            "known_label_anomaly": rid == KNOWN_LABEL_ANOMALY,
            "wearer_logit": wearer_logit,
            "env_logit": env_logit,
            "wearer_label": wearer_label,
            "env_label": env_label,
        }
        rows.append(row)
        env_only = np.flatnonzero(np.logical_and(wearer_label == 0, env_label == 1))
        sample = _sample_evenly(env_only, max_env_frames_per_recording)
        if len(sample):
            decision_numeric.append(physical[sample])
            decision_noise.append(np.full(len(sample), row["noise_category"], dtype=object))
            decision_speaker.append(np.full(len(sample), row["self_speaker"], dtype=object))
            decision_target.append((wearer_logit[sample] > 0.0).astype(np.int8))
            decision_recording.append(np.full(len(sample), rid, dtype=object))
        print(f"[{index}/{len(records)}] scored {rid}", flush=True)

    all_metrics = error_metrics(
        _concatenate(rows, "wearer_logit"), _concatenate(rows, "env_logit"),
        _concatenate(rows, "wearer_label"), _concatenate(rows, "env_label"),
    )
    by_self = _group_metrics(rows, "self_speaker")
    by_other = _group_metrics(rows, "other_speaker")
    by_noise = _group_metrics(rows, "noise_category")
    by_recording = _group_metrics(rows, "session_id")
    incremental = identity_incremental_value(
        np.concatenate(decision_numeric), np.concatenate(decision_noise),
        np.concatenate(decision_speaker), np.concatenate(decision_target),
        np.concatenate(decision_recording),
    )
    result = {
        "status": "SCORED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "channel": channel,
        "official_dev_used": False,
        "official_eval_used": False,
        "data_boundary": (
            "MMCSG official train only: frozen G2's wearer-disjoint internal validation split. "
            "No official dev/eval access and no model fitting/tuning."
        ),
        "identity_holdout": {
            "n_internal_train_wearers": len(train_wearers),
            "n_internal_val_wearers": len(val_wearers),
            "wearer_identity_overlap_train_val": len(train_wearers & val_wearers),
            "pass": True,
        },
        "operating_point": {
            "wearer_logit_threshold": 0.0,
            "false_wearer_definition": "environment-only frame with wearer logit > 0",
        },
        "overall": all_metrics,
        "per_self_speaker": by_self,
        "per_other_speaker": by_other,
        "per_noise_category": by_noise,
        "per_recording_session": by_recording,
        "distribution_summaries": {
            "self_speaker_solo_auroc": distribution_summary(by_self, "solo_auroc_wearer_minus_environment_logit"),
            "self_speaker_false_wearer_rate": distribution_summary(by_self, "false_wearer_rate_on_environment_only"),
            "recording_session_solo_auroc": distribution_summary(by_recording, "solo_auroc_wearer_minus_environment_logit"),
            "recording_session_false_wearer_rate": distribution_summary(by_recording, "false_wearer_rate_on_environment_only"),
        },
        "decision_conditioned_identity_increment": incremental,
        "known_label_quality_context": {
            "recording_id": KNOWN_LABEL_ANOMALY,
            "status": "RETAINED_NOT_RELABELED_OR_EXCLUDED",
            "evidence_artifact": "evaluation/geowearnet/mmcsg/results/error_analysis.json",
            "note": "The known isolated RTTM role-swap suspicion is retained in every aggregate and flagged in groups.",
        },
        "feature_controls": list(FEATURE_NAMES),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-env-frames-per-recording", type=int, default=MAX_ENV_FRAMES_PER_RECORDING)
    args = parser.parse_args()
    report = run(args.checkpoint, args.out, args.max_env_frames_per_recording)
    print(json.dumps({
        "status": report["status"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "overall": report["overall"],
        "identity_increment": report["decision_conditioned_identity_increment"].get("verdict"),
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
