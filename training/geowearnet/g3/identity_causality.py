"""G3 identity-shortcut causality tests on held-out synthetic speakers.

This experiment is deliberately separate from MMCSG official dev. It uses the
frozen G2 checkpoint and the synthetic test speaker pool, whose speakers are
disjoint from the synthetic training pool. The strongest test holds the exact
source waveform fixed and changes only the acoustic geometry.

The decision thresholds below are predeclared for this experiment:
  * role-flip accuracy >= 0.80;
  * geometry-dominance median ratio >= 1.50;
  * grouped role-flip accuracy >= 0.75.

They are not product acceptance thresholds. They only classify whether this
diagnostic provides enough causal evidence to replace the old SUSPECTED label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .. import scenes as SC
from .. import simulate_s1 as SIM
from ..acoustics import SAMPLE_RATE, log_mel
from ..data import physical_feature_indices
from ..features import extract_features
from ..mmcsg.norm_stats import path_for
from ..mmcsg_transfer import load_checkpoint
from ..manifests import load_speaker_index


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = REPO_ROOT / "training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt"
DEFAULT_OUT = REPO_ROOT / "evaluation/geowearnet/g3/identity_causality_report.json"
ROLE_FLIP_MIN = 0.80
GROUPED_ROLE_FLIP_MIN = 0.75
DOMINANCE_MIN = 1.50


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stats(checkpoint: dict) -> Dict[str, np.ndarray] | None:
    cfg = checkpoint.get("config", {})
    if cfg.get("normalization", "global") == "none":
        return None
    channel = cfg.get("channel", 2)
    path = path_for(channel, bool(cfg.get("drop_amplitude_features", False)))
    if not path.is_file():
        raise FileNotFoundError(f"missing normalization stats: {path}")
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def _device() -> SIM.DeviceConfig:
    return SIM.DeviceConfig(
        tilt_db_per_oct=0.0,
        resonances=[],
        hp_cutoff_hz=40.0,
        recording_gain_db=0.0,
        compressor_ratio=1.0,
        clip_drive=None,
        wind_lf_db=None,
        noise_snr_db=None,
    )


def _audio_through_geometry(source: np.ndarray, rig: SIM.Rig, room: SIM.Room,
                            geometry: SIM.SourceGeometry, seed: int,
                            generation: str = "S2") -> np.ndarray:
    # Reusing the seed keeps reflection/tail randomness matched while the
    # source geometry changes, so role flips do not accidentally compare two
    # unrelated rooms or device responses.
    rng = np.random.default_rng(seed)
    ir, _ = SIM.build_ir(geometry, rig, room, rng, generation)
    return SIM.apply_device_chain(
        SIM.convolve(source, ir), _device(), np.random.default_rng(seed)
    ).astype(np.float32)


def _prepare(model, model_cfg, train_cfg: dict, stats: Dict[str, np.ndarray] | None,
             audio: np.ndarray) -> Dict[str, np.ndarray | float]:
    lm = log_mel(audio, n_mels=model_cfg.n_mels, sr=SAMPLE_RATE)
    keep = physical_feature_indices(bool(train_cfg.get("drop_amplitude_features", False)))
    if model_cfg.use_physical_features:
        ph = extract_features(audio)[:, keep]
    else:
        ph = np.zeros((len(lm), 0), dtype=np.float32)
    n = min(len(lm), len(ph)) if model_cfg.use_physical_features else len(lm)
    lm = np.ascontiguousarray(lm[:n], dtype=np.float32)
    ph = np.ascontiguousarray(ph[:n], dtype=np.float32)
    raw_lm, raw_ph = lm.copy(), ph.copy()
    if train_cfg.get("normalization", "global") == "global":
        assert stats is not None
        lm = (lm - stats["mel_mean"]) / stats["mel_std"]
        if ph.shape[1]:
            ph = (ph - stats["phys_mean"]) / stats["phys_std"]

    x = torch.from_numpy(lm[None])
    p = torch.from_numpy(ph[None]) if ph.shape[1] else None
    with torch.inference_mode():
        output = model(x, p)
        trunk = model._trunk(x)
        fused = model.embed(x, p)
    return {
        "wearer_logit": float(output["wearer_logits"][0].mean().item()),
        "environment_logit": float(output["environment_logits"][0].mean().item()),
        "raw_lm": raw_lm,
        "raw_ph": raw_ph,
        "input": np.concatenate([lm, ph], axis=1),
        "early": trunk[0].numpy(),
        "late": fused[0].numpy(),
    }


def _prob(logit: float) -> float:
    return float(1.0 / (1.0 + math.exp(-np.clip(logit, -60.0, 60.0))))


def _summary(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.concatenate([x.mean(axis=0), x.std(axis=0)]).astype(np.float32)


def _bootstrap(values: Iterable[float], seed: int = 0, n: int = 1000) -> List[float]:
    x = np.asarray(list(values), dtype=np.float64)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _classifier_baselines(rows: List[Dict[str, object]], feature_names: List[str]) -> Dict[str, object]:
    labels = np.array([r["speaker"] for r in rows])
    unique = sorted(set(labels.tolist()))
    label_id = {x: i for i, x in enumerate(unique)}
    y = np.array([label_id[x] for x in labels])
    out: Dict[str, object] = {"n_labels": len(unique), "chance_accuracy": 1 / max(len(unique), 1)}
    for name in feature_names:
        X = np.stack([r[name] for r in rows])
        train = np.array([i for i, r in enumerate(rows) if int(r["repeat"]) < 2])
        test = np.array([i for i, r in enumerate(rows) if int(r["repeat"]) == 2])
        scaler = StandardScaler().fit(X[train])
        clf = LogisticRegression(max_iter=1000, random_state=0).fit(scaler.transform(X[train]), y[train])
        accuracy = float(clf.score(scaler.transform(X[test]), y[test]))
        out[name] = {
            "train_samples": int(len(train)),
            "test_samples": int(len(test)),
            "heldout_accuracy": accuracy,
            "accuracy_over_chance": accuracy / out["chance_accuracy"],
        }
    return out


def _read_source(pool: SC.SpeechPool, speaker: str, seed: int, seconds: float = 6.0) -> np.ndarray:
    return pool.read_speech(speaker, int(seconds * SAMPLE_RATE), np.random.default_rng(seed))


def run(checkpoint_path: Path, out_path: Path, n_speakers: int = 20,
        repeats: int = 3, seed: int = 9031) -> Dict[str, object]:
    torch.set_num_threads(1)
    model, model_cfg, checkpoint = load_checkpoint(checkpoint_path)
    train_cfg = checkpoint.get("config", {})
    stats = _stats(checkpoint)
    model.eval()

    pool = SC.SpeechPool(load_speaker_index("test"))
    speakers = pool.speakers[:n_speakers]
    if len(speakers) < 6:
        raise RuntimeError(f"need at least 6 held-out synthetic speakers, got {len(speakers)}")

    baseline_rows: List[Dict[str, object]] = []
    rig = SIM.Rig(0.084, 0.125, np.deg2rad(70.0))
    room = SIM.Room(80.0, 0.45)
    wearer_geometry = SIM.SourceGeometry(rig.mouth_range_m, rig.mouth_angle_rad, "wearer")
    env_geometry = SIM.SourceGeometry(2.0, np.deg2rad(125.0), "environment")
    base_ir_rng = np.random.default_rng(seed)
    base_ir, _ = SIM.build_ir(wearer_geometry, rig, room, base_ir_rng, "S2")

    for si, speaker in enumerate(speakers):
        for repeat in range(repeats):
            source = _read_source(pool, speaker, seed + si * 100 + repeat)
            audio = SIM.apply_device_chain(
                SIM.convolve(source, base_ir), _device(), np.random.default_rng(seed + si * 100 + repeat)
            ).astype(np.float32)
            f = _prepare(model, model_cfg, train_cfg, stats, audio)
            baseline_rows.append({
                "speaker": speaker,
                "repeat": repeat,
                "physical": _summary(f["raw_ph"]),
                "log_mel": _summary(f["raw_lm"]),
                "input": _summary(f["input"]),
                "early": _summary(f["early"]),
                "late": _summary(f["late"]),
                "final_pre_head": _summary(f["late"]),
                "wearer_probability": _prob(float(f["wearer_logit"])),
            })

    pair_rows = []
    grouped = {speaker: [] for speaker in speakers}
    identity_deltas = []
    geometry_deltas = []
    pitch_deltas = []
    geometry_perturb_deltas = []

    for si, speaker in enumerate(speakers):
        for repeat in range(repeats):
            pair_seed = seed + 10_000 + si * 100 + repeat
            source = _read_source(pool, speaker, pair_seed)
            rig = SIM.Rig(
                float(np.random.default_rng(pair_seed).uniform(0.078, 0.094)),
                float(np.random.default_rng(pair_seed + 1).uniform(0.105, 0.145)),
                float(np.deg2rad(np.random.default_rng(pair_seed + 2).uniform(55.0, 90.0))),
            )
            room = SIM.Room(80.0, 0.45)
            w_geom = SIM.SourceGeometry(rig.mouth_range_m, rig.mouth_angle_rad, "wearer")
            e_geom = SIM.SourceGeometry(
                float(np.random.default_rng(pair_seed + 3).uniform(1.5, 3.0)),
                float(np.deg2rad(np.random.default_rng(pair_seed + 4).uniform(110.0, 145.0))),
                "environment",
            )
            w_audio = _audio_through_geometry(source, rig, room, w_geom, pair_seed)
            e_audio = _audio_through_geometry(source, rig, room, e_geom, pair_seed)
            w = _prepare(model, model_cfg, train_cfg, stats, w_audio)
            e = _prepare(model, model_cfg, train_cfg, stats, e_audio)
            w_prob, e_prob = _prob(w["wearer_logit"]), _prob(e["wearer_logit"])
            role_delta = w_prob - e_prob
            role_correct = role_delta > 0.0
            geometry_deltas.append(abs(role_delta))
            grouped[speaker].append(bool(role_correct))

            # Same wearer geometry, different held-out speaker: identity
            # perturbation with the transfer function held constant.
            other = speakers[(si + 1 + repeat) % len(speakers)]
            other_source = _read_source(pool, other, pair_seed + 5000)
            other_audio = _audio_through_geometry(other_source, rig, room, w_geom, pair_seed)
            other_f = _prepare(model, model_cfg, train_cfg, stats, other_audio)
            identity_delta = abs(w_prob - _prob(other_f["wearer_logit"]))
            identity_deltas.append(identity_delta)

            # Geometry perturbation with source and identity fixed.
            g2 = SIM.SourceGeometry(
                min(e_geom.range_m * 1.35, 5.0),
                min(e_geom.angle_rad + np.deg2rad(12.0), np.pi),
                "environment",
            )
            e2_audio = _audio_through_geometry(source, rig, room, g2, pair_seed)
            e2 = _prepare(model, model_cfg, train_cfg, stats, e2_audio)
            geometry_perturb_deltas.append(abs(e_prob - _prob(e2["wearer_logit"])))

            # Mild pitch perturbation as an identity-cue control. This is
            # diagnostic only; it is not used to select a checkpoint.
            try:
                import librosa

                shifted = librosa.effects.pitch_shift(source, sr=SAMPLE_RATE, n_steps=1.0)
                shifted = np.asarray(shifted[: len(source)], dtype=np.float32)
                if len(shifted) < len(source):
                    shifted = np.pad(shifted, (0, len(source) - len(shifted)))
                pitch_audio = _audio_through_geometry(shifted, rig, room, w_geom, pair_seed)
                pitch_f = _prepare(model, model_cfg, train_cfg, stats, pitch_audio)
                pitch_deltas.append(abs(w_prob - _prob(pitch_f["wearer_logit"])))
            except Exception:
                pass

            pair_rows.append({
                "speaker": speaker,
                "repeat": repeat,
                "wearer_probability": w_prob,
                "same_speaker_environment_probability": e_prob,
                "role_delta": role_delta,
                "role_flip_correct": bool(role_correct),
                "same_geometry_different_speaker_abs_delta": identity_delta,
            })

    grouped_accuracy = {k: float(np.mean(v)) for k, v in grouped.items() if v}
    role_accuracy = float(np.mean([r["role_flip_correct"] for r in pair_rows]))
    dominance = float(np.median(geometry_deltas) / max(np.median(identity_deltas), 1e-9))
    grouped_mean = float(np.mean(list(grouped_accuracy.values())))

    # Three deliberately different held-out identities per speaker are used in
    # the identity baselines. This is not a claim of a new MMCSG participant
    # split; it is the synthetic test speaker split, documented as such.
    baselines = _classifier_baselines(
        baseline_rows,
        ["physical", "log_mel", "input", "early", "late", "final_pre_head"],
    )

    content_deltas = []
    for speaker in speakers:
        speaker_rows = [r for r in baseline_rows if r["speaker"] == speaker]
        for left in range(len(speaker_rows)):
            for right in range(left + 1, len(speaker_rows)):
                content_deltas.append(abs(
                    float(speaker_rows[left]["wearer_probability"])
                    - float(speaker_rows[right]["wearer_probability"])
                ))

    original_probe_path = REPO_ROOT / "evaluation/geowearnet/mmcsg/results/identity_session_probe.json"
    original_probe = None
    if original_probe_path.is_file():
        original_probe = json.loads(original_probe_path.read_text())

    if role_accuracy >= ROLE_FLIP_MIN and grouped_mean >= GROUPED_ROLE_FLIP_MIN and dominance >= DOMINANCE_MIN:
        verdict = "IDENTITY_INFORMATION_PRESENT_BUT_NOT_CAUSAL"
    elif role_accuracy < 0.60 and dominance < 1.0:
        verdict = "IDENTITY_SHORTCUT_CONFIRMED"
    else:
        verdict = "IDENTITY_SHORTCUT_UNRESOLVED"

    result = {
        "status": "OK",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "official_dev_used": False,
        "protocol": {
            "synthetic_split": "test",
            "n_speakers": len(speakers),
            "speaker_ids": speakers,
            "repeats_per_speaker": repeats,
            "generation": "S2",
            "duration_s": 6.0,
            "same_speaker_role_flip": "same source waveform, fixed rig/room/device seed, wearer vs environment geometry",
            "same_geometry_speaker_swap": "same wearer geometry, different held-out synthetic speaker",
            "grouped_generalization": "role-flip accuracy grouped by synthetic test speaker",
            "no_mmcsg_dev_or_g2_overwrite": True,
        },
        "original_probe_context": {
            "identity_probe_artifact": "evaluation/geowearnet/mmcsg/results/identity_session_probe.json",
            "identity_target": "closed-set self_speaker label over held-out recording windows",
            "session_target": "recording ID over a within-recording temporal split",
            "not_tested_by_original": "leave-person-out generalization and same-speaker geometry counterfactuals",
            "original_result": original_probe,
        },
        "representation_identity_baselines": baselines,
        "same_speaker_role_flip": {
            "n_pairs": len(pair_rows),
            "role_flip_accuracy": role_accuracy,
            "bootstrap_ci95": _bootstrap([float(r["role_flip_correct"]) for r in pair_rows], seed + 1),
            "mean_probability_delta": float(np.mean([r["role_delta"] for r in pair_rows])),
            "median_probability_delta": float(np.median([r["role_delta"] for r in pair_rows])),
        },
        "same_geometry_speaker_swap": {
            "n_pairs": len(identity_deltas),
            "mean_abs_probability_delta": float(np.mean(identity_deltas)),
            "median_abs_probability_delta": float(np.median(identity_deltas)),
            "bootstrap_ci95_mean": _bootstrap(identity_deltas, seed + 2),
        },
        "same_speaker_different_content_control": {
            "n_pairs": len(content_deltas),
            "content_isolation": "same held-out speaker, same wearer geometry and device response, different source clip",
            "mean_abs_probability_delta": float(np.mean(content_deltas)) if content_deltas else None,
            "median_abs_probability_delta": float(np.median(content_deltas)) if content_deltas else None,
            "bootstrap_ci95_mean": _bootstrap(content_deltas, seed + 4),
        },
        "geometry_sensitivity": {
            "same_source_role_flip_abs_delta_median": float(np.median(geometry_deltas)),
            "same_source_role_flip_abs_delta_bootstrap_ci95_mean": _bootstrap(geometry_deltas, seed + 3),
            "same_source_environment_perturb_abs_delta_median": float(np.median(geometry_perturb_deltas)),
            "identity_abs_delta_median": float(np.median(identity_deltas)),
            "GEOMETRY_DOMINANCE_RATIO": dominance,
        },
        "voice_perturbation_control": {
            "n_pairs": len(pitch_deltas),
            "pitch_shift_semitones": 1.0,
            "mean_abs_probability_delta": float(np.mean(pitch_deltas)) if pitch_deltas else None,
            "median_abs_probability_delta": float(np.median(pitch_deltas)) if pitch_deltas else None,
        },
        "leave_identity_out": {
            "grouping": "synthetic test speaker; speakers are disjoint from synthetic training pool",
            "mean_group_role_flip_accuracy": grouped_mean,
            "median_group_role_flip_accuracy": float(np.median(list(grouped_accuracy.values()))),
            "p10_group_role_flip_accuracy": float(np.percentile(list(grouped_accuracy.values()), 10)),
            "worst_group_role_flip_accuracy": float(np.min(list(grouped_accuracy.values()))),
            "variance_group_role_flip_accuracy": float(np.var(list(grouped_accuracy.values()))),
            "per_speaker": grouped_accuracy,
        },
        "decision_rule": {
            "role_flip_min": ROLE_FLIP_MIN,
            "grouped_role_flip_min": GROUPED_ROLE_FLIP_MIN,
            "geometry_dominance_min": DOMINANCE_MIN,
            "result": "geometry-dominant evidence clears causal shortcut concern only when all three pass",
        },
        "verdict": verdict,
        "notes": [
            "Speaker identity remains decodable from ordinary and model representations by design; decodability alone is not a shortcut verdict.",
            "This experiment is causal evidence on synthetic held-out speakers, not Mentra-hardware validation.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-speakers", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9031)
    args = parser.parse_args()
    result = run(args.checkpoint, args.out, args.n_speakers, args.repeats, args.seed)
    print(json.dumps({
        "status": result["status"],
        "verdict": result["verdict"],
        "role_flip_accuracy": result["same_speaker_role_flip"]["role_flip_accuracy"],
        "geometry_dominance_ratio": result["geometry_sensitivity"]["GEOMETRY_DOMINANCE_RATIO"],
        "leave_identity_out_mean": result["leave_identity_out"]["mean_group_role_flip_accuracy"],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
