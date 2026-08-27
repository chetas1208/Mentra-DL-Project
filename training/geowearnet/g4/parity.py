"""G4 WS3 -- live/offline routing parity, measured rather than assumed.

The same recorded audio is pushed through the offline product evaluator and
through the REAL live receiver components (``MentraInferenceConsumer`` ->
``AudioFrontend`` -> ``StreamingGateRouter`` -> PCM16 encode), 10 ms frame by
10 ms frame, and every difference is measured.

The comparison is deliberately DECOMPOSED, because "live differs from
offline" is useless as a finding. Four configurations isolate each cause:

  A_OFFLINE_REF        dense probabilities + the offline GeoWearGate
                       -- the exact code path that produced the G3 product
                       matrix. The reference.
  B_ROUTER_DENSE       dense probabilities + the new StreamingGateRouter fed
                       in 10 ms blocks.
                       -> isolates the ROUTER IMPLEMENTATION. Expected to be
                          bit-identical; anything else is a bug.
  C_LIVE_PROBS_OFFLINE_GATE
                       live 200 ms held probabilities + the offline gate.
                       -> isolates the DETECTOR CADENCE.
  D_LIVE_FULL          the real consumer (2 s rolling window, 200 ms hop,
                       final-frame logit) + the real AudioFrontend envelope
                       + the real float32->PCM16 encode.
                       -> the actual live path, end to end.

Reported per item and in aggregate: gate state agreement, gain envelope
error, routed-PCM error, wearer-onset gate-open timing deltas, and the PCM16
quantisation contribution reported separately so it is never confused with a
routing difference.

Data boundary: GeoWearNet-internal validation windows derived from official
MMCSG train only. No official dev/eval look. No Mentra-device audio exists in
this environment and none is simulated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from evaluation.agent_audio import mmcsg_bridge as B
from evaluation.agent_audio.gate import GatePolicy, GeoWearGate
from evaluation.agent_audio.run_matrix import mmcsg_items
from mentra.audio.consumer import MentraInferenceConsumer, pcm16_bytes_to_float32
from mentra.audio.frame import AudioFrame, MessageType
from server.audio.frontend import AudioFrontend
from server.audio.streaming_gate import StreamingGateRouter
from server.models.geowearnet import GeoWearNetDetector
from training.geowearnet.g4.probsource import live_cadence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = REPO_ROOT / "training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt"
DEFAULT_OUT = REPO_ROOT / "evaluation/geowearnet/g4/live_offline_parity.json"

SR = 16000
FRAME_SAMPLES = 160          # 10 ms -- the transport's real frame size
FRAME_MS = 10.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    """Byte-for-byte the encoder ``scripts/mentra/run_receiver.py`` uses."""
    x = np.asarray(samples, dtype=np.float32)
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _frames(audio: np.ndarray) -> List[np.ndarray]:
    n = len(audio) // FRAME_SAMPLES
    return [audio[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] for i in range(n)]


# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------
def run_offline(audio: np.ndarray, pw: np.ndarray, pe: np.ndarray,
                policy: GatePolicy) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate = GeoWearGate(policy, sr=SR)
    routed, gain = gate.apply(audio, pw, pe)
    return routed, gain, gate.frame_states


def run_router(audio: np.ndarray, pw: np.ndarray, pe: np.ndarray, policy: GatePolicy,
               preroll_ms: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense probabilities through the live router, one 10 ms block at a time.

    Audio is truncated to the frames that actually have probabilities. The
    feature extractor emits slightly fewer frames than the audio contains
    (the analysis window does not reach the final samples), and the offline
    ``apply_gain_envelope`` handles that tail by holding the last gain while
    the router -- correctly -- keeps advancing its release ramp. Comparing the
    two over that tail would report a difference that is an artefact of the
    offline helper's tail convention, not of the router. So the comparison is
    made over the region where both are defined.
    """
    router = StreamingGateRouter(policy, sr=SR, preroll_ms=preroll_ms)
    n_frames = min(len(audio) // FRAME_SAMPLES, len(pw), len(pe))
    out, gains, states = [], [], []
    for i, block in enumerate(_frames(audio)[:n_frames]):
        router.update_probabilities(pw[i], pe[i])
        routed, info = router.process(block)
        out.append(routed)
        gains.append(info.gains)
        states.append(info.states)
    return (np.concatenate(out) if out else np.zeros(0, np.float32),
            np.concatenate(gains) if gains else np.zeros(0, np.float32),
            np.concatenate(states) if states else np.zeros(0, np.int64))


def run_live_full(audio: np.ndarray, detector: GeoWearNetDetector, policy: GatePolicy,
                  preroll_ms: float = 0.0, context_s: float = 2.0, hop_s: float = 0.2
                  ) -> Dict[str, object]:
    """The REAL live path: real consumer, real frontend, real PCM16 encode.

    This is not a reimplementation of the receiver's audio handling; it calls
    the same objects ``scripts/mentra/run_receiver.py`` calls, in the same
    order, with the same 10 ms AudioFrames the browser transport sends.
    """
    consumer = MentraInferenceConsumer(
        detector, sample_rate=SR, context_s=context_s, hop_s=hop_s,
        wearer_high_threshold=0.60, wearer_low_threshold=0.40,
        environment_high_threshold=0.60, environment_low_threshold=0.40,
    )
    router = StreamingGateRouter(policy, sr=SR, preroll_ms=preroll_ms)
    frontend = AudioFrontend("geowear_envelope", router=router)
    frontend.reset()

    out_float, out_pcm, gains, states = [], [], [], []
    decisions: List[Dict[str, float]] = []
    for i, block in enumerate(_frames(audio)):
        frame = AudioFrame(
            sequence_number=i, capture_timestamp_ns=i * 10_000_000,
            sample_rate=SR, channels=1, bits_per_sample=16,
            payload=float32_to_pcm16_bytes(block),
            message_type=MessageType.AUDIO_FRAME,
        )
        samples = pcm16_bytes_to_float32(frame.payload)
        result = consumer.consume_frame(frame)
        if result is not None:
            router.update_probabilities(result.wearer_score, result.environment_score)
            decisions.append({
                "frame": i,
                "p_wearer": float(result.wearer_score),
                "p_env": float(result.environment_score),
                "consumer_state": result.state,
            })
        processed = frontend.process(samples, consumer.current_state)
        out_float.append(processed.audio)
        out_pcm.append(pcm16_bytes_to_float32(float32_to_pcm16_bytes(processed.audio)))
        gains.append(router.gain)
        states.append(router.gate.state)

    return {
        "audio_float": np.concatenate(out_float) if out_float else np.zeros(0, np.float32),
        "audio_pcm16": np.concatenate(out_pcm) if out_pcm else np.zeros(0, np.float32),
        "gain": np.asarray(gains, dtype=np.float32),
        "states": np.asarray(states, dtype=np.int64),
        "decisions": decisions,
    }


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def _err_db(reference: np.ndarray, other: np.ndarray) -> float | None:
    n = min(len(reference), len(other))
    if n == 0:
        return None
    diff = reference[:n] - other[:n]
    num = float(np.sum(reference[:n] ** 2))
    den = float(np.sum(diff ** 2))
    if den <= 0.0:
        return float("inf")
    if num <= 0.0:
        return None
    return float(10.0 * np.log10(num / den))


def compare(reference: Dict[str, np.ndarray], other: Dict[str, np.ndarray]) -> Dict[str, object]:
    ra, oa = reference["audio"], other["audio"]
    rg, og = reference["gain"], other["gain"]
    rs, os_ = reference["states"], other["states"]
    n_a, n_g, n_s = min(len(ra), len(oa)), min(len(rg), len(og)), min(len(rs), len(os_))
    corr = None
    if n_a > 1 and np.std(ra[:n_a]) > 0 and np.std(oa[:n_a]) > 0:
        corr = float(np.corrcoef(ra[:n_a], oa[:n_a])[0, 1])
    return {
        "audio_max_abs_diff": float(np.max(np.abs(ra[:n_a] - oa[:n_a]))) if n_a else None,
        "audio_rms_diff": float(np.sqrt(np.mean((ra[:n_a] - oa[:n_a]) ** 2))) if n_a else None,
        "audio_error_snr_db": _err_db(ra, oa),
        "audio_pearson_r": corr,
        "audio_bit_identical": bool(n_a and np.array_equal(ra[:n_a], oa[:n_a])),
        "gain_max_abs_diff": float(np.max(np.abs(rg[:n_g] - og[:n_g]))) if n_g else None,
        "gain_mae": float(np.mean(np.abs(rg[:n_g] - og[:n_g]))) if n_g else None,
        "state_agreement": float(np.mean(rs[:n_s] == os_[:n_s])) if n_s else None,
        "open_fraction_reference": float(np.mean(rg[:n_g] >= 0.5)) if n_g else None,
        "open_fraction_other": float(np.mean(og[:n_g] >= 0.5)) if n_g else None,
    }


def onset_open_latencies(gain: np.ndarray, wearer_active: np.ndarray,
                         horizon_frames: int = 100) -> List[float]:
    x = np.asarray(wearer_active, dtype=bool)
    if x.size == 0:
        return []
    onsets = np.flatnonzero(np.logical_and(x, np.concatenate(([True], ~x[:-1]))))
    out = []
    for onset in onsets:
        end = min(len(gain), onset + horizon_frames)
        if end <= onset:
            continue
        opened = np.flatnonzero(gain[onset:end] >= 0.5)
        out.append(float(opened[0] * FRAME_MS) if opened.size else float("inf"))
    return out


def _latency_delta_summary(reference: List[float], other: List[float]) -> Dict[str, object]:
    pairs = [(r, o) for r, o in zip(reference, other) if np.isfinite(r) and np.isfinite(o)]
    both_missing = sum(1 for r, o in zip(reference, other)
                       if not np.isfinite(r) and not np.isfinite(o))
    disagree = sum(1 for r, o in zip(reference, other)
                   if np.isfinite(r) != np.isfinite(o))
    if not pairs:
        return {"n_paired": 0, "n_both_never_opened": both_missing,
                "n_open_disagreements": disagree}
    delta = np.asarray([o - r for r, o in pairs], dtype=np.float64)
    return {
        "n_paired": len(pairs),
        "n_both_never_opened": both_missing,
        "n_open_disagreements": disagree,
        "median_delta_ms": float(np.median(delta)),
        "mean_delta_ms": float(np.mean(delta)),
        "p95_abs_delta_ms": float(np.percentile(np.abs(delta), 95)),
        "max_abs_delta_ms": float(np.max(np.abs(delta))),
    }


# ---------------------------------------------------------------------------
def run(checkpoint_path: Path = DEFAULT_CHECKPOINT, out_path: Path = DEFAULT_OUT,
        n_recordings: int = 12, window_s: float = 20.0, preroll_ms: float = 0.0,
        policy_name: str = "A_balanced") -> Dict[str, object]:
    torch.set_num_threads(2)
    policy = GatePolicy(name=policy_name)
    predictor = B.GeoWearNetPredictor(checkpoint_path)
    detector = GeoWearNetDetector(checkpoint_path, num_threads=2)
    items = mmcsg_items(n_recordings, windows_per_rec=1, window_s=window_s)

    per_item: List[Dict[str, object]] = []
    agg: Dict[str, List[Dict[str, object]]] = {"B_ROUTER_DENSE": [],
                                               "C_LIVE_PROBS_OFFLINE_GATE": [],
                                               "D_LIVE_FULL": []}
    latency_pairs: Dict[str, Tuple[List[float], List[float]]] = {
        k: ([], []) for k in agg}
    pcm_quant: List[Dict[str, object]] = []

    for index, record in enumerate(items, start=1):
        item = record["item"]
        audio = np.ascontiguousarray(item.audio, dtype=np.float32)
        pw, pe = predictor.predict(audio)
        n = int(min(len(pw), len(pe)))
        pw, pe = pw[:n].astype(np.float32), pe[:n].astype(np.float32)
        lpw, lpe = live_cadence(pw, pe)

        a_audio, a_gain, a_states = run_offline(audio, pw, pe, policy)
        b_audio, b_gain, b_states = run_router(audio, pw, pe, policy, preroll_ms)
        c_audio, c_gain, c_states = run_offline(audio, lpw, lpe, policy)
        d = run_live_full(audio, detector, policy, preroll_ms=preroll_ms)

        ref = {"audio": a_audio, "gain": a_gain, "states": a_states}
        configs = {
            "B_ROUTER_DENSE": {"audio": b_audio, "gain": b_gain, "states": b_states},
            "C_LIVE_PROBS_OFFLINE_GATE": {"audio": c_audio, "gain": c_gain, "states": c_states},
            "D_LIVE_FULL": {"audio": d["audio_float"], "gain": d["gain"], "states": d["states"]},
        }
        row: Dict[str, object] = {"item_id": item.item_id,
                                  "duration_s": round(item.duration_s, 2)}
        ref_lat = onset_open_latencies(a_gain, item.wearer_active)
        for name, cfg in configs.items():
            result = compare(ref, cfg)
            agg[name].append(result)
            row[name] = result
            other_lat = onset_open_latencies(cfg["gain"], item.wearer_active)
            latency_pairs[name][0].extend(ref_lat)
            latency_pairs[name][1].extend(other_lat)

        quant = compare({"audio": d["audio_float"], "gain": d["gain"], "states": d["states"]},
                        {"audio": d["audio_pcm16"], "gain": d["gain"], "states": d["states"]})
        pcm_quant.append(quant)
        row["D_LIVE_FULL_pcm16_quantisation_only"] = {
            "audio_max_abs_diff": quant["audio_max_abs_diff"],
            "audio_error_snr_db": quant["audio_error_snr_db"],
        }
        row["n_detector_decisions"] = len(d["decisions"])
        per_item.append(row)
        print(f"[parity {index}/{len(items)}] {item.item_id} "
              f"router_identical={row['B_ROUTER_DENSE']['audio_bit_identical']} "
              f"live_state_agreement={row['D_LIVE_FULL']['state_agreement']:.4f}", flush=True)

    def _agg(rows: List[Dict[str, object]], key: str) -> float | None:
        values = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(values)) if values else None

    summary = {}
    for name, rows in agg.items():
        summary[name] = {
            "n_items": len(rows),
            "audio_bit_identical_rate": float(np.mean([r["audio_bit_identical"] for r in rows])),
            "mean_state_agreement": _agg(rows, "state_agreement"),
            "min_state_agreement": float(min(r["state_agreement"] for r in rows)),
            "mean_gain_mae": _agg(rows, "gain_mae"),
            "max_gain_max_abs_diff": float(max(r["gain_max_abs_diff"] for r in rows)),
            "mean_audio_error_snr_db": _agg(rows, "audio_error_snr_db"),
            "mean_audio_pearson_r": _agg(rows, "audio_pearson_r"),
            "mean_open_fraction_reference": _agg(rows, "open_fraction_reference"),
            "mean_open_fraction_other": _agg(rows, "open_fraction_other"),
            "wearer_onset_open_latency_delta": _latency_delta_summary(*latency_pairs[name]),
        }

    report = {
        "status": "MEASURED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "workstream": "G4_WS3_LIVE_OFFLINE_PARITY",
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "official_dev_used": False,
        "official_eval_used": False,
        "mentra_hardware_audio_used": False,
        "data_boundary": (
            f"{len(items)} MMCSG GeoWearNet-internal validation windows derived only from "
            "official train. No Mentra-device audio exists in this environment."
        ),
        "gate_policy": policy.to_json(),
        "preroll_ms": preroll_ms,
        "live_receiver_settings": {"context_s": 2.0, "hop_s": 0.2,
                                   "frame_ms": FRAME_MS,
                                   "thresholds": {"wearer_on": 0.60, "wearer_off": 0.40,
                                                  "env_on": 0.60, "env_off": 0.40}},
        "configurations": {
            "A_OFFLINE_REF": "dense probabilities + offline GeoWearGate (the G3 product-matrix path)",
            "B_ROUTER_DENSE": "dense probabilities + StreamingGateRouter (isolates router implementation)",
            "C_LIVE_PROBS_OFFLINE_GATE": "200 ms held probabilities + offline gate (isolates detector cadence)",
            "D_LIVE_FULL": "real MentraInferenceConsumer + AudioFrontend envelope + PCM16 encode",
        },
        "summary": summary,
        "pcm16_quantisation_only": {
            "mean_audio_error_snr_db": float(np.mean(
                [q["audio_error_snr_db"] for q in pcm_quant
                 if q["audio_error_snr_db"] is not None and np.isfinite(q["audio_error_snr_db"])])),
            "max_audio_max_abs_diff": float(max(q["audio_max_abs_diff"] for q in pcm_quant)),
            "note": "Reported separately so PCM16 rounding is never counted as a routing difference.",
        },
        "per_item": per_item,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


HOP_GRID_S = (0.20, 0.10, 0.05)
SWEEP_PREROLL_MS = (0.0, 100.0, 150.0)


def run_hop_sweep(checkpoint_path: Path, n_recordings: int, window_s: float,
                  policy_name: str) -> Dict[str, object]:
    """WS1 follow-up: the receiver's 200 ms detector cadence is the dominant
    source of live/offline divergence. This measures, on the same windows,
    what a faster cadence actually buys (state agreement, onset timing,
    first-word retention) and what it costs (real wall-clock detector compute
    per second of audio, on this CPU, at 2 threads).

    Nothing here changes a default. It is the evidence a cadence change would
    need, produced now so the decision is not made on intuition later."""
    import time as _time

    from training.geowearnet.g4.first_word import first_word_events, _mean_gain

    torch.set_num_threads(2)
    policy = GatePolicy(name=policy_name)
    predictor = B.GeoWearNetPredictor(checkpoint_path)
    detector = GeoWearNetDetector(checkpoint_path, num_threads=2)
    items = mmcsg_items(n_recordings, windows_per_rec=1, window_s=window_s)

    from training.geowearnet.g4.probsource import shift_gain_for_preroll

    rows = []
    for hop_s in HOP_GRID_S:
        # Pre-roll does not change what the detector computes -- it only shifts
        # which audio a decided gain lands on. So the expensive live pass runs
        # once per cadence, and each pre-roll is applied to the resulting
        # envelope. That shift is asserted equal to the router's delay line in
        # tests/audio/test_streaming_gate.py, so this is an exact
        # reorganisation of the work, not an approximation.
        acc: Dict[float, Dict[str, list]] = {
            p: {"latency_ref": [], "latency_live": [], "retained": [], "agree": []}
            for p in SWEEP_PREROLL_MS}
        audio_seconds, elapsed = 0.0, 0.0
        for record in items:
            item = record["item"]
            audio = np.ascontiguousarray(item.audio, dtype=np.float32)
            pw, pe = predictor.predict(audio)
            n = int(min(len(pw), len(pe)))
            _, ref_gain, ref_states = run_offline(audio, pw[:n], pe[:n], policy)

            t0 = _time.perf_counter()
            live = run_live_full(audio, detector, policy, preroll_ms=0.0, hop_s=hop_s)
            elapsed += _time.perf_counter() - t0
            audio_seconds += item.duration_s

            events = first_word_events(item.wearer_active, record.get("self_words", []))
            ref_latency = onset_open_latencies(ref_gain, item.wearer_active)
            for preroll in SWEEP_PREROLL_MS:
                gain = shift_gain_for_preroll(live["gain"], preroll)
                m = min(len(ref_states), len(live["states"]))
                acc[preroll]["agree"].append(
                    float(np.mean(ref_states[:m] == live["states"][:m])))
                acc[preroll]["latency_ref"].extend(ref_latency)
                acc[preroll]["latency_live"].extend(
                    onset_open_latencies(gain, item.wearer_active))
                for event in events:
                    g = _mean_gain(gain, event["word_start_s"],
                                   min(event["word_end_s"], event["word_start_s"] + 0.250))
                    if g is not None:
                        acc[preroll]["retained"].append(g)

        rtf = round(elapsed / max(audio_seconds, 1e-9), 4)
        for preroll in SWEEP_PREROLL_MS:
            a = acc[preroll]
            rows.append({
                "hop_s": hop_s,
                "preroll_ms": preroll,
                "added_algorithmic_delay_ms": preroll,
                "mean_state_agreement_vs_offline": float(np.mean(a["agree"])),
                "wearer_onset_open_latency_delta": _latency_delta_summary(
                    a["latency_ref"], a["latency_live"]),
                "first_250ms_retention_pass_rate": (
                    float(np.mean([g >= 0.5 for g in a["retained"]])) if a["retained"] else None),
                "n_first_words": len(a["retained"]),
                "real_time_factor_full_live_path": rtf,
                "wall_clock_s": round(elapsed, 2),
                "audio_s": round(audio_seconds, 1),
            })
            print(f"[hop {hop_s} preroll {preroll:.0f}ms] "
                  f"agreement={rows[-1]['mean_state_agreement_vs_offline']:.4f} "
                  f"first250={rows[-1]['first_250ms_retention_pass_rate']} RTF={rtf}", flush=True)
    return {
        "note": ("Real-time factor is the WHOLE live path (feature extraction, model, "
                 "envelope, PCM16 encode) on this shared CPU host at 2 torch threads, "
                 "measured end to end -- not a model-only microbenchmark. RTF depends only "
                 "on the cadence, not on the pre-roll, which is why it repeats across "
                 "pre-roll rows for one hop."),
        "threads": 2,
        "state_agreement_caveat": (
            "state agreement is measured on the un-shifted gate state sequence, so it is a "
            "property of the cadence alone; pre-roll changes WHICH AUDIO a state lands on, "
            "not the state sequence itself."),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-recordings", type=int, default=12)
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--preroll-ms", type=float, default=0.0)
    parser.add_argument("--policy", default="A_balanced")
    parser.add_argument("--hop-sweep", action="store_true",
                        help="also measure what reducing the receiver's detector cadence buys, "
                             "and what it costs in CPU")
    args = parser.parse_args()
    report = run(args.checkpoint, args.out, args.n_recordings, args.window_s,
                 args.preroll_ms, args.policy)
    if args.hop_sweep:
        report["detector_cadence_sweep"] = run_hop_sweep(
            args.checkpoint, args.n_recordings, args.window_s, args.policy)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
