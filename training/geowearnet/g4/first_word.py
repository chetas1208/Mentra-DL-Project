"""G4 WS2 -- first-word retention.

G3 reported a first-word gate-exposure proxy of 59.21% (balanced policy) with
a 73.68% no-smoothing control, and correctly refused to retune on that
observation alone. 59% is a product-breaking number: if two of every five
agent commands lose their first word, the agent mishears the command even
when the detector is right about *who* is talking. WS2 treats that as a
first-class product defect and measures a real fix.

WHAT IS MEASURED (all on GeoWearNet-INTERNAL validation windows derived from
official MMCSG train only -- the official dev/eval splits are untouched)

    first_100ms_retention    mean gate gain over the first 100 ms of the first
                             labelled wearer word after each wearer onset
    first_250ms_retention    same over the first 250 ms (directly comparable
                             to G3's published proxy)
    first_token_retention    mean gate gain over that word's WHOLE span
    wake_word_retention      first_token_retention restricted to turns that
                             begin after >= 1.0 s of wearer silence -- the
                             "user turns to the agent and starts talking" case
    bystander_word_exposure  mean gate gain over bystander word spans in
                             wearer-inactive frames, plus the fraction of
                             those words exposed at gain >= 0.5. This is the
                             LEAKAGE COST that any retention gain must be
                             paid for with.
    added_delay_ms           the pre-roll's real algorithmic delay

WHAT THIS IS NOT: an ASR word-error measurement. These are router-exposure
proxies, the same class of measurement G3 used, so the before/after numbers
are comparable. The ASR-based confirmation of the selected point is a
separate step (``--confirm-asr``).

Every configuration is evaluated under BOTH probability sources -- the dense
offline one the product matrix used, and the 200 ms held signal the live
receiver actually produces -- because a fix that only works offline is not a
fix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from evaluation.agent_audio import mmcsg_bridge as B
from evaluation.agent_audio.gate import GatePolicy, GeoWearGate
from evaluation.agent_audio.run_matrix import mmcsg_items
from training.geowearnet.g4.probsource import (live_cadence, shift_gain_for_preroll,
                                               verify_rolling_window_parity)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = REPO_ROOT / "training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt"
DEFAULT_OUT = REPO_ROOT / "evaluation/geowearnet/g4/first_word_retention.json"

FRAME_MS = 10.0
FIRST_WORD_SEARCH_S = 1.5      # same window G3 used to find "the first word"
WAKE_TURN_SILENCE_S = 1.0      # wearer silence before a turn for it to count as a wake

# ---- PREDECLARED grid. Fixed before any number was looked at. --------------
PREROLL_MS = (0.0, 50.0, 100.0, 150.0, 200.0)


def gate_policies() -> List[GatePolicy]:
    """The transition-parameter variants tested against each pre-roll value.

    A_balanced is the published product policy and the baseline every "after"
    number is compared against. The other two vary exactly one mechanism each
    so an improvement can be attributed:
      * A_fast_attack removes the 10 ms gain ramp (onset shaping only)
      * A_early_open lowers the hysteresis ON threshold (decision only)
    E_no_smoothing is retained as G3's causal control, not as a candidate.
    """
    return [
        GatePolicy(name="A_balanced"),
        GatePolicy(name="A_fast_attack", attack_ms=0.0),
        GatePolicy(name="A_early_open", wearer_on=0.50, wearer_off=0.35),
        GatePolicy(name="E_no_smoothing", attack_ms=0.0, release_ms=0.0,
                   min_state_ms=0.0, hangover_ms=0.0),
    ]


# ---- PREDECLARED selection rule, fixed before results existed --------------
SELECTION_RULE = {
    "objective": "maximise first_token_retention pass rate under the LIVE (200 ms held) probability source",
    "constraint_leakage": "bystander_word_exposure pass rate must not exceed the A_balanced/preroll=0 "
                          "baseline by more than 2.0 percentage points",
    "constraint_delay": "added algorithmic delay from pre-roll must be <= 150 ms",
    "tie_break": "lowest added delay, then lowest bystander exposure",
}
MAX_LEAKAGE_INCREASE_PP = 2.0
MAX_ADDED_DELAY_MS = 150.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rising_edges(active: np.ndarray) -> np.ndarray:
    x = np.asarray(active, dtype=bool)
    if x.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.logical_and(x, np.concatenate(([True], ~x[:-1]))))


def _mean_gain(gain: np.ndarray, start_s: float, end_s: float) -> float | None:
    a = max(0, int(np.floor(start_s * 1000.0 / FRAME_MS)))
    b = min(len(gain), max(a + 1, int(np.ceil(end_s * 1000.0 / FRAME_MS))))
    if b <= a:
        return None
    return float(np.mean(gain[a:b]))


def first_word_events(wearer_active: np.ndarray,
                      self_words: Sequence[Tuple[float, float, str]]) -> List[dict]:
    """One event per wearer speech onset that has a labelled first word.

    ``is_wake`` marks onsets preceded by >= WAKE_TURN_SILENCE_S of wearer
    inactivity: the realistic "start of a command to the agent" case, where
    losing the first word is most damaging.
    """
    events = []
    active = np.asarray(wearer_active, dtype=bool)
    for onset in rising_edges(active):
        onset_s = onset * FRAME_MS / 1000.0
        candidates = [w for w in self_words
                      if w[1] > onset_s and w[0] < onset_s + FIRST_WORD_SEARCH_S]
        if not candidates:
            continue
        start, end, text = min(candidates, key=lambda w: (w[0], w[1]))
        start = max(start, onset_s, 0.0)
        end = max(end, start + 0.01)
        silence_before = onset * FRAME_MS / 1000.0
        if onset > 0:
            prev_active = np.flatnonzero(active[:onset])
            if prev_active.size:
                silence_before = (onset - prev_active[-1] - 1) * FRAME_MS / 1000.0
        events.append({
            "onset_frame": int(onset),
            "word_start_s": float(start),
            "word_end_s": float(end),
            "word": text,
            "is_wake": bool(silence_before >= WAKE_TURN_SILENCE_S),
        })
    return events


def bystander_word_events(wearer_active: np.ndarray, env_active: np.ndarray,
                          other_words: Sequence[Tuple[float, float, str]]) -> List[dict]:
    """Bystander words spoken while the wearer is NOT active.

    Overlapped bystander words are excluded on purpose: a gain-only router
    cannot suppress them without deleting wearer speech, so counting them
    would blame the pre-roll for the overlap ceiling G3 already quantified.
    """
    wearer = np.asarray(wearer_active, dtype=bool)
    env = np.asarray(env_active, dtype=bool)
    env_only = np.logical_and(env, ~wearer)
    events = []
    for start, end, text in other_words:
        a = max(0, int(np.floor(start * 1000.0 / FRAME_MS)))
        b = min(len(env_only), max(a + 1, int(np.ceil(end * 1000.0 / FRAME_MS))))
        if b <= a:
            continue
        if not env_only[a:b].any():
            continue
        events.append({"word_start_s": float(start), "word_end_s": float(end), "word": text})
    return events


def _rate(values: Sequence[float], threshold: float = 0.5) -> float | None:
    if not values:
        return None
    return float(np.mean([v >= threshold for v in values]))


def _summary(values: Sequence[float]) -> Dict[str, object]:
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return {"n": 0, "mean": None, "p50": None, "pass_rate_ge_0_5": None}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "p50": float(np.median(x)),
        "pass_rate_ge_0_5": float(np.mean(x >= 0.5)),
    }


def score_config(items: List[dict], probs: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 policy: GatePolicy, preroll_ms: float, source: str) -> Dict[str, object]:
    """Score one (policy, pre-roll, probability-source) configuration."""
    first100, first250, first_token, wake_token, bystander = [], [], [], [], []
    for record in items:
        item = record["item"]
        pw, pe = probs[f"{item.item_id}|{source}"]
        gate = GeoWearGate(policy, sr=16000)
        gain = gate.run(pw, pe)
        gain = shift_gain_for_preroll(gain, preroll_ms)

        for event in first_word_events(item.wearer_active, record.get("self_words", [])):
            s, e = event["word_start_s"], event["word_end_s"]
            g100 = _mean_gain(gain, s, min(e, s + 0.100))
            g250 = _mean_gain(gain, s, min(e, s + 0.250))
            gtok = _mean_gain(gain, s, e)
            if g100 is not None:
                first100.append(g100)
            if g250 is not None:
                first250.append(g250)
            if gtok is not None:
                first_token.append(gtok)
                if event["is_wake"]:
                    wake_token.append(gtok)
        for event in bystander_word_events(item.wearer_active, item.env_active,
                                           record.get("other_words", [])):
            g = _mean_gain(gain, event["word_start_s"], event["word_end_s"])
            if g is not None:
                bystander.append(g)

    return {
        "policy": policy.name,
        "preroll_ms": float(preroll_ms),
        "probability_source": source,
        "added_delay_ms": float(preroll_ms),
        "first_100ms_retention": _summary(first100),
        "first_250ms_retention": _summary(first250),
        "first_token_retention": _summary(first_token),
        "wake_word_retention": _summary(wake_token),
        "bystander_word_exposure": _summary(bystander),
    }


def select_pareto(configs: List[dict]) -> Dict[str, object]:
    """Apply the predeclared selection rule to the LIVE-source configurations."""
    live = [c for c in configs if c["probability_source"] == "live_200ms_held"]
    baseline = next((c for c in live
                     if c["policy"] == "A_balanced" and c["preroll_ms"] == 0.0), None)
    if baseline is None:
        return {"status": "NO_BASELINE"}
    base_leak = baseline["bystander_word_exposure"]["pass_rate_ge_0_5"] or 0.0

    eligible = []
    for c in live:
        if c["policy"] == "E_no_smoothing":
            continue  # control, not a candidate
        if c["added_delay_ms"] > MAX_ADDED_DELAY_MS:
            continue
        leak = c["bystander_word_exposure"]["pass_rate_ge_0_5"] or 0.0
        if (leak - base_leak) * 100.0 > MAX_LEAKAGE_INCREASE_PP:
            continue
        eligible.append((c, leak))

    if not eligible:
        return {"status": "NO_ELIGIBLE_CONFIG", "rule": SELECTION_RULE,
                "baseline": baseline}
    best, best_leak = max(
        eligible,
        key=lambda pair: (pair[0]["first_token_retention"]["pass_rate_ge_0_5"] or 0.0,
                          -pair[0]["added_delay_ms"], -pair[1]),
    )
    return {
        "status": "SELECTED",
        "rule": SELECTION_RULE,
        "baseline": baseline,
        "selected": best,
        "n_eligible": len(eligible),
        "delta_first_token_pp": round(
            ((best["first_token_retention"]["pass_rate_ge_0_5"] or 0.0)
             - (baseline["first_token_retention"]["pass_rate_ge_0_5"] or 0.0)) * 100.0, 2),
        "delta_first_250ms_pp": round(
            ((best["first_250ms_retention"]["pass_rate_ge_0_5"] or 0.0)
             - (baseline["first_250ms_retention"]["pass_rate_ge_0_5"] or 0.0)) * 100.0, 2),
        "delta_bystander_exposure_pp": round((best_leak - base_leak) * 100.0, 2),
    }


def run(checkpoint_path: Path = DEFAULT_CHECKPOINT, out_path: Path = DEFAULT_OUT,
        n_recordings: int = 38, window_s: float = 20.0) -> Dict[str, object]:
    torch.set_num_threads(2)
    predictor = B.GeoWearNetPredictor(checkpoint_path)
    items = mmcsg_items(n_recordings, windows_per_rec=1, window_s=window_s)

    probs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    parity_probes: List[Dict[str, float]] = []
    for index, record in enumerate(items, start=1):
        item = record["item"]
        pw, pe = predictor.predict(item.audio)
        n = int(min(len(pw), len(pe), len(item.wearer_active), len(item.env_active)))
        pw, pe = pw[:n].astype(np.float32), pe[:n].astype(np.float32)
        probs[f"{item.item_id}|offline_dense"] = (pw, pe)
        probs[f"{item.item_id}|live_200ms_held"] = live_cadence(pw, pe)
        if index <= 5:
            parity_probes.append(
                verify_rolling_window_parity(predictor, item.audio, pw, pe))
        print(f"[probs {index}/{len(items)}] {item.item_id}", flush=True)

    configs: List[dict] = []
    for policy in gate_policies():
        for preroll in PREROLL_MS:
            for source in ("offline_dense", "live_200ms_held"):
                configs.append(score_config(items, probs, policy, preroll, source))
        print(f"[scored] {policy.name}", flush=True)

    selection = select_pareto(configs)
    report = {
        "status": "SCORED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "workstream": "G4_WS2_FIRST_WORD_RETENTION",
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "official_dev_used": False,
        "official_eval_used": False,
        "mentra_hardware_audio_used": False,
        "data_boundary": (
            f"{len(items)} MMCSG GeoWearNet-internal validation windows ({window_s:.0f} s each) "
            "derived only from official train; wearer-disjoint from G2 inner train. "
            "No Mentra-device audio exists in this environment and none was simulated."
        ),
        "measurement_class": (
            "gate-exposure proxy (mean applied gain over labelled word spans), NOT ASR word error"
        ),
        "n_items": len(items),
        "total_audio_minutes": float(sum(r["item"].duration_s for r in items) / 60.0),
        "rolling_window_parity_probes": parity_probes,
        "preroll_grid_ms": list(PREROLL_MS),
        "policies": [p.to_json() for p in gate_policies()],
        "configs": configs,
        "selection": selection,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-recordings", type=int, default=38)
    parser.add_argument("--window-s", type=float, default=20.0)
    args = parser.parse_args()
    report = run(args.checkpoint, args.out, args.n_recordings, args.window_s)
    print(json.dumps({"status": report["status"],
                      "selection": report["selection"].get("status"),
                      "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
