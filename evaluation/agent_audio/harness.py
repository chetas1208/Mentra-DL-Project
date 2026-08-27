"""P1.1 -- the agent-audio evaluation harness.

Takes an `EvalItem` (mixed audio + ground-truth SELF/OTHER activity + per-
speaker reference transcripts + optional commands + optional clean wearer
reference) and a `ProcessedAudio` (whatever a pipeline produced) and emits
ONE flat row of product metrics. Every other module in this lane -- pipeline
matrix, oracle comparison, overlap analysis, stress bench -- is a loop over
this function.

Row contents
    wearer_wer / wearer_ter / wearer_deletion_rate / wearer_retention
    bystander_leakage_rate                       <- the first-class metric
    speaker_attribution_error_rate + attribution_frac_{self,other,overlap}
    target_activity_precision/recall/f1          (gate decision vs truth)
    energy_leak_fraction                         (ASR-free complement)
    wearer_command_retention / false_agent_command_rate
    leaked_words_by_state / other_ref_words_by_state / self_ref_words_by_state
                                                 (feeds P1.6)
    latency_algorithmic_ms / latency_measured_ms / rtf
    asr_available                                (never silently faked)

Aggregation (`aggregate_rows`) micro-averages word-rate metrics from raw
counts and ALWAYS emits both an `official` and an `anomaly_excluded` block
per P1.13 -- a caller cannot obtain one without the other.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import anomalies as anom
from .asr import ASRBackend, ASRResult
from .commands import aggregate_command_metrics, command_metrics
from .metrics import (activity_prf, aggregate, attribute_hyp_words,
                      bystander_leakage_rate, bootstrap_ci_rate, energy_leakage,
                      wearer_transcription_metrics)
from .metrics import _levenshtein_counts
from .asr import tokenize
from .overlap import word_states
from .pipelines import EvalItem, ProcessedAudio

SR = 16000


def _leaked_words_by_state(other_words, hyp_text: str, wearer_active, env_active) -> List[int]:
    """Bucket the bystander's reference words by their dominant ground-truth
    state, and count how many of each bucket survived into the hypothesis.

    Survival is decided per-word by a fuzzy presence test against the
    hypothesis token multiset. This is an approximation (a proper alignment
    cannot attribute a hypothesis word to a *state*), and it is used ONLY for
    the P1.6 state attribution -- the headline BLR itself is computed by full
    sequence alignment in `metrics.bystander_leakage_rate`, not by this.
    """
    states = word_states(other_words, wearer_active, env_active)
    hyp = tokenize(hyp_text)
    pool: Dict[str, int] = {}
    for t in hyp:
        pool[t] = pool.get(t, 0) + 1
    leaked = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    for (s, e, word), st in zip(other_words, states):
        toks = tokenize(word)
        if not toks:
            continue
        total[st] += 1
        t = toks[0]
        if pool.get(t, 0) > 0:
            pool[t] -= 1
            leaked[st] += 1
    return leaked, total


def _ref_words_by_state(words, wearer_active, env_active) -> List[int]:
    states = word_states(words, wearer_active, env_active)
    out = [0, 0, 0, 0]
    for st in states:
        out[st] += 1
    return out


def evaluate_item(item: EvalItem, proc: ProcessedAudio, backend: ASRBackend,
                  compute_asr: bool = True) -> Dict[str, object]:
    """One item x one pipeline -> one metric row."""
    row: Dict[str, object] = {
        "item_id": item.item_id,
        "recording_id": item.recording_id or item.item_id,
        "scenario": item.scenario,
        "source": item.source,
        "pipeline": proc.pipeline,
        "detector_input": proc.detector_input,
        "duration_s": item.duration_s,
        "is_known_anomaly": anom.is_anomalous(item.recording_id or item.item_id),
    }
    row.update({k: v for k, v in proc.latency.items()})
    row.update({f"t_{k}": v for k, v in proc.timings.items()})

    # ---- gate decision quality (ASR-free, always available) ----------------
    if proc.gate_gain is not None:
        row.update(activity_prf(proc.gate_gain > 0.5, item.wearer_active))
        row["mean_gate_gain"] = float(np.mean(proc.gate_gain))
        row["gate_open_fraction"] = float(np.mean(proc.gate_gain > 0.5))
    el = energy_leakage(proc.audio, item.wearer_active, item.env_active, sr=SR)
    if el.get("defined"):
        row["energy_leak_fraction"] = el["energy_leak_fraction"]
        row["energy_wearer_only"] = el["energy_wearer_only"]
        row["energy_other_only"] = el["energy_other_only"]

    # (per-state reference word bookkeeping for P1.6 needs TIMED reference
    #  words, which only `evaluate_item_with_words` receives -- see below.)

    # ---- ASR-domain metrics ------------------------------------------------
    if not compute_asr or not backend.available:
        row["asr_available"] = False
        row["status"] = "NO_ASR"
        return row

    t0 = time.perf_counter()
    res: ASRResult = backend.transcribe_cached(proc.audio, SR)
    row["asr_available"] = True
    row["asr_cached"] = res.cached
    row["latency_measured_ms"] = 1000.0 * (float(res.latency_s) if not res.cached else 0.0)
    row["rtf"] = res.rtf if not res.cached else None
    row["hypothesis"] = res.text

    row.update(wearer_transcription_metrics(item.self_text, res.text))
    row.update(bystander_leakage_rate(item.other_text, res.text))

    if res.has_timestamps:
        att = attribute_hyp_words(res.words, item.wearer_active, item.env_active)
        row.update({k: v for k, v in att.items() if k != "per_word"})
        # Timestamp-based decomposition of what the agent RECEIVED, by the
        # ground-truth state each recognised word actually landed in. This is
        # unambiguous (a word's time span is known) and is the sound basis for
        # the P1.6 overlap attribution -- unlike the token-bag approximation
        # below, which cannot tell a bystander's "the" from the wearer's.
        c = att["attribution_counts"]
        row["hyp_words_by_state"] = [c["none"], c["self"], c["other"], c["overlap"]]

    if item.wearer_commands or item.bystander_commands:
        row.update({k: v for k, v in command_metrics(
            res.text, item.wearer_commands, item.bystander_commands).items()
            if not k.endswith("_detail")})

    row["status"] = "OK"
    return row


def evaluate_item_with_words(item: EvalItem, proc: ProcessedAudio, backend: ASRBackend,
                             self_words=None, other_words=None) -> Dict[str, object]:
    """`evaluate_item` plus the per-state word bookkeeping P1.6 needs, which
    requires timed reference words (MMCSG TSV / stress-bench construction)."""
    row = evaluate_item(item, proc, backend)
    if other_words:
        leaked, total = _leaked_words_by_state(other_words, row.get("hypothesis", ""),
                                               item.wearer_active, item.env_active)
        row["leaked_words_by_state"] = leaked
        row["other_ref_words_by_state"] = total
    if self_words:
        row["self_ref_words_by_state"] = _ref_words_by_state(
            self_words, item.wearer_active, item.env_active)
    return row


# ---------------------------------------------------------------------------
# aggregation with mandatory anomaly-sensitivity reporting (P1.13)
# ---------------------------------------------------------------------------
def aggregate_rows(rows: Sequence[Dict[str, object]], n_boot: int = 400) -> Dict[str, object]:
    """Always returns BOTH the official aggregate and the anomaly-excluded
    sensitivity aggregate. This is not optional -- P1.13 requires that a
    reader can see how much a known annotation defect moved the number."""
    all_rows, clean_rows = anom.split_rows_by_anomaly(list(rows))

    def block(rs: Sequence[Dict[str, object]]) -> Dict[str, object]:
        b = aggregate(rs)
        b.update(aggregate_command_metrics(rs))
        # recording-level bootstrap CI on the headline metric
        num = [float(r.get("n_other_words_leaked", 0) or 0) for r in rs]
        den = [float(r.get("n_other_ref_words", 0) or 0) for r in rs]
        if sum(den) > 0:
            pt, lo, hi = bootstrap_ci_rate(num, den, n_boot=n_boot)
            b["bystander_leakage_rate_ci95"] = [lo, hi]
        numw = [float(r.get("n_del", 0) or 0) for r in rs]
        denw = [float(r.get("n_self_ref_words", 0) or 0) for r in rs]
        if sum(denw) > 0:
            pt, lo, hi = bootstrap_ci_rate(numw, denw, n_boot=n_boot)
            b["wearer_deletion_rate_ci95"] = [lo, hi]
        return b

    n_anom = len(all_rows) - len(clean_rows)
    return {
        "official": block(all_rows),
        "anomaly_excluded": block(clean_rows),
        "anomaly_reporting": {
            "policy": "P1.13 -- both blocks always emitted; neither replaces the other",
            "n_rows_total": len(all_rows),
            "n_rows_excluded_as_anomalous": n_anom,
            "excluded_recording_ids": sorted(anom.anomalous_recording_ids()),
            "registry": str(anom.REGISTRY_PATH),
        },
    }


def write_result(obj: Dict[str, object], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    tmp.replace(path)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
