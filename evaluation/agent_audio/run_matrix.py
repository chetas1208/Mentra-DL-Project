"""Runner for the P1.5 pipeline matrix (+P1.3 oracle, +P1.6 overlap analysis).

    python3 -m evaluation.agent_audio.run_matrix --source mmcsg  --windows-per-rec 2
    python3 -m evaluation.agent_audio.run_matrix --source stress --seeds 3
    python3 -m evaluation.agent_audio.run_matrix --policy-sweep --source stress

CPU-only. Never touches the official dev/eval splits (the bridge refuses
them outright). Never writes into any training run directory. Safe to run
alongside a live GPU campaign -- the only shared resource is CPU, and the
default settings keep this to a single process with no worker pool.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import mmcsg_bridge as B
from .asr import availability_report as asr_report, get_backend
from .denoise import Denoiser, availability_report as dn_report
from .gate import GatePolicy, policy_variants
from .harness import aggregate_rows, evaluate_item_with_words, write_result
from .overlap import (bystander_leakage_ceiling, overlap_attributable_leakage,
                      received_words_by_state, separator_value_estimate,
                      state_occupancy, wearer_deletion_ceiling)
from .pipelines import PIPELINES, EvalItem, PipelineContext, run_pipeline
from . import stressbench as SB

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "evaluation/agent_audio/results"

# The six required rows, plus one diagnostic row that makes the P1.6
# separator-value argument computable (oracle with overlap MUTED).
MATRIX: List[Dict[str, object]] = [
    {"row": "RAW", "pipeline": "RAW", "policy": None},
    {"row": "RNNOISE", "pipeline": "RNNOISE", "policy": None},
    {"row": "GEOWEAR_GATE", "pipeline": "GEOWEAR_GATE", "policy": None},
    {"row": "RNNOISE_GEOWEAR_GATE", "pipeline": "RNNOISE_GEOWEAR_GATE", "policy": None},
    {"row": "ORACLE_GATE", "pipeline": "ORACLE_GATE", "policy": None},
    {"row": "RNNOISE_ORACLE_GATE", "pipeline": "RNNOISE_ORACLE_GATE", "policy": None},
    {"row": "ORACLE_GATE_MUTE_OVERLAP", "pipeline": "ORACLE_GATE", "policy": "mute_overlap"},
]


def _policy(kind: Optional[str], base: GatePolicy) -> GatePolicy:
    if kind == "mute_overlap":
        import dataclasses
        return dataclasses.replace(base, name=base.name + "_mute_overlap",
                                   overlap_policy="mute")
    return base


# ---------------------------------------------------------------------------
# item construction
# ---------------------------------------------------------------------------
def mmcsg_items(n_recordings: int, windows_per_rec: int, window_s: float,
                channel: int = B.DEPLOYMENT_CHANNEL) -> List[Dict[str, object]]:
    """Windows from the GeoWearNet-INTERNAL val split (official train-derived,
    wearer-disjoint). The official dev/eval splits are refused by the bridge."""
    recs = B.internal_split("val")[:n_recordings]
    out = []
    for r in recs:
        full = B.load_recording(r, channel=channel)
        # pick evenly-spaced windows, skipping the first/last 5 s
        usable = max(full.duration_s - 10.0, window_s)
        for k in range(windows_per_rec):
            start = 5.0 + (usable - window_s) * (k / max(windows_per_rec - 1, 1)) \
                if windows_per_rec > 1 else 5.0
            w = B.slice_recording(full, start, window_s)
            item = EvalItem(
                item_id=w.recording_id, audio=w.audio,
                wearer_active=w.wearer_active, env_active=w.env_active,
                self_text=w.self_text, other_text=w.other_text,
                scenario="mmcsg_conversation", source="mmcsg_internal_val",
                recording_id=full.recording_id,
            )
            out.append({"item": item, "self_words": w.self_words, "other_words": w.other_words})
        del full
    return out


def stress_items(seeds: int, duration_s: float) -> List[Dict[str, object]]:
    out = []
    for it in SB.build_suite(seeds_per_scenario=seeds, duration_s=duration_s):
        # timed reference words are not available from LibriSpeech transcripts,
        # so per-state word bookkeeping uses the whole-utterance span
        sw = [(0.0, len(it.clean_wearer) / SB.SR, w) for w in it.self_text.split()] \
            if it.self_text else []
        ow = [(0.0, len(it.clean_bystander) / SB.SR, w) for w in it.other_text.split()] \
            if it.other_text else []
        item = EvalItem(
            item_id=it.item_id, audio=it.audio,
            wearer_active=it.wearer_active, env_active=it.env_active,
            self_text=it.self_text, other_text=it.other_text,
            wearer_commands=it.wearer_commands, bystander_commands=it.bystander_commands,
            scenario=it.scenario, source="stress_bench", recording_id=it.item_id,
            clean_wearer=it.clean_wearer,
        )
        out.append({"item": item, "self_words": sw, "other_words": ow, "meta": it.meta})
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def run(items: List[Dict[str, object]], ctx: PipelineContext, backend,
        matrix: Sequence[Dict[str, object]] = MATRIX, verbose: bool = True
        ) -> Dict[str, object]:
    rows_by_pipeline: Dict[str, List[Dict[str, object]]] = {m["row"]: [] for m in matrix}
    t_start = time.time()
    for i, rec in enumerate(items):
        item: EvalItem = rec["item"]
        for m in matrix:
            pol = _policy(m["policy"], ctx.policy)
            proc = run_pipeline(m["pipeline"], item, ctx, policy=pol)
            row = evaluate_item_with_words(item, proc, backend,
                                           self_words=rec.get("self_words"),
                                           other_words=rec.get("other_words"))
            row["matrix_row"] = m["row"]
            row["gate_policy"] = pol.name
            rows_by_pipeline[m["row"]].append(row)
        # per-item prediction cache is only useful within the item
        ctx._pred_cache.clear()
        if verbose and (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(items)}] {time.time()-t_start:.0f}s elapsed", flush=True)
    return rows_by_pipeline


def summarise(rows_by_pipeline: Dict[str, List[Dict[str, object]]],
              items: List[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {"pipelines": {}}
    for name, rows in rows_by_pipeline.items():
        agg = aggregate_rows(rows)
        # two independent attributions, reported together on purpose:
        #  * token-bag (approximate; cannot disambiguate common words)
        #  * timestamp-based (exact; the one to trust)
        agg["overlap_analysis_token_bag_approx"] = overlap_attributable_leakage(rows)
        agg["received_words_by_state"] = received_words_by_state(rows)
        summary["pipelines"][name] = agg

    # P1.6 headline: ceilings + separator value
    all_rows = next(iter(rows_by_pipeline.values()))
    summary["overlap_bottleneck"] = {
        "bystander_leakage_ceiling": bystander_leakage_ceiling(all_rows),
        "wearer_deletion_ceiling": wearer_deletion_ceiling(all_rows),
    }
    op = rows_by_pipeline.get("ORACLE_GATE")
    om = rows_by_pipeline.get("ORACLE_GATE_MUTE_OVERLAP")
    if op and om:
        summary["overlap_bottleneck"]["separator_value_estimate"] = separator_value_estimate(
            summary["pipelines"]["ORACLE_GATE"]["official"],
            summary["pipelines"]["ORACLE_GATE_MUTE_OVERLAP"]["official"])
    if "ORACLE_GATE" in summary["pipelines"]:
        summary["overlap_bottleneck"]["leakage_attribution_oracle_gate_token_bag"] = \
            summary["pipelines"]["ORACLE_GATE"]["overlap_analysis_token_bag_approx"]
        summary["overlap_bottleneck"]["received_words_by_state_oracle_gate"] = \
            summary["pipelines"]["ORACLE_GATE"]["received_words_by_state"]
    # The assumption-free version of the same question: muting ALL overlap is
    # the most a gain-only router can possibly do about state 11, so the
    # leakage it removes is an upper bound on overlap's contribution.
    po = summary["pipelines"].get("ORACLE_GATE", {}).get("official", {})
    mo = summary["pipelines"].get("ORACLE_GATE_MUTE_OVERLAP", {}).get("official", {})
    if po.get("bystander_leakage_rate") is not None and \
            mo.get("bystander_leakage_rate") is not None:
        pl, ml = po["bystander_leakage_rate"], mo["bystander_leakage_rate"]
        summary["overlap_bottleneck"]["overlap_share_of_residual_leakage_exact"] = {
            "oracle_pass_overlap_blr": pl,
            "oracle_mute_overlap_blr": ml,
            "leakage_removed_by_muting_all_overlap": pl - ml,
            "share_of_oracle_residual_leakage_from_overlap": (
                (pl - ml) / pl if pl > 0 else None),
            "wearer_deletion_cost": (mo.get("wearer_deletion_rate", 0)
                                     - po.get("wearer_deletion_rate", 0)),
            "method": ("assumption-free: muting ALL overlap is the maximum a gain-only "
                       "router can do about state 11, so the leakage it removes bounds "
                       "overlap's contribution to residual leakage from above."),
        }

    # The floor BLR cannot go below (shared vocabulary between speakers).
    # Computed ASR-free from the reference transcripts alone, so it is a
    # property of the corpus, not of any pipeline. A pipeline whose measured
    # BLR sits at this floor has suppressed the bystander completely.
    from .metrics import lexical_coincidence_floor
    fn = fd = 0
    for r in items:
        it = r["item"]
        f = lexical_coincidence_floor(it.self_text, it.other_text)
        if f.get("defined"):
            fn += f["n_other_words_present_in_wearer_reference"]
            fd += f["n_other_ref_words"]
    summary["bystander_leakage_floor"] = {
        "lexical_coincidence_floor": (fn / fd) if fd else None,
        "n_other_ref_words": fd,
        "n_other_words_also_said_by_wearer": fn,
        "meaning": ("BLR cannot meaningfully fall below this value: these bystander words "
                    "are also spoken by the wearer, so they appear even in a perfectly "
                    "clean wearer-only transcript. A pipeline measuring at this floor has "
                    "achieved complete bystander suppression; BLR differences smaller than "
                    "the floor are not evidence."),
    }

    # corpus-level state occupancy over the evaluated windows
    W = np.concatenate([r["item"].wearer_active for r in items]) if items else np.zeros(0)
    E = np.concatenate([r["item"].env_active for r in items]) if items else np.zeros(0)
    summary["state_occupancy"] = state_occupancy(W, E)

    # per-scenario breakdown (the stress bench's whole point)
    by_scen: Dict[str, Dict[str, object]] = {}
    for name, rows in rows_by_pipeline.items():
        for r in rows:
            by_scen.setdefault(r.get("scenario", "?"), {}).setdefault(name, []).append(r)
    summary["by_scenario"] = {
        sc: {p: aggregate_rows(rs)["official"] for p, rs in d.items()}
        for sc, d in by_scen.items()
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["mmcsg", "stress"], default="stress")
    ap.add_argument("--n-recordings", type=int, default=38)
    ap.add_argument("--windows-per-rec", type=int, default=2)
    ap.add_argument("--window-s", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stress-duration-s", type=float, default=12.0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--checkpoint-pattern", default="g2_sim_ft_60m")
    ap.add_argument("--policy-sweep", action="store_true",
                    help="evaluate every GatePolicy variant on GEOWEAR_GATE + ORACLE_GATE")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-asr", action="store_true")
    a = ap.parse_args()

    print("== availability ==")
    print(json.dumps({"asr": asr_report(), "rnnoise": dn_report(),
                      "stress_bench": SB.availability_report()}, indent=2))

    ckpt = Path(a.checkpoint) if a.checkpoint else B.find_checkpoint(a.checkpoint_pattern)
    predictor = None
    if ckpt and Path(ckpt).exists():
        predictor = B.GeoWearNetPredictor(Path(ckpt))
        print("== detector ==")
        print(json.dumps(predictor.describe(), indent=2, default=str))
    else:
        print(f"!! no GeoWearNet checkpoint matched {a.checkpoint_pattern!r}; "
              "predicted-gate rows will be vacuous")

    backend = get_backend("null" if a.no_asr else "auto")
    ctx = PipelineContext(predictor=predictor, policy=GatePolicy(name="A_balanced"),
                          denoiser=Denoiser())

    print(f"== building items (source={a.source}) ==", flush=True)
    if a.source == "mmcsg":
        items = mmcsg_items(a.n_recordings, a.windows_per_rec, a.window_s)
    else:
        items = stress_items(a.seeds, a.stress_duration_s)
    total_s = sum(r["item"].duration_s for r in items)
    print(f"   {len(items)} items, {total_s/60:.1f} min of audio", flush=True)

    if a.policy_sweep:
        matrix = []
        for p in policy_variants():
            matrix.append({"row": f"GEOWEAR_GATE[{p.name}]", "pipeline": "GEOWEAR_GATE",
                           "policy": None, "_policy_obj": p})
            matrix.append({"row": f"ORACLE_GATE[{p.name}]", "pipeline": "ORACLE_GATE",
                           "policy": None, "_policy_obj": p})
        rows_by_pipeline = {m["row"]: [] for m in matrix}
        t0 = time.time()
        for i, rec in enumerate(items):
            item = rec["item"]
            for m in matrix:
                proc = run_pipeline(m["pipeline"], item, ctx, policy=m["_policy_obj"])
                row = evaluate_item_with_words(item, proc, backend,
                                               self_words=rec.get("self_words"),
                                               other_words=rec.get("other_words"))
                row["matrix_row"] = m["row"]
                row["gate_policy"] = m["_policy_obj"].name
                rows_by_pipeline[m["row"]].append(row)
            ctx._pred_cache.clear()
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(items)}] {time.time()-t0:.0f}s", flush=True)
        summary = summarise(rows_by_pipeline, items)
        summary["mode"] = "policy_sweep"
        summary["policies"] = [p.to_json() for p in policy_variants()]
    else:
        rows_by_pipeline = run(items, ctx, backend)
        summary = summarise(rows_by_pipeline, items)
        summary["mode"] = "pipeline_matrix"

    summary["meta"] = {
        "source": a.source,
        "n_items": len(items),
        "total_audio_minutes": total_s / 60.0,
        "checkpoint": str(ckpt) if ckpt else None,
        "detector": predictor.describe() if predictor else None,
        "asr": asr_report(),
        "rnnoise": dn_report(),
        "gate_policy": ctx.policy.to_json(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_dev_split_touched": False,
    }
    if a.source == "stress":
        summary["meta"]["DISCLAIMER"] = SB.DISCLAIMER
        summary["meta"]["is_synthetic"] = True

    tag = "policysweep" if a.policy_sweep else "matrix"
    out = Path(a.out) if a.out else RESULTS / f"agent_audio_{tag}_{a.source}.json"
    write_result(summary, out)
    print(f"\nwrote {out}")

    # terse console table
    print(f"\n{'pipeline':30s} {'wearerWER':>10s} {'BLR':>8s} {'wDel':>7s} {'F1':>6s} {'latms':>7s}")
    for name, agg in summary["pipelines"].items():
        o = agg["official"]
        print(f"{name:30s} {o.get('wearer_wer', float('nan')):10.3f} "
              f"{o.get('bystander_leakage_rate', float('nan')):8.3f} "
              f"{o.get('wearer_deletion_rate', float('nan')):7.3f} "
              f"{o.get('target_activity_f1', float('nan')):6.3f} "
              f"{o.get('latency_algorithmic_ms', float('nan')):7.1f}")


if __name__ == "__main__":
    main()
