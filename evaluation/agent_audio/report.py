"""Renders the P1 product-evaluation results into a Markdown report.

Reads only the JSON artifacts written by `run_matrix.py`; computes nothing
itself, so the report can never disagree with the measurements. Sections
sourced from the synthetic stress bench are labelled as such, every time.

    python3 -m evaluation.agent_audio.report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "evaluation/agent_audio/results"
OUT = REPO_ROOT / "docs/geowearnet_product_eval_report.md"

ROW_ORDER = ["RAW", "RNNOISE", "GEOWEAR_GATE", "RNNOISE_GEOWEAR_GATE",
             "ORACLE_GATE", "RNNOISE_ORACLE_GATE", "ORACLE_GATE_MUTE_OVERLAP"]


def _load(p: Path) -> Optional[dict]:
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def f(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _matrix_table(summary: dict, block: str = "official") -> str:
    lines = ["| pipeline | wearer WER | wearer TER | **bystander leakage** | wearer deletion | wearer retention | activity F1 | attrib err | latency ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    pipes = summary.get("pipelines", {})
    for name in ROW_ORDER + [k for k in pipes if k not in ROW_ORDER]:
        if name not in pipes:
            continue
        o = pipes[name].get(block, {})
        lines.append(
            f"| `{name}` | {f(o.get('wearer_wer'))} | {f(o.get('wearer_ter'))} | "
            f"**{f(o.get('bystander_leakage_rate'))}** | {f(o.get('wearer_deletion_rate'))} | "
            f"{f(o.get('wearer_retention'))} | {f(o.get('target_activity_f1'))} | "
            f"{f(o.get('speaker_attribution_error_rate'))} | "
            f"{f(o.get('latency_algorithmic_ms'), 1)} |")
    return "\n".join(lines)


def _ci_table(summary: dict) -> str:
    lines = ["| pipeline | bystander leakage [95% CI] | wearer deletion [95% CI] |",
             "|---|---|---|"]
    for name, agg in summary.get("pipelines", {}).items():
        o = agg.get("official", {})
        bl = o.get("bystander_leakage_rate_ci95")
        wd = o.get("wearer_deletion_rate_ci95")
        lines.append(f"| `{name}` | {f(o.get('bystander_leakage_rate'))} "
                     f"[{f(bl[0]) if bl else '—'}, {f(bl[1]) if bl else '—'}] | "
                     f"{f(o.get('wearer_deletion_rate'))} "
                     f"[{f(wd[0]) if wd else '—'}, {f(wd[1]) if wd else '—'}] |")
    return "\n".join(lines)


def _overlap_section(summary: dict) -> str:
    ob = summary.get("overlap_bottleneck", {}) or {}
    occ = (summary.get("state_occupancy", {}) or {}).get("occupancy", {})
    out = ["### State occupancy over the evaluated windows", "",
           "| state | fraction |", "|---|---|"]
    for k, v in occ.items():
        out.append(f"| `{k}` | {f(v)} |")

    # `run_matrix` names this explicitly as a token-bag approximation. Accept
    # the older key too so reports remain compatible with artifacts produced
    # before that name was clarified.
    lk = (ob.get("leakage_attribution_oracle_gate_token_bag") or
          ob.get("leakage_attribution_oracle_gate"))
    if lk:
        out += ["", "### Where the leakage lives (ORACLE gate — perfect detection)", "",
                "| state | bystander ref words | leaked | leakage rate |", "|---|---|---|---|"]
        lw = lk.get("leaked_words_by_state", {})
        ow = lk.get("other_ref_words_by_state", {})
        lr = lk.get("leakage_rate_by_state", {})
        for k in ow:
            out.append(f"| `{k}` | {f(ow.get(k), 0)} | {f(lw.get(k), 0)} | {f(lr.get(k))} |")
        pct = lk.get("pct_of_leakage_from_overlap")
        out += ["", f"**Share of residual bystander leakage attributable to state 11 "
                    f"(overlap): {f(pct, 1)}%**"]

    c = ob.get("bystander_leakage_ceiling") or {}
    d = ob.get("wearer_deletion_ceiling") or {}
    if c.get("defined"):
        out += ["", "### Router ceilings (what a gain-only router cannot beat)", "",
                f"- pass-overlap → minimum achievable bystander leakage rate: "
                f"**{f(c.get('min_possible_bystander_leakage_rate_pass_overlap'))}**",
                f"- mute-overlap → minimum achievable bystander leakage rate: **0.0**, "
                f"at a wearer-deletion cost of "
                f"**{f(d.get('min_wearer_deletion_rate_if_overlap_muted'))}**"]

    sv = ob.get("separator_value_estimate") or {}
    if sv.get("defined"):
        out += ["", "### Separator value estimate (P1.6 → P1.14 go/no-go)", "",
                f"- leakage reducible by muting overlap: "
                f"**{f(sv.get('leakage_reducible_by_muting'))}**",
                f"- wearer-speech cost of muting overlap: "
                f"**{f(sv.get('wearer_deletion_cost_of_muting'))}**", "",
                "> " + str(sv.get("interpretation", "")).replace("\n", " ")]
    return "\n".join(out)


def _scenario_table(summary: dict) -> str:
    by = summary.get("by_scenario", {}) or {}
    if not by:
        return ""
    pipes = [p for p in ROW_ORDER if any(p in d for d in by.values())]
    lines = ["| scenario | " + " | ".join(f"`{p}`" for p in pipes) + " |",
             "|---" * (len(pipes) + 1) + "|"]
    for sc in sorted(by):
        cells = []
        for p in pipes:
            o = by[sc].get(p, {})
            cells.append(f"{f(o.get('bystander_leakage_rate'))} / {f(o.get('wearer_wer'))}")
        lines.append(f"| {sc} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build(mmcsg: Optional[dict], stress: Optional[dict],
          sweep: Optional[dict]) -> str:
    L: List[str] = []
    A = L.append
    A("# GeoWearNet P1 — Product Audio Evaluation Report")
    A("")
    A("Generated by `evaluation/agent_audio/report.py` from measured artifacts in")
    A("`evaluation/agent_audio/results/`. Every number below is computed, not estimated.")
    A("")
    A("**What this measures:** not detector AUROC, but what the voice agent actually")
    A("receives — wearer WER/TER, **bystander leakage rate**, wearer deletion, false")
    A("agent command rate, latency. See `docs/geowearnet_product_architecture.md`.")
    A("")
    A("**The oracle gate is the load-bearing comparison.** It is the same gate, same")
    A("policy, same smoothing, same ASR — fed ground-truth activity instead of")
    A("predictions. So `RAW → ORACLE` is everything detection-plus-routing can deliver,")
    A("`ORACLE → GEOWEAR` is what is currently lost to detector error, and whatever the")
    A("oracle still gets wrong is the routing architecture's own ceiling.")
    A("")

    if mmcsg:
        m = mmcsg.get("meta", {})
        A("---")
        A("")
        A("## 1. Pipeline matrix — REAL wearable audio (MMCSG)")
        A("")
        A(f"- source: `{m.get('source')}` — GeoWearNet-internal val split "
          f"(derived from official MMCSG **train**; wearer-disjoint)")
        A(f"- {m.get('n_items')} windows, {f(m.get('total_audio_minutes'), 1)} minutes of audio")
        A(f"- detector: `{(m.get('detector') or {}).get('train_run')}` "
          f"({(m.get('detector') or {}).get('params')} params, "
          f"arch `{(m.get('detector') or {}).get('arch')}`)")
        A(f"- ASR: `{(m.get('asr') or {}).get('backend_id')}`")
        A(f"- denoiser: `{(m.get('rnnoise') or {}).get('backend')}`")
        A(f"- **official MMCSG dev split touched: "
          f"{m.get('official_dev_split_touched')}**")
        A("")
        A("### Official labels (all recordings, corpus labels exactly as shipped)")
        A("")
        A(_matrix_table(mmcsg, "official"))
        A("")
        A("### Anomaly-sensitivity (MMCSG-ANOM-001 excluded) — P1.13")
        A("")
        anom = (mmcsg.get("pipelines", {}).get("RAW", {}) or {}).get("anomaly_reporting", {})
        A(f"Excludes {anom.get('n_rows_excluded_as_anomalous', 0)} of "
          f"{anom.get('n_rows_total', 0)} windows belonging to the confirmed "
          f"SELF/OTHER label inversion recorded in "
          f"`evaluation/geowearnet/mmcsg/known_annotation_anomalies.json`. "
          f"Both views are always reported; neither replaces the other.")
        A("")
        A(_matrix_table(mmcsg, "anomaly_excluded"))
        A("")
        A("### Confidence intervals (recording-level bootstrap)")
        A("")
        A(_ci_table(mmcsg))
        A("")
        A("## 2. Overlap bottleneck (P1.6) — real audio")
        A("")
        A(_overlap_section(mmcsg))
        A("")

    if stress:
        m = stress.get("meta", {})
        A("---")
        A("")
        A("## 3. SYNTHETIC PRODUCT STRESS BENCH (P1.7)")
        A("")
        A(f"> **{m.get('DISCLAIMER', '')}**")
        A("")
        A("Digitally mixed LibriSpeech (CC BY 4.0) + MUSAN (CC BY 4.0) under a")
        A("synthetic room model. Useful because it varies noise type, SNR, bystander")
        A("loudness and overlap deliberately and repeatably, and preserves the clean")
        A("sources so retention/leakage are exact. It is **not** shop-floor evidence.")
        A("")
        A(f"- {m.get('n_items')} items, {f(m.get('total_audio_minutes'), 1)} minutes")
        A("")
        A(_matrix_table(stress, "official"))
        A("")
        sc = _scenario_table(stress)
        if sc:
            A("### Per scenario — `bystander leakage / wearer WER`")
            A("")
            A(sc)
            A("")
        A("### Overlap analysis — synthetic")
        A("")
        A(_overlap_section(stress))
        A("")

    if sweep:
        A("---")
        A("")
        A("## 4. Gate policy sweep (P1.2)")
        A("")
        A(_matrix_table(sweep, "official"))
        A("")

    A("---")
    A("")
    A("## Provenance and constraints honoured")
    A("")
    A("- The official MMCSG **dev** split was never evaluated by this lane; ")
    A("  `evaluation/agent_audio/mmcsg_bridge.py` refuses it outright. Final dev ")
    A("  evaluation is reserved for `training/geowearnet/mmcsg/final_selection.py` ")
    A("  under `dev_guard.py` (P1.11).")
    A("- No raw MMCSG audio, RTTM, or TSV file was modified.")
    A("- No GPU was used; no training was launched; no checkpoint was written.")
    A("- WearerSepNet remains **specification only** — see ")
    A("  `docs/geowearnet_wearersepnet_spec.md`.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--mmcsg-path", default=str(RESULTS / "agent_audio_matrix_mmcsg.json"))
    ap.add_argument("--stress-path", default=str(RESULTS / "agent_audio_matrix_stress.json"))
    ap.add_argument("--sweep-path", default=str(RESULTS / "agent_audio_policysweep_stress.json"))
    a = ap.parse_args()
    mmcsg = _load(Path(a.mmcsg_path))
    stress = _load(Path(a.stress_path))
    sweep = _load(Path(a.sweep_path))
    md = build(mmcsg, stress, sweep)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md)
    print(f"wrote {p} ({len(md)} chars); "
          f"mmcsg={'yes' if mmcsg else 'no'} stress={'yes' if stress else 'no'} "
          f"sweep={'yes' if sweep else 'no'}")


if __name__ == "__main__":
    main()
