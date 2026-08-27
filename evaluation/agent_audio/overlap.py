"""P1.6 -- overlap limit analysis. The module that decides whether
WearerSepNet needs to exist.

THE ARGUMENT
------------
A gain-only router can perfectly solve states 00 / 10 / 01:
    00 silence            -> mute, nothing lost
    10 wearer only        -> pass, wearer fully retained, zero leakage
    01 environment only   -> mute, zero leakage, nothing of the wearer lost
It CANNOT solve state 11 (both talking). Multiplying the mixture by a scalar
gain either passes both voices or mutes both voices. There is no third
option available to a router.

Therefore: if most residual bystander leakage after oracle gating lives in
state 11, the remaining product error is genuinely a SEPARATION problem and
WearerSepNet is justified. If instead most leakage lives in states 01/10,
the remaining error is a DETECTION problem and the right investment is a
better detector, not a separator.

`overlap_attributable_leakage()` computes exactly that split, and
`bystander_leakage_ceiling()` computes the best any router can do.

WHY THIS IS ENERGY-DOMAIN AND WORD-DOMAIN
-----------------------------------------
Word-domain leakage (the ASR-based BLR) cannot be cleanly attributed to a
frame state, because a leaked word may straddle states. So we report BOTH:
  * an ENERGY-domain attribution over frames (exact, unambiguous), and
  * a WORD-domain attribution using each bystander reference word's dominant
    state (approximate at word boundaries, but it is the metric that maps to
    what the agent actually receives).
Neither is presented alone.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import STATE_NAMES

SR = 16000
HOP_S = 0.01


def state_array(wearer_active: np.ndarray, env_active: np.ndarray) -> np.ndarray:
    n = min(len(wearer_active), len(env_active))
    w = np.asarray(wearer_active[:n]).astype(np.int64)
    e = np.asarray(env_active[:n]).astype(np.int64)
    return w + 2 * e


def state_occupancy(wearer_active: np.ndarray, env_active: np.ndarray) -> Dict[str, object]:
    st = state_array(wearer_active, env_active)
    n = max(len(st), 1)
    return {
        "n_frames": int(len(st)),
        "occupancy": {STATE_NAMES[k]: float((st == k).sum()) / n for k in range(4)},
        "n_frames_by_state": {STATE_NAMES[k]: int((st == k).sum()) for k in range(4)},
        "seconds_by_state": {STATE_NAMES[k]: float((st == k).sum() * HOP_S) for k in range(4)},
    }


def word_states(words: Sequence[Tuple[float, float, str]], wearer_active: np.ndarray,
                env_active: np.ndarray) -> List[int]:
    """Dominant ground-truth state over each reference word's span."""
    st = state_array(wearer_active, env_active)
    out = []
    for s, e, _ in words:
        i0 = max(int(np.floor(s / HOP_S)), 0)
        i1 = min(max(int(np.ceil(e / HOP_S)), i0 + 1), len(st))
        if i1 <= i0:
            out.append(0)
            continue
        seg = st[i0:i1]
        out.append(int(np.bincount(seg, minlength=4).argmax()))
    return out


def overlap_attributable_leakage(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Given per-item rows carrying `leaked_words_by_state` (a 4-vector of
    bystander words that leaked, bucketed by their dominant state) and
    `other_ref_words_by_state`, compute the headline P1.6 number:

        % of total bystander leakage attributable to state 11 (overlap)

    That single percentage is what determines WearerSepNet's urgency.
    """
    leaked = np.zeros(4, dtype=np.float64)
    total = np.zeros(4, dtype=np.float64)
    for r in rows:
        lw = r.get("leaked_words_by_state")
        ow = r.get("other_ref_words_by_state")
        if lw is not None:
            leaked += np.asarray(lw, dtype=np.float64)
        if ow is not None:
            total += np.asarray(ow, dtype=np.float64)
    tot_leak = leaked.sum()
    out = {
        "leaked_words_by_state": {STATE_NAMES[k]: float(leaked[k]) for k in range(4)},
        "other_ref_words_by_state": {STATE_NAMES[k]: float(total[k]) for k in range(4)},
        "total_leaked_words": float(tot_leak),
        "total_other_ref_words": float(total.sum()),
    }
    if tot_leak > 0:
        out["fraction_of_leakage_from_overlap"] = float(leaked[3] / tot_leak)
        out["fraction_of_leakage_from_env_only"] = float(leaked[2] / tot_leak)
        out["pct_of_leakage_from_overlap"] = float(100.0 * leaked[3] / tot_leak)
    else:
        out["fraction_of_leakage_from_overlap"] = None
        out["pct_of_leakage_from_overlap"] = None
    # per-state leakage RATE (how leaky is each state, independent of how
    # common it is) -- distinguishes "overlap is leaky" from "overlap is big"
    out["leakage_rate_by_state"] = {
        STATE_NAMES[k]: (float(leaked[k] / total[k]) if total[k] > 0 else None) for k in range(4)
    }
    return out


def received_words_by_state(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """TIMESTAMP-BASED decomposition of what the agent actually received.

    Each recognised word is bucketed by the ground-truth state its time span
    fell in. This is exact where the token-bag attribution is not: it cannot
    confuse a bystander's "the" with the wearer's, because it never compares
    word identities at all -- only when the word was spoken.

    Reading of each bucket:
      10_wearer_only        legitimate -- what the agent is supposed to hear
      01_environment_only   PURE LEAKAGE -- the wearer was silent here
      11_overlap            AMBIGUOUS -- both were talking; without a
                            separator this word cannot be assigned to a
                            speaker at all. That ambiguity is precisely the
                            thing WearerSepNet would resolve.
      00_silence            spurious -- ASR hallucination or bleed
    """
    tot = np.zeros(4, dtype=np.float64)
    for r in rows:
        v = r.get("hyp_words_by_state")
        if v is not None:
            tot += np.asarray(v, dtype=np.float64)
    n = tot.sum()
    out = {
        "n_hyp_words": float(n),
        "hyp_words_by_state": {STATE_NAMES[k]: float(tot[k]) for k in range(4)},
    }
    if n <= 0:
        return out
    out["frac_by_state"] = {STATE_NAMES[k]: float(tot[k] / n) for k in range(4)}
    pure_leak, ambiguous = float(tot[2]), float(tot[3])
    out["pure_leakage_words_env_only"] = pure_leak
    out["ambiguous_words_overlap"] = ambiguous
    denom = pure_leak + ambiguous
    out["max_share_of_non_wearer_content_from_overlap"] = (
        float(ambiguous / denom) if denom > 0 else None)
    out["note"] = (
        "max_share_of_non_wearer_content_from_overlap is an UPPER bound on the share "
        "of problematic content attributable to overlap: it optimistically assumes "
        "every state-11 word is a bystander word, when in reality many are the "
        "wearer's own simultaneous speech. If even this upper bound is small, "
        "overlap is not where the product's residual error lives.")
    return out


def bystander_leakage_ceiling(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """The floor on bystander leakage that ANY gain-only router must accept,
    assuming a perfect detector and a pass-overlap policy: every bystander
    word whose dominant state is 11 gets through, because muting it would
    delete the wearer's simultaneous words.

    Reported next to the measured oracle number: if measured oracle leakage
    is at this floor, routing is done and only separation can improve it.
    """
    total = np.zeros(4, dtype=np.float64)
    for r in rows:
        ow = r.get("other_ref_words_by_state")
        if ow is not None:
            total += np.asarray(ow, dtype=np.float64)
    n_other = total.sum()
    if n_other <= 0:
        return {"defined": False}
    return {
        "defined": True,
        "n_other_ref_words": float(n_other),
        "n_other_words_in_overlap": float(total[3]),
        "min_possible_bystander_leakage_rate_pass_overlap": float(total[3] / n_other),
        "min_possible_bystander_leakage_rate_mute_overlap": 0.0,
        "note": ("mute-overlap achieves zero leakage only by deleting the wearer's "
                 "own simultaneous words -- see wearer_deletion_rate in the same row. "
                 "The pass-overlap floor is the leakage a router must accept if it "
                 "refuses to delete wearer speech."),
    }


def wearer_deletion_ceiling(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Mirror of the above: if overlap is MUTED, this much of the wearer's own
    speech is destroyed. The two ceilings together define the router's
    Pareto frontier -- and the area between them is exactly the value a
    separator could add."""
    total = np.zeros(4, dtype=np.float64)
    for r in rows:
        sw = r.get("self_ref_words_by_state")
        if sw is not None:
            total += np.asarray(sw, dtype=np.float64)
    n_self = total.sum()
    if n_self <= 0:
        return {"defined": False}
    return {
        "defined": True,
        "n_self_ref_words": float(n_self),
        "n_self_words_in_overlap": float(total[3]),
        "min_wearer_deletion_rate_if_overlap_muted": float(total[3] / n_self),
        "min_wearer_deletion_rate_if_overlap_passed": 0.0,
    }


def separator_value_estimate(oracle_pass_row: Dict[str, object],
                             oracle_mute_row: Dict[str, object]) -> Dict[str, object]:
    """Quantifies the gap a separator would have to close.

    A perfect router must choose a point on the segment between
    (pass-overlap: low deletion, high leakage) and (mute-overlap: zero
    overlap leakage, high deletion). A perfect separator would reach
    (low deletion, low leakage) -- the corner neither router policy can
    occupy. The size of that corner gap IS the value of WearerSepNet.
    """
    def g(row, k):
        v = row.get(k)
        return float(v) if v is not None else None

    pl, pd = g(oracle_pass_row, "bystander_leakage_rate"), g(oracle_pass_row, "wearer_deletion_rate")
    ml, md = g(oracle_mute_row, "bystander_leakage_rate"), g(oracle_mute_row, "wearer_deletion_rate")
    if None in (pl, pd, ml, md):
        return {"defined": False}
    return {
        "defined": True,
        "oracle_pass_overlap": {"bystander_leakage_rate": pl, "wearer_deletion_rate": pd},
        "oracle_mute_overlap": {"bystander_leakage_rate": ml, "wearer_deletion_rate": md},
        "leakage_reducible_by_muting": pl - ml,
        "wearer_deletion_cost_of_muting": md - pd,
        "separator_target_corner": {"bystander_leakage_rate": ml, "wearer_deletion_rate": pd},
        "interpretation": (
            "A separator is worth building iff BOTH gaps are large: muting overlap must "
            "buy a meaningful leakage reduction AND cost a meaningful amount of wearer "
            "speech. If muting costs almost no wearer speech, just mute overlap and skip "
            "the separator. If muting buys almost no leakage reduction, overlap is not "
            "where the problem lives and the separator would solve nothing."),
    }
