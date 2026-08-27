"""P1.8 -- agent command metrics.

The autobody technician's real interaction is not "say words near a
microphone", it is "issue a command the agent must act on":

    "Hey glasses, what part is this?"
    "Hey glasses, do we have a replacement in stock?"
    "Show me the repair procedure."

Two failure modes matter and they are asymmetric in cost:

  WEARER COMMAND RETENTION  (higher is better)
      the wearer issued a command and the agent still receives it after
      processing. A missed command is an annoyance -- the technician repeats
      themselves.

  FALSE AGENT COMMAND RATE  (lower is better)
      a phrase the WEARER DID NOT SAY reaches the agent as a command --
      because a coworker said it, or because gating stitched fragments into
      something command-like. This is the expensive failure: the agent acts
      (orders a part, opens a procedure, takes a photo) on someone who is not
      the wearer and did not intend to command anything. In a shared-glasses
      deployment this is also a trust/authority problem, not just an accuracy
      problem.

Both are measured on the ASR hypothesis of the PROCESSED audio, against
per-speaker reference transcripts, so they compose with the P1.5 pipeline
matrix without special-casing.

MATCHING
--------
A command is "present" in a hypothesis if its normalised token sequence
appears as a fuzzy subsequence with word-error <= `tolerance` (default 0.34,
i.e. up to a third of the command's words may be wrong -- ASR on noisy
wearable audio is not going to be exact, and requiring exact match would
make every metric read 0 and tell us nothing). Matching is done with a
sliding windowed edit distance, so a command embedded in a longer utterance
is found.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

import numpy as np

from .asr import tokenize
from .metrics import _levenshtein_counts

WAKE_PHRASE = "hey glasses"

# Autobody / warehouse-shop flavoured command set. These are the phrases the
# P1.7 stress bench actually renders as wearer or bystander speech.
COMMAND_SET: List[Dict[str, str]] = [
    {"id": "cmd_what_part", "text": "hey glasses what part is this", "intent": "identify_part"},
    {"id": "cmd_in_stock", "text": "hey glasses do we have a replacement in stock", "intent": "check_inventory"},
    {"id": "cmd_repair_proc", "text": "show me the repair procedure", "intent": "open_procedure"},
    {"id": "cmd_torque_spec", "text": "hey glasses what is the torque spec for this bolt", "intent": "lookup_spec"},
    {"id": "cmd_order_part", "text": "hey glasses order a new one", "intent": "place_order"},
    {"id": "cmd_take_photo", "text": "hey glasses take a picture of this", "intent": "camera_tool_call"},
]

# Phrases a BYSTANDER says. Some are deliberately command-shaped: that is the
# whole point -- a coworker saying "order a new one" must NOT reach the agent.
BYSTANDER_PHRASES: List[Dict[str, str]] = [
    {"id": "byst_chat", "text": "did you see the game last night", "command_shaped": False},
    {"id": "byst_lunch", "text": "i'm heading out for lunch in ten minutes", "command_shaped": False},
    {"id": "byst_order", "text": "hey glasses order a new one", "command_shaped": True},
    {"id": "byst_stock", "text": "do we have a replacement in stock", "command_shaped": True},
    {"id": "byst_photo", "text": "hey glasses take a picture of this", "command_shaped": True},
]


@dataclasses.dataclass
class CommandMatch:
    command_id: str
    present: bool
    best_error: float
    match_start_word: Optional[int]


def fuzzy_contains(hyp_tokens: Sequence[str], phrase_tokens: Sequence[str],
                   tolerance: float = 0.34) -> CommandMatch:
    """Sliding windowed edit distance: is `phrase` present anywhere in `hyp`?"""
    n = len(phrase_tokens)
    if n == 0:
        return CommandMatch("", False, 1.0, None)
    if not hyp_tokens:
        return CommandMatch("", False, 1.0, None)
    best, best_i = 1.0, None
    # allow the matched span to be a bit shorter/longer than the phrase
    for width in {max(1, n - 2), max(1, n - 1), n, n + 1, n + 2}:
        for i in range(0, max(len(hyp_tokens) - width + 1, 1)):
            win = hyp_tokens[i:i + width]
            c = _levenshtein_counts(phrase_tokens, win)
            err = (c["sub"] + c["del"] + c["ins"]) / n
            if err < best:
                best, best_i = err, i
    return CommandMatch("", best <= tolerance, float(best), best_i)


def command_metrics(hyp_text: str,
                    wearer_commands: Sequence[str],
                    bystander_commands: Sequence[str],
                    tolerance: float = 0.34) -> Dict[str, object]:
    """Core P1.8 metric block.

    `wearer_commands`   -- commands the WEARER actually issued (should survive)
    `bystander_commands`-- command-shaped phrases the BYSTANDER said
                           (must NOT survive)
    """
    hyp = tokenize(hyp_text)

    retained, retained_detail = 0, []
    for c in wearer_commands:
        m = fuzzy_contains(hyp, tokenize(c), tolerance)
        retained += int(m.present)
        retained_detail.append({"command": c, "retained": m.present, "error": m.best_error})

    injected, injected_detail = 0, []
    for c in bystander_commands:
        m = fuzzy_contains(hyp, tokenize(c), tolerance)
        injected += int(m.present)
        injected_detail.append({"command": c, "injected": m.present, "error": m.best_error})

    wake_in_hyp = fuzzy_contains(hyp, tokenize(WAKE_PHRASE), tolerance)
    wearer_said_wake = any(WAKE_PHRASE in c.lower() for c in wearer_commands)
    bystander_said_wake = any(WAKE_PHRASE in c.lower() for c in bystander_commands)

    n_w, n_b = len(wearer_commands), len(bystander_commands)
    return {
        "n_wearer_commands": n_w,
        "n_wearer_commands_retained": retained,
        "wearer_command_retention": (retained / n_w) if n_w else None,

        "n_bystander_commands": n_b,
        "n_false_agent_commands": injected,
        "false_agent_command_rate": (injected / n_b) if n_b else None,

        "wake_phrase_in_hypothesis": wake_in_hyp.present,
        "wake_phrase_retained": bool(wake_in_hyp.present and wearer_said_wake),
        "wake_phrase_falsely_injected": bool(
            wake_in_hyp.present and not wearer_said_wake and bystander_said_wake),

        "wearer_command_detail": retained_detail,
        "bystander_command_detail": injected_detail,
        "match_tolerance": tolerance,
    }


def aggregate_command_metrics(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Micro-averaged over commands, not macro-averaged over scenarios."""
    def s(k):
        return int(sum(int(r.get(k, 0) or 0) for r in rows))

    n_w, n_b = s("n_wearer_commands"), s("n_bystander_commands")
    out: Dict[str, object] = {
        "n_items": len(rows),
        "n_wearer_commands": n_w,
        "n_wearer_commands_retained": s("n_wearer_commands_retained"),
        "n_bystander_commands": n_b,
        "n_false_agent_commands": s("n_false_agent_commands"),
        "n_wake_phrase_falsely_injected": s("wake_phrase_falsely_injected"),
    }
    if n_w:
        out["wearer_command_retention"] = out["n_wearer_commands_retained"] / n_w
    if n_b:
        out["false_agent_command_rate"] = out["n_false_agent_commands"] / n_b
        out["wake_phrase_false_injection_rate"] = out["n_wake_phrase_falsely_injected"] / n_b
    return out
