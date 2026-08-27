"""Shared probability sources for the G4 live-vs-offline measurements.

The offline product evaluator runs GeoWearNet over a whole recording and gets
one probability pair per 10 ms frame. The live receiver re-scores a rolling
``context_s`` window every ``hop_s`` and reads only the final frame's logit,
then holds that value until the next hop. Those are different signals, and
every G4 number states which one it used.

``live_cadence`` builds the live signal from the dense one by subsampling at
the hop and zero-order-holding. That shortcut is only legitimate because the
model is a causal TCN whose 1000 ms receptive field fits inside the live
2000 ms rolling window, so the final frame of a rolling window equals the
same frame of a full-recording pass. G3 measured that streaming parity at a
max wearer-logit difference of 4.68e-06; ``verify_rolling_window_parity``
re-measures it here rather than trusting the earlier number.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

SR = 16000
FRAME_MS = 10.0


def live_cadence(p_wearer: np.ndarray, p_env: np.ndarray, hop_ms: float = 200.0,
                 context_ms: float = 2000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Dense per-frame probabilities -> the live receiver's held signal.

    The receiver emits nothing until it has ``hop_ms`` of audio, and the first
    decision it can make lands on the frame at ``hop_ms``. Before that the
    envelope sees the router's initial (0, 0) SILENCE probabilities, which is
    exactly what the live path does at the start of a stream -- so the leading
    frames are filled with zeros rather than with a probability the live
    receiver could not have known yet.
    """
    pw = np.asarray(p_wearer, dtype=np.float32)
    pe = np.asarray(p_env, dtype=np.float32)
    n = int(min(len(pw), len(pe)))
    hop_frames = max(1, int(round(hop_ms / FRAME_MS)))
    out_w = np.zeros(n, dtype=np.float32)
    out_e = np.zeros(n, dtype=np.float32)
    # decisions are published at frame indices hop-1, 2*hop-1, ... (the last
    # frame of each completed hop) and apply from the following frame onward
    for start in range(hop_frames, n + hop_frames, hop_frames):
        decision_frame = min(start - 1, n - 1)
        end = min(start + hop_frames, n)
        if start >= n:
            break
        out_w[start:end] = pw[decision_frame]
        out_e[start:end] = pe[decision_frame]
    del context_ms
    return out_w, out_e


def verify_rolling_window_parity(predictor, audio: np.ndarray, dense_w: np.ndarray,
                                 dense_e: np.ndarray, context_s: float = 2.0,
                                 hop_s: float = 0.2, n_probes: int = 8) -> Dict[str, float]:
    """Measure, not assume, that a rolling-window score equals the dense score.

    Runs the real detector on real rolling windows at ``n_probes`` positions
    and compares the final frame's probability against the corresponding dense
    frame. A large deviation here would invalidate ``live_cadence``.
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    ctx = int(round(context_s * SR))
    hop = int(round(hop_s * SR))
    n_frames = int(min(len(dense_w), len(dense_e)))
    total_hops = max(1, (len(audio) - ctx) // hop)
    probe_hops = np.unique(np.linspace(1, total_hops, num=min(n_probes, total_hops)).astype(int))

    dw, de = [], []
    for k in probe_hops:
        end = ctx + int(k) * hop
        if end > len(audio):
            break
        window = audio[end - ctx:end]
        wpw, wpe = predictor.predict(window)
        if len(wpw) == 0:
            continue
        # Align by frame COUNT, not by sample arithmetic: the feature
        # extractor's analysis window means the last mel frame of a window
        # does not end at the window's last sample. The window starts at a
        # whole-frame boundary, so its frame j is dense frame
        # (window_start/hop) + j.
        hop_samples = int(round(SR * FRAME_MS / 1000.0))
        window_start_frame = (end - ctx) // hop_samples
        frame_index = window_start_frame + len(wpw) - 1
        if frame_index >= n_frames:
            continue
        dw.append(abs(float(wpw[-1]) - float(dense_w[frame_index])))
        de.append(abs(float(wpe[-1]) - float(dense_e[frame_index])))
    if not dw:
        return {"n_probes": 0}
    return {
        "n_probes": len(dw),
        "max_abs_wearer_prob_diff": float(np.max(dw)),
        "mean_abs_wearer_prob_diff": float(np.mean(dw)),
        "max_abs_env_prob_diff": float(np.max(de)),
        "mean_abs_env_prob_diff": float(np.mean(de)),
    }


def shift_gain_for_preroll(gain: np.ndarray, preroll_ms: float) -> np.ndarray:
    """Offline equivalent of the router's pre-roll delay line.

    The router emits audio ``preroll_ms`` behind the gate, so audio frame *i*
    is multiplied by the gain decided at frame ``i + preroll_frames``. Keeping
    the audio in place and shifting the gain left is exactly the same
    operation and is asserted equal to the live router in
    ``tests/audio/test_streaming_gate.py``.
    """
    g = np.asarray(gain, dtype=np.float32)
    d = int(round(preroll_ms / FRAME_MS))
    if d <= 0:
        return g
    if d >= len(g):
        return np.full_like(g, g[-1] if len(g) else 0.0)
    return np.concatenate([g[d:], np.full(d, g[-1], dtype=np.float32)])
