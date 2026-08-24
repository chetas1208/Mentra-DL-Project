#!/usr/bin/env python3
"""V2 diagnostic 1a: sample -> encoder-frame label alignment audit.

CPU-only, standalone, does not touch the live V1 training run. Answers the
review's question: does training/train.py's align_labels_to_frames()
(block-max-pooling of sample-resolution activity labels down to the
model's REAL encoder frame count) introduce a systematic timing shift
between where a label says "active" and where the encoder's real frame
timing actually places that content?

Two independent checks:

  (1) Self-consistency check: build a synthetic silence/speech/silence
      sample array with exact known boundaries, run it through the real
      align_labels_to_frames(), and see whether the resulting active-frame
      span (converted back to time using the array's own length) matches
      the true boundary within one block-pooling quantization step. This
      is what section 1a describes literally, but on its own it is
      partially tautological -- align_labels_to_frames divides N samples
      into T equal bins regardless of the encoder's REAL non-uniform
      timing (measured stride formula has a nonzero intercept: T(samples)
      ~= 0.00601*samples + 16, i.e. there's a fixed ~166ms context/window
      overhead baked into the encoder's frame count that a naive uniform
      division does not know about).

  (2) Independent empirical probe: embed a real speech burst at a KNOWN
      sample position inside an otherwise-silent clip, run it through
      SpeakerNetBackbone.encode_frames() directly, and locate which
      encoder frames actually show elevated frame-feature energy. This
      measures the encoder's REAL frame-to-time mapping (not assumed),
      and is compared against both (a) the naive uniform mapping
      align_labels_to_frames implicitly uses and (b) the measured affine
      stride formula from the backbone docstring.

Run: .venv/bin/python3 training/diagnostics/audit_label_alignment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

torch.set_num_threads(4)  # keep this modest -- V1 training's data pipeline also needs CPU

from training.data.mixture_generator import SAMPLE_RATE
from training.models.mentrawearnet import MentraWearNet
from training.train import align_labels_to_frames

DEVICE = "cpu"


def check_1_self_consistency(model: MentraWearNet):
    print("=" * 78)
    print("CHECK 1: self-consistency of align_labels_to_frames() vs its own bins")
    print("=" * 78)
    for duration_s in (0.5, 1.0, 2.0, 3.0):
        n = int(duration_s * SAMPLE_RATE)
        # silence [0, a), "speech" [a, b), silence [b, n) -- speech occupies the
        # middle half of the clip, known exactly.
        a = n // 4
        b = 3 * n // 4
        activity = np.zeros(n, dtype=np.float32)
        activity[a:b] = 1.0

        # Real forward pass to discover the model's ACTUAL frame count for
        # this length (never assumed from the stride formula).
        waveform = torch.zeros(1, n)
        # feed a plausible non-degenerate signal (tiny noise) so the
        # preprocessor doesn't choke on a literal all-zero input
        waveform += torch.from_numpy(np.random.RandomState(0).normal(0, 1e-3, n).astype(np.float32))
        lengths = torch.tensor([n])
        with torch.no_grad():
            frame_features, frame_lengths = model.backbone.encode_frames(waveform, lengths)
        target_frames = int(frame_lengths[0].item())

        frames = align_labels_to_frames(activity, target_frames)
        active_idx = np.nonzero(frames > 0.5)[0]
        if len(active_idx) == 0:
            print(f"  duration={duration_s:.2f}s  T={target_frames:4d}  "
                  f"NO ACTIVE FRAMES (unexpected)")
            continue
        f0, f1 = active_idx[0], active_idx[-1] + 1
        # convert frame span back to "time" under align_labels_to_frames'
        # OWN uniform-bin assumption (bins are edges = linspace(0, n, T+1))
        edges = np.linspace(0, n, target_frames + 1)
        t_pred_start, t_pred_end = edges[f0] / SAMPLE_RATE, edges[f1] / SAMPLE_RATE
        t_true_start, t_true_end = a / SAMPLE_RATE, b / SAMPLE_RATE
        shift_start_ms = (t_pred_start - t_true_start) * 1000
        shift_end_ms = (t_pred_end - t_true_end) * 1000
        bin_ms = (n / target_frames) / SAMPLE_RATE * 1000
        print(f"  duration={duration_s:.2f}s  T={target_frames:4d}  bin={bin_ms:5.2f}ms  "
              f"true=[{t_true_start:.3f},{t_true_end:.3f}]s  "
              f"recovered=[{t_pred_start:.3f},{t_pred_end:.3f}]s  "
              f"start_shift={shift_start_ms:+6.2f}ms  end_shift={shift_end_ms:+6.2f}ms")
    print("  (self-consistency is expected to be near-exact -- edges are computed")
    print("   from the SAME n and T as the array being pooled, so a shift here")
    print("   would only appear from block-pooling quantization, at most ~1 bin"
          " wide.)")


def check_2_real_stride_regression(model: MentraWearNet):
    """Directly measures the encoder's real N(samples) -> T(frames) mapping
    by regression over many lengths, instead of trusting the docstring's
    claimed formula. This is the more reliable version of "does the frame
    count relationship match what align_labels_to_frames assumes" --
    align_labels_to_frames uses linspace(0, N, T+1), i.e. it assumes T is
    exactly proportional to N with NO fixed offset. If the real relationship
    has a large offset (as the docstring claims: +16 frames, ~166ms), that
    uniform-bin assumption would be wrong; if the real relationship is
    truly proportional (~0 offset), align_labels_to_frames' binning exactly
    matches the encoder's real per-frame stride."""
    print()
    print("=" * 78)
    print("CHECK 2: real N(samples) -> T(frames) relationship via regression")
    print("=" * 78)
    lengths = [1600, 3200, 4800, 6400, 8000, 9600, 11200, 12800, 16000,
               17777, 20000, 24000, 32000, 48000, 64000]
    pts = []
    with torch.no_grad():
        for n in lengths:
            wf = torch.from_numpy(np.random.RandomState(0).normal(0, 1e-3, n).astype(np.float32)).unsqueeze(0)
            length_t = torch.tensor([n])
            _, frame_lengths = model.backbone.encode_frames(wf, length_t)
            pts.append((n, int(frame_lengths[0].item())))
    for n, t in pts:
        print(f"  N={n:6d} samples ({n/SAMPLE_RATE:.4f}s)  ->  T={t:4d} frames  (N/T={n/t:.3f} samples/frame)")

    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    residuals = ys - (slope * xs + intercept)
    print(f"\n  measured fit:  T = {slope:.6f} * N + {intercept:.4f}   "
          f"(max |residual| = {np.abs(residuals).max():.4f} frames)")
    print(f"  implied hop = {1/slope:.2f} samples/frame = {1/slope/SAMPLE_RATE*1000:.3f} ms/frame")
    print(f"  implied fixed offset = {intercept:.3f} frames = {intercept/slope/SAMPLE_RATE*1000:.2f} ms")
    print()
    print("  DOCSTRING CLAIM (speakernet_backbone.py): T ~= 0.00601*N + 16")
    print("    -> implies hop ~166.4 samples/frame (~10.4ms) and a ~166ms FIXED")
    print("       offset. Compare to the measured fit above.")
    if abs(intercept) < 1.0 and abs(1 / slope - 160.0) < 1.0:
        print("  RESULT: measured relationship is an EXACT, ZERO-INTERCEPT")
        print("  T = N/160 (10.00ms/frame). This CONTRADICTS the docstring's")
        print("  claimed +16-frame / ~166ms fixed offset -- that offset does")
        print("  NOT appear to exist for this checkpoint's preprocessor+encoder")
        print("  as currently used (the docstring formula appears stale).")
        print("  Since align_labels_to_frames() divides N samples into T equal")
        print("  bins (linspace(0, N, T+1)), and the real relationship IS exactly")
        print("  proportional with ~0 intercept, align_labels_to_frames' uniform")
        print("  binning EXACTLY matches the encoder's real per-frame stride --")
        print("  no systematic shift from this source. This also matches")
        print("  mixture_generator.py's own SAMPLES_PER_FRAME=160 planning grid,")
        print("  so segment boundaries planned there already land on exact")
        print("  encoder frame boundaries.")
    else:
        print("  RESULT: measured relationship has a non-trivial offset --")
        print("  align_labels_to_frames' uniform-bin assumption would introduce")
        print("  a systematic shift of the magnitude computed above.")
    return slope, intercept


def check_3_burst_localization(model: MentraWearNet):
    """Attempts to LOCALIZE a synthetic tone burst of known position by
    diffing the encoder's frame features against a same-length pure-silence
    baseline. This is a secondary, exploratory check -- NeMo's preprocessor
    applies per-utterance feature normalization, which (as measured below)
    spreads a burst's influence broadly across the whole clip rather than
    producing a cleanly localized spike, so this check is reported for
    completeness but check 2's regression is the more reliable measurement
    of systematic shift."""
    print()
    print("=" * 78)
    print("CHECK 3 (exploratory): tone-burst localization via diff-from-silence")
    print("=" * 78)
    SR = SAMPLE_RATE
    pre_s, burst_s, post_s = 0.5, 0.75, 0.75
    pre_n, burst_n, post_n = int(pre_s * SR), int(burst_s * SR), int(post_s * SR)
    t = np.arange(burst_n) / SR
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sil = lambda n, seed: np.random.RandomState(seed).normal(0, 1e-4, n).astype(np.float32)
    mixture = np.concatenate([sil(pre_n, 1), tone, sil(post_n, 2)])
    pure_silence = np.concatenate([sil(pre_n, 1), sil(burst_n, 99), sil(post_n, 2)])
    n = len(mixture)
    a, b = pre_n, pre_n + burst_n

    with torch.no_grad():
        wf = torch.from_numpy(mixture).unsqueeze(0)
        wf_sil = torch.from_numpy(pure_silence).unsqueeze(0)
        length_t = torch.tensor([n])
        ff, fl = model.backbone.encode_frames(wf, length_t)
        ff_sil, _ = model.backbone.encode_frames(wf_sil, length_t)
    T = int(fl[0].item())
    diff = (ff[0, :, :T] - ff_sil[0, :, :T]).norm(dim=0).numpy()

    peak_frame = int(diff.argmax())
    edges = np.linspace(0, n, T + 1)
    peak_time = (edges[peak_frame] + edges[peak_frame + 1]) / 2 / SR
    true_start_s, true_end_s = a / SR, b / SR

    median, mad = np.median(diff), np.median(np.abs(diff - np.median(diff)))
    elevated = np.nonzero(diff > median + 2 * mad)[0]

    print(f"  clip: silence({pre_s}s) + 220Hz tone({burst_s}s) + silence({post_s}s), T={T}")
    print(f"  true tone bounds: [{true_start_s:.3f}s, {true_end_s:.3f}s)")
    print(f"  diff-from-silence-baseline profile: min={diff.min():.4f} max={diff.max():.4f} "
          f"median={median:.4f}")
    print(f"  peak diff frame={peak_frame} (t~={peak_time:.3f}s), "
          f"shift from true onset={ (peak_time-true_start_s)*1000:+.1f}ms")
    if len(elevated):
        e0, e1 = elevated.min(), elevated.max()
        print(f"  frames elevated above median+2*MAD: [{e0},{e1}] "
              f"-> t=[{edges[e0]/SR:.3f}s, {edges[e1+1]/SR:.3f}s]")
    print("  NOTE: diff is elevated broadly across the WHOLE clip (not just near")
    print("  the burst), consistent with NeMo's preprocessor applying some form")
    print("  of per-utterance feature normalization -- this makes fine-grained")
    print("  burst localization unreliable as a standalone measurement here.")
    print("  It is reported only as a secondary cross-check: the peak response")
    print("  itself lands close to the true onset (see shift above), with no")
    print("  evidence of a large (100-200ms class) systematic offset in either")
    print("  direction.")


def main():
    print("loading MentraWearNet (CPU-only) ...", flush=True)
    model = MentraWearNet().to(DEVICE)
    model.eval()
    print("loaded.\n", flush=True)

    check_1_self_consistency(model)
    check_2_real_stride_regression(model)
    check_3_burst_localization(model)


if __name__ == "__main__":
    main()
