#!/usr/bin/env python3
"""Queue/jitter-buffer tests (sprint spec section 22: buffer tests, sequence tests)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from mentra.audio.frame import AudioFrame, FrameFlags
from mentra.audio.jitter_buffer import JitterBuffer
from mentra.audio.queue import BoundedAudioQueue

PASS = []
FAIL = []


def check(name, condition):
    (PASS if condition else FAIL).append(name)
    print(f"{'PASS' if condition else 'FAIL'}: {name}")


def make_frame(seq: int, n_samples: int = 160) -> AudioFrame:  # 160 samples = 10ms @16kHz
    return AudioFrame(seq, time.monotonic_ns(), 16000, 1, 16, payload=b"\x00\x01" * n_samples)


def test_queue_never_blocks_on_overflow():
    q = BoundedAudioQueue(max_frames=3)
    for i in range(5):
        q.push(make_frame(i))
    check("queue_overflow: depth capped at max_frames", q.depth() == 3)
    check("queue_overflow: dropped count correct", q.stats.frames_dropped_overflow == 2)
    frame = q.pop(timeout=0)
    check("queue_overflow: oldest frame was dropped, frame 2 is now oldest", frame.sequence_number == 2)


def test_queue_pop_timeout():
    q = BoundedAudioQueue(max_frames=3)
    t0 = time.monotonic()
    result = q.pop(timeout=0.05)
    elapsed = time.monotonic() - t0
    check("queue_pop_timeout: returns None on empty", result is None)
    check("queue_pop_timeout: respects timeout duration", 0.04 <= elapsed <= 0.2)


def test_jitter_buffer_normal_sequence():
    # target_ms=0: no smoothing cushion held back, isolates ordering
    # correctness from the (separately-tested) latency-cushion behavior.
    jb = JitterBuffer(target_ms=0, max_ms=100, sample_rate=16000)
    for i in range(5):
        jb.push(make_frame(i))
    released = jb.release_ready()
    check("jitter_normal: releases in order", [f.sequence_number for f in released] == list(range(5)))
    check("jitter_normal: no missing/duplicate/out-of-order flagged",
          jb.stats.missing_sequence_numbers == 0 and jb.stats.duplicates == 0)


def test_jitter_buffer_holds_back_cushion():
    # With target_ms=30 and only 50ms buffered, the buffer should hold back
    # roughly target_ms worth as a smoothing cushion rather than draining
    # to empty -- that's the whole point of a jitter buffer.
    jb = JitterBuffer(target_ms=30, max_ms=100, sample_rate=16000)
    for i in range(5):  # 5 * 10ms = 50ms pushed
        jb.push(make_frame(i))
    released = jb.release_ready()
    check("jitter_cushion: releases some but not all frames", 0 < len(released) < 5)
    check("jitter_cushion: released frames still in correct order",
          [f.sequence_number for f in released] == list(range(len(released))))
    check("jitter_cushion: remaining buffered ms is close to target_ms",
          jb.depth_ms() <= 30)


def test_jitter_buffer_duplicate():
    jb = JitterBuffer(target_ms=30, max_ms=100, sample_rate=16000)
    jb.push(make_frame(0))
    jb.push(make_frame(0))  # duplicate
    jb.push(make_frame(1))
    check("jitter_duplicate: detected", jb.stats.duplicates == 1)


def test_jitter_buffer_out_of_order():
    jb = JitterBuffer(target_ms=0, max_ms=100, sample_rate=16000)
    jb.push(make_frame(0))
    jb.push(make_frame(2))  # arrives before 1
    jb.push(make_frame(1))
    released = jb.release_ready()
    check("jitter_out_of_order: still releases in correct sequence order",
          [f.sequence_number for f in released] == [0, 1, 2])


def test_jitter_buffer_missing_forces_release_at_max_ms():
    jb = JitterBuffer(target_ms=20, max_ms=50, sample_rate=16000)
    # seq 0 present, seq 1 MISSING, seq 2..7 present -- enough buffered ms to force past the gap
    jb.push(make_frame(0))
    for i in range(2, 8):
        jb.push(make_frame(i))
    released = jb.release_ready()
    check("jitter_missing: forces release past the gap under buffer pressure", len(released) > 1)
    check("jitter_missing: missing_sequence_numbers counted", jb.stats.missing_sequence_numbers >= 1)
    discontinuity_flagged = any(f.flags & FrameFlags.DISCONTINUITY for f in released)
    check("jitter_missing: discontinuity flag set on the frame after the gap", discontinuity_flagged)


if __name__ == "__main__":
    test_queue_never_blocks_on_overflow()
    test_queue_pop_timeout()
    test_jitter_buffer_normal_sequence()
    test_jitter_buffer_holds_back_cushion()
    test_jitter_buffer_duplicate()
    test_jitter_buffer_out_of_order()
    test_jitter_buffer_missing_forces_release_at_max_ms()

    print()
    print(f"PASS: {len(PASS)}  FAIL: {len(FAIL)}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
