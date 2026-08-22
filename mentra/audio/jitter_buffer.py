"""Sequence-aware jitter buffer (sprint spec section 12). Bounded -- never
becomes a large latency-producing queue. Tracks out-of-order, late,
missing, and duplicate frames; marks discontinuities rather than silently
concatenating audio across a gap."""
from __future__ import annotations

from dataclasses import dataclass, field

from mentra.audio.frame import AudioFrame, FrameFlags


@dataclass
class JitterBufferStats:
    frames_released: int = 0
    out_of_order: int = 0
    duplicates: int = 0
    missing_sequence_numbers: int = 0
    late_dropped: int = 0


class JitterBuffer:
    def __init__(self, target_ms: float, max_ms: float, sample_rate: int):
        if max_ms < target_ms:
            raise ValueError("max_ms must be >= target_ms")
        self.target_ms = target_ms
        self.max_ms = max_ms
        self.sample_rate = sample_rate
        self._pending: dict[int, AudioFrame] = {}
        self._next_expected_seq: int | None = None
        self._seen_seqs: set[int] = set()
        self.stats = JitterBufferStats()

    def _buffered_ms(self) -> float:
        total_samples = sum(f.sample_count for f in self._pending.values())
        return (total_samples / self.sample_rate) * 1000 if self.sample_rate else 0.0

    def push(self, frame: AudioFrame) -> None:
        seq = frame.sequence_number

        if seq in self._seen_seqs:
            self.stats.duplicates += 1
            return

        if self._next_expected_seq is not None and seq < self._next_expected_seq:
            self.stats.late_dropped += 1
            return

        self._seen_seqs.add(seq)
        self._pending[seq] = frame

        if self._next_expected_seq is None:
            self._next_expected_seq = seq

        # if this frame filled a gap, the frames after it may now be out-of-order
        # relative to arrival time but that's expected -- release() handles ordering
        if seq != self._next_expected_seq and seq > self._next_expected_seq:
            self.stats.out_of_order += 1

    def release_ready(self) -> list[AudioFrame]:
        """Returns frames ready for consumption, in sequence order. A frame
        is released once buffered_ms reaches target_ms, OR immediately if
        buffered_ms exceeds max_ms (bounded latency takes priority over
        strict ordering when the buffer is under pressure)."""
        released: list[AudioFrame] = []
        buffered_ms = self._buffered_ms()
        force_release = buffered_ms >= self.max_ms

        while self._pending and (buffered_ms >= self.target_ms or force_release):
            if self._next_expected_seq in self._pending:
                frame = self._pending.pop(self._next_expected_seq)
                released.append(frame)
                self._next_expected_seq += 1
            elif force_release:
                # gap we're not willing to wait out any longer -- mark
                # discontinuity and jump to the earliest frame we actually have
                self.stats.missing_sequence_numbers += 1
                earliest_seq = min(self._pending.keys())
                frame = self._pending.pop(earliest_seq)
                frame.flags |= FrameFlags.DISCONTINUITY
                released.append(frame)
                self._next_expected_seq = earliest_seq + 1
            else:
                break  # waiting for the missing frame, buffer not under pressure yet
            buffered_ms = self._buffered_ms()

        self.stats.frames_released += len(released)
        return released

    def depth_ms(self) -> float:
        return self._buffered_ms()

    def flush(self) -> list[AudioFrame]:
        """Force-releases everything still buffered, regardless of
        target_ms/max_ms, in sequence order (gaps marked as discontinuities
        same as release_ready()). Call this when a stream ends -- otherwise
        the last target_ms worth of frames never gets released, since
        release_ready() only fires reactively when a new frame arrives."""
        released: list[AudioFrame] = []
        while self._pending:
            if self._next_expected_seq in self._pending:
                frame = self._pending.pop(self._next_expected_seq)
                released.append(frame)
                self._next_expected_seq += 1
            else:
                self.stats.missing_sequence_numbers += 1
                earliest_seq = min(self._pending.keys())
                frame = self._pending.pop(earliest_seq)
                frame.flags |= FrameFlags.DISCONTINUITY
                released.append(frame)
                self._next_expected_seq = earliest_seq + 1
        self.stats.frames_released += len(released)
        return released
