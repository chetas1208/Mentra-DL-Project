"""Bounded producer/consumer audio queue (sprint spec section 13). Real-time
priority: fresh audio beats old audio -- overflow drops the OLDEST queued
frame, never blocks the producer (the Bluetooth/capture callback must never
stall on a full queue)."""
from __future__ import annotations

import collections
import threading
from dataclasses import dataclass

from mentra.audio.frame import AudioFrame


@dataclass
class QueueStats:
    frames_pushed: int = 0
    frames_popped: int = 0
    frames_dropped_overflow: int = 0


class BoundedAudioQueue:
    def __init__(self, max_frames: int):
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self.max_frames = max_frames
        self._deque: collections.deque[AudioFrame] = collections.deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self.stats = QueueStats()

    def push(self, frame: AudioFrame) -> None:
        """Never blocks. If full, drops the oldest frame to make room --
        real-time audio prioritizes freshness over completeness."""
        with self._not_empty:
            if len(self._deque) >= self.max_frames:
                self._deque.popleft()
                self.stats.frames_dropped_overflow += 1
            self._deque.append(frame)
            self.stats.frames_pushed += 1
            self._not_empty.notify()

    def pop(self, timeout: float | None = None) -> AudioFrame | None:
        """Blocks up to `timeout` seconds waiting for a frame. Returns None on timeout."""
        with self._not_empty:
            if not self._deque:
                if not self._not_empty.wait(timeout=timeout):
                    return None
            if not self._deque:
                return None
            frame = self._deque.popleft()
            self.stats.frames_popped += 1
            return frame

    def depth(self) -> int:
        with self._lock:
            return len(self._deque)

    def depth_ms(self, sample_rate: int) -> float:
        with self._lock:
            total_samples = sum(f.sample_count for f in self._deque)
        return (total_samples / sample_rate) * 1000 if sample_rate else 0.0
