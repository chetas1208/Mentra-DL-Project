"""G4 WS6 -- automatic validation of a research capture session.

A pilot recording session is expensive: someone has to physically wear the
glasses and talk. Discovering afterwards that the file was silent, clipped,
resampled by the browser, or written at the wrong rate wastes the whole
session. This module runs every cheap check that can catch that, immediately,
so a bad take is caught while the wearer is still in the room.

Checks are graded PASS / WARN / FAIL:
  FAIL  the capture is unusable as evidence (unreadable, empty, wrong rate,
        wrong channel count, effectively silent, or grossly clipped)
  WARN  usable but worth knowing about (short, quiet, mildly clipped, little
        detected speech activity)
  PASS  nothing to flag

Nothing here is Mentra-specific: it validates whatever audio it was given.
The metadata layer is where a capture declares what device it came from, and
``mentra.capture.session`` refuses to label a capture as Mentra audio unless
the operator explicitly declared a Mentra source.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Thresholds. Deliberately loose: this is a "did the recording chain work"
# check, not an audio-quality opinion.
MIN_DURATION_S = 1.0
SHORT_DURATION_S = 5.0
SILENT_RMS = 1e-4          # below this the channel is dead, not quiet
QUIET_RMS = 3e-3
CLIP_THRESHOLD = 0.999
CLIP_FAIL_RATE = 0.01      # >1% of samples pinned at full scale
CLIP_WARN_RATE = 0.001
LOW_ACTIVITY_RATE = 0.05   # <5% of frames above the speech-activity floor
ACTIVITY_FRAME_MS = 20.0
ACTIVITY_MARGIN_DB = 12.0  # a frame counts as active this far above the noise floor


@dataclasses.dataclass
class Check:
    name: str
    status: str            # "PASS" | "WARN" | "FAIL"
    detail: str
    value: Optional[float] = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def speech_activity_rate(audio: np.ndarray, sample_rate: int) -> float:
    """Fraction of frames whose energy sits well above the recording's own
    noise floor.

    Deliberately NOT a VAD model: this is a recording-chain sanity check, and
    a model here would add a dependency and a failure mode for no gain. The
    floor is the 10th percentile frame energy of the recording itself, so it
    adapts to a noisy shop floor instead of assuming a quiet room.
    """
    hop = max(1, int(round(sample_rate * ACTIVITY_FRAME_MS / 1000.0)))
    n = audio.size // hop
    if n < 4:
        return 0.0
    frames = audio[:n * hop].reshape(n, hop)
    energy = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)) + 1e-12
    floor = float(np.percentile(energy, 10))
    threshold = floor * (10.0 ** (ACTIVITY_MARGIN_DB / 20.0))
    return float(np.mean(energy > threshold))


def validate_audio(audio: np.ndarray, sample_rate: int, channels: int,
                   *, expected_sample_rate: Optional[int] = None,
                   expected_channels: int = 1, label: str = "audio") -> List[Check]:
    checks: List[Check] = []
    x = np.asarray(audio, dtype=np.float32).reshape(-1)

    if not np.isfinite(x).all():
        checks.append(Check(f"{label}.finite", "FAIL",
                            "contains NaN or Inf samples"))
        x = np.nan_to_num(x)
    else:
        checks.append(Check(f"{label}.finite", "PASS", "all samples finite"))

    if sample_rate <= 0:
        checks.append(Check(f"{label}.sample_rate", "FAIL",
                            f"non-positive sample rate {sample_rate}", float(sample_rate)))
    elif expected_sample_rate is not None and sample_rate != expected_sample_rate:
        checks.append(Check(f"{label}.sample_rate", "FAIL",
                            f"expected {expected_sample_rate} Hz, got {sample_rate} Hz",
                            float(sample_rate)))
    else:
        checks.append(Check(f"{label}.sample_rate", "PASS", f"{sample_rate} Hz",
                            float(sample_rate)))

    if channels != expected_channels:
        checks.append(Check(f"{label}.channels", "FAIL",
                            f"expected {expected_channels} channel(s), got {channels}",
                            float(channels)))
    else:
        checks.append(Check(f"{label}.channels", "PASS", f"{channels} channel(s)",
                            float(channels)))

    duration = x.size / float(sample_rate) if sample_rate > 0 else 0.0
    if duration < MIN_DURATION_S:
        checks.append(Check(f"{label}.duration_s", "FAIL",
                            f"{duration:.2f} s is below the {MIN_DURATION_S} s minimum", duration))
    elif duration < SHORT_DURATION_S:
        checks.append(Check(f"{label}.duration_s", "WARN",
                            f"{duration:.2f} s is short for a pilot take", duration))
    else:
        checks.append(Check(f"{label}.duration_s", "PASS", f"{duration:.2f} s", duration))

    rms = _rms(x)
    if rms < SILENT_RMS:
        checks.append(Check(f"{label}.rms", "FAIL",
                            f"RMS {rms:.2e} -- the channel is effectively silent", rms))
    elif rms < QUIET_RMS:
        checks.append(Check(f"{label}.rms", "WARN",
                            f"RMS {rms:.2e} is very quiet; check mic gain/placement", rms))
    else:
        checks.append(Check(f"{label}.rms", "PASS", f"RMS {rms:.4f}", rms))

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    checks.append(Check(f"{label}.peak", "PASS", f"peak {peak:.4f}", peak))

    clip_rate = float(np.mean(np.abs(x) >= CLIP_THRESHOLD)) if x.size else 0.0
    if clip_rate > CLIP_FAIL_RATE:
        checks.append(Check(f"{label}.clipping", "FAIL",
                            f"{clip_rate * 100:.2f}% of samples at full scale", clip_rate))
    elif clip_rate > CLIP_WARN_RATE:
        checks.append(Check(f"{label}.clipping", "WARN",
                            f"{clip_rate * 100:.3f}% of samples at full scale", clip_rate))
    else:
        checks.append(Check(f"{label}.clipping", "PASS",
                            f"{clip_rate * 100:.3f}% of samples at full scale", clip_rate))

    activity = speech_activity_rate(x, sample_rate) if sample_rate > 0 else 0.0
    if activity < LOW_ACTIVITY_RATE:
        checks.append(Check(f"{label}.speech_activity", "WARN",
                            f"only {activity * 100:.1f}% of frames rise above the noise floor; "
                            "the take may contain no speech", activity))
    else:
        checks.append(Check(f"{label}.speech_activity", "PASS",
                            f"{activity * 100:.1f}% of frames above the noise floor", activity))

    return checks


def validate_file(path: Path, *, expected_sample_rate: Optional[int] = None,
                  expected_channels: int = 1, label: Optional[str] = None) -> List[Check]:
    """Read-back check: a file that cannot be reopened and decoded is a FAIL
    no matter how good the in-memory buffer looked."""
    import soundfile as sf

    label = label or path.stem
    if not path.is_file():
        return [Check(f"{label}.readable", "FAIL", f"missing file {path}")]
    try:
        data, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception as exc:  # noqa: BLE001 -- any decode failure is a FAIL
        return [Check(f"{label}.readable", "FAIL", f"unreadable: {exc}")]

    checks = [Check(f"{label}.readable", "PASS",
                    f"{path.name} decoded, {data.shape[0]} frames x {data.shape[1]} ch")]
    checks.extend(validate_audio(data[:, 0], sample_rate, data.shape[1],
                                 expected_sample_rate=expected_sample_rate,
                                 expected_channels=expected_channels, label=label))
    return checks


def overall_status(checks: List[Check]) -> str:
    statuses = {c.status for c in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def summarise(checks: List[Check]) -> Dict[str, object]:
    return {
        "status": overall_status(checks),
        "n_checks": len(checks),
        "n_fail": sum(1 for c in checks if c.status == "FAIL"),
        "n_warn": sum(1 for c in checks if c.status == "WARN"),
        "failures": [c.to_json() for c in checks if c.status == "FAIL"],
        "warnings": [c.to_json() for c in checks if c.status == "WARN"],
        "checks": [c.to_json() for c in checks],
    }


def check_rate_consistency(raw_duration_s: float, derived_duration_s: float,
                           tolerance_s: float = 0.25) -> Check:
    """The derived 16 kHz stream and the native-rate raw stream must describe
    the same span of time. A mismatch means the resampler drifted or one of
    the two streams dropped frames -- the single most likely silent failure in
    a browser capture chain, and invisible in either file on its own."""
    delta = abs(raw_duration_s - derived_duration_s)
    if not math.isfinite(delta):
        return Check("streams.duration_match", "FAIL", "non-finite duration")
    if delta > max(tolerance_s, 0.02 * max(raw_duration_s, derived_duration_s)):
        return Check("streams.duration_match", "FAIL",
                     f"raw {raw_duration_s:.2f} s vs derived {derived_duration_s:.2f} s "
                     f"differ by {delta:.2f} s", delta)
    return Check("streams.duration_match", "PASS",
                 f"raw {raw_duration_s:.2f} s vs derived {derived_duration_s:.2f} s", delta)
