"""G4 WS6 -- research capture session writer.

WHERE THE AUDIO COMES FROM (read this before assuming a Bluetooth stack)
-----------------------------------------------------------------------
This project already has a real, working, deployed capture chain:

    Mentra Live (or any input device) paired to the OPERATOR's own machine
      -> that machine's browser: getUserMedia + AudioWorklet
         (web/app/composables/useMicCapture.ts, native sample rate, AGC/NS/EC
          requested off and the GRANTED settings read back, never assumed)
      -> the binary MTRA WebSocket protocol (web/app/utils/mentraProtocol.ts)
      -> server/audio/remote_receiver.py on this host
      -> this module

No Bluetooth, serial, or adb access is needed on the server for that to work,
because pairing happens on the operator's machine, not here. What this module
adds is the research bookkeeping the live demo path never needed: a session
identity, an anonymous wearer id, a condition label, the device/format
provenance, both the native-rate and the derived 16 kHz streams, and
automatic validation.

HONESTY RULES ENFORCED IN CODE, NOT JUST IN DOCS
------------------------------------------------
* ``source_kind`` is declared by the operator and copied verbatim into the
  metadata. ``SYNTHETIC_TEST_INPUT_NOT_MENTRA_AUDIO`` is a first-class value
  used by the self-test, and any session whose source is not an explicit
  Mentra device is stamped ``is_mentra_hardware_audio: false``.
* ``assert_mentra_hardware`` exists so downstream evaluation code can refuse
  to treat a synthetic or browser-mic capture as Mentra pilot evidence.
* Raw capture stays local: the session directory is created under a
  gitignored capture root and nothing is uploaded anywhere.
"""
from __future__ import annotations

import dataclasses
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from mentra.capture import validation as V

DEFAULT_SAMPLE_RATE = 16000

#: Declared provenance of a capture. The value is written verbatim into the
#: metadata and decides whether the capture may ever be cited as Mentra
#: hardware evidence.
SOURCE_KINDS = (
    "MENTRA_LIVE_GLASSES_MIC",             # the real product device
    "MENTRA_COMPATIBLE_GLASSES_MIC",       # another MentraOS-compatible device
    "BROWSER_DEFAULT_MIC",                 # laptop/phone mic -- NOT Mentra evidence
    "SYNTHETIC_TEST_INPUT_NOT_MENTRA_AUDIO",  # corpus audio replayed through the chain
)
MENTRA_HARDWARE_SOURCES = ("MENTRA_LIVE_GLASSES_MIC", "MENTRA_COMPATIBLE_GLASSES_MIC")


def default_capture_root() -> Path:
    env = os.environ.get("MENTRA_CAPTURE_DIR")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[2] / "captures"


def anonymous_wearer_id() -> str:
    """A random per-person handle. Deliberately NOT derived from anything
    about the person: no name, no voice hash, no device id. Reusing the same
    handle across sessions is the operator's choice, recorded on paper, not
    something this code can reconstruct."""
    return f"W-{uuid.uuid4().hex[:8]}"


class CaptureMetadataError(ValueError):
    pass


@dataclasses.dataclass
class CaptureMetadata:
    """Everything needed to interpret a take months later."""
    session_id: str
    wearer_id: str
    condition: str
    source_kind: str
    started_utc: str
    raw_sample_rate: int
    raw_channels: int
    derived_sample_rate: int = DEFAULT_SAMPLE_RATE
    derived_channels: int = 1
    bits_per_sample: int = 16
    # Capture-chain provenance, read back from the browser rather than assumed
    echo_cancellation: Optional[bool] = None
    noise_suppression: Optional[bool] = None
    auto_gain_control: Optional[bool] = None
    device_label: Optional[str] = None
    device_id_hash: Optional[str] = None
    user_agent: Optional[str] = None
    platform: Optional[str] = None
    client_version: Optional[str] = None
    # Free-form operator notes and any device/firmware fields the operator
    # could read off the app. Unknown is recorded as null, never guessed.
    firmware_version: Optional[str] = None
    mentraos_version: Optional[str] = None
    glasses_model: Optional[str] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.source_kind not in SOURCE_KINDS:
            raise CaptureMetadataError(
                f"unknown source_kind {self.source_kind!r}; choose one of {SOURCE_KINDS}")
        if not self.session_id:
            raise CaptureMetadataError("session_id is required")
        if not self.wearer_id:
            raise CaptureMetadataError("wearer_id is required")
        if not self.condition:
            raise CaptureMetadataError(
                "condition is required -- an unlabelled take cannot be compared to anything")
        if self.raw_sample_rate <= 0:
            raise CaptureMetadataError(f"invalid raw_sample_rate {self.raw_sample_rate}")
        if self.raw_channels <= 0:
            raise CaptureMetadataError(f"invalid raw_channels {self.raw_channels}")

    @property
    def is_mentra_hardware_audio(self) -> bool:
        return self.source_kind in MENTRA_HARDWARE_SOURCES

    def to_json(self) -> dict:
        payload = dataclasses.asdict(self)
        payload["is_mentra_hardware_audio"] = self.is_mentra_hardware_audio
        return payload

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "CaptureMetadata":
        fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in payload.items() if k in fields}
        missing = {"session_id", "wearer_id", "condition", "source_kind",
                   "raw_sample_rate", "raw_channels"} - set(kwargs)
        if missing:
            raise CaptureMetadataError(f"capture metadata is missing {sorted(missing)}")
        kwargs.setdefault("started_utc", datetime.now(timezone.utc).isoformat())
        kwargs["raw_sample_rate"] = int(kwargs["raw_sample_rate"])
        kwargs["raw_channels"] = int(kwargs["raw_channels"])
        return cls(**kwargs)  # type: ignore[arg-type]


class CaptureSession:
    """Accumulates one take and writes it out with validation.

    Two streams are kept because they answer different questions:
      * ``raw``     the least-processed audio the application layer can see,
                    at the capture device's native rate. This is the artifact
                    a future domain-gap analysis needs.
      * ``derived`` the mono 16 kHz stream the model actually consumes, so a
                    detector result can always be reproduced exactly.
    """

    def __init__(self, metadata: CaptureMetadata, root: Optional[Path] = None):
        self.metadata = metadata
        self.root = Path(root) if root is not None else default_capture_root()
        self.dir = self.root / metadata.session_id
        self._raw: List[np.ndarray] = []
        self._derived: List[np.ndarray] = []
        self.closed = False
        self.result: Optional[Dict[str, object]] = None

    # -- ingestion -----------------------------------------------------------
    def add_raw_pcm16(self, payload: bytes, sample_rate: int, channels: int) -> None:
        if self.closed:
            raise RuntimeError("capture session already closed")
        if sample_rate != self.metadata.raw_sample_rate:
            raise ValueError(
                f"raw chunk sample rate {sample_rate} does not match the declared "
                f"{self.metadata.raw_sample_rate}; the capture chain changed mid-session")
        if channels != self.metadata.raw_channels:
            raise ValueError(
                f"raw chunk channel count {channels} does not match the declared "
                f"{self.metadata.raw_channels}")
        self._raw.append(np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0)

    def add_derived_pcm16(self, payload: bytes) -> None:
        if self.closed:
            raise RuntimeError("capture session already closed")
        self._derived.append(np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0)

    # -- output --------------------------------------------------------------
    @property
    def raw_duration_s(self) -> float:
        n = sum(a.size for a in self._raw)
        return n / float(self.metadata.raw_sample_rate * self.metadata.raw_channels)

    @property
    def derived_duration_s(self) -> float:
        return sum(a.size for a in self._derived) / float(self.metadata.derived_sample_rate)

    def close(self) -> Dict[str, object]:
        """Write raw.wav, derived_16k.wav, metadata.json and validation.json.

        Always writes something, even for a failed take: a FAIL result that
        exists on disk is far more useful than a session that silently
        vanished."""
        import soundfile as sf

        if self.closed and self.result is not None:
            return self.result
        self.dir.mkdir(parents=True, exist_ok=True)
        raw = (np.concatenate(self._raw) if self._raw
               else np.zeros(0, dtype=np.float32))
        derived = (np.concatenate(self._derived) if self._derived
                   else np.zeros(0, dtype=np.float32))

        raw_path = self.dir / "raw.wav"
        derived_path = self.dir / "derived_16k.wav"
        if self.metadata.raw_channels > 1 and raw.size:
            raw_out = raw.reshape(-1, self.metadata.raw_channels)
        else:
            raw_out = raw
        sf.write(str(raw_path), raw_out, self.metadata.raw_sample_rate, subtype="PCM_16")
        sf.write(str(derived_path), derived, self.metadata.derived_sample_rate, subtype="PCM_16")

        checks = []
        checks.extend(V.validate_file(raw_path, expected_sample_rate=self.metadata.raw_sample_rate,
                                      expected_channels=self.metadata.raw_channels, label="raw"))
        checks.extend(V.validate_file(derived_path,
                                      expected_sample_rate=self.metadata.derived_sample_rate,
                                      expected_channels=1, label="derived_16k"))
        checks.append(V.check_rate_consistency(self.raw_duration_s, self.derived_duration_s))

        metadata = self.metadata.to_json()
        metadata["ended_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["raw_duration_s"] = round(self.raw_duration_s, 3)
        metadata["derived_duration_s"] = round(self.derived_duration_s, 3)
        metadata["files"] = {"raw": raw_path.name, "derived_16k": derived_path.name}
        (self.dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        summary = V.summarise(checks)
        summary["session_id"] = self.metadata.session_id
        summary["is_mentra_hardware_audio"] = self.metadata.is_mentra_hardware_audio
        summary["source_kind"] = self.metadata.source_kind
        (self.dir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")

        self.closed = True
        self.result = {"session_dir": str(self.dir), "metadata": metadata,
                       "validation": summary}
        return self.result


def assert_mentra_hardware(session_dir: Path) -> Dict[str, object]:
    """Refuse to proceed unless this capture really came from Mentra hardware.

    Any G4 workstream that claims a Mentra result must call this. It is the
    code-level guard against the exact failure mode this campaign was warned
    about: presenting corpus, browser-mic, or synthetic audio as Mentra pilot
    data."""
    metadata_path = Path(session_dir) / "metadata.json"
    if not metadata_path.is_file():
        raise CaptureMetadataError(f"no capture metadata at {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    source = metadata.get("source_kind")
    if source not in MENTRA_HARDWARE_SOURCES:
        raise CaptureMetadataError(
            f"capture {session_dir} declares source_kind={source!r}, which is NOT Mentra "
            "hardware audio. It must not be reported as a Mentra device result.")
    return metadata


def load_sessions(root: Optional[Path] = None) -> List[Dict[str, object]]:
    root = Path(root) if root is not None else default_capture_root()
    if not root.is_dir():
        return []
    out = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            continue
        validation_path = metadata_path.parent / "validation.json"
        out.append({
            "session_dir": str(metadata_path.parent),
            "metadata": metadata,
            "validation": (json.loads(validation_path.read_text())
                           if validation_path.is_file() else None),
        })
    return out
