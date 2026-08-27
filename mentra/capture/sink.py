"""G4 WS6 -- the receiver-side capture sink.

One object owns the whole server-side capture lifecycle so that the live
receiver entrypoint and the end-to-end self-test exercise the SAME code. If
they had their own copies, a self-test pass would prove nothing about the
receiver an operator actually runs during a pilot.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional

from mentra.audio.frame import AudioFrame
from mentra.capture.session import CaptureMetadata, CaptureSession


class CaptureSink:
    """Routes CAPTURE_META / CAPTURE_RAW_PCM / derived AUDIO_FRAME payloads
    into a ``CaptureSession``, and finalises it when the transport closes.

    Deliberately tolerant of a bad take: a rejected metadata payload or a
    mismatched raw chunk is reported and counted, never raised into the
    receiver's frame loop, because dropping a live connection mid-pilot is a
    worse outcome than a flagged capture file.
    """

    def __init__(self, root: Path, log: Optional[Callable[[str], None]] = None):
        self.root = Path(root)
        self.log = log or (lambda message: print(message, flush=True))
        self.session: Optional[CaptureSession] = None
        self.rejected_meta = 0
        self.rejected_raw_chunks = 0
        self.last_result: Optional[Dict[str, object]] = None

    # -- callbacks used by MentraRemoteReceiver ------------------------------
    def on_capture_meta(self, session_id: str, payload: bytes) -> bytes:
        del session_id
        if self.session is not None and not self.session.closed:
            self.close("superseded by a new capture session")
        try:
            metadata = CaptureMetadata.from_json(json.loads(payload.decode("utf-8")))
        except Exception as exc:  # noqa: BLE001 -- surface, never silently drop
            self.rejected_meta += 1
            self.log(f"capture metadata rejected: {exc}")
            return json.dumps({"status": "REJECTED", "error": str(exc)}).encode("utf-8")

        self.session = CaptureSession(metadata, root=self.root)
        self.log(
            f"capture session {metadata.session_id} opened (wearer={metadata.wearer_id} "
            f"condition={metadata.condition!r} source={metadata.source_kind} "
            f"mentra_hardware={metadata.is_mentra_hardware_audio}) -> {self.session.dir}"
        )
        return json.dumps({
            "status": "OPEN",
            "session_id": metadata.session_id,
            "session_dir": str(self.session.dir),
            "is_mentra_hardware_audio": metadata.is_mentra_hardware_audio,
        }).encode("utf-8")

    def on_capture_raw(self, session_id: str, frame: AudioFrame) -> None:
        del session_id
        if self.session is None or self.session.closed:
            return
        try:
            self.session.add_raw_pcm16(frame.payload, frame.sample_rate, frame.channels)
        except ValueError as exc:
            self.rejected_raw_chunks += 1
            self.log(f"capture raw chunk rejected: {exc}")

    def on_derived_frame(self, payload: bytes) -> None:
        """The 16 kHz stream the model consumes, archived verbatim so a
        detector result from a pilot take can be reproduced exactly."""
        if self.session is None or self.session.closed:
            return
        self.session.add_derived_pcm16(payload)

    def close(self, reason: str) -> Optional[Dict[str, object]]:
        if self.session is None or self.session.closed:
            return self.last_result
        result = self.session.close()
        validation = result["validation"]
        self.log(
            f"capture session {self.session.metadata.session_id} closed ({reason}): "
            f"{validation['status']} ({validation['n_fail']} fail / "
            f"{validation['n_warn']} warn) -> {result['session_dir']}"
        )
        self.last_result = result
        return result
