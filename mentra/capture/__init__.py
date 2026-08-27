"""G4 WS6 -- research capture for real wearable pilot sessions.

The audio arrives through the project's existing browser -> WebSocket capture
chain (see ``mentra.capture.session`` for the full path). This package adds
the research bookkeeping: session identity, anonymous wearer id, condition
label, provenance, both native-rate and derived 16 kHz streams, and automatic
validation of every take.
"""
from mentra.capture.session import (CaptureMetadata, CaptureMetadataError, CaptureSession,
                                    MENTRA_HARDWARE_SOURCES, SOURCE_KINDS,
                                    anonymous_wearer_id, assert_mentra_hardware,
                                    default_capture_root, load_sessions)
from mentra.capture.validation import (Check, overall_status, summarise, validate_audio,
                                       validate_file)

__all__ = [
    "CaptureMetadata", "CaptureMetadataError", "CaptureSession", "SOURCE_KINDS",
    "MENTRA_HARDWARE_SOURCES", "anonymous_wearer_id", "assert_mentra_hardware",
    "default_capture_root", "load_sessions", "Check", "overall_status", "summarise",
    "validate_audio", "validate_file",
]
