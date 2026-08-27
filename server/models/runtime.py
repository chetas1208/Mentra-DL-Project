"""Model construction for the Mentra receiver."""
from __future__ import annotations

from pathlib import Path


def build_detector(model_id: str, speaker_model_path: str,
                   geowearnet_checkpoint: str | Path | None, num_threads: int = 1):
    """Build the detector selected by ``MENTRA_MODEL``.

    The default SpeakerNet path remains unchanged. GeoWearNet E1 is kept in
    the capability registry for research reporting but is not silently routed
    to a different runtime implementation.
    """
    if model_id == "speakernet":
        from research.sherpa_onnx.detector import SherpaOnnxWearerDetector

        return SherpaOnnxWearerDetector(speaker_model_path, num_threads=num_threads)

    if model_id == "geowearnet_g2":
        if geowearnet_checkpoint is None:
            raise ValueError("geowearnet_g2 requires a checkpoint path")
        from .geowearnet import GeoWearNetDetector

        return GeoWearNetDetector(geowearnet_checkpoint, num_threads=num_threads)

    if model_id == "geowearnet_e1":
        raise RuntimeError(
            "MENTRA_MODEL=geowearnet_e1 is registered for capability reporting, "
            "but no live E1 detector is wired; use speakernet or geowearnet_g2"
        )

    raise KeyError(f"unsupported live receiver model: {model_id!r}")
