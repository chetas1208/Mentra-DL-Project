"""Live detector selection and GeoWearNet runtime adapter tests."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from server.models import geowearnet as GW
from server.models.runtime import build_detector


def test_default_runtime_selects_speakernet(monkeypatch):
    class FakeSpeakerNet:
        def __init__(self, path, num_threads):
            assert path == "speaker.onnx"
            assert num_threads == 2

    monkeypatch.setattr("research.sherpa_onnx.detector.SherpaOnnxWearerDetector", FakeSpeakerNet)
    result = build_detector("speakernet", "speaker.onnx", None, num_threads=2)
    assert isinstance(result, FakeSpeakerNet)


def test_e1_runtime_is_explicitly_rejected():
    with pytest.raises(RuntimeError, match="no live E1 detector is wired"):
        build_detector("geowearnet_e1", "speaker.onnx", None)


def test_geowearnet_detector_returns_probability_from_final_frame(monkeypatch, tmp_path):
    class FakeModel:
        def eval(self):
            return self

        def __call__(self, mel, physical):
            assert mel.shape[0] == 1
            assert physical is not None
            return {
                "wearer_logits": torch.tensor([[-2.0, 2.0]]),
                "environment_logits": torch.tensor([[1.0, -1.0]]),
            }

    config = SimpleNamespace(n_mels=64, n_physical_features=14, use_physical_features=True)
    checkpoint = {"config": {"normalization": "none", "drop_amplitude_features": False}}
    monkeypatch.setattr(GW, "load_checkpoint", lambda path: (FakeModel(), config, checkpoint))

    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    detector = GW.GeoWearNetDetector(checkpoint_path)
    result = detector.process(np.zeros(1600, dtype=np.float32), 16_000)

    assert result["wearer_logit"] == pytest.approx(2.0)
    assert result["wearer_score"] == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert result["environment_logit"] == pytest.approx(-1.0)


def test_geowearnet_detector_rejects_enrollment(monkeypatch, tmp_path):
    config = SimpleNamespace(n_mels=64, n_physical_features=14, use_physical_features=True)
    checkpoint = {"config": {"normalization": "none", "drop_amplitude_features": False}}

    class FakeModel:
        def eval(self):
            return self

    monkeypatch.setattr(GW, "load_checkpoint", lambda path: (FakeModel(), config, checkpoint))
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    detector = GW.GeoWearNetDetector(checkpoint_path)

    with pytest.raises(RuntimeError, match="enrollment-free"):
        detector.enroll([])
