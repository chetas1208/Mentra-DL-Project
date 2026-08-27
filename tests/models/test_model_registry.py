"""Multi-model runtime registry: real models, real checkpoint, real failures.

Nothing here mocks the models the live receiver actually serves. The frozen G2
checkpoint is loaded from disk and its sha256 re-verified, and SpeakerNet's
real ONNX runtime is used -- because the whole point of the registry is that
``ready=true`` means a genuine forward pass succeeded, not that a file path
was configured.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from server.models.capabilities import LIVE_SERVABLE_MODELS
from server.models.registry import (G2_FROZEN_SHA256, ModelRegistry, SessionRuntime,
                                    sha256_file)

SPEAKER_MODEL = "models/sherpa-speaker/nemo_en_speakerverification_speakernet.onnx"
G2_CHECKPOINT = "training/geowearnet/mmcsg/frozen/g2_selected_07c43c3d9e37.pt"

requires_models = pytest.mark.skipif(
    not (Path(SPEAKER_MODEL).is_file() and Path(G2_CHECKPOINT).is_file()),
    reason="live model artifacts not present in this checkout",
)


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    reg = ModelRegistry(speaker_model_path=SPEAKER_MODEL,
                        geowearnet_checkpoint=G2_CHECKPOINT,
                        default_model="speakernet")
    reg.load_all()
    return reg


# --- catalog --------------------------------------------------------------
@requires_models
def test_registry_serves_exactly_the_two_live_models(registry: ModelRegistry):
    catalog = registry.catalog()
    assert catalog["defaultModel"] == "speakernet"
    assert [m["id"] for m in catalog["models"]] == LIVE_SERVABLE_MODELS == [
        "speakernet", "geowearnet_g2"]
    # Both genuinely loaded, not merely configured.
    assert all(m["ready"] for m in catalog["models"]), catalog


@requires_models
def test_catalog_payload_is_valid_json_for_the_frontend(registry: ModelRegistry):
    payload = json.loads(registry.catalog_payload())
    assert payload["type"] == "model_catalog"
    for entry in payload["models"]:
        assert isinstance(entry["displayName"], str) and entry["displayName"]
        assert isinstance(entry["ready"], bool)
        assert entry["unavailableReason"] is None or isinstance(entry["unavailableReason"], str)


@requires_models
def test_display_names_are_product_names_not_research_ids(registry: ModelRegistry):
    names = {m["id"]: m["displayName"] for m in registry.catalog()["models"]}
    assert names["speakernet"] == "SpeakerNet"
    assert names["geowearnet_g2"] == "GeoWearNet G2"


# --- capability truth -----------------------------------------------------
@requires_models
def test_speakernet_capabilities_are_accurate(registry: ModelRegistry):
    cap = registry.capabilities_for("speakernet")
    assert cap.requiresEnrollment is True
    assert cap.supportsEnrollment is True
    assert cap.supportsEnrollmentFreeDetection is False
    assert cap.supportsEnvironmentActivity is False   # never fabricate one for SpeakerNet
    assert cap.supportsOverlap is False
    assert cap.supportsSourceSeparation is False
    assert cap.experimental is False


@requires_models
def test_geowearnet_capabilities_are_accurate(registry: ModelRegistry):
    cap = registry.capabilities_for("geowearnet_g2")
    assert cap.requiresEnrollment is False
    assert cap.supportsEnrollment is False
    assert cap.supportsEnrollmentFreeDetection is True
    assert cap.supportsEnvironmentActivity is True
    assert cap.supportsOverlap is True
    assert cap.supportsSourceSeparation is False      # routing, never separation
    assert cap.experimental is True                   # not Mentra-hardware validated
    assert set(cap.outputLabels) == {"wearer", "environment", "overlap", "silence"}


# --- checkpoint identity --------------------------------------------------
def test_frozen_checkpoint_sha256_matches_the_pinned_value():
    assert Path(G2_CHECKPOINT).is_file()
    assert sha256_file(G2_CHECKPOINT) == G2_FROZEN_SHA256


@requires_models
def test_registry_reports_the_verified_checkpoint_hash(registry: ModelRegistry):
    loaded = registry.get("geowearnet_g2")
    assert loaded.checkpoint_sha256 == G2_FROZEN_SHA256
    entry = loaded.catalog_entry()
    assert entry["checkpointSha256Prefix"] == G2_FROZEN_SHA256[:12]


def test_a_tampered_checkpoint_is_refused_rather_than_served(tmp_path: Path):
    """A byte-level change must fail the load, not produce a quiet warning."""
    tampered = tmp_path / "tampered.pt"
    shutil.copyfile(G2_CHECKPOINT, tampered)
    with open(tampered, "ab") as handle:
        handle.write(b"\x00")

    reg = ModelRegistry(speaker_model_path=SPEAKER_MODEL, geowearnet_checkpoint=tampered,
                        default_model="geowearnet_g2", model_ids=["geowearnet_g2"])
    reg.load_all()
    loaded = reg.get("geowearnet_g2")
    assert loaded.ready is False
    assert "sha256 mismatch" in (loaded.failure_reason or "")


# --- real inference -------------------------------------------------------
@requires_models
def test_geowearnet_produces_real_bounded_scores(registry: ModelRegistry):
    runtime = registry.create_session_runtime("geowearnet_g2")
    result = runtime.detector.process(
        np.zeros(2 * 16_000, dtype=np.float32).astype(np.float32), 16_000)
    for key in ("wearer_score", "environment_score"):
        value = result[key]
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0


@requires_models
def test_geowearnet_refuses_enrollment_instead_of_faking_one(registry: ModelRegistry):
    runtime = registry.create_session_runtime("geowearnet_g2")
    with pytest.raises(RuntimeError, match="enrollment-free"):
        runtime.enroll([(np.zeros(16_000, dtype=np.float32), 16_000)])


# --- failure isolation ----------------------------------------------------
def test_one_model_failing_to_load_never_disables_the_other(tmp_path: Path):
    reg = ModelRegistry(speaker_model_path=SPEAKER_MODEL,
                        geowearnet_checkpoint=tmp_path / "does-not-exist.pt",
                        default_model="speakernet")
    reg.load_all()

    assert reg.is_ready("speakernet") is True
    assert reg.is_ready("geowearnet_g2") is False
    entry = reg.get("geowearnet_g2").catalog_entry()
    assert entry["ready"] is False
    assert "not found" in entry["unavailableReason"]
    # The catalog is still well-formed -- one optional model being down is not
    # a backend outage.
    assert json.loads(reg.catalog_payload())["defaultModel"] == "speakernet"


def test_speakernet_failing_to_load_never_disables_geowearnet(tmp_path: Path):
    reg = ModelRegistry(speaker_model_path=str(tmp_path / "missing.onnx"),
                        geowearnet_checkpoint=G2_CHECKPOINT,
                        default_model="geowearnet_g2")
    reg.load_all()
    assert reg.is_ready("speakernet") is False
    assert reg.is_ready("geowearnet_g2") is True


def test_a_dead_model_cannot_be_bound_to_a_session(tmp_path: Path):
    reg = ModelRegistry(speaker_model_path=SPEAKER_MODEL,
                        geowearnet_checkpoint=tmp_path / "missing.pt",
                        default_model="speakernet")
    reg.load_all()
    with pytest.raises(Exception):
        reg.create_session_runtime("geowearnet_g2")


# --- selection resolution -------------------------------------------------
@requires_models
def test_no_requested_model_falls_back_to_the_default(registry: ModelRegistry):
    assert registry.resolve_requested_model(None) == ("speakernet", None)


@requires_models
def test_unknown_model_id_is_an_error_not_a_silent_fallback(registry: ModelRegistry):
    model, error = registry.resolve_requested_model("wearersepnet")
    assert model is None
    assert "unknown model id" in error


def test_unavailable_model_reports_why_and_never_substitutes_another(tmp_path: Path):
    reg = ModelRegistry(speaker_model_path=SPEAKER_MODEL,
                        geowearnet_checkpoint=tmp_path / "missing.pt",
                        default_model="speakernet")
    reg.load_all()
    model, error = reg.resolve_requested_model("geowearnet_g2")
    assert model is None                       # never silently served SpeakerNet instead
    assert "GeoWearNet G2 unavailable" in error


@requires_models
def test_a_registry_default_must_itself_be_live_servable():
    with pytest.raises(ValueError, match="not one of the live-servable models"):
        ModelRegistry(speaker_model_path=SPEAKER_MODEL, geowearnet_checkpoint=G2_CHECKPOINT,
                      default_model="geowearnet_e1")


@requires_models
def test_sessions_get_independent_runtimes_sharing_one_loaded_model(registry: ModelRegistry):
    a = registry.create_session_runtime("speakernet")
    b = registry.create_session_runtime("speakernet")
    assert isinstance(a, SessionRuntime) and isinstance(b, SessionRuntime)
    assert a is not b
    assert a.consumer is not b.consumer
    assert a.detector is not b.detector            # private enrollment slot per session
    assert a.detector.extractor is b.detector.extractor  # one loaded ONNX runtime
