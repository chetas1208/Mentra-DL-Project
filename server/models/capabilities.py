"""Backend model-capability registry (Workstreams AW, AX).

The web console already implements a real capability handshake: it parses the
STREAM_ACCEPTED payload with `web/app/utils/modelCapabilities.ts` and renders
"NOT AVAILABLE" when the server sends nothing, rather than guessing a default.
Until now the server sent `b""`, so no model ever advertised itself.

This module supplies that payload. It is the ONLY thing GeoWearNet needs from
the backend to become selectable -- no frontend change, no redesign.

ACTIVATION IS CONFIGURATION-DRIVEN, never hard-coded:

    MENTRA_MODEL=geowearnet_e1  python3 -m scripts.mentra.run_receiver ...

The default stays `speakernet` so existing deployments behave exactly as before.
Existing models are registered here, not replaced: SpeakerNet and MentraWearNet
remain available and untouched.

The field names and types mirror `ModelCapabilities` in
`web/app/utils/modelCapabilities.ts` exactly; `validate_against_frontend_contract()`
asserts that, so the two can never silently drift apart.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional

# Must match ModelCapabilities in web/app/utils/modelCapabilities.ts
FRONTEND_REQUIRED_FIELDS = {
    "modelId": str,
    "modelVersion": str,
    "ready": bool,
    "task": str,
    "requiresEnrollment": bool,
    "supportsEnrollment": bool,
    "supportsEnrollmentFreeDetection": bool,
    "supportsPassivePersonalization": bool,
    "supportsOverlap": bool,
    "supportsEnvironmentActivity": bool,
    "supportsSourceSeparation": bool,
    "audioPolicy": str,
    "experimental": bool,
    "outputLabels": list,
}


@dataclass
class ModelCapabilities:
    modelId: str
    modelVersion: str
    ready: bool
    task: str
    requiresEnrollment: bool
    supportsEnrollment: bool
    supportsEnrollmentFreeDetection: bool
    supportsPassivePersonalization: bool
    supportsOverlap: bool
    supportsEnvironmentActivity: bool
    supportsSourceSeparation: bool
    audioPolicy: str = "passthrough"
    experimental: bool = False
    outputLabels: List[str] = field(default_factory=list)

    def to_payload(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")


REGISTRY: Dict[str, ModelCapabilities] = {
    # --- existing, unchanged, still selectable ---
    "speakernet": ModelCapabilities(
        modelId="speakernet",
        modelVersion="nemo-speakerverification-1",
        ready=True,
        task="speaker_verification",
        requiresEnrollment=True,
        supportsEnrollment=True,
        supportsEnrollmentFreeDetection=False,
        supportsPassivePersonalization=False,
        supportsOverlap=False,
        supportsEnvironmentActivity=False,
        supportsSourceSeparation=False,
        audioPolicy="passthrough",
        outputLabels=["wearer", "not_wearer"],
    ),
    "mentrawearnet": ModelCapabilities(
        modelId="mentrawearnet",
        modelVersion="v3",
        ready=True,
        task="enrollment_conditioned_wearer_detection",
        requiresEnrollment=True,
        supportsEnrollment=True,
        supportsEnrollmentFreeDetection=False,
        supportsPassivePersonalization=False,
        supportsOverlap=True,
        supportsEnvironmentActivity=True,
        supportsSourceSeparation=False,
        audioPolicy="passthrough",
        outputLabels=["wearer", "environment", "overlap", "silence"],
    ),
    # --- GeoWearNet: the enrollment-free branch ---
    #
    # `ready=False` until a checkpoint is actually wired into the live inference
    # path. Advertising ready=True on the strength of simulation results alone
    # would be exactly the overclaim this campaign forbids: GeoWearNet is
    # SIMULATION-VALIDATED, not MENTRA-PILOT-VALIDATED.
    # `set_geowearnet_ready()` flips it once a real checkpoint is loaded.
    "geowearnet_e1": ModelCapabilities(
        modelId="geowearnet_e1",
        modelVersion="e1-sim-v1",
        ready=False,
        task="enrollment_free_wearer_detection",
        requiresEnrollment=False,
        supportsEnrollment=False,
        supportsEnrollmentFreeDetection=True,
        supportsPassivePersonalization=False,
        supportsOverlap=True,
        supportsEnvironmentActivity=True,
        supportsSourceSeparation=False,
        audioPolicy="passthrough",
        experimental=True,
        outputLabels=["wearer", "environment", "overlap", "silence"],
    ),
    # --- GeoWearNet G2: real-wearable-domain (MMCSG) checkpoint. ---
    #
    # Distinct model id from `geowearnet_e1` on purpose: G1 is 100% simulation
    # -trained, G2 is trained/fine-tuned on REAL Aria smart-glasses audio
    # (MMCSG). Keeping both registered lets the backend serve either one and
    # keeps the provenance of any served prediction unambiguous. `ready=False`
    # until `set_geowearnet_g2_ready()` is called with an actual selected
    # checkpoint (see training/geowearnet/mmcsg/campaign_real.py /
    # docs/geowearnet_g2_mmcsg_report.md for the selection). Even once ready,
    # this is MMCSG-validated (real Aria hardware), NOT Mentra-hardware
    # validated -- the frontend/console must not imply otherwise.
    "geowearnet_g2": ModelCapabilities(
        modelId="geowearnet_g2",
        modelVersion="g2-real-mmcsg-v1",
        ready=False,
        task="enrollment_free_wearer_detection",
        requiresEnrollment=False,
        supportsEnrollment=False,
        supportsEnrollmentFreeDetection=True,
        supportsPassivePersonalization=False,
        supportsOverlap=True,
        supportsEnvironmentActivity=True,
        supportsSourceSeparation=False,
        audioPolicy="passthrough",
        experimental=True,
        outputLabels=["wearer", "environment", "overlap", "silence"],
    ),
}

DEFAULT_MODEL = "speakernet"
ENV_VAR = "MENTRA_MODEL"

# User-facing names for the web console's model selector. Deliberately short
# product names, not research nomenclature: the exact checkpoint/version stays
# in `modelVersion` (and the debug drawer), never in the main dashboard.
# This is a presentation lookup, NOT a field on ModelCapabilities -- adding a
# field there would break `validate_against_frontend_contract()` and the
# frontend's strict `parseCapabilities` type guard.
DISPLAY_NAMES: Dict[str, str] = {
    "speakernet": "SpeakerNet",
    "mentrawearnet": "MentraWearNet",
    "geowearnet_e1": "GeoWearNet E1",
    "geowearnet_g2": "GeoWearNet G2",
}

# Models with an actual live runtime behind them (see server/models/runtime.py).
# `mentrawearnet` and `geowearnet_e1` stay registered for capability reporting
# and research provenance but have no live detector, so they are never offered
# in the web console's selector -- advertising a model the backend cannot serve
# would be exactly the fake-readiness this contract exists to prevent.
LIVE_SERVABLE_MODELS: List[str] = ["speakernet", "geowearnet_g2"]


def display_name(model_id: str) -> str:
    return DISPLAY_NAMES.get(model_id, model_id)


def active_model_id() -> str:
    return os.environ.get(ENV_VAR, DEFAULT_MODEL)


def active_capabilities(model_id: Optional[str] = None) -> ModelCapabilities:
    mid = model_id or active_model_id()
    if mid not in REGISTRY:
        raise KeyError(f"unknown {ENV_VAR}={mid!r}; known models: {sorted(REGISTRY)}")
    return REGISTRY[mid]


# Mirrors server.audio.frontend.SUPPORTED_POLICIES. Duplicated deliberately so
# the capability contract does not drag numpy/evaluation imports into the web
# contract check; tests/audio/test_frontend_envelope.py asserts no drift.
#
# G4 WS40 -- the experimental live gate configuration, documented but NOT
# defaulted:
#     MENTRA_MODEL=geowearnet_g2 \
#     MENTRA_AUDIO_POLICY=geowear_envelope \
#     MENTRA_GATE_POLICY=A_balanced MENTRA_GATE_PREROLL_MS=150 \
#     python3 scripts/mentra/run_receiver.py --port <unused>
# That configuration reports experimental=true, requiresEnrollment=false and
# supportsSourceSeparation=false. The DEFAULT remains passthrough on
# speakernet, and no `geowearnet_g4` model id is registered because G4 produced
# no new checkpoint -- the frozen G2 parent is unchanged, and inventing a G4
# model id would imply a Mentra-adapted model that does not exist.
AUDIO_POLICIES = ("passthrough", "geowear_gate", "rnnoise_geowear_gate",
                  "geowear_envelope", "rnnoise_geowear_envelope")


def with_audio_policy(capabilities: ModelCapabilities, audio_policy: str) -> ModelCapabilities:
    """Return a per-receiver capability payload without mutating the registry.

    A policy is a live-pipeline choice, not a property of a model checkpoint;
    mutating the global registry would make a later client see another
    receiver's policy.  Source separation stays false for every current
    policy because no routing option extracts a source waveform -- the
    envelope policies shape a per-frame GAIN on the original mixture, and an
    OVERLAP frame stays mixed.
    """
    if audio_policy not in AUDIO_POLICIES:
        raise ValueError(
            f"unsupported audio policy {audio_policy!r}; choose one of {sorted(AUDIO_POLICIES)}")
    return replace(
        capabilities,
        audioPolicy=audio_policy,
        experimental=capabilities.experimental or audio_policy != "passthrough",
    )


def set_geowearnet_ready(checkpoint_path: str, model_version: Optional[str] = None) -> ModelCapabilities:
    """Flip GeoWearNet (G1, simulation) to ready once a real checkpoint is
    loaded into the inference path. Kept explicit so `ready` can never drift
    true by accident."""
    cap = REGISTRY["geowearnet_e1"]
    cap.ready = True
    if model_version:
        cap.modelVersion = model_version
    return cap


def set_geowearnet_g2_ready(checkpoint_path: str, model_version: Optional[str] = None) -> ModelCapabilities:
    """Flip GeoWearNet G2 (real MMCSG-trained/fine-tuned) to ready once its
    selected checkpoint is loaded into the inference path."""
    cap = REGISTRY["geowearnet_g2"]
    cap.ready = True
    if model_version:
        cap.modelVersion = model_version
    return cap


def validate_against_frontend_contract() -> Dict[str, object]:
    """Assert every registered model satisfies the frontend's parser exactly.

    `parseCapabilities` returns null (-> "NOT AVAILABLE") on any type mismatch,
    so a silent drift here would look like a UI bug. This makes it a test
    failure instead (Workstream BH).
    """
    report: Dict[str, object] = {"models": {}}
    for mid, cap in REGISTRY.items():
        d = json.loads(cap.to_payload())
        assert set(d.keys()) == set(FRONTEND_REQUIRED_FIELDS.keys()), (
            f"{mid}: field set differs from the frontend contract: "
            f"extra={set(d) - set(FRONTEND_REQUIRED_FIELDS)} "
            f"missing={set(FRONTEND_REQUIRED_FIELDS) - set(d)}")
        for k, t in FRONTEND_REQUIRED_FIELDS.items():
            assert isinstance(d[k], t), f"{mid}.{k} should be {t.__name__}, got {type(d[k]).__name__}"
        report["models"][mid] = d
    for geo_id in ("geowearnet_e1", "geowearnet_g2"):
        geo = REGISTRY[geo_id]
        assert geo.requiresEnrollment is False
        assert geo.supportsEnrollmentFreeDetection is True
        assert geo.supportsOverlap is True
        assert geo.supportsEnvironmentActivity is True
        assert geo.supportsSourceSeparation is False
    report["geowearnet_contract_ok"] = True
    report["default_model"] = DEFAULT_MODEL
    report["env_var"] = ENV_VAR
    return report


if __name__ == "__main__":
    print(json.dumps(validate_against_frontend_contract(), indent=2))
    print("active:", active_model_id())
