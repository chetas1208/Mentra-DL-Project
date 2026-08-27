"""Multi-model runtime registry for the live Mentra receiver.

ONE audio pipeline, N selectable models. The receiver's transport, jitter
buffer, AudioFrame contract, audio frontend and ASR are entirely unaware of
which model a given session picked:

    audio source -> AudioFrame -> LiveSession -> SessionRuntime -> detector
                                              -> common result contract

Two responsibilities live here and nowhere else:

1. PROCESS-LEVEL LOADING. Each model's weights are loaded at most once per
   process (``LoadedModel``), never per frame and never per session. A model
   that fails to load is recorded with its failure reason and reported
   ``ready=false``; it can never corrupt the registry or the other model
   (Phase 30/31 -- SpeakerNet must keep working if GeoWearNet is missing, and
   vice versa).

2. PER-SESSION BINDING. ``create_session_runtime()`` returns a
   ``SessionRuntime`` that owns everything stream-local (rolling PCM window,
   hysteresis latches, and -- for SpeakerNet -- the enrolled embedding) while
   SHARING the immutable loaded weights. Model choice is therefore session
   state, not process state: two concurrent sessions can run different models
   without restarting the server and without seeing each other's identity,
   latches or scores.

Readiness is never assumed. ``geowearnet_g2`` is advertised ready ONLY after
the frozen checkpoint exists, its sha256 matches ``G2_FROZEN_SHA256``, the
normalization artifact resolves, the architecture loads, the runtime adapter
initialises, and a real forward pass returns finite scores in [0, 1].
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mentra.audio.consumer import MentraInferenceConsumer
from server.models.capabilities import (DEFAULT_MODEL, LIVE_SERVABLE_MODELS, REGISTRY,
                                        ModelCapabilities, display_name,
                                        set_geowearnet_g2_ready, with_audio_policy)
from server.models.runtime import build_detector

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

# The one frozen, wearable-domain (MMCSG) GeoWearNet checkpoint this build is
# allowed to serve. Re-verified on every load; a mismatch is a hard load
# failure, never a warning, so a swapped/corrupted file can never be served
# under the G2 name.
G2_FROZEN_SHA256 = "07c43c3d9e37dbd490ae0477f46ff515e1e19e8fe0a20687ec9d13a8555c68ef"

# Per-model consumer thresholds. SpeakerNet keeps the historical defaults
# (cosine similarity against an enrolled embedding, binary hysteresis).
# GeoWearNet G2 uses the predeclared product threshold pair on two real
# calibrated probabilities, which is what lets the consumer publish OVERLAP
# instead of collapsing it into "not the wearer".
CONSUMER_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "speakernet": {},
    "geowearnet_g2": {
        "wearer_high_threshold": 0.60,
        "wearer_low_threshold": 0.40,
        "environment_high_threshold": 0.60,
        "environment_low_threshold": 0.40,
    },
}


class ModelLoadError(RuntimeError):
    """A model could not be brought to a genuinely servable state."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class LoadedModel:
    """One model's process-level state: immutable weights + honest readiness."""

    model_id: str
    display_name: str
    capabilities: ModelCapabilities
    detector: object | None = None
    checkpoint_sha256: Optional[str] = None
    failure_reason: Optional[str] = None
    # Guards inference on the SHARED detector object. The live receiver drives
    # every session from one asyncio event loop, so inference is already
    # serialised in practice; this makes that safe rather than incidental if a
    # future caller ever dispatches a forward pass to a thread executor. It is
    # per-model, deliberately not one global lock -- SpeakerNet inference must
    # never be able to block a GeoWearNet session.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def ready(self) -> bool:
        return self.detector is not None and self.capabilities.ready

    def catalog_entry(self) -> Dict[str, object]:
        """Flat entry for the model_catalog control message.

        Field names are the EXISTING capability-schema names, not a parallel
        vocabulary, so the frontend needs no second contract to understand it.
        """
        cap = self.capabilities
        return {
            "id": self.model_id,
            "displayName": self.display_name,
            "modelVersion": cap.modelVersion,
            "ready": self.ready,
            "experimental": cap.experimental,
            "requiresEnrollment": cap.requiresEnrollment,
            "supportsEnrollment": cap.supportsEnrollment,
            "supportsEnrollmentFreeDetection": cap.supportsEnrollmentFreeDetection,
            "supportsEnvironmentActivity": cap.supportsEnvironmentActivity,
            "supportsOverlap": cap.supportsOverlap,
            "supportsSourceSeparation": cap.supportsSourceSeparation,
            "unavailableReason": self.failure_reason,
            # Provenance for the debug drawer only -- never the main dashboard.
            "checkpointSha256Prefix": (self.checkpoint_sha256[:12]
                                       if self.checkpoint_sha256 else None),
        }


class SessionRuntime:
    """One model bound to ONE session. Owns every stream-local thing.

    Nothing here is shared with another session: the rolling PCM window and
    hysteresis latches live in this object's own ``MentraInferenceConsumer``,
    and SpeakerNet's enrolled embedding lives in this object's own shallow
    detector view (which shares the loaded ONNX runtime but never the
    embedding). That is the shared-glasses guarantee at the model layer: no
    identity, latch or rolling audio can survive into another session.
    """

    def __init__(self, loaded: LoadedModel, context_s: float, hop_s: float):
        if not loaded.ready:
            raise ModelLoadError(
                f"model {loaded.model_id!r} is not ready: {loaded.failure_reason or 'unknown reason'}")
        self._loaded = loaded
        self.model_id = loaded.model_id
        self.display_name = loaded.display_name
        self.capabilities = loaded.capabilities
        self.model_version = loaded.capabilities.modelVersion
        self.checkpoint_sha256 = loaded.checkpoint_sha256
        # SpeakerNet keeps its enrolled embedding on the detector itself, so a
        # session gets its OWN shallow view: same loaded ONNX extractor, a
        # private `_wearer_embedding` slot. GeoWearNet G2 is enrollment-free
        # and stateless across calls, so the shared object is used directly.
        self.detector = (
            copy.copy(loaded.detector) if loaded.capabilities.supportsEnrollment
            else loaded.detector
        )
        self.consumer = MentraInferenceConsumer(
            self.detector, context_s=context_s, hop_s=hop_s,
            **CONSUMER_THRESHOLDS.get(loaded.model_id, {}),
        )

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        """Drop every stream-local value; keep the bound model and (for
        SpeakerNet) this session's own enrollment lifecycle, which the
        transport layer owns explicitly."""
        self.consumer.reset()

    def enroll(self, segments: Sequence[Tuple[np.ndarray, int]]) -> None:
        if not self.capabilities.supportsEnrollment:
            raise RuntimeError(
                f"{self.display_name} is enrollment-free and does not accept enrollment")
        self.consumer.enroll(list(segments))

    def infer(self, frame):
        with self._loaded.lock:
            return self.consumer.consume_frame(frame)

    @property
    def current_state(self) -> str:
        return self.consumer.current_state


class ModelRegistry:
    """Process-level catalog of loadable models + per-session binding factory."""

    def __init__(self, speaker_model_path: str,
                 geowearnet_checkpoint: str | Path | None = None,
                 default_model: str = DEFAULT_MODEL,
                 audio_policy: str = "passthrough",
                 speaker_threads: int = 1,
                 geowearnet_threads: int = 1,
                 context_s: float = 2.0, hop_s: float = 0.2,
                 speaker_enrollment: Optional[Sequence[Tuple[np.ndarray, int]]] = None,
                 model_ids: Optional[Sequence[str]] = None):
        self.speaker_model_path = speaker_model_path
        self.geowearnet_checkpoint = geowearnet_checkpoint
        self.audio_policy = audio_policy
        self.speaker_threads = speaker_threads
        self.geowearnet_threads = geowearnet_threads
        self.context_s = context_s
        self.hop_s = hop_s
        self.speaker_enrollment = list(speaker_enrollment) if speaker_enrollment else None
        self.model_ids: List[str] = list(model_ids or LIVE_SERVABLE_MODELS)
        if default_model not in self.model_ids:
            raise ValueError(
                f"default model {default_model!r} is not one of the live-servable models "
                f"{self.model_ids}")
        self.default_model = default_model
        self.models: Dict[str, LoadedModel] = {}

    # -- loading -----------------------------------------------------------
    def load_all(self) -> Dict[str, LoadedModel]:
        """Load every configured model. A failure in one is contained: it is
        recorded as ``ready=false`` with a reason and the others still load."""
        for model_id in self.model_ids:
            self.models[model_id] = self._load_one(model_id)
        if not self.models[self.default_model].ready:
            logger.warning("default model %s is NOT ready: %s", self.default_model,
                           self.models[self.default_model].failure_reason)
        return self.models

    def _base_capabilities(self, model_id: str) -> ModelCapabilities:
        # A private copy per registry: the module-level REGISTRY entries are
        # shared, mutable dataclasses, and a receiver must never be able to
        # mutate another component's view of a model.
        return with_audio_policy(replace(REGISTRY[model_id]), self.audio_policy)

    def _load_one(self, model_id: str) -> LoadedModel:
        caps = self._base_capabilities(model_id)
        loaded = LoadedModel(model_id=model_id, display_name=display_name(model_id),
                             capabilities=caps)
        try:
            if model_id == "geowearnet_g2":
                self._load_geowearnet_g2(loaded)
            elif model_id == "speakernet":
                self._load_speakernet(loaded)
            else:
                raise ModelLoadError(f"no live loader for model id {model_id!r}")
        except Exception as exc:  # noqa: BLE001 -- failure isolation is the point
            loaded.detector = None
            loaded.capabilities = replace(loaded.capabilities, ready=False)
            loaded.failure_reason = f"{type(exc).__name__}: {exc}"
            logger.warning("model %s failed to load: %s", model_id, loaded.failure_reason)
        return loaded

    def _load_speakernet(self, loaded: LoadedModel) -> None:
        path = Path(self.speaker_model_path)
        if not path.is_file():
            raise ModelLoadError(f"SpeakerNet model not found: {path}")
        detector = build_detector("speakernet", str(path), None, self.speaker_threads)
        if self.speaker_enrollment:
            # Enrolled ONCE on the prototype so every session starts from the
            # same documented placeholder, exactly as the single-model receiver
            # did. A session's own ENROLL_AUDIO then replaces the embedding on
            # that session's private detector view only.
            detector.enroll(list(self.speaker_enrollment))
            probe = detector.process(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
            score = float(probe["wearer_score"])
            if not np.isfinite(score):
                raise ModelLoadError("SpeakerNet probe forward pass returned a non-finite score")
        loaded.detector = detector
        loaded.checkpoint_sha256 = None  # large ONNX; hashed on demand, not at every boot
        loaded.capabilities = replace(loaded.capabilities, ready=True)

    def _load_geowearnet_g2(self, loaded: LoadedModel) -> None:
        if self.geowearnet_checkpoint is None:
            raise ModelLoadError("no GeoWearNet G2 checkpoint configured")
        path = Path(self.geowearnet_checkpoint).expanduser()
        if not path.is_file():
            raise ModelLoadError(f"GeoWearNet G2 checkpoint not found: {path}")
        digest = sha256_file(path)
        if digest != G2_FROZEN_SHA256:
            raise ModelLoadError(
                f"GeoWearNet G2 checkpoint sha256 mismatch: expected {G2_FROZEN_SHA256}, "
                f"got {digest} for {path}")
        # Constructing the adapter validates the architecture, the training
        # config's feature contract, and the presence of the normalization
        # artifact -- all of which must hold before ready may become true.
        detector = build_detector("geowearnet_g2", self.speaker_model_path, path,
                                  self.geowearnet_threads)
        probe = detector.process(np.zeros(2 * SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        for key in ("wearer_score", "environment_score"):
            value = float(probe[key])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ModelLoadError(
                    f"GeoWearNet G2 probe forward pass returned an invalid {key}={value!r}")
        # The explicit, documented opt-in gate in capabilities.py -- called only
        # now, after a real checkpoint has actually produced real scores.
        set_geowearnet_g2_ready(str(path))
        loaded.detector = detector
        loaded.checkpoint_sha256 = digest
        loaded.capabilities = replace(loaded.capabilities, ready=True)

    # -- queries -----------------------------------------------------------
    def get(self, model_id: str) -> LoadedModel:
        if model_id not in self.models:
            raise KeyError(f"unknown model {model_id!r}; known: {sorted(self.models)}")
        return self.models[model_id]

    def is_ready(self, model_id: str) -> bool:
        return model_id in self.models and self.models[model_id].ready

    def catalog(self) -> Dict[str, object]:
        return {
            "type": "model_catalog",
            "defaultModel": self.default_model,
            "models": [self.models[m].catalog_entry() for m in self.model_ids
                       if m in self.models],
        }

    def catalog_payload(self) -> bytes:
        return json.dumps(self.catalog()).encode("utf-8")

    def capabilities_for(self, model_id: str) -> ModelCapabilities:
        return self.get(model_id).capabilities

    # -- per-session binding ----------------------------------------------
    def create_session_runtime(self, model_id: Optional[str] = None) -> SessionRuntime:
        """Bind ONE model to one session. Never mutates process state."""
        chosen = model_id or self.default_model
        loaded = self.get(chosen)
        return SessionRuntime(loaded, context_s=self.context_s, hop_s=self.hop_s)

    def resolve_requested_model(self, requested: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Validate a client's requested model id.

        Returns ``(model_id, error)``. Exactly one is non-None. An unknown id
        and an unavailable model are different failures and are reported as
        such -- neither silently falls back to the default, because a client
        that asked for GeoWearNet must never be told it got GeoWearNet while
        actually being served SpeakerNet.
        """
        if requested is None:
            return self.default_model, None
        if requested not in self.models:
            return None, f"unknown model id {requested!r}"
        loaded = self.models[requested]
        if not loaded.ready:
            return None, f"{loaded.display_name} unavailable: {loaded.failure_reason or 'not ready'}"
        return requested, None


def build_registry_from_args(speaker_model_path: str,
                             geowearnet_checkpoint: str | Path | None,
                             default_model: str,
                             audio_policy: str,
                             context_s: float, hop_s: float,
                             speaker_threads: int, geowearnet_threads: int,
                             speaker_enrollment_loader: Optional[Callable[[], Sequence[Tuple[np.ndarray, int]]]] = None,
                             ) -> ModelRegistry:
    """Convenience constructor used by the receiver entrypoint."""
    enrollment = speaker_enrollment_loader() if speaker_enrollment_loader is not None else None
    registry = ModelRegistry(
        speaker_model_path=speaker_model_path,
        geowearnet_checkpoint=geowearnet_checkpoint,
        default_model=default_model,
        audio_policy=audio_policy,
        speaker_threads=speaker_threads,
        geowearnet_threads=geowearnet_threads,
        context_s=context_s, hop_s=hop_s,
        speaker_enrollment=enrollment,
    )
    registry.load_all()
    return registry
