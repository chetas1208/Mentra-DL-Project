"""WearerSepNet -- INTERFACE STUB ONLY. NOT AN IMPLEMENTATION.

    ####################################################################
    #  THERE IS NO MODEL HERE AND THERE MUST NOT BE ONE YET.           #
    #  No architecture is selected. No training code exists. No GPU     #
    #  time may be spent on this. See docs/geowearnet_wearersepnet_    #
    #  spec.md, which fixes the acceptance criteria BEFORE any model    #
    #  exists, and §1's predeclared go/no-go test which must pass on    #
    #  real measurements before this is built at all.                   #
    ####################################################################

This module exists so that:
  1. the pipeline in `evaluation/agent_audio/` can be written against a
     stable interface today, and
  2. `PassThroughWearerSep` gives that pipeline a real, runnable object --
     an explicit no-op that is honest about doing nothing -- instead of
     `None` checks scattered through the routing code.

`PassThroughWearerSep` returns its input unchanged. That is exactly the
current oracle-gate "pass overlap" behaviour, so wiring it in changes no
measured number. Any future real implementation must beat it on the
acceptance suite in the spec, and the fact that it is a trivial baseline is
the point: a separator that cannot beat "do nothing" is not worth shipping.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Protocol, runtime_checkable

import numpy as np

SR = 16000
FRAME_HOP_MS = 10.0
HOP = int(SR * FRAME_HOP_MS / 1000.0)

# Budget constants from the spec, exposed so tests/tools can assert against
# them rather than re-typing numbers that could drift out of sync.
PARAM_TARGET_MIN = 500_000
PARAM_TARGET_MAX = 3_000_000
SYSTEM_PARAM_HARD_LIMIT = 10_000_000
MAX_ADDED_LATENCY_MS = 32.0

CANDIDATE_ARCHITECTURES = (
    "stft_mask_causal_tcn",
    "small_spectral_crnn",
    "light_convtasnet",
)
SELECTED_ARCHITECTURE = None      # deliberately unset -- see the spec, §6

CONDITIONING_MODES = ("none", "probabilities", "physical_features", "embedding")


@runtime_checkable
class WearerSepNetInterface(Protocol):
    """The contract fixed in docs/geowearnet_wearersepnet_spec.md §2."""

    def reset(self) -> None:
        """Clear all streaming state."""
        ...

    def process_frame(self, pcm_frame: np.ndarray, p_wearer: float,
                      p_environment: float,
                      geo_features: Optional[np.ndarray] = None) -> np.ndarray:
        """One 10 ms frame in -> one 10 ms frame of wearer-dominant PCM out.
        Strictly causal: must not use any sample after `pcm_frame`."""
        ...


@dataclasses.dataclass
class WearerSepConfig:
    conditioning: str = "probabilities"
    frame_hop_ms: float = FRAME_HOP_MS
    sample_rate: int = SR

    def __post_init__(self):
        if self.conditioning not in CONDITIONING_MODES:
            raise ValueError(f"conditioning must be one of {CONDITIONING_MODES}")


class PassThroughWearerSep:
    """The honest no-op. Returns the mixture unchanged.

    Equivalent to the gate's `overlap_policy="pass_raw"`. Serves as:
      * a runnable placeholder so pipeline code needs no None-checks, and
      * the baseline any real separator must beat on the acceptance suite.
    """

    def __init__(self, config: Optional[WearerSepConfig] = None):
        self.config = config or WearerSepConfig()
        self.is_implemented = False
        self.n_parameters = 0

    def reset(self) -> None:
        return None

    def process_frame(self, pcm_frame: np.ndarray, p_wearer: float,
                      p_environment: float,
                      geo_features: Optional[np.ndarray] = None) -> np.ndarray:
        return np.ascontiguousarray(pcm_frame, dtype=np.float32)

    def process(self, pcm: np.ndarray, p_wearer, p_environment) -> np.ndarray:
        return np.ascontiguousarray(pcm, dtype=np.float32)

    def describe(self) -> dict:
        return {
            "name": "PassThroughWearerSep",
            "is_implemented": False,
            "status": "SPEC_ONLY_NO_MODEL_TRAINED",
            "n_parameters": 0,
            "selected_architecture": SELECTED_ARCHITECTURE,
            "candidate_architectures": list(CANDIDATE_ARCHITECTURES),
            "spec": "docs/geowearnet_wearersepnet_spec.md",
            "note": ("Returns the mixture unchanged. Identical to the GeoWear Gate's "
                     "pass_raw overlap policy, so wiring it in changes no measured "
                     "number. Any real implementation must beat this on the spec's "
                     "acceptance suite."),
        }


def build(config: Optional[WearerSepConfig] = None):
    """Factory. Deliberately cannot return a trained model -- there is none.

    Kept so the eventual implementation has an obvious single entry point, and
    so any code that calls it today gets the honest pass-through rather than
    a plausible-looking but untrained network.
    """
    return PassThroughWearerSep(config)


if __name__ == "__main__":
    import json
    print(json.dumps(build().describe(), indent=2))
