"""Agent-audio product evaluation lane (P1).

The GeoWearNet campaigns measure *detection* quality (AUROC, EER, frame F1).
This package measures the thing the customer actually experiences:

    wearer speech + bystander speech + machinery noise
        -> [optional denoise] -> [GeoWear Gate] -> ASR / voice agent
        -> did the agent hear the WEARER, and only the wearer?

Primary axis is recognition/agent correctness (wearer WER/TER, BYSTANDER
LEAKAGE RATE, false agent command rate), NOT perceptual scores. Perceptual
metrics (SI-SDR/PESQ/DNSMOS) are secondary and must never override
downstream agent accuracy -- see `docs/geowearnet_wearersepnet_spec.md`.

Modules
-------
asr          P1.9  provider-agnostic ASR interface + cached sherpa-onnx backend
metrics      P1.1  WER/TER, bystander leakage, deletion, attribution, latency
commands     P1.8  agent command phrase set + false-command / retention metrics
gate         P1.2  GeoWear Gate v0 (streaming router) + P1.3 oracle gate
denoise      P1.4  RNNoise machinery-noise baseline (optional, isolated)
pipelines    P1.5  the six-way pipeline matrix
overlap      P1.6  4-state (00/10/01/11) conditioned analysis
stressbench  P1.7  SYNTHETIC PRODUCT STRESS BENCH (LibriSpeech + MUSAN)
mmcsg_bridge       read-only bridge to MMCSG audio/RTTM/word transcripts
anomalies    P1.13 known annotation anomaly registry accessor
harness      P1.1  top-level evaluation orchestration

Hard rules honoured by every module here:
  * raw MMCSG audio/RTTM/TSV files are opened read-only, never written;
  * the official MMCSG **dev** split is never evaluated by this package --
    dev is reserved for `training/geowearnet/mmcsg/final_selection.py`,
    which is guarded by `dev_guard.py`;
  * nothing here touches GPU, trains anything, or writes into
    `training/geowearnet/runs/` or `training/geowearnet/mmcsg/runs/`.
"""

__all__ = [
    "asr", "metrics", "commands", "gate", "denoise",
    "pipelines", "overlap", "stressbench", "mmcsg_bridge", "anomalies", "harness",
]
