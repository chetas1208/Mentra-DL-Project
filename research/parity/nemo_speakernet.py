"""Trainable NeMo SpeakerNet wrapper for parity testing against the ONNX
baseline (sprint spec section 12). Deterministic inference only -- no
training-time augmentation, model.eval() enforced.
"""
from __future__ import annotations

import numpy as np
import torch


class NemoSpeakerNetWrapper:
    def __init__(self, model_name: str = "speakerverification_speakernet", device: str = "cpu"):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=model_name)
        self.model.eval()
        self.model.to(device)
        self.device = device
        torch.manual_seed(0)

    @torch.no_grad()
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """samples: float32 mono in [-1, 1] at sample_rate (must be 16000).
        Returns L2-normalized embedding as numpy array, matching the
        normalization convention used by research/sherpa_onnx/detector.py."""
        if sample_rate != 16000:
            raise ValueError(f"expected 16000 Hz, got {sample_rate}")

        audio = torch.from_numpy(samples).float().unsqueeze(0).to(self.device)
        length = torch.tensor([samples.shape[0]], dtype=torch.int64).to(self.device)

        _, emb = self.model.forward(input_signal=audio, input_signal_length=length)
        emb = emb.squeeze(0).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    @property
    def dim(self) -> int:
        return self.model.decoder.final.out_features if hasattr(self.model.decoder, "final") else -1
