"""Trainable SpeakerNet backbone wrapper (sprint spec section 3, "GO" phase).

Wraps the real pretrained NeMo EncDecSpeakerLabelModel's preprocessor +
encoder + decoder pooling/projection -- does not reimplement SpeakerNet
from scratch. Exposes both:

  encode_frames()  -> [B, 1500, T] frame-level features, BEFORE statistics
                       pooling. This is the entire point: SpeakerNet's own
                       verification pipeline pools to one fixed embedding
                       per utterance, which is why it needs >=1.0s of audio
                       to work well (measured: 6.25% EER at 1.0s vs 2.08%
                       at 2.0s+). MentraWearNet needs frame-level resolution
                       to do better at short context.

  encode_speaker()  -> [B, 256] L2-normalized speaker embedding, via the
                        SAME encoder instance (mandatory -- one shared
                        backbone, not two copies; see section 6). Used only
                        for enrollment, computed once and cached at
                        deployment time, never per live-audio call.

The original 7205-way VoxCeleb classifier (model.decoder.final) is
intentionally never touched here -- it's a training-time artifact of the
pretrained checkpoint, not part of what MentraWearNet needs, and is
excluded from the deployment parameter count (see count_params.py).

Frame stride measured empirically (2026-08-21, see docs/MENTRAWEARNET_ARCHITECTURE.md):
T(samples) ~= 0.00601 * samples + 16 (~10.4ms/frame hop, ~166ms fixed
window overhead). Not a textbook derivation -- measured directly from this
checkpoint's preprocessor+encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpeakerNetBackbone(nn.Module):
    def __init__(self, model_name: str = "speakerverification_speakernet"):
        super().__init__()
        import nemo.collections.asr as nemo_asr

        nemo_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=model_name)

        # Reuse the real pretrained modules directly -- no reimplementation.
        self.preprocessor = nemo_model.preprocessor
        self.encoder = nemo_model.encoder
        # decoder._pooling (StatsPoolLayer) + decoder.emb_layers[0] (Linear
        # 3000->256, BatchNorm, ReLU) are the embedding path we keep.
        # decoder.final (Linear 256->7205) is the classifier we DROP.
        self.pooling = nemo_model.decoder._pooling
        self.embedding_projection = nemo_model.decoder.emb_layers[0]

        self.frame_channels = 1500  # measured: ConvASREncoder's last Jasper block output channels
        self.embedding_dim = 256    # measured: decoder.emb_layers[0] output

    def encode_frames(self, waveform: torch.Tensor, lengths: torch.Tensor):
        """waveform: [B, N] float32 in [-1, 1]. Returns (frame_features [B, 1500, T], frame_lengths [B])."""
        proc_sig, proc_len = self.preprocessor(input_signal=waveform, length=lengths)
        frame_features, frame_lengths = self.encoder(audio_signal=proc_sig, length=proc_len)
        return frame_features, frame_lengths

    def encode_speaker(self, waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """waveform: [B, N] float32 in [-1, 1]. Returns L2-normalized [B, 256] embedding.
        Shares the SAME encoder as encode_frames -- verified by
        tests/test_shared_backbone.py (single set of encoder parameters,
        not two)."""
        frame_features, frame_lengths = self.encode_frames(waveform, lengths)
        pooled = self.pooling(frame_features, frame_lengths)  # [B, 3000] (mean+std over valid frames)
        emb = self.embedding_projection(pooled)  # [B, 256]
        return nn.functional.normalize(emb, p=2, dim=-1)

    def num_backbone_params(self) -> int:
        return (sum(p.numel() for p in self.preprocessor.parameters()) +
                sum(p.numel() for p in self.encoder.parameters()) +
                sum(p.numel() for p in self.pooling.parameters()) +
                sum(p.numel() for p in self.embedding_projection.parameters()))
