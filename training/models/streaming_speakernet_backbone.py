"""StreamingSpeakerNetBackbone: strict-causal (C0) variant of the pretrained
SpeakerNet encoder (sprint spec sections 11-14, Phase 2).

Does NOT duplicate the offline SpeakerNetBackbone or its weights (section
12) -- takes the same class of pretrained model, then in-place patches:

  1. Every kernel>1 MaskedConv1d's padding from symmetric ("same", k//2
     both sides) to causal (k-1 on the left, 0 on the right), using NeMo's
     own asymmetric-padding mechanism (MaskedConv1d.pad_layer +
     nn.ConstantPad1d) rather than hand-rolled forward hooks -- this is
     the "use NeMo's future-context mechanism where possible" instruction.
     Zero new parameters: same conv kernels, just where they're centered.

  2. Feature normalization from SpeakerNet's default `per_feature` (proven
     unbounded -- see docs/SPEAKERNET_CONTEXT_REPORT.md, computed from
     normalize_batch() source) to a FIXED global CMVN computed once from
     real training-adjacent audio (scripts/model/compute_fixed_cmvn.py),
     applied identically to every frame regardless of future context.

After both patches, the only remaining future dependency is the STFT's
10ms centered-window lookahead (section 6) -- accepted as a fixed, tiny,
measured cost rather than eliminated, since making the STFT itself causal
would require changing window centering in a way NeMo doesn't expose
cleanly and 10ms is negligible next to the 310ms this patch removes.

This is the "SpeakerNet-C0" variant from the spec: strict causal in the
convolutional stack, 10ms residual STFT lookahead, fixed-not-adaptive
normalization. SpeakerNet-LowRC (bounded, not zero, lookahead) is not
implemented -- deferred, see docs/TODO.md.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn


def _total_context(masked_conv1d) -> int:
    """Original symmetric right-context (== left-context) in frames for one
    kernel>1 layer, before any patching -- the fixed budget that gets
    redistributed between left/right by _make_causal / _make_bounded_causal."""
    conv = masked_conv1d.conv
    kernel_size = conv.kernel_size[0]
    dilation = conv.dilation[0]
    if kernel_size == 1:
        return 0
    return dilation * (kernel_size - 1)


def _make_causal(masked_conv1d) -> None:
    """In-place: converts one MaskedConv1d from symmetric to strict-causal
    padding (right_context=0). Reuses the exact same nn.Conv1d weight
    tensor -- no new parameters."""
    _make_bounded_causal(masked_conv1d, right_context_frames=0)


def _make_bounded_causal(masked_conv1d, right_context_frames: int) -> None:
    """In-place: converts one MaskedConv1d from symmetric padding to
    asymmetric padding with a specific bounded right-context (LowRC,
    sprint spec Track B section 7). right_context_frames=0 is identical to
    _make_causal (C0). The total context budget (left+right) is preserved
    exactly as the original symmetric padding -- only where it's spent
    changes, so output length and receptive field size are unaffected,
    just its temporal centering."""
    conv = masked_conv1d.conv
    kernel_size = conv.kernel_size[0]
    dilation = conv.dilation[0]
    if kernel_size == 1:
        return  # no receptive field to redistribute
    total_context = dilation * (kernel_size - 1)
    right_pad = min(right_context_frames, total_context)
    left_pad = total_context - right_pad
    masked_conv1d.pad_layer = nn.ConstantPad1d((left_pad, right_pad), value=0.0)
    conv.padding = (0,)
    masked_conv1d._padding = (left_pad, right_pad)
    masked_conv1d.same_padding = False
    masked_conv1d.same_padding_asymmetric = True  # length-preserving, redistributed left/right


class FixedCMVN(nn.Module):
    """Replaces per-utterance per_feature normalization with a fixed global
    mean/std per mel channel (section 5, option A). Registered as buffers
    so it moves with .to(device) correctly."""

    def __init__(self, stats_path: Path):
        super().__init__()
        stats = json.loads(Path(stats_path).read_text())
        mean = torch.tensor(stats["fixed_mean"], dtype=torch.float32)
        std = torch.tensor(stats["fixed_std"], dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, -1, 1))
        self.register_buffer("std", std.view(1, -1, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, n_mels, T] raw log-mel, pre-normalization
        return (features - self.mean) / self.std


class StreamingSpeakerNetBackbone(nn.Module):
    DEFAULT_CMVN_PATH = Path(__file__).parent / "speakernet_fixed_cmvn.json"

    def __init__(self, model_name: str = "speakerverification_speakernet",
                 cmvn_stats_path: Path | None = None,
                 max_lookahead_frames: int = 0):
        """max_lookahead_frames=0 gives strict-causal (C0). A positive value
        distributes that many frames of right-context proportionally across
        the 7 kernel>1 layers (LowRC, Track B) -- each layer gets
        min(its original right-context, round(budget * its share of the
        total 31-frame original right-context)), so the budget is spent
        preferentially on layers that originally had more context rather
        than split evenly regardless of kernel size."""
        super().__init__()
        import nemo.collections.asr as nemo_asr

        nemo_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=model_name)
        # deepcopy so patching this instance's conv padding never touches
        # the offline SpeakerNetBackbone's shared module tree if both are
        # constructed in the same process (they load independent pretrained
        # instances either way via from_pretrained, but explicit is safer).
        self.featurizer = nemo_model.preprocessor.featurizer
        self.encoder = copy.deepcopy(nemo_model.encoder)
        self.pooling = nemo_model.decoder._pooling
        self.embedding_projection = nemo_model.decoder.emb_layers[0]

        self.max_lookahead_frames = max_lookahead_frames

        causal_layers = []
        for block in self.encoder.encoder:  # nn.Sequential of JasperBlock
            for layer in block.mconv:
                if hasattr(layer, "conv") and hasattr(layer, "pad_layer") and layer.conv.kernel_size[0] > 1:
                    causal_layers.append(layer)

        total_original_context = sum(_total_context(l) for l in causal_layers)  # 31, measured

        n_patched = 0
        allocated_so_far = 0
        for i, layer in enumerate(causal_layers):
            layer_original = _total_context(layer)
            if max_lookahead_frames <= 0 or total_original_context == 0:
                layer_budget = 0
            elif i == len(causal_layers) - 1:
                layer_budget = max_lookahead_frames - allocated_so_far  # last layer: remainder, avoids rounding loss
            else:
                layer_budget = round(max_lookahead_frames * (layer_original / total_original_context))
            layer_budget = max(0, min(layer_budget, layer_original))
            allocated_so_far += layer_budget
            _make_bounded_causal(layer, right_context_frames=layer_budget)
            n_patched += 1
        self.n_causal_patched_layers = n_patched
        self.allocated_lookahead_frames = min(allocated_so_far, total_original_context)

        cmvn_path = cmvn_stats_path or self.DEFAULT_CMVN_PATH
        self.fixed_cmvn = FixedCMVN(cmvn_path)

        self.frame_channels = 1500
        self.embedding_dim = 256

    def _extract_raw_features(self, waveform: torch.Tensor, lengths: torch.Tensor):
        """Runs the featurizer (STFT+mel+log) WITHOUT SpeakerNet's default
        per_feature normalization, then applies fixed CMVN instead."""
        raw_features, feat_lengths = self.featurizer(waveform, lengths)
        normalized = self.fixed_cmvn(raw_features)
        return normalized, feat_lengths

    def encode_frames(self, waveform: torch.Tensor, lengths: torch.Tensor):
        features, feat_lengths = self._extract_raw_features(waveform, lengths)
        frame_features, frame_lengths = self.encoder(audio_signal=features, length=feat_lengths)
        return frame_features, frame_lengths

    def encode_speaker(self, waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        frame_features, frame_lengths = self.encode_frames(waveform, lengths)
        pooled = self.pooling(frame_features, frame_lengths)
        emb = self.embedding_projection(pooled)
        return nn.functional.normalize(emb, p=2, dim=-1)

    def num_backbone_params(self) -> int:
        return (sum(p.numel() for p in self.encoder.parameters()) +
                sum(p.numel() for p in self.pooling.parameters()) +
                sum(p.numel() for p in self.embedding_projection.parameters()))
        # featurizer has no trainable params (mel filterbank is a fixed buffer);
        # fixed_cmvn is also non-trainable buffers, correctly excluded.
