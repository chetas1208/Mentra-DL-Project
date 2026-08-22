#!/usr/bin/env python3
"""Compute the exact future-context (right-context) budget of the pretrained
SpeakerNet encoder from its real module graph -- not estimated from memory
(sprint spec section 7/8). Walks ConvASREncoder's Jasper blocks, sums the
right-context contributed by every kernel>1 conv layer (stride is 1
throughout this encoder, confirmed by inspection, so right-context
accumulates additively across sequential layers)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import nemo.collections.asr as nemo_asr


def main():
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name="speakerverification_speakernet")

    preproc_cfg = model.cfg.preprocessor
    window_size_ms = preproc_cfg.window_size * 1000
    hop_ms = preproc_cfg.window_stride * 1000
    stft_lookahead_ms = window_size_ms / 2  # centered STFT window

    print(f"STFT window_size={window_size_ms}ms hop={hop_ms}ms")
    print(f"STFT centering lookahead: {stft_lookahead_ms}ms (half the analysis window)")
    print(f"normalize type: {preproc_cfg.normalize!r}")
    if preproc_cfg.normalize == "per_feature":
        print("  -> UNBOUNDED future dependency: mean/std computed over the FULL")
        print("     valid-length utterance (confirmed by reading normalize_batch()")
        print("     source -- x_mean_numerator sums over the entire time axis using")
        print("     seq_len, not a bounded window). This dominates the conv-padding")
        print("     right-context below for any utterance longer than ~1-2 seconds.")

    print()
    print("=== Jasper block right-context (kernel>1 conv layers only) ===")
    total_right_context_frames = 0
    encoder_blocks = list(model.encoder.encoder)  # nn.Sequential of JasperBlock
    for i, block in enumerate(encoder_blocks):
        block_context = 0
        layer_details = []
        for layer in block.mconv:
            if hasattr(layer, "conv") and hasattr(layer.conv, "kernel_size"):
                k = layer.conv.kernel_size[0]
                d = layer.conv.dilation[0]
                if k > 1:
                    ctx = ((k - 1) // 2) * d
                    block_context += ctx
                    layer_details.append(f"k={k} d={d} groups={layer.conv.groups} -> +{ctx} frames")
        total_right_context_frames += block_context
        print(f"block {i}: {layer_details} => block right-context = {block_context} frames")

    total_ms = total_right_context_frames * hop_ms
    print()
    print(f"TOTAL conv right-context: {total_right_context_frames} frames = {total_ms:.1f}ms")
    print(f"TOTAL bounded future context (STFT + conv): {stft_lookahead_ms + total_ms:.1f}ms")
    print(f"PLUS unbounded per_feature normalization (see above) -- this is the")
    print(f"dominant non-causal component, not the {total_ms:.1f}ms of conv lookahead.")


if __name__ == "__main__":
    main()
